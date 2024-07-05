# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

import peft
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset

from operators.efficient_linear import EfficientMemoryLinear
from operators.efficient_gelu import EfficientMemoryGELU
from operators.efficient_silu import EfficientMemorySiLU
from operators.efficient_layernorm import EfficientMemoryLayerNorm
from operators.efficient_rmsnorm import EfficientMemoryRMSNorm
from operators.efficient_rmsnorm_gemma import EfficientMemoryRMSNormGemma
from operators.efficient_dropout import EfficientMemoryDropout
from operators.efficient_hadamard import EfficientMemoryHadamard
from operators.efficient_gemm import EfficientMemoryGEMM, EfficientMemoryGEMMWithSoftmax
from operators.efficient_softmax import EfficientMemorySoftmax

import os

os.environ["WANDB_PROJECT"] = "gsm8k-svd"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="LoftQ/Mistral-7B-v0.1-4bit-64rank",
        metadata={"help": "Path to the model."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."
        },
    )
    lora_init: bool = field(
        default=False,
        metadata={
            "help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."
        },
    )
    full_precision: bool = field(
        default=False,
        metadata={
            "help": "False: Use bitsandbytes Linear4bit, real quantization"
            "True: Use quantization equivalent fp16/fp32 weights."
            "Note: Set True for data parallel training"
        },
    )
    rank: int = field(
        default=64,
        metadata={
            "help": "Rank of LoRA adapters. LoftQ does not require this config. Used for fp16 LoRA or QLoRA."
        },
    )
    bits: int = field(
        default=4,
        metadata={
            "help": "Bit of the backbone. LoftQ does not require this config. Used for QLoRA."
        },
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    transform_bp_enable: bool = (
        field(
            default=False,
            metadata={
                "help": "True: Use transform block; False: Do not use transform block"
            },
        ),
    )
    gradient_checkpointing_enable: bool = field(
        default=False,
        metadata={
            "help": "True: Use gradient checkpointing; False: Do not use gradient checkpointing"
        },
    )
    flash_attention: bool = field(
        default=False,
        metadata={
            "help": "True: Use Flash Attention; False: Do not use Flash Attention"
        },
    )
    linear_outlier_ratio: float = field(
        default=0.01,
        metadata={"help": "Linear outlier ratio"},
    )
    linear_sub_outlier_ratio: float = field(
        default=0.2,
        metadata={"help": "Linear sub outlier ratio"},
    )
    linear_sub_outlier_bit: int = field(
        default=8,
        metadata={"help": "Linear sub outlier ratio"},
    )
    linear_sub_outlier_quant_method: str = field(
        default="per-tensor",
        metadata={"help": "Quantization method"},
    )
    linear_rank: int = field(
        default=16,
        metadata={"help": "Linear rank"},
    )
    silu_outlier_ratio: float = field(
        default=0.01,
        metadata={"help": "SiLU outlier ratio"},
    )
    silu_sub_outlier_ratio: float = field(
        default=0.2,
        metadata={"help": "SiLU sub outlier ratio"},
    )
    silu_sub_outlier_bit: int = field(
        default=8,
        metadata={"help": "SiLU sub outlier ratio"},
    )
    silu_sub_outlier_quant_method: str = field(
        default="per-tensor",
        metadata={"help": "Quantization method"},
    )
    silu_rank: int = field(
        default=16,
        metadata={"help": "SiLU rank"},
    )
    layernorm_outlier_ratio: float = field(
        default=0.01,
        metadata={"help": "LayerNorm outlier ratio"},
    )
    layernorm_sub_outlier_ratio: float = field(
        default=0.2,
        metadata={"help": "LayerNorm sub outlier ratio"},
    )
    layernorm_sub_outlier_bit: int = field(
        default=8,
        metadata={"help": "LayerNorm sub outlier bit"},
    )
    layernorm_sub_outlier_quant_method: str = field(
        default="per-tensor",
        metadata={"help": "Quantization method"},
    )
    layernorm_rank: int = field(
        default=16,
        metadata={"help": "LayerNorm rank"},
    )
    softmax_outlier_ratio: float = field(
        default=0.01,
        metadata={"help": "Softmax outlier ratio"},
    )
    softmax_sub_outlier_ratio: float = field(
        default=0.2,
        metadata={"help": "Softmax sub outlier ratio"},
    )
    softmax_sub_outlier_bit: int = field(
        default=8,
        metadata={"help": "Softmax sub outlier bit"},
    )
    softmax_rank: int = field(
        default=16,
        metadata={"help": "Softmax rank"},
    )
    gemm_outlier_ratio: float = field(
        default=0.01,
        metadata={"help": "GEMM outlier ratio"},
    )
    gemm_sub_outlier_ratio: float = field(
        default=0.2,
        metadata={"help": "GEMM sub outlier ratio"},
    )
    gemm_sub_outlier_bit: int = field(
        default=8,
        metadata={"help": "GEMM sub outlier bit"},
    )
    gemm_sub_outlier_quant_method: str = field(
        default="per-tensor",
        metadata={"help": "Quantization method"},
    )
    gemm_rank: int = field(
        default=16,
        metadata={"help": "GEMM rank"},
    )
    hadamard_outlier_ratio: float = field(
        default=0.01,
        metadata={"help": "Hadamard outlier ratio"},
    )
    hadamard_sub_outlier_ratio: float = field(
        default=0.2,
        metadata={"help": "Hadamard sub outlier ratio"},
    )
    hadamard_sub_outlier_bit: int = field(
        default=8,
        metadata={"help": "Hadamard sub outlier bit"},
    )
    hadamard_sub_outlier_quant_method: str = field(
        default="per-tensor",
        metadata={"help": "Quantization method"},
    )
    hadamard_rank: int = field(
        default=16,
        metadata={"help": "Hadamard rank"},
    )
    extract_mode: bool = field(
        default=False, metadata={"help": "whether to extract the hidden states"}
    )
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    init_lora_weights: str = field(
        default="loftq",
        metadata={
            "help": "init_lora_weights (`['gaussian', 'loftq', 'pissa', 'pissa_init']`):"
        },
    )


@dataclass
class DataArguments:
    data_name: str = field(default="gsm8k", metadata={"help": "Dataset name."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )


class GACTTrainer(Trainer):
    def __init__(self, extract_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extract_mode = extract_mode

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = super().compute_loss(model, inputs, return_outputs)
        # #! only for debug
        # allocated = torch.cuda.memory_allocated()
        # reserved = torch.cuda.memory_reserved()
        # print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        # print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
        return loss


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    # sources are questions, and targets are answers
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources = [f"{example['question']}{QUESTION_PROMPT}" for example in raw_data]
        targets = [
            f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT)
            for example in raw_data
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        #! a tricky way to pad the input_ids and labels to 16's multiple
        # 1. find the max length of input_ids
        max_len = max([len(input_id) for input_id in input_ids])
        # 2. pad the input_ids and labels to 8's multiple
        # max_len = 512  #! only for profile
        max_len = (max_len + 7) // 8 * 8
        # 3. generate a max_len tensor
        max_len_tensor = torch.randn(max_len).to(torch.int64)
        # 4. append the max_len tensor to the input_ids and labels
        input_ids.append(max_len_tensor)
        labels.append(max_len_tensor)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # delete the max_len tensor
        input_ids = input_ids[:-1]
        labels = labels[:-1]

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_name, "main")
    train_set = dataset["train"]
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    print("train_dataset: ", train_dataset)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def replace_module_for_quantization(module, compress_config, model_args):
    for name, child in module.named_children():
        if (
            isinstance(child, torch.nn.Linear)
            and (child.weight.requires_grad)
            and (child.in_features > 100)
            and ("lm_head" not in name)
        ):
            original_weight_data = child.weight.data
            original_bias_data = child.bias.data if child.bias is not None else None
            new_child = EfficientMemoryLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                outlier_ratio=compress_config["linear"]["outlier_ratio"],
                sub_outlier_ratio=compress_config["linear"]["sub_outlier_ratio"],
                rank=compress_config["linear"]["rank"],
                sub_outlier_bit=compress_config["linear"]["sub_outlier_bit"],
                sub_outlier_quantize_method=compress_config["linear"][
                    "sub_outlier_quant_method"
                ],
            )
            new_child.weight.data = original_weight_data
            if child.bias is not None:
                new_child.bias.data = original_bias_data
            setattr(module, name, new_child)

        elif "SiLU" in child.__class__.__name__:
            new_child = EfficientMemorySiLU(
                outlier_ratio=compress_config["silu"]["outlier_ratio"],
                sub_outlier_ratio=compress_config["silu"]["sub_outlier_ratio"],
                rank=compress_config["silu"]["rank"],
                sub_outlier_bit=compress_config["silu"]["sub_outlier_bit"],
                sub_outlier_quantize_method=compress_config["silu"][
                    "sub_outlier_quant_method"
                ],
            )
            setattr(module, name, new_child)

        elif "GELU" in child.__class__.__name__:
            new_child = EfficientMemoryGELU(
                outlier_ratio=compress_config["silu"]["outlier_ratio"],
                sub_outlier_ratio=compress_config["silu"]["sub_outlier_ratio"],
                rank=compress_config["silu"]["rank"],
                sub_outlier_bit=compress_config["silu"]["sub_outlier_bit"],
                sub_outlier_quantize_method=compress_config["silu"][
                    "sub_outlier_quant_method"
                ],
            )
            setattr(module, name, new_child)

        elif "RMSNorm" in child.__class__.__name__:
            original_weight_data = child.weight.data
            if "gemma" in model_args.model_name_or_path:
                new_child = EfficientMemoryRMSNormGemma(
                    normalized_shape=child.weight.data.shape,
                    eps=child.variance_epsilon,
                    elementwise_affine=True,
                    bias=False,
                    outlier_ratio=compress_config["layernorm"]["outlier_ratio"],
                    sub_outlier_ratio=compress_config["layernorm"][
                        "sub_outlier_ratio"
                    ],
                    rank=compress_config["layernorm"]["rank"],
                    sub_outlier_bit=compress_config["layernorm"]["sub_outlier_bit"],
                    sub_outlier_quantize_method=compress_config["layernorm"][
                        "sub_outlier_quant_method"
                    ],
                )
            else:
                new_child = EfficientMemoryRMSNorm(
                    normalized_shape=child.weight.data.shape,
                    eps=child.variance_epsilon,
                    elementwise_affine=True,
                    bias=False,
                    outlier_ratio=compress_config["layernorm"]["outlier_ratio"],
                    sub_outlier_ratio=compress_config["layernorm"][
                        "sub_outlier_ratio"
                    ],
                    rank=compress_config["layernorm"]["rank"],
                    sub_outlier_bit=compress_config["layernorm"]["sub_outlier_bit"],
                    sub_outlier_quantize_method=compress_config["layernorm"][
                        "sub_outlier_quant_method"
                    ],
                )
            new_child.weight.data = original_weight_data
            setattr(module, name, new_child)
            
        elif "LayerNorm" in child.__class__.__name__:
            original_weight_data = child.weight.data
            original_bias_data = child.bias.data
            new_child = EfficientMemoryLayerNorm(
                normalized_shape=child.weight.data.shape,
                elementwise_affine=True,
                bias=child.bias is not None,
                outlier_ratio=compress_config["layernorm"]["outlier_ratio"],
                sub_outlier_ratio=compress_config["layernorm"]["sub_outlier_ratio"],
                rank=compress_config["layernorm"]["rank"],
                sub_outlier_bit=compress_config["layernorm"]["sub_outlier_bit"],
                sub_outlier_quantize_method=compress_config["layernorm"][
                    "sub_outlier_quant_method"
                ],
            )
            new_child.weight.data = original_weight_data
            if child.bias is not None:
                new_child.bias.data = original_bias_data
            setattr(module, name, new_child)
            
        elif isinstance(child, torch.nn.Softmax):
            new_child = EfficientMemorySoftmax(
                outlier_ratio=compress_config["softmax"]["outlier_ratio"]
            )
            setattr(module, name, new_child)
            
        elif isinstance(child, torch.nn.Dropout):
            setattr(module, name, EfficientMemoryDropout(0.0))
            
        elif "Hadamard" in child.__class__.__name__:
            new_child = EfficientMemoryHadamard(
                outlier_ratio_1=compress_config["hadamard"]["outlier_ratio"],
                sub_outlier_ratio_1=compress_config["hadamard"]["sub_outlier_ratio"],
                sub_outlier_bit_1=compress_config["hadamard"]["sub_outlier_bit"],
                sub_outlier_quantize_method_1=compress_config["hadamard"][
                    "sub_outlier_quant_method"
                ],
                rank=compress_config["hadamard"]["rank"],
                outlier_ratio_2=compress_config["hadamard"]["outlier_ratio"],
                sub_outlier_ratio_2=compress_config["hadamard"]["sub_outlier_ratio"],
                sub_outlier_bit_2=compress_config["hadamard"]["sub_outlier_bit"],
                sub_outlier_quantize_method_2=compress_config["hadamard"][
                    "sub_outlier_quant_method"
                ],
            )
            setattr(module, name, new_child)
            
        elif "GEMM" in child.__class__.__name__:
            if "gemm1" in name:
                new_child = EfficientMemoryGEMM(
                    outlier_ratio_1=compress_config["gemm"]["outlier_ratio"],
                    sub_outlier_ratio_1=compress_config["gemm"]["sub_outlier_ratio"],
                    sub_outlier_bit_1=compress_config["gemm"]["sub_outlier_bit"],
                    sub_outlier_quantize_method_1=compress_config["gemm"][
                        "sub_outlier_quant_method"
                    ],
                    rank=compress_config["gemm"]["rank"],
                    outlier_ratio_2=compress_config["gemm"]["outlier_ratio"],
                    sub_outlier_ratio_2=compress_config["gemm"]["sub_outlier_ratio"],
                    sub_outlier_bit_2=compress_config["gemm"]["sub_outlier_bit"],
                    sub_outlier_quantize_method_2=compress_config["gemm"][
                        "sub_outlier_quant_method"
                    ],
                )
            else:
                new_child = EfficientMemoryGEMMWithSoftmax(
                    outlier_ratio_1=compress_config["softmax"]["outlier_ratio"],
                    outlier_ratio_2=compress_config["gemm"]["outlier_ratio"],
                    sub_outlier_ratio_2=compress_config["gemm"]["sub_outlier_ratio"],
                    sub_outlier_bit_2=compress_config["gemm"]["sub_outlier_bit"],
                    sub_outlier_quantize_method_2=compress_config["gemm"][
                        "sub_outlier_quant_method"
                    ],
                    rank=compress_config["gemm"]["rank"],
                )
            setattr(module, name, new_child)
            
        else:
            replace_module_for_quantization(child, compress_config, model_args)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    compress_config = {
        "linear": {
            "outlier_ratio": model_args.linear_outlier_ratio,
            "sub_outlier_ratio": model_args.linear_sub_outlier_ratio,
            "sub_outlier_bit": model_args.linear_sub_outlier_bit,
            "sub_outlier_quant_method": model_args.linear_sub_outlier_quant_method,
            "rank": model_args.linear_rank,
        },
        "silu": {
            "outlier_ratio": model_args.silu_outlier_ratio,
            "sub_outlier_ratio": model_args.silu_sub_outlier_ratio,
            "sub_outlier_bit": model_args.silu_sub_outlier_bit,
            "sub_outlier_quant_method": model_args.silu_sub_outlier_quant_method,
            "rank": model_args.silu_rank,
        },
        "layernorm": {
            "outlier_ratio": model_args.layernorm_outlier_ratio,
            "sub_outlier_ratio": model_args.layernorm_sub_outlier_ratio,
            "sub_outlier_bit": model_args.layernorm_sub_outlier_bit,
            "sub_outlier_quant_method": model_args.layernorm_sub_outlier_quant_method,
            "rank": model_args.layernorm_rank,
        },
        "softmax": {
            "outlier_ratio": model_args.softmax_outlier_ratio,
            "sub_outlier_ratio": model_args.softmax_sub_outlier_ratio,
            "sub_outlier_bit": model_args.softmax_sub_outlier_bit,
            "rank": model_args.softmax_rank,
        },
        "gemm": {
            "outlier_ratio": model_args.gemm_outlier_ratio,
            "sub_outlier_ratio": model_args.gemm_sub_outlier_ratio,
            "sub_outlier_bit": model_args.gemm_sub_outlier_bit,
            "sub_outlier_quant_method": model_args.gemm_sub_outlier_quant_method,
            "rank": model_args.gemm_rank,
        },
        "hadamard": {
            "outlier_ratio": model_args.hadamard_outlier_ratio,
            "sub_outlier_ratio": model_args.hadamard_sub_outlier_ratio,
            "sub_outlier_bit": model_args.hadamard_sub_outlier_bit,
            "sub_outlier_quant_method": model_args.hadamard_sub_outlier_quant_method,
            "rank": model_args.hadamard_rank,
        },
    }
    
    print(compress_config)

    if model_args.full_precision:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            attn_implementation=(
                "flash_attention_2" if model_args.flash_attention else "eager"
            ),
            trust_remote_code=True,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            use_cache=False if model_args.gradient_checkpointing_enable else True,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            attn_implementation=(
                "flash_attention_2" if model_args.flash_attention else "eager"
            ),
            trust_remote_code=True,
        )
        model = peft.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=model_args.gradient_checkpointing_enable,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        if model_args.gradient_checkpointing_enable:
            print("Gradient Checkpointing is enabled")

    ##########################
    #       Peft Model       #
    ##########################
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            init_lora_weights=True,
        )
        model = get_peft_model(model, lora_config)
    elif model_args.adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            model_args.adapter_name_or_path,
            is_trainable=True,
            token=model_args.token,
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            model_args.model_name_or_path,
            subfolder="loftq_init",
            is_trainable=True,
            token=model_args.token,
        )

    # # # # replace the module
    if model_args.transform_bp_enable:
        print('replacing the modules...')
        replace_module_for_quantization(model, compress_config, model_args)
    print(model)

    # get the model name
    for name, module in model.named_modules():
        module.name = name

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split("/")[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )
    trainer = GACTTrainer(
        extract_mode=model_args.extract_mode,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
