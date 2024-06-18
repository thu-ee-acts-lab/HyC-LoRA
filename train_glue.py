import argparse
import logging
import math
import os
import random
import torch
from pathlib import Path

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
import wandb

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

import peft
from peft import (
    LoraConfig,
    PeftModelForSequenceClassification,
)

from operators.efficient_linear import EfficientMemoryLinear
from operators.efficient_gelu import EfficientMemoryGELU
from operators.efficient_layernorm import EfficientMemoryLayerNorm
from operators.efficient_dropout import EfficientMemoryDropout
from operators.efficient_gemm import EfficientMemoryGEMM, EfficientMemoryGEMMWithSoftmax
from operators.efficient_softmax import EfficientMemorySoftmax

logger = logging.getLogger(__name__)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

metric_key = {
    "mrpc": "f1",
    "sst2": "accuracy",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "qqp": "f1",
    "stsb": "pearsonr",
    "cola": "matthews_correlation",
    "wnli": "accuracy",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation-file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use-slow-tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay to use."
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=1.0,
        help="Maximum norm of gradient.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num-warmup-steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub-token", type=str, help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--layer-num", type=int, default=24, help="Number of Bert layers"
    )
    parser.add_argument("--hidden-size", type=int, default=1024, help="hidden size")
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=4096,
        help="customize intermediate size",
    )
    parser.add_argument(
        "--ckpt", action="store_true", help="enable gradient checkpoint"
    )
    parser.add_argument("--r", action="store_true", help="lora rank", default=16)
    # config about sparse bp
    parser.add_argument("--sparse-bp", action="store_true", help="enable sparse bp")
    parser.add_argument(
        "--sparse-bp-freeze-range",
        type=str,
        default="[i for i in range(12)]",
        help="sparse bp range",
    )
    parser.add_argument(
        "--sparse-bp-freeze-layer", type=str, default="[]", help="sparse bp layer"
    )
    parser.add_argument(
        "--linear-outlier-ratio", type=float, default=0.01, help="linear outlier ratio"
    )
    parser.add_argument(
        "--linear-sub-outlier-ratio",
        type=float,
        default=0.2,
        help="linear sub outlier ratio",
    )
    parser.add_argument(
        "--linear-sub-outlier-bit",
        type=int,
        default=8,
        help="linear sub outlier ratio",
    )
    parser.add_argument(
        "--linear-sub-outlier-quant-method",
        type=str,
        default="per-tensor",
        help="Quantization method",
    )
    parser.add_argument("--linear-rank", type=int, default=16, help="Linear rank")

    # silu
    parser.add_argument(
        "--silu-outlier-ratio", type=float, default=0.01, help="SiLU outlier ratio"
    )
    parser.add_argument(
        "--silu-sub-outlier-ratio",
        type=float,
        default=0.2,
        help="SiLU sub outlier ratio",
    )
    parser.add_argument(
        "--silu-sub-outlier-bit", type=int, default=8, help="SiLU sub outlier ratio"
    )
    parser.add_argument(
        "--silu-sub-outlier-quant-method",
        type=str,
        default="per-tensor",
        help="Quantization method",
    )
    parser.add_argument("--silu-rank", type=int, default=16, help="SiLU rank")

    # layernorm
    parser.add_argument(
        "--layernorm-outlier-ratio",
        type=float,
        default=0.01,
        help="LayerNorm outlier ratio",
    )
    parser.add_argument(
        "--layernorm-sub-outlier-ratio",
        type=float,
        default=0.2,
        help="LayerNorm sub outlier ratio",
    )
    parser.add_argument(
        "--layernorm-sub-outlier-bit",
        type=int,
        default=8,
        help="LayerNorm sub outlier bit",
    )
    parser.add_argument(
        "--layernorm-sub-outlier-quant-method",
        type=str,
        default="per-tensor",
        help="Quantization method",
    )
    parser.add_argument("--layernorm-rank", type=int, default=16, help="LayerNorm rank")

    # softmax
    parser.add_argument(
        "--softmax-outlier-ratio",
        type=float,
        default=0.01,
        help="Softmax outlier ratio",
    )
    parser.add_argument(
        "--softmax-sub-outlier-ratio",
        type=float,
        default=0.2,
        help="Softmax sub outlier ratio",
    )
    parser.add_argument(
        "--softmax-sub-outlier-bit", type=int, default=8, help="Softmax sub outlier bit"
    )
    parser.add_argument("--softmax-rank", type=int, default=16, help="Softmax rank")

    # gemm
    parser.add_argument(
        "--gemm-outlier-ratio", type=float, default=0.01, help="GEMM outlier ratio"
    )
    parser.add_argument(
        "--gemm-sub-outlier-ratio",
        type=float,
        default=0.2,
        help="GEMM sub outlier ratio",
    )
    parser.add_argument(
        "--gemm-sub-outlier-bit", type=int, default=8, help="GEMM sub outlier bit"
    )
    parser.add_argument(
        "--gemm-sub-outlier-quant-method",
        type=str,
        default="per-tensor",
        help="Quantization method",
    )
    parser.add_argument("--gemm-rank", type=int, default=16, help="GEMM rank")

    args = parser.parse_args()

    # Sanity checks
    if (
        args.task_name is None
        and args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert (
            args.output_dir is not None
        ), "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    compress_config = {
        "linear": {
            "outlier_ratio": args.linear_outlier_ratio,
            "sub_outlier_ratio": args.linear_sub_outlier_ratio,
            "sub_outlier_bit": args.linear_sub_outlier_bit,
            "sub_outlier_quant_method": args.linear_sub_outlier_quant_method,
            "rank": args.linear_rank,
        },
        "silu": {
            "outlier_ratio": args.silu_outlier_ratio,
            "sub_outlier_ratio": args.silu_sub_outlier_ratio,
            "sub_outlier_bit": args.silu_sub_outlier_bit,
            "sub_outlier_quant_method": args.silu_sub_outlier_quant_method,
            "rank": args.silu_rank,
        },
        "layernorm": {
            "outlier_ratio": args.layernorm_outlier_ratio,
            "sub_outlier_ratio": args.layernorm_sub_outlier_ratio,
            "sub_outlier_bit": args.layernorm_sub_outlier_bit,
            "sub_outlier_quant_method": args.layernorm_sub_outlier_quant_method,
            "rank": args.layernorm_rank,
        },
        "softmax": {
            "outlier_ratio": args.softmax_outlier_ratio,
            "sub_outlier_ratio": args.softmax_sub_outlier_ratio,
            "sub_outlier_bit": args.softmax_sub_outlier_bit,
            "rank": args.softmax_rank,
        },
        "gemm": {
            "outlier_ratio": args.gemm_outlier_ratio,
            "sub_outlier_ratio": args.gemm_sub_outlier_ratio,
            "sub_outlier_bit": args.gemm_sub_outlier_bit,
            "sub_outlier_quant_method": args.gemm_sub_outlier_quant_method,
            "rank": args.gemm_rank,
        },
    }

    print(f"Compress config: {compress_config}")

    return args, compress_config


def replace_module(module, compress_config):
    for name, child in module.named_children():
        if (
            isinstance(child, torch.nn.Linear)
            and (child.weight.requires_grad)
            and (
                name != "class_intermediate"
                and name != "out_proj"
                and child.in_features > 100
            )
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
            replace_module(child, compress_config)


def main():
    args, compress_config = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(device_placement=False)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "./glue.py",
            args.task_name,
        )
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (
            args.train_file if args.train_file is not None else args.valid_file
        ).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["classifier"],
        ),
        ignore_mismatched_sizes=True,
        device_map="auto",
    )
    use_gradient_checkpointing = args.ckpt
    model = peft.prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=use_gradient_checkpointing
    )

    if args.ckpt:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=args.r,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["dense", "key", "value", "query"],
    )
    model = PeftModelForSequenceClassification(model, peft_config)

    # print trainable parameter ratio:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params
    print(
        f"Total params: {total_params}, Trainable params: {trainable_params}, Trainable ratio: {trainable_ratio}"
    )

    # replace efficient modules
    replace_module(model, compress_config)

    for name, module in model.named_modules():
        module.name = name

    print(model)

    # update optimization level, this is only for logging output
    if args.ckpt:
        args.opt_level += "_ckpt"

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != "label"
        ]
        if (
            "sentence1" in non_label_column_names
            and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts, padding=padding, max_length=args.max_length, truncation=True
        )

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets[
        "validation_matched" if args.task_name == "mnli" else "validation"
    ]

    train_max_length = 0
    dev_max_length = 0
    for item in train_dataset:
        if len(item["input_ids"]) > train_max_length:
            train_max_length = len(item["input_ids"])
    for item in eval_dataset:
        if len(item["input_ids"]) > dev_max_length:
            dev_max_length = len(item["input_ids"])
    logger.info("Train max length: %d" % train_max_length)
    logger.info("Dev max length: %d" % dev_max_length)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    train_step = 0
    completed_steps = 0
    best_metric = 0
    model.to(args.device)

    # define wandb logger
    wandb_project = f"{args.model_name_or_path.split('/')[-1]}_{args.task_name}"
    wandb_name = f"lr_{args.learning_rate}_linear_{args.linear_rank}_{args.linear_sub_outlier_bit}_{args.linear_outlier_ratio}_silu_{args.silu_rank}_{args.silu_sub_outlier_bit}_{args.silu_outlier_ratio}_layernorm_{args.layernorm_rank}_{args.layernorm_sub_outlier_bit}_{args.layernorm_outlier_ratio}_softmax_{args.softmax_rank}_{args.softmax_sub_outlier_bit}_{args.softmax_outlier_ratio}_gemm_{args.gemm_rank}_{args.gemm_sub_outlier_bit}_{args.gemm_outlier_ratio}"
    wandb.init(project=wandb_project, name=wandb_name)
    wandb.define_metric("train_step")
    wandb.define_metric("val_step")
    wandb.define_metric("train_*", step_metric="train_step")
    wandb.define_metric("val_*", step_metric="val_step")
    wandb.log({"trainable_ratio": trainable_ratio})

    for epoch in range(args.num_train_epochs):
        model.train()
        for _, batch in enumerate(train_dataloader):
            train_step += 1

            for k, v in batch.items():
                batch[k] = v.to(args.device)

            outputs = model(**batch)

            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps

            # log the train loss
            wandb.log({"train_loss": loss.item(), "train_step": train_step})

            optimizer.zero_grad()
            loss.backward()
            torch.utils.checkpoint.first_iter = False

            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), args.max_gradient_norm
            )
            optimizer.step()  #! optimizer generate there
            lr_scheduler.step()
            completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        with torch.no_grad():
            model.eval()
            for _, batch in enumerate(eval_dataloader):
                for k, v in batch.items():
                    batch[k] = v.to(args.device)
                outputs = model(**batch)
                predictions = (
                    outputs.logits.argmax(dim=-1)
                    if not is_regression
                    else outputs.logits.squeeze()
                )
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

        eval_metric = metric.compute()
        print(f"epoch {epoch}: {eval_metric}")

        for key, value in eval_metric.items():
            wandb.log({f"val_{key}": value, "val_step": epoch})

        if eval_metric[metric_key[args.task_name]] > best_metric:
            best_metric = eval_metric[metric_key[args.task_name]]
            wandb.log({"best_val_acc": best_metric, "val_step": epoch})

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}",
                    blocking=False,
                    auto_lfs_prune=True,
                )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        with open(os.path.join(args.output_dir, "result.txt"), "a") as f:
            f.write(
                "lr:%f, bsz:%d, result:%f\n"
                % (
                    args.learning_rate,
                    args.per_device_train_batch_size,
                    best_metric,
                )
            )


if __name__ == "__main__":
    main()
