import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MAX_NEW_TOKENS = 128
model_name = '/home/yujin-wa20/gemma-2b-nf4'

text = 'Is Taiwan a country?\n'
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(text, return_tensors="pt").input_ids

max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

# n_gpus = torch.cuda.device_count()
# max_memory = {i: max_memory for i in range(n_gpus)}
# print(max_memory)

quantization_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.float16,
  bnb_4bit_quant_type='nf4'
)
print(torch.cuda.max_memory_reserved() / 1024 / 1024, 'MiB')
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  quantization_config=quantization_config,
  low_cpu_mem_usage=True
)
print(torch.cuda.max_memory_reserved() / 1024 / 1024, 'MiB')
generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
print(torch.cuda.max_memory_reserved() / 1024 / 1024, 'MiB')