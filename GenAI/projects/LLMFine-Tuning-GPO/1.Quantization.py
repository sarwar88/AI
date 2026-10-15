!pip install -q -U transformers bitsandbytes trl peft datasets huggingface_hub

from huggingface_hub import login
login()

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch


model_name = "meta-llama/Llama-3.2-1B-Instruct"

quantization_config_8bit = BitsAndBytesConfig(load_in_8bit=True)

quantization_config_4bit = BitsAndBytesConfig(load_in_4bit=True)


model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config_4bit,
    device_map="auto"
)


model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config_8bit,
    device_map="auto"
)


model_16bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model_32bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)


import torch

def print_model_size_in_mb_gb(model):
    """
    Verilen PyTorch modelinin (parametreler ve buffer'lar dahil)
    toplam boyutunu MB ve GB cinsinden ekrana yazdırır.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_bytes = param_size + buffer_size
    size_mb = size_bytes / (1024**2)
    size_gb = size_mb / 1024

    print(f"Model toplam boyutu: {size_mb:.2f} MB ({size_gb:.2f} GB)")


print_model_size_in_mb_gb(model_4bit)
print_model_size_in_mb_gb(model_8bit)
print_model_size_in_mb_gb(model_16bit)
print_model_size_in_mb_gb(model_32bit)

import torch

torch.set_printoptions(precision=8, sci_mode=False)

layer_quantized = model_32bit.model.layers[0]
weight_quantized = layer_quantized.self_attn.q_proj.weight

print("32 bit Ağırlık Değeri:")
print(weight_quantized)

import torch

torch.set_printoptions(precision=8, sci_mode=False)

layer_quantized = model_8bit.model.layers[0]
weight_quantized = layer_quantized.self_attn.q_proj.weight

print("8 bit Ağırlık Değeri:")
print(weight_quantized)

layer_quantized = model_32bit.model.layers[0]
weight_quantized_max = layer_quantized.self_attn.q_proj.weight.max()
weight_quantized_min = layer_quantized.self_attn.q_proj.weight.min()

print("32 bit Ağırlık Değeri:")
print(f"weight_quantized_max {weight_quantized_max}")
print(f"weight_quantized_min {weight_quantized_min}")

layer_quantized = model_8bit.model.layers[0]
weight_quantized_max = layer_quantized.self_attn.q_proj.weight.max()
weight_quantized_min = layer_quantized.self_attn.q_proj.weight.min()

print("8 bit Ağırlık Değeri:")
print(f"weight_quantized_max {weight_quantized_max}")
print(f"weight_quantized_min {weight_quantized_min}")

model_32bit

model_8bit

model_4bit

model_16bit

import torch

torch.cuda.empty_cache()
start_mem = torch.cuda.memory_allocated(device=0)
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config_8bit,
    device_map="auto"
)
end_mem = torch.cuda.memory_allocated(device=0)
used_mem = end_mem - start_mem
print(f"8-bit model GPU memory usage: {used_mem/1024**2:.2f} MB")
