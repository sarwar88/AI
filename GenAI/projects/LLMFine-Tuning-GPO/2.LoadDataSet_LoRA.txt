!pip install -q -U transformers datasets bitsandbytes  trl peft  huggingface_hub

from huggingface_hub import notebook_login

notebook_login()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# model_name = "Cagatayd/llama3.2-1B-Instruct-Egitim"
# model_name = "unsloth/Llama-3.2-1B-Instruct"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

config_8bit = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(


                                                  model_name,
                                                  quantization_config=config_8bit,
                                                  device_map= "auto",
                                                  trust_remote_code =True

                                                  )


model_8bit

tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left",trust_remote_code= True)

from datasets import load_dataset

dataset = load_dataset("MattCoddity/dockerNLcommands")
dataset

dataset['train'][0]

from datasets import DatasetDict

train_val_split = dataset['train'].train_test_split(test_size=0.2, seed=42)

dataset = DatasetDict(

                      {'train':train_val_split['train'],
                       'validation':train_val_split['test']}

)
dataset

def to_chat_template(example):


    messages = [

                {"role": 'system','content':example['instruction']},
                {"role": 'user','content':example['input']},
                {"role": 'assistant','content':example['output']}

                ]


    return {'text':messages}

dataset = dataset.map(to_chat_template)
dataset



dataset['train']['text'][0]

tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer_mistral.apply_chat_template(dataset['train']['text'][0],tokenize=False)

tokenizer.apply_chat_template(dataset['train']['text'][0],tokenize=False)

def apply_chat_temp(example):

    new_text = tokenizer.apply_chat_template(example['text'],tokenize=False)

    return {'text':new_text}


dataset = dataset.map(apply_chat_temp)
dataset

dataset['train']['text'][0]

tokenizer(dataset['train']['text'][0])

def tokenize_fn(example):

    return tokenizer(example['text'])


tokenized_dataset = dataset.map(tokenize_fn,batched=True,remove_columns=['input', 'output', 'instruction', 'text'])
tokenized_dataset

len(tokenized_dataset['train']['input_ids'][1])

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False,return_tensors="pt")


tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token

from torch.utils.data import DataLoader


data_loader = DataLoader(
    tokenized_dataset['train'],
    batch_size=2,
    collate_fn=data_collator
)
for step, batch in enumerate(data_loader):
    print(f"Batch {step}")
    print("input_ids.shape:", batch["input_ids"].shape)

    print("attention_mask.shape:", batch["attention_mask"].shape)


    if step >= 3:
        break

# 2 2

# [1,2,3,4,5,6]
# [pad,pad,pad,1,2,3] -100



# [1,2,3,4,5]
# [pad,pad,1,2,3]


model_8bit

from peft import  LoraConfig, get_peft_model
import copy

model_8bit_clone = copy.deepcopy(model_8bit)


lora_config = LoraConfig(

                         r= 64,
                         lora_alpha = 32,
                         target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
                         lora_dropout = 0.05,
                         bias = "none",
                         task_type = "CAUSAL_LM"

)

model_8bit_lora = get_peft_model(model_8bit_clone,lora_config)
model_8bit_lora.print_trainable_parameters()

model_8bit

model_8bit_lora

from transformers import TrainingArguments
from trl import SFTTrainer


training_args = TrainingArguments(

                                  output_dir = "./resutls",
                                  per_device_train_batch_size = 2,
                                  per_device_eval_batch_size = 2,
                                  eval_steps = 10,
                                  evaluation_strategy = "steps",
                                  save_steps = 20,
                                  save_strategy = "steps",
                                  # num_train_epochs = 1,
                                  max_steps = 60,
                                  learning_rate = 3e-5,
                                  weight_decay = 0.01,
                                  warmup_steps = 5,
                                  fp16 = True,
                                  gradient_accumulation_steps = 4,
                                  optim = "adamw_torch",
                                  logging_steps = 10,
                                  report_to = "none"


                                  )

tokenized_dataset

trainer = SFTTrainer(

                     model =model_8bit_lora,
                     train_dataset =tokenized_dataset['train'],
                     eval_dataset = tokenized_dataset['validation'],
                     args = training_args,
                     data_collator= data_collator,
                     tokenizer = tokenizer
)

trainer.train()

model_8bit_lora

import torch

torch.set_printoptions(precision=8, sci_mode=False)

layer0 = model_8bit_lora.model.model.layers[0]

model_8bit_weights = layer0.self_attn.q_proj.weight
print("model_8bit_weights:", model_8bit_weights)

import torch

torch.set_printoptions(precision=8, sci_mode=False)

layer0 = model_8bit_lora.model.model.layers[0]

# LoRA A matrisi
weight_A = layer0.self_attn.q_proj.lora_A["default"].weight
print("LoRA A weight:", weight_A)

weight_A.dtype


base_model = AutoModelForCausalLM.from_pretrained(


                                                  model_name,

                                                  device_map= "auto",
                                                  trust_remote_code =True

                                                  )


base_model

lora_path = "./lora_adapters"
model_8bit_lora.save_pretrained(lora_path)

from peft import PeftModel

base_model =PeftModel.from_pretrained(base_model,lora_path)
base_model

base_model = base_model.merge_and_unload()
base_model

base_model.push_to_hub("Cagatayd/Llama3.2-doker-egitim")
tokenizer.push_to_hub("Cagatayd/Llama3.2-doker-egitim")

# import torch
# from transformers import pipeline

# model_id = "Cagatayd/Llama3.2-doker-egitim"
# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]
# outputs = pipe(
#     messages,
#     max_new_tokens=256,
# )
# print(outputs[0]["generated_text"][-1])


dataset['train'][1]

import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    tokenizer = tokenizer
)
messages = [
    {"role": "system", "content": "translate this sentence in docker command"},
    {"role": "user", "content": "Please show me the Docker containers that have exited and are related to the mongo image."},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1]['content'])


import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    tokenizer = tokenizer
)
messages = [
    {"role": "system", "content": "translate this sentence in docker command"},
    {"role": "user", "content": "Please show me the Docker containers that have exited and are related to the mongo image."},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
    temperature = 0.5,
    top_p  = 0.9,
    top_k = 10,
    do_sample = True,
    repetition_penalty = 1.2


)
print(outputs[0]["generated_text"][-1]['content'])


