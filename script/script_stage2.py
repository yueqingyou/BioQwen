import os
import math
import copy
import torch
import bitsandbytes as bnb
from typing import Any, Dict, List
from trl import get_kbit_device_map
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          BitsAndBytesConfig, 
                          Trainer, 
                          TrainingArguments,
                          pipeline)

train_mode = 'qlora'
max_length = 1024

# merged weight by notebook's relative section
model_path = 'BioQwen-stage1-merged/'
print(f'model_path: {model_path}')

###### DATA ######
###### NER + RE + MT 18w + QA 18w + GC 3w = 404190 ######
general_data = load_dataset('json', data_files='../data/COIG-CQIA/COIG-CQIA-full.jsonl')
selected_general_data = general_data.map(remove_columns=['task_type', 'domain', 'metadata', 'answer_from', 'human_verified', 'copyright'])
selected_general_data = selected_general_data['train']
selected_qa_data = load_from_disk('filtered_qa_data/')
other_data = load_dataset('json', data_files='../data/Taiyi_Instruction_Data_001/Taiyi_Instruction_Data_001.jsonl')
filtered_other_data = other_data.filter(lambda x: x['category'] in ['NER', 'RE', 'MT-zh2en', 'MT-en2zh'])
def filter_fields(example):
    example = example['conversation'][0]
    return {
        'instruction': example['human'],
        'input': '',
        'output': example['assistant']
    }
selected_filtered_other_data = filtered_other_data.map(filter_fields, remove_columns=filtered_other_data['train'].column_names)
selected_filtered_other_data = selected_filtered_other_data['train']
cognition_data = load_dataset('json', data_files='../data/self_cognition.json')
cognition_data = cognition_data['train']

tokenizer = AutoTokenizer.from_pretrained(model_path)
selected_general_data = selected_general_data.filter(lambda x: len(tokenizer.encode(x['instruction'] + x['input'] + x['output'])) < max_length - 100)
selected_filtered_other_data = selected_filtered_other_data.filter(lambda x: len(tokenizer.encode(x['instruction'] + x['input'])) < (max_length * 0.5) and len(tokenizer.encode(x['output'])) < (max_length * 0.5))

stage2_data = concatenate_datasets([selected_qa_data, selected_general_data, cognition_data, selected_filtered_other_data])
print(stage2_data)

def check_lang_fast(batch):
    langs = []
    
    for text in batch['instruction']:
        english_count, chinese_count = 0, 0
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                chinese_count += 1
            elif 'a' <= char.lower() <= 'z':
                english_count += 1
        lang = 'zh' if chinese_count > english_count else 'en'
        langs.append(lang)
    batch['lang'] = langs
    
    return batch

add_lang_stage2_data = stage2_data.map(check_lang_fast, batched=True)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
# tokenizer.add_special_tokens({'pad_token': '<|padoftext|>'})
# tokenizer.pad_token_id = 151646
# for flash atten
tokenizer.padding_side = 'left'
print(tokenizer)

def tokenize_chatml(batch):
    input_ids = []
    attention_mask = []
    target_mask = []

    zh_system = "你是千问生物智能助手，一个专注于生物领域的先进人工智能。"
    en_system = "You are BIO-QWEN, an advanced AI specializing in the field of biology."
    
    for lang, instruction, input_text, output in zip(batch['lang'], batch['instruction'], batch['input'], batch['output']):
        system_greeting = zh_system if lang == 'zh' else en_system
        system_format = f'<|im_start|>system\n{system_greeting}<|im_end|>\n'
        user_format = f'<|im_start|>user\n{instruction.strip() + input_text.strip()}<|im_end|>\n<|im_start|>assistant\n'
        assistant_format = f'{output.strip()}<|im_end|>\n'
        chatml_text = f'{system_format}{user_format}{assistant_format}'

        encoded_length = len(tokenizer.encode(chatml_text, add_special_tokens=False))
        min_length = len(tokenizer.encode(system_format + user_format, add_special_tokens=False))
        
        while encoded_length > max_length:
            last_newline = chatml_text.rfind('\n')
            if last_newline > min_length:
                chatml_text = chatml_text[:last_newline]
                encoded_length = len(tokenizer.encode(chatml_text, add_special_tokens=False))
            else:
                chatml_text = chatml_text[:max_length]
                break
                
        if not chatml_text.endswith('<|im_end|>\n'):
            chatml_text += '<|im_end|>\n'

        chatml_text += tokenizer.eos_token
                
        tokenized = tokenizer(chatml_text, add_special_tokens=False, return_tensors="pt")
        input_ids.append(tokenized.input_ids.squeeze().tolist())
        attention_mask.append(tokenized.attention_mask.squeeze().tolist())

        start_idx = len(tokenizer.encode(system_format, add_special_tokens=False)) + len(tokenizer.encode(user_format, add_special_tokens=False))
        end_idx = len(tokenizer.encode(chatml_text, add_special_tokens=False))
        label = [0] * start_idx + [1] * (end_idx - start_idx)
        target_mask.append(label)

        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        target_mask = target_mask[:max_length]
        assert len(input_ids) == len(attention_mask) == len(target_mask)

    batch['input_ids'] = input_ids
    batch['attention_mask'] = attention_mask
    batch['target_mask'] = target_mask

    return batch

tokenized_stage2_data = add_lang_stage2_data.map(tokenize_chatml, 
                                                 batched=True, 
                                                 remove_columns=add_lang_stage2_data.column_names, 
                                                 num_proc=64
                                                )

###### Stage 2 Model ######
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
                                             model_path, 
                                             torch_dtype=torch.bfloat16, 
                                             attn_implementation='flash_attention_2',  
                                             quantization_config=bnb_config if train_mode == 'qlora' else None,
                                             device_map=get_kbit_device_map() if train_mode == 'qlora' else None,
                                            )

if train_mode == 'qlora':
    # QLora: casts all the non int8 modules to full precision (fp32) for stability
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    # Disable caching to be compatible with gradient checkpointing
    model.config.use_cache = False
elif train_mode == 'lora':
    # LoRA: Enables the gradients for the input embeddings
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
# tokenizer.add_special_tokens({'pad_token': '<|padoftext|>'})
# tokenizer.pad_token_id = 151646

def find_all_linear_names(model, train_mode):
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    return lora_module_names

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=find_all_linear_names(model, train_mode),
    bias='none',
    task_type='CAUSAL_LM'
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

class SFTDataCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        batch_max_len = min(max(lengths), self.max_seq_length)

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        # truncate and padding
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = batch_max_len - len(input_ids)

            # flash atten padding_side=left
            # Ensure padding is applied to the left
            input_ids = [self.pad_token_id] * padding_len + input_ids
            attention_mask = [0] * padding_len + attention_mask
            target_mask = [0] * padding_len + target_mask
            
            # Truncate from the right if necessary
            input_ids = input_ids[-self.max_seq_length:]
            attention_mask = attention_mask[-self.max_seq_length:]
            target_mask = target_mask[-self.max_seq_length:]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels
        }
        return inputs

data_collator = SFTDataCollator(tokenizer, max_length)

###### Stage 2 Train ######
split_stage2_data = tokenized_stage2_data.train_test_split(test_size=500, seed=42)
train_data = split_stage2_data['train']
val_data = split_stage2_data['test']
world_size = torch.cuda.device_count()
num_train_epochs = 5
per_device_train_batch_size = 4
per_device_eval_batch_size = per_device_train_batch_size
gradient_accumulation_steps = 4
weight_decay = 1e-5
weight_decay = 0
training_nums = len(train_data)
warmup_ratio = 0.1
learning_rate = 2e-4
logging_steps = 10

batch_size = per_device_train_batch_size * world_size * gradient_accumulation_steps
t_total = training_nums // batch_size * num_train_epochs
warmup_steps = int(t_total * warmup_ratio) if warmup_ratio > 0.0 else warmup_steps

save_steps = eval_steps = t_total // 10
print(t_total, eval_steps, save_steps, logging_steps, warmup_steps)

training_args = TrainingArguments(
    'BioQwen-stage2',
    per_device_train_batch_size=per_device_train_batch_size, # 8
    per_device_eval_batch_size=per_device_eval_batch_size, # 8
    # evaluation_strategy='epoch',
    evaluation_strategy='steps',
    eval_steps=eval_steps,
    gradient_accumulation_steps=gradient_accumulation_steps, # 2
    num_train_epochs=num_train_epochs, # 5
    weight_decay=weight_decay,
    warmup_steps=warmup_steps,
    lr_scheduler_type='constant_with_warmup',
    learning_rate=learning_rate,
    save_steps=eval_steps,
    # fp16=True,
    # resume_from_checkpoint=True,
    logging_steps=logging_steps,
    report_to='wandb',
    remove_unused_columns=False,
    optim='paged_adamw_32bit',
    seed=42,
    max_grad_norm=0.3,
    ddp_find_unused_parameters=False,
    bf16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
