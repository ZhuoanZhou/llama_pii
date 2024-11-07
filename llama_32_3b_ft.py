# Train Llama 3.2 3b instruct on "pii-masking-300k-with-new-labels-v2"

import argparse
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
import datasets
from functools import partial
import os
from peft import (
    get_peft_config,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
    PeftModel
    )
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from datasets import load_dataset
import re
from transformers import PreTrainedTokenizerFast

import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

hf_oken = "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"

from tqdm import tqdm, trange

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    print('Num of GPUs', n_gpus)
    max_memory = f'{40960}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range( n_gpus)},
    )

    #model = AutoModelForCausalLM.from_pretrained(
            #model_name,
            #quantization_config=bnb_config,
            #device=7,
    #        )
    #tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    path = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    #tokenizer = PreTrainedTokenizerFast(model_name)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# Load dataset from huggingFace
#dataset = load_dataset("ai4privacy/pii-masking-300k")
#english_dataset = dataset.filter(lambda example: example['language'] == 'English')
#eval_dataset  = english_dataset["validation"]
#dataset = english_dataset["train"]

dataset_path = "pii-masking-300k-with-new-labels-v2/train/train"

dataset = load_from_disk(dataset_path)
eval_dataset_path = "pii-masking-300k-with-new-labels-v2/train/test"
eval_dataset  = load_from_disk(eval_dataset_path)
#eval_dataset  = dataset["test"]
#dataset = dataset["train"]

print(f'Number of prompts: {len(dataset)}')
print(f'Column names are: {dataset.column_names}')

print(f'Number of prompts: {len(eval_dataset)}')
print(f'Column names are: {eval_dataset.column_names}')

def create_prompt_formats(sample):

    """
    Format various fields of the sample ('source_text', 'target_text')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """

    #INTRO_BLURB = "Extract the explicitly mentioned soft skills from the following sentence. If soft skills are not mentioned answer “None”. Only soft skills from the following list are allowed:\n"
    INTRO_BLURB = "Identify and mask private identity information (PII) in the text."
    INSTRUCTION_KEY = "### Text:"
    #INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Masked text:"
    END_KEY = "### End"
    
    blurb = f"{INTRO_BLURB}"

    instruction = f"{INSTRUCTION_KEY}\n{sample['source_text']}"
    #input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}\n{sample['target_text']}"
    end = f"{END_KEY}"
    parts = [part for part in [blurb, instruction, response, end] if part]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt
    return sample

print(create_prompt_formats(dataset[0]))

# It reformulate the data into the following format:
'''
Identify and mask private identity information (PII) in the text.

### Text:
<input text>

### Masked text:
<output text>

### End
'''

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)#, batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        #remove_columns=["context", "response", "text", "category"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config
def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config

def create_ptuning_config(modeuls):
    config = PromptEncoderConfig(task_type="CAUSAL_LM", num_virtual_tokens=40, encoder_hidden_size=128)
    return config

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

#model_name = "meta-llama/Meta-Llama-3-8B" 
model_name = "meta-llama/Llama-3.2-3B-Instruct"

bnb_config = create_bnb_config()

model, tokenizer = load_model(model_name, bnb_config)

max_length = get_max_length(model)

dataset = preprocess_dataset(tokenizer, max_length, 1234, dataset)
eval_dataset = preprocess_dataset(tokenizer, max_length, 1234, eval_dataset)
def train(model, tokenizer, dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    #ptune_config = create_ptuning_config(modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset = eval_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=1000,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            eval_strategy = 'steps',
            #evaluation_strategy = 'steps',
            eval_steps = 50,
            save_steps = 50,
            load_best_model_at_end=True,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],

    )

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    ###

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    print(trainer.evaluate())
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


output_dir = "results/llama3_2_3b_v2/final_checkpoint"
train(model, tokenizer, dataset, output_dir)
lora_flag =True
if lora_flag:
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map='auto', torch_dtype=torch.bfloat16)
    #model = PeftModel.from_pretrained(output_dir, device_map = 'auto', torch_dtype = torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = "results/llama3_2_3b_v2/final_merged_checkpoint"
    os.makedirs(output_merged_dir, exist_ok=True)
    model.save_pretrained(output_merged_dir, safe_serialization=True)

    # save tokenizer for easy inference
    #path = "/home/tuf72076/pennmutual/results/enumeration_v2/final_merged_checkpoint"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_merged_dir)
else:
    #from huggingface_hub import notebook_login
    #notebook_login()
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map='auto', torch_dtype=torch.bfloat16)
    #model = AutoModelForCausalLM.from_pretrained(
    #    model_name,
    #    quantization_config=bnb_config,
    #    device_map="auto", # dispatch efficiently the model on the available ressources
    #    #max_memory = {i: max_memory for i in range( n_gpus)},
    #)
    #model = PeftModel.from_pretrained(model, output_dir)

    #model = model.merge_and_unload()
    model.push_to_hub("Misaka19487/llama2_indeed_r4", create_pr=1)
    #output_merged_dir = "results/llama2-13b-trina-ptune/final_merged_checkpoint"
    #os.makedirs(output_merged_dir, exist_ok=True)
    #model.save_pretrained(output_merged_dir, safe_serialization=True)

    # save tokenizer for easy inference
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #tokenizer.save_pretrained(output_merged_dir)

    
    #from huggingface_hub import notebook_login
    #notebook_login()
