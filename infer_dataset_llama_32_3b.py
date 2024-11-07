import torch
import csv
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
from functools import partial

#model_name = "meta-llama/Llama-2-13b-chat-hf" 
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto')

output_dir = "results/llama3_2_3b_v2/final_merged_checkpoint"
tokenizer_dir = "results/llama3_2_3b_v2/final_merged_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(output_dir, device_map = 'auto')

# set end token
model.generation_config.pad_token_id = tokenizer.pad_token_id

dataset_original = load_from_disk("pii-masking-300k-with-new-labels-v2/test")

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )
    
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
    
    blurb = f"{INTRO_BLURB}"

    instruction = f"{INSTRUCTION_KEY}\n{sample['source_text']}"
    #input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}\n"
    parts = [part for part in [blurb, instruction, response] if part]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt
    return sample

print("Preprocessing dataset...")
dataset = dataset_original.map(create_prompt_formats)#, batched=True)

# Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
_preprocessing_function = partial(preprocess_batch, max_length=1024, tokenizer=tokenizer)
dataset = dataset.map(
    _preprocessing_function,
    batched=True,
    #remove_columns=["sample_id", "Ranges", "number", "prompt", "response", "ground_truth", "CoT_demo_prompt", "text"],
)

print("size of test set before filter:", len(dataset))

# Filter out samples that have input_ids exceeding max_length
dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < 1024)

print("size of test set after filter:", len(dataset))

dataset_texts = dataset['text']

#print(dataset)

# Check if CUDA (GPU support) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("dataset:")
outputs = []
for i in tqdm(dataset.iter(batch_size=1),total=len(dataset)):
    input_ids = tokenizer(i['text'], return_tensors="pt").input_ids.cuda()
    print(len(input_ids[0]))
    generated_ids = model.generate(input_ids, max_new_tokens=len(input_ids[0])+25,  do_sample=True, temperature=0.1, num_return_sequences=1, num_beams = 1)
    out = tokenizer.batch_decode(generated_ids)
    outputs.append(out[0])
    #print(out[0])
    #break
dataset = dataset.add_column('output', outputs)
dataset.to_csv("llama32_3b_pii_test_output_v2.csv")