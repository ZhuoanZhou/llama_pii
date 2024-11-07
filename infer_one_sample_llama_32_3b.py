import torch
import csv
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, MllamaForConditionalGeneration, AutoProcessor
from datasets import load_dataset, load_from_disk
from functools import partial
import re

#model_name = "meta-llama/Llama-2-13b-chat-hf" 
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto')

#output_dir = "meta-llama/Llama-3.2-11B-Vision-Instruct"
#tokenizer_dir = "meta-llama/Llama-3.2-11B-Vision-Instruct"
#tokenizer = AutoProcessor.from_pretrained(tokenizer_dir)

output_dir = "results/llama3_2_3b_qlora_pii/final_merged_checkpoint"
tokenizer_dir = "results/llama3_2_3b_qlora_pii/final_merged_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

#model = AutoModelForCausalLM.from_pretrained(output_dir, quantization_config=bnb_config, device_map = 'auto')
model = AutoModelForCausalLM.from_pretrained(output_dir, device_map = 'auto')
#model = MllamaForConditionalGeneration.from_pretrained(output_dir, device_map = 'auto')

#prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n    Identify and mask private identity information (PII) in the text.\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n    ### Text:\n{\n  "suicide_risk_assessment": [\n    {\n      "username": "bxsmjy7352",\n      "email": "80F@aol.com",\n      "id_card": "YQ30924HU",\n      "telephone": "+54.50 737 9577",\n      "country": "GB",\n      "address": {\n        "building": "421",\n        "street": "Measham Road",\n        "city": "Ashby-de-la-Zouch",\n        "state": "ENG",\n        "postcode": "LE65 2PF, LE65 2TX",\n        "secondary_address": "Loft 296"\n      \n\n### Masked text:\n    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

prompt = """Identify and mask private identity information (PII) in the text.

### Text:
**Guest Injury Report**

**Date:** 14/11/2008

**Time:** 17:52:39

**Location:** 684

---

**Guest 1 Details:**
- **Title:** Duchess
- **Name:** Puspita Langenick
- **IP Address:** e00e:f623:c39e:5a15:3da9:c968:c58c:e325
- **Password:** m7Fk7/J&\\
- **Check-in Time:** 8 PM

---

**Guest 2 Details:**
- **Title:** Arch

### Masked text:
"""

#input_ids = tokenizer(None, prompt, return_tensors="pt").input_ids.cuda()

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

print("length of promt tokens:",len(input_ids[0]))

start=time.time()
generated_ids = model.generate(input_ids, max_new_tokens=len(input_ids[0]),  do_sample=True, temperature=0.1, num_return_sequences=1, num_beams = 1)
print("runtime:",time.time()-start)

out = tokenizer.batch_decode(generated_ids)

# Function to extract the masked text
def find_all_non_empty_masked_texts(text):
    # Regular expression to find all subsets of text between "### Masked text:" and "### End of (anything)"
    pattern = r'### masked text:(.*?)### end.*?(\n|$)'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    non_empty_matches = [match[0].strip() for match in matches if match[0].strip()]
    if non_empty_matches:
        return non_empty_matches[0]
    else:
        return ""

out = find_all_non_empty_masked_texts(out[0])
print("input:\n", prompt)
print("\noutput:\n",out)
