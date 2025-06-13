import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import json
import argparse

print("Starting script execution...")
output_dir = "results/minif2f/Godel-Prover-SFT"
n = 2
gpu=1
split = 'none'

print("Loading data from file...")
data_path = "./datasets/minif2f.jsonl"
# Initialize an empty list to hold the dictionaries
data_list = []

# Open the file and read each line
with open(data_path, 'r') as file:
    for line in file:
        # Parse the JSON object and append it to the list
        # if data_split is not None and prob['split'] not in data_split:
        #     continue
        data = json.loads(line)
        # if (data["split"] == args.split) or (args.split == "none"):
        #     data_list.append(data)
        if split == "none":
            data_list.append(data)
        else:
            try:
                int_split = int(split)
            except:
                int_split = None
                pass
            if isinstance(int_split, int):
                if (int(data["split"]) == int(split)):
                    data_list.append(data)
            else:
                if ((data["split"]) == (split)):
                    data_list.append(data)

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

print("Processing data and preparing model inputs...")
model_inputs = []
for data in data_list:
        # model_inputs.append("Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}\n/-{informal_prefix}-/ \n{formal_statement}".format(
        model_inputs.append("Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
                header=data.get('header', LEAN4_DEFAULT_HEADER),
                informal_prefix=data.get('informal_prefix', str()),
                formal_statement=data['formal_statement'],
            )
        )

print("Loading tokenizer...")
model_name = "Qwen/Qwen2.5-Coder-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully")

print("Starting LLM model initialization...")
print(f"Model name: {model_name}")
print(f"GPU count: {gpu}")
print(f"Max model length: 2048")
print(f"Swap space: 8")

try:
    print("Attempting to initialize LLM...")
    model = LLM(
        model=model_name,
        seed=1,
        trust_remote_code=True,
        swap_space=8,
        tensor_parallel_size=gpu,
        max_model_len=2048,
    )
    print("LLM model initialized successfully")
except Exception as e:
    print(f"Error during LLM initialization: {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    print(f"Full traceback:\n{traceback.format_exc()}")
    raise

print("Setting up sampling parameters...")
sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=2048,
    top_p=0.95,
    n=1,
)

print("Starting model generation...")
model_outputs = model.generate(
    model_inputs,
    sampling_params,
    use_tqdm=True,
)

assert len(model_outputs) == len(model_inputs)

def extrac_code(inputs):
    try:
        return re.search(r'```lean4\n(.*?)\n```', inputs, re.DOTALL).group(1)
    except:
        return "None"

to_inference_codes = []
for i in range(len(data_list)):
    data_list[i]["model_input"] = model_inputs[i]
    data_list[i]["model_outputs"] = [output.text for output in model_outputs[i].outputs]
    data_list[i]["full_code"] = [extrac_code(model_inputs[i] + output.text) for output in model_outputs[i].outputs]
    if "problem_id" in data_list[i]:
        to_inference_codes += [{"name": data_list[i]["problem_id"], "code": code} for code in data_list[i]["full_code"]]
    else:
        to_inference_codes += [{"name": data_list[i]["name"], "code": code} for code in data_list[i]["full_code"]]

import os
os.makedirs(output_dir, exist_ok=True)

output_file_path = F'{output_dir}/full_records.json'
print(F"Outputting to {output_file_path}")
# Dump the list to a JSON file with indents
with open(output_file_path, 'w') as json_file:
    json.dump(data_list, json_file, indent=4)

toinfer_file_path = F'{output_dir}/to_inference_codes.json'
print(F"Outputting to {toinfer_file_path}")
# Dump the list to a JSON file with indents
with open(toinfer_file_path, 'w') as json_file:
    json.dump(to_inference_codes, json_file, indent=4)
# for data
#     model_outputs[0].outputs[0].text
