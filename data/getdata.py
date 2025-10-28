from datasets import load_dataset

# 加载数据集
dataset = load_dataset("yahma/alpaca-cleaned")

# 保存数据集到本地
dataset.save_to_disk("/path/to/save/dataset")

# git clone git@hf.co:tloen/alpaca-lora-7b


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("baffo32/decapoda-research-llama-7B-hf")
model = AutoModelForCausalLM.from_pretrained("baffo32/decapoda-research-llama-7B-hf")
