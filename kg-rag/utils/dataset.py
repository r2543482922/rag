from torch.utils.data import Dataset
from kg.retriever import KGRetriever
from transformers import AutoTokenizer
from config.settings import settings
import torch


class MedicalDataset(Dataset):
    def __init__(self, file_path: str):
        self.data = self._load_data(file_path)
        self.retriever = KGRetriever()
        self.tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_path)

    def _load_data(self, path):
        with open(path) as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        retrieved = self.retriever.retrieve(item["question"])

        # 构建Prompt
        prompt = f"知识：{retrieved['texts']}\nKG：{retrieved['kg_triples']}\n问题：{item['question']}\n答案："

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=settings.cutoff_len,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0).clone()
        }