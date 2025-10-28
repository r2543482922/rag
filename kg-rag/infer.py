from transformers import AutoTokenizer, pipeline
from model.lora import apply_lora
from kg.retriever import KGRetriever
from config.settings import settings
import torch
from transformers import (
    AutoModelForCausalLM,
)


def infer(question: str):
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        device_map="auto"
    )
    model = apply_lora(model).eval()

    # 知识检索
    retriever = KGRetriever()
    context = retriever.retrieve(question)

    # 构建Prompt
    prompt = f"知识：{context['texts']}\nKG：{context['kg_triples']}\n问题：{question}\n答案："

    # 生成回答
    tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_path)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(infer("糖尿病有哪些常见并发症？"))