# ner_deepseek.py
import os
from openai import OpenAI

client = OpenClient(
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-xxx"),  # 免费领
    base_url="https://api.deepseek.com/v1"
)

SYSTEM_PROMPT = (
    "你是一名医学信息抽取专家。请从用户提问中抽取出所有实体，"
    "并以 JSON 列表返回，格式：[\"实体1\", \"实体2\", ...]。"
    "实体类型包括：疾病、药品、检查、手术、症状、解剖部位。"
)

def extract_entities_deepseek(text: str) -> list[str]:
    response = client.chat.completions.create(
        model="deepseek-chat",          # 即 DeepSeek-V2
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0.0,
        max_tokens=256
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return []      # 异常时返回空列表，可降级