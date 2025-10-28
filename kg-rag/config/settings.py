from dataclasses import dataclass


@dataclass
class Settings:
    # 模型参数
    model_name: str = "deepseek-ai/deepseek-llm-7b"
    tokenizer_path: str = "deepseek-ai/deepseek-llm-7b"

    # LoRA参数
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_targets: list = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # 知识图谱参数
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_auth: tuple = ("neo4j", "password")
    max_kg_triples: int = 5

    # 训练参数
    batch_size: int = 4
    learning_rate: float = 1e-5
    epochs: int = 10
    cutoff_len: int = 2048  # 适配DeepSeek长上下文


settings = Settings()