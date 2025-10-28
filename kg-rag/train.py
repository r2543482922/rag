from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from model.lora import apply_lora
from model.kg_fusion import KnowledgeFusion
from utils.dataset import MedicalDataset
from config.settings import settings
import torch


class KGTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fusion = KnowledgeFusion(settings.hidden_size)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(input_ids=inputs["input_ids"])
        fused = self.fusion(outputs.last_hidden_state, inputs["kg_data"])
        loss = model.compute_loss(fused, inputs["labels"])
        return (loss, outputs) if return_outputs else loss


def train():
    # 初始化模型
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = apply_lora(model)

    # 数据加载
    dataset = MedicalDataset("./data/raw/train.jsonl")

    # 训练参数
    args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.epochs,
        logging_dir="./logs",
        save_strategy="epoch",
        fp16=True
    )

    # 开始训练
    trainer = KGTrainer(
        model=model,
        args=args,
        train_dataset=dataset
    )
    trainer.train()


if __name__ == "__main__":
    train()