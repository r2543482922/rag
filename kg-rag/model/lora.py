from peft import LoraConfig, get_peft_model
from config.settings import settings

def apply_lora(model):
    config = LoraConfig(
        r=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        target_modules=settings.lora_targets,
        lora_dropout=settings.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, config)