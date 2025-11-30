# dpo_train.py

import os
import sys
from typing import List, Optional

import fire
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, set_peft_model_state_dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import DPOTrainer  # 引入 DPOTrainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import TrainerCallback, TrainerState, TrainerControl


# ------------------------------------------------------------------
# Prompter 类 (从您的 SFT 脚本复制)
# ------------------------------------------------------------------
class Prompter:
    import json
    import os.path as osp
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        # 假设 templates 文件夹存在，且包含 med_template.json
        file_name = self.osp.join("templates", f"{template_name}.json")
        if not self.osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name, encoding="utf-8") as fp:
            self.template = self.json.load(fp)

    def generate_prompt(
            self,
            instruction: str,
            input: Optional[str] = None,
            label: Optional[str] = None,
            knowledge: Optional[str] = None,
    ) -> str:
        # 注意: 在 DPO 数据中，RAG context 已经拼接到 instruction 中了
        if knowledge:
            input = (knowledge + "\n\n" + (input or "")).strip()

        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        return res


# ------------------------------------------------------------------
# 主 DPO 训练函数
# ------------------------------------------------------------------
def train_dpo(
        base_model: str = "ziqingyang/chinese-llama-2-7b",
        data_path: str = "single_turn_dpo_preference_data.jsonl",  # ⚠️ 使用生成的数据文件
        output_dir: str = "./lora-dpo-chinese-llama2-med",
        # 训练超参
        batch_size: int = 16,
        micro_batch_size: int = 2,
        num_epochs: int = 1,  # DPO 训练 epoch 不宜过多
        learning_rate: float = 5e-7,  # ⚠️ DPO 学习率应非常低
        cutoff_len: int = 512,
        # DPO 特有超参
        dpo_beta: float = 0.1,  # 控制偏好强度
        # LoRA 超参 (建议 R 值大一点，目标模块多一点)
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"],
        # 其他
        group_by_length: bool = False,
        wandb_project: str = "llama_med_dpo",
        wandb_run_name: str = "",
        resume_from_checkpoint: Optional[str] = None,
        prompt_template_name: str = "med_template",
        load_in_4bit: bool = True,  # 启用 4bit 量化以节省显存
):
    print("DPO fine-tuning Chinese-LLaMA-2-7B with RAG Feedback...")
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    # DDP Setup
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Quantization Setup
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # ------------------------------------------------------------------
    # 1. 模型加载 (Policy Model & Reference Model)
    # ------------------------------------------------------------------
    print("⏳ Loading Policy Model (trainable) and Reference Model (frozen)...")

    # Policy Model (参与训练)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    # Reference Model (冻结参数)
    model_ref = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.pad_token_id or 0
    tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # 2. 数据加载 & 格式化
    # ------------------------------------------------------------------
    def preprocess_dpo_data(data_point):
        # ⚠️ 确保数据格式与 create_dpo_data.py 的输出一致
        return {
            "prompt": data_point["prompt"],
            "chosen": data_point["chosen"],
            "rejected": data_point["rejected"],
        }

    data = load_dataset("json", data_files=data_path)
    # DPO 不需要 val_set，因为偏好数据集通常较小且训练目标明确
    train_data = data["train"].shuffle(seed=2023).map(preprocess_dpo_data)

    # ------------------------------------------------------------------
    # 3. LoRA 配置
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ------------------------------------------------------------------
    # 4. Trainer
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=8,
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=32,
        output_dir=output_dir,
        save_total_limit=3,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if wandb_project else None,
        run_name=wandb_run_name if wandb_project else None,
    )

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=dpo_beta,
        train_dataset=train_data,
        tokenizer=tokenizer,
        peft_config=lora_config,
        max_length=cutoff_len,
        max_prompt_length=cutoff_len,
    )

    # 5. 训练
    model.config.use_cache = False

    # Resume Checkpoint Logic (Simplified for DPO LoRA)
    if resume_from_checkpoint:
        print(f"Restarting from {resume_from_checkpoint}")
        adapter_path = os.path.join(resume_from_checkpoint, "adapter_model.bin")
        if os.path.exists(adapter_path):
            adapters_weights = torch.load(adapter_path)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print("Adapter checkpoint not found.")

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    dpo_trainer.train()

    # 6. 保存 DPO 后的 LoRA 权重
    dpo_trainer.model.save_pretrained(output_dir)
    print("\nDPO LoRA adapter saved to", output_dir)


if __name__ == "__main__":
    fire.Fire(train_dpo)