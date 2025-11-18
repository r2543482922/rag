#!/usr/bin/env python3
"""
Chinese-LLaMA-2-7B + LoRA + RAG 微调脚本
"""
import os
import sys
from typing import List, Optional

import fire
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import TrainerCallback, TrainerState, TrainerControl

# ------------------------------------------------------------------
# 可选：RAG 检索器（如不需要，把 USE_RAG=False）
# ------------------------------------------------------------------
USE_RAG = False          # 如需 RAG，改为 True 并保证 retriever.py 存在
if USE_RAG:
    from retriever import MedRetriever

# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
class Prompter:
    """
    最小化 prompt 模板加载器
    """
    import json
    import os.path as osp
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
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
        # 将知识拼到 input 之前
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
# Trainer callback：只保存 adapter
# ------------------------------------------------------------------
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        kwargs["model"].save_pretrained(checkpoint_folder)
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


# ------------------------------------------------------------------
# 主训练函数
# ------------------------------------------------------------------
def train(
    base_model: str = "ziqingyang/chinese-llama-2-7b",
    data_path: str = "data/llama_data.json",
    output_dir: str = "./lora-chinese-llama2-med",
    # 训练超参
    batch_size: int = 128,
    micro_batch_size: int = 64,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 500,
    # LoRA 超参
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    # LLM 超参
    train_on_inputs: bool = False,
    group_by_length: bool = False,
    # wandb
    wandb_project: str = "llama_med",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: Optional[str] = None,
    prompt_template_name: str = "med_template",
):
    print("LoRA fine-tuning Chinese-LLaMA-2-7B …")
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # wandb 初始化
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_watch:
        os.environ["WANDB_WATCH"] = wandb_watch
    if wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # ------------------------------------------------------------------
    # 模型 & tokenizer
    # ------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model.tie_weights()  # 关键：消除 transformers>=4.35 的报错

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.pad_token_id or 0
    tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # 可选：RAG 检索器
    # ------------------------------------------------------------------
    retriever = None
    if USE_RAG:
        retriever = MedRetriever(index_path="med.index")

    # ------------------------------------------------------------------
    # 数据处理
    # ------------------------------------------------------------------
    def tokenize(prompt: str, add_eos: bool = True):
        res = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            add_eos
            and res["input_ids"][-1] != tokenizer.eos_token_id
            and len(res["input_ids"]) < cutoff_len
        ):
            res["input_ids"].append(tokenizer.eos_token_id)
            res["attention_mask"].append(1)
        res["labels"] = res["input_ids"].copy()
        return res

    def generate_and_tokenize_prompt(data_point):
        instruction = data_point["instruction"]
        inp = data_point.get("input", "")
        outp = data_point["output"]

        knowledge = ""
        if retriever:
            query = f"{instruction} {inp}".strip()
            knowledge = retriever.retrieve(query)

        full_prompt = prompter.generate_prompt(
            instruction=instruction,
            input=inp,
            label=outp,
            knowledge=knowledge,
        )
        tokenized = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                instruction=instruction, input=inp, knowledge=knowledge
            )
            user_len = len(tokenize(user_prompt, add_eos=False)["input_ids"])
            tokenized["labels"] = [-100] * user_len + tokenized["labels"][user_len:]
        return tokenized

    # ------------------------------------------------------------------
    # 数据集加载 & 划分
    # ------------------------------------------------------------------
    data = load_dataset("json", data_files=data_path)
    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, seed=2023)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    model = prepare_model_for_int8_training(model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.1,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=8,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=32 if val_set_size > 0 else None,
            save_steps=32,
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=bool(val_set_size > 0),
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if wandb_project else None,
            run_name=wandb_run_name if wandb_project else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[SavePeftModelCallback],
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)
    print("\nLoRA adapter saved to", output_dir)


# ------------------------------------------------------------------
# Fire CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    fire.Fire(train)