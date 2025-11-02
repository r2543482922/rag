# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
finetune.py - LoRA/PEFT 微调脚本（改进版）
主要改进点：
- 更稳健的 tokenizer pad token 处理（避免直接写 pad_token_id = 0）
- 更健壮的 resume_from_checkpoint 检测（避免传 False 给 Trainer.train）
- 保存训练 metadata（config + 版本信息）
- 设置随机 seed 以便复现
- 在使用 8-bit 模式时避免盲目 resize embeddings
- 改进 SavePeftModelCallback，确保能保存 PEFT adapter 格式
"""
import os
import sys
import json
import random
import platform
from typing import List, Optional

import fire
import torch
import transformers
import wandb
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_tokenizer_has_pad(tokenizer, model, load_in_8bit: bool):
    """
    确保 tokenizer 有 pad_token:
    - 优先使用 eos_token 作为 pad（不需要 resize embedding）
    - 如果没有 eos_token，则添加 [PAD] 并在非 8-bit 情况下 resize embedding
    """
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Tokenizer had no pad_token; set pad_token = eos_token")
        else:
            # 添加 pad token（会改变 vocab size，需 resize embeddings）
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print("Added [PAD] token to tokenizer")
            if not load_in_8bit:
                try:
                    model.resize_token_embeddings(len(tokenizer))
                    print("Resized model embeddings to match new tokenizer length")
                except Exception as e:
                    print("Warning: resize_token_embeddings failed or unsupported in current mode:", e)
    return tokenizer


def train(
        # model/data params
        base_model: str = "",  # required
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 8,
        num_epochs: int = 10,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 500,
        seed: Optional[int] = 42,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # llm hyperparams
        train_on_inputs: bool = False,
        group_by_length: bool = False,
        # wandb
        wandb_project: str = "llama_med",
        wandb_run_name: str = "",
        wandb_watch: str = "",
        wandb_log_model: str = "",
        resume_from_checkpoint: Optional[str] = None,
        prompt_template_name: str = "alpaca",
        # device / model loading
        load_in_8bit: bool = True,
):
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"load_in_8bit: {load_in_8bit}\n"
        )

    assert base_model, "Please specify a --base_model"

    set_seed(seed)

    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK") or 0)
        device_map = {"": local_rank}
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

    # wandb env
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    # ensure pad token is set safely
    tokenizer = ensure_tokenizer_has_pad(tokenizer, model, load_in_8bit)

    tokenizer.padding_side = "left"  # for batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                len(result.get("input_ids", [])) > 0
                and result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        # expects data_point has instruction, input (optional), output
        full_prompt = prompter.generate_prompt(
            data_point.get("instruction", ""),
            data_point.get("input", ""),
            data_point.get("output", None),
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(data_point.get("instruction", ""), data_point.get("input", ""))
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model) if load_in_8bit else model

    # LoRA config
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # load dataset
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # resume logic: check if resume path exists and contains adapter weights
    resume_checkpoint = None
    if resume_from_checkpoint:
        if os.path.isdir(resume_from_checkpoint):
            # try several known names
            adapter_bin = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            pytorch_bin = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
            # Also accept HuggingFace save_pretrained folder that contains adapter_config.json or adapter_model.bin
            if os.path.exists(adapter_bin):
                resume_checkpoint = resume_from_checkpoint
            elif os.path.exists(pytorch_bin):
                resume_checkpoint = resume_from_checkpoint
            else:
                # maybe user passed path to adapter file directly
                if os.path.exists(resume_from_checkpoint) and resume_from_checkpoint.endswith(".bin"):
                    resume_checkpoint = resume_from_checkpoint
        elif os.path.exists(resume_from_checkpoint):
            resume_checkpoint = resume_from_checkpoint

        if resume_checkpoint:
            print(f"Resuming from checkpoint: {resume_checkpoint}")
            try:
                # try to load adapter/pytorch file into PEFT model
                if os.path.isdir(resume_checkpoint):
                    adapter_path = os.path.join(resume_checkpoint, "adapter_model.bin")
                    pytorch_path = os.path.join(resume_checkpoint, "pytorch_model.bin")
                    if os.path.exists(adapter_path):
                        adapters_weights = torch.load(adapter_path, map_location="cpu")
                    elif os.path.exists(pytorch_path):
                        adapters_weights = torch.load(pytorch_path, map_location="cpu")
                    else:
                        # try peft saved folder: let PeftModel.from_pretrained handle it in downstream
                        adapters_weights = None
                    if adapters_weights is not None:
                        set_peft_model_state_dict(model, adapters_weights)
                else:
                    adapters_weights = torch.load(resume_checkpoint, map_location="cpu")
                    set_peft_model_state_dict(model, adapters_weights)
            except Exception as e:
                print("Warning: failed to load adapter weights from resume path:", e)
                resume_checkpoint = None
        else:
            print("resume_from_checkpoint passed but no valid checkpoint found; continuing from base model.")
            resume_checkpoint = None

    # transparency for trainable parameters
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # prepare train/val datasets
    if val_set_size > 0 and "train" in data:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=seed or 42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt) if "train" in data else data.map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # Save PEFT adapter on save (callback)
    class SavePeftModelCallback(TrainerCallback):
        def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            # kwargs["model"] should be the peft model wrapper
            ckpt_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            try:
                kwargs["model"].save_pretrained(ckpt_folder)
                # Optionally remove large pytorch_model.bin if present (we prefer adapter files)
                pytorch_model_path = os.path.join(ckpt_folder, "pytorch_model.bin")
                if os.path.exists(pytorch_model_path):
                    try:
                        os.remove(pytorch_model_path)
                    except Exception as e:
                        print("Warning: unable to remove pytorch_model.bin:", e)
            except Exception as e:
                print("Warning: failed to save PEFT model at save callback:", e)
            return control

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
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[SavePeftModelCallback],
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        try:
            model = torch.compile(model)
        except Exception as e:
            print("torch.compile failed or is not recommended for this model:", e)

    # Save metadata/config for reproducibility
    os.makedirs(output_dir, exist_ok=True)
    meta = {
        "base_model": base_model,
        "data_path": data_path,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "cutoff_len": cutoff_len,
        "val_set_size": val_set_size,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
        "train_on_inputs": train_on_inputs,
        "group_by_length": group_by_length,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "resume_from_checkpoint": resume_checkpoint,
        "seed": seed,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "platform": platform.platform(),
    }
    with open(os.path.join(output_dir, "train_metadata.json"), "w", encoding="utf-8") as fw:
        json.dump(meta, fw, ensure_ascii=False, indent=2)

    # call trainer.train with resume checkpoint if valid path else None
    resume_arg = resume_checkpoint if resume_checkpoint else None
    trainer.train(resume_from_checkpoint=resume_arg)

    # final save
    model.save_pretrained(output_dir)
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print("Warning: failed to save tokenizer:", e)

    print("\nTraining finished. Saved model and tokenizer to", output_dir)


if __name__ == "__main__":
    fire.Fire(train)