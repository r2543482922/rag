#!/bin/bash

# 设置实验标签
exp_tag="llama3_chinese_v1"

# 设置WandB参数（可选）
export WANDB_PROJECT="llama3_chinese"
export WANDB_WATCH="all"
export WANDB_LOG_MODEL="true"

# 运行训练脚本
python finetuneLlama3.py \
    --base_model "shenzhi-wang/Llama3-8B-Chinese-Chat" \
    --data_path "./data/chinese_instruction_data.json" \
    --output_dir "./lora-llama3-chinese-${exp_tag}" \
    --prompt_template_name "chinese_template" \
    --micro_batch_size 4 \
    --batch_size 128 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --cutoff_len 512 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --val_set_size 500 \
    --wandb_run_name "${exp_tag}" \
    --group_by_length \
    --resume_from_checkpoint None 2>&1 | tee training.log

echo "Training completed! Output saved to: ./lora-llama3-chinese-${exp_tag}"