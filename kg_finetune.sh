#!/bin/bash

# Configuration
export WANDB_PROJECT="llama_med"
exp_tag="medical_finetune_v1"

# Training parameters
python finetune.py \
    --base_model 'decapoda-research/llama-7b' \
    --data_path './data/llama_data.json' \
    --output_dir './lora-llama-med-2' \
    --prompt_template_name 'med_template' \
    --micro_batch_size 8 \
    --batch_size 64 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --val_set_size 200 \
    --wandb_run_name $exp_tag \
    --lora_target_modules "q_proj" "v_proj"kg_finetune.sh