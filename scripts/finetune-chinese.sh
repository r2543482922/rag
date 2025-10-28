exp_tag="e3"
python finetune.py \
  --base_model ziqingyang/chinese-llama-2-7b \
  --data_path ./data/llama_data.json \
  --output_dir ./lora-chinese-llama2-med-$exp_tag \
  --prompt_template_name med_template \
  --micro_batch_size 64 \
  --batch_size 128 \
  --wandb_run_name $exp_tag