#!/bin/bash

cd /data/sjh/LLMwork/Qwen

python finetune.py \
  --model_name_or_path /data/sjh/LLMwork/Qwen/finetune/models/Qwen-7B-Chat \
  --data_path /data/sjh/LLMwork/Qwen/finetune/data/my_sft_data.json \
  --output_dir /data/sjh/LLMwork/Qwen/finetune/qwen-lora-output \
  --model_max_length 1024 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs 128 \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules c_attn \
  --use_lora True \
  --q_lora False \
  --fp16 False \
  --bf16 False \
  --logging_steps 10 \
  --save_steps 100 \
  --report_to none \
  --do_train