#!/bin/bash

python3 main.py \
    --project-name text2sql \
    --run-name v1_r16_a32 \
    --dataset-path 100suping/ko-bird-sql-schema \
    --model-type qwen-2.5 \
    --model-name unsloth/Qwen2.5-Coder-32B-Instruct \
    --quant-bit 8 \
    --r 16 \
    --lora-alpha 32 \
    --lora-dropout 0 \
    --max-seq-length 4096 \
    --output-dir outputs \
    --save-steps 50 \
    --logging-steps 4 \
    --epochs 3 \
    --batch-size 32 \
    --warmup-steps 20 \
    --lr 2e-4 \
    --gradient-accumulation-steps 4 \
    --verbose false \
    --report-to wandb \
    --seed 42 \
    --test-run false