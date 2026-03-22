#!/usr/bin/env bash
set -euo pipefail

: "${SDXL_DIR:?Please export SDXL_DIR to your local SDXL base model path.}"

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
OUTPUT_DIR="${OUTPUT_DIR:-$AUTODL_TMP/outputs/oft_sdxl_lego}"

mkdir -p "$OUTPUT_DIR"

python3 scripts/train_oft_sdxl.py \
  --pretrained_model_name_or_path "$SDXL_DIR" \
  --instance_data_dir train_data/clean \
  --output_dir "$OUTPUT_DIR" \
  --instance_prompt "studio photo of a skslego sports car made of toy bricks, white seamless background" \
  --validation_prompt "studio photo of a skslego supercar made of toy bricks, white seamless background" \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 800 \
  --learning_rate 6e-5 \
  --resolution 1024 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --use_8bit_adam \
  --save_initial_validation
