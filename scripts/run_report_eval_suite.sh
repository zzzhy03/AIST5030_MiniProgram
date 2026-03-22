#!/usr/bin/env bash
set -euo pipefail

: "${SDXL_DIR:?Please export SDXL_DIR to your local SDXL base model path.}"

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
ADAPTER_DIR="${ADAPTER_DIR:-$AUTODL_TMP/outputs/oft_sdxl_lego/final_adapter}"
RUN_DIR="${RUN_DIR:-$(dirname "$ADAPTER_DIR")}"
REPORT_TAG="${REPORT_TAG:-$(basename "$ADAPTER_DIR")}"
EVAL_DIR="${EVAL_DIR:-outputs/report_eval/$REPORT_TAG}"
IMG2IMG_INPUT_DIR="${IMG2IMG_INPUT_DIR:-testset}"
IMG2IMG_MAX_INPUTS="${IMG2IMG_MAX_INPUTS:-0}"

TXT2IMG_STEPS="${TXT2IMG_STEPS:-30}"
IMG2IMG_STEPS="${IMG2IMG_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
IMG2IMG_STRENGTH="${IMG2IMG_STRENGTH:-0.55}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"

mkdir -p "$EVAL_DIR/txt2img" "$EVAL_DIR/img2img"

BASE_ADAPTER_DIR="$EVAL_DIR/_base_model_without_adapter"
MANIFEST_PATH="$EVAL_DIR/report_eval_manifest.md"
LOSS_CSV_PATH="$RUN_DIR/logs/train_loss.csv"
LOSS_PLOT_PATH="$EVAL_DIR/training_loss.png"
IMG2IMG_PROMPT="a skslego version of this car, toy brick model, studio product photo, white background"

PROMPT_IDS=(
  "prompt_01_red_sports"
  "prompt_02_white_supercar"
  "prompt_03_black_racing"
)
PROMPTS=(
  "studio photo of a skslego red sports car made of toy bricks, white seamless background"
  "studio photo of a skslego white supercar made of toy bricks, white seamless background"
  "studio photo of a skslego black racing car made of toy bricks, three-quarter front view, white studio background"
)
PROMPT_SEEDS=(42 52 62)

{
  printf '# Report Evaluation Manifest\n\n'
  printf -- '- `adapter_dir`: `%s`\n' "$ADAPTER_DIR"
  printf -- '- `run_dir`: `%s`\n' "$RUN_DIR"
  printf -- '- `eval_dir`: `%s`\n' "$EVAL_DIR"
  printf -- '- `txt2img_steps`: `%s`\n' "$TXT2IMG_STEPS"
  printf -- '- `img2img_steps`: `%s`\n' "$IMG2IMG_STEPS"
  printf -- '- `guidance_scale`: `%s`\n' "$GUIDANCE_SCALE"
  printf -- '- `img2img_strength`: `%s`\n' "$IMG2IMG_STRENGTH"
  printf -- '- `resolution`: `%sx%s`\n' "$WIDTH" "$HEIGHT"
  printf '\n'
} > "$MANIFEST_PATH"

if [[ -f "$LOSS_CSV_PATH" ]]; then
  python3 scripts/plot_loss.py \
    --input "$LOSS_CSV_PATH" \
    --output "$LOSS_PLOT_PATH" \
    --title "Training Loss ($REPORT_TAG)"
  {
    printf '## Loss Curve\n\n'
    printf -- '- `csv`: `%s`\n' "$LOSS_CSV_PATH"
    printf -- '- `plot`: `%s`\n\n' "$LOSS_PLOT_PATH"
  } >> "$MANIFEST_PATH"
fi

{
  printf '## txt2img Prompts\n\n'
} >> "$MANIFEST_PATH"

for index in "${!PROMPT_IDS[@]}"; do
  prompt_id="${PROMPT_IDS[$index]}"
  prompt_text="${PROMPTS[$index]}"
  seed="${PROMPT_SEEDS[$index]}"
  base_output="$EVAL_DIR/txt2img/${prompt_id}_base.png"
  oft_output="$EVAL_DIR/txt2img/${prompt_id}_oft.png"

  python3 scripts/infer_oft_sdxl.py \
    --pretrained_model_name_or_path "$SDXL_DIR" \
    --adapter_dir "$BASE_ADAPTER_DIR" \
    --prompt "$prompt_text" \
    --seed "$seed" \
    --num_inference_steps "$TXT2IMG_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --output "$base_output"

  python3 scripts/infer_oft_sdxl.py \
    --pretrained_model_name_or_path "$SDXL_DIR" \
    --adapter_dir "$ADAPTER_DIR" \
    --prompt "$prompt_text" \
    --seed "$seed" \
    --num_inference_steps "$TXT2IMG_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --output "$oft_output"

  {
    printf '### %s\n\n' "$prompt_id"
    printf -- '- `prompt`: `%s`\n' "$prompt_text"
    printf -- '- `seed`: `%s`\n' "$seed"
    printf -- '- `base`: `%s`\n' "$base_output"
    printf -- '- `oft`: `%s`\n\n' "$oft_output"
  } >> "$MANIFEST_PATH"
done

if [[ "$IMG2IMG_MAX_INPUTS" =~ ^[1-9][0-9]*$ ]]; then
  mapfile -t IMG2IMG_INPUTS < <(
    find "$IMG2IMG_INPUT_DIR" -maxdepth 1 -type f \
      \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) \
      | sort | head -n "$IMG2IMG_MAX_INPUTS"
  )
else
  mapfile -t IMG2IMG_INPUTS < <(
    find "$IMG2IMG_INPUT_DIR" -maxdepth 1 -type f \
      \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) \
      | sort
  )
fi

{
  printf '## img2img Inputs\n\n'
  printf -- '- `prompt`: `%s`\n\n' "$IMG2IMG_PROMPT"
} >> "$MANIFEST_PATH"

if [[ "${#IMG2IMG_INPUTS[@]}" -eq 0 ]]; then
  {
    printf 'No img2img inputs were found in `%s`.\n' "$IMG2IMG_INPUT_DIR"
  } >> "$MANIFEST_PATH"
else
  for input_path in "${IMG2IMG_INPUTS[@]}"; do
    input_name="$(basename "$input_path")"
    input_stem="${input_name%.*}"
    input_ext=".${input_name##*.}"
    copied_input="$EVAL_DIR/img2img/${input_stem}_input${input_ext}"
    base_output="$EVAL_DIR/img2img/${input_stem}_base.png"
    oft_output="$EVAL_DIR/img2img/${input_stem}_oft.png"

    cp "$input_path" "$copied_input"

    python3 scripts/infer_oft_sdxl.py \
      --pretrained_model_name_or_path "$SDXL_DIR" \
      --adapter_dir "$BASE_ADAPTER_DIR" \
      --image "$input_path" \
      --prompt "$IMG2IMG_PROMPT" \
      --strength "$IMG2IMG_STRENGTH" \
      --seed 42 \
      --num_inference_steps "$IMG2IMG_STEPS" \
      --guidance_scale "$GUIDANCE_SCALE" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --output "$base_output"

    python3 scripts/infer_oft_sdxl.py \
      --pretrained_model_name_or_path "$SDXL_DIR" \
      --adapter_dir "$ADAPTER_DIR" \
      --image "$input_path" \
      --prompt "$IMG2IMG_PROMPT" \
      --strength "$IMG2IMG_STRENGTH" \
      --seed 42 \
      --num_inference_steps "$IMG2IMG_STEPS" \
      --guidance_scale "$GUIDANCE_SCALE" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --output "$oft_output"

    {
      printf '### %s\n\n' "$input_stem"
      printf -- '- `input`: `%s`\n' "$copied_input"
      printf -- '- `base`: `%s`\n' "$base_output"
      printf -- '- `oft`: `%s`\n\n' "$oft_output"
    } >> "$MANIFEST_PATH"
  done
fi

printf 'Saved report evaluation outputs to: %s\n' "$EVAL_DIR"
printf 'Saved evaluation manifest to: %s\n' "$MANIFEST_PATH"
