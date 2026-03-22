# Mini-Project: Parameter-efficient Finetuning for Pretrained Foundation Models

This project investigates parameter-efficient finetuning of Stable Diffusion XL with Orthogonal Finetuning (OFT) for LEGO-style car image generation. The goal is to adapt a pretrained text-to-image model to a narrow visual domain while keeping the number of trainable parameters small.

After finetuning, the adapted model is intended to support two types of generation:

- `txt2img` generation of LEGO-style car images from prompts
- `img2img` stylization of real-car photos into LEGO-like renderings

## Method

The implementation uses Stable Diffusion XL base 1.0 as the backbone model and inserts OFT adapters into the trainable modules of the diffusion model. Training is performed on a local image set with a DreamBooth-style workflow. The repository includes:

- an SDXL OFT training script
- an inference script for both `txt2img` and `img2img`
- a plotting utility for loss curves
- a compact training configuration file

## Dataset

The experiment uses a local training set of 36 PNG images. In the provided repository configuration, the training directory is `train_data/clean`, while the training script itself accepts any image directory passed through `--instance_data_dir`.

## Repository Layout

```text
AIST5030_MiniProgram/
├── Mini-Project_ Parameter-efficient Finetuning for Pretrained Models.pdf
├── README.md
├── .gitignore
├── requirements.txt
├── configs/
│   └── sdxl_oft_lego.yaml
├── logs/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── report/
│   ├── figures/
│   │   └── .gitkeep
│   ├── tables/
│   │   └── .gitkeep
│   └── project_report.md
├── scripts/
│   ├── infer_oft_sdxl.py
│   ├── plot_loss.py
│   ├── run_oft_sdxl_lego.sh
│   └── train_oft_sdxl.py
└── train_data/
    └── ...
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the memory-optimized SDXL setup used in this project, `bitsandbytes` and `xformers` may also be installed:

```bash
pip install bitsandbytes xformers
```

## Training

Example server environment:

```bash
export AUTODL_TMP=/root/autodl-tmp
export HF_HOME=$AUTODL_TMP/hf
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export DIFFUSERS_CACHE=$HF_HOME/diffusers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$AUTODL_TMP/torch
export PIP_CACHE_DIR=$AUTODL_TMP/pip-cache
mkdir -p "$AUTODL_TMP/outputs" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

export HF_TOKEN="xxx"
export SDXL_DIR="/root/autodl-tmp/models/sdxl-base-1.0"
```

Default launch command:

```bash
bash scripts/run_oft_sdxl_lego.sh
```

Equivalent direct invocation:

```bash
python3 scripts/train_oft_sdxl.py \
  --pretrained_model_name_or_path "$SDXL_DIR" \
  --instance_data_dir train_data/clean \
  --output_dir "$AUTODL_TMP/outputs/oft_sdxl_lego" \
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
  --use_8bit_adam
```

## Inference

Prompt-based generation:

```bash
python3 scripts/infer_oft_sdxl.py \
  --pretrained_model_name_or_path "$SDXL_DIR" \
  --adapter_dir "$AUTODL_TMP/outputs/oft_sdxl_lego/final_adapter" \
  --prompt "studio photo of a skslego red racing car made of toy bricks, white seamless background" \
  --output outputs/lego_prompt.png
```

Image-guided generation:

```bash
python3 scripts/infer_oft_sdxl.py \
  --pretrained_model_name_or_path "$SDXL_DIR" \
  --adapter_dir "$AUTODL_TMP/outputs/oft_sdxl_lego/final_adapter" \
  --image path/to/real_car.jpg \
  --prompt "a skslego version of this car, toy brick model, studio product photo, white background" \
  --strength 0.55 \
  --output outputs/lego_img2img.png
```

## Outputs

Training and inference produce the following artifacts:

- adapter checkpoints
- final adapter weights
- training loss logs
- validation images
- generated images

## Loss Curve

```bash
python scripts/plot_loss.py \
  --input logs/train_loss.csv \
  --output report/figures/training_loss.png \
  --title "Training Loss"
```

## Report

The project report draft is stored at `report/project_report.md`.
