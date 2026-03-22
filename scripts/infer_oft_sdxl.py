#!/usr/bin/env python3
"""Run SDXL inference with a locally saved OFT adapter bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from peft import OFTConfig, set_peft_model_state_dict
from transformers import AutoTokenizer, PretrainedConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--image", type=Path, default=None, help="Optional image path for img2img.")
    parser.add_argument("--strength", type=float, default=0.55)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--adapter_name", type=str, default="default")
    return parser.parse_args()


def import_text_encoder_cls(model_path: str, revision: str | None, subfolder: str) -> type:
    config = PretrainedConfig.from_pretrained(model_path, revision=revision, subfolder=subfolder)
    architecture = config.architectures[0]
    if architecture == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    if architecture == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    raise ValueError(f"Unsupported text encoder architecture: {architecture}")


def load_adapter_component(model: torch.nn.Module, adapter_component_dir: Path, adapter_name: str) -> None:
    if not adapter_component_dir.exists():
        return
    config = OFTConfig.from_pretrained(adapter_component_dir)
    model.add_adapter(config, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    state_dict = load_file(str(adapter_component_dir / "adapter_model.safetensors"))
    incompatible = set_peft_model_state_dict(model, state_dict, adapter_name=adapter_name)
    if incompatible is not None:
        unexpected_keys = getattr(incompatible, "unexpected_keys", None)
        if unexpected_keys:
            print(f"Warning: unexpected keys while loading {adapter_component_dir}: {unexpected_keys}")


def prepare_image(image_path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        use_fast=False,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )

    text_encoder_one_cls = import_text_encoder_cls(args.pretrained_model_name_or_path, args.revision, "text_encoder")
    text_encoder_two_cls = import_text_encoder_cls(args.pretrained_model_name_or_path, args.revision, "text_encoder_2")

    text_encoder_one = text_encoder_one_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    text_encoder_two = text_encoder_two_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )

    load_adapter_component(unet, args.adapter_dir / "unet", adapter_name=args.adapter_name)
    load_adapter_component(text_encoder_one, args.adapter_dir / "text_encoder", adapter_name=args.adapter_name)
    load_adapter_component(text_encoder_two, args.adapter_dir / "text_encoder_2", adapter_name=args.adapter_name)

    pipeline_cls = StableDiffusionXLImg2ImgPipeline if args.image else StableDiffusionXLPipeline
    pipeline = pipeline_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        torch_dtype=torch_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    generator = None
    if args.seed is not None and device.type == "cuda":
        generator = torch.Generator(device=device).manual_seed(args.seed)

    pipeline_args = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }
    if args.image:
        pipeline_args["image"] = prepare_image(args.image, args.width, args.height)
        pipeline_args["strength"] = args.strength
    else:
        pipeline_args["height"] = args.height
        pipeline_args["width"] = args.width

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.no_grad():
            image = pipeline(**pipeline_args).images[0]

    image.save(args.output)
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
