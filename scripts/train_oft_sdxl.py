#!/usr/bin/env python3
"""Train an SDXL OFT adapter on a local image folder."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from peft import OFTConfig
from peft.utils import get_peft_model_state_dict
from transformers import AutoTokenizer, PretrainedConfig


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_UNET_TARGETS = [
    "proj_in",
    "proj_out",
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
]
DEFAULT_TEXT_ENCODER_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


@dataclass
class Example:
    image_path: Path
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--instance_prompt", type=str, required=True)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--validation_image", type=Path, default=None)
    parser.add_argument("--validation_strength", type=float, default=0.55)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_steps", type=int, default=200)
    parser.add_argument("--save_initial_validation", action="store_true")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--caption_extension", type=str, default=".txt")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--text_encoder_lr", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--prediction_type", choices=("epsilon", "v_prediction"), default=None)
    parser.add_argument("--noise_offset", type=float, default=0.0)
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--adapter_name", type=str, default="default")
    parser.add_argument("--oft_rank", type=int, default=None)
    parser.add_argument("--oft_block_size", type=int, default=None)
    parser.add_argument("--oft_dropout", type=float, default=0.0)
    parser.add_argument("--coft", action="store_true")
    parser.add_argument("--coft_eps", type=float, default=6e-5)
    parser.add_argument("--block_share", action="store_true")
    parser.add_argument("--num_cayley_neumann_terms", type=int, default=5)
    parser.add_argument("--unet_target_modules", type=str, default=",".join(DEFAULT_UNET_TARGETS))
    parser.add_argument(
        "--text_encoder_target_modules",
        type=str,
        default=",".join(DEFAULT_TEXT_ENCODER_TARGETS),
    )
    parser.add_argument("--validation_guidance_scale", type=float, default=7.5)
    parser.add_argument("--validation_num_steps", type=int, default=30)
    args = parser.parse_args()
    if args.oft_rank is None and args.oft_block_size is None:
        args.oft_rank = 8
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def maybe_read_caption(image_path: Path, caption_extension: str, fallback_prompt: str) -> str:
    caption_path = image_path.with_suffix(caption_extension)
    if caption_path.exists():
        text = caption_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return fallback_prompt


class LocalImagePromptDataset(Dataset):
    def __init__(
        self,
        instance_data_dir: Path,
        instance_prompt: str,
        resolution: int,
        repeats: int,
        center_crop: bool,
        random_flip: bool,
        caption_extension: str,
    ) -> None:
        if not instance_data_dir.exists():
            raise FileNotFoundError(f"Instance data directory not found: {instance_data_dir}")

        image_paths = sorted(
            path for path in instance_data_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No images found in {instance_data_dir}")

        self.examples = [
            Example(
                image_path=path,
                prompt=maybe_read_caption(path, caption_extension=caption_extension, fallback_prompt=instance_prompt),
            )
            for path in image_paths
        ]
        self.num_instance_images = len(self.examples)
        self._length = self.num_instance_images * max(repeats, 1)
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.has_custom_prompts = any(example.prompt != instance_prompt for example in self.examples)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index % self.num_instance_images]
        image = Image.open(example.image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = (image.height, image.width)
        image = self.resize(image)

        if self.center_crop:
            y1 = max(0, int(round((image.height - self.resolution) / 2.0)))
            x1 = max(0, int(round((image.width - self.resolution) / 2.0)))
        else:
            y1 = 0 if image.height == self.resolution else random.randint(0, image.height - self.resolution)
            x1 = 0 if image.width == self.resolution else random.randint(0, image.width - self.resolution)

        image = crop(image, top=y1, left=x1, height=self.resolution, width=self.resolution)

        if self.random_flip and random.random() < 0.5:
            image = ImageOps.mirror(image)

        pixel_values = self.to_tensor(image)
        return {
            "pixel_values": pixel_values,
            "prompt": example.prompt,
            "original_size": original_size,
            "crop_top_left": (y1, x1),
            "image_path": str(example.image_path),
        }


def collate_examples(examples: list[dict[str, object]]) -> dict[str, object]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]
    image_paths = [example["image_path"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
        "image_paths": image_paths,
    }


def tokenize_prompt(tokenizer: AutoTokenizer, prompt: str | list[str]) -> torch.Tensor:
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids


def encode_prompt(
    text_encoders: list[torch.nn.Module],
    tokenizers: list[AutoTokenizer] | None,
    prompt: str | list[str] | None,
    text_input_ids_list: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds_list = []
    pooled_prompt_embeds = None

    for index, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            text_input_ids = tokenize_prompt(tokenizers[index], prompt)
        else:
            if text_input_ids_list is None:
                raise ValueError("text_input_ids_list must be provided when tokenizers is None")
            text_input_ids = text_input_ids_list[index]

        prompt_outputs = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
            return_dict=False,
        )
        pooled_prompt_embeds = prompt_outputs[0]
        prompt_hidden_states = prompt_outputs[-1][-2]
        bs_embed, seq_len, _ = prompt_hidden_states.shape
        prompt_hidden_states = prompt_hidden_states.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_hidden_states)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(prompt_embeds.shape[0], -1)
    return prompt_embeds, pooled_prompt_embeds


def build_add_time_ids(
    original_sizes: list[tuple[int, int]],
    crop_top_lefts: list[tuple[int, int]],
    resolution: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    target_size = (resolution, resolution)
    add_time_ids = [list(original + crop_top_left + target_size) for original, crop_top_left in zip(original_sizes, crop_top_lefts)]
    return torch.tensor(add_time_ids, device=device, dtype=dtype)


def cast_trainable_params(models: Iterable[torch.nn.Module]) -> None:
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()


def count_trainable_params(models: Iterable[torch.nn.Module]) -> int:
    total = 0
    for model in models:
        total += sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total


def parse_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def create_oft_config(args: argparse.Namespace, target_modules: list[str]) -> OFTConfig:
    if args.oft_rank is not None and args.oft_block_size is not None:
        raise ValueError("Specify either --oft_rank or --oft_block_size, not both.")

    config_kwargs = {
        "module_dropout": args.oft_dropout,
        "target_modules": target_modules,
        "init_weights": True,
        "coft": args.coft,
        "eps": args.coft_eps,
        "block_share": args.block_share,
        "use_cayley_neumann": True,
        "num_cayley_neumann_terms": args.num_cayley_neumann_terms,
        "bias": "none",
    }
    if args.oft_rank is not None:
        config_kwargs["r"] = args.oft_rank
    if args.oft_block_size is not None:
        config_kwargs["oft_block_size"] = args.oft_block_size

    return OFTConfig(**config_kwargs)


def save_adapter_component(model: torch.nn.Module, component_dir: Path, adapter_name: str) -> None:
    component_dir.mkdir(parents=True, exist_ok=True)
    state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)
    save_file(state_dict, str(component_dir / "adapter_model.safetensors"))


def save_training_state(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
    trainable_params: int,
    train_text_encoder: bool,
    validation_prompt: str | None,
) -> None:
    metadata = {
        "base_model": args.pretrained_model_name_or_path,
        "instance_data_dir": str(args.instance_data_dir),
        "instance_prompt": args.instance_prompt,
        "validation_prompt": validation_prompt,
        "resolution": args.resolution,
        "global_step": global_step,
        "epoch": epoch,
        "adapter_name": args.adapter_name,
        "train_text_encoder": train_text_encoder,
        "trainable_params": trainable_params,
        "oft_rank": args.oft_rank,
        "oft_block_size": args.oft_block_size,
        "unet_target_modules": parse_target_modules(args.unet_target_modules),
        "text_encoder_target_modules": parse_target_modules(args.text_encoder_target_modules),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def save_adapter_bundle(
    save_dir: Path,
    *,
    args: argparse.Namespace,
    unet: torch.nn.Module,
    text_encoder_one: torch.nn.Module,
    text_encoder_two: torch.nn.Module,
    unet_config: OFTConfig,
    text_encoder_config: OFTConfig | None,
    global_step: int,
    epoch: int,
    trainable_params: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    unet_dir = save_dir / "unet"
    save_adapter_component(unet, unet_dir, adapter_name=args.adapter_name)
    unet_config.save_pretrained(unet_dir)

    if args.train_text_encoder and text_encoder_config is not None:
        te1_dir = save_dir / "text_encoder"
        te2_dir = save_dir / "text_encoder_2"
        save_adapter_component(text_encoder_one, te1_dir, adapter_name=args.adapter_name)
        save_adapter_component(text_encoder_two, te2_dir, adapter_name=args.adapter_name)
        text_encoder_config.save_pretrained(te1_dir)
        text_encoder_config.save_pretrained(te2_dir)

    save_training_state(
        save_dir,
        args=args,
        global_step=global_step,
        epoch=epoch,
        trainable_params=trainable_params,
        train_text_encoder=args.train_text_encoder,
        validation_prompt=args.validation_prompt,
    )


def load_validation_image(image_path: Path, resolution: int) -> Image.Image:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return image


def maybe_autocast(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_validation(
    *,
    args: argparse.Namespace,
    device: torch.device,
    step: int,
    tokenizer_one: AutoTokenizer,
    tokenizer_two: AutoTokenizer,
    text_encoder_one: torch.nn.Module,
    text_encoder_two: torch.nn.Module,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    generator: torch.Generator | None,
) -> None:
    if not args.validation_prompt:
        return

    validation_dir = args.output_dir / "validation" / f"step-{step:06d}"
    validation_dir.mkdir(parents=True, exist_ok=True)

    was_training = {
        "unet": unet.training,
        "te1": text_encoder_one.training,
        "te2": text_encoder_two.training,
    }
    unet.eval()
    text_encoder_one.eval()
    text_encoder_two.eval()

    pipeline_cls = StableDiffusionXLImg2ImgPipeline if args.validation_image else StableDiffusionXLPipeline
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
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    pipeline_args = {
        "prompt": args.validation_prompt,
        "num_inference_steps": args.validation_num_steps,
        "guidance_scale": args.validation_guidance_scale,
        "generator": generator,
    }
    if args.validation_image:
        pipeline_args["image"] = load_validation_image(args.validation_image, args.resolution)
        pipeline_args["strength"] = args.validation_strength

    with torch.no_grad():
        with maybe_autocast(device, args.mixed_precision):
            images = pipeline(
                **pipeline_args,
                num_images_per_prompt=args.num_validation_images,
            ).images

    for index, image in enumerate(images):
        image.save(validation_dir / f"sample_{index:02d}.png")

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if was_training["unet"]:
        unet.train()
    if was_training["te1"]:
        text_encoder_one.train()
    if was_training["te2"]:
        text_encoder_two.train()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.mixed_precision != "no":
        print("CUDA is not available. Falling back to full precision on CPU.")
        args.mixed_precision = "no"

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )

    text_encoder_one_cls = import_text_encoder_cls(args.pretrained_model_name_or_path, args.revision, "text_encoder")
    text_encoder_two_cls = import_text_encoder_cls(args.pretrained_model_name_or_path, args.revision, "text_encoder_2")

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

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

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    unet_config = create_oft_config(args, parse_target_modules(args.unet_target_modules))
    unet.add_adapter(unet_config, adapter_name=args.adapter_name)
    unet.set_adapter(args.adapter_name)

    text_encoder_config = None
    if args.train_text_encoder:
        text_encoder_config = create_oft_config(args, parse_target_modules(args.text_encoder_target_modules))
        text_encoder_one.add_adapter(text_encoder_config, adapter_name=args.adapter_name)
        text_encoder_two.add_adapter(text_encoder_config, adapter_name=args.adapter_name)
        text_encoder_one.set_adapter(args.adapter_name)
        text_encoder_two.set_adapter(args.adapter_name)

    trainable_models = [unet]
    if args.train_text_encoder:
        trainable_models.extend([text_encoder_one, text_encoder_two])
    cast_trainable_params(trainable_models)

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xformers is not available but --enable_xformers_memory_efficient_attention was set")
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=torch.float32)
    if args.train_text_encoder:
        text_encoder_one.to(device, dtype=torch.float32)
        text_encoder_two.to(device, dtype=torch.float32)
    else:
        text_encoder_one.to(device, dtype=weight_dtype)
        text_encoder_two.to(device, dtype=weight_dtype)

    train_dataset = LocalImagePromptDataset(
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        resolution=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        caption_extension=args.caption_extension,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_examples,
    )

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.train_batch_size * args.gradient_accumulation_steps
        if args.train_text_encoder:
            args.text_encoder_lr = args.text_encoder_lr * args.train_batch_size * args.gradient_accumulation_steps

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise SystemExit("bitsandbytes is required for --use_8bit_adam") from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer_grouped_parameters = [
        {
            "params": [param for param in unet.parameters() if param.requires_grad],
            "lr": args.learning_rate,
        }
    ]
    if args.train_text_encoder:
        optimizer_grouped_parameters.append(
            {
                "params": [param for model in (text_encoder_one, text_encoder_two) for param in model.parameters() if param.requires_grad],
                "lr": args.text_encoder_lr,
            }
        )

    optimizer = optimizer_cls(
        optimizer_grouped_parameters,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps
    num_train_epochs = max(args.num_train_epochs, math.ceil(max_train_steps / update_steps_per_epoch))
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.mixed_precision == "fp16")

    unique_prompts = {example.prompt for example in train_dataset.examples}
    cached_prompt_embeds = None
    cached_pooled_prompt_embeds = None
    if not args.train_text_encoder and len(unique_prompts) == 1:
        only_prompt = next(iter(unique_prompts))
        with torch.no_grad():
            cached_prompt_embeds, cached_pooled_prompt_embeds = encode_prompt(
                [text_encoder_one, text_encoder_two],
                [tokenizer_one, tokenizer_two],
                only_prompt,
            )
        cached_prompt_embeds = cached_prompt_embeds.to(device=device, dtype=weight_dtype)
        cached_pooled_prompt_embeds = cached_pooled_prompt_embeds.to(device=device, dtype=weight_dtype)

    trainable_params = count_trainable_params(trainable_models)
    summary = {
        "device": str(device),
        "mixed_precision": args.mixed_precision,
        "trainable_params": trainable_params,
        "dataset_images": train_dataset.num_instance_images,
        "dataset_length_after_repeats": len(train_dataset),
        "num_train_epochs": num_train_epochs,
        "max_train_steps": max_train_steps,
    }
    print(json.dumps(summary, indent=2))
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    loss_csv_path = logs_dir / "train_loss.csv"
    csv_file = loss_csv_path.open("w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "epoch", "loss", "lr"])

    generator = None
    if args.seed is not None and device.type == "cuda":
        generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.save_initial_validation and args.validation_prompt:
        run_validation(
            args=args,
            device=device,
            step=0,
            tokenizer_one=tokenizer_one,
            tokenizer_two=tokenizer_two,
            text_encoder_one=text_encoder_one,
            text_encoder_two=text_encoder_two,
            vae=vae,
            unet=unet,
            generator=generator,
        )

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    progress_bar = tqdm(total=max_train_steps, desc="Training")

    last_epoch = 0
    for epoch in range(num_train_epochs):
        last_epoch = epoch
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()

        for step, batch in enumerate(train_dataloader):
            if global_step >= max_train_steps:
                break

            pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
            model_input = model_input.to(device=device, dtype=weight_dtype)

            noise = torch.randn_like(model_input)
            if args.noise_offset:
                noise = noise + args.noise_offset * torch.randn(
                    (model_input.shape[0], model_input.shape[1], 1, 1),
                    device=device,
                    dtype=model_input.dtype,
                )
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (model_input.shape[0],),
                device=device,
            ).long()
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            if args.train_text_encoder:
                tokens_one = tokenize_prompt(tokenizer_one, batch["prompts"]).to(device)
                tokens_two = tokenize_prompt(tokenizer_two, batch["prompts"]).to(device)
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    [text_encoder_one, text_encoder_two],
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=[tokens_one, tokens_two],
                )
                prompt_embeds = prompt_embeds.to(device=device, dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=weight_dtype)
            else:
                if cached_prompt_embeds is None:
                    with torch.no_grad():
                        prompt_embeds, pooled_prompt_embeds = encode_prompt(
                            [text_encoder_one, text_encoder_two],
                            [tokenizer_one, tokenizer_two],
                            batch["prompts"],
                        )
                    prompt_embeds = prompt_embeds.to(device=device, dtype=weight_dtype)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=weight_dtype)
                else:
                    batch_size = noisy_model_input.shape[0]
                    prompt_embeds = cached_prompt_embeds.repeat(batch_size, 1, 1)
                    pooled_prompt_embeds = cached_pooled_prompt_embeds.repeat(batch_size, 1)

            add_time_ids = build_add_time_ids(
                batch["original_sizes"],
                batch["crop_top_lefts"],
                resolution=args.resolution,
                device=device,
                dtype=weight_dtype,
            )

            target = (
                noise
                if noise_scheduler.config.prediction_type == "epsilon"
                else noise_scheduler.get_velocity(model_input, noise, timesteps)
            )

            added_cond_kwargs = {
                "time_ids": add_time_ids,
                "text_embeds": pooled_prompt_embeds,
            }

            with maybe_autocast(device, args.mixed_precision):
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            loss_to_backprop = loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            should_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader)
            if not should_step:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [param for group in optimizer.param_groups for param in group["params"]],
                args.max_grad_norm,
            )

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress_bar.update(1)
            current_lr = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
            csv_writer.writerow([global_step, epoch, f"{loss.item():.8f}", f"{current_lr:.10f}"])
            csv_file.flush()

            if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                run_validation(
                    args=args,
                    device=device,
                    step=global_step,
                    tokenizer_one=tokenizer_one,
                    tokenizer_two=tokenizer_two,
                    text_encoder_one=text_encoder_one,
                    text_encoder_two=text_encoder_two,
                    vae=vae,
                    unet=unet,
                    generator=generator,
                )

            if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                checkpoint_dir = args.output_dir / f"checkpoint-{global_step:06d}"
                save_adapter_bundle(
                    checkpoint_dir,
                    args=args,
                    unet=unet,
                    text_encoder_one=text_encoder_one,
                    text_encoder_two=text_encoder_two,
                    unet_config=unet_config,
                    text_encoder_config=text_encoder_config,
                    global_step=global_step,
                    epoch=epoch,
                    trainable_params=trainable_params,
                )

            if global_step >= max_train_steps:
                break

    progress_bar.close()
    csv_file.close()

    final_dir = args.output_dir / "final_adapter"
    save_adapter_bundle(
        final_dir,
        args=args,
        unet=unet,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        unet_config=unet_config,
        text_encoder_config=text_encoder_config,
        global_step=global_step,
        epoch=last_epoch,
        trainable_params=trainable_params,
    )

    if args.validation_prompt:
        run_validation(
            args=args,
            device=device,
            step=global_step,
            tokenizer_one=tokenizer_one,
            tokenizer_two=tokenizer_two,
            text_encoder_one=text_encoder_one,
            text_encoder_two=text_encoder_two,
            vae=vae,
            unet=unet,
            generator=generator,
        )

    print(f"Training finished. Final adapter saved to: {final_dir}")
    print(f"Loss log saved to: {loss_csv_path}")


if __name__ == "__main__":
    main()
