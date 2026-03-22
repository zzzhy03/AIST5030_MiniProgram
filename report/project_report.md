# AIST5030 Mini-Project Report: Parameter-efficient Finetuning for Pretrained Foundation Models

**Task:** LEGO-style car image generation with SDXL  
**Model:** Stable Diffusion XL base 1.0  
**Method:** Orthogonal Finetuning (OFT)

## Abstract

This project studies parameter-efficient adaptation of Stable Diffusion XL for LEGO-style car image generation. The goal is to make the model produce more consistent toy-brick vehicle images under a clean studio-photography setup while keeping the finetuning footprint small. Orthogonal Finetuning is used as the adaptation mechanism because it modifies the pretrained backbone in a structured and parameter-efficient way. The training set is composed of curated LEGO car images collected for this project, and the resulting system supports both prompt-based generation and image-conditioned stylization through `img2img`. The repository contains the training script, inference script, training configuration, and report draft for the experiment.

## 1. Task and Objective

The downstream task in this project is LEGO-style car generation. More specifically, the model is intended to generate images that preserve the overall semantics of a sports car or supercar while rendering the object as a LEGO build composed of toy bricks and photographed against a plain background. This task is a reasonable fit for a parameter-efficient finetuning project because the target domain is visually narrow and style-heavy, while the base model already has strong general image-generation capability.

The practical objective is twofold. First, the adapted model is expected to improve prompt-based generation of LEGO cars in `txt2img` mode. Second, it also supports `img2img` inference so that a real-car photo can be converted into a LEGO-like rendering. The evaluation therefore focuses on visual consistency, faithfulness to the LEGO style, and stability across repeated prompts.

## 2. Dataset and Experimental Setup

The training set currently contains 36 curated LEGO car images. These images are visually consistent: the object is a LEGO car, the viewpoint is close to product photography, and the background is relatively clean. This makes the dataset suitable for a DreamBooth-style local image finetuning workflow in which the goal is to adapt SDXL toward a narrow but coherent visual domain.

During training, the images are resized to 1024 resolution and normalized to the standard `[-1, 1]` diffusion input range. The intended execution environment is the target GPU server where SDXL is already available under a local cache path.

## 3. Method

The implementation uses Stable Diffusion XL base 1.0 as the pretrained foundation model and applies OFT adapters to the UNet. The training objective is the standard diffusion denoising objective used in SDXL finetuning. The repository script `scripts/train_oft_sdxl.py` loads the model components locally, inserts OFT adapters into the selected target modules, and writes adapter checkpoints without modifying the full pretrained weights.

The default training configuration uses resolution 1024, batch size 1, gradient accumulation 4, maximum 800 optimization steps, learning rate `6e-5`, mixed precision `fp16`, gradient checkpointing, and optional memory optimizations through `bitsandbytes` and `xformers`. The prompt design introduces the token `skslego` as a compact trigger for the learned domain and keeps the visual description anchored to studio photography and toy-brick structure.

## 4. Evaluation Protocol

The project evaluation contains both a training-loss figure and a before-versus-after comparison. During finetuning, `scripts/train_oft_sdxl.py` exports a step-wise CSV log under the output directory, and `scripts/plot_loss.py` converts that log into a figure suitable for inclusion in the report. In addition, `scripts/infer_oft_sdxl.py` generates matched prompt samples from the base model and the finetuned adapter.

The comparison protocol is defined as follows. First, a fixed set of LEGO-car prompts is used to compare the base SDXL model and the OFT-adapted model in `txt2img` mode. Second, one or more real-car photographs are tested in `img2img` mode to determine whether the finetuned model preserves major car structure while moving the rendering toward the LEGO domain. At the time of repository packaging, the official server-side artifacts had not yet been committed, so this document records the evaluation design and the code path used to generate the final empirical evidence.

## 5. Discussion

This project is technically well matched to parameter-efficient finetuning. The base model already has strong prior knowledge about cars, product photography, and composition, while the finetuning task mainly requires adaptation to a narrow visual domain defined by brick texture, toy geometry, and a clean studio setup. OFT is therefore a reasonable choice because it allows the project to focus the adaptation on a compact set of trainable parameters instead of updating the entire SDXL backbone.

The main limitation is dataset size. Thirty-six training images are enough to study the feasibility of the adaptation, but they are still too few to support broad coverage of car categories, camera viewpoints, and lighting variations. The final empirical analysis therefore needs to pay attention not only to visual quality but also to overfitting behavior, prompt sensitivity, and the extent to which the learned LEGO appearance generalizes beyond the specific training examples.

## 6. Conclusion

This repository presents a complete SDXL OFT project setup for LEGO-style car generation. The task definition, dataset organization, training code, inference code, and report structure are aligned around a single experiment design intended for execution on the target server. The remaining empirical material consists of the adapter outputs, the loss curve, and the qualitative comparisons generated by that run.

## References

- Hugging Face PEFT documentation for Orthogonal Finetuning.
- Hugging Face Diffusers documentation for DreamBooth and SDXL training.
- Stable Diffusion XL base 1.0 model release.
