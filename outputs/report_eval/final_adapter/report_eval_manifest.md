# Report Evaluation Manifest

- `adapter_dir`: `/root/autodl-tmp/outputs/oft_sdxl_lego/final_adapter`
- `run_dir`: `/root/autodl-tmp/outputs/oft_sdxl_lego`
- `eval_dir`: `outputs/report_eval/final_adapter`
- `txt2img_steps`: `30`
- `img2img_steps`: `30`
- `guidance_scale`: `7.5`
- `img2img_strength`: `0.55`
- `resolution`: `1024x1024`

## Loss Curve

- `csv`: `/root/autodl-tmp/outputs/oft_sdxl_lego/logs/train_loss.csv`
- `plot`: `outputs/report_eval/final_adapter/training_loss.png`

## txt2img Prompts

### prompt_01_red_sports

- `prompt`: `studio photo of a skslego red sports car made of toy bricks, white seamless background`
- `seed`: `42`
- `base`: `outputs/report_eval/final_adapter/txt2img/prompt_01_red_sports_base.png`
- `oft`: `outputs/report_eval/final_adapter/txt2img/prompt_01_red_sports_oft.png`

### prompt_02_white_supercar

- `prompt`: `studio photo of a skslego white supercar made of toy bricks, white seamless background`
- `seed`: `52`
- `base`: `outputs/report_eval/final_adapter/txt2img/prompt_02_white_supercar_base.png`
- `oft`: `outputs/report_eval/final_adapter/txt2img/prompt_02_white_supercar_oft.png`

### prompt_03_black_racing

- `prompt`: `studio photo of a skslego black racing car made of toy bricks, three-quarter front view, white studio background`
- `seed`: `62`
- `base`: `outputs/report_eval/final_adapter/txt2img/prompt_03_black_racing_base.png`
- `oft`: `outputs/report_eval/final_adapter/txt2img/prompt_03_black_racing_oft.png`

## img2img Inputs

- `prompt`: `a skslego version of this car, toy brick model, studio product photo, white background`

### 0

- `input`: `outputs/report_eval/final_adapter/img2img/0_input.png`
- `base`: `outputs/report_eval/final_adapter/img2img/0_base.png`
- `oft`: `outputs/report_eval/final_adapter/img2img/0_oft.png`

### 1

- `input`: `outputs/report_eval/final_adapter/img2img/1_input.png`
- `base`: `outputs/report_eval/final_adapter/img2img/1_base.png`
- `oft`: `outputs/report_eval/final_adapter/img2img/1_oft.png`

### 2

- `input`: `outputs/report_eval/final_adapter/img2img/2_input.png`
- `base`: `outputs/report_eval/final_adapter/img2img/2_base.png`
- `oft`: `outputs/report_eval/final_adapter/img2img/2_oft.png`

### 3

- `input`: `outputs/report_eval/final_adapter/img2img/3_input.png`
- `base`: `outputs/report_eval/final_adapter/img2img/3_base.png`
- `oft`: `outputs/report_eval/final_adapter/img2img/3_oft.png`
