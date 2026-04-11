# CLAUDE.md — DoRA Fine-Tuning SDXL on H200

## Project Goal
Fine-tune Stable Diffusion XL (SDXL) using **DoRA (Weight-Decomposed Low-Rank Adaptation)** on an NVIDIA H200 GPU. DoRA extends LoRA by decomposing pretrained weights into magnitude and direction, updating each independently for superior fine-tune quality at near-identical parameter cost.

---

## Hardware & Environment
- **GPU:** NVIDIA H200 (141 GB HBM3e) — single node unless told otherwise
- **OS:** Linux (Ubuntu 22.04+)
- **Python:** 3.10+
- **CUDA:** 12.1+
- **Precision:** BF16 throughout (H200 has native BF16 throughput advantage)

---

## Stack
```
diffusers>=0.27.0
peft>=0.10.0          # DoRA support added in 0.9.0
transformers>=4.38.0
accelerate>=0.27.0
torch>=2.2.0
bitsandbytes>=0.43.0  # optional: for 8-bit Adam
xformers              # optional: for memory-efficient attention
datasets
Pillow
wandb                 # for experiment tracking
```

Install with:
```bash
pip install diffusers peft transformers accelerate bitsandbytes xformers datasets wandb
```

---

## Key Files to Create / Maintain
```
.
├── CLAUDE.md               ← this file
├── train_dora_sdxl.py      ← main training script
├── config/
│   └── dora_sdxl.yaml      ← hyperparameter config
├── data/
│   └── dataset.py          ← dataset loading & preprocessing
├── utils/
│   ├── checkpoint.py       ← save/resume logic
│   └── logging.py          ← wandb + console logging
└── inference.py            ← run inference with trained DoRA weights
```

---

## DoRA Configuration (PEFT)
DoRA is enabled in PEFT via `use_dora=True` on a `LoraConfig`:

```python
from peft import LoraConfig, get_peft_model

dora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    init_lora_weights="gaussian",
    target_modules=[
        "to_q", "to_k", "to_v", "to_out.0",     # self-attention
        "proj_in", "proj_out",                     # transformer proj
        "ff.net.0.proj", "ff.net.2",              # feed-forward
    ],
    lora_dropout=0.05,
    use_dora=True,   # ← this is the key flag that enables DoRA
    bias="none",
)
```

Apply to the UNet:
```python
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="unet",
    torch_dtype=torch.bfloat16,
)
unet = get_peft_model(unet, dora_config)
unet.print_trainable_parameters()
```

---

## Training Loop Expectations
- Use `accelerate` for distributed/mixed-precision management
- Gradient checkpointing: `unet.enable_gradient_checkpointing()`
- Optimizer: AdamW (or `bitsandbytes.optim.AdamW8bit` for memory savings)
- Scheduler: cosine with warmup
- Loss: standard diffusion MSE loss on noise prediction
- Batch size: start with 4–8 per GPU given H200 VRAM
- Learning rate: 1e-4 for DoRA (slightly higher than LoRA is fine due to magnitude decoupling)

```python
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, unet.parameters()),
    lr=1e-4,
    weight_decay=1e-2,
)
```

---

## Data Format
Expect an image-caption dataset structured as:
```
data/images/
    img_001.jpg
    img_002.jpg
    ...
data/captions/
    img_001.txt
    img_002.txt
    ...
```
Or a HuggingFace dataset with `image` and `text` columns.
SDXL requires images at 1024×1024 (or multi-aspect with bucketing).

---

## Saving & Loading DoRA Weights
```python
# Save
unet.save_pretrained("./checkpoints/dora-sdxl-step-1000")

# Load for inference
from peft import PeftModel
unet = UNet2DConditionModel.from_pretrained(...)
unet = PeftModel.from_pretrained(unet, "./checkpoints/dora-sdxl-step-1000")
```

---

## Inference Script Expectations
`inference.py` should:
1. Load base SDXL pipeline
2. Load DoRA weights into UNet via `PeftModel.from_pretrained`
3. Accept a prompt via CLI arg
4. Output image to `./outputs/`

```bash
python inference.py --prompt "a photo of <subject>" --steps 30 --guidance 7.5
```

---

## Coding Standards
- Type hints on all functions
- Docstrings on all classes and public methods
- Config via YAML (use `omegaconf` or `pydantic`), not hardcoded values
- All file paths relative to project root
- Reproducibility: always set and log random seeds
- Log: step loss, grad norm, lr, and sample images every N steps to wandb

---

## Common Pitfalls to Avoid
1. **Don't apply DoRA to the text encoders** — target UNet only (CLIP encoders are usually frozen)
2. **Don't forget to freeze VAE** — `vae.requires_grad_(False)`
3. **Check PEFT version** — `use_dora=True` requires `peft>=0.9.0`
4. **BF16 + DoRA:** the magnitude vector is small; ensure it doesn't underflow — keep in FP32 if needed via `lora_dtype=torch.float32` in config
5. **SDXL uses two text encoders** — `CLIPTextModel` and `CLIPTextModelWithProjection`; both need to be loaded and frozen

---

## References
- DoRA paper: "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al., 2024) — https://arxiv.org/abs/2402.09353
- PEFT DoRA docs: https://huggingface.co/docs/peft/developer_guides/lora#weight-decomposed-low-rank-adaptation-dora
- Diffusers SDXL training: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py
