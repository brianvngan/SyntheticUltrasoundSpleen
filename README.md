# DoRA SDXL Fine-Tuning Project

A complete implementation of **Weight-Decomposed Low-Rank Adaptation (DoRA)** for fine-tuning Stable Diffusion XL (SDXL) on NVIDIA H200 GPU.

## 📋 Project Overview

This project enables fine-tuning SDXL using DoRA on ultrasound images (specifically spleen ultrasound images) without using segmentation masks, as per professor's requirements. The project includes:

- **Advanced Training Pipeline**: Full DoRA implementation with gradient checkpointing and mixed precision
- **Comprehensive API**: High-level Python API, Flask REST API, and CLI interface
- **Dataset Management**: Unified dataset loading from multiple ultrasound sources
- **Inference Engine**: Generate high-quality ultrasound images with trained DoRA weights
- **Experiment Tracking**: W&B integration for monitoring training progress

## 🏗️ Project Structure

```
.
├── config/
│   └── dora_sdxl.yaml              # Main configuration file
├── data/
│   ├── __init__.py
│   └── dataset.py                  # Dataset loading & preprocessing
├── utils/
│   ├── __init__.py
│   ├── logging.py                  # W&B + console logging
│   └── checkpoint.py               # Checkpoint save/load logic
├── api/
│   ├── __init__.py
│   ├── core.py                     # High-level Python API
│   ├── app.py                      # Flask REST API
│   └── cli.py                      # Command-line interface
├── train_dora_sdxl.py              # Main training script
├── inference.py                    # Inference/generation script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd /path/to/Impact

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/dora_sdxl.yaml` to customize:
- Training hyperparameters
- DoRA settings (rank, target modules)
- Dataset paths and sizes
- Logging and checkpoint settings

Key configuration sections:
```yaml
training:
  num_epochs: 10
  learning_rate: 1.0e-4
  batch_size: 4

dora:
  r: 64
  use_dora: true
  target_modules: [...]

data:
  dataset_names: ["deepspv", "simulation"]
  image_size: 1024
```

### 3. Training

#### Using Python API

```python
from api.core import DoRAProject

project = DoRAProject(
    config_path="config/dora_sdxl.yaml",
    device="cuda"
)

# Prepare and train
project.train(num_epochs=10)
```

#### Using CLI

```bash
# Prepare training
python -m api.cli training prepare

# Start training
python -m api.cli training start --epochs 10
```

#### Using Training Script

```bash
python train_dora_sdxl.py --config config/dora_sdxl.yaml
```

#### Using Flask API

```bash
# Start API server
python -m api.app --port 5000

# In another terminal, send requests
curl http://localhost:5000/training/prepare
curl -X POST http://localhost:5000/training/start \
  -H "Content-Type: application/json" \
  -d '{"num_epochs": 10}'
```

### 4. Inference

#### Using Python API

```python
from api.core import DoRAProject
from pathlib import Path

project = DoRAProject(device="cuda")

images = project.generate(
    prompt="a high quality ultrasound image of the spleen",
    dora_weights_path="./checkpoints/final_model",
    num_images=4,
)

project.save_generated_images(images, output_dir="outputs")
```

#### Using CLI

```bash
python -m api.cli inference generate \
  --weights ./checkpoints/final_model \
  --prompt "ultrasound image of spleen" \
  --num-images 4 \
  --steps 50
```

#### Using Inference Script

```bash
python inference.py \
  --dora-weights ./checkpoints/final_model \
  --prompt "ultrasound image of spleen" \
  --num-images 4
```

#### Using Flask API

```bash
# Start API
python -m api.app --port 5000

# Generate images
curl -X POST http://localhost:5000/inference/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "ultrasound image of spleen",
    "dora_weights_path": "./checkpoints/final_model",
    "num_images": 4
  }'
```

## 📊 API Reference

### Python API (High-Level)

```python
from api import DoRAProject, DoRAConfig

# Initialize project
project = DoRAProject(config_path="config/dora_sdxl.yaml", device="cuda")

# Configuration Management
project.config.get("training.learning_rate")
project.config.set("training.learning_rate", 5e-5)
project.config.save("config_backup.yaml")

# Dataset Management
project.dataset_manager.load()
train_loader = project.dataset_manager.get_train_dataloader()

# Model Management
project.model_manager.load_base_models()
project.model_manager.apply_dora()
project.model_manager.enable_gradient_checkpointing()

# Training
project.train(num_epochs=10)

# Inference
project.prepare_inference("./checkpoints/final_model")
images = project.inference_manager.generate(
    prompt="ultrasound image",
    num_images=4,
)
```

### CLI Interface

```bash
# Configuration
python -m api.cli config show
python -m api.cli config get training.learning_rate
python -m api.cli config set training.learning_rate 5e-5
python -m api.cli config save --output new_config.yaml

# Dataset
python -m api.cli dataset load
python -m api.cli dataset info

# Model
python -m api.cli model load
python -m api.cli model apply-dora

# Training
python -m api.cli training prepare
python -m api.cli training start --epochs 10

# Inference
python -m api.cli inference prepare --weights ./checkpoints/final_model
python -m api.cli inference generate \
  --weights ./checkpoints/final_model \
  --prompt "spleen ultrasound" \
  --num-images 4 \
  --steps 50 \
  --guidance-scale 7.5
```

### Flask REST API

**Base URL:** `http://localhost:5000`

#### Health & Status

- `GET /health` - Health check
- `GET /status` - Project status
- `GET /api/version` - API version

#### Configuration

- `GET /config` - Get configuration
- `PUT /config` - Update configuration
- `POST /config/save` - Save configuration

#### Dataset Management

- `GET /datasets` - Get dataset info
- `POST /datasets/load` - Load datasets

#### Model Management

- `GET /models/info` - Get model information
- `POST /models/load` - Load base models
- `POST /models/apply-dora` - Apply DoRA to UNet

#### Training

- `POST /training/prepare` - Prepare for training
- `POST /training/start` - Start training

#### Inference

- `POST /inference/prepare` - Prepare inference
- `POST /inference/generate` - Generate images
- `GET /inference/image/<filename>` - Download generated image

## 🔑 Key Features

### DoRA Implementation
- ✅ Weight-decomposed low-rank adaptation
- ✅ Magnitude and direction decoupling
- ✅ Configurable rank and alpha
- ✅ Supports multiple target modules

### Training Features
- ✅ Mixed precision (BF16) for H200
- ✅ Gradient checkpointing for memory efficiency
- ✅ Learning rate scheduling (cosine with warmup)
- ✅ Checkpoint saving and resumption
- ✅ Validation image generation during training

### Dataset Features
- ✅ Multi-source ultrasound dataset support
- ✅ **No mask usage** (images only, as requested)
- ✅ Data augmentation (rotation, flip, color jitter)
- ✅ Automatic train/test splitting
- ✅ Memory-efficient batch loading

### Logging & Monitoring
- ✅ Weights & Biases integration
- ✅ Console logging
- ✅ Training progress tracking
- ✅ Gradient and learning rate monitoring
- ✅ Sample image logging

## 📚 Data Format

The project expects processed ultrasound datasets in NumPy format:

```
Processed_NPZ_Dataset/
├── DeepSPV_Synthetic_Dataset/
│   ├── synthetic_img_00001.npz
│   └── ...
├── US simulation & segmentation/
│   ├── synthetic_img_*.npz
│   └── ...
└── US.v7i.yolov8-obb Roboflow/
    ├── real_img_*.npz
    └── ...
```

Each `.npz` file contains:
- `image`: RGB ultrasound image (uint8, 1024×1024)
- `label`: "Synthetic" or "Real" classification
- `original_path`: Original file path metadata

**Important:** Only the `image` field is used during training. Masks are NOT used per professor's requirements.

## 🎯 Configuration Examples

### Minimal Configuration

```yaml
training:
  num_epochs: 3
  batch_size: 2
  learning_rate: 1e-4

data:
  dataset_names: ["deepspv"]
  train_split: 0.8
  image_size: 512  # Smaller for faster testing
```

### High-Performance Configuration (H200)

```yaml
training:
  num_epochs: 20
  max_steps: 100000
  batch_size: 8
  gradient_accumulation_steps: 4
  mixed_precision: "bf16"
  gradient_checkpointing: true

dora:
  r: 128  # Larger rank for more capacity
  lora_alpha: 128

data:
  dataset_names: ["deepspv", "simulation", "roboflow"]
  image_size: 1024
  augmentation: true

logging:
  use_wandb: true
  wandb_project: "dora-sdxl-production"
```

## 📈 Monitoring Training

### Using W&B Dashboard

1. Set `wandb_entity` in config to your W&B username
2. Training automatically logs to W&B
3. Monitor at: https://wandb.ai/your-username/dora-sdxl

### Using Console Logs

Training logs are saved to `logs/` directory:
```bash
tail -f logs/dora_sdxl_training_*.log
```

### Checkpoint Management

Trained models are saved to `checkpoints/` directory:
```
checkpoints/
├── checkpoint-best/           # Best model during training
├── checkpoint-epoch-001-step-001000/
├── final_model/               # Final trained model
└── ...
```

## 🔧 Troubleshooting

### CUDA Out of Memory

1. Reduce batch size: `training.batch_size: 2`
2. Enable gradient checkpointing: `training.gradient_checkpointing: true`
3. Reduce LoRA rank: `dora.r: 32`
4. Use smaller image size: `data.image_size: 512`

### Slow Dataset Loading

1. Increase workers: `training.num_workers: 8`
2. Enable pin_memory in dataloader (default: True)

### Model not improving

1. Try higher learning rate: `training.learning_rate: 5e-4`
2. Increase warmup steps: `training.warmup_steps: 2000`
3. Verify dataset loading: `python -m api.cli dataset info`

## 📝 References

- **DoRA Paper**: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- **PEFT Library**: https://huggingface.co/docs/peft/
- **Diffusers**: https://huggingface.co/docs/diffusers/
- **SDXL Model**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

## 📄 License

This project is part of the IMPACT research program.

## 👥 Contributing

For questions or contributions, please contact the project maintainers.

---

**Last Updated:** April 11, 2026
**Version:** 1.0.0
