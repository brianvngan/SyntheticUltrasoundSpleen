# 📚 Complete API Reference

## Quick Reference Guide

All available APIs, commands, and endpoints for the DoRA SDXL project.

---

## 🐍 Python API Reference

### Import
```python
from api import DoRAProject, DoRAConfig
```

### Configuration Management
```python
config = DoRAConfig("config/dora_sdxl.yaml")

# Get values
lr = config.get("training.learning_rate")
batch_size = config.get("training.batch_size")

# Set values
config.set("training.learning_rate", 5e-5)
config.set("training.batch_size", 8)

# Save configuration
config.save("my_config.yaml")

# Get all config as dict
full_config = config.to_dict()
```

### Dataset Management
```python
project = DoRAProject()

# Load datasets
train_loader, test_loader = project.dataset_manager.load()

# Get dataset info
train_dataset = project.dataset_manager.train_dataset
test_dataset = project.dataset_manager.test_dataset
print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# Get dataloaders
train_loader = project.dataset_manager.get_train_dataloader()
test_loader = project.dataset_manager.get_test_dataloader()
```

### Model Management
```python
project = DoRAProject(device="cuda")

# Load base models
project.model_manager.load_base_models()

# Apply DoRA
project.model_manager.apply_dora()

# Enable optimizations
project.model_manager.enable_gradient_checkpointing()

# Move to device
project.model_manager.to("cuda")

# Get models dict
models = project.model_manager.get_models()
```

### Training
```python
project = DoRAProject()

# Prepare for training
project.prepare_training()

# Train
project.train(num_epochs=10)

# Or with custom config
project.config.set("training.num_epochs", 20)
project.config.set("training.batch_size", 8)
project.train()
```

### Inference
```python
project = DoRAProject(device="cuda")

# Prepare inference pipeline
project.prepare_inference("./checkpoints/final_model")

# Generate images
images = project.inference_manager.generate(
    prompt="ultrasound image of spleen",
    num_images=4,
    num_steps=50,
    guidance_scale=7.5,
)

# Save images
saved_paths = project.save_generated_images(
    images,
    output_dir="outputs"
)

# Or use high-level method
images = project.generate(
    prompt="spleen ultrasound",
    dora_weights_path="./checkpoints/final_model",
    num_images=4,
)
```

---

## 💻 CLI Reference

### Configuration Commands

```bash
# Show all configuration
python -m api.cli config show

# Get specific setting
python -m api.cli config get training.learning_rate
python -m api.cli config get dora.r
python -m api.cli config get data.image_size

# Set configuration value
python -m api.cli config set training.learning_rate 5e-5
python -m api.cli config set training.batch_size 8
python -m api.cli config set dora.r 128

# Save configuration
python -m api.cli config save
python -m api.cli config save --output custom_config.yaml
```

### Dataset Commands

```bash
# Load datasets
python -m api.cli dataset load

# Get dataset info
python -m api.cli dataset info
```

### Model Commands

```bash
# Load base SDXL models
python -m api.cli model load

# Apply DoRA to UNet
python -m api.cli model apply-dora
```

### Training Commands

```bash
# Prepare for training
python -m api.cli training prepare

# Start training
python -m api.cli training start --epochs 10
python -m api.cli training start --epochs 20
```

### Inference Commands

```bash
# Prepare inference
python -m api.cli inference prepare --weights ./checkpoints/final_model

# Generate images
python -m api.cli inference generate \
  --weights ./checkpoints/final_model \
  --prompt "ultrasound image of spleen" \
  --num-images 4

# With additional options
python -m api.cli inference generate \
  --weights ./checkpoints/final_model \
  --prompt "high quality spleen ultrasound" \
  --num-images 10 \
  --steps 50 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output-dir outputs
```

---

## 🌐 REST API Endpoints

### Start Server

```bash
python -m api.app --port 5000 --device cuda
```

### Health & Status Endpoints

```bash
# Health check
GET /health
Response: {"status": "healthy", "timestamp": "..."}

# Project status
GET /status
Response: {"status": "initialized", "device": "cuda", ...}

# API version
GET /api/version
Response: {"version": "1.0.0", "api": "DoRA SDXL API"}
```

### Configuration Endpoints

```bash
# Get configuration
GET /config
Response: {config dict}

# Update configuration
PUT /config
Body: {"training.learning_rate": 5e-5, "training.batch_size": 8}
Response: {"status": "success", "config": {...}}

# Save configuration
POST /config/save
Body: {"output_path": "custom_config.yaml"}
Response: {"status": "success", "path": "..."}
```

### Dataset Endpoints

```bash
# Get dataset info
GET /datasets
Response: {
  "train_samples": 1000,
  "test_samples": 250,
  "train_batches": 250,
  "batch_size": 4
}

# Load datasets
POST /datasets/load
Response: {
  "status": "success",
  "train_samples": 1000,
  "test_samples": 250
}
```

### Model Endpoints

```bash
# Get model info
GET /models/info
Response: {"unet": {...}, "vae": {...}, ...}

# Load base models
POST /models/load
Response: {"status": "success", "message": "Models loaded"}

# Apply DoRA
POST /models/apply-dora
Response: {"status": "success", "message": "DoRA applied"}
```

### Training Endpoints

```bash
# Prepare training
POST /training/prepare
Response: {"status": "success"}

# Start training
POST /training/start
Body: {"num_epochs": 10}
Response: {"status": "success", "message": "Training started"}
```

### Inference Endpoints

```bash
# Prepare inference
POST /inference/prepare
Body: {"dora_weights_path": "./checkpoints/final_model"}
Response: {"status": "success"}

# Generate images
POST /inference/generate
Body: {
  "prompt": "ultrasound image",
  "dora_weights_path": "./checkpoints/final_model",
  "num_images": 4,
  "num_steps": 50,
  "guidance_scale": 7.5,
  "seed": 42,
  "output_dir": "outputs"
}
Response: {
  "status": "success",
  "saved_paths": ["outputs/generated_0000.png", ...]
}

# Download image
GET /inference/image/<filename>
Response: Image file
```

---

## 📝 Configuration Keys Reference

### Training Parameters
```yaml
training:
  num_epochs                    # Number of training epochs
  max_steps                     # Override epochs with max steps
  warmup_steps                  # LR warmup steps
  learning_rate                 # Learning rate
  weight_decay                  # AdamW weight decay
  gradient_accumulation_steps   # Gradient accumulation
  batch_size                    # Batch size per GPU
  num_workers                   # DataLoader workers
  mixed_precision               # bf16 or fp32
  gradient_checkpointing        # Enable gradient checkpointing
  seed                          # Random seed
```

### DoRA Configuration
```yaml
dora:
  r                            # LoRA rank
  lora_alpha                   # Scaling factor
  lora_dropout                 # Dropout probability
  use_dora                     # Enable DoRA
  init_lora_weights            # Weight initialization
  bias                         # Bias training
  target_modules               # Modules to apply DoRA to
```

### Data Configuration
```yaml
data:
  dataset_names                # List of datasets
  train_split                  # Train/test split
  image_size                   # Target image size
  center_crop                  # Center crop
  random_flip                  # Random horizontal flip
  random_rotation              # Random rotation degrees
  normalize                    # Normalize to [-1, 1]
  augmentation                 # Enable augmentation
```

### Model Configuration
```yaml
model:
  pretrained_model_name_or_path  # Base model ID
  freeze_vae                     # Freeze VAE
  freeze_text_encoders           # Freeze text encoders
  unet_dtype                     # UNet precision
  vae_dtype                      # VAE precision
  text_encoder_dtype             # Text encoder precision
```

### Validation Configuration
```yaml
validation:
  validate_every_n_steps         # Validation frequency
  num_validation_images          # Number of validation images
  validation_steps               # Denoising steps
  guidance_scale                 # Classifier-free guidance
  negative_prompt                # Negative prompt
  seed                           # Validation seed
```

### Logging Configuration
```yaml
logging:
  log_every_n_steps              # Logging frequency
  log_gradients                  # Log gradients
  log_learning_rate              # Log LR
  log_sample_images              # Log sample images
  use_wandb                      # Use W&B
  wandb_project                  # W&B project
  wandb_entity                   # W&B entity
  wandb_notes                    # W&B notes
```

### Checkpointing Configuration
```yaml
checkpointing:
  save_checkpoint_every_n_steps  # Checkpoint frequency
  save_best_model                # Save best model
  checkpoint_dir                 # Checkpoint directory
  resume_from_checkpoint         # Resume path
  keep_last_n_checkpoints        # Keep last N
```

---

## 🎯 Common Workflows

### Training Workflow
```bash
# 1. Validate setup
python validate_setup.py

# 2. Configure (optional)
python -m api.cli config set training.batch_size 8

# 3. Prepare
python -m api.cli training prepare

# 4. Train
python -m api.cli training start --epochs 10

# 5. Monitor
# Check logs or W&B dashboard
```

### Inference Workflow
```bash
# 1. Prepare inference
python -m api.cli inference prepare --weights ./checkpoints/final_model

# 2. Generate
python -m api.cli inference generate \
  --weights ./checkpoints/final_model \
  --prompt "spleen ultrasound" \
  --num-images 4
```

### API Server Workflow
```bash
# 1. Start server
python -m api.app --port 5000

# 2. Load data
curl -X POST http://localhost:5000/datasets/load

# 3. Train
curl -X POST http://localhost:5000/training/start \
  -d '{"num_epochs": 10}'

# 4. Generate
curl -X POST http://localhost:5000/inference/generate \
  -d '{"prompt": "ultrasound", "num_images": 4}'
```

---

## 📊 Example Requests

### Python API
```python
from api import DoRAProject

project = DoRAProject()
project.prepare_training()
project.train(num_epochs=10)
```

### CLI
```bash
python -m api.cli training start --epochs 10
```

### REST API
```bash
curl -X POST http://localhost:5000/training/start \
  -H "Content-Type: application/json" \
  -d '{"num_epochs": 10}'
```

---

## 🔍 Help & Information

```bash
# Show all commands
python -m api.cli --help

# Show command help
python -m api.cli training --help
python -m api.cli inference --help

# Show API info
curl http://localhost:5000/api/version

# Show config
python -m api.cli config show

# Validate setup
python validate_setup.py
```

---

## 💾 File Locations

- Configuration: `config/dora_sdxl.yaml`
- Checkpoints: `checkpoints/`
- Generated images: `outputs/`
- Logs: `logs/`
- Processed datasets: `Processed Datasets/Processed_NPZ_Dataset/`

---

## ✨ Quick Reference

| Task | Command |
|------|---------|
| Validate setup | `python validate_setup.py` |
| Load data | `python -m api.cli dataset load` |
| Train | `python -m api.cli training start --epochs 10` |
| Generate | `python -m api.cli inference generate --prompt "ultrasound"` |
| Show config | `python -m api.cli config show` |
| Start API | `python -m api.app --port 5000` |
| View docs | See README.md |

---

**All APIs, commands, and endpoints are fully documented and ready to use!**

