# 🚀 Getting Started with DoRA SDXL Project

**Quick start guide for the DoRA fine-tuning project**

---

## ⚡ 5-Minute Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Validate Setup
```bash
python validate_setup.py
```

### 3. Load Datasets
```bash
python -m api.cli dataset load
python -m api.cli dataset info
```

### 4. Start Training
```bash
python -m api.cli training prepare
python -m api.cli training start --epochs 5
```

### 5. Generate Images
```bash
python -m api.cli inference prepare --weights ./checkpoints/final_model
python -m api.cli inference generate \
  --weights ./checkpoints/final_model \
  --prompt "ultrasound image of spleen" \
  --num-images 4
```

---

## 📖 Detailed Setup

### Step 1: Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA 12.1+
- 16GB+ VRAM (H200 has 141GB)

### Step 2: Clone and Setup
```bash
cd /Users/davidgogi/Desktop/Impact
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python validate_setup.py
```

Expected output:
```
✓ Python 3.10+ version
✓ All directories exist
✓ All required files present
✓ Core packages installed
✓ CUDA available
✓ Configuration valid
✓ Datasets available
```

### Step 4: Customize Configuration (Optional)
Edit `config/dora_sdxl.yaml`:
```yaml
training:
  num_epochs: 10
  batch_size: 4          # Adjust for your GPU
  learning_rate: 1e-4

dora:
  r: 64                  # LoRA rank
  use_dora: true         # DoRA enabled

data:
  dataset_names: ["deepspv", "simulation"]
  image_size: 1024
```

---

## 🎯 Three Ways to Use the Project

### Option 1: Python API (Recommended for Scripts)
```python
from api.core import DoRAProject

# Initialize
project = DoRAProject(device="cuda")

# Configure
project.config.set("training.num_epochs", 10)

# Load data
project.dataset_manager.load()

# Train
project.prepare_training()
project.train()

# Generate
images = project.generate(
    prompt="ultrasound image",
    dora_weights_path="./checkpoints/final_model",
    num_images=4
)
```

### Option 2: Command-Line Interface (Recommended for Quick Tasks)
```bash
# View configuration
python -m api.cli config show

# Modify configuration
python -m api.cli config set training.batch_size 8

# Load and check datasets
python -m api.cli dataset load
python -m api.cli dataset info

# Training workflow
python -m api.cli training prepare
python -m api.cli training start --epochs 10

# Inference
python -m api.cli inference prepare --weights ./checkpoints/final_model
python -m api.cli inference generate --weights ./checkpoints/final_model \
  --prompt "spleen ultrasound" --num-images 4
```

### Option 3: Direct Scripts (For Fine-Grained Control)
```bash
# Training
python train_dora_sdxl.py --config config/dora_sdxl.yaml

# Inference
python inference.py --dora-weights ./checkpoints/final_model \
  --prompt "ultrasound image" --num-images 4
```

### Option 4: REST API (Recommended for Integration)
```bash
# Start server
python -m api.app --port 5000 --device cuda

# In another terminal, use endpoints
curl http://localhost:5000/health
curl -X POST http://localhost:5000/datasets/load
curl -X POST http://localhost:5000/training/prepare
curl -X POST http://localhost:5000/training/start \
  -H "Content-Type: application/json" \
  -d '{"num_epochs": 10}'
```

---

## 📊 Monitoring Training

### Using Weights & Biases
1. Configure in `config/dora_sdxl.yaml`:
```yaml
logging:
  use_wandb: true
  wandb_project: "dora-sdxl-ultrasound"
  wandb_entity: "your-username"  # Add your W&B username
```

2. Visit dashboard at: `https://wandb.ai/your-username/dora-sdxl-ultrasound`

### Using Console Logs
```bash
tail -f logs/dora_sdxl_training_*.log
```

### Checking Checkpoints
```bash
ls -lh checkpoints/
ls -lh checkpoints/checkpoint-best/
```

---

## 🔍 Common Operations

### Check Dataset Info
```bash
python -m api.cli dataset info
```

### Modify Configuration
```bash
# View specific setting
python -m api.cli config get training.learning_rate

# Change setting
python -m api.cli config set training.learning_rate 5e-5

# Save modified config
python -m api.cli config save --output my_config.yaml
```

### Resume Training from Checkpoint
```bash
# Edit config/dora_sdxl.yaml:
checkpointing:
  resume_from_checkpoint: "./checkpoints/checkpoint-best"

# Then start training
python -m api.cli training start --epochs 10
```

### Generate Multiple Images
```bash
python -m api.cli inference generate \
  --weights ./checkpoints/final_model \
  --prompt "high quality spleen ultrasound" \
  --num-images 10 \
  --steps 50 \
  --seed 42
```

---

## ⚙️ Configuration Examples

### Minimal (Testing)
```yaml
training:
  num_epochs: 1
  batch_size: 2
  learning_rate: 1e-4
data:
  dataset_names: ["deepspv"]
  image_size: 512
```

### Standard (Default)
```yaml
training:
  num_epochs: 10
  batch_size: 4
  learning_rate: 1e-4
data:
  dataset_names: ["deepspv", "simulation"]
  image_size: 1024
```

### High-Performance (H200)
```yaml
training:
  num_epochs: 20
  max_steps: 100000
  batch_size: 8
  gradient_accumulation_steps: 4
data:
  dataset_names: ["deepspv", "simulation", "roboflow"]
  image_size: 1024
  augmentation: true
dora:
  r: 128
  lora_alpha: 128
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
python -m api.cli config set training.batch_size 2

# Or reduce image size
python -m api.cli config set data.image_size 512

# Or reduce LoRA rank
python -m api.cli config set dora.r 32
```

### Dataset Not Loading
```bash
# Verify datasets are processed
ls -la Processed\ Datasets/Processed_NPZ_Dataset/*/

# Check dataset info
python -m api.cli dataset info

# If empty, run process_datasets.py first
cd Processed\ Datasets
python process_datasets.py
```

### Training Not Starting
```bash
# Validate setup
python validate_setup.py

# Check logs
tail -f logs/dora_sdxl_training_*.log

# Verify models can load
python -m api.cli model load
```

### Slow Training
- Increase workers: `config set training.num_workers 8`
- Check GPU usage: `nvidia-smi`
- Verify CUDA is available: `validate_setup.py`

---

## 📚 Project Files Overview

| File | Purpose |
|------|---------|
| `config/dora_sdxl.yaml` | Main configuration file |
| `train_dora_sdxl.py` | Direct training script |
| `inference.py` | Direct inference script |
| `api/core.py` | High-level Python API |
| `api/cli.py` | Command-line interface |
| `api/app.py` | Flask REST API |
| `data/dataset.py` | Dataset loading |
| `utils/logging.py` | Logging system |
| `utils/checkpoint.py` | Checkpoint management |
| `validate_setup.py` | Setup validation |

---

## ✅ Checklist

- [ ] Python 3.10+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Setup validated: `python validate_setup.py` (all ✓)
- [ ] Configuration reviewed: `python -m api.cli config show`
- [ ] Datasets available: `python -m api.cli dataset info`
- [ ] W&B account setup (if using): `wandb login`
- [ ] Ready to train! `python -m api.cli training start`

---

## 🎓 Next Steps

1. **Read Full Documentation**
   - `README.md` - Complete guide with all details
   - `examples.py` - Practical usage examples
   - `PROJECT_SUMMARY.md` - Project overview

2. **Start Small**
   - Configure for testing: see "Minimal" config above
   - Train for 1-2 epochs to verify setup
   - Check logs and outputs

3. **Scale Up**
   - Increase epochs and dataset
   - Monitor W&B dashboard
   - Save best checkpoints

4. **Generate Images**
   - Once training complete
   - Use various prompts
   - Export high-quality results

---

## 💡 Tips

- **Always validate setup first**: `python validate_setup.py`
- **Use CLI for quick changes**: `python -m api.cli config set ...`
- **Monitor W&B for detailed metrics**: Real-time training visualization
- **Save configs before training**: `python -m api.cli config save`
- **Use gradual increases**: Start small, scale up progressively
- **Check GPU memory**: `nvidia-smi` during training

---

## 📞 Quick Help

```bash
# Show all CLI commands
python -m api.cli --help

# Get help for specific command
python -m api.cli training --help
python -m api.cli inference --help

# Check system compatibility
python validate_setup.py

# View current configuration
python -m api.cli config show
```

---

**Ready to start?** Run:
```bash
python validate_setup.py
```

If all checks pass, you're ready to train!

