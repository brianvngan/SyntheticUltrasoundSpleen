# Complete Code Explanation

This document explains the entire DoRA SDXL project codebase, breaking it down into digestible sections.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Key Components](#key-components)
4. [File Structure](#file-structure)
5. [How Everything Works Together](#how-everything-works-together)

---

## Project Overview

### What is this project?

This is a **fine-tuning framework** for **Stable Diffusion XL (SDXL)** using **DoRA (Weight-Decomposed Low-Rank Adaptation)** on ultrasound medical images.

### Why DoRA?

- **LoRA** (Low-Rank Adaptation) is efficient but has limitations
- **DoRA** improves upon LoRA by separating weight updates into:
  - **Magnitude component** (how much to change)
  - **Direction component** (which way to change)
  - This gives better quality with minimal extra parameters

### Why H200?

The NVIDIA H200 GPU has:
- **141 GB HBM3e** memory (huge!)
- Native **BF16** precision support (fast & accurate)
- Perfect for training large models like SDXL

### The Goal

Fine-tune SDXL to generate better ultrasound images using:
- **DeepSPV Synthetic Dataset** (400 images)
- **US Simulation Dataset** (400 images)  
- **Roboflow Real Ultrasounds** (445 images)

**IMPORTANT:** Professor said NOT to use masks → we only load images, never segmentation masks

---

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                      │
├─────────────────────────────────────────────────────────────┤
│  Streamlit UI  │  REST API (Flask)  │  CLI  │  Python API  │
└────────┬─────────────┬──────────────────────┬───────────────┘
         │             │                      │
         └─────────────┴──────────┬───────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │    api/core.py             │
                    │  (Main API Logic)          │
                    │  - DoRAProject             │
                    │  - DatasetManager          │
                    │  - ModelManager            │
                    │  - TrainingManager         │
                    │  - InferenceManager        │
                    └────────────┬────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
   ┌─────────┐           ┌──────────────┐        ┌──────────────┐
   │ Training │           │   Inference  │        │ Dataset      │
   │ Pipeline │           │  Pipeline    │        │ Management   │
   └─────────┘           └──────────────┘        └──────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
   train_dora_sdxl.py  inference.py              data/dataset.py
```

### The Three-Tier API Design

We built **3 different ways to use the project**:

#### 1. **Python API** (Most programmatic)
```python
from api import DoRAProject

project = DoRAProject()
project.train(num_epochs=10)
project.generate(prompt="ultrasound image")
```

#### 2. **REST API** (Most scalable)
```bash
python -m api.app  # Starts Flask server
curl http://localhost:5000/training/start
```

#### 3. **CLI** (Most practical)
```bash
python -m api.cli training start --epochs 10
python -m api.cli inference generate --prompt "ultrasound"
```

---

## Key Components

### 1. **app_ui.py** - Streamlit Web Dashboard

**Purpose:** Beautiful demo UI to showcase the project

**Structure:**
```python
# Setup
st.set_page_config(...)  # Configure page
st.markdown(CSS_STYLING)  # Add custom styling

# Sidebar navigation
with st.sidebar:
    page = st.radio("Navigation", [...pages...])

# Conditional rendering
if page == "🏠 Dashboard":
    # Show dashboard content
elif page == "⚙️ Configuration":
    # Show config tabs
```

**Key Features:**
- 6 different pages accessible from sidebar
- Sliders for numeric parameters
- Text inputs for prompts
- Demo charts and data
- No actual training needed - purely for demonstration

**Example - Dashboard Page:**
```python
# Display 4 metrics in a row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Training Steps", "0", "Ready to start")

# Display charts
st.line_chart(df.set_index("Step")["Loss"], height=300)
```

---

### 2. **config/dora_sdxl.yaml** - Configuration File

**Purpose:** Single source of truth for all hyperparameters

**Structure:**
```yaml
training:
  num_epochs: 10                    # How many times through dataset
  learning_rate: 1.0e-4             # How fast model updates
  batch_size: 4                     # Images per GPU update
  gradient_accumulation_steps: 2    # Accumulate gradients before update
  mixed_precision: "bf16"           # Use BF16 for H200
  gradient_checkpointing: true      # Save memory during training

dora:
  r: 64                             # Rank of LoRA adaptation
  lora_alpha: 64                    # Scaling factor
  use_dora: true                    # CRITICAL: Enable DoRA
  target_modules: [...]             # Which model parts to train

data:
  dataset_names: ["deepspv", "simulation"]  # Which datasets
  image_size: 1024                  # SDXL wants 1024x1024
  train_split: 0.8                  # 80% train, 20% test
```

**How it works:**
1. Python loads this YAML file
2. Converts to dictionary
3. All code references values via `config.get("training.learning_rate")`
4. Can be modified programmatically or edited directly

---

### 3. **data/dataset.py** - Data Loading

**Purpose:** Load ultrasound images and prepare them for training

**Key Class: `UltrasoundImageDataset`**

```python
class UltrasoundImageDataset(Dataset):
    """
    Loads images from NPZ files.
    CRITICAL: Only loads 'image', NEVER loads 'mask'
    """
    
    def __init__(self, image_paths, image_size=1024, augmentation=True):
        self.image_paths = image_paths
        self.transform = self._build_transform()
    
    def __getitem__(self, idx):
        # Load NPZ file
        npz_data = np.load(self.image_paths[idx])
        
        # CRITICAL: Only get image, not mask
        image = npz_data['image']
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        # Apply augmentations
        image = self.transform(image)
        
        # Return: pixel values + caption
        return {
            'pixel_values': image,
            'text': 'ultrasound image',
            'original_path': str(self.image_paths[idx])
        }
```

**Data Augmentation Pipeline:**
```python
transforms.Compose([
    transforms.CenterCrop(...),           # Crop to center
    transforms.Resize(1024),              # Resize to 1024x1024
    transforms.RandomHorizontalFlip(0.5), # Random flip
    transforms.ColorJitter(...),          # Random color changes
    transforms.RandomAffine(...),         # Random rotation/scale
])
```

**Key Point:** 
- We load from files like `synthetic_img_00001.npz`
- Each NPZ contains: `{'image': array, 'other_data': ...}`
- We ONLY use `image`, never `mask` (professor's requirement)
- Applied to every image for variety

---

### 4. **train_dora_sdxl.py** - Training Script

**Purpose:** Main training loop that fine-tunes SDXL with DoRA

**Flow:**

```
1. Load Configuration
   └─ Read dora_sdxl.yaml
   
2. Load Models
   ├─ VAE (compress images)
   ├─ UNet (denoise)
   ├─ Text Encoders (understand prompts)
   
3. Apply DoRA
   └─ Wrap UNet with PEFT LoRA + use_dora=True
   
4. Setup Training
   ├─ Optimizer (AdamW)
   ├─ Scheduler (Cosine warmup)
   ├─ Accelerator (multi-GPU)
   
5. Training Loop
   ├─ For each epoch:
   │  └─ For each batch:
   │     ├─ Load images + prompts
   │     ├─ Encode prompts to embeddings
   │     ├─ Add noise to images
   │     ├─ Predict noise with UNet
   │     ├─ Calculate loss
   │     ├─ Backward pass
   │     └─ Update weights
   │
   └─ Validation: Generate images to check quality
   
6. Save Checkpoints
   └─ Save UNet weights + optimizer state
```

**Key Functions:**

#### `create_dora_config(config)`
```python
def create_dora_config(config):
    """Create PEFT config with DoRA enabled"""
    return LoraConfig(
        r=64,                          # Rank
        lora_alpha=64,                 # Scaling
        use_dora=True,                 # CRITICAL: DoRA enabled
        target_modules=['to_q', 'to_k', 'to_v', ...],
        lora_dropout=0.05,
    )
```

#### `encode_prompt(text_encoders, tokenizers, prompt, device)`
```python
def encode_prompt(text_encoders, tokenizers, prompt, device):
    """
    SDXL uses TWO text encoders (unlike SDXL 1.0 which uses one)
    
    1. CLIPTextModel (main embeddings)
    2. CLIPTextModelWithProjection (additional embeddings)
    
    We concatenate both for best quality
    """
    embeddings = []
    for tokenizer, encoder in zip(tokenizers, text_encoders):
        tokens = tokenizer(prompt)
        embedding = encoder(tokens)
        embeddings.append(embedding)
    
    return torch.cat(embeddings, dim=-1)
```

#### `train_step(images, prompts, ...)`
```python
def train_step(images, prompts):
    """One training iteration"""
    
    # 1. Encode prompts to embeddings
    embeddings = encode_prompt(prompts)
    
    # 2. Encode images to latent space
    latents = vae.encode(images).latent_dist.sample()
    
    # 3. Add random noise
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, 1000, ...)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # 4. Predict noise with UNet
    predicted_noise = unet(
        noisy_latents,
        timesteps,
        embeddings
    )
    
    # 5. Calculate loss
    loss = F.mse_loss(predicted_noise, noise)
    
    # 6. Backward pass
    loss.backward()
    
    # 7. Update weights
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()
```

**Important Concept - Denoising Training:**

SDXL is trained via diffusion:
1. Start with random noise
2. Gradually denoise over 1000 steps
3. Model learns to remove noise while conditioning on prompts

We're fine-tuning this by:
- Adding noise to ultrasound images
- Training model to predict the noise
- On ultrasound-specific text prompts

---

### 5. **api/core.py** - High-Level API

**Purpose:** Unified Python interface to entire project

**Main Class: `DoRAProject`**

```python
class DoRAProject:
    """Main entry point for all operations"""
    
    def __init__(self, config_path="config/dora_sdxl.yaml"):
        self.config = DoRAConfig(config_path)
        self.dataset_manager = DatasetManager(self.config)
        self.model_manager = ModelManager(self.config)
        self.training_manager = TrainingManager(self.config)
        self.inference_manager = InferenceManager(self.config)
    
    def train(self, num_epochs=None):
        """Train the model"""
        # Prepare
        self.dataset_manager.load()
        self.model_manager.load()
        self.training_manager.prepare()
        
        # Execute training
        self.training_manager.train(num_epochs)
    
    def generate(self, prompt, num_images=1, steps=50):
        """Generate images"""
        self.inference_manager.prepare()
        return self.inference_manager.generate(prompt, num_images, steps)
```

**Sub-managers:**

1. **`DoRAConfig`** - Configuration management
   ```python
   config.get("training.learning_rate")     # Get value
   config.set("training.learning_rate", 2e-4)  # Set value
   config.save()                            # Save to YAML
   ```

2. **`DatasetManager`** - Load and manage datasets
   ```python
   train_loader, test_loader = dataset_manager.load()
   ```

3. **`ModelManager`** - Load models and apply DoRA
   ```python
   model_manager.load()  # Load SDXL base
   model_manager.apply_dora()  # Wrap with LoRA
   ```

4. **`TrainingManager`** - Execute training
   ```python
   training_manager.train(num_epochs=10)
   ```

5. **`InferenceManager`** - Generate images
   ```python
   images = inference_manager.generate("ultrasound image")
   ```

---

### 6. **api/app.py** - REST API

**Purpose:** HTTP API for web/service integration

**Structure:**

```python
# Initialize Flask
app = Flask(__name__)

# Initialize project
project = DoRAProject()

# Define endpoints
@app.route('/health', methods=['GET'])
def health():
    return {"status": "healthy"}

@app.route('/config', methods=['GET'])
def get_config():
    return project.config.to_dict()

@app.route('/training/start', methods=['POST'])
def start_training():
    data = request.get_json()
    num_epochs = data.get('num_epochs', 10)
    project.train(num_epochs)
    return {"status": "training started"}

@app.route('/inference/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt')
    images = project.generate(prompt)
    return {"images": images}
```

**How to use:**
```bash
# Start server
python -m api.app --port 5000

# Make requests
curl -X GET http://localhost:5000/config
curl -X POST http://localhost:5000/training/start \
  -H "Content-Type: application/json" \
  -d '{"num_epochs": 10}'
```

---

### 7. **api/cli.py** - Command-Line Interface

**Purpose:** Command-line tool for easy interaction

**Structure:**

```python
import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command')

# Config commands
config_parser = subparsers.add_parser('config')
config_parser.add_argument('action', choices=['show', 'get', 'set', 'save'])

# Training commands
training_parser = subparsers.add_parser('training')
training_parser.add_argument('action', choices=['prepare', 'start'])
training_parser.add_argument('--epochs', type=int, default=10)

# Inference commands
inference_parser = subparsers.add_parser('inference')
inference_parser.add_argument('action', choices=['prepare', 'generate'])
inference_parser.add_argument('--prompt', type=str)
```

**How to use:**
```bash
# Show current configuration
python -m api.cli config show

# Get specific value
python -m api.cli config get --key training.learning_rate

# Start training
python -m api.cli training start --epochs 10

# Generate images
python -m api.cli inference generate --prompt "ultrasound image"
```

---

### 8. **utils/logging.py** - Logging System

**Purpose:** Unified logging to console and W&B

```python
class DoRALogger:
    def __init__(self, use_wandb=True):
        self.use_wandb = use_wandb
    
    def log_metrics(self, step, metrics):
        # Log to console
        print(f"Step {step}: {metrics}")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_images(self, images, captions):
        # Log generated images to W&B
        if self.use_wandb:
            wandb.log({"images": wandb.Image(img) for img in images})
```

---

### 9. **utils/checkpoint.py** - Model Saving

**Purpose:** Save and load model checkpoints

```python
class CheckpointManager:
    def save_checkpoint(self, step, model, optimizer, metrics):
        """Save training checkpoint"""
        checkpoint = {
            'step': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, f"checkpoints/step_{step}.pt")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        return checkpoint['model_state'], checkpoint['optimizer_state']
    
    def save_final_model(self, model):
        """Save final trained model"""
        model.save_pretrained("checkpoints/final_model")
```

---

### 10. **inference.py** - Standalone Inference

**Purpose:** Generate images from trained model

```python
def generate_image(prompt, num_steps=50, guidance_scale=7.5, seed=42):
    """Generate an image from a prompt"""
    
    # Load trained model
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    )
    
    # Load fine-tuned weights
    pipeline.unet = PeftModel.from_pretrained(
        pipeline.unet,
        "checkpoints/final_model"
    )
    
    # Generate
    with torch.no_grad():
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed)
        ).images[0]
    
    return image
```

---

## File Structure

```
/Users/davidgogi/Desktop/Impact/
├── app_ui.py                          # Streamlit UI
├── train_dora_sdxl.py                 # Main training script
├── inference.py                       # Image generation
├── validate_setup.py                  # Setup validation
├── requirements.txt                   # Python dependencies
│
├── config/
│   └── dora_sdxl.yaml                # Configuration (YAML)
│
├── data/
│   └── dataset.py                    # Dataset loading
│
├── utils/
│   ├── logging.py                    # Logging utilities
│   └── checkpoint.py                 # Checkpoint management
│
├── api/
│   ├── core.py                       # High-level API
│   ├── app.py                        # REST API (Flask)
│   └── cli.py                        # CLI interface
│
├── checkpoints/                      # Saved models (generated)
│   ├── checkpoint-xxxx/              # Training checkpoints
│   └── final_model/                  # Final trained model
│
└── outputs/                          # Generated images (generated)
```

---

## How Everything Works Together

### Scenario 1: Training via Streamlit UI

```
1. User opens http://localhost:8501
2. User adjusts sliders on Configuration page
3. User clicks "Start Training" button
4. Browser sends configuration to backend
5. train_dora_sdxl.py starts training
6. Progress updates in Dashboard
7. Logs sent to W&B
8. Checkpoints saved to ./checkpoints/
```

### Scenario 2: Training via CLI

```
1. User runs: python -m api.cli training start --epochs 10
2. CLI parses arguments
3. API creates DoRAProject
4. DatasetManager loads datasets
5. ModelManager loads SDXL + applies DoRA
6. TrainingManager executes training loop
7. CheckpointManager saves progress
8. Done!
```

### Scenario 3: Training via Python API

```python
from api import DoRAProject

project = DoRAProject()
project.train(num_epochs=10)
images = project.generate("ultrasound image")
```

### Scenario 4: Training via REST API

```bash
# Start server
python -m api.app --port 5000

# Call endpoints
curl -X POST http://localhost:5000/training/start \
  -H "Content-Type: application/json" \
  -d '{"num_epochs": 10}'

curl -X POST http://localhost:5000/inference/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ultrasound image"}'
```

### Data Flow During Training

```
Dataset Files (NPZ)
      ↓
   dataset.py (UltrasoundImageDataset)
      ↓
   DataLoader (batches of 4)
      ↓
   train_dora_sdxl.py (training loop)
      ├─ Encode prompt → embeddings
      ├─ Encode image → latent space
      ├─ Add noise
      ├─ Predict noise with UNet
      ├─ Calculate loss
      └─ Update weights
      ↓
   CheckpointManager (save weights)
      ↓
   ./checkpoints/final_model/
```

---

## Key Concepts Explained

### What is DoRA?

Traditional LoRA (Low-Rank Adaptation):
```
W_new = W_original + LoRA_update
```

DoRA decomposition:
```
W_new = m * (d / ||d||) * W_original
   where: m = magnitude (scalar)
          d = direction (vector)
```

Benefits:
- Better expressive power
- Faster convergence
- Minimal overhead (few extra parameters)

### What is BF16?

- **bfloat16** = Brain Float 16
- Uses 16 bits instead of 32
- Maintains range but reduces precision
- H200 has native hardware support → 2x faster
- Good enough for training (not inference)

### What is Gradient Checkpointing?

```
Normal: Store all intermediate values → High memory

Checkpointing: Only store some intermediate values
              → Lower memory
              → Recalculate others during backward pass
              → Slower but saves memory
```

Perfect for H200 with huge memory!

### What is Mixed Precision?

```
Some computations in FP32 (precision)
Some computations in BF16 (speed)
Best of both worlds!
```

---

## Summary

**The complete flow:**

1. **Configuration** (YAML) → defines all settings
2. **Data Loading** (dataset.py) → loads ultrasound images (NO masks!)
3. **Model Loading** (train_dora_sdxl.py) → loads SDXL
4. **DoRA Application** (PEFT) → wraps UNet with trainable adapters
5. **Training Loop** → fine-tunes on ultrasound images
6. **Checkpointing** → saves progress
7. **Inference** → generates new ultrasound images
8. **APIs** (Python/REST/CLI) → interface for everything

**User can interact via:**
- Streamlit UI (visual, interactive)
- CLI (command-line, practical)
- REST API (web/service, scalable)
- Python API (programmatic, flexible)

All three use the same underlying `api/core.py` → guaranteed consistency!
