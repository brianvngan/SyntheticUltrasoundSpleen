# Synthetic Ultrasound Spleen — DDPM

A Denoising Diffusion Probabilistic Model (DDPM) that generates synthetic spleen ultrasound images using a U-Net backbone with attention. Trained on preprocessed `.npz` image files and designed to run on a GPU cluster (HPC).

---

## Project Structure

```
SyntheticUltrasoundSpleen/
├── ddpm.ipynb                        # Main training notebook
├── README.md
├── Preprocessed_Roboflow/            # Input dataset (.npz files)
├── ddpm_chimera_checkpoints/         # Saved model checkpoints
└── ddpm_chimera_results/             # Generated images & loss curves
```

---

## Requirements

### Hardware
- NVIDIA GPU (CUDA required)
- Recommended: 16GB+ VRAM for batch size 8 at 256×256

### Software

```bash
pip install torch torchvision tqdm matplotlib numpy
```

Or with conda:

```bash
conda create -n ddpm python=3.10
conda activate ddpm
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install tqdm matplotlib numpy
```

Verify your GPU is available:

```python
import torch
print(torch.cuda.is_available())       # Should print True
print(torch.cuda.get_device_name(0))   # Your GPU name
```

---

## Running on CHIMERA24 (DGX H200)

All IMPACT students have access to **CHIMERA24** (the new DGX H200).

> **Returning user?** If you've already set up on Chimera before, skip the one-time setup and jump to [Per-session workflow](#per-session-workflow).

---

### One-time setup (skip if already done)

The steps below are **per-account**, not per-project. If Miniconda is already installed and you have an `IMPACT` conda env from previous work, skip straight to the per-session workflow.

<details>
<summary><b>1. Install Miniconda</b> — skip if <code>which conda</code> already returns a path</summary>

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Mini*
./Mini*
```
</details>

<details>
<summary><b>2. Create the IMPACT conda env</b> — skip if <code>conda env list</code> already shows <code>IMPACT</code></summary>

```bash
conda create -n IMPACT python=3.10 -y
conda activate IMPACT
pip install jupyterlab ipykernel
```
</details>

<details>
<summary><b>3. One-time LD_LIBRARY_PATH setup</b> — skip if <code>ls $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh</code> already exists</summary>

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Paste:

```bash
export LD_LIBRARY_PATH=$(python - <<'PY'
import site, os, glob
paths=[]
for p in site.getsitepackages():
    paths += glob.glob(os.path.join(p,"nvidia","*","lib"))
print(":".join(paths))
PY
):$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Reload:

```bash
conda deactivate
conda activate IMPACT
```
</details>

### Project dependencies (one-time per project)

Inside the `IMPACT` env, install this project's requirements:

```bash
conda activate IMPACT
pip install torch torchvision tqdm matplotlib numpy
```

---

### Per-session workflow

Do these every time you come back to train.

#### 1. SSH into the headnode

```bash
ssh <your-username>@chimera.umb.edu
```

**DO NOT RUN JOBS ON THE HEADNODE.** Allocate a compute node instead.

#### 2. Allocate an H200

**Quick test / sanity-check (60 min, 1 GPU):**

```bash
salloc -c2 -A impact -q aicore --gres=gpu:1 --mem=32G -w chimera24 -p AICORE_H200 -t 60
```

**Full training run (8 hours, 3 GPUs):**

```bash
salloc -c6 -A impact -q aicore --gres=gpu:3 --mem=96G -w chimera24 -p AICORE_H200 -t 480
```

#### 3. Activate the env and sanity-check the GPU

```bash
conda activate IMPACT
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print `True NVIDIA H200`.

#### 4. Launch the notebook

```bash
cd /path/to/SyntheticUltrasoundSpleen
jupyter lab --no-browser --port=8888
```

Then open `ddpm.ipynb` and run all cells.

---

## Dataset

The model expects grayscale ultrasound images stored as `.npz` files, each containing an `image` key with a 2D NumPy array.

Expected dataset path (edit in the notebook config):
```
/path/to/your/Preprocessed_Roboflow/
```

Each `.npz` file should contain:
```python
{'image': np.ndarray}  # shape: (H, W), dtype: uint8 or float
```

Images are automatically normalized to `[-1, 1]` during loading.

The dataset is split **90% train / 10% validation** automatically.

---

## Configuration

At the top of `ddpm.ipynb`, update these paths and hyperparameters to match your setup:

```python
DATA_PATH    = '/path/to/Preprocessed_Roboflow'   # Input .npz files
OUTPUT_DIR   = '/path/to/ddpm_checkpoints'         # Where to save model weights
RESULTS_DIR  = '/path/to/ddpm_results'             # Where to save generated images

IMAGE_SIZE   = 256     # Input/output resolution
BATCH_SIZE   = 8       # Reduce if you run out of VRAM
LR           = 5e-5    # Learning rate
NUM_EPOCHS   = 60      # Total training epochs
T            = 1000    # Diffusion timesteps
SAVE_EVERY   = 5       # Save checkpoint every N epochs
NUM_GENERATE = 200     # How many synthetic images to generate after training
```

---

## Pipeline

### 1. Prepare Data
Place your preprocessed `.npz` files in the `DATA_PATH` directory. Make sure each file has an `image` key.

### 2. Train the Model
Open `ddpm.ipynb` and run all cells in order. The notebook will:
- Load and split the dataset
- Build a U-Net with sinusoidal time embeddings and multi-head attention
- Train using the DDPM noise-prediction objective (MSE loss)
- Save checkpoints every `SAVE_EVERY` epochs to `OUTPUT_DIR`
- Save the best model (lowest val loss) as `best.pt`
- Generate sample images every 10 epochs for visual progress checks

### 3. Monitor Training
Loss is printed each epoch:
```
Epoch   1 | train=0.1234 | val=0.1456
Epoch   2 | train=0.1102 | val=0.1311
...
```
A loss curve is saved to `RESULTS_DIR/loss_curve.png` after training.

### 4. Generate Synthetic Images
After training, the notebook automatically generates `NUM_GENERATE` synthetic images using the best checkpoint. Outputs are saved as:
- `ddpm_synthetic.npz` — all generated images stacked into a single compressed array
- `real_vs_synthetic.png` — side-by-side comparison grid of real vs generated images

---

## Model Architecture

| Component | Details |
|-----------|---------|
| Backbone | U-Net with skip connections |
| Base channels | 128 |
| Time embedding | Sinusoidal positional encoding → MLP |
| Attention | Multi-head self-attention (4 heads) at bottleneck |
| Normalization | GroupNorm throughout |
| Activation | SiLU (Swish) |
| Parameters | ~100M |

---

## Outputs

| File | Description |
|------|-------------|
| `epoch-XXX.pt` | Periodic checkpoints (model + optimizer state) |
| `best.pt` | Best model weights by validation loss |
| `samples_epochXXX.png` | 4 generated samples at that epoch |
| `loss_curve.png` | Train/val MSE loss over epochs |
| `ddpm_synthetic.npz` | Final batch of generated images |
| `real_vs_synthetic.png` | Comparison grid: real vs synthetic |

---

## Loading a Checkpoint

```python
import torch
from pathlib import Path

checkpoint = torch.load('ddpm_chimera_checkpoints/best.pt')
model.load_state_dict(checkpoint)
model.eval()
```

Or to resume training from a full checkpoint:

```python
ckpt = torch.load('ddpm_chimera_checkpoints/epoch-060.pt')
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
start_epoch = ckpt['epoch'] + 1
```

---

## Notes

- Training on 60 epochs at 256×256 with batch size 8 takes several hours on a modern GPU
- Mixed precision (`torch.amp`) is used automatically for faster training and lower memory usage
- Gradient clipping (max norm = 1.0) is applied for training stability
- If you run out of VRAM, reduce `BATCH_SIZE` or `base_ch` in the UNet constructor
