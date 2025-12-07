# What to Change in Your Colab Notebook for TPU

## The Short Answer

Replace your current installation cell with this:

### OLD (what you have now):
```python
!pip install -r requirements.txt
!pip install hydra-core --upgrade
```

### NEW (for TPU):
```python
# Install PyTorch XLA for TPU
!pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# Install other dependencies
!pip install accelerate transformers hydra-core omegaconf pandas numpy scipy scikit-learn tqdm datasets tokenizers safetensors huggingface-hub
```

## Commands to Run

### Step 1: Enable TPU in Colab
**Before running ANY cells:**
- Click: **Runtime → Change runtime type → TPU v2**
- Click **Save**

### Step 2: After cloning the repo, verify TPU:
```python
import torch_xla.core.xla_model as xm
print(f"TPU cores: {xm.xrt_world_size()}")  # Should print 8
```

### Step 3: Create TPU-optimized config:
```python
# Add this cell after slicing data
import os
os.makedirs("labtop/src/config/train", exist_ok=True)

with open("labtop/src/config/train/train_small_tpu.yaml", "w") as f:
    f.write("""defaults:
  - train_base

epochs: 2
batch_size: 8        # TPU-optimized
gradient_accumulation_steps: 4
use_wandb: false
patience: 1
max_seq_len: 512
lr: 1e-4
""")
```

### Step 4: Train with TPU config:

**OLD:**
```python
!python src/scripts/train.py data=mimiciv_small train=train_small max_seq_len=512
```

**NEW:**
```python
!python src/scripts/train.py data=mimiciv_small train=train_small_tpu max_seq_len=512
```

## Complete Cell-by-Cell Changes

Here's what cells need to change in your existing notebook:

### Cell: "Install Dependencies"
**Replace:**
```python
!pip install -r requirements.txt
!pip install hydra-core --upgrade
```

**With:**
```python
!pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html
!pip install accelerate transformers hydra-core omegaconf pandas numpy scipy scikit-learn tqdm datasets tokenizers safetensors huggingface-hub
```

### Cell: "Create Configs"
**Replace:**
```python
with open("src/config/train/train_small.yaml", "w") as f:
    f.write("""
defaults:
  - train_base

epochs: 2
batch_size: 2
gradient_accumulation_steps: 8
use_wandb: false
patience: 1
max_seq_len: 512
""")
```

**With:**
```python
with open("labtop/src/config/train/train_small_tpu.yaml", "w") as f:
    f.write("""defaults:
  - train_base

epochs: 2
batch_size: 8        # Optimized for TPU
gradient_accumulation_steps: 4
use_wandb: false
patience: 1
max_seq_len: 512
lr: 1e-4
""")
```

### Cell: "Train"
**Replace:**
```python
!python src/scripts/train.py data=mimiciv_small train=train_small max_seq_len=512
```

**With:**
```python
!python src/scripts/train.py data=mimiciv_small train=train_small_tpu max_seq_len=512
```

### Cell: "Evaluate"
**Replace:**
```python
!python src/scripts/evaluate.py data=mimiciv_small train=train_small max_seq_len=512
```

**With:**
```python
!python src/scripts/evaluate.py data=mimiciv_small train=train_small_tpu max_seq_len=512
```

## What Stays the Same

✅ All data download code - **no changes**
✅ Mounting Google Drive - **no changes**
✅ Cloning repository - **no changes**
✅ Slicing data - **no changes**
✅ Preprocessing - **no changes**

## Why These Changes?

1. **PyTorch XLA**: TPUs use a different PyTorch backend (XLA compiler)
2. **Batch size = 8**: TPUs perform best with batch sizes divisible by 8
3. **train_small_tpu**: A new config file optimized for TPU batch sizing
4. **HuggingFace Accelerate**: Automatically detects and uses TPU (no code changes needed!)

## That's It!

The beauty of using HuggingFace Accelerate is that your actual training code (`trainer.py`) **doesn't need ANY changes**. Accelerate automatically:
- Detects TPU
- Places tensors on TPU
- Handles distributed training across 8 cores
- Manages gradient synchronization

Just install the right PyTorch version and adjust batch sizes! 🚀

