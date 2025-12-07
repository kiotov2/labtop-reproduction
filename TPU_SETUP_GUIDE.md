# Google Colab TPU Setup Guide for LabTOP

## Quick Start

1. **Open the notebook**: Upload `labtop_colab_tpu.ipynb` to Google Colab
2. **Enable TPU**: Runtime → Change runtime type → **TPU v2**
3. **Run cells sequentially** from top to bottom

## What's Different for TPU?

### 1. **PyTorch XLA Installation**
TPUs require PyTorch XLA instead of regular PyTorch:
```bash
pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

### 2. **Batch Size Optimization**
TPUs work best with batch sizes divisible by 8 (per core):
- **Per-core batch size**: 8
- **Number of TPU cores**: 8 (TPU v2)
- **Total batch size**: 8 × 8 = 64
- **With gradient accumulation (4 steps)**: 64 × 4 = **256 effective batch size**

### 3. **Accelerate Auto-Detection**
The good news: **HuggingFace Accelerate automatically detects TPU!**
- No code changes needed in `trainer.py`
- Accelerate handles device placement
- Just install PyTorch XLA and it works

## Key Changes from Your Original Notebook

### Before (CPU/GPU):
```python
!pip install -r requirements.txt
!python src/scripts/train.py data=mimiciv_small train=train_small
```

### After (TPU):
```python
# Install TPU-specific PyTorch
!pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html
!pip install accelerate transformers hydra-core omegaconf ...

# Verify TPU
import torch_xla.core.xla_model as xm
print(f"TPU cores: {xm.xrt_world_size()}")

# Train (same command, but uses train_small_tpu config)
!python src/scripts/train.py data=mimiciv_small train=train_small_tpu
```

## Configuration Files

### TPU-Optimized Training Config
Created at: `labtop/src/config/train/train_small_tpu.yaml`

```yaml
defaults:
  - train_base

epochs: 2
batch_size: 8        # Per-core (optimal for TPU)
gradient_accumulation_steps: 4
use_wandb: false
patience: 1
max_seq_len: 512
lr: 1e-4
```

## Performance Tips

### 1. **Batch Size Guidelines**
- Always use multiples of 8 per core
- TPU v2 has 8 cores
- Good batch sizes: 8, 16, 32, 64 per core

### 2. **Data Loading**
- Use `num_workers > 0` in DataLoader (already configured)
- Avoid CPU bottlenecks

### 3. **Monitoring TPU Utilization**
```python
import torch_xla.debug.metrics as met
print(met.metrics_report())
```

## Troubleshooting

### Issue: "TPU not detected"
**Solution**: Make sure Runtime is set to TPU v2
- Runtime → Change runtime type → TPU

### Issue: "Out of memory"
**Solution**: Reduce batch size
```yaml
batch_size: 4  # Instead of 8
```

### Issue: "Slow training"
**Possible causes**:
1. Batch size too small (use 8+ per core)
2. Too many data transfer operations (minimize CPU↔TPU transfers)
3. Check TPU utilization with metrics

### Issue: "Import error: torch_xla"
**Solution**: Reinstall PyTorch XLA
```bash
!pip uninstall torch torch_xla -y
!pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

## What You DON'T Need to Change

✅ **Trainer code** - Accelerate handles TPU automatically
✅ **Model architecture** - Works as-is
✅ **Data preprocessing** - No changes needed
✅ **Evaluation code** - Works as-is

## Expected Speedup

Compared to CPU:
- **~50-100x faster** for training
- TPU v2 is optimized for matrix operations (transformers are perfect)

Compared to GPU (T4):
- **~5-10x faster** depending on batch size

## Cost Considerations

- **Colab Free**: Limited TPU hours per week
- **Colab Pro**: More TPU hours
- **Colab Pro+**: Even more TPU access

## Next Steps After Training

The trained model will be saved to:
```
/content/labtop-reproduction/labtop/trained_models/
```

To save to Google Drive:
```python
!cp -r trained_models /content/drive/MyDrive/labtop_models/
```

## References

- [PyTorch XLA Documentation](https://pytorch.org/xla/)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate)
- [Google Cloud TPU](https://cloud.google.com/tpu)

