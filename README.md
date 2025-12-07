# LabTOP Reproduction - CS598 DL4H

This is a reproduction and extension of the LabTOP paper for CS598 Deep Learning for Healthcare at UIUC.

**Authors:** Anish Sao, Kristopher Iotov

**Original Paper:** LabTOP: A Unified Model for Lab Test Outcome Prediction on Electronic Health Records

**Original Repository:** https://github.com/sujeongim/LabTOP

## Project Overview

LabTOP is a unified autoregressive model that predicts lab test outcomes on Electronic Health Records. Unlike traditional methods that classify discrete ranges or estimate limited subsets of tests, LabTOP performs continuous numerical predictions across diverse lab items using generative modeling.

This reproduction extends the original work by:
- Implementing TPU training support via Google Colab
- Creating data slicing utilities for resource-constrained experimentation
- Optimizing memory usage for cloud TPU environments
- Providing complete end-to-end training pipelines

## Quick Start with Google Colab TPU

The fastest way to train LabTOP is using our pre-configured Colab notebook with TPU acceleration.

### Prerequisites

1. Google Colab Pro account with TPU access
2. PhysioNet credentialed account with MIMIC-IV v2.2 access
3. Google Drive with at least 5GB free space

### Setup Steps

1. Open `labtop_colab_tpu.ipynb` in Google Colab
2. Change runtime to TPU v5e-1 or v6e-1: Runtime > Change runtime type > TPU
3. Update your PhysioNet credentials in Step 2
4. Run all cells sequentially

The notebook handles:
- MIMIC-IV data download and storage in Google Drive
- Dataset slicing to 200 ICU stays for manageable training
- Automatic decompression and preprocessing
- TPU-optimized training configuration
- Model evaluation and metrics

### Expected Runtime

- Data download: 30-60 minutes first time only
- Data slicing: 5-10 minutes
- Preprocessing: 10-15 minutes
- Training 5 epochs: 20-30 minutes on TPU
- Evaluation: 5 minutes

Total: Approximately 1.5-2 hours for complete pipeline.

## Local Development Setup

For local experimentation without GPU or TPU access, use the CPU-friendly small-scale mode.

### Installation

```bash
git clone https://github.com/kiotov2/labtop-reproduction.git
cd labtop-reproduction
conda env create -f environment.yml
conda activate labtop
```

### Data Slicing

Create a manageable subset of MIMIC-IV:

```bash
python scripts/slice_mimic.py \
  --source /path/to/mimiciv-2.2 \
  --dest ./data_small \
  --n_stays 200
```

This generates:
- 200 ICU stays with all related events
- Compressed CSV files matching MIMIC-IV format
- Approximately 50-100MB total size versus 50GB for full dataset

Note: Sliced data remains covered by PhysioNet Data Use Agreement. Do not share publicly.

### Configuration

Create configuration files for small-scale training:

```bash
# Create data config
cat > labtop/src/config/data/mimiciv_small.yaml << EOF
defaults:
  - mimiciv

raw_data_path: ./data_small
min_los: 1
debug_table_sample_ratio: 1.0
EOF

# Create training config
cat > labtop/src/config/train/train_small.yaml << EOF
defaults:
  - train_base

epochs: 5
batch_size: 2
gradient_accumulation_steps: 8
use_wandb: false
patience: 1
max_seq_len: 512
lr: 1e-4
EOF
```

### Preprocessing

```bash
cd labtop
python src/scripts/preprocess.py data=mimiciv_small max_seq_len=512
```

### Training

```bash
python src/scripts/train.py \
  data=mimiciv_small \
  train=train_small \
  max_seq_len=512
```

### Evaluation

```bash
python src/scripts/evaluate.py \
  data=mimiciv_small \
  train=train_small \
  max_seq_len=512 \
  +test.model_path=./trained_models/YOUR_MODEL_PATH
```

## Project Structure

```
labtop-reproduction/
├── labtop/                         # Main package directory
│   ├── src/
│   │   ├── config/                # Hydra configuration files
│   │   │   ├── data/              # Dataset configs
│   │   │   └── train/             # Training configs
│   │   ├── core/
│   │   │   ├── models/            # Model architecture and training
│   │   │   └── utils/             # Helper utilities
│   │   └── scripts/
│   │       ├── preprocess.py      # Data preprocessing
│   │       ├── train.py           # Training script
│   │       └── evaluate.py        # Evaluation script
├── scripts/
│   └── slice_mimic.py             # Dataset slicing utility
├── labtop_colab_tpu.ipynb         # Complete Colab TPU notebook
├── TPU_SETUP_GUIDE.md             # Detailed TPU setup instructions
└── environment.yml                # Conda environment specification
```

## TPU Training Details

### Memory Optimization

TPU training uses reduced batch sizes to prevent out-of-memory errors:
- Batch size per core: 2
- Gradient accumulation steps: 8
- Effective batch size: 16
- Maximum sequence length: 512

### Model Checkpointing

The codebase includes TPU-compatible model saving that bypasses Accelerate's `save_state` method:
- Models saved as `pytorch_model.bin`
- Tokenizer and config saved alongside weights
- Automatic best model tracking based on validation loss

### Known Issues

1. Accelerate's `save_state` may fail on TPU environments. The trainer includes fallback manual saving.
2. Small validation sets require adjusted subset sizes. Patched to use `min(len(dataset))`.
3. MIMIC-IV download can be slow. Files persist in Google Drive between sessions.

## Results

Training on 200 ICU stays subset:
- Training time: Approximately 20-30 minutes for 5 epochs on TPU v5e-1
- Validation loss: Results vary based on random seed and data split
- Model size: Approximately 400MB saved checkpoint

Full results and metrics available in the `results/` directory after training.

## Configuration Management

This project uses Hydra for configuration management. Override any parameter via command line:

```bash
python src/scripts/train.py \
  data=mimiciv_small \
  train.batch_size=4 \
  train.epochs=10 \
  max_seq_len=1024
```

Common overrides:
- `data.raw_data_path`: Path to MIMIC-IV data
- `train.batch_size`: Batch size per device
- `train.epochs`: Number of training epochs
- `max_seq_len`: Maximum sequence length
- `train.lr`: Learning rate

## Troubleshooting

### TPU not detected in Colab
- Verify runtime type: Runtime > Change runtime type > TPU v5e-1
- Restart runtime after changing
- Check `PJRT_DEVICE` environment variable shows TPU not CPU

### Missing or corrupted MIMIC-IV files
- Re-download specific files using wget with PhysioNet credentials
- Verify file sizes are non-zero
- Check Google Drive storage quota

### Out of memory during training
- Reduce `train.batch_size` to 1
- Increase `train.gradient_accumulation_steps` proportionally
- Reduce `max_seq_len` from 512 to 256

### Model weights not saving
- Check `trained_models/` directory exists
- Verify trainer.py includes manual save fallback
- Ensure main process completes without interruption

## Citation

```bibtex
@article{im2025labtop,
  title={LabTOP: A Unified Model for Lab Test Outcome Prediction on Electronic Health Records},
  author={Im, Sujeong and Oh, Jungwoo and Choi, Edward},
  journal={arXiv preprint arXiv:2502.14259},
  year={2025}
}
```

## License

This project follows the same license as the original LabTOP repository. MIMIC-IV data usage is governed by the PhysioNet Data Use Agreement.

## Acknowledgments

- Original LabTOP authors: Sujeong Im, Jungwoo Oh, Edward Choi
- CS598 DL4H course staff at UIUC
- PhysioNet for MIMIC-IV dataset access
