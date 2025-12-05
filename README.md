# LabTOP Reproduction - CS598 DL4H

This is a reproduction of the LabTOP paper for CS598 Deep Learning for Healthcare at UIUC.

**Authors:** Anish Sao, Kristopher Iotov

**Original Paper:** [LabTOP: A Unified Model for Lab Test Outcome Prediction on Electronic Health Records](https://arxiv.org/abs/2502.14259)

**Original Repository:** [https://github.com/sujeongim/LabTOP](https://github.com/sujeongim/LabTOP)

## Overview

LabTOP is a unified model that predicts lab test outcomes by leveraging autoregressive generative modeling on EHR data. Unlike conventional methods that estimate only a subset of lab tests or classify discrete value ranges, LabTOP performs continuous numerical predictions for a diverse range of lab items.

## Results

No results yet.

## Setup

### Full-Scale Setup

See the original repository for complete setup instructions.

### Small-Scale Mode (CPU-Friendly)

For quick local experiments without GPU access, you can create a small subset of MIMIC-IV:

#### Step 1: Slice the Dataset

```bash
python scripts/slice_mimic.py \
  --source /path/to/mimic-iv-3.1 \
  --dest ./data_small \
  --n_stays 200
```

This creates a manageable subset with:
- 200 ICU stays (sufficient for train/val/test splits)
- All related events (labs, medications, procedures, etc.)
- Compressed `.csv.gz` files matching MIMIC-IV format
- ~50-100MB total size (vs. ~50GB for full dataset)

**Important:** The sliced data is still covered by PhysioNet DUA. Do not upload to GitHub or share publicly.

#### Step 2: Update Configuration

Create a config override for small-scale mode:

```yaml
# labtop/src/config/data/mimiciv_small.yaml
defaults:
  - mimiciv

raw_data_path: ./data_small
min_los: 1
```

#### Step 3: Run Preprocessing

```bash
cd labtop/src
python scripts/preprocess.py data=mimiciv_small
```

#### Step 4: Train (CPU-Friendly Settings)

```bash
python scripts/train.py \
  data=mimiciv_small \
  max_seq_len=512 \
  train.epochs=2 \
  train.batch_size=2
```

**Expected Runtime:**
- Slicing: ~5-10 minutes
- Preprocessing: ~10-15 minutes
- Training (2 epochs): ~30-60 minutes on CPU

This allows you to validate the pipeline and get baseline results for your report.

## Citation

```bibtex
@article{im2025labtop,
  title={LabTOP: A Unified Model for Lab Test Outcome Prediction on Electronic Health Records},
  author={Im, Sujeong and Oh, Jungwoo and Choi, Edward},
  journal={arXiv preprint arXiv:2502.14259},
  year={2025}
}
```
