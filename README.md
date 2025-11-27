# ADetective: Oligodendrocyte AD Pathology Classifier

Binary classification of Alzheimer's Disease neuropathology (High vs Not AD) using single-cell RNA-seq data from oligodendrocytes in the SEA-AD dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Full Workflow](#full-workflow)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Details](#model-details)
- [Google Colab Guide](#google-colab-guide)
  - [Accelerate Integration](#accelerate-integration-colab-support)

## Overview

This project implements three complementary deep learning approaches for classifying AD pathology in oligodendrocytes:

1. **MLP Baseline** (Phase 3): Simple multi-layer perceptron with batch normalization and dropout
2. **Custom Transformer** (Phase 4): Gene-as-sequence transformer with Flash Attention support
3. **scGPT Fine-tuning** (Phase 5): Leveraging a pretrained foundation model with 33M cell pre-training

## Dataset

- **Source**: SEAAD (Single-nucleus Enrichment and Sequencing of Alzheimer Disease)
- **File**: `SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad` (35GB)
- **Format**: AnnData H5AD (sparse matrix CSR format)
- **Cell Type**: Oligodendrocytes only (~80K cells after filtering)
- **Labels**: Binary (High AD vs Not AD)
  - Label 1: High AD neuropathology
  - Label 0: Not AD pathology
  - Excluded: Low and Intermediate

## Quick Start

### Google Colab (Recommended)

**1. Clone repo in Colab cell:**
```python
!git clone https://github.com/[username]/ADetective.git
%cd ADetective
```

**2. Install dependencies:**
```python
!pip install -q -r requirements-colab.txt
```

**3. Run all training in sequence:**

Data is assumed to be at `/content/SEAAD_A9_RNAseq_DREAM.Cleaned.h5ad` in Colab.

```bash
# Train MLP (5-10 minutes on T4 GPU)
!python scripts/train_mlp.py \
    --data-dir ./results/processed \
    --output-dir ./results/mlp \
    --batch-size 32 --learning-rate 1e-3 --epochs 30

# Train Transformer (10-15 minutes)
!python scripts/train_transformer.py \
    --data-dir ./results/processed \
    --output-dir ./results/transformer \
    --batch-size 32 --learning-rate 1e-4 --epochs 30

# Fine-tune scGPT (15-20 minutes, requires 16GB VRAM)
!python scripts/train_scgpt.py \
    --data-dir ./results/processed \
    --output-dir ./results/scgpt \
    --batch-size 16 --learning-rate 1e-5 --epochs 15

# Compare all models
!python scripts/compare_models.py \
    --results-dir ./results --output-dir ./results/comparison
```

**4. Load and preprocess data first (one-time setup):**
```bash
!python scripts/load.py \
    --data-path /content/SEAAD_A9_RNAseq_DREAM.Cleaned.h5ad \
    --output-dir ./results \
    --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 --n-hvgs 2000
```

**5. Save results to Drive:**
```bash
!cp -r ./results /content/drive/MyDrive/ADetective_Results
```

### Local Development

```bash
# Clone repository
git clone https://github.com/[username]/ADetective.git
cd ADetective

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy cleaned dataset
cp /path/to/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad ./data/

# Run training commands (see Full Workflow section below)
```

## Full Workflow

### Phase 1: Data Preparation

The dataset must be cleaned and preprocessed before training. This typically involves:

**Input**: `SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad` (35GB, raw)

**Processing**:
```bash
python scripts/prepare_data.py \
    --input-path /path/to/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad \
    --output-path ./data/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad \
    --cell-type Oligodendrocyte
```

**What happens**:
1. Load 35GB H5AD file with 1.3M cells across all types
2. Filter to oligodendrocytes (~80K cells)
3. Map ADNC status to binary labels (High=1, Not AD=0)
4. Exclude Low and Intermediate pathology
5. Select 2000 highly variable genes (HVGs)
6. Standardize expression (z-score by training set)
7. Create train/val/test splits (70%/10%/20%) at **donor level** to prevent data leakage
8. Save processed file: `SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad` (~500MB)

**Output**: `./data/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad` (cleaned, ready for training)

#### Data Split Details
- **Train**: 70% of unique donors → ~56K cells
- **Validation**: 10% of unique donors → ~8K cells
- **Test**: 20% of unique donors → ~16K cells
- **Stratification**: Balanced by ADNC label (High vs Not AD)

### Phase 2: Model Training

#### Step 1: Train MLP Baseline

**Command:**
```bash
python scripts/train_mlp.py \
    --data-dir ./results/processed \
    --output-dir ./results/mlp \
    --batch-size 32 --learning-rate 1e-3 --epochs 30 \
    --hidden-dims 512 256 128 --dropout-rate 0.3
```

**Architecture**:
- Input: 2000 genes
- Hidden: 512 → 256 → 128 → 1
- Batch Norm + ReLU + Dropout(0.3)
- ~1.3M parameters

**Training**:
- Optimizer: AdamW (lr=1e-3)
- Loss: BCEWithLogitsLoss
- Epochs: 30 (with early stopping, patience=10)
- Batch size: 32-64
- Duration: 5-10 min on T4 GPU

**Output**:
- `./results/mlp/checkpoint.pt` - trained weights
- `./results/mlp/results.json` - metrics (accuracy, F1, ROC-AUC)
- `./results/mlp/training_history.json` - per-epoch metrics

**Expected Performance**: ~78% accuracy, ~0.75 F1

#### Step 2: Train Custom Transformer

**Command:**
```bash
python scripts/train_transformer.py \
    --data-dir ./results/processed \
    --output-dir ./results/transformer \
    --batch-size 32 --learning-rate 1e-4 --epochs 30 \
    --d-model 128 --nhead 8 --num-layers 3 --dim-feedforward 256
```

**Novel Architecture**:
- Treats genes as sequence tokens
- Expression values scale token embeddings (multiplicative)
- CLS token for cell-level classification
- 3 transformer layers, 8 heads, d_model=128
- ~300K parameters
- Flash Attention via PyTorch 2.0 (transparent, no API changes)

**Training**:
- Optimizer: AdamW (lr=1e-4, lower for stability)
- Loss: BCEWithLogitsLoss
- Epochs: 30 (early stopping, patience=10)
- Batch size: 32
- Duration: 10-15 min on T4 GPU

**Output**:
- `./results/transformer/checkpoint.pt` - trained weights
- `./results/transformer/results.json` - metrics
- `./results/transformer/training_history.json` - per-epoch metrics
- `./results/transformer/attention_weights.npy` (optional) - for interpretation

**Expected Performance**: ~80% accuracy, ~0.78 F1 (modest improvement over MLP if gene interactions matter)

#### Step 3: Fine-tune scGPT (Optional, requires more VRAM)

**Command:**
```bash
python scripts/train_scgpt.py \
    --data-dir ./results/processed \
    --output-dir ./results/scgpt \
    --batch-size 16 --learning-rate 1e-5 --epochs 15 \
    --d-model 512 --nhead 8 --num-layers 12 --freeze-layers 6
```

**Foundation Model**:
- Pre-trained on 33M cells across diverse tissues
- 12 transformer layers, d_model=512, 8 heads
- ~120M parameters (only top 6 layers trainable)
- Expression values tokenized into 51 bins

**Fine-tuning Strategy**:
- Freeze bottom 6 layers (preserve general knowledge)
- Train top 6 layers + classification head
- Gene vocabulary alignment (handles out-of-vocabulary)

**Training**:
- Optimizer: AdamW (lr=1e-5, very low for pretrained)
- Loss: BCEWithLogitsLoss
- Epochs: 15 (early stopping, patience=3)
- Batch size: 16 (smaller due to model size)
- Duration: 15-20 min on T4 GPU

**Output**:
- `./results/scgpt/checkpoint.pt` - fine-tuned weights
- `./results/scgpt/results.json` - metrics
- `./results/scgpt/training_history.json` - per-epoch metrics

**Expected Performance**: ~82% accuracy, ~0.80 F1 (best if knowledge transfer helps)

#### Step 4: Compare All Models

**Command:**
```bash
python scripts/compare_models.py \
    --results-dir ./results \
    --output-dir ./results/comparison
```

**Creates**:
- `./results/comparison/comparison_table.csv` - side-by-side metrics
- `./results/comparison/comparison_plot.png` - visualization
- Accuracy, F1, ROC-AUC, Precision, Recall comparison

**Example Output**:
```
Model         | Accuracy | F1 Score | ROC-AUC
MLP           | 78.5%    | 0.752    | 0.820
Transformer   | 80.2%    | 0.781    | 0.842
scGPT         | 82.1%    | 0.805    | 0.862
```

## Project Structure

```
ADetective/
├── src/                          # Source code
│   ├── data/
│   │   ├── dataset.py           # Lazy-loading PyTorch Dataset
│   │   └── loaders.py           # DataLoader utilities
│   ├── models/
│   │   ├── mlp.py               # MLP classifier + trainer
│   │   ├── transformer.py       # Transformer with Flash Attention
│   │   └── scgpt_wrapper.py     # scGPT fine-tuning wrapper
│   ├── training/
│   │   └── evaluator.py         # Training evaluation logic
│   ├── eval/
│   │   └── metrics.py           # Classification metrics & visualization
│   └── utils/
│       └── config.py            # Configuration loading
├── scripts/
│   ├── prepare_data.py          # Data preprocessing pipeline
│   ├── train_mlp.py             # MLP training script
│   ├── train_transformer.py     # Transformer training script
│   ├── train_scgpt.py           # scGPT fine-tuning script
│   ├── compare_models.py        # Cross-model comparison
│   └── load.py                  # Data loading utility
├── configs/
│   ├── data_config.yaml         # Dataset column mapping
│   ├── mlp_config.yaml          # MLP hyperparameters
│   ├── transformer_config.yaml  # Transformer hyperparameters
│   └── scgpt_config.yaml        # scGPT fine-tuning config
├── notebooks/                    # Jupyter notebooks (development)
├── plan/                         # Project planning documents
├── results/                      # Output directory (git-ignored)
│   ├── mlp/                     # MLP results
│   ├── transformer/             # Transformer results
│   ├── scgpt/                   # scGPT results
│   ├── comparison/              # Cross-model comparison
│   └── processed/               # Processed data splits
├── requirements.txt             # Base dependencies
├── requirements_scgpt.txt       # scGPT-specific dependencies
├── requirements-colab.txt       # Google Colab optimized
├── setup.py                     # Package configuration
├── README.md                    # This file
└── .gitignore
```

## Data Processing

### Data Filtering & Labels

**Input**: `SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad` (1.3M cells, 36.6K genes, 35GB)

**Output**: `SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad` (80K oligodendrocytes, 2K HVGs, 500MB)

**Processing Pipeline**:
1. **Cell Type Filtering**: Keep only oligodendrocytes (column: `Subclass == "Oligodendrocyte"`)
2. **Label Creation**: Map ADNC to binary labels
   - High → 1 (positive class)
   - Not AD → 0 (negative class)
   - Exclude: Low, Intermediate
3. **Gene Selection**: 2000 highly variable genes (HVGs)
4. **Normalization**: Z-score standardization using training set statistics
5. **Train/Val/Test Split**: Donor-level stratified split
   - Train: 70% of donors → ~56K cells
   - Validation: 10% of donors → ~8K cells
   - Test: 20% of donors → ~16K cells

**Why Donor-Level Split?**
- Prevents data leakage (cells from same donor must stay together)
- Ensures fair evaluation on unseen donors
- Reflects real-world deployment scenario

### Configuration Details

Edit `configs/data_config.yaml` to customize:
```yaml
cell_type_column: "Subclass"
cell_type_value: "Oligodendrocyte"
donor_id_column: "Donor ID"
adnc_column: "ADNC"
n_hvgs: 2000  # Highly variable genes to select
train_split: 0.7
val_split: 0.1
test_split: 0.2
random_seed: 42
```

## Model Details

### MLP Baseline

**Purpose**: Simple baseline for comparison

**Architecture**:
```
Input (2000) → BN → 512 → ReLU → Dropout(0.3)
            → BN → 256 → ReLU → Dropout(0.3)
            → BN → 128 → ReLU → Dropout(0.3)
            → Linear(1) → Sigmoid
```

**Hyperparameters**:
- Batch size: 32-64
- Learning rate: 1e-3
- Optimizer: AdamW
- Loss: BCEWithLogitsLoss
- Early stopping: patience=10

**File**: `src/models/mlp.py:338`

**Training time**: ~5-10 min (T4 GPU)

### Custom Transformer

**Purpose**: Capture gene-gene interactions and gene importance

**Novel Design**:
- Treats each gene as a sequence token
- Expression value scales token embedding (multiplicative scaling)
- CLS token aggregates cell-wide information via attention
- Positional encoding enables gene order awareness

**Architecture**:
```
Expression vector [2000]
    ↓
Gene embeddings × scaled by expression [2000, 128]
    ↓
[CLS] token prepended [2001, 128]
    ↓
3 × Transformer layers (8 heads, d_model=128, ff_dim=256)
    ↓
CLS output [128]
    ↓
Classification head: LayerNorm → Linear(64) → GELU → Linear(1)
```

**Key Features**:
- Flash Attention via PyTorch 2.0 (automatic, transparent)
- Pre-norm architecture for stability
- Gradient checkpointing to save memory
- Attention weight visualization for interpretability

**Hyperparameters**:
- Batch size: 32
- Learning rate: 1e-4
- Optimizer: AdamW
- Loss: BCEWithLogitsLoss
- Early stopping: patience=10

**File**: `src/models/transformer.py:521`

**Training time**: ~10-15 min (T4 GPU)

### scGPT Fine-tuning

**Purpose**: Leverage large-scale pre-training on 33M cells

**Foundation Model**:
- Pre-trained on diverse tissues (brain, immune, gut, etc.)
- 12 transformer layers, 512 dimensions, 8 heads
- ~120M parameters
- Masked gene expression prediction objective

**Fine-tuning Strategy**:
- Freeze layers 0-5 (preserve general knowledge)
- Train layers 6-11 + classification head
- Gene vocabulary alignment (91-95% overlap typically)
- Expression binned into 51 discrete tokens

**Hyperparameters**:
- Batch size: 16 (smaller due to model size)
- Learning rate: 1e-5 (very low for pretrained)
- Optimizer: AdamW
- Loss: BCEWithLogitsLoss
- Early stopping: patience=3 (early convergence expected)

**File**: `src/models/scgpt_wrapper.py:100`

**Training time**: ~15-20 min (T4 GPU, 16GB VRAM)

## Google Colab Guide

### Step-by-Step Setup

**Cell 1: Clone and Install**
```python
!git clone https://github.com/[username]/ADetective.git
%cd ADetective
!pip install -q -r requirements-colab.txt
```

**Cell 2: Mount Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# List files to verify
!ls /content/drive/MyDrive/ | grep -i seaad
```

**Cell 3: Setup Directories**
```bash
!mkdir -p ./data ./results
!cp /content/drive/MyDrive/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad ./data/
!ls -lh ./data/
```

**Cell 4: Train All Models** (can be run as separate cells)

All training scripts now support **Hugging Face Accelerate** by default. See the [Accelerate Integration](#accelerate-integration-colab-support) section below.

```bash
# MLP - Quick baseline (uses Accelerate automatically)
python scripts/train_mlp.py \
    --data-dir ./results/processed \
    --output-dir ./results/mlp \
    --batch-size 32 --learning-rate 1e-3 --epochs 30

# Or with explicit accelerate launch for single GPU
accelerate launch scripts/train_mlp.py \
    --data-dir ./results/processed \
    --output-dir ./results/mlp \
    --batch-size 32 --learning-rate 1e-3 --epochs 30
```

**Cell 5: Save Results**
```bash
!cp -r ./results /content/drive/MyDrive/ADetective_Results_$(date +%Y%m%d)
```

### Accelerate Integration (Colab Support)

All three training scripts have been integrated with **Hugging Face Accelerate** library for distributed training support.

#### What is Accelerate?

Accelerate is a library that:
- Handles device placement automatically (GPU/CPU/multi-GPU)
- Supports mixed precision training (fp16/bf16)
- Simplifies distributed training code
- Works seamlessly with Colab, local GPUs, and multi-GPU setups

#### Running on Colab (Single GPU)

**Option 1: Default (Recommended)**
```bash
python scripts/train_mlp.py \
    --data-dir ./results/processed \
    --output-dir ./results/mlp \
    --batch-size 32 --learning-rate 1e-3 --epochs 30
```
- Uses Accelerate automatically with default settings
- Accelerate auto-detects 1 GPU and configures optimally

**Option 2: With Explicit Accelerate Launch**
```bash
accelerate launch scripts/train_mlp.py \
    --data-dir ./results/processed \
    --output-dir ./results/mlp \
    --batch-size 32 --learning-rate 1e-3 --epochs 30
```
- More explicit control
- Better for debugging

**Option 3: Disable Accelerate (Fall Back to Single GPU)**
```bash
python scripts/train_mlp.py \
    --data-dir ./results/processed \
    --output-dir ./results/mlp \
    --batch-size 32 --learning-rate 1e-3 --epochs 30 \
    --no-accelerate
```

#### Command-Line Flags

All training scripts support Accelerate control:

```bash
# Enable Accelerate (default)
--use-accelerate          # Explicitly enable (default is True)

# Disable Accelerate
--no-accelerate           # Use single GPU without Accelerate wrapper
```

#### Training Scripts with Accelerate Support

**1. MLP Training (train_mlp.py)**
```bash
python scripts/train_mlp.py \
    --data-dir ./results/processed \
    --output-dir ./results/mlp \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --epochs 30 \
    --hidden-dims 512 256 128 \
    --dropout-rate 0.3 \
    --gradient-clip 1.0 \
    --patience 10
```

**2. Transformer Training (train_transformer.py)**
```bash
python scripts/train_transformer.py \
    --data-dir ./results/processed \
    --output-dir ./results/transformer \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --epochs 30 \
    --d-model 128 \
    --nhead 8 \
    --num-layers 3 \
    --dim-feedforward 256 \
    --dropout 0.1 \
    --gradient-clip 1.0 \
    --patience 10
```

**3. scGPT Fine-tuning (train_scgpt.py)**
```bash
python scripts/train_scgpt.py \
    --data-dir ./results/processed \
    --output-dir ./results/scgpt \
    --batch-size 16 \
    --learning-rate 1e-5 \
    --epochs 15 \
    --n-bins 51 \
    --d-model 512 \
    --nhead 8 \
    --num-layers 12 \
    --freeze-layers 6 \
    --warmup-steps 500 \
    --patience 3
```

#### Accelerate Features Used

- **Mixed Precision Training**: Automatic fp16/bf16 support
- **Gradient Clipping**: Safer training with `accelerator.clip_grad_norm_()`
- **Backward Pass**: Unified `accelerator.backward()` method
- **Device Management**: Automatic device placement (no manual `.to(device)` needed in dataloader)
- **Model Wrapping**: Proper model unwrapping for checkpointing and inference

#### Example: Complete Colab Workflow with Accelerate

```python
# Cell 1: Setup
!git clone https://github.com/[username]/ADetective.git
%cd ADetective
!pip install -q -r requirements-colab.txt

# Cell 2: Mount and copy data
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p ./results/processed
!cp /content/drive/MyDrive/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad ./data/

# Cell 3: Preprocess (one-time)
!python scripts/prepare_data.py \
    --input-path ./data/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad \
    --output-path ./results/processed \
    --cell-type Oligodendrocyte

# Cell 4: Train MLP with Accelerate
!python scripts/train_mlp.py \
    --data-dir ./results/processed \
    --output-dir ./results/mlp \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --epochs 30

# Cell 5: Train Transformer with Accelerate
!python scripts/train_transformer.py \
    --data-dir ./results/processed \
    --output-dir ./results/transformer \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --epochs 30

# Cell 6: Fine-tune scGPT with Accelerate
!python scripts/train_scgpt.py \
    --data-dir ./results/processed \
    --output-dir ./results/scgpt \
    --batch-size 16 \
    --learning-rate 1e-5 \
    --epochs 15

# Cell 7: Save to Drive
!cp -r ./results /content/drive/MyDrive/ADetective_Results_$(date +%Y%m%d)
```

#### Accelerate Configuration

For advanced users, Accelerate can be configured via:

1. **Environment Variables**:
   ```bash
   export ACCELERATE_GPU_PER_DEVICE=1
   export ACCELERATE_MIXED_PRECISION=bf16
   python scripts/train_mlp.py --data-dir ...
   ```

2. **Accelerate Config File** (optional):
   ```bash
   accelerate config
   # Answers questions about your setup
   # Creates ~/.huggingface/accelerate/default_config.yaml
   ```

3. **Command-line Flags**:
   ```bash
   accelerate launch \
       --num_processes 1 \
       --mixed_precision bf16 \
       scripts/train_mlp.py --data-dir ...
   ```

#### Troubleshooting Accelerate Issues

**Issue: "RuntimeError: Expected all tensors to be on the same device"**
- Solution: Ensure you're not manually moving tensors with `.to(device)` when using Accelerate
- Accelerate handles device placement automatically
- If needed, disable: `python scripts/train_mlp.py --no-accelerate`

**Issue: "CUDA out of memory"**
- Accelerate uses same memory as non-distributed training
- Reduce batch size: `--batch-size 16` instead of 32
- Or disable mixed precision if needed

**Issue: Slower training than expected**
- Accelerate adds minimal overhead (~2-3%)
- On single GPU, performance should be equivalent
- If significantly slower, check if GPU is actually being used: `!nvidia-smi`

### Troubleshooting

**Issue: GPU Memory Error**
```python
# Reduce batch size in training script, or:
import torch
torch.cuda.empty_cache()
```

**Issue: H5AD Load Failure (Compression Filter)**
```python
# If original file has compression issues, use this workaround:
import anndata as ad
import scanpy as sc

# Method 1: Try direct load first
try:
    adata = ad.read_h5ad('./data/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad')
except:
    # Method 2: Install hdf5plugin (for Colab)
    !pip install -q hdf5plugin
    import hdf5plugin  # Register plugins
    adata = ad.read_h5ad('./data/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad')
```

**Issue: scGPT Training OutOfMemory**
```bash
# Use smaller batch size
# Edit training script or modify accelerate config:
!accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 2 \
    scripts/train_scgpt.py --data-path ./data/... --cell-type Oligodendrocyte
```

**Issue: Session Timeout**
- Colab sessions can timeout after 12 hours
- Results are automatically saved to `./results`
- Copy results to Drive frequently: `!cp -r ./results /content/drive/MyDrive/`

### Performance Tips

1. **Use Colab GPU T4 or V100** (not TPU for this task)
   - Go to Runtime > Change runtime type > GPU

2. **Monitor GPU Memory**
   ```python
   !nvidia-smi
   ```

3. **Optimize Data I/O**
   ```bash
   # Keep data in Colab local storage (faster than Drive)
   !cp /content/drive/MyDrive/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad /content/data/
   ```

4. **Enable Mixed Precision** (already configured)
   - Saves memory and speeds up training

5. **Check GPU Compute Capability**
   ```python
   import torch
   print(torch.cuda.get_device_capability(0))
   # Should be (7, 5) or higher for Flash Attention
   ```

## Results Interpretation

### Output Files

After training, check:
```
./results/
├── mlp/checkpoint.pt           # Model weights
├── mlp/results.json            # Final metrics
├── mlp/training_history.json   # Per-epoch logs
├── transformer/checkpoint.pt
├── transformer/results.json
├── scgpt/checkpoint.pt
└── comparison/
    ├── comparison_table.csv    # Side-by-side comparison
    └── comparison_plot.png     # Visualization
```

### Metrics Explained

- **Accuracy**: Overall correctness (% predictions right)
- **F1 Score**: Harmonic mean of precision and recall (0-1 scale)
- **ROC-AUC**: Area under receiver-operating characteristic curve (0-1 scale, 0.5=random)
- **Precision**: % of positive predictions that are correct (catch real AD cases)
- **Recall**: % of actual AD cases that model identifies (minimize false negatives)

## Hardware Requirements

### Minimum
- GPU: NVIDIA T4 (15GB VRAM)
- CPU: 4 cores, 16GB RAM
- Storage: 50GB (for data + models)

### Recommended
- GPU: NVIDIA V100+ (32GB VRAM)
- CPU: 8+ cores, 32GB RAM
- Storage: 100GB (for checkpoints and experiments)

### Google Colab
- Free tier: T4 GPU, 12.7GB VRAM (sufficient for all models)
- Pro tier: V100 GPU, 16GB VRAM (recommended for scGPT)

## Dependencies

**Core Stack**:
- PyTorch 2.0+ (includes Flash Attention support)
- Hugging Face Accelerate (distributed training)
- AnnData + Scanpy (single-cell workflows)
- scikit-learn (metrics)

**Optional**:
- scGPT (for Phase 5)
- hdf5plugin (for H5AD compression workarounds)

See `requirements.txt` for full list.

## Key Features

- ✅ Donor-level data splitting (prevents leakage)
- ✅ Three complementary model architectures
- ✅ Mixed precision training (bf16)
- ✅ Flash Attention optimization (PyTorch 2.0)
- ✅ Hugging Face Accelerate integration (distributed training)
- ✅ Comprehensive metrics and visualization
- ✅ Reproducible with fixed seeds
- ✅ Google Colab compatible (single & multi-GPU)
- ✅ Modular, extensible design

## Citation

If you use this code, please cite:
```
@software{adetective2024,
  title={ADetective: Oligodendrocyte AD Pathology Classifier},
  author={Duc Hong},
  year={2024},
  url={https://github.com/[username]/ADetective}
}
```

## License

MIT License - see LICENSE file

## References

- **SEAAD Dataset**: [Allen Institute](https://www.allencelltypes.org/seaad/)
- **scGPT**: Wang et al., "scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics"
- **Flash Attention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- **Accelerate**: Hugging Face [Accelerate Library](https://huggingface.co/docs/accelerate)

## Support

For issues and questions:
1. Check [GitHub Issues](https://github.com/[username]/ADetective/issues)
2. Review [Google Colab Guide](#google-colab-guide) section
3. Check individual training script help: `python scripts/train_mlp.py --help`
