# MLP Baseline Training Guide

This guide explains how to train the MLP (Multi-Layer Perceptron) baseline model for oligodendrocyte AD pathology classification.

## Overview

The MLP baseline is a simple fully-connected neural network that serves as a baseline for performance comparison with more complex models (Transformer, scGPT, etc.).

**Architecture**:
- Input: 2000 highly variable genes
- Hidden layers: 512 → 256 → 128 neurons
- Output: 2 classes (Not AD vs High AD)
- Regularization: Batch Normalization + Dropout
- Training: Mixed precision (BF16) with gradient clipping

**Expected Performance**:
- Target accuracy: >85%
- Target F1: >0.80
- Target ROC-AUC: >0.85

## Prerequisites

1. **Data preprocessing** must be completed first:
   ```bash
   python scripts/prepare_data.py
   ```
   This creates:
   - `results/processed_data/train_data.h5ad`
   - `results/processed_data/val_data.h5ad`
   - `results/processed_data/test_data.h5ad`

2. **Dependencies** must be installed:
   ```bash
   pip install -r requirements.txt
   ```

## Training

### Quick Start (Default Configuration)

```bash
python scripts/training/train_mlp.py
```

This runs training with default configuration:
- Batch size: 32
- Learning rate: 1e-3
- Epochs: 100 (with early stopping after 15 epochs without improvement)
- Mixed precision: BF16
- Output: `results/mlp_baseline/`

### Custom Configuration

```bash
python scripts/training/train_mlp.py \
  --data-dir results/processed_data \
  --config configs/mlp_config.yaml \
  --output-dir results/mlp_baseline_custom
```

### Configuration File

Edit `configs/mlp_config.yaml` to customize:

**Architecture**:
- `input_dim`: Number of input genes (default: 2000)
- `hidden_dims`: Hidden layer sizes (default: [512, 256, 128])
- `dropout_rate`: Dropout probability (default: 0.3)
- `batch_norm`: Use batch normalization (default: true)

**Training**:
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Initial LR (default: 1e-3)
- `epochs`: Max epochs (default: 100)
- `early_stopping.patience`: Stop if no improvement for N epochs (default: 15)
- `mixed_precision`: "bf16" or "fp16" (default: "bf16")

**Example - Larger Model**:
```yaml
architecture:
  hidden_dims: [1024, 512, 256]  # Larger
  dropout_rate: 0.4

training:
  batch_size: 16  # Smaller for more GPU memory
  learning_rate: 5.0e-4
```

## Output Structure

After training, results are saved to `results/mlp_baseline/`:

```
results/mlp_baseline/
├── model.pt                    # Trained model weights
├── config.yaml                 # Configuration used
├── training_history.json       # Loss/accuracy curves
├── metrics.json                # Final metrics for train/val/test
└── plots/
    ├── training_history.png    # Loss and accuracy curves
    ├── confusion_matrix_train.png
    ├── confusion_matrix_val.png
    ├── confusion_matrix_test.png
    ├── roc_curve_train.png
    ├── roc_curve_val.png
    └── roc_curve_test.png
```

### Key Files

**metrics.json** - Final evaluation metrics:
```json
{
  "train": {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.96,
    "f1": 0.95,
    "roc_auc": 0.98
  },
  "validation": {...},
  "test": {...}
}
```

**training_history.json** - Per-epoch training curves for plotting:
```json
{
  "train_loss": [0.6, 0.5, ...],
  "train_accuracy": [0.65, 0.75, ...],
  "val_loss": [0.58, 0.52, ...],
  "val_accuracy": [0.68, 0.76, ...]
}
```

## Using Trained Model

### Load and Evaluate

```python
import torch
from src.models.mlp import MLPClassifier
from src.eval.metrics import ModelEvaluator
from src.data.dataset import create_data_loaders

# Load model
model = MLPClassifier(
    input_dim=2000,
    hidden_dims=[512, 256, 128],
    output_dim=2,
)
model.load_state_dict(torch.load("results/mlp_baseline/model.pt"))

# Create evaluator
evaluator = ModelEvaluator(model, device="cuda")

# Load test data
_, _, test_loader = create_data_loaders(
    "results/processed_data/train_data.h5ad",
    "results/processed_data/val_data.h5ad",
    "results/processed_data/test_data.h5ad",
)

# Evaluate
metrics = evaluator.evaluate(test_loader, "test")
print(f"Test F1: {metrics['f1']:.4f}")

# Generate predictions
preds, probs, labels = evaluator.predict(test_loader)
```

### Make Predictions on New Data

```python
import torch
import numpy as np

# Load model
model.eval()
device = "cuda"

# Prepare your expression data (N cells × 2000 genes)
X = np.random.randn(10, 2000).astype(np.float32)  # Example
X = torch.from_numpy(X).to(device)

# Predict
with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)

print(f"Predictions: {preds.cpu().numpy()}")
print(f"Probabilities: {probs.cpu().numpy()}")
```

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA OOM errors:

1. **Reduce batch size**:
   ```bash
   python scripts/training/train_mlp.py --config configs/mlp_config.yaml
   ```
   Edit config: `batch_size: 16` (or lower)

2. **Reduce model size**:
   Edit config: `hidden_dims: [256, 128]` (smaller)

3. **Use CPU** (slower):
   Edit config: `device: "cpu"`

### Slow Training

If training is slow:

1. Check GPU utilization: `nvidia-smi`
2. Reduce `num_workers` in config if CPU bottleneck
3. Increase batch size if GPU underutilized (watch memory)

### Poor Performance

If metrics are low (F1 < 0.7):

1. **Check data**: Run `python scripts/explore_data.py` to verify preprocessing
2. **Try different hyperparameters**:
   - Lower learning rate: 5e-4
   - Increase epochs: 200
   - Reduce dropout: 0.2
3. **Check label distribution**: Are classes balanced in train/val/test?

## Comparison with Other Models

### Expected Performance

| Model | Accuracy | F1 | ROC-AUC | Training Time |
|-------|----------|-----|---------|---|
| MLP Baseline | ~85% | ~0.80 | ~0.85 | ~5 min |
| Transformer | ~87% | ~0.82 | ~0.87 | ~15 min |
| scGPT Fine-tune | ~90% | ~0.85 | ~0.90 | ~1 hour |

*Approximate figures based on typical runs*

## Advanced Usage

### Early Stopping

Early stopping automatically stops training if validation loss doesn't improve:

```yaml
early_stopping:
  enabled: true
  patience: 15  # Stop after 15 epochs without improvement
  metric: "val_loss"
  mode: "min"
```

### Learning Rate Warmup

Linear warmup helps stabilize training at the beginning:

```yaml
warmup:
  enabled: true
  steps: 500  # Linearly increase LR over 500 steps
```

### Gradient Clipping

Prevents exploding gradients:

```yaml
training:
  gradient_clipping: 1.0  # Clip gradients to norm ≤ 1.0
```

## Next Steps

After training the MLP baseline:

1. **Evaluate on test set**: Check final metrics
2. **Compare results**: Baseline for Transformer and scGPT models
3. **Analyze predictions**: Which samples are misclassified?
4. **Feature importance**: Use gradient-based attribution methods

---

For questions or issues, refer to `DATA_PIPELINE.md` and main `README.md`.
