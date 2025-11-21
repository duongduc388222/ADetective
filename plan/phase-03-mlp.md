# Phase 03: MLP Baseline Implementation

## Objective
Implement a Multi-Layer Perceptron (MLP) baseline model with Accelerate for bf16 mixed precision training and comprehensive evaluation metrics.

## Duration
1-2 hours

## Tasks

### 3.1 MLP Model Architecture
Create `src/models/mlp.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 1,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Multi-Layer Perceptron for binary classification.

        Args:
            input_dim: Number of input features (genes)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for binary classification)
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
        """
        super(MLPClassifier, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        self.use_batch_norm = use_batch_norm

        # Build layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Activation function
        self.activation = self._get_activation(activation)

        # Initialize weights
        self._initialize_weights()

    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }
        return activations.get(activation, nn.ReLU())

    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for layer in self.layers:
            if isinstance(self.activation, nn.ReLU) or isinstance(self.activation, nn.LeakyReLU):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropouts[i](x)

        # Output layer (no activation for logits)
        x = self.output_layer(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
```

### 3.2 Training Utilities
Create `src/training/trainer.py`:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        mixed_precision: str = 'bf16',
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        """
        Model trainer with Accelerate support.

        Args:
            model: PyTorch model to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            mixed_precision: Mixed precision mode ('no', 'fp16', 'bf16')
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        # Loss function for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Prepare model and optimizer with Accelerate
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def create_data_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True
    ) -> DataLoader:
        """Create DataLoader from numpy arrays."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)

        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Prepare with Accelerate
        dataloader = self.accelerator.prepare(dataloader)

        return dataloader

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training", disable=not self.accelerator.is_local_main_process)

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass
                self.accelerator.backward(loss)

                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Statistics
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        for inputs, targets in val_loader:
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Statistics
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            # Collect for metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total,
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets)
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 20,
        patience: int = 5,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            patience: Early stopping patience
            save_path: Path to save best model

        Returns:
            Training history and best metrics
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.accelerator.device}")
        logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")

        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])

                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    best_model_state = self.accelerator.unwrap_model(self.model).state_dict()
                else:
                    patience_counter += 1

                # Logging
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.2f}%"
                )

                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.2f}%"
                )

        # Load best model
        if best_model_state is not None:
            self.accelerator.unwrap_model(self.model).load_state_dict(best_model_state)

        # Save model
        if save_path and self.accelerator.is_local_main_process:
            self.save_model(save_path)

        return self.history

    def save_model(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.accelerator.device)
        self.accelerator.unwrap_model(self.model).load_state_dict(
            checkpoint['model_state_dict']
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Model loaded from {path}")
```

### 3.3 Training Script
Create `scripts/train_mlp.py`:
```python
#!/usr/bin/env python3
"""
Train MLP baseline model for Oligodendrocyte AD pathology classification.
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scanpy as sc
import yaml
from pathlib import Path

from src.models.mlp import MLPClassifier
from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(adata_path):
    """Load and prepare data for training."""
    # Load processed data
    adata = sc.read_h5ad(adata_path)

    # Split data
    train_mask = adata.obs['split'] == 'train'
    val_mask = adata.obs['split'] == 'val'
    test_mask = adata.obs['split'] == 'test'

    # Get expression matrices and labels
    X_train = adata[train_mask].X
    y_train = adata[train_mask].obs['label'].values

    X_val = adata[val_mask].X
    y_val = adata[val_mask].obs['label'].values

    X_test = adata[test_mask].X
    y_test = adata[test_mask].obs['label'].values

    # Convert sparse to dense if needed
    if hasattr(X_train, 'todense'):
        X_train = X_train.todense()
        X_val = X_val.todense()
        X_test = X_test.todense()

    # Convert to numpy arrays
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    X_test = np.asarray(X_test)

    print(f"Data shapes:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val: {X_val.shape}, {y_val.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    parser = argparse.ArgumentParser(description='Train MLP model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to processed h5ad file')
    parser.add_argument('--config', type=str, default='configs/mlp_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./results/mlp',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--precision', type=str, default='bf16',
                       choices=['no', 'fp16', 'bf16'],
                       help='Mixed precision training')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'model': {
                'hidden_dims': [512, 256, 128],
                'dropout_rate': 0.3,
                'use_batch_norm': True,
                'activation': 'relu'
            },
            'training': {
                'weight_decay': 1e-4,
                'gradient_accumulation_steps': 1,
                'max_grad_norm': 1.0,
                'patience': 5
            }
        }

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(args.data_path)

    # Create model
    input_dim = X_train.shape[1]
    model = MLPClassifier(
        input_dim=input_dim,
        **config['model']
    )

    print(f"\nModel architecture:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimensions: {config['model']['hidden_dims']}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        learning_rate=args.learning_rate,
        mixed_precision=args.precision,
        **config['training']
    )

    # Create data loaders
    train_loader = trainer.create_data_loader(
        X_train, y_train,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = trainer.create_data_loader(
        X_val, y_val,
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = trainer.create_data_loader(
        X_test, y_test,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Train model
    model_path = os.path.join(args.output_dir, 'best_model.pth')
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        patience=config['training']['patience'],
        save_path=model_path
    )

    # Evaluate on test set
    print("\n=== Test Set Evaluation ===")
    evaluator = ModelEvaluator(trainer.model, trainer.accelerator)
    test_metrics = evaluator.evaluate(test_loader)

    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # Save results
    results = {
        'config': config,
        'args': vars(args),
        'history': history,
        'test_metrics': test_metrics
    }

    results_path = os.path.join(args.output_dir, 'results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)

    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main()
```

### 3.4 MLP Configuration
Create `configs/mlp_config.yaml`:
```yaml
# MLP Model Configuration
model:
  hidden_dims: [512, 256, 128]
  dropout_rate: 0.3
  use_batch_norm: true
  activation: relu

# Training Configuration
training:
  weight_decay: 0.0001
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  patience: 5

# Data Configuration
data:
  batch_size: 64
  num_workers: 0

# Experiment Configuration
experiment:
  seed: 42
  num_epochs: 30
  learning_rate: 0.001
  precision: bf16
```

## Validation Checklist
- [ ] MLP model architecture implemented with configurable layers
- [ ] Batch normalization and dropout for regularization
- [ ] Accelerate integration for bf16 mixed precision
- [ ] Training loop with early stopping
- [ ] Model checkpointing and loading
- [ ] Comprehensive evaluation metrics (accuracy, F1, ROC-AUC)
- [ ] Configuration system for hyperparameters
- [ ] Results saved for comparison

## Expected Performance
- Training should converge within 20-30 epochs
- Baseline accuracy: 70-80%
- F1 score: 0.65-0.75
- ROC-AUC: 0.75-0.85

## Troubleshooting
- If CUDA out of memory: Reduce batch size or hidden dimensions
- If loss is NaN: Check for bf16 support or switch to fp16/fp32
- If poor performance: Adjust learning rate or add more regularization

## Next Steps
- Train MLP baseline and record performance
- Use these results as benchmark for transformer models
- Proceed to Phase 04 for custom transformer implementation