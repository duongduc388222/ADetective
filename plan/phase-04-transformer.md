# Phase 04: Custom Transformer Implementation

## Objective
Implement a custom Transformer model that treats genes as sequence tokens with expression values, leveraging PyTorch 2.0's automatic Flash Attention optimization.

## Duration
2-3 hours

## Tasks

### 4.1 Transformer Model Architecture
Create `src/models/transformer.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class GeneTransformer(nn.Module):
    def __init__(
        self,
        num_genes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_length: int = 2048,
        use_cls_token: bool = True,
        expression_scaling: str = 'multiplicative'
    ):
        """
        Transformer model for gene expression classification.

        Args:
            num_genes: Number of genes in the input
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            use_cls_token: Whether to use CLS token for classification
            expression_scaling: How to combine gene embeddings with expression values
                              ('multiplicative', 'additive', 'concatenate')
        """
        super(GeneTransformer, self).__init__()

        self.num_genes = num_genes
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.expression_scaling = expression_scaling

        # Gene embeddings
        self.gene_embeddings = nn.Embedding(num_genes, d_model)

        # CLS token embedding if used
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Expression value processing
        if expression_scaling == 'additive':
            self.expression_projection = nn.Linear(1, d_model)
        elif expression_scaling == 'concatenate':
            self.gene_embeddings = nn.Embedding(num_genes, d_model // 2)
            self.expression_projection = nn.Linear(1, d_model // 2)
            self.fusion_layer = nn.Linear(d_model, d_model)

        # Positional encoding (optional, genes already have fixed positions)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        # Initialize gene embeddings
        nn.init.normal_(self.gene_embeddings.weight, mean=0.0, std=0.02)

        # Initialize classifier
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def create_gene_representations(
        self,
        expression_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Create gene representations from expression values.

        Args:
            expression_values: Tensor of shape (batch_size, num_genes)

        Returns:
            Gene representations of shape (batch_size, num_genes, d_model)
        """
        batch_size = expression_values.size(0)

        # Create gene indices
        gene_indices = torch.arange(self.num_genes, device=expression_values.device)
        gene_indices = gene_indices.unsqueeze(0).expand(batch_size, -1)

        # Get gene embeddings
        gene_embeds = self.gene_embeddings(gene_indices)  # (batch, num_genes, d_model)

        if self.expression_scaling == 'multiplicative':
            # Scale embeddings by expression values
            expression_values = expression_values.unsqueeze(-1)  # (batch, num_genes, 1)
            # Apply soft scaling to avoid extreme values
            expression_scale = torch.tanh(expression_values / 5.0) + 1.0
            gene_representations = gene_embeds * expression_scale

        elif self.expression_scaling == 'additive':
            # Add expression embedding to gene embedding
            expression_values = expression_values.unsqueeze(-1)  # (batch, num_genes, 1)
            expression_embeds = self.expression_projection(expression_values)
            gene_representations = gene_embeds + expression_embeds

        elif self.expression_scaling == 'concatenate':
            # Concatenate gene and expression embeddings
            expression_values = expression_values.unsqueeze(-1)
            expression_embeds = self.expression_projection(expression_values)
            concat_embeds = torch.cat([gene_embeds, expression_embeds], dim=-1)
            gene_representations = self.fusion_layer(concat_embeds)

        else:
            gene_representations = gene_embeds

        return gene_representations

    def forward(self, expression_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer.

        Args:
            expression_values: Tensor of shape (batch_size, num_genes)

        Returns:
            Logits of shape (batch_size, 1)
        """
        batch_size = expression_values.size(0)

        # Create gene representations
        gene_representations = self.create_gene_representations(expression_values)

        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            sequence = torch.cat([cls_tokens, gene_representations], dim=1)
        else:
            sequence = gene_representations

        # Apply positional encoding (optional)
        sequence = self.positional_encoding(sequence)

        # Pass through transformer
        transformer_output = self.transformer(sequence)

        # Extract representation for classification
        if self.use_cls_token:
            # Use CLS token output
            cell_representation = transformer_output[:, 0, :]
        else:
            # Use mean pooling
            cell_representation = transformer_output.mean(dim=1)

        # Classification
        logits = self.classifier(cell_representation)

        return logits

    def get_attention_weights(
        self,
        expression_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights for interpretability.

        Returns attention weights from the last layer.
        """
        self.eval()
        with torch.no_grad():
            batch_size = expression_values.size(0)

            # Create gene representations
            gene_representations = self.create_gene_representations(expression_values)

            # Add CLS token
            if self.use_cls_token:
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                sequence = torch.cat([cls_tokens, gene_representations], dim=1)
            else:
                sequence = gene_representations

            # Apply positional encoding
            sequence = self.positional_encoding(sequence)

            # Get attention weights from transformer
            # Note: This requires modifying the transformer to return attention weights
            # For now, return the sequence for compatibility
            return sequence, sequence


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FlashTransformer(GeneTransformer):
    """
    Transformer with Flash Attention optimization.

    PyTorch 2.0+ automatically uses Flash Attention when available.
    This class ensures optimal settings for Flash Attention.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure we're using PyTorch 2.0+ scaled_dot_product_attention
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Set SDPA backend preferences for Flash Attention
            # This happens automatically in PyTorch 2.0+
            self._use_flash_attn = True
        else:
            self._use_flash_attn = False
            print("Warning: Flash Attention not available. Using standard attention.")

    def forward(self, expression_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Flash Attention optimization.

        Flash Attention is automatically used by PyTorch 2.0+ when:
        - CUDA device with compute capability >= 8.0
        - Sequence length and dimensions meet requirements
        - No attention mask is used (or causal mask)
        """
        # PyTorch 2.0 automatically optimizes this
        return super().forward(expression_values)
```

### 4.2 Transformer Training Script
Create `scripts/train_transformer.py`:
```python
#!/usr/bin/env python3
"""
Train Transformer model for Oligodendrocyte AD pathology classification.
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scanpy as sc
import yaml
import torch
from pathlib import Path

from src.models.transformer import FlashTransformer
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

def check_flash_attention():
    """Check if Flash Attention is available."""
    print("\n=== Flash Attention Status ===")

    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Check compute capability
        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = major + minor / 10
        print(f"Compute capability: {compute_capability}")

        if compute_capability >= 8.0:
            print("✓ GPU supports Flash Attention (compute capability >= 8.0)")
        else:
            print("✗ GPU does not support Flash Attention (compute capability < 8.0)")
    else:
        print("CUDA available: No")
        print("✗ Flash Attention requires CUDA GPU")

    # Check if scaled_dot_product_attention is available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("✓ PyTorch 2.0+ scaled_dot_product_attention available")
    else:
        print("✗ PyTorch 2.0+ required for automatic Flash Attention")

    print("=" * 30)

def main():
    parser = argparse.ArgumentParser(description='Train Transformer model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to processed h5ad file')
    parser.add_argument('--config', type=str, default='configs/transformer_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./results/transformer',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--precision', type=str, default='bf16',
                       choices=['no', 'fp16', 'bf16'],
                       help='Mixed precision training')
    parser.add_argument('--use-flash-attn', action='store_true',
                       help='Use Flash Attention (automatic in PyTorch 2.0+)')
    args = parser.parse_args()

    # Check Flash Attention availability
    check_flash_attention()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'model': {
                'd_model': 128,
                'nhead': 8,
                'num_layers': 3,
                'dim_feedforward': 256,
                'dropout': 0.1,
                'use_cls_token': True,
                'expression_scaling': 'multiplicative'
            },
            'training': {
                'weight_decay': 1e-4,
                'gradient_accumulation_steps': 2,
                'max_grad_norm': 1.0,
                'patience': 5
            }
        }

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(args.data_path)

    # Create model
    num_genes = X_train.shape[1]
    model = FlashTransformer(
        num_genes=num_genes,
        **config['model']
    )

    print(f"\nTransformer architecture:")
    print(f"  Number of genes: {num_genes}")
    print(f"  Model dimension: {config['model']['d_model']}")
    print(f"  Number of heads: {config['model']['nhead']}")
    print(f"  Number of layers: {config['model']['num_layers']}")
    print(f"  Expression scaling: {config['model']['expression_scaling']}")
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
        'test_metrics': test_metrics,
        'flash_attn_available': hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    }

    results_path = os.path.join(args.output_dir, 'results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)

    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main()
```

### 4.3 Transformer Configuration
Create `configs/transformer_config.yaml`:
```yaml
# Transformer Model Configuration
model:
  d_model: 128
  nhead: 8
  num_layers: 3
  dim_feedforward: 256
  dropout: 0.1
  use_cls_token: true
  expression_scaling: multiplicative  # multiplicative, additive, or concatenate
  max_seq_length: 2048

# Training Configuration
training:
  weight_decay: 0.0001
  gradient_accumulation_steps: 2  # Useful for larger models
  max_grad_norm: 1.0
  patience: 5

# Data Configuration
data:
  batch_size: 32  # Smaller than MLP due to memory requirements
  num_workers: 0

# Experiment Configuration
experiment:
  seed: 42
  num_epochs: 30
  learning_rate: 0.0001  # Lower than MLP
  precision: bf16
  use_flash_attn: true  # Automatic in PyTorch 2.0+
```

## Validation Checklist
- [ ] Transformer architecture with gene-as-token representation
- [ ] Gene embeddings combined with expression values
- [ ] CLS token for cell-level classification
- [ ] Flash Attention support via PyTorch 2.0
- [ ] Mixed precision training with bf16
- [ ] Gradient accumulation for memory efficiency
- [ ] Model interpretability via attention weights
- [ ] Configuration system for hyperparameters

## Memory Optimization Strategies
1. **Batch Size**: Start with 32, reduce if OOM
2. **Gradient Accumulation**: Simulate larger batches
3. **Mixed Precision**: Use bf16/fp16 to save memory
4. **Sequence Length**: Use HVGs to limit to ~2000 genes
5. **Model Size**: Reduce d_model or num_layers if needed

## Flash Attention Requirements
- PyTorch 2.0+ (automatic optimization)
- CUDA GPU with compute capability >= 8.0 (A100, H100)
- For older GPUs (V100, T4), standard attention is used
- No code changes needed - PyTorch handles optimization

## Expected Performance
- May achieve similar or slightly better performance than MLP
- Attention weights can provide gene importance insights
- Training may take longer due to model complexity
- Benefits most when gene interactions are important

## Troubleshooting
- **OOM Error**: Reduce batch_size or increase gradient_accumulation_steps
- **Slow Training**: Ensure Flash Attention is active (check GPU compatibility)
- **Poor Performance**: Try different expression_scaling methods
- **Instability**: Reduce learning rate or increase warmup steps

## Next Steps
- Compare transformer performance with MLP baseline
- Analyze attention weights for biological insights
- Proceed to Phase 05 for foundation model fine-tuning