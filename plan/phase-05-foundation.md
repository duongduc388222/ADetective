# Phase 05: Foundation Model Fine-tuning (scGPT)

## Objective
Fine-tune scGPT or other single-cell foundation models for AD pathology classification, leveraging pretrained knowledge from millions of cells.

## Duration
3-4 hours

## Tasks

### 5.1 scGPT Wrapper Implementation
Create `src/models/scgpt_wrapper.py`:
```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class scGPTWrapper(nn.Module):
    """
    Wrapper for scGPT foundation model fine-tuning.

    This wrapper handles:
    - Gene vocabulary alignment
    - Expression binning/tokenization
    - Classification head addition
    - Fine-tuning setup
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        n_bins: int = 51,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        freeze_layers: int = 0,
        dropout: float = 0.1,
        use_fast_tokenizer: bool = True
    ):
        """
        Initialize scGPT wrapper.

        Args:
            pretrained_path: Path to pretrained scGPT checkpoint
            vocab_path: Path to gene vocabulary file
            n_bins: Number of expression bins for tokenization
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            freeze_layers: Number of layers to freeze from bottom
            dropout: Dropout rate
            use_fast_tokenizer: Use optimized tokenizer
        """
        super().__init__()

        self.n_bins = n_bins
        self.d_model = d_model
        self.freeze_layers = freeze_layers

        # Load or create gene vocabulary
        self.gene_vocab = self._load_gene_vocab(vocab_path)
        self.vocab_size = len(self.gene_vocab)

        # Expression binning thresholds
        self.bin_edges = self._create_bin_edges(n_bins)

        # Token embeddings (genes + expression bins + special tokens)
        self.n_tokens = self.vocab_size * n_bins + 10  # +10 for special tokens
        self.token_embeddings = nn.Embedding(self.n_tokens, d_model)

        # Special token IDs
        self.cls_token_id = self.n_tokens - 3
        self.pad_token_id = self.n_tokens - 2
        self.mask_token_id = self.n_tokens - 1

        # Position embeddings
        self.position_embeddings = nn.Embedding(4096, d_model)  # Max 4096 genes

        # Layer norm
        self.ln_input = nn.LayerNorm(d_model)

        # Transformer encoder (scGPT architecture)
        if pretrained_path and Path(pretrained_path).exists():
            self._load_pretrained_model(pretrained_path)
        else:
            # Create new transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
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

        # Freeze layers if specified
        self._freeze_layers()

    def _load_gene_vocab(self, vocab_path: Optional[str]) -> Dict[str, int]:
        """Load gene vocabulary from file or create default."""
        if vocab_path and Path(vocab_path).exists():
            import json
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            logger.info(f"Loaded gene vocabulary with {len(vocab)} genes")
        else:
            # Create dummy vocabulary (will be replaced with actual genes)
            vocab = {f"GENE_{i}": i for i in range(2000)}
            logger.warning("Using dummy gene vocabulary. Replace with actual genes!")
        return vocab

    def _create_bin_edges(self, n_bins: int) -> torch.Tensor:
        """Create expression bin edges for discretization."""
        # Create log-spaced bins for expression values
        # Assuming log-normalized data roughly in range [0, 10]
        edges = torch.linspace(0, 10, n_bins + 1)
        return edges

    def _load_pretrained_model(self, checkpoint_path: str):
        """Load pretrained scGPT model."""
        logger.info(f"Loading pretrained model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load transformer weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Filter for transformer weights
            transformer_state = {k: v for k, v in state_dict.items()
                               if 'transformer' in k}
            self.transformer.load_state_dict(transformer_state, strict=False)

        # Load embeddings if available
        if 'token_embeddings' in checkpoint:
            self.token_embeddings.load_state_dict(
                checkpoint['token_embeddings'], strict=False
            )

        logger.info("Pretrained weights loaded successfully")

    def _freeze_layers(self):
        """Freeze specified number of bottom transformer layers."""
        if self.freeze_layers > 0:
            # Freeze embeddings
            for param in self.token_embeddings.parameters():
                param.requires_grad = False
            for param in self.position_embeddings.parameters():
                param.requires_grad = False

            # Freeze transformer layers
            if hasattr(self, 'transformer'):
                layers = self.transformer.encoder.layers
                for i in range(min(self.freeze_layers, len(layers))):
                    for param in layers[i].parameters():
                        param.requires_grad = False

            logger.info(f"Froze {self.freeze_layers} transformer layers")

    def tokenize_expression(
        self,
        expression_matrix: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize expression matrix into token IDs.

        Args:
            expression_matrix: Expression values (batch_size, n_genes)
            gene_names: List of gene names

        Returns:
            token_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        """
        batch_size, n_genes = expression_matrix.shape
        device = expression_matrix.device

        # Map genes to vocabulary IDs
        gene_ids = []
        valid_mask = []
        for gene in gene_names:
            if gene in self.gene_vocab:
                gene_ids.append(self.gene_vocab[gene])
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        # Filter expression matrix for valid genes
        valid_indices = [i for i, valid in enumerate(valid_mask) if valid]
        expression_filtered = expression_matrix[:, valid_indices]
        gene_ids = torch.tensor(gene_ids, device=device)

        # Discretize expression values into bins
        expression_bins = torch.bucketize(expression_filtered, self.bin_edges.to(device))
        expression_bins = torch.clamp(expression_bins, 0, self.n_bins - 1)

        # Create token IDs: gene_id * n_bins + bin_id
        token_ids = gene_ids.unsqueeze(0) * self.n_bins + expression_bins

        # Add CLS token at the beginning
        cls_tokens = torch.full((batch_size, 1), self.cls_token_id,
                               dtype=torch.long, device=device)
        token_ids = torch.cat([cls_tokens, token_ids], dim=1)

        # Create attention mask (all ones for valid tokens)
        attention_mask = torch.ones_like(token_ids)

        return token_ids, attention_mask

    def forward(
        self,
        expression_matrix: torch.Tensor,
        gene_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            expression_matrix: Expression values (batch_size, n_genes)
            gene_names: List of gene names (optional)

        Returns:
            Logits for binary classification (batch_size, 1)
        """
        batch_size = expression_matrix.size(0)
        device = expression_matrix.device

        # Use dummy gene names if not provided
        if gene_names is None:
            gene_names = [f"GENE_{i}" for i in range(expression_matrix.size(1))]

        # Tokenize expression
        token_ids, attention_mask = self.tokenize_expression(
            expression_matrix, gene_names
        )

        # Get token embeddings
        token_embeds = self.token_embeddings(token_ids)

        # Add position embeddings
        positions = torch.arange(token_ids.size(1), device=device)
        position_embeds = self.position_embeddings(positions)
        embeddings = token_embeds + position_embeds.unsqueeze(0)

        # Layer norm
        embeddings = self.ln_input(embeddings)

        # Pass through transformer
        # Create attention mask for transformer (1 = attend, 0 = ignore)
        attn_mask = (1.0 - attention_mask.float()) * -10000.0
        transformer_output = self.transformer(embeddings, src_key_padding_mask=attn_mask)

        # Extract CLS token representation
        cls_output = transformer_output[:, 0, :]

        # Classification
        logits = self.classifier(cls_output)

        return logits


class scGPTFineTuner:
    """
    Helper class for fine-tuning scGPT models.
    """

    def __init__(
        self,
        model: scGPTWrapper,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01
    ):
        """
        Initialize fine-tuner.

        Args:
            model: scGPT model to fine-tune
            learning_rate: Learning rate for fine-tuning
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters()
                              if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    def create_optimizer_and_scheduler(
        self,
        num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """
        Create optimizer and learning rate scheduler.

        Args:
            num_training_steps: Total number of training steps

        Returns:
            Optimizer and scheduler
        """
        # Separate parameters for different weight decay
        no_decay = ["bias", "LayerNorm.weight", "ln"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Cosine schedule with warmup
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / \
                      float(max(1, num_training_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return optimizer, scheduler

    def align_gene_vocabulary(
        self,
        dataset_genes: List[str],
        model_genes: List[str]
    ) -> Dict[str, str]:
        """
        Align dataset genes with model vocabulary.

        Args:
            dataset_genes: Gene names from dataset
            model_genes: Gene names from model vocabulary

        Returns:
            Mapping from dataset genes to model genes
        """
        gene_mapping = {}

        # Direct matching (case-insensitive)
        model_genes_upper = {g.upper(): g for g in model_genes}

        for gene in dataset_genes:
            gene_upper = gene.upper()
            if gene_upper in model_genes_upper:
                gene_mapping[gene] = model_genes_upper[gene_upper]

        logger.info(f"Matched {len(gene_mapping)} / {len(dataset_genes)} genes")

        # Report unmatched genes
        unmatched = set(dataset_genes) - set(gene_mapping.keys())
        if unmatched:
            logger.warning(f"Unmatched genes: {len(unmatched)} (will be ignored)")
            if len(unmatched) < 20:
                logger.warning(f"Examples: {list(unmatched)[:20]}")

        return gene_mapping
```

### 5.2 scGPT Installation Script
Create `scripts/install_scgpt.sh`:
```bash
#!/bin/bash
# Script to install scGPT and dependencies

echo "Installing scGPT dependencies..."

# Install scGPT from GitHub (if not on PyPI)
if ! pip show scgpt > /dev/null 2>&1; then
    echo "Installing scGPT from GitHub..."
    pip install git+https://github.com/bowang-lab/scGPT.git
fi

# Install Flash Attention for efficiency
echo "Checking Flash Attention..."
python -c "import flash_attn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Flash Attention..."
    pip install flash-attn --no-build-isolation
fi

# Download pretrained weights
WEIGHTS_DIR="./pretrained_models"
mkdir -p $WEIGHTS_DIR

# Download scGPT whole-human model (example URL - replace with actual)
if [ ! -f "$WEIGHTS_DIR/scGPT_human.pth" ]; then
    echo "Downloading pretrained scGPT model..."
    # wget https://example.com/scGPT_human.pth -O $WEIGHTS_DIR/scGPT_human.pth
    echo "Please download pretrained weights manually from scGPT repository"
fi

# Download gene vocabulary
if [ ! -f "$WEIGHTS_DIR/gene_vocab.json" ]; then
    echo "Creating gene vocabulary..."
    python -c "
import json
import requests

# Example: Create vocabulary from common human genes
# In practice, use the actual vocabulary from scGPT
genes = ['APOE', 'APP', 'MAPT', 'PSEN1', 'PSEN2']  # Add more genes
vocab = {gene: i for i, gene in enumerate(genes)}

with open('$WEIGHTS_DIR/gene_vocab.json', 'w') as f:
    json.dump(vocab, f)
print(f'Created vocabulary with {len(vocab)} genes')
"
fi

echo "scGPT setup complete!"
```

### 5.3 scGPT Training Script
Create `scripts/train_scgpt.py`:
```python
#!/usr/bin/env python3
"""
Fine-tune scGPT foundation model for Oligodendrocyte AD pathology classification.
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

from src.models.scgpt_wrapper import scGPTWrapper, scGPTFineTuner
from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data_with_genes(adata_path):
    """Load data and gene names for scGPT."""
    # Load processed data
    adata = sc.read_h5ad(adata_path)

    # Get gene names
    gene_names = list(adata.var_names)
    print(f"Dataset contains {len(gene_names)} genes")

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

    return X_train, y_train, X_val, y_val, X_test, y_test, gene_names

def main():
    parser = argparse.ArgumentParser(description='Fine-tune scGPT model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to processed h5ad file')
    parser.add_argument('--pretrained-path', type=str,
                       default='./pretrained_models/scGPT_human.pth',
                       help='Path to pretrained scGPT checkpoint')
    parser.add_argument('--vocab-path', type=str,
                       default='./pretrained_models/gene_vocab.json',
                       help='Path to gene vocabulary')
    parser.add_argument('--config', type=str, default='configs/scgpt_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./results/scgpt',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--precision', type=str, default='bf16',
                       choices=['no', 'fp16', 'bf16'],
                       help='Mixed precision training')
    parser.add_argument('--freeze-layers', type=int, default=6,
                       help='Number of transformer layers to freeze')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # Default configuration for scGPT
        config = {
            'model': {
                'n_bins': 51,
                'd_model': 512,
                'nhead': 8,
                'num_layers': 12,
                'dropout': 0.1,
                'use_fast_tokenizer': True
            },
            'training': {
                'weight_decay': 0.01,
                'gradient_accumulation_steps': 4,
                'max_grad_norm': 1.0,
                'patience': 3,
                'warmup_steps': 500
            }
        }

    # Prepare data with gene names
    X_train, y_train, X_val, y_val, X_test, y_test, gene_names = \
        prepare_data_with_genes(args.data_path)

    # Create scGPT model
    model = scGPTWrapper(
        pretrained_path=args.pretrained_path if Path(args.pretrained_path).exists() else None,
        vocab_path=args.vocab_path if Path(args.vocab_path).exists() else None,
        freeze_layers=args.freeze_layers,
        **config['model']
    )

    print(f"\nscGPT architecture:")
    print(f"  Model dimension: {config['model']['d_model']}")
    print(f"  Number of heads: {config['model']['nhead']}")
    print(f"  Number of layers: {config['model']['num_layers']}")
    print(f"  Expression bins: {config['model']['n_bins']}")
    print(f"  Frozen layers: {args.freeze_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create fine-tuner
    finetuner = scGPTFineTuner(
        model=model,
        learning_rate=args.learning_rate,
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay']
    )

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        learning_rate=args.learning_rate,
        mixed_precision=args.precision,
        **{k: v for k, v in config['training'].items()
           if k not in ['warmup_steps']}
    )

    # Override optimizer with fine-tuning specific one
    num_training_steps = len(X_train) // args.batch_size * args.num_epochs
    optimizer, scheduler = finetuner.create_optimizer_and_scheduler(num_training_steps)
    trainer.optimizer = optimizer

    # Create custom data loader that passes gene names
    class GeneExpressionDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, gene_names):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y).reshape(-1, 1)
            self.gene_names = gene_names

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # Create datasets
    train_dataset = GeneExpressionDataset(X_train, y_train, gene_names)
    val_dataset = GeneExpressionDataset(X_val, y_val, gene_names)
    test_dataset = GeneExpressionDataset(X_test, y_test, gene_names)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Prepare with accelerator
    train_loader = trainer.accelerator.prepare(train_loader)
    val_loader = trainer.accelerator.prepare(val_loader)
    test_loader = trainer.accelerator.prepare(test_loader)

    # Fine-tune model
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
        'pretrained_used': Path(args.pretrained_path).exists()
    }

    results_path = os.path.join(args.output_dir, 'results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)

    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main()
```

### 5.4 scGPT Configuration
Create `configs/scgpt_config.yaml`:
```yaml
# scGPT Model Configuration
model:
  n_bins: 51  # Number of expression bins
  d_model: 512  # Model dimension (matches pretrained)
  nhead: 8  # Number of attention heads
  num_layers: 12  # Number of transformer layers
  dropout: 0.1
  use_fast_tokenizer: true

# Fine-tuning Configuration
training:
  weight_decay: 0.01  # Higher weight decay for fine-tuning
  gradient_accumulation_steps: 4  # More accumulation due to memory
  max_grad_norm: 1.0
  patience: 3  # Lower patience for fine-tuning
  warmup_steps: 500

# Data Configuration
data:
  batch_size: 16  # Smaller batch size due to model size
  num_workers: 0

# Experiment Configuration
experiment:
  seed: 42
  num_epochs: 15  # Fewer epochs for fine-tuning
  learning_rate: 0.00005  # Lower learning rate for fine-tuning
  precision: bf16
  freeze_layers: 6  # Freeze bottom 6 layers
```

## Validation Checklist
- [ ] scGPT wrapper implemented with tokenization
- [ ] Gene vocabulary alignment functionality
- [ ] Expression binning for discrete tokens
- [ ] Pretrained weight loading capability
- [ ] Layer freezing for efficient fine-tuning
- [ ] Classification head added to foundation model
- [ ] Custom optimizer with warmup schedule
- [ ] Memory-efficient training with gradient accumulation

## Installation Requirements
```bash
# Install scGPT (if available)
pip install scgpt

# Or install from source
git clone https://github.com/bowang-lab/scGPT.git
cd scGPT
pip install -e .

# Additional dependencies
pip install flash-attn --no-build-isolation
pip install transformers>=4.30.0
```

## Expected Challenges & Solutions

### Gene Vocabulary Mismatch
- **Problem**: Dataset genes don't match pretrained vocabulary
- **Solution**: Implement flexible mapping, use only matched genes
- **Impact**: May lose some signal from unmatched genes

### Memory Constraints
- **Problem**: Large model doesn't fit in GPU memory
- **Solution**:
  - Use gradient accumulation (steps=4-8)
  - Reduce batch size (8-16)
  - Freeze more layers
  - Use gradient checkpointing

### Slow Training
- **Problem**: Fine-tuning is very slow
- **Solution**:
  - Freeze more layers (keep only top 3-4 trainable)
  - Use mixed precision (bf16/fp16)
  - Reduce sequence length if possible

## Expected Performance
- May achieve similar or better performance than custom models
- Benefits from pretrained knowledge of gene relationships
- Performance depends heavily on vocabulary overlap
- Fine-tuning typically converges faster than training from scratch

## Next Steps
- Compare all three models (MLP, Transformer, scGPT)
- Analyze which approach works best for this dataset
- Proceed to Phase 06 for final testing and documentation