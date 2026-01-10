#!/usr/bin/env python3
"""
Train CellFM classifier for oligodendrocyte AD classification.

Finetunes CellFM foundation model backbone with a classification head.

This script:
1. Loads vocab-aligned h5ad files (prepared with --foundation-model-mode)
2. Creates PyTorch dataloaders with sparse-to-dense conversion
3. Loads CellFM pretrained backbone
4. Initializes CellFMClassifier with classification head
5. Trains using PyTorch Lightning (or Accelerate if --use-accelerate)
6. Evaluates on test set
7. Saves results and checkpoints

Usage:
    python scripts/train_cellfm.py \\
        --data-dir ./data/cellfm_prepared \\
        --backbone-path /path/to/cellfm_weights.pt \\
        --output-dir ./results/cellfm

    # With backbone freezing (recommended for small datasets)
    python scripts/train_cellfm.py \\
        --data-dir ./data/cellfm_prepared \\
        --backbone-path /path/to/cellfm_weights.pt \\
        --freeze-backbone

    # Full finetuning (larger datasets)
    python scripts/train_cellfm.py \\
        --data-dir ./data/cellfm_prepared \\
        --backbone-path /path/to/cellfm_weights.pt \\
        --no-freeze-backbone \\
        --learning-rate 1e-5

    # With Accelerate for distributed training
    python scripts/train_cellfm.py \\
        --data-dir ./data/cellfm_prepared \\
        --backbone-path /path/to/cellfm_weights.pt \\
        --use-accelerate
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import anndata as ad
from scipy import sparse

# Optional imports - will be checked at runtime
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from torchmetrics import Accuracy, F1Score, AUROC, Precision, Recall
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def check_flash_attention():
    """Check if Flash Attention is available."""
    logger.info("\n=== Flash Attention Status ===")
    logger.info(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = major + minor / 10
        logger.info(f"Compute capability: {compute_capability}")

        if compute_capability >= 8.0:
            logger.info("GPU supports Flash Attention (compute capability >= 8.0)")
        else:
            logger.warning("GPU does not support Flash Attention (compute capability < 8.0)")
    else:
        logger.warning("CUDA available: No")
        logger.warning("Flash Attention requires CUDA GPU")

    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        logger.info("PyTorch 2.0+ scaled_dot_product_attention available")
    else:
        logger.warning("PyTorch 2.0+ required for automatic Flash Attention")

    logger.info("=" * 30)


# =============================================================================
# Dataset
# =============================================================================

class ScanpyDataset(Dataset):
    """
    Dataset for loading vocab-aligned h5ad files.

    Handles sparse matrices efficiently by converting to dense only at access time.
    """

    def __init__(self, h5ad_path: str):
        """
        Initialize dataset from h5ad file.

        Args:
            h5ad_path: Path to h5ad file with gene expression and labels
        """
        self.h5ad_path = h5ad_path
        logger.info(f"Loading dataset from {h5ad_path}")

        self.adata = ad.read_h5ad(h5ad_path)
        self.n_genes = self.adata.n_vars
        self.n_cells = self.adata.n_obs
        self.is_sparse = sparse.issparse(self.adata.X)

        logger.info(f"  Shape: ({self.n_cells:,}, {self.n_genes:,})")
        logger.info(f"  Sparse: {self.is_sparse}")

        # Validate labels exist
        if "label" not in self.adata.obs.columns:
            raise ValueError(f"'label' column not found in obs. Available: {list(self.adata.obs.columns)}")

        # Get label distribution
        label_counts = self.adata.obs["label"].value_counts()
        logger.info(f"  Labels: {label_counts.to_dict()}")

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (gene_expression_tensor, label_tensor)
        """
        # Get gene expression - convert sparse to dense
        if self.is_sparse:
            x = np.asarray(self.adata.X[idx].todense()).flatten().astype(np.float32)
        else:
            x = np.asarray(self.adata.X[idx]).flatten().astype(np.float32)

        # Get label
        y = int(self.adata.obs["label"].iloc[idx])

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# =============================================================================
# Model Components
# =============================================================================

# Import CellFM backbone loader from cellfm_backbone.py
try:
    from cellfm_backbone import load_cellfm_backbone
except ImportError:
    # Fallback if cellfm_backbone.py is not in the same directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from cellfm_backbone import load_cellfm_backbone
    except ImportError:
        def load_cellfm_backbone(
            weights_path: str,
            config_path: Optional[str] = None,
            strict: bool = False,
        ) -> Tuple[nn.Module, int]:
            """Placeholder - cellfm_backbone.py not found."""
            raise ImportError(
                "\n" + "=" * 70 + "\n"
                "CellFM backbone loader not found!\n"
                "=" * 70 + "\n\n"
                "Please ensure cellfm_backbone.py exists in scripts/\n"
                "Or install it with:\n"
                "  See docs/CELLFM_FINETUNING_GUIDE.md for instructions\n"
                + "=" * 70
            )


class CellFMClassifier(nn.Module):
    """
    CellFM backbone + MLP classification head.

    Architecture:
        backbone(x) -> features (B, hidden_dim)
        classifier(features) -> logits (B, num_classes)

    Classification head structure:
        Linear(hidden_dim, head_hidden)
        -> LayerNorm(head_hidden)
        -> ReLU
        -> Dropout(dropout)
        -> Linear(head_hidden, num_classes)
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 1536,
        num_classes: int = 2,
        head_hidden: int = 512,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        """
        Initialize CellFM classifier.

        Args:
            backbone: Pretrained CellFM encoder
            hidden_dim: Output dimension of backbone (default: 1536)
            num_classes: Number of output classes (default: 2 for binary)
            head_hidden: Hidden dimension in classification head
            dropout: Dropout rate in classification head
            freeze_backbone: If True, freeze backbone parameters
        """
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.freeze_backbone = freeze_backbone

        # Freeze backbone if requested
        if freeze_backbone:
            logger.info("Freezing backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Set to eval mode for frozen backbone
            self.backbone.eval()

        # Classification head: Linear -> LayerNorm -> ReLU -> Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

        # Initialize classifier weights
        self._init_classifier()

        # Log parameter counts
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Model parameters:")
        logger.info(f"  Backbone: {backbone_params:,} (frozen: {freeze_backbone})")
        logger.info(f"  Classifier: {classifier_params:,}")
        logger.info(f"  Trainable: {trainable_params:,}")

    def _init_classifier(self):
        """Initialize classifier weights using Kaiming initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Gene expression tensor (batch_size, n_genes)

        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        # Get features from backbone
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        # Classification head
        logits = self.classifier(features)
        return logits

    def train(self, mode: bool = True):
        """Override train to keep frozen backbone in eval mode."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


# =============================================================================
# Trainer for Accelerate Mode (similar to MLPTrainer/TransformerTrainer)
# =============================================================================

class CellFMTrainer:
    """
    Custom trainer for CellFM with Accelerate support.

    Similar to MLPTrainer and TransformerTrainer patterns.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = "cuda",
        use_accelerate: bool = False,
        accelerator=None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: CellFMClassifier model
            config: Training configuration dict
            device: Device to use
            use_accelerate: Whether to use HuggingFace Accelerate
            accelerator: Accelerator instance if using Accelerate
            class_weights: Optional class weights for loss
        """
        self.model = model
        self.config = config
        self.device = device
        self.use_accelerate = use_accelerate
        self.accelerator = accelerator

        # Move class weights to device
        if class_weights is not None:
            if use_accelerate and accelerator is not None:
                class_weights = class_weights.to(accelerator.device)
            else:
                class_weights = class_weights.to(device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create scheduler (optional)
        self.scheduler = self._create_scheduler()

        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Move model to device if not using Accelerate
        if not use_accelerate:
            self.model = self.model.to(device)

    def _is_main_process(self) -> bool:
        """Check if this is the main process (for logging in distributed training)."""
        if not self.use_accelerate or self.accelerator is None:
            return True
        return self.accelerator.is_main_process

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        training_config = self.config.get("training", {})
        lr = training_config.get("learning_rate", 1e-4)
        weight_decay = training_config.get("weight_decay", 0.01)

        # Only optimize trainable parameters
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def _create_scheduler(self):
        """Create learning rate scheduler (optional)."""
        training_config = self.config.get("training", {})
        warmup_config = training_config.get("warmup", {})

        if warmup_config.get("enabled", False):
            warmup_steps = warmup_config.get("steps", 100)
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        return None

    def train_epoch(self, train_loader, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        training_config = self.config.get("training", {})
        gradient_clip = training_config.get("gradient_clipping", 1.0)

        for batch_idx, (x, y) in enumerate(train_loader):
            # Move to device if not using Accelerate
            if not self.use_accelerate:
                x = x.to(self.device)
                y = y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)

            # Backward pass
            if self.use_accelerate and self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                if self.use_accelerate and self.accelerator is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), gradient_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def validate(self, val_loader) -> Tuple[float, float, Dict]:
        """
        Validate model.

        Returns:
            Tuple of (average_loss, accuracy, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                if not self.use_accelerate:
                    x = x.to(self.device)
                    y = y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)[:, 1]

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "roc_auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        }

        return avg_loss, metrics["accuracy"], metrics

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
    ) -> Dict:
        """
        Train model for specified epochs.

        Returns:
            Training history dict
        """
        training_config = self.config.get("training", {})
        early_stopping = training_config.get("early_stopping", {})
        patience = early_stopping.get("patience", 5) if early_stopping.get("enabled", True) else num_epochs

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            # Validate
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            # Log only from main process (avoids duplicate logs in distributed training)
            if self._is_main_process():
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_metrics['f1']:.4f}"
                )

            # Early stopping (only check on main process, then sync)
            should_stop = False
            if self._is_main_process():
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        should_stop = True

            # Synchronize early stopping across all processes
            if self.use_accelerate and self.accelerator is not None:
                should_stop_tensor = torch.tensor([should_stop], device=self.accelerator.device)
                should_stop_tensor = self.accelerator.gather(should_stop_tensor)
                should_stop = should_stop_tensor[0].item()

            if should_stop:
                break

        return history


# =============================================================================
# PyTorch Lightning Module
# =============================================================================

class CellFMLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for CellFM finetuning.

    Handles training loop, validation, metrics, and optimization.
    """

    def __init__(
        self,
        model: CellFMClassifier,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize Lightning module.

        Args:
            model: CellFMClassifier instance
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for AdamW
            class_weights: Optional class weights for imbalanced data
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Store class weights as buffer (not a parameter)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # Loss function with optional class weights
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # Metrics - separate instances for each phase
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_auroc = AUROC(task="binary")

        self.test_acc = Accuracy(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_auroc = AUROC(task="binary")

        # Save hyperparameters (exclude model to avoid serialization issues)
        self.save_hyperparameters(ignore=["model", "class_weights"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute predictions for accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute predictions and probabilities
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]

        # Update metrics
        self.val_acc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)
        self.val_auroc(probs, y)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_auroc", self.val_auroc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute predictions and probabilities
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]

        # Update metrics
        self.test_acc(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_f1(preds, y)
        self.test_auroc(probs, y)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        # Only optimize parameters with requires_grad=True
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune CellFM for oligodendrocyte AD classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with frozen backbone
  python scripts/train_cellfm.py \\
    --data-dir ./data/cellfm_prepared \\
    --backbone-path /path/to/cellfm_weights.pt \\
    --freeze-backbone

  # Full finetuning with lower learning rate
  python scripts/train_cellfm.py \\
    --data-dir ./data/cellfm_prepared \\
    --backbone-path /path/to/cellfm_weights.pt \\
    --no-freeze-backbone \\
    --learning-rate 1e-5

  # Custom training configuration
  python scripts/train_cellfm.py \\
    --data-dir ./data/cellfm_prepared \\
    --backbone-path /path/to/cellfm_weights.pt \\
    --output-dir ./results/cellfm_experiment \\
    --batch-size 64 \\
    --epochs 20 \\
    --patience 7
        """,
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with train.h5ad, val.h5ad, test.h5ad (prepared with --foundation-model-mode)",
    )
    parser.add_argument(
        "--backbone-path",
        type=str,
        required=True,
        help="Path to CellFM pretrained weights (.pt or .ckpt)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to CellFM config file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/cellfm",
        help="Directory to save results and checkpoints (default: ./results/cellfm)",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Maximum number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization (default: 0.01)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )

    # Model arguments
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=True,
        help="Freeze backbone and train only classifier head (default: True)",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Finetune entire model including backbone",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=1536,
        help="Hidden dimension of backbone output (default: 1536, will be overridden if read from model)",
    )
    parser.add_argument(
        "--head-hidden",
        type=int,
        default=512,
        help="Hidden dimension of classification head (default: 512)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate in classification head (default: 0.3)",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (default: 1.0)",
    )

    # Accelerate arguments
    parser.add_argument(
        "--use-accelerate",
        action="store_true",
        default=False,
        help="Use HuggingFace Accelerate for distributed training (default: False)",
    )
    parser.add_argument(
        "--no-accelerate",
        dest="use_accelerate",
        action="store_false",
        help="Disable Accelerate and use PyTorch Lightning (default)",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main training function."""
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("CellFM FINETUNING")
    logger.info("=" * 80)

    # Check Lightning availability for non-Accelerate mode
    if not args.use_accelerate and not LIGHTNING_AVAILABLE:
        logger.error("PyTorch Lightning not available. Install with: pip install pytorch-lightning torchmetrics")
        logger.error("Or use --use-accelerate for Accelerate-based training")
        return False

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nConfiguration:")
    logger.info(f"  Data directory: {data_dir}")
    logger.info(f"  Backbone path: {args.backbone_path}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Freeze backbone: {args.freeze_backbone}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Early stopping patience: {args.patience}")
    logger.info(f"  Use Accelerate: {args.use_accelerate}")

    # Step 0: Check Flash Attention
    logger.info("\n" + "=" * 80)
    logger.info("STEP 0: Checking Flash Attention")
    logger.info("=" * 80)
    check_flash_attention()

    # Step 1: Load datasets
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Datasets")
    logger.info("=" * 80)

    try:
        train_dataset = ScanpyDataset(data_dir / "train.h5ad")
        val_dataset = ScanpyDataset(data_dir / "val.h5ad")
        test_dataset = ScanpyDataset(data_dir / "test.h5ad")
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error("Make sure you've run prepare_seaad.py with --foundation-model-mode first")
        return False

    # Verify gene dimensions match
    assert train_dataset.n_genes == val_dataset.n_genes == test_dataset.n_genes, \
        "Gene dimensions must match across splits!"
    n_genes = train_dataset.n_genes
    logger.info(f"\nDatasets loaded: {n_genes:,} genes per sample")

    # Step 2: Create dataloaders
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Creating DataLoaders")
    logger.info("=" * 80)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Step 3: Calculate class weights
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Calculating Class Weights")
    logger.info("=" * 80)

    labels = train_dataset.adata.obs["label"].values.astype(int)
    class_counts = np.bincount(labels)
    # Inverse frequency weighting
    class_weights = torch.tensor(
        len(labels) / (len(class_counts) * class_counts),
        dtype=torch.float32,
    )
    logger.info(f"Class counts: {dict(enumerate(class_counts))}")
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Step 4: Load CellFM backbone
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Loading CellFM Backbone")
    logger.info("=" * 80)

    try:
        backbone, hidden_dim = load_cellfm_backbone(
            args.backbone_path,
            config_path=args.config_path,
            strict=False,
        )
        logger.info(f"Backbone loaded with hidden_dim={hidden_dim}")
    except NotImplementedError as e:
        logger.error(str(e))
        return False
    except Exception as e:
        logger.error(f"Failed to load backbone: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Create classifier model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Creating CellFM Classifier")
    logger.info("=" * 80)

    model = CellFMClassifier(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_classes=2,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    )

    # =========================================================================
    # Training Mode: Accelerate or PyTorch Lightning
    # =========================================================================

    if args.use_accelerate:
        # =====================================================================
        # ACCELERATE MODE (similar to train_mlp.py / train_transformer.py)
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Setting Up Accelerate Training")
        logger.info("=" * 80)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        training_config = {
            "training": {
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "optimizer": "adamw",
                "epochs": args.epochs,
                "gradient_clipping": args.gradient_clip,
                "early_stopping": {"enabled": True, "patience": args.patience},
                "warmup": {"enabled": False},
            },
            "logging": {"log_frequency": 50},
        }

        # Initialize Accelerator
        accelerator_instance = None
        try:
            from accelerate import Accelerator
            accelerator_instance = Accelerator()
            logger.info(f"Initialized Accelerator on device: {accelerator_instance.device}")

            # Prepare model and dataloaders with Accelerate
            model, train_loader, val_loader, test_loader = accelerator_instance.prepare(
                model, train_loader, val_loader, test_loader
            )
            logger.info("Model and dataloaders prepared with Accelerate")
        except ImportError:
            logger.warning("Accelerate not available, falling back to standard training")
            accelerator_instance = None

        # Create trainer
        trainer = CellFMTrainer(
            model,
            config=training_config,
            device=device,
            use_accelerate=(accelerator_instance is not None),
            accelerator=accelerator_instance,
            class_weights=class_weights,
        )

        # Step 7: Train
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Training")
        logger.info("=" * 80)

        history = trainer.fit(train_loader, val_loader, num_epochs=args.epochs)

        # Save checkpoint
        checkpoint_path = output_dir / "checkpoint.pt"
        if accelerator_instance is not None:
            model_to_save = accelerator_instance.unwrap_model(model)
        else:
            model_to_save = model
        torch.save(model_to_save.state_dict(), checkpoint_path)
        logger.info(f"Saved model checkpoint to {checkpoint_path}")

        # Step 8: Test evaluation
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: Test Evaluation")
        logger.info("=" * 80)

        _, _, test_metrics = trainer.validate(test_loader)

        # Step 9: Save results
        logger.info("\n" + "=" * 80)
        logger.info("STEP 9: Saving Results")
        logger.info("=" * 80)

        results = {
            "config": {
                "data_dir": str(data_dir),
                "backbone_path": args.backbone_path,
                "freeze_backbone": args.freeze_backbone,
                "hidden_dim": hidden_dim,
                "head_hidden": args.head_hidden,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "patience": args.patience,
                "use_accelerate": True,
            },
            "data": {
                "n_genes": n_genes,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "class_weights": class_weights.tolist(),
            },
            "training_history": {
                "train_loss": history["train_loss"],
                "train_accuracy": history["train_accuracy"],
                "val_loss": history["val_loss"],
                "val_accuracy": history["val_accuracy"],
            },
            "test_metrics": test_metrics,
        }

    else:
        # =====================================================================
        # PYTORCH LIGHTNING MODE (default)
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Creating Lightning Module")
        logger.info("=" * 80)

        lightning_module = CellFMLightningModule(
            model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            class_weights=class_weights,
        )

        # Step 7: Setup callbacks
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Setting Up Training")
        logger.info("=" * 80)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                mode="min",
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath=output_dir,
                filename="cellfm-{epoch:02d}-{val_f1:.3f}",
                monitor="val_f1",
                mode="max",
                save_top_k=1,
                save_last=True,
            ),
        ]

        # Create trainer with gradient clipping
        pl_trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            devices="auto",
            callbacks=callbacks,
            default_root_dir=output_dir,
            enable_progress_bar=True,
            log_every_n_steps=10,
            gradient_clip_val=args.gradient_clip,
        )

        # Step 8: Train
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: Training")
        logger.info("=" * 80)

        pl_trainer.fit(lightning_module, train_loader, val_loader)

        # Step 9: Test evaluation
        logger.info("\n" + "=" * 80)
        logger.info("STEP 9: Test Evaluation")
        logger.info("=" * 80)

        test_results = pl_trainer.test(lightning_module, test_loader, ckpt_path="best")

        # Step 10: Save results
        logger.info("\n" + "=" * 80)
        logger.info("STEP 10: Saving Results")
        logger.info("=" * 80)

        test_metrics = test_results[0] if test_results else {}

        results = {
            "config": {
                "data_dir": str(data_dir),
                "backbone_path": args.backbone_path,
                "freeze_backbone": args.freeze_backbone,
                "hidden_dim": hidden_dim,
                "head_hidden": args.head_hidden,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "patience": args.patience,
                "use_accelerate": False,
            },
            "data": {
                "n_genes": n_genes,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "class_weights": class_weights.tolist(),
            },
            "test_metrics": test_metrics,
        }

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)

    if test_metrics:
        logger.info(f"\nFinal Test Performance:")
        # Handle both Lightning and Accelerate metric formats
        acc_key = "test_acc" if "test_acc" in test_metrics else "accuracy"
        prec_key = "test_precision" if "test_precision" in test_metrics else "precision"
        rec_key = "test_recall" if "test_recall" in test_metrics else "recall"
        f1_key = "test_f1" if "test_f1" in test_metrics else "f1"
        auc_key = "test_auroc" if "test_auroc" in test_metrics else "roc_auc"

        logger.info(f"  Accuracy:  {test_metrics.get(acc_key, 'N/A'):.4f}")
        logger.info(f"  Precision: {test_metrics.get(prec_key, 'N/A'):.4f}")
        logger.info(f"  Recall:    {test_metrics.get(rec_key, 'N/A'):.4f}")
        logger.info(f"  F1 Score:  {test_metrics.get(f1_key, 'N/A'):.4f}")
        logger.info(f"  ROC-AUC:   {test_metrics.get(auc_key, 'N/A'):.4f}")

    logger.info(f"\nResults saved to: {output_dir}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
