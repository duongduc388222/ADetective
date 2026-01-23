#!/usr/bin/env python3
"""
Train CellFM classifier for oligodendrocyte AD classification.

Finetunes CellFM foundation model backbone with a classification head.
Supports 4-class ADNC classification (Not AD, Low, Intermediate, High)
and cross-region validation.

This script:
1. Loads vocab-aligned h5ad files (prepared with --foundation-model-mode)
2. Creates PyTorch dataloaders with sparse-to-dense conversion
3. Loads CellFM pretrained backbone
4. Initializes CellFMClassifier with classification head
5. Trains using PyTorch Lightning (or Accelerate if --use-accelerate)
6. Evaluates on test set
7. Saves results and checkpoints

Usage:
    # Same-region training (single data directory)
    python scripts/train_cellfm.py \\
        --data-dir ./data/cellfm_prepared \\
        --backbone-path /path/to/cellfm_weights.pt \\
        --output-dir ./results/cellfm

    # Cross-region training (separate train and test directories)
    python scripts/train_cellfm.py \\
        --train-dir ./data/cellfm_mtg \\
        --test-dir ./data/cellfm_a9 \\
        --backbone-path /path/to/cellfm_weights.pt \\
        --output-dir ./results/cellfm_mtg_to_a9

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import anndata as ad
from scipy import sparse

# Import metadata processor for --use-metadata option
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.metadata_processor import MetadataProcessor
    METADATA_AVAILABLE = True
except ImportError:
    METADATA_AVAILABLE = False

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
    Optionally includes processed metadata features.
    """

    def __init__(
        self,
        h5ad_path: str,
        metadata_array: Optional[np.ndarray] = None,
    ):
        """
        Initialize dataset from h5ad file.

        Args:
            h5ad_path: Path to h5ad file with gene expression and labels
            metadata_array: Optional pre-processed metadata array (n_cells, n_metadata_features)
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

        # Store metadata if provided
        self.metadata = metadata_array
        self.use_metadata = metadata_array is not None
        if self.use_metadata:
            assert len(metadata_array) == self.n_cells, \
                f"Metadata length ({len(metadata_array)}) must match n_cells ({self.n_cells})"
            self.n_metadata = metadata_array.shape[1]
            logger.info(f"  Metadata features: {self.n_metadata}")

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            If use_metadata: Tuple of (gene_expression, metadata, label)
            Else: Tuple of (gene_expression, label)
        """
        # Get gene expression - convert sparse to dense
        if self.is_sparse:
            x = np.asarray(self.adata.X[idx].todense()).flatten().astype(np.float32)
        else:
            x = np.asarray(self.adata.X[idx]).flatten().astype(np.float32)

        # Get label
        y = int(self.adata.obs["label"].iloc[idx])

        if self.use_metadata:
            meta = self.metadata[idx].astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(meta), torch.tensor(y, dtype=torch.long)
        else:
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
    CellFM backbone + MLP classification head with optional metadata fusion.

    Architecture:
        backbone(x) -> cell_embedding (B, hidden_dim)
        [Optional: concat(cell_embedding, metadata) -> features (B, hidden_dim + n_metadata)]
        classifier(features) -> logits (B, num_classes)

    Classification head structure:
        Linear(hidden_dim [+ n_metadata], head_hidden)
        -> LayerNorm(head_hidden)
        -> ReLU
        -> Dropout(dropout)
        -> Linear(head_hidden, num_classes)
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 1536,
        num_classes: int = 4,
        head_hidden: int = 512,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        n_metadata: int = 0,
    ):
        """
        Initialize CellFM classifier.

        Args:
            backbone: Pretrained CellFM encoder
            hidden_dim: Output dimension of backbone (default: 1536)
            num_classes: Number of output classes (default: 4 for ADNC)
            head_hidden: Hidden dimension in classification head
            dropout: Dropout rate in classification head
            freeze_backbone: If True, freeze backbone parameters
            n_metadata: Number of metadata features to concatenate (0 = no metadata)
        """
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.freeze_backbone = freeze_backbone
        self.n_metadata = n_metadata
        self.use_metadata = n_metadata > 0

        # Freeze backbone if requested
        if freeze_backbone:
            logger.info("Freezing backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Set to eval mode for frozen backbone
            self.backbone.eval()

        # Classification head input dimension: cell embedding + optional metadata
        classifier_input_dim = hidden_dim + n_metadata

        if self.use_metadata:
            logger.info(f"Metadata fusion enabled: {n_metadata} features")
            logger.info(f"  Classifier input: {hidden_dim} (embedding) + {n_metadata} (metadata) = {classifier_input_dim}")

        # Classification head: Linear -> LayerNorm -> ReLU -> Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, head_hidden),
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

    def forward(
        self,
        x: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Gene expression tensor (batch_size, n_genes)
            metadata: Optional metadata tensor (batch_size, n_metadata)

        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        # Get features from backbone
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        # Concatenate metadata if provided
        if self.use_metadata:
            if metadata is None:
                raise ValueError("Model was initialized with n_metadata>0 but no metadata provided")
            features = torch.cat([features, metadata], dim=1)

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
        label_smoothing: float = 0.0,
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
            label_smoothing: Label smoothing factor (default: 0.0)
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

        # Loss function with optional label smoothing
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

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
        from tqdm import tqdm

        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        training_config = self.config.get("training", {})
        gradient_clip = training_config.get("gradient_clipping", 1.0)
        log_frequency = self.config.get("logging", {}).get("log_frequency", 50)

        # Create progress bar (only on main process)
        if self._is_main_process():
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch+1}",
                leave=True,
                ncols=100,
            )
        else:
            pbar = enumerate(train_loader)

        for batch_idx, batch in pbar:
            # Unpack batch - may have 2 or 3 elements depending on metadata
            if len(batch) == 3:
                x, metadata, y = batch
            else:
                x, y = batch
                metadata = None

            # Move to device if not using Accelerate
            if not self.use_accelerate:
                x = x.to(self.device)
                y = y.to(self.device)
                if metadata is not None:
                    metadata = metadata.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x, metadata=metadata)
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
            batch_loss = loss.item()
            total_loss += batch_loss
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Update progress bar with current loss
            if self._is_main_process() and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })

        # Close progress bar
        if self._is_main_process() and hasattr(pbar, 'close'):
            pbar.close()

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def validate(self, val_loader, num_classes: int = 4) -> Tuple[float, float, Dict]:
        """
        Validate model.

        Args:
            val_loader: Validation dataloader
            num_classes: Number of classes for multi-class metrics

        Returns:
            Tuple of (average_loss, accuracy, metrics_dict)
        """
        from tqdm import tqdm

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        # Create progress bar for validation (only on main process)
        if self._is_main_process():
            pbar = tqdm(
                val_loader,
                total=len(val_loader),
                desc="Validating",
                leave=False,
                ncols=100,
            )
        else:
            pbar = val_loader

        with torch.no_grad():
            for batch in pbar:
                # Unpack batch - may have 2 or 3 elements depending on metadata
                if len(batch) == 3:
                    x, metadata, y = batch
                else:
                    x, y = batch
                    metadata = None

                if not self.use_accelerate:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    if metadata is not None:
                        metadata = metadata.to(self.device)

                logits = self.model(x, metadata=metadata)
                loss = self.criterion(logits, y)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)  # (batch, num_classes)

                all_preds.extend(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # Close progress bar
        if self._is_main_process() and hasattr(pbar, 'close'):
            pbar.close()

        # Concatenate all probabilities
        all_probs = np.vstack(all_probs)  # (N, num_classes)

        avg_loss = total_loss / len(val_loader)

        # Multi-class metrics
        label_names = ["Not AD", "Low", "Intermediate", "High"][:num_classes]

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "macro_f1": f1_score(all_labels, all_preds, average='macro', zero_division=0),
            "weighted_f1": f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            "macro_precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
            "macro_recall": recall_score(all_labels, all_preds, average='macro', zero_division=0),
            "per_class_f1": f1_score(all_labels, all_preds, average=None, zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        }

        # Add ROC-AUC (one-vs-rest for multi-class)
        try:
            if len(set(all_labels)) > 1:
                metrics["roc_auc_ovr"] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            metrics["roc_auc_ovr"] = 0.0

        # Add classification report
        metrics["classification_report"] = classification_report(
            all_labels, all_preds,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        # Use macro_f1 as the primary metric for consistency
        metrics["f1"] = metrics["macro_f1"]

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
    Supports multi-class (4-class ADNC) classification.
    """

    def __init__(
        self,
        model: CellFMClassifier,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        class_weights: Optional[torch.Tensor] = None,
        num_classes: int = 4,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize Lightning module.

        Args:
            model: CellFMClassifier instance
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for AdamW
            class_weights: Optional class weights for imbalanced data
            num_classes: Number of classes (default: 4 for ADNC)
            label_smoothing: Label smoothing factor (default: 0.0)
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        # Store class weights as buffer (not a parameter)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # Loss function with optional class weights and label smoothing
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=label_smoothing)

        # Metrics - multi-class configuration
        metric_task = "multiclass"
        metric_kwargs = {"task": metric_task, "num_classes": num_classes}

        # Training metrics
        self.train_acc = Accuracy(**metric_kwargs)

        # Validation metrics
        self.val_acc = Accuracy(**metric_kwargs)
        self.val_precision = Precision(**metric_kwargs, average='macro')
        self.val_recall = Recall(**metric_kwargs, average='macro')
        self.val_f1 = F1Score(**metric_kwargs, average='macro')
        self.val_auroc = AUROC(**metric_kwargs, average='macro')

        # Test metrics
        self.test_acc = Accuracy(**metric_kwargs)
        self.test_precision = Precision(**metric_kwargs, average='macro')
        self.test_recall = Recall(**metric_kwargs, average='macro')
        self.test_f1 = F1Score(**metric_kwargs, average='macro')
        self.test_auroc = AUROC(**metric_kwargs, average='macro')

        # Save hyperparameters (exclude model to avoid serialization issues)
        self.save_hyperparameters(ignore=["model", "class_weights"])

    def forward(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        return self.model(x, metadata=metadata)

    def training_step(self, batch, batch_idx):
        """Training step."""
        # Unpack batch - may have 2 or 3 elements depending on metadata
        if len(batch) == 3:
            x, metadata, y = batch
        else:
            x, y = batch
            metadata = None
        logits = self(x, metadata=metadata)
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
        # Unpack batch - may have 2 or 3 elements depending on metadata
        if len(batch) == 3:
            x, metadata, y = batch
        else:
            x, y = batch
            metadata = None
        logits = self(x, metadata=metadata)
        loss = self.criterion(logits, y)

        # Compute predictions and probabilities
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)  # (batch, num_classes) for multi-class

        # Update metrics
        self.val_acc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)
        self.val_auroc(probs, y)  # Multi-class AUROC expects full probability matrix

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_auroc", self.val_auroc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        # Unpack batch - may have 2 or 3 elements depending on metadata
        if len(batch) == 3:
            x, metadata, y = batch
        else:
            x, y = batch
            metadata = None
        logits = self(x, metadata=metadata)
        loss = self.criterion(logits, y)

        # Compute predictions and probabilities
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)  # (batch, num_classes) for multi-class

        # Update metrics
        self.test_acc(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_f1(preds, y)
        self.test_auroc(probs, y)  # Multi-class AUROC expects full probability matrix

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
        default=None,
        help="Directory with train.h5ad, val.h5ad, test.h5ad (same-region mode)",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Training data directory (cross-region mode: contains train.h5ad, val.h5ad)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Test data directory (cross-region mode: contains train.h5ad to use as test set)",
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
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of output classes (default: 4 for ADNC)",
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
        default=1e-5,
        help="Learning rate for classifier head (default: 1e-5, optimized for finetuning)",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Learning rate for backbone (if different from --learning-rate). "
             "Default: 1/10th of --learning-rate for full finetuning",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs (default: 20)",
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
        default=7,
        help="Early stopping patience (default: 7)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate scheduler (default: 100)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for CrossEntropyLoss (default: 0.1)",
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

    # Metadata arguments
    parser.add_argument(
        "--use-metadata",
        action="store_true",
        default=False,
        help="Include donor metadata (APOE, Sex, Age) as additional features (default: False)",
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

    # Validate data directory arguments
    cross_region_mode = args.train_dir is not None and args.test_dir is not None
    same_region_mode = args.data_dir is not None

    if not cross_region_mode and not same_region_mode:
        logger.error("Either --data-dir OR both --train-dir and --test-dir must be provided")
        return False

    if cross_region_mode and same_region_mode:
        logger.warning("Both --data-dir and --train-dir/--test-dir provided. Using cross-region mode.")

    # Setup paths
    if cross_region_mode:
        train_dir = Path(args.train_dir)
        test_dir = Path(args.test_dir)
        data_dir = train_dir  # For logging compatibility
        logger.info("\n*** CROSS-REGION MODE ***")
        logger.info(f"  Training region: {train_dir}")
        logger.info(f"  Test region: {test_dir}")
    else:
        data_dir = Path(args.data_dir)
        train_dir = data_dir
        test_dir = data_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nConfiguration:")
    if cross_region_mode:
        logger.info(f"  Train directory: {train_dir}")
        logger.info(f"  Test directory: {test_dir}")
    else:
        logger.info(f"  Data directory: {data_dir}")
    logger.info(f"  Backbone path: {args.backbone_path}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Num classes: {args.num_classes}")
    logger.info(f"  Freeze backbone: {args.freeze_backbone}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    if args.backbone_lr:
        logger.info(f"  Backbone LR: {args.backbone_lr}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Early stopping patience: {args.patience}")
    logger.info(f"  Warmup steps: {args.warmup_steps}")
    logger.info(f"  Label smoothing: {args.label_smoothing}")
    logger.info(f"  Use Accelerate: {args.use_accelerate}")
    logger.info(f"  Use metadata: {args.use_metadata}")

    # Step 0: Check Flash Attention
    logger.info("\n" + "=" * 80)
    logger.info("STEP 0: Checking Flash Attention")
    logger.info("=" * 80)
    check_flash_attention()

    # Step 1: Load datasets
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Datasets")
    logger.info("=" * 80)

    # Initialize metadata variables
    n_metadata = 0
    train_meta, val_meta, test_meta = None, None, None
    metadata_processor = None

    try:
        if cross_region_mode:
            # Cross-region mode: train/val from train_dir, test from test_dir
            logger.info(f"Loading training data from: {train_dir}")
            train_h5ad_path = train_dir / "train.h5ad"
            val_h5ad_path = train_dir / "val.h5ad"

            logger.info(f"Loading test data from: {test_dir}")
            test_h5ad_path = test_dir / "train.h5ad"
            if not test_h5ad_path.exists():
                test_h5ad_path = test_dir / "test.h5ad"
        else:
            # Same-region mode: all from data_dir
            train_h5ad_path = data_dir / "train.h5ad"
            val_h5ad_path = data_dir / "val.h5ad"
            test_h5ad_path = data_dir / "test.h5ad"

        # Process metadata if requested
        if args.use_metadata:
            if not METADATA_AVAILABLE:
                logger.error("Metadata processing not available. Install src.data.metadata_processor")
                return False

            logger.info("\n  Processing metadata features...")
            metadata_processor = MetadataProcessor()

            # Load h5ad files to access obs for metadata
            train_adata = ad.read_h5ad(train_h5ad_path)
            val_adata = ad.read_h5ad(val_h5ad_path)
            test_adata = ad.read_h5ad(test_h5ad_path)

            # Fit on training data, transform all
            train_meta = metadata_processor.fit_transform(train_adata.obs)
            val_meta = metadata_processor.transform(val_adata.obs)
            test_meta = metadata_processor.transform(test_adata.obs)

            n_metadata = metadata_processor.get_metadata_dim()
            metadata_features = metadata_processor.get_feature_names()

            logger.info(f"  Extracted {n_metadata} metadata features:")
            for i, feat in enumerate(metadata_features):
                logger.info(f"    {i+1}. {feat}")

            # Save metadata processor for reproducibility
            metadata_processor.save(output_dir / "metadata_processor.pkl")

            # Save metadata feature names
            import json
            metadata_info = {
                "metadata_dim": n_metadata,
                "feature_names": metadata_features,
            }
            with open(output_dir / "metadata_features.json", "w") as f:
                json.dump(metadata_info, f, indent=2)
            logger.info(f"  Saved metadata processor to {output_dir / 'metadata_processor.pkl'}")

        # Create datasets (with or without metadata)
        train_dataset = ScanpyDataset(train_h5ad_path, metadata_array=train_meta)
        val_dataset = ScanpyDataset(val_h5ad_path, metadata_array=val_meta)
        test_dataset = ScanpyDataset(test_h5ad_path, metadata_array=test_meta)

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
        num_classes=args.num_classes,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        n_metadata=n_metadata,
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
                "backbone_lr": args.backbone_lr if args.backbone_lr else args.learning_rate / 10,
                "weight_decay": args.weight_decay,
                "optimizer": "adamw",
                "epochs": args.epochs,
                "gradient_clipping": args.gradient_clip,
                "early_stopping": {"enabled": True, "patience": args.patience},
                "warmup": {"enabled": True, "steps": args.warmup_steps},
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
            label_smoothing=args.label_smoothing,
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
                "data_dir": str(data_dir) if not cross_region_mode else None,
                "train_dir": str(train_dir) if cross_region_mode else None,
                "test_dir": str(test_dir) if cross_region_mode else None,
                "cross_region_mode": cross_region_mode,
                "backbone_path": args.backbone_path,
                "num_classes": args.num_classes,
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
                "use_metadata": args.use_metadata,
                "n_metadata": n_metadata,
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
            num_classes=args.num_classes,
            label_smoothing=args.label_smoothing,
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
                "data_dir": str(data_dir) if not cross_region_mode else None,
                "train_dir": str(train_dir) if cross_region_mode else None,
                "test_dir": str(test_dir) if cross_region_mode else None,
                "cross_region_mode": cross_region_mode,
                "backbone_path": args.backbone_path,
                "num_classes": args.num_classes,
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
                "use_metadata": args.use_metadata,
                "n_metadata": n_metadata,
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
        logger.info(f"\nFinal Test Performance ({args.num_classes}-class):")
        # Handle both Lightning and Accelerate metric formats
        acc_key = "test_acc" if "test_acc" in test_metrics else "accuracy"
        prec_key = "test_precision" if "test_precision" in test_metrics else "macro_precision"
        rec_key = "test_recall" if "test_recall" in test_metrics else "macro_recall"
        f1_key = "test_f1" if "test_f1" in test_metrics else "macro_f1"
        auc_key = "test_auroc" if "test_auroc" in test_metrics else "roc_auc_ovr"

        acc_val = test_metrics.get(acc_key)
        prec_val = test_metrics.get(prec_key)
        rec_val = test_metrics.get(rec_key)
        f1_val = test_metrics.get(f1_key)
        auc_val = test_metrics.get(auc_key)

        logger.info(f"  Accuracy:        {acc_val:.4f}" if acc_val is not None else "  Accuracy: N/A")
        logger.info(f"  Macro Precision: {prec_val:.4f}" if prec_val is not None else "  Macro Precision: N/A")
        logger.info(f"  Macro Recall:    {rec_val:.4f}" if rec_val is not None else "  Macro Recall: N/A")
        logger.info(f"  Macro F1 Score:  {f1_val:.4f}" if f1_val is not None else "  Macro F1 Score: N/A")
        logger.info(f"  Macro ROC-AUC:   {auc_val:.4f}" if auc_val is not None else "  Macro ROC-AUC: N/A")

        # Show per-class F1 if available (Accelerate mode)
        if "per_class_f1" in test_metrics:
            label_names = ["Not AD", "Low", "Intermediate", "High"][:args.num_classes]
            logger.info(f"\n  Per-class F1:")
            for i, (name, f1) in enumerate(zip(label_names, test_metrics["per_class_f1"])):
                logger.info(f"    {name}: {f1:.4f}")

        # Show confusion matrix if available
        if "confusion_matrix" in test_metrics:
            logger.info(f"\n  Confusion Matrix:")
            cm = test_metrics["confusion_matrix"]
            label_names = ["Not AD", "Low", "Intermediate", "High"][:args.num_classes]
            logger.info(f"    Rows=True, Cols=Predicted")
            header = "           " + "  ".join([f"{n[:6]:>6}" for n in label_names])
            logger.info(header)
            for i, row in enumerate(cm):
                row_str = "  ".join([f"{v:>6}" for v in row])
                logger.info(f"    {label_names[i]:>6}  {row_str}")

    logger.info(f"\nResults saved to: {output_dir}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
