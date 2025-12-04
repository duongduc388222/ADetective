#!/usr/bin/env python3
"""
Phase 5: Fine-tune scGPT foundation model for AD pathology classification.

This script:
1. Loads preprocessed datasets from Phase 2
2. Creates scGPT-format tokenized dataloaders
3. Initializes scGPT wrapper with actual scGPT library
4. Fine-tunes with layer freezing strategy
5. Evaluates on test set
6. Saves results and trained model
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import anndata as ad
from accelerate import Accelerator
import scipy.sparse
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.scgpt_wrapper import scGPTWrapper, scGPTFineTuner
from src.eval.metrics import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(data_dir: Path) -> tuple:
    """Load preprocessed h5ad files from Phase 2."""
    logger.info(f"Loading preprocessed data from {data_dir}")

    train_path = data_dir / "train.h5ad"
    val_path = data_dir / "val.h5ad"
    test_path = data_dir / "test.h5ad"

    # Check files exist
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

    # Load h5ad files
    logger.info(f"Loading train data from {train_path}")
    train_data = ad.read_h5ad(train_path)

    logger.info(f"Loading val data from {val_path}")
    val_data = ad.read_h5ad(val_path)

    logger.info(f"Loading test data from {test_path}")
    test_data = ad.read_h5ad(test_path)

    # Extract features and labels (handle sparse matrices for numpy 2.x compatibility)
    X_train = (train_data.X.toarray() if scipy.sparse.issparse(train_data.X) else train_data.X).astype(np.float32)
    y_train = train_data.obs["label"].values.astype(np.int64)

    X_val = (val_data.X.toarray() if scipy.sparse.issparse(val_data.X) else val_data.X).astype(np.float32)
    y_val = val_data.obs["label"].values.astype(np.int64)

    X_test = (test_data.X.toarray() if scipy.sparse.issparse(test_data.X) else test_data.X).astype(np.float32)
    y_test = test_data.obs["label"].values.astype(np.int64)

    logger.info(f"Train: {X_train.shape}, {y_train.shape}")
    logger.info(f"Val:   {X_val.shape}, {y_val.shape}")
    logger.info(f"Test:  {X_test.shape}, {y_test.shape}")

    # Get gene names
    gene_names = list(train_data.var_names)
    logger.info(f"Number of genes: {len(gene_names)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, gene_names


def tokenize_batch(
    expression_data: np.ndarray,
    gene_names: list,
    scgpt_wrapper: scGPTWrapper,
    max_len: int = 2048,
) -> dict:
    """
    Tokenize a batch of expression data for scGPT.

    Args:
        expression_data: (batch_size, num_genes) expression matrix
        gene_names: List of gene names
        scgpt_wrapper: scGPT wrapper with tokenization methods
        max_len: Maximum sequence length

    Returns:
        Dict with tokenized batch data
    """
    batch_tokens = {
        "gene_ids": [],
        "values": [],
        "padding_mask": [],
    }

    for i in range(expression_data.shape[0]):
        expr_values = expression_data[i]

        # Prepare expression data in scGPT format
        expression_dict = {
            "gene_names": gene_names,
            "values": expr_values,
        }

        # Tokenize using wrapper
        tokens = scgpt_wrapper.tokenize_and_pad_batch(
            expression_dict, max_len=max_len
        )

        batch_tokens["gene_ids"].append(tokens["gene_ids"])
        batch_tokens["values"].append(tokens["values"])
        batch_tokens["padding_mask"].append(tokens["padding_mask"])

    # Stack into batch tensors
    batch_tokens["gene_ids"] = torch.stack(batch_tokens["gene_ids"])
    batch_tokens["values"] = torch.stack(batch_tokens["values"])
    batch_tokens["padding_mask"] = torch.stack(batch_tokens["padding_mask"])

    return batch_tokens


class scGPTDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for scGPT tokenized data."""

    def __init__(
        self, expression_data: np.ndarray, labels: np.ndarray,
        gene_names: list, scgpt_wrapper: scGPTWrapper,
        max_len: int = 2048,
    ):
        """Initialize dataset."""
        self.expression_data = expression_data
        # Use torch.tensor() instead of torch.from_numpy() for NumPy 2.x compatibility
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.gene_names = gene_names
        self.scgpt_wrapper = scgpt_wrapper
        self.max_len = max_len

    def __len__(self):
        return self.expression_data.shape[0]

    def __getitem__(self, idx):
        """Get a single sample."""
        expr_values = self.expression_data[idx]

        # Tokenize single sample
        expression_dict = {
            "gene_names": self.gene_names,
            "values": expr_values,
        }
        tokens = self.scgpt_wrapper.tokenize_and_pad_batch(
            expression_dict, max_len=self.max_len
        )

        return {
            "gene_ids": tokens["gene_ids"],
            "values": tokens["values"],
            "padding_mask": tokens["padding_mask"],
            "label": self.labels[idx],
        }


def collate_scgpt_batch(batch):
    """Collate function for scGPT dataloaders."""
    return {
        "gene_ids": torch.stack([b["gene_ids"] for b in batch]),
        "values": torch.stack([b["values"] for b in batch]),
        "padding_mask": torch.stack([b["padding_mask"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
    }


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gene_names: list,
    scgpt_wrapper: scGPTWrapper,
    batch_size: int = 16,
    max_len: int = 2048,
    use_weighted_sampling: bool = True,
) -> tuple:
    """Create PyTorch dataloaders with scGPT tokenization and weighted sampling for imbalanced data."""
    logger.info(
        f"Creating scGPT dataloaders with batch_size={batch_size}, max_len={max_len}"
    )

    train_dataset = scGPTDataset(X_train, y_train, gene_names, scgpt_wrapper, max_len)
    val_dataset = scGPTDataset(X_val, y_val, gene_names, scgpt_wrapper, max_len)
    test_dataset = scGPTDataset(X_test, y_test, gene_names, scgpt_wrapper, max_len)

    # AD-specific: Use weighted sampling for imbalanced dataset
    if use_weighted_sampling:
        # Calculate class weights for weighted sampling
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(y_train),
            replacement=True
        )
        logger.info(f"Using weighted sampling: class weights = {class_weights}")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_scgpt_batch,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_scgpt_batch,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_scgpt_batch,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_scgpt_batch,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


class scGPTTrainer:
    """Trainer for scGPT fine-tuning."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_labels: np.ndarray,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_accelerate: bool = True,
        accelerator=None,
        eval_metric: str = "f1",
    ):
        """Initialize trainer."""
        self.use_accelerate = use_accelerate
        self.eval_metric = eval_metric  # Track metric for early stopping

        # Initialize or use provided Accelerator
        if self.use_accelerate:
            if accelerator is not None:
                # Use provided accelerator (already prepared everything)
                self.accelerator = accelerator
                logger.info(f"Using provided Accelerator instance")
                logger.info(f"  Device: {self.accelerator.device}")
                logger.info(f"  Process index: {self.accelerator.process_index}")
                logger.info(f"  Number of processes: {self.accelerator.num_processes}")
                # Model/optimizer/scheduler already prepared in main()
                self.model = model
                self.optimizer = optimizer
                self.scheduler = scheduler
            else:
                # Fallback: create new accelerator and prepare here
                self.accelerator = Accelerator()
                logger.info(f"Created new Accelerator instance in trainer")
                logger.info(f"  Device: {self.accelerator.device}")
                logger.info(f"  Process index: {self.accelerator.process_index}")
                logger.info(f"  Number of processes: {self.accelerator.num_processes}")
                self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                    model, optimizer, scheduler
                )
                logger.info("Model, optimizer, and scheduler prepared with Accelerate")
        else:
            self.accelerator = None
            self.device = device
            logger.info(f"Initializing scGPTTrainer on device: {device}")
            self.model = model.to(device)
            self.optimizer = optimizer
            self.scheduler = scheduler

        # Calculate class weights for imbalanced dataset
        pos_count = (train_labels == 1).sum()
        neg_count = (train_labels == 0).sum()
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)

        logger.info(f"Class distribution: Positive={pos_count}, Negative={neg_count}")
        logger.info(f"Positive class weight: {pos_weight.item():.4f}")

        # Move pos_weight to correct device
        if self.use_accelerate:
            pos_weight = pos_weight.to(self.accelerator.device)
        else:
            pos_weight = pos_weight.to(device)

        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        logger.info(f"Using '{eval_metric}' metric for early stopping")

    def train_epoch(self, train_loader, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # No need to move to device with Accelerate - it handles this
            if not self.use_accelerate:
                gene_ids = batch["gene_ids"].to(self.device)
                values = batch["values"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                labels = batch["labels"].to(self.device).float().unsqueeze(1)
            else:
                gene_ids = batch["gene_ids"]
                values = batch["values"]
                padding_mask = batch["padding_mask"]
                labels = batch["labels"].float().unsqueeze(1)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(
                gene_ids=gene_ids,
                values=values,
                src_key_padding_mask=padding_mask,
            )
            loss = self.criterion(logits, labels)

            # Backward pass with Accelerate
            if self.use_accelerate:
                self.accelerator.backward(loss)
            else:
                loss.backward()

            # Gradient clipping
            if self.use_accelerate:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # Metrics
            with torch.no_grad():
                preds = (logits > 0).float()
                accuracy = (preds == labels).float().mean()

                # Diagnostic logging for oscillating accuracy detection
                if (batch_idx + 1) % 50 == 0:
                    logit_stats = {
                        'min': logits.min().item(),
                        'max': logits.max().item(),
                        'mean': logits.mean().item(),
                        'std': logits.std().item()
                    }
                    pred_dist = preds.sum().item() / len(preds)
                    label_dist = labels.sum().item() / len(labels)

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1

            # Logging (only on main process when using Accelerate)
            if (batch_idx + 1) % 50 == 0:
                if not self.use_accelerate or self.accelerator.is_main_process:
                    logger.info(
                        f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f} | Acc: {accuracy.item():.4f}"
                    )
                    logger.info(
                        f"  Logits: min={logit_stats['min']:.3f}, max={logit_stats['max']:.3f}, "
                        f"mean={logit_stats['mean']:.3f}, std={logit_stats['std']:.3f} | "
                        f"Pred class 1: {pred_dist:.2%} | True class 1: {label_dist:.2%}"
                    )

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        return avg_loss, avg_accuracy

    def validate(self, val_loader) -> tuple:
        """Validate on validation set and compute F1 score for AD classification."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                # No need to move to device with Accelerate - it handles this
                if not self.use_accelerate:
                    gene_ids = batch["gene_ids"].to(self.device)
                    values = batch["values"].to(self.device)
                    padding_mask = batch["padding_mask"].to(self.device)
                    labels = batch["labels"].to(self.device).float().unsqueeze(1)
                else:
                    gene_ids = batch["gene_ids"]
                    values = batch["values"]
                    padding_mask = batch["padding_mask"]
                    labels = batch["labels"].float().unsqueeze(1)

                logits = self.model(
                    gene_ids=gene_ids,
                    values=values,
                    src_key_padding_mask=padding_mask,
                )
                loss = self.criterion(logits, labels)
                preds = (logits > 0).float()
                accuracy = (preds == labels).float().mean()

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1

                # Collect predictions for F1 score computation
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        # Compute F1 score for binary classification
        all_preds = np.concatenate(all_preds, axis=0).flatten().astype(int)
        all_labels = np.concatenate(all_labels, axis=0).flatten().astype(int)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)

        return avg_loss, avg_accuracy, val_f1

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 15,
        patience: int = 3,
    ) -> dict:
        """Train model with early stopping based on selected metric (F1 or loss)."""
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }

        if not self.use_accelerate or self.accelerator.is_main_process:
            logger.info(f"Starting fine-tuning: {num_epochs} epochs, patience={patience}")
            logger.info(f"Early stopping metric: {self.eval_metric}")

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            # Validate (now returns loss, accuracy, AND F1)
            val_loss, val_acc, val_f1 = self.validate(val_loader)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["val_f1"].append(val_f1)

            if not self.use_accelerate or self.accelerator.is_main_process:
                logger.info(
                    f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
                )

            # Early stopping based on selected metric
            if self.eval_metric == "f1":
                # Maximize F1 score
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.patience_counter = 0
                    if not self.use_accelerate or self.accelerator.is_main_process:
                        logger.info(f"  → New best F1 score: {val_f1:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= patience:
                        if not self.use_accelerate or self.accelerator.is_main_process:
                            logger.info(
                                f"Early stopping triggered (patience {patience} reached, best F1: {self.best_val_f1:.4f})"
                            )
                        break
            else:
                # Minimize loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if not self.use_accelerate or self.accelerator.is_main_process:
                        logger.info(f"  → New best validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= patience:
                        if not self.use_accelerate or self.accelerator.is_main_process:
                            logger.info(
                                f"Early stopping triggered (patience {patience} reached, best loss: {self.best_val_loss:.4f})"
                            )
                        break

        return history


def parse_arguments():
    """Parse command-line arguments for scGPT fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune scGPT foundation model for oligodendrocyte AD classification")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with preprocessed data (train.h5ad, val.h5ad, test.h5ad)")
    parser.add_argument("--output-dir", type=str, default="./results/scgpt", help="Directory to save results and checkpoint")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate for fine-tuning (default: 5e-6, optimized for AD)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for regularization (default: 0.01)")
    parser.add_argument("--n-bins", type=int, default=51, help="Number of expression bins (default: 51)")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension (default: 512)")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers (default: 12)")
    parser.add_argument("--freeze-layers", type=int, default=8, help="Number of layers to freeze - AD optimized (default: 8)")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps for scheduler - AD optimized (default: 1000)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience - AD optimized (default: 5)")
    parser.add_argument("--max-sequence-length", type=int, default=1200, help="Max sequence length for tokens - AD optimized (default: 1200)")
    parser.add_argument("--eval-metric", type=str, default="f1", choices=["f1", "loss"], help="Metric for best model selection (default: f1)")
    parser.add_argument("--use-weighted-sampling", action="store_true", default=True, help="Use weighted sampling for imbalanced data (default: True)")
    parser.add_argument("--focal-loss-gamma", type=float, default=0.0, help="Focal loss gamma for hard example mining (default: 0.0, use 2.0 for AD)")
    parser.add_argument("--expression-noise-std", type=float, default=0.0, help="Std dev of Gaussian noise for augmentation (default: 0.0)")
    parser.add_argument("--use-accelerate", action="store_true", default=False, help="Use Accelerate for distributed training (default: False)")
    parser.add_argument("--no-accelerate", dest="use_accelerate", action="store_false", help="Disable Accelerate and use single GPU/CPU")
    parser.add_argument("--pretrained-path", type=str, default=None, help="Path to pretrained scGPT checkpoint (.pt file)")
    parser.add_argument("--vocab-path", type=str, default=None, help="Path to pretrained gene vocabulary (vocab.json)")
    return parser.parse_args()


def main():
    """Fine-tune scGPT model."""
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("PHASE 5: SCGPT FOUNDATION MODEL FINE-TUNING (WITH ACTUAL scGPT LIBRARY)")
    logger.info("=" * 80)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nData directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Load data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Preprocessed Data")
    logger.info("=" * 80)
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, gene_names = load_data(data_dir)
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        logger.error("Make sure you've run scripts/load.py and scripts/prepare_data.py first")
        return False

    # Step 2: Initialize scGPT wrapper
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Initializing scGPT Wrapper")
    logger.info("=" * 80)

    try:
        model = scGPTWrapper(
            gene_names=gene_names,
            pretrained_path=args.pretrained_path,  # Load pretrained weights if provided
            vocab_path=args.vocab_path,  # Load pretrained vocabulary if provided
            n_bins=args.n_bins,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            freeze_layers=args.freeze_layers,
            dropout=0.1,
            do_mvc=False,  # Disabled for classification fine-tuning
            do_dab=False,
            do_ecs=False,
            use_batch_labels=False,
            explicit_zero_prob=False,
            use_fast_transformer=True,
        )
        logger.info("✓ scGPT model initialized successfully")

        # Log pretrained weight usage
        if args.pretrained_path:
            logger.info(f"  Loaded pretrained checkpoint from: {args.pretrained_path}")
        else:
            logger.info("  Training from scratch (no pretrained weights)")
        if args.vocab_path:
            logger.info(f"  Loaded pretrained vocabulary from: {args.vocab_path}")
        else:
            logger.info("  Created vocabulary from dataset genes")
    except ImportError as e:
        logger.error(f"Failed to initialize scGPT: {e}")
        logger.error("Install scGPT with: pip install scgpt")
        return False

    # Step 3: Create dataloaders
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Creating scGPT-format Dataloaders")
    logger.info("=" * 80)
    max_len = min(args.max_sequence_length, X_train.shape[1])
    logger.info(f"Using max sequence length: {max_len} (default: {args.max_sequence_length}, available genes: {X_train.shape[1]})")

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        gene_names, model,
        batch_size=args.batch_size,
        max_len=max_len,
        use_weighted_sampling=args.use_weighted_sampling,
    )

    # Step 4: Create fine-tuner
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Creating Fine-tuner")
    logger.info("=" * 80)

    finetuner = scGPTFineTuner(
        model=model,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
    )

    # Create optimizer and scheduler
    num_training_steps = len(X_train) // args.batch_size * args.epochs
    optimizer, scheduler = finetuner.create_optimizer_and_scheduler(num_training_steps)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Using Accelerate: {args.use_accelerate}")

    # Initialize Accelerator and prepare everything if using Accelerate
    accelerator_instance = None
    if args.use_accelerate:
        logger.info("\n" + "=" * 80)
        logger.info("Preparing Components with Accelerate")
        logger.info("=" * 80)
        accelerator_instance = Accelerator()
        logger.info(f"Initialized Accelerator on device: {accelerator_instance.device}")

        # Prepare ALL components together - this ensures dataloaders are on correct device
        model, optimizer, scheduler, train_loader, val_loader, test_loader = accelerator_instance.prepare(
            model, optimizer, scheduler, train_loader, val_loader, test_loader
        )
        logger.info("✓ Model, optimizer, scheduler, and dataloaders prepared with Accelerate")

    trainer = scGPTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_labels=y_train,
        device=device,
        use_accelerate=args.use_accelerate,
        accelerator=accelerator_instance,
        eval_metric=args.eval_metric,  # AD-optimized: use F1 for early stopping
    )

    # Step 5: Fine-tune model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Fine-tuning Model")
    logger.info("=" * 80)
    history = trainer.fit(train_loader, val_loader, num_epochs=args.epochs, patience=args.patience)

    # Save checkpoint (unwrap model if using Accelerate)
    checkpoint_path = output_dir / "checkpoint.pt"
    if args.use_accelerate:
        model_to_save = trainer.accelerator.unwrap_model(trainer.model)
    else:
        model_to_save = model
    torch.save(model_to_save.state_dict(), checkpoint_path)
    logger.info(f"Saved model checkpoint to {checkpoint_path}")

    # Step 6: Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Test Set Evaluation")
    logger.info("=" * 80)
    # Use unwrapped model for evaluation
    eval_model = model_to_save
    evaluator = ModelEvaluator(eval_model, device=device)
    test_metrics = evaluator.evaluate(test_loader, dataset_name="test")

    # Classification report
    evaluator.get_classification_report(test_loader, dataset_name="test")

    # Step 7: Save results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Saving Results")
    logger.info("=" * 80)

    results = {
        "config": {
            "num_genes": len(gene_names),
            "n_bins": args.n_bins,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "freeze_layers": args.freeze_layers,
            "batch_size": args.batch_size,
            "max_sequence_length": max_len,
            "model_type": "scGPT (actual library)",
            # AD-specific configurations
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "patience": args.patience,
            "eval_metric": args.eval_metric,
            "use_weighted_sampling": args.use_weighted_sampling,
        },
        "training_history": {
            "train_loss": history["train_loss"],
            "train_accuracy": history["train_accuracy"],
            "val_loss": history["val_loss"],
            "val_accuracy": history["val_accuracy"],
            "val_f1": history["val_f1"],  # AD-specific: F1 score for early stopping
        },
        "test_metrics": test_metrics,
        # Pretrained weight tracking
        "pretrained_used": args.pretrained_path is not None,
        "pretrained_checkpoint": args.pretrained_path if args.pretrained_path else "None (trained from scratch)",
        "pretrained_vocab": args.vocab_path if args.vocab_path else "Created from dataset genes",
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINE-TUNING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nFinal Test Performance:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Model checkpoint: {checkpoint_path}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
