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
from torch.utils.data import DataLoader, TensorDataset
import anndata as ad
from accelerate import Accelerator

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

    # Extract features and labels
    X_train = np.asarray(train_data.X, dtype=np.float32)
    y_train = train_data.obs["label"].values.astype(np.int64)

    X_val = np.asarray(val_data.X, dtype=np.float32)
    y_val = val_data.obs["label"].values.astype(np.int64)

    X_test = np.asarray(test_data.X, dtype=np.float32)
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
        self.labels = torch.from_numpy(labels).long()
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
) -> tuple:
    """Create PyTorch dataloaders with scGPT tokenization."""
    logger.info(
        f"Creating scGPT dataloaders with batch_size={batch_size}, max_len={max_len}"
    )

    train_dataset = scGPTDataset(X_train, y_train, gene_names, scgpt_wrapper, max_len)
    val_dataset = scGPTDataset(X_val, y_val, gene_names, scgpt_wrapper, max_len)
    test_dataset = scGPTDataset(X_test, y_test, gene_names, scgpt_wrapper, max_len)

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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_accelerate: bool = True,
    ):
        """Initialize trainer."""
        self.use_accelerate = use_accelerate

        # Initialize Accelerator
        if self.use_accelerate:
            self.accelerator = Accelerator()
            logger.info(f"Initializing scGPTTrainer with Accelerate")
            logger.info(f"  Device: {self.accelerator.device}")
            logger.info(f"  Process index: {self.accelerator.process_index}")
            logger.info(f"  Number of processes: {self.accelerator.num_processes}")
        else:
            self.accelerator = None
            self.device = device
            logger.info(f"Initializing scGPTTrainer on device: {device}")
            model = model.to(device)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Prepare model, optimizer, and scheduler with Accelerate
        if self.use_accelerate:
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )
            logger.info("Model, optimizer, and scheduler prepared with Accelerate")
        else:
            self.device = device

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0

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

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        return avg_loss, avg_accuracy

    def validate(self, val_loader) -> tuple:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

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

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        return avg_loss, avg_accuracy

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 15,
        patience: int = 3,
    ) -> dict:
        """Train model with early stopping."""
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        if not self.use_accelerate or self.accelerator.is_main_process:
            logger.info(f"Starting fine-tuning: {num_epochs} epochs, patience={patience}")

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            if not self.use_accelerate or self.accelerator.is_main_process:
                logger.info(
                    f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

            # Early stopping
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
                            f"Early stopping triggered (patience {patience} reached)"
                        )
                    break

        return history


def parse_arguments():
    """Parse command-line arguments for scGPT fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune scGPT foundation model for oligodendrocyte AD classification")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with preprocessed data (train.h5ad, val.h5ad, test.h5ad)")
    parser.add_argument("--output-dir", type=str, default="./results/scgpt", help="Directory to save results and checkpoint")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for fine-tuning (default: 1e-5)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for regularization (default: 0.01)")
    parser.add_argument("--n-bins", type=int, default=51, help="Number of expression bins (default: 51)")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension (default: 512)")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers (default: 12)")
    parser.add_argument("--freeze-layers", type=int, default=6, help="Number of layers to freeze (default: 6)")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps for scheduler (default: 500)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3)")
    parser.add_argument("--use-accelerate", action="store_true", default=False, help="Use Accelerate for distributed training (default: False)")
    parser.add_argument("--no-accelerate", dest="use_accelerate", action="store_false", help="Disable Accelerate and use single GPU/CPU")
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
            pretrained_path=None,  # No pretrained weights unless provided
            vocab_path=None,  # Create vocabulary from dataset genes
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
    except ImportError as e:
        logger.error(f"Failed to initialize scGPT: {e}")
        logger.error("Install scGPT with: pip install scgpt")
        return False

    # Step 3: Create dataloaders
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Creating scGPT-format Dataloaders")
    logger.info("=" * 80)
    max_len = min(2048, X_train.shape[1])

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        gene_names, model,
        batch_size=args.batch_size,
        max_len=max_len,
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

    trainer = scGPTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_accelerate=args.use_accelerate,
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
        },
        "training_history": {
            "train_loss": history["train_loss"],
            "train_accuracy": history["train_accuracy"],
            "val_loss": history["val_loss"],
            "val_accuracy": history["val_accuracy"],
        },
        "test_metrics": test_metrics,
        "pretrained_used": False,
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
