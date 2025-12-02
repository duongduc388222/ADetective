#!/usr/bin/env python3
"""
Phase 4: Train Transformer model for oligodendrocyte AD classification.

This script:
1. Loads preprocessed datasets from Phase 2
2. Creates PyTorch dataloaders
3. Initializes and trains Transformer model
4. Evaluates on test set
5. Saves results and trained model
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import FlashTransformer, TransformerTrainer
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

    # Extract features and labels (handle sparse matrices)
    def to_dense_array(X):
        """Convert matrix to dense numpy array (handles both sparse and dense)."""
        if hasattr(X, 'toarray'):
            # Sparse matrix
            return X.toarray().astype(np.float32)
        else:
            # Dense matrix
            return np.asarray(X, dtype=np.float32)

    X_train = to_dense_array(train_data.X)
    y_train = train_data.obs["label"].values.astype(np.int64)

    X_val = to_dense_array(val_data.X)
    y_val = val_data.obs["label"].values.astype(np.int64)

    X_test = to_dense_array(test_data.X)
    y_test = test_data.obs["label"].values.astype(np.int64)

    logger.info(f"Train: {X_train.shape}, {y_train.shape}")
    logger.info(f"Val:   {X_val.shape}, {y_val.shape}")
    logger.info(f"Test:  {X_test.shape}, {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> tuple:
    """Create PyTorch dataloaders."""
    logger.info(f"Creating dataloaders with batch size {batch_size}")

    def create_loader(X, y, shuffle=True):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = create_loader(X_train, y_train, shuffle=True)
    val_loader = create_loader(X_val, y_val, shuffle=False)
    test_loader = create_loader(X_test, y_test, shuffle=False)

    return train_loader, val_loader, test_loader


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
            logger.info("✓ GPU supports Flash Attention (compute capability >= 8.0)")
        else:
            logger.warning("✗ GPU does not support Flash Attention (compute capability < 8.0)")
    else:
        logger.warning("CUDA available: No")
        logger.warning("✗ Flash Attention requires CUDA GPU")

    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        logger.info("✓ PyTorch 2.0+ scaled_dot_product_attention available")
    else:
        logger.warning("✗ PyTorch 2.0+ required for automatic Flash Attention")

    logger.info("=" * 30)


def parse_arguments():
    """Parse command-line arguments for Transformer training."""
    parser = argparse.ArgumentParser(description="Train Transformer model for oligodendrocyte AD classification")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with preprocessed data (train.h5ad, val.h5ad, test.h5ad)")
    parser.add_argument("--output-dir", type=str, default="./results/transformer", help="Directory to save results and checkpoint")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for AdamW optimizer (default: 1e-4)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for regularization (default: 1e-4)")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension (default: 128)")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers (default: 3)")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="FFN dimension (default: 256)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value (default: 1.0)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--expression-scaling", type=str, default="multiplicative", choices=["multiplicative", "additive", "concatenate"], help="Expression scaling method")
    parser.add_argument("--use-accelerate", action="store_true", default=False, help="Use Accelerate for distributed training (default: False)")
    parser.add_argument("--no-accelerate", dest="use_accelerate", action="store_false", help="Disable Accelerate and use single GPU/CPU")
    return parser.parse_args()


def main():
    """Train Transformer model."""
    args = parse_arguments()
    logger.info("=" * 80)
    logger.info("PHASE 4: TRANSFORMER TRAINING")
    logger.info("=" * 80)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nData directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Step 0: Check Flash Attention
    logger.info("\n" + "=" * 80)
    logger.info("STEP 0: Checking Flash Attention")
    logger.info("=" * 80)
    check_flash_attention()

    # Step 1: Load data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Preprocessed Data")
    logger.info("=" * 80)
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir)
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        logger.error("Make sure you've run scripts/load.py and scripts/prepare_data.py first")
        return False

    # Step 2: Create dataloaders
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Creating Dataloaders")
    logger.info("=" * 80)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=args.batch_size
    )

    # Step 3: Initialize model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Initializing Transformer Model")
    logger.info("=" * 80)
    num_genes = X_train.shape[1]

    model = FlashTransformer(
        num_genes=num_genes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=min(num_genes + 1, 2048),  # +1 for CLS token
        use_cls_token=True,
        expression_scaling=args.expression_scaling,
    )

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Step 4: Create trainer
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Creating Trainer")
    logger.info("=" * 80)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Using Accelerate: {args.use_accelerate}")

    # Initialize Accelerator and prepare everything if using Accelerate
    accelerator_instance = None
    if args.use_accelerate:
        logger.info("\n" + "=" * 80)
        logger.info("Preparing Components with Accelerate")
        logger.info("=" * 80)
        from accelerate import Accelerator
        accelerator_instance = Accelerator()
        logger.info(f"Initialized Accelerator on device: {accelerator_instance.device}")

        # Prepare ALL components together - this ensures dataloaders are on correct device
        model, train_loader, val_loader, test_loader = accelerator_instance.prepare(
            model, train_loader, val_loader, test_loader
        )
        logger.info("✓ Model and dataloaders prepared with Accelerate")

    trainer = TransformerTrainer(
        model,
        config=training_config,
        device=device,
        use_accelerate=args.use_accelerate,
        accelerator=accelerator_instance,
    )

    # Step 5: Train model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Training Model")
    logger.info("=" * 80)
    history = trainer.fit(train_loader, val_loader, num_epochs=args.epochs)

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
            "num_genes": num_genes,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "batch_size": args.batch_size,
        },
        "training_history": {
            "train_loss": history["train_loss"],
            "train_accuracy": history["train_accuracy"],
            "val_loss": history["val_loss"],
            "val_accuracy": history["val_accuracy"],
        },
        "test_metrics": test_metrics,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
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
