#!/usr/bin/env python3
"""
Phase 3: Train MLP baseline model for oligodendrocyte AD classification.

This script:
1. Loads preprocessed datasets from Phase 2
2. Creates PyTorch dataloaders
3. Initializes and trains MLP model
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import anndata as ad

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mlp import MLPClassifier, MLPTrainer
from src.eval.metrics import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments for MLP training."""
    parser = argparse.ArgumentParser(description="Train MLP baseline model for oligodendrocyte AD classification")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with preprocessed data (train.h5ad, val.h5ad, test.h5ad)")
    parser.add_argument("--output-dir", type=str, default="./results/mlp", help="Directory to save results and checkpoint (default: ./results/mlp)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for AdamW optimizer (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for regularization (default: 1e-4)")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128], help="Hidden layer dimensions (default: 512 256 128)")
    parser.add_argument("--dropout-rate", type=float, default=0.3, help="Dropout rate (default: 0.3)")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value (default: 1.0)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--use-accelerate", action="store_true", default=True, help="Use Accelerate for distributed training (default: True)")
    parser.add_argument("--no-accelerate", dest="use_accelerate", action="store_false", help="Disable Accelerate and use single GPU/CPU")
    return parser.parse_args()


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

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
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


def main():
    """Train MLP model."""
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("PHASE 3: MLP BASELINE TRAINING")
    logger.info("=" * 80)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nData directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Hyperparameters:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Hidden dims: {args.hidden_dims}")
    logger.info(f"  Dropout: {args.dropout_rate}")

    # Step 1: Load data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Preprocessed Data")
    logger.info("=" * 80)
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir)
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        logger.error("Make sure you've run scripts/load.py first")
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
    logger.info("STEP 3: Initializing MLP Model")
    logger.info("=" * 80)
    input_dim = X_train.shape[1]
    output_dim = 2

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=output_dim,
        dropout_rate=args.dropout_rate,
        batch_norm=True,
        activation="relu",
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

    trainer = MLPTrainer(
        model,
        config=training_config,
        device=device,
        use_accelerate=args.use_accelerate,
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
            "input_dim": input_dim,
            "hidden_dims": args.hidden_dims,
            "output_dim": output_dim,
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
