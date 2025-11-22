#!/usr/bin/env python3
"""
Phase 3: Train MLP Baseline Model

This script trains a simple fully-connected neural network baseline for
binary classification of AD pathology in oligodendrocytes.

Usage:
    python scripts/training/train_mlp.py
    python scripts/training/train_mlp.py --data-dir results/processed_data
    python scripts/training/train_mlp.py --config configs/mlp_config.yaml
"""

import logging
import sys
import json
from pathlib import Path
from typing import Optional

import click
import yaml
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.models.mlp import MLPClassifier, MLPTrainer
from src.eval.metrics import ModelEvaluator
from src.data.dataset import create_data_loaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = "configs/mlp_config.yaml"

    logger.info(f"Loading config from {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def train_baseline(
    data_dir: str = "results/processed_data",
    config_path: str = "configs/mlp_config.yaml",
    output_dir: str = "results/mlp_baseline",
):
    """
    Train MLP baseline model.

    Args:
        data_dir: Directory with preprocessed data
        config_path: Path to configuration YAML
        output_dir: Output directory for results
    """
    logger.info("=" * 80)
    logger.info("MLP BASELINE TRAINING")
    logger.info("=" * 80)

    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(config_path)
    set_seed(config["reproducibility"]["seed"])

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to {output_dir / 'config.yaml'}")

    # Check device
    device = config["device"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)

    data_dir = Path(data_dir)
    train_path = data_dir / "train_data.h5ad"
    val_path = data_dir / "val_data.h5ad"
    test_path = data_dir / "test_data.h5ad"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            logger.error(f"Data file not found: {path}")
            logger.error(f"Run 'python scripts/prepare_data.py' first to preprocess data")
            return False

    train_loader, val_loader, test_loader = create_data_loaders(
        str(train_path),
        str(val_path),
        str(test_path),
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=config["data"].get("pin_memory", True),
    )

    # Create model
    logger.info("\n" + "=" * 80)
    logger.info("BUILDING MODEL")
    logger.info("=" * 80)

    model_config = config["model"]["architecture"]
    model = MLPClassifier(
        input_dim=model_config["input_dim"],
        hidden_dims=model_config["hidden_dims"],
        output_dim=model_config["output_dim"],
        dropout_rate=model_config["dropout_rate"],
        batch_norm=model_config["batch_norm"],
        activation=model_config["activation"],
    )

    # Create trainer
    trainer = MLPTrainer(model, config, device=device)

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80)

    history = trainer.fit(train_loader, val_loader)

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {k: [float(v) for v in vs] for k, vs in history.items()}
        json.dump(history_json, f, indent=2)
    logger.info(f"Saved training history to {output_dir / 'training_history.json'}")

    # Save model
    logger.info("\n" + "=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)

    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")

    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)

    evaluator = ModelEvaluator(model, device=device)

    # Train set
    train_metrics = evaluator.evaluate(train_loader, dataset_name="train")

    # Validation set
    val_metrics = evaluator.evaluate(val_loader, dataset_name="validation")

    # Test set
    test_metrics = evaluator.evaluate(test_loader, dataset_name="test")

    # Classification reports
    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION REPORTS")
    logger.info("=" * 80)

    evaluator.get_classification_report(train_loader, "train")
    evaluator.get_classification_report(val_loader, "validation")
    evaluator.get_classification_report(test_loader, "test")

    # Save metrics
    results = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {output_dir / 'metrics.json'}")

    # Plotting
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING PLOTS")
    logger.info("=" * 80)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Training history
    evaluator.plot_training_history(history, save_path=str(plots_dir / "training_history.png"))

    # Confusion matrices
    evaluator.plot_confusion_matrix(
        train_loader, save_path=str(plots_dir / "confusion_matrix_train.png")
    )
    evaluator.plot_confusion_matrix(
        val_loader, save_path=str(plots_dir / "confusion_matrix_val.png")
    )
    evaluator.plot_confusion_matrix(
        test_loader, save_path=str(plots_dir / "confusion_matrix_test.png")
    )

    # ROC curves
    evaluator.plot_roc_curve(train_loader, save_path=str(plots_dir / "roc_curve_train.png"))
    evaluator.plot_roc_curve(val_loader, save_path=str(plots_dir / "roc_curve_val.png"))
    evaluator.plot_roc_curve(test_loader, save_path=str(plots_dir / "roc_curve_test.png"))

    logger.info(f"Saved plots to {plots_dir}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    return True


@click.command()
@click.option(
    "--data-dir",
    default="results/processed_data",
    help="Directory with preprocessed data (default: results/processed_data)",
)
@click.option(
    "--config",
    "config_path",
    default="configs/mlp_config.yaml",
    help="Path to config YAML (default: configs/mlp_config.yaml)",
)
@click.option(
    "--output-dir",
    default="results/mlp_baseline",
    help="Output directory for results (default: results/mlp_baseline)",
)
def main(data_dir: str, config_path: str, output_dir: str):
    """Train MLP baseline model for oligodendrocyte AD classification."""
    try:
        success = train_baseline(
            data_dir=data_dir,
            config_path=config_path,
            output_dir=output_dir,
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
