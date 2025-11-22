#!/usr/bin/env python3
"""
Train MLP on clinical and pathology variables for AD classification.
"""

import logging
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ClinicalMLPClassifier(nn.Module):
    """Simple MLP for clinical variable classification."""

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))  # Binary classification

        self.network = nn.Sequential(*layers)

        # Log info
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Clinical MLP: {input_dim} → {hidden_dims} → 2")
        logger.info(f"Total parameters: {total_params:,}")

    def forward(self, x):
        return self.network(x)


def train_clinical_mlp():
    """Train MLP on clinical variables."""

    logger.info("=" * 80)
    logger.info("CLINICAL MLP TRAINING")
    logger.info("=" * 80)

    # Setup
    config = Config()
    data_dir = Path(config.get("output.results_dir")) / "clinical_data"
    output_dir = Path(config.get("output.results_dir")) / "clinical_mlp"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    logger.info(f"\n[1/4] Loading clinical data")
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val: {X_val.shape}")
    logger.info(f"  Test: {X_test.shape}")

    # Create dataloaders
    logger.info(f"\n[2/4] Creating dataloaders")
    batch_size = 256
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    # Model
    logger.info(f"\n[3/4] Building and training model")
    input_dim = X_train.shape[1]
    model = ClinicalMLPClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout=0.3)
    model.to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 100
    early_stopping_patience = 15
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    logger.info(f"  Starting training: {epochs} epochs, early stopping patience={early_stopping_patience}")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, preds = logits.max(1)
            train_correct += (preds == y_batch).sum().item()
            train_total += X_batch.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, preds = logits.max(1)
                val_correct += (preds == y_batch).sum().item()
                val_total += X_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    logger.info(f"Loaded best model from epoch {epochs - patience_counter}")

    # Evaluate
    logger.info(f"\n[4/4] Evaluating on test set")

    def predict(model, loader, device):
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.softmax(logits, dim=1)
                _, preds = logits.max(1)
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y_batch.numpy())
        return np.concatenate(all_preds), np.concatenate(all_probs), np.concatenate(all_labels)

    # Predictions
    train_preds, train_probs, train_labels = predict(model, train_loader, device)
    val_preds, val_probs, val_labels = predict(model, val_loader, device)
    test_preds, test_probs, test_labels = predict(model, test_loader, device)

    # Metrics
    def compute_metrics(y_true, y_pred, y_probs):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_probs[:, 1])
        except:
            auc = 0.0
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

    train_metrics = compute_metrics(train_labels, train_preds, train_probs)
    val_metrics = compute_metrics(val_labels, val_preds, val_probs)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)

    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Val metrics: {val_metrics}")
    logger.info(f"Test metrics: {test_metrics}")

    # Save results
    results = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"\n✓ TEST SET PERFORMANCE:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {test_metrics['auc']:.4f}")

    # Visualizations
    logger.info(f"\nGenerating visualizations...")

    # Training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(["Not AD", "High"])
    ax.set_yticklabels(["Not AD", "High"])
    ax.set_title("Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(test_labels, test_probs[:, 1])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"AUC = {test_metrics['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test Set ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved training_history.png")
    logger.info(f"  ✓ Saved confusion_matrix.png")
    logger.info(f"  ✓ Saved roc_curve.png")

    logger.info(f"\n" + "=" * 80)
    logger.info(f"Training Complete!")
    logger.info(f"=" * 80)

    return True


if __name__ == "__main__":
    success = train_clinical_mlp()
    sys.exit(0 if success else 1)
