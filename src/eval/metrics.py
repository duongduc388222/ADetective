"""
Evaluation metrics and utilities for classification models.

Computes standard ML metrics: accuracy, precision, recall, F1, ROC-AUC, etc.
Also handles visualization and saving results.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluator for binary classification models.

    Computes:
    - Accuracy, Precision, Recall, F1
    - ROC-AUC
    - Confusion matrix
    - Classification report
    - Visualizations (ROC curve, confusion matrix)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            device: Device to use ("cuda" or "cpu")
        """
        self.model = model
        self.device = device
        self.model.to(device)

    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions on a dataset.

        Args:
            data_loader: DataLoader with (X, y) tuples

        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for X, y in data_loader:
                X = X.to(self.device)
                logits = self.model(X)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(y.numpy())

        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Handle both binary (shape: N,1) and multiclass (shape: N,2+) outputs
        if logits.shape[1] == 1:
            # Binary classification: use sigmoid
            probs_positive = 1 / (1 + np.exp(-logits.flatten()))  # Sigmoid
            preds = (probs_positive > 0.5).astype(int)
            # Create 2-column probs array: [P(class 0), P(class 1)]
            probs = np.column_stack([1 - probs_positive, probs_positive])
        else:
            # Multiclass: use softmax
            preds = np.argmax(logits, axis=1)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # Softmax

        return preds, probs, labels

    def evaluate(
        self,
        data_loader,
        dataset_name: str = "test",
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset and compute all metrics.

        Args:
            data_loader: DataLoader with (X, y) tuples
            dataset_name: Name of dataset (for logging)

        Returns:
            Dictionary with all metrics
        """
        preds, probs, labels = self.predict(data_loader)

        # Compute metrics
        metrics = {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probs[:, 1])),
        }

        logger.info(f"\n{dataset_name.upper()} SET EVALUATION:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def get_classification_report(
        self,
        data_loader,
        dataset_name: str = "test",
    ) -> str:
        """
        Get detailed classification report.

        Args:
            data_loader: DataLoader with (X, y) tuples
            dataset_name: Name of dataset (for logging)

        Returns:
            Classification report string
        """
        preds, _, labels = self.predict(data_loader)

        report = classification_report(
            labels,
            preds,
            target_names=["Not AD (0)", "High AD (1)"],
            digits=4,
        )

        logger.info(f"\n{dataset_name.upper()} CLASSIFICATION REPORT:\n{report}")
        return report

    def get_confusion_matrix(
        self,
        data_loader,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Get confusion matrix.

        Args:
            data_loader: DataLoader with (X, y) tuples
            normalize: Normalize by row (True) or not (False)

        Returns:
            Confusion matrix
        """
        preds, _, labels = self.predict(data_loader)
        cm = confusion_matrix(labels, preds)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        return cm

    def plot_confusion_matrix(
        self,
        data_loader,
        save_path: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Plot and optionally save confusion matrix.

        Args:
            data_loader: DataLoader with (X, y) tuples
            save_path: Path to save plot (optional)
            normalize: Normalize by row (True) or not (False)
        """
        cm = self.get_confusion_matrix(data_loader, normalize=normalize)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else ".0f",
            cmap="Blues",
            xticklabels=["Not AD", "High AD"],
            yticklabels=["Not AD", "High AD"],
            cbar_kws={"label": "Count" if not normalize else "Proportion"},
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved confusion matrix to {save_path}")

        return plt.gcf()

    def plot_roc_curve(
        self,
        data_loader,
        save_path: Optional[str] = None,
    ):
        """
        Plot and optionally save ROC curve.

        Args:
            data_loader: DataLoader with (X, y) tuples
            save_path: Path to save plot (optional)
        """
        preds, probs, labels = self.predict(data_loader)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved ROC curve to {save_path}")

        return plt.gcf()

    def plot_training_history(
        self,
        history: Dict[str, list],
        save_path: Optional[str] = None,
    ):
        """
        Plot training history (loss and accuracy curves).

        Args:
            history: Dictionary with training history
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Loss plot
        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[1].plot(history["train_accuracy"], label="Train Acc", linewidth=2)
        axes[1].plot(history["val_accuracy"], label="Val Acc", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved training history plot to {save_path}")

        return fig
