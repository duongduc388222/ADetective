import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_curve, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics."""

    def __init__(self, model: nn.Module, accelerator=None):
        """
        Initialize evaluator.

        Args:
            model: Trained model to evaluate
            accelerator: Accelerator instance for distributed evaluation
        """
        self.model = model
        self.accelerator = accelerator

    @torch.no_grad()
    def evaluate(self, data_loader) -> Dict[str, Any]:
        """
        Evaluate model on data loader.

        Args:
            data_loader: DataLoader with test data

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        all_predictions = []
        all_probabilities = []
        all_targets = []

        for inputs, targets in data_loader:
            # Forward pass
            outputs = self.model(inputs)

            # Get probabilities
            probabilities = torch.sigmoid(outputs)

            # Get binary predictions
            predictions = (probabilities > 0.5).float()

            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        # Convert to arrays
        y_true = np.array(all_targets).flatten()
        y_pred = np.array(all_predictions).flatten()
        y_prob = np.array(all_probabilities).flatten()

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred) * 100,
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'predictions': y_pred,
            'probabilities': y_prob,
            'targets': y_true
        }

        # Add confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # Add classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=['Not AD', 'High AD'],
            output_dict=True
        )

        return metrics

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not AD', 'High AD'],
            yticklabels=['Not AD', 'High AD']
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")

        plt.show()

    def create_evaluation_report(
        self,
        metrics: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive evaluation report.

        Args:
            metrics: Dictionary of evaluation metrics
            save_path: Path to save report

        Returns:
            Report as string
        """
        report = "=" * 50 + "\n"
        report += "MODEL EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"

        # Overall metrics
        report += "Overall Performance:\n"
        report += f"  Accuracy: {metrics['accuracy']:.2f}%\n"
        report += f"  F1 Score: {metrics['f1']:.4f}\n"
        report += f"  ROC-AUC: {metrics['roc_auc']:.4f}\n\n"

        # Confusion matrix
        cm = metrics['confusion_matrix']
        report += "Confusion Matrix:\n"
        report += f"              Predicted\n"
        report += f"              Not AD  High AD\n"
        report += f"  Actual Not AD   {cm[0,0]:5d}   {cm[0,1]:5d}\n"
        report += f"        High AD   {cm[1,0]:5d}   {cm[1,1]:5d}\n\n"

        # Per-class metrics
        cls_report = metrics['classification_report']
        report += "Per-Class Metrics:\n"
        for class_name in ['Not AD', 'High AD']:
            stats = cls_report[class_name]
            report += f"  {class_name}:\n"
            report += f"    Precision: {stats['precision']:.4f}\n"
            report += f"    Recall: {stats['recall']:.4f}\n"
            report += f"    F1-Score: {stats['f1-score']:.4f}\n"
            report += f"    Support: {stats['support']}\n"

        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")

        return report


class MetadataEnhancer:
    """Add donor metadata features to improve classification."""

    def __init__(self, metadata_columns: list):
        """
        Initialize metadata enhancer.

        Args:
            metadata_columns: List of metadata columns to include
        """
        self.metadata_columns = metadata_columns
        self.scalers = {}

    def prepare_metadata(self, adata) -> np.ndarray:
        """
        Extract and prepare metadata features.

        Args:
            adata: AnnData object with metadata in obs

        Returns:
            Metadata features array
        """
        import pandas as pd

        metadata_features = []

        for col in self.metadata_columns:
            if col not in adata.obs.columns:
                logger.warning(f"Metadata column {col} not found")
                continue

            values = adata.obs[col].values

            if col == 'Age at Death':
                # Normalize age
                values = (values - 80) / 10  # Center around 80, scale by 10
                metadata_features.append(values.reshape(-1, 1))

            elif col == 'Sex':
                # Binary encode sex
                values = (values == 'Female').astype(float)
                metadata_features.append(values.reshape(-1, 1))

            elif col == 'APOE Genotype':
                # Count APOE4 alleles
                apoe4_count = np.zeros(len(values))
                for i, genotype in enumerate(values):
                    if pd.isna(genotype):
                        apoe4_count[i] = 0
                    else:
                        apoe4_count[i] = genotype.count('4')
                metadata_features.append(apoe4_count.reshape(-1, 1))

        if metadata_features:
            return np.hstack(metadata_features)
        else:
            return np.zeros((len(adata), 0))

    def combine_with_expression(
        self,
        expression_data: np.ndarray,
        metadata_data: np.ndarray
    ) -> np.ndarray:
        """
        Combine expression and metadata features.

        Args:
            expression_data: Gene expression features
            metadata_data: Metadata features

        Returns:
            Combined feature array
        """
        return np.hstack([expression_data, metadata_data])
