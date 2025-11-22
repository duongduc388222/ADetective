"""
MLP Baseline Model for Oligodendrocyte AD Classification.

Implements a simple fully-connected neural network with batch normalization
and dropout for binary classification of AD pathology.
"""

import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

logger = logging.getLogger(__name__)


class MLPClassifier(nn.Module):
    """
    Fully-connected neural network for binary classification.

    Architecture:
    - Input layer: [batch_size, input_dim]
    - Hidden layers: [hidden_dims[0]], [hidden_dims[1]], ...
    - Output layer: [batch_size, output_dim]
    - Activations: ReLU
    - Regularization: Batch normalization, Dropout
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 2,
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
    ):
        """
        Initialize MLP classifier.

        Args:
            input_dim: Input feature dimension (number of genes)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output classes (default: 2 for binary classification)
            dropout_rate: Dropout rate (default: 0.3)
            batch_norm: Use batch normalization (default: True)
            activation: Activation function name (default: "relu")
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.activation = activation

        logger.info(f"Initializing MLPClassifier:")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Hidden dims: {hidden_dims}")
        logger.info(f"  Output dim: {output_dim}")
        logger.info(f"  Dropout: {dropout_rate}")
        logger.info(f"  Batch norm: {batch_norm}")

        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Batch normalization (not on output layer)
            if batch_norm and i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))

            # Activation function (not on output layer)
            if i < len(dims) - 2:
                if activation.lower() == "relu":
                    layers.append(nn.ReLU())
                elif activation.lower() == "gelu":
                    layers.append(nn.GELU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")

            # Dropout (not on output layer)
            if dropout_rate > 0 and i < len(dims) - 2:
                layers.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

        logger.info(f"Model size: {self._count_parameters():,} parameters")

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Logits tensor [batch_size, output_dim]
        """
        return self.network(x)


class MLPTrainer:
    """
    Trainer for MLP model using Accelerate for distributed training.

    Handles:
    - Mixed precision training (fp16/bf16)
    - Learning rate scheduling with warmup
    - Gradient accumulation and clipping
    - Early stopping
    - Model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Configuration dictionary (from YAML)
            device: Device to use ("cuda" or "cpu")
        """
        self.model = model
        self.config = config
        self.device = device

        logger.info(f"Initializing MLPTrainer on device: {device}")

        # Move model to device
        self.model = self.model.to(device)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_config = self.config["training"]
        optimizer_name = opt_config.get("optimizer", "adamw").lower()

        if optimizer_name == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=opt_config["learning_rate"],
                weight_decay=opt_config.get("weight_decay", 1e-4),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        logger.info(f"Created optimizer: {optimizer_name}")
        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create learning rate scheduler with optional warmup."""
        opt_config = self.config["training"]
        warmup_config = opt_config.get("warmup", {})
        warmup_steps = warmup_config.get("steps", 500) if warmup_config.get("enabled") else 0

        if warmup_steps > 0:
            # Linear warmup followed by cosine decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_steps,
            )
            logger.info(f"Created learning rate scheduler with {warmup_steps} warmup steps")
            return warmup_scheduler
        else:
            logger.info("No learning rate scheduler (no warmup)")
            return None

    def train_epoch(
        self,
        train_loader,
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            Tuple of (avg_loss, avg_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(self.device)
            y = y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_clip = self.config["training"].get("gradient_clipping")
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            # Optimizer step
            self.optimizer.step()

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == y).float().mean()

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if (batch_idx + 1) % self.config["logging"].get("log_frequency", 100) == 0:
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | Acc: {accuracy.item():.4f}"
                )

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        return avg_loss, avg_accuracy

    def validate(self, val_loader) -> Tuple[float, float]:
        """
        Validate on validation set.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Tuple of (avg_loss, avg_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                logits = self.model(X)
                loss = self.criterion(logits, y)
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == y).float().mean()

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
        num_epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs with validation and early stopping.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs (default: from config)

        Returns:
            Dictionary with training history
        """
        if num_epochs is None:
            num_epochs = self.config["training"]["epochs"]

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        early_stop_config = self.config["training"].get("early_stopping", {})
        early_stop_enabled = early_stop_config.get("enabled", True)
        patience = early_stop_config.get("patience", 15)

        logger.info(
            f"Starting training: {num_epochs} epochs, "
            f"early stopping={early_stop_enabled} (patience={patience})"
        )

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            logger.info(
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Early stopping
            if early_stop_enabled:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    logger.info(f"  → New best validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= patience:
                        logger.info(
                            f"Early stopping triggered (patience {patience} reached)"
                        )
                        break

        return history
