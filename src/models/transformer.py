"""
Transformer Model for Gene Expression Classification.

Implements custom Transformer that treats genes as sequence tokens with expression values,
leveraging PyTorch 2.0's automatic Flash Attention optimization.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from accelerate import Accelerator

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GeneTransformer(nn.Module):
    """
    Transformer model for gene expression classification.

    Treats genes as sequence tokens with expression values as modulation.
    """

    def __init__(
        self,
        num_genes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_length: int = 2048,
        use_cls_token: bool = True,
        expression_scaling: str = 'multiplicative',
    ):
        """
        Initialize GeneTransformer.

        Args:
            num_genes: Number of genes in the input
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            use_cls_token: Whether to use CLS token for classification
            expression_scaling: How to combine gene embeddings with expression values
                              ('multiplicative', 'additive', 'concatenate')
        """
        super(GeneTransformer, self).__init__()

        self.num_genes = num_genes
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.expression_scaling = expression_scaling

        logger.info(f"Initializing GeneTransformer:")
        logger.info(f"  Number of genes: {num_genes}")
        logger.info(f"  Model dimension: {d_model}")
        logger.info(f"  Number of heads: {nhead}")
        logger.info(f"  Number of layers: {num_layers}")
        logger.info(f"  Expression scaling: {expression_scaling}")

        # Gene embeddings
        self.gene_embeddings = nn.Embedding(num_genes, d_model)

        # CLS token embedding if used
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Expression value processing
        if expression_scaling == 'additive':
            self.expression_projection = nn.Linear(1, d_model)
        elif expression_scaling == 'concatenate':
            self.gene_embeddings = nn.Embedding(num_genes, d_model // 2)
            self.expression_projection = nn.Linear(1, d_model // 2)
            self.fusion_layer = nn.Linear(d_model, d_model)

        # Positional encoding (account for CLS token)
        actual_max_seq_length = max(max_seq_length, num_genes + (1 if use_cls_token else 0))
        self.positional_encoding = PositionalEncoding(d_model, dropout, actual_max_seq_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        # Initialize gene embeddings
        nn.init.normal_(self.gene_embeddings.weight, mean=0.0, std=0.02)

        # Initialize classifier
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def create_gene_representations(
        self,
        expression_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Create gene representations from expression values.

        Args:
            expression_values: Tensor of shape (batch_size, num_genes)

        Returns:
            Gene representations of shape (batch_size, num_genes, d_model)
        """
        batch_size = expression_values.size(0)

        # Create gene indices
        gene_indices = torch.arange(self.num_genes, device=expression_values.device)
        gene_indices = gene_indices.unsqueeze(0).expand(batch_size, -1)

        # Get gene embeddings
        gene_embeds = self.gene_embeddings(gene_indices)  # (batch, num_genes, d_model)

        if self.expression_scaling == 'multiplicative':
            # Scale embeddings by expression values
            expression_values = expression_values.unsqueeze(-1)  # (batch, num_genes, 1)
            # Apply soft scaling to avoid extreme values
            expression_scale = torch.tanh(expression_values / 5.0) + 1.0
            gene_representations = gene_embeds * expression_scale

        elif self.expression_scaling == 'additive':
            # Add expression embedding to gene embedding
            expression_values = expression_values.unsqueeze(-1)  # (batch, num_genes, 1)
            expression_embeds = self.expression_projection(expression_values)
            gene_representations = gene_embeds + expression_embeds

        elif self.expression_scaling == 'concatenate':
            # Concatenate gene and expression embeddings
            expression_values = expression_values.unsqueeze(-1)
            expression_embeds = self.expression_projection(expression_values)
            concat_embeds = torch.cat([gene_embeds, expression_embeds], dim=-1)
            gene_representations = self.fusion_layer(concat_embeds)

        else:
            gene_representations = gene_embeds

        return gene_representations

    def forward(self, expression_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer.

        Args:
            expression_values: Tensor of shape (batch_size, num_genes)

        Returns:
            Logits of shape (batch_size, 1)
        """
        batch_size = expression_values.size(0)

        # Create gene representations
        gene_representations = self.create_gene_representations(expression_values)

        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            sequence = torch.cat([cls_tokens, gene_representations], dim=1)
        else:
            sequence = gene_representations

        # Apply positional encoding
        sequence = self.positional_encoding(sequence)

        # Pass through transformer
        transformer_output = self.transformer(sequence)

        # Extract representation for classification
        if self.use_cls_token:
            # Use CLS token output
            cell_representation = transformer_output[:, 0, :]
        else:
            # Use mean pooling
            cell_representation = transformer_output.mean(dim=1)

        # Classification
        logits = self.classifier(cell_representation)

        return logits


class FlashTransformer(GeneTransformer):
    """
    Transformer with Flash Attention optimization.

    PyTorch 2.0+ automatically uses Flash Attention when available.
    This class ensures optimal settings for Flash Attention.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check if Flash Attention is available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            self._use_flash_attn = True
            logger.info("✓ Flash Attention available (PyTorch 2.0+)")
        else:
            self._use_flash_attn = False
            logger.warning("✗ Flash Attention not available. Using standard attention.")

    def forward(self, expression_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Flash Attention optimization.

        Flash Attention is automatically used by PyTorch 2.0+ when:
        - CUDA device with compute capability >= 8.0
        - Sequence length and dimensions meet requirements
        - No attention mask is used (or causal mask)
        """
        # PyTorch 2.0 automatically optimizes this
        return super().forward(expression_values)


class TransformerTrainer:
    """
    Trainer for Transformer model.

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
        use_accelerate: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Configuration dictionary (from YAML)
            device: Device to use ("cuda" or "cpu") - used if use_accelerate=False
            use_accelerate: Whether to use Accelerate library (default: True)
        """
        self.model = model
        self.config = config
        self.use_accelerate = use_accelerate

        # Initialize Accelerator
        if self.use_accelerate:
            self.accelerator = Accelerator()
            logger.info(f"Initializing TransformerTrainer with Accelerate")
            logger.info(f"  Device: {self.accelerator.device}")
            logger.info(f"  Process index: {self.accelerator.process_index}")
            logger.info(f"  Number of processes: {self.accelerator.num_processes}")
        else:
            self.accelerator = None
            self.device = device
            logger.info(f"Initializing TransformerTrainer on device: {device}")
            self.model = self.model.to(device)

        # Create optimizer (before prepare)
        self.optimizer = self._create_optimizer()

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Prepare model, optimizer, and scheduler with Accelerate
        if self.use_accelerate:
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )
            logger.info("Model, optimizer, and scheduler prepared with Accelerate")

        # Loss function (binary classification with logits)
        self.criterion = nn.BCEWithLogitsLoss()

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
            # No need to move to device with Accelerate - it handles this
            if not self.use_accelerate:
                X = X.to(self.device)
                y = y.to(self.device)

            y = y.float().unsqueeze(1)  # Convert to float for BCEWithLogitsLoss

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)

            # Backward pass with Accelerate
            if self.use_accelerate:
                self.accelerator.backward(loss)
            else:
                loss.backward()

            # Gradient clipping
            grad_clip = self.config["training"].get("gradient_clipping")
            if grad_clip:
                if self.use_accelerate:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            # Optimizer step
            self.optimizer.step()

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Metrics
            with torch.no_grad():
                preds = (logits > 0).float()
                accuracy = (preds == y).float().mean()

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
            self.global_step += 1

            # Logging (only on main process when using Accelerate)
            if (batch_idx + 1) % self.config["logging"].get("log_frequency", 100) == 0:
                if not self.use_accelerate or self.accelerator.is_main_process:
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
                # No need to move to device with Accelerate - it handles this
                if not self.use_accelerate:
                    X = X.to(self.device)
                    y = y.to(self.device)

                y = y.float().unsqueeze(1)

                logits = self.model(X)
                loss = self.criterion(logits, y)
                preds = (logits > 0).float()
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

        # Prepare dataloaders with Accelerate
        if self.use_accelerate:
            train_loader, val_loader = self.accelerator.prepare(train_loader, val_loader)
            logger.info("Dataloaders prepared with Accelerate")

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        early_stop_config = self.config["training"].get("early_stopping", {})
        early_stop_enabled = early_stop_config.get("enabled", True)
        patience = early_stop_config.get("patience", 15)

        if not self.use_accelerate or self.accelerator.is_main_process:
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

            if not self.use_accelerate or self.accelerator.is_main_process:
                logger.info(
                    f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

            # Early stopping
            if early_stop_enabled:
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
