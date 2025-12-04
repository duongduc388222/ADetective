"""
scGPT Wrapper for Fine-tuning Foundation Model.

Wraps scGPT (https://github.com/bowang-lab/scGPT) for fine-tuning on AD pathology
classification with gene vocabulary alignment, expression tokenization, and efficient fine-tuning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

logger = logging.getLogger(__name__)

# Try to import actual scGPT library
try:
    import scgpt as scg
    from scgpt.model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.loss import masked_mse_loss, criterion_neg_log_bernoulli
    SCGPT_AVAILABLE = True
except ImportError:
    logger.warning(
        "scGPT library not installed. Install with: "
        "pip install scgpt or git+https://github.com/bowang-lab/scGPT.git"
    )
    SCGPT_AVAILABLE = False


class scGPTWrapper(nn.Module):
    """
    Wrapper for scGPT foundation model fine-tuning.

    This wrapper provides:
    - Integration with actual scGPT's TransformerModel
    - Gene vocabulary management using GeneVocab
    - Expression binning and tokenization
    - Layer freezing for efficient fine-tuning
    - Support for multi-task training (classification + MVC)
    """

    def __init__(
        self,
        gene_names: List[str],
        pretrained_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        n_bins: int = 51,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        freeze_layers: int = 0,
        dropout: float = 0.1,
        do_mvc: bool = True,
        do_dab: bool = False,
        do_ecs: bool = False,
        use_batch_labels: bool = False,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = True,
    ):
        """
        Initialize scGPT wrapper.

        Args:
            gene_names: List of gene names in dataset
            pretrained_path: Path to pretrained scGPT checkpoint
            vocab_path: Path to gene vocabulary file (GeneVocab format)
            n_bins: Number of expression bins (scGPT standard: 51)
            d_model: Model embedding dimension (scGPT standard: 512)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers (scGPT standard: 12)
            freeze_layers: Number of layers to freeze from bottom
            dropout: Dropout rate
            do_mvc: Enable Masked Value Correction
            do_dab: Enable Domain Adaptation by Batch norm
            do_ecs: Enable Elastic Cell Similarity
            use_batch_labels: Use batch labels for DSBN
            explicit_zero_prob: Model explicit zero probability
            use_fast_transformer: Use Flash Attention if available
        """
        super().__init__()

        if not SCGPT_AVAILABLE:
            raise ImportError(
                "scGPT library is required. Install with:\n"
                "pip install scgpt\n"
                "or from source:\n"
                "pip install git+https://github.com/bowang-lab/scGPT.git"
            )

        self.gene_names = gene_names
        self.n_bins = n_bins
        self.d_model = d_model
        self.freeze_layers = freeze_layers

        logger.info(f"Initializing scGPT Wrapper:")
        logger.info(f"  Number of genes: {len(gene_names)}")
        logger.info(f"  Model dimension: {d_model}")
        logger.info(f"  Number of heads: {nhead}")
        logger.info(f"  Number of layers: {num_layers}")
        logger.info(f"  Expression bins: {n_bins}")
        logger.info(f"  Frozen layers: {freeze_layers}")

        # Load or create gene vocabulary
        self.gene_vocab = self._load_gene_vocab(vocab_path, gene_names)

        # Create scGPT's TransformerModel
        # IMPORTANT: Use d_hid=d_model (512) to match pretrained weights architecture
        # The pretrained checkpoint uses linear1/linear2 with shape [512, 512], NOT [2048, 512]
        self.transformer = TransformerModel(
            ntoken=len(self.gene_vocab),
            d_model=d_model,
            nhead=nhead,
            d_hid=d_model,  # Match pretrained checkpoint: d_hid = d_model (512), not d_model * 4
            nlayers=num_layers,
            nlayers_cls=1,  # Single classification head layer
            n_input_bins=n_bins,
            dropout=dropout,
            do_mvc=do_mvc,
            do_dab=do_dab,
            ecs_threshold=0.8 if do_ecs else 0.0,
            use_batch_labels=use_batch_labels,
            domain_spec_batchnorm=False,  # Can be enabled with batch labels
            explicit_zero_prob=explicit_zero_prob,
            use_fast_transformer=use_fast_transformer,
            pre_norm=False,
            vocab=self.gene_vocab,
        )

        # Classification head (for AD classification task)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Load pretrained weights if available
        if pretrained_path and Path(pretrained_path).exists():
            self._load_pretrained_model(pretrained_path)

        # Freeze layers if specified
        self._freeze_layers()

        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        logger.info(
            f"Trainable parameters: "
            f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        )

    def _load_gene_vocab(
        self, vocab_path: Optional[str], gene_names: List[str]
    ) -> "GeneVocab":
        """Load or create GeneVocab from scGPT."""
        if vocab_path and Path(vocab_path).exists():
            try:
                # Load pretrained vocabulary
                vocab = GeneVocab.from_file(vocab_path)
                logger.info(
                    f"Loaded pretrained gene vocabulary with {len(vocab)} genes"
                )
            except Exception as e:
                logger.warning(f"Could not load vocab from {vocab_path}: {e}")
                logger.info("Creating new vocabulary from dataset genes")
                vocab = GeneVocab(gene_list_or_vocab=gene_names)
        else:
            # Create vocabulary from dataset genes
            vocab = GeneVocab(gene_list_or_vocab=gene_names)
            logger.info(f"Created gene vocabulary from dataset with {len(vocab)} genes")

        # Add special tokens if not present
        special_tokens = ["<pad>", "<cls>", "<eos>", "<mask>"]
        for token in special_tokens:
            if token not in vocab:
                vocab.append_token(token)

        # Set default index for padding
        vocab.set_default_index(vocab["<pad>"])

        return vocab

    def _freeze_layers(self):
        """Freeze specified number of bottom transformer layers."""
        if self.freeze_layers > 0:
            # Freeze gene encoder (contains gene embedding layer)
            for param in self.transformer.encoder.parameters():
                param.requires_grad = False

            # Freeze value encoder (expression value embeddings)
            for param in self.transformer.value_encoder.parameters():
                param.requires_grad = False

            # Freeze batch encoder if it exists
            if hasattr(self.transformer, "batch_encoder") and self.transformer.batch_encoder is not None:
                for param in self.transformer.batch_encoder.parameters():
                    param.requires_grad = False

            # Freeze transformer encoder layers
            if hasattr(self.transformer, "transformer_encoder"):
                # Access layers through transformer_encoder, not encoder
                if hasattr(self.transformer.transformer_encoder, "layers"):
                    layers = self.transformer.transformer_encoder.layers
                    for i in range(min(self.freeze_layers, len(layers))):
                        for param in layers[i].parameters():
                            param.requires_grad = False

            logger.info(f"Froze {self.freeze_layers} transformer layers and all embedding encoders")

    def _load_pretrained_model(self, checkpoint_path: str):
        """Load pretrained scGPT model from checkpoint."""
        logger.info(f"Loading pretrained scGPT from {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                model_state = checkpoint["state_dict"]
            else:
                model_state = checkpoint

            # Remove 'module.' prefix if present (from DataParallel)
            model_state = {
                k.replace("module.", ""): v for k, v in model_state.items()
            }

            # Load weights with strict=False to allow missing classifier head
            missing_keys, unexpected_keys = self.transformer.load_state_dict(
                model_state, strict=False
            )

            logger.info("Pretrained weights loaded successfully")
            if missing_keys:
                logger.info(f"Missing keys (expected): {missing_keys[:5]}")
            if unexpected_keys:
                logger.info(f"Unexpected keys: {unexpected_keys[:5]}")

            # Load vocabulary if present in checkpoint
            if "vocab" in checkpoint:
                self.gene_vocab = checkpoint["vocab"]
                logger.info(f"Loaded vocabulary with {len(self.gene_vocab)} genes")

        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise

    def tokenize_and_pad_batch(
        self,
        expression_data: Dict[str, np.ndarray],
        max_len: int = 2048,
        pad_value: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize expression data to scGPT format.

        Args:
            expression_data: Dict with keys:
                - "gene_names": List[str] - gene names
                - "values": np.ndarray of shape (n_genes,) - expression values
            max_len: Maximum sequence length
            pad_value: Padding value for binned expression

        Returns:
            Dict with:
                - "gene_ids": token IDs (seq_len,)
                - "values": binned expression values (seq_len,)
                - "padding_mask": boolean mask (seq_len,)
        """
        gene_names = expression_data["gene_names"]
        values = expression_data["values"]

        # Map genes to vocabulary
        gene_ids = []
        gene_values = []
        for gene, value in zip(gene_names, values):
            if gene in self.gene_vocab and value > 0:  # Filter zero genes
                gene_ids.append(self.gene_vocab[gene])
                gene_values.append(value)

        # Limit sequence length
        if len(gene_ids) > max_len:
            # Keep top genes by expression
            sorted_idx = np.argsort(gene_values)[::-1][:max_len]
            gene_ids = [gene_ids[i] for i in sorted_idx]
            gene_values = [gene_values[i] for i in sorted_idx]

        # Bin expression values (0-51)
        binned_values = self._bin_expression(np.array(gene_values))

        # Pad to max_len
        pad_len = max_len - len(gene_ids)
        if pad_len > 0:
            pad_token_id = self.gene_vocab["<pad>"]
            gene_ids = gene_ids + [pad_token_id] * pad_len
            binned_values = np.concatenate([binned_values, np.zeros(pad_len, dtype=int)])

        gene_ids = np.array(gene_ids, dtype=np.int64)
        binned_values = np.array(binned_values, dtype=np.int64)

        # Create padding mask (True for padding positions)
        padding_mask = gene_ids == self.gene_vocab["<pad>"]

        return {
            "gene_ids": torch.tensor(gene_ids, dtype=torch.long),
            "values": torch.tensor(binned_values, dtype=torch.float),
            "padding_mask": torch.tensor(padding_mask),
        }

    def _bin_expression(self, values: np.ndarray) -> np.ndarray:
        """
        Bin expression values using quantile-based binning (scGPT approach).

        Args:
            values: Non-zero expression values

        Returns:
            Binned values in range [1, n_bins]
        """
        if len(values) == 0:
            return np.array([], dtype=int)

        # Quantile-based binning
        quantiles = np.linspace(0, 100, self.n_bins)
        bin_edges = np.percentile(values, quantiles)

        # Digitize: returns bin index (1 to n_bins)
        binned = np.digitize(values, bin_edges[1:-1]) + 1
        binned = np.clip(binned, 1, self.n_bins)

        return binned

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        batch_labels: Optional[torch.Tensor] = None,
        output_cell_embeddings: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass using scGPT model.

        Args:
            gene_ids: Gene token IDs from vocabulary (batch_size, seq_len)
            values: Binned expression values (batch_size, seq_len)
            src_key_padding_mask: Padding mask - True for padding positions
            batch_labels: Batch labels for domain adaptation (optional)
            output_cell_embeddings: Whether to return cell embeddings

        Returns:
            Classification logits (batch_size, 1)
        """
        # Get transformer outputs
        output_dict = self.transformer(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels,
            CLS=True,
            MVC=False,  # No masking during inference
            ECS=False,
        )

        # Extract cell embeddings (CLS token or mean pooling from scGPT)
        # scGPT returns 'cell_emb' for the cell-level representation
        if "cell_emb" in output_dict:
            cell_emb = output_dict["cell_emb"]
        else:
            # Fallback: use CLS token output (first token)
            cell_emb = output_dict["last_hidden_state"][:, 0, :]

        # Classification head
        logits = self.classifier(cell_emb)

        return logits


class scGPTFineTuner:
    """Helper class for fine-tuning scGPT models with custom objectives."""

    def __init__(
        self,
        model: scGPTWrapper,
        learning_rate: float = 1e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
    ):
        """
        Initialize fine-tuner.

        Args:
            model: scGPT wrapper model to fine-tune
            learning_rate: Learning rate (lower for fine-tuning)
            warmup_steps: Number of warmup steps
            weight_decay: L2 regularization weight
        """
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        logger.info(
            f"Percentage trainable: {100 * trainable_params / total_params:.2f}%"
        )

    def create_optimizer_and_scheduler(
        self,
        num_training_steps: int,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """
        Create optimizer and learning rate scheduler.

        Uses different weight decay for embeddings and biases (no decay).

        Args:
            num_training_steps: Total number of training steps

        Returns:
            Optimizer and scheduler tuple
        """
        # Separate parameters for different weight decay
        no_decay = ["bias", "LayerNorm", "ln", "embedding"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # AdamW optimizer with epsilon for stability
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Cosine learning rate schedule with warmup
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(
                max(1, num_training_steps - self.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return optimizer, scheduler
