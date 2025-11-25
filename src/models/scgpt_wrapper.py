"""
scGPT Wrapper for Fine-tuning Foundation Model.

Wraps scGPT for fine-tuning on AD pathology classification with:
- Gene vocabulary alignment
- Expression binning/tokenization
- Classification head addition
- Efficient fine-tuning setup
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class scGPTWrapper(nn.Module):
    """
    Wrapper for scGPT foundation model fine-tuning.

    Handles:
    - Gene vocabulary alignment
    - Expression binning/tokenization
    - Classification head addition
    - Fine-tuning setup
    """

    def __init__(
        self,
        num_genes: int,
        pretrained_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        n_bins: int = 51,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        freeze_layers: int = 0,
        dropout: float = 0.1,
        use_fast_tokenizer: bool = True,
    ):
        """
        Initialize scGPT wrapper.

        Args:
            num_genes: Number of genes in dataset
            pretrained_path: Path to pretrained scGPT checkpoint
            vocab_path: Path to gene vocabulary file
            n_bins: Number of expression bins for tokenization
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            freeze_layers: Number of layers to freeze from bottom
            dropout: Dropout rate
            use_fast_tokenizer: Use optimized tokenizer
        """
        super().__init__()

        self.num_genes = num_genes
        self.n_bins = n_bins
        self.d_model = d_model
        self.freeze_layers = freeze_layers

        logger.info(f"Initializing scGPTWrapper:")
        logger.info(f"  Number of genes: {num_genes}")
        logger.info(f"  Model dimension: {d_model}")
        logger.info(f"  Number of heads: {nhead}")
        logger.info(f"  Number of layers: {num_layers}")
        logger.info(f"  Expression bins: {n_bins}")
        logger.info(f"  Frozen layers: {freeze_layers}")

        # Load or create gene vocabulary
        self.gene_vocab = self._load_gene_vocab(vocab_path, num_genes)
        self.vocab_size = len(self.gene_vocab)

        # Expression binning thresholds
        self.register_buffer('bin_edges', self._create_bin_edges(n_bins))

        # Token embeddings (genes * bins + special tokens)
        self.n_tokens = self.vocab_size * n_bins + 10  # +10 for special tokens
        self.token_embeddings = nn.Embedding(self.n_tokens, d_model)

        # Special token IDs
        self.cls_token_id = self.n_tokens - 3
        self.pad_token_id = self.n_tokens - 2
        self.mask_token_id = self.n_tokens - 1

        # Position embeddings
        self.position_embeddings = nn.Embedding(4096, d_model)  # Max 4096 genes

        # Layer norm
        self.ln_input = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
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

        # Load pretrained weights if available
        if pretrained_path and Path(pretrained_path).exists():
            self._load_pretrained_model(pretrained_path)

        # Freeze layers if specified
        self._freeze_layers()

    def _load_gene_vocab(self, vocab_path: Optional[str], num_genes: int) -> Dict[str, int]:
        """Load gene vocabulary from file or create default."""
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            logger.info(f"Loaded gene vocabulary with {len(vocab)} genes")
        else:
            # Create dummy vocabulary (will be replaced with actual genes)
            vocab = {f"GENE_{i}": i for i in range(min(num_genes, 2000))}
            logger.warning(f"Using dummy gene vocabulary with {len(vocab)} genes")
        return vocab

    def _create_bin_edges(self, n_bins: int) -> torch.Tensor:
        """Create expression bin edges for discretization."""
        # Create log-spaced bins for expression values
        # Assuming log-normalized data roughly in range [0, 10]
        edges = torch.linspace(0, 10, n_bins + 1)
        return edges

    def _initialize_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _load_pretrained_model(self, checkpoint_path: str):
        """Load pretrained scGPT model."""
        logger.info(f"Loading pretrained model from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Load transformer weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Filter for transformer weights
                transformer_state = {
                    k.replace('transformer.', ''): v
                    for k, v in state_dict.items()
                    if 'transformer' in k
                }
                if transformer_state:
                    self.transformer.load_state_dict(transformer_state, strict=False)

            # Load embeddings if available
            if 'token_embeddings' in checkpoint:
                self.token_embeddings.load_state_dict(
                    checkpoint['token_embeddings'], strict=False
                )

            logger.info("Pretrained weights loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load pretrained model: {e}")

    def _freeze_layers(self):
        """Freeze specified number of bottom transformer layers."""
        if self.freeze_layers > 0:
            # Freeze embeddings
            for param in self.token_embeddings.parameters():
                param.requires_grad = False
            for param in self.position_embeddings.parameters():
                param.requires_grad = False

            # Freeze transformer layers
            if hasattr(self, 'transformer'):
                layers = self.transformer.layers
                for i in range(min(self.freeze_layers, len(layers))):
                    for param in layers[i].parameters():
                        param.requires_grad = False

            logger.info(f"Froze {self.freeze_layers} transformer layers and embeddings")

    def tokenize_expression(
        self,
        expression_matrix: torch.Tensor,
        gene_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize expression matrix into token IDs.

        Args:
            expression_matrix: Expression values (batch_size, n_genes)
            gene_indices: Indices of genes in vocabulary (batch_size, n_genes)

        Returns:
            token_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        """
        batch_size, n_genes = expression_matrix.shape
        device = expression_matrix.device

        # Use gene indices if provided, otherwise use sequential indices
        if gene_indices is None:
            gene_indices = torch.arange(
                min(n_genes, self.vocab_size), device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # Clamp to valid vocab size
        gene_indices = torch.clamp(gene_indices, 0, self.vocab_size - 1)

        # Discretize expression values into bins
        expression_bins = torch.bucketize(
            expression_matrix, self.bin_edges.to(device)
        )
        expression_bins = torch.clamp(expression_bins, 0, self.n_bins - 1)

        # Create token IDs: gene_id * n_bins + bin_id
        token_ids = gene_indices * self.n_bins + expression_bins

        # Add CLS token at the beginning
        cls_tokens = torch.full(
            (batch_size, 1), self.cls_token_id,
            dtype=torch.long, device=device
        )
        token_ids = torch.cat([cls_tokens, token_ids], dim=1)

        # Create attention mask (all ones for valid tokens)
        attention_mask = torch.ones_like(token_ids)

        return token_ids, attention_mask

    def forward(self, expression_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            expression_matrix: Expression values (batch_size, n_genes)

        Returns:
            Logits for binary classification (batch_size, 1)
        """
        batch_size = expression_matrix.size(0)
        device = expression_matrix.device

        # Tokenize expression
        token_ids, attention_mask = self.tokenize_expression(expression_matrix)

        # Get token embeddings
        token_embeds = self.token_embeddings(token_ids)

        # Add position embeddings
        positions = torch.arange(token_ids.size(1), device=device)
        position_embeds = self.position_embeddings(positions)
        embeddings = token_embeds + position_embeds.unsqueeze(0)

        # Layer norm
        embeddings = self.ln_input(embeddings)

        # Create attention mask for transformer (1.0 = attend, 0.0 = ignore)
        # For padding tokens, we don't attend
        src_key_padding_mask = (attention_mask == 0)

        # Pass through transformer
        transformer_output = self.transformer(
            embeddings, src_key_padding_mask=src_key_padding_mask
        )

        # Extract CLS token representation
        cls_output = transformer_output[:, 0, :]

        # Classification
        logits = self.classifier(cls_output)

        return logits


class scGPTFineTuner:
    """Helper class for fine-tuning scGPT models."""

    def __init__(
        self,
        model: scGPTWrapper,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
    ):
        """
        Initialize fine-tuner.

        Args:
            model: scGPT model to fine-tune
            learning_rate: Learning rate for fine-tuning
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters()
                              if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    def create_optimizer_and_scheduler(
        self,
        num_training_steps: int,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """
        Create optimizer and learning rate scheduler.

        Args:
            num_training_steps: Total number of training steps

        Returns:
            Optimizer and scheduler
        """
        # Separate parameters for different weight decay
        no_decay = ["bias", "LayerNorm", "ln"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # AdamW optimizer
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Cosine schedule with warmup
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / \
                      float(max(1, num_training_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return optimizer, scheduler

    def align_gene_vocabulary(
        self,
        dataset_genes: List[str],
        model_genes: List[str],
    ) -> Dict[str, str]:
        """
        Align dataset genes with model vocabulary.

        Args:
            dataset_genes: Gene names from dataset
            model_genes: Gene names from model vocabulary

        Returns:
            Mapping from dataset genes to model genes
        """
        gene_mapping = {}

        # Direct matching (case-insensitive)
        model_genes_upper = {g.upper(): g for g in model_genes}

        for gene in dataset_genes:
            gene_upper = gene.upper()
            if gene_upper in model_genes_upper:
                gene_mapping[gene] = model_genes_upper[gene_upper]

        logger.info(f"Matched {len(gene_mapping)} / {len(dataset_genes)} genes")

        # Report unmatched genes
        unmatched = set(dataset_genes) - set(gene_mapping.keys())
        if unmatched:
            logger.warning(f"Unmatched genes: {len(unmatched)}")

        return gene_mapping
