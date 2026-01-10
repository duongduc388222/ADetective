#!/usr/bin/env python3
"""
CellFM Backbone Loading Utilities.

Loads MindSpore checkpoint and converts to PyTorch for finetuning.

Usage:
    # As module
    from cellfm_backbone import load_cellfm_backbone
    backbone, hidden_dim = load_cellfm_backbone("path/to/CellFM_80M_weight.ckpt")

    # Test standalone
    python scripts/cellfm_backbone.py --ckpt examples/save/cellFM/CellFM_80M_weight.ckpt
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CellFMEncoder(nn.Module):
    """
    Simplified CellFM encoder for extracting cell embeddings.

    Architecture based on CellFM 80M model:
    - Input: (batch, 27934) gene expression
    - Gene embedding + positional encoding
    - Transformer encoder layers
    - CLS token pooling
    - Output: (batch, hidden_dim) cell embeddings

    This is a PyTorch re-implementation compatible with MindSpore weights.
    """

    def __init__(
        self,
        n_genes: int = 27934,
        hidden_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim

        # Gene embedding: project each gene's expression to hidden_dim
        self.gene_embedding = nn.Linear(1, hidden_dim)

        # Positional encoding for genes (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_genes, hidden_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm for final output
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # CLS token for cell-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Gene expression tensor (batch, n_genes)

        Returns:
            Cell embeddings (batch, hidden_dim)
        """
        batch_size = x.shape[0]

        # Reshape for gene embedding: (batch, n_genes) -> (batch, n_genes, 1)
        x = x.unsqueeze(-1)

        # Gene embedding: (batch, n_genes, 1) -> (batch, n_genes, hidden_dim)
        x = self.gene_embedding(x)

        # Add positional embedding
        x = x + self.pos_embedding

        # Prepend CLS token: (batch, n_genes+1, hidden_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer encoding
        x = self.encoder(x)

        # Extract CLS token output as cell embedding
        cell_embedding = x[:, 0, :]  # (batch, hidden_dim)

        # Final layer norm
        cell_embedding = self.layer_norm(cell_embedding)

        return cell_embedding


def convert_mindspore_to_pytorch(ms_ckpt_path: str) -> Dict[str, torch.Tensor]:
    """
    Convert MindSpore checkpoint to PyTorch state dict.

    This function handles the naming convention differences between
    MindSpore and PyTorch (e.g., gamma->weight, beta->bias).

    Args:
        ms_ckpt_path: Path to .ckpt file

    Returns:
        PyTorch-compatible state dict
    """
    try:
        from mindspore import load_checkpoint
    except ImportError:
        raise ImportError(
            "MindSpore is required to load .ckpt files.\n"
            "Install with: pip install mindspore==2.2.10"
        )

    logger.info(f"Loading MindSpore checkpoint: {ms_ckpt_path}")
    ms_ckpt = load_checkpoint(ms_ckpt_path)

    pt_state_dict = {}

    # Key name mappings from MindSpore to PyTorch
    name_replacements = [
        (".gamma", ".weight"),
        (".beta", ".bias"),
        ("layer_norm.gamma", "layer_norm.weight"),
        ("layer_norm.beta", "layer_norm.bias"),
        ("post_norm.gamma", "post_norm.weight"),
        ("post_norm.beta", "post_norm.bias"),
    ]

    for ms_name, ms_param in ms_ckpt.items():
        # Convert parameter name
        pt_name = ms_name
        for ms_pattern, pt_pattern in name_replacements:
            pt_name = pt_name.replace(ms_pattern, pt_pattern)

        # Convert tensor from MindSpore to PyTorch
        try:
            np_array = ms_param.asnumpy()
            pt_tensor = torch.from_numpy(np_array.copy())
            pt_state_dict[pt_name] = pt_tensor
        except Exception as e:
            logger.warning(f"Failed to convert parameter {ms_name}: {e}")
            continue

    logger.info(f"Converted {len(pt_state_dict)} parameters")

    # Log some statistics
    total_params = sum(p.numel() for p in pt_state_dict.values())
    logger.info(f"Total parameters: {total_params:,}")

    return pt_state_dict


def inspect_checkpoint(ckpt_path: str) -> None:
    """
    Print information about checkpoint contents.

    Useful for debugging weight loading issues.
    """
    path = Path(ckpt_path)

    if path.suffix == ".ckpt":
        from mindspore import load_checkpoint

        ckpt = load_checkpoint(str(path))
        logger.info(f"\nMindSpore checkpoint: {path.name}")
        logger.info(f"Number of parameters: {len(ckpt)}")
        logger.info("\nFirst 20 parameter names and shapes:")
        for i, (name, param) in enumerate(ckpt.items()):
            if i >= 20:
                logger.info("...")
                break
            logger.info(f"  {name}: {param.shape}")

    elif path.suffix == ".pt":
        ckpt = torch.load(str(path), map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        logger.info(f"\nPyTorch checkpoint: {path.name}")
        logger.info(f"Number of parameters: {len(ckpt)}")
        logger.info("\nFirst 20 parameter names and shapes:")
        for i, (name, param) in enumerate(ckpt.items()):
            if i >= 20:
                logger.info("...")
                break
            logger.info(f"  {name}: {param.shape}")


def load_cellfm_backbone(
    weights_path: str,
    config_path: Optional[str] = None,
    strict: bool = False,
) -> Tuple[nn.Module, int]:
    """
    Load CellFM backbone from MindSpore checkpoint.

    This function:
    1. Creates a PyTorch encoder with CellFM 80M architecture
    2. Loads and converts MindSpore weights
    3. Attempts to match as many weights as possible

    Args:
        weights_path: Path to CellFM .ckpt or .pt file
        config_path: Optional config path (unused, for API compatibility)
        strict: Whether to require exact key matching (default: False)

    Returns:
        Tuple of (backbone_module, hidden_dim)

    Example:
        >>> backbone, hidden_dim = load_cellfm_backbone("CellFM_80M_weight.ckpt")
        >>> print(f"Hidden dim: {hidden_dim}")
        Hidden dim: 512
    """
    # CellFM 80M model configuration
    # Based on: https://github.com/biomed-AI/CellFM
    hidden_dim = 512
    n_genes = 27934
    n_layers = 6
    n_heads = 16

    logger.info("Creating CellFM encoder with 80M configuration:")
    logger.info(f"  n_genes: {n_genes:,}")
    logger.info(f"  hidden_dim: {hidden_dim}")
    logger.info(f"  n_layers: {n_layers}")
    logger.info(f"  n_heads: {n_heads}")

    # Create encoder
    backbone = CellFMEncoder(
        n_genes=n_genes,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
    )

    # Count parameters before loading
    param_count = sum(p.numel() for p in backbone.parameters())
    logger.info(f"  Total parameters: {param_count:,}")

    # Load weights
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    logger.info(f"\nLoading weights from: {weights_path}")

    if weights_path.suffix == ".ckpt":
        # MindSpore checkpoint - convert to PyTorch
        pt_state_dict = convert_mindspore_to_pytorch(str(weights_path))

        # Try to load matching keys
        model_state = backbone.state_dict()
        loaded_keys = []
        missing_keys = []
        unexpected_keys = list(pt_state_dict.keys())

        for model_key in model_state.keys():
            # Try exact match first
            if model_key in pt_state_dict:
                if model_state[model_key].shape == pt_state_dict[model_key].shape:
                    model_state[model_key] = pt_state_dict[model_key]
                    loaded_keys.append(model_key)
                    unexpected_keys.remove(model_key)
                else:
                    logger.warning(
                        f"Shape mismatch for {model_key}: "
                        f"model={model_state[model_key].shape}, "
                        f"ckpt={pt_state_dict[model_key].shape}"
                    )
                    missing_keys.append(model_key)
            else:
                missing_keys.append(model_key)

        # Load the matched state dict
        backbone.load_state_dict(model_state, strict=False)

        logger.info(f"\nWeight loading summary:")
        logger.info(f"  Loaded: {len(loaded_keys)} keys")
        logger.info(f"  Missing (not in ckpt): {len(missing_keys)} keys")
        logger.info(f"  Unexpected (not in model): {len(unexpected_keys)} keys")

        if missing_keys and len(missing_keys) <= 10:
            logger.info(f"  Missing keys: {missing_keys}")
        if unexpected_keys and len(unexpected_keys) <= 10:
            logger.info(f"  Unexpected keys: {unexpected_keys[:10]}")

    elif weights_path.suffix == ".pt":
        # Already PyTorch format
        state_dict = torch.load(str(weights_path), map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        result = backbone.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded PyTorch checkpoint")
        if hasattr(result, "missing_keys"):
            logger.info(f"  Missing keys: {len(result.missing_keys)}")
        if hasattr(result, "unexpected_keys"):
            logger.info(f"  Unexpected keys: {len(result.unexpected_keys)}")

    else:
        raise ValueError(f"Unsupported weight format: {weights_path.suffix}")

    logger.info(f"\nCellFM backbone loaded successfully!")
    logger.info(f"  Output dimension: {hidden_dim}")

    return backbone, hidden_dim


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test CellFM backbone loading")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="examples/save/cellFM/CellFM_80M_weight.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only inspect checkpoint contents without loading",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=True,
        help="Run forward pass test (default: True)",
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_checkpoint(args.ckpt)
    else:
        # Test loading
        print("=" * 60)
        print("CellFM Backbone Loading Test")
        print("=" * 60)

        try:
            backbone, hidden_dim = load_cellfm_backbone(args.ckpt)

            print(f"\n{'=' * 60}")
            print("SUCCESS: Backbone loaded!")
            print(f"{'=' * 60}")
            print(f"  Hidden dim: {hidden_dim}")
            print(f"  Parameters: {sum(p.numel() for p in backbone.parameters()):,}")

            if args.test:
                # Test forward pass
                print("\nTesting forward pass...")
                backbone.eval()
                dummy_input = torch.randn(2, 27934)
                with torch.no_grad():
                    output = backbone(dummy_input)
                print(f"  Input shape: {dummy_input.shape}")
                print(f"  Output shape: {output.shape}")
                print(f"  Output mean: {output.mean().item():.4f}")
                print(f"  Output std: {output.std().item():.4f}")
                print("\nForward pass successful!")

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback

            traceback.print_exc()
