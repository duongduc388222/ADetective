"""
PyTorch Dataset and DataLoader utilities for SEAAD data.

Handles loading preprocessed H5AD files and creating PyTorch tensors.
"""

import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import anndata as ad
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class SEAADDataset(Dataset):
    """
    PyTorch Dataset for SEAAD preprocessed data.

    Loads expression data and labels from H5AD file.
    """

    def __init__(
        self,
        data_path: str,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize dataset from H5AD file.

        Args:
            data_path: Path to preprocessed H5AD file
            dtype: PyTorch data type (default: float32)
        """
        logger.info(f"Loading dataset from {data_path}")

        # Load AnnData
        adata = ad.read_h5ad(data_path)
        logger.info(f"Loaded data: {adata.shape}")

        # Get expression data
        X = adata.X
        if hasattr(X, "toarray"):  # Sparse matrix
            X = X.toarray()
        X = X.astype(np.float32)

        # Get labels
        if "label" not in adata.obs:
            raise ValueError("Column 'label' not found in observations")

        y = adata.obs["label"].values.astype(np.int64)

        # Convert to tensors
        self.X = torch.from_numpy(X).to(dtype)
        self.y = torch.from_numpy(y)

        logger.info(f"Dataset shapes:")
        logger.info(f"  Expression: {self.X.shape}")
        logger.info(f"  Labels: {self.y.shape}")
        logger.info(f"  Label distribution: {np.bincount(y)}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (expression, label)
        """
        return self.X[idx], self.y[idx]


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.

    Args:
        train_path: Path to training data H5AD file
        val_path: Path to validation data H5AD file
        test_path: Path to test data H5AD file
        batch_size: Batch size (default: 32)
        num_workers: Number of data loading workers (default: 0)
        pin_memory: Pin memory for faster transfer (default: True)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info(f"Creating data loaders with batch_size={batch_size}")

    # Create datasets
    train_dataset = SEAADDataset(train_path)
    val_dataset = SEAADDataset(val_path)
    test_dataset = SEAADDataset(test_path)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
