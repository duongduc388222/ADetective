"""
PyTorch Dataset and DataLoader utilities for SEAAD data.

Implements memory-efficient lazy loading with optional caching
for handling large preprocessed H5AD files.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import anndata as ad
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse

logger = logging.getLogger(__name__)


class SEAADDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for SEAAD preprocessed data.

    Uses lazy loading to only load individual samples on-demand,
    with optional LRU caching for frequently accessed samples.
    Supports both in-memory and backed (memory-mapped) H5AD files.
    """

    def __init__(
        self,
        data_path: str,
        dtype: torch.dtype = torch.float32,
        use_cache: bool = False,
        cache_size: int = 128,
    ):
        """
        Initialize dataset with lazy loading.

        Args:
            data_path: Path to preprocessed H5AD file
            dtype: PyTorch data type (default: float32)
            use_cache: Whether to use LRU caching for samples (default: False)
            cache_size: Maximum number of samples to cache (default: 128)
        """
        self.data_path = Path(data_path)
        self.dtype = dtype
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.cache_order = []  # Track access order for LRU eviction

        logger.info(f"Loading dataset metadata from {data_path}")

        # Load metadata only (file will remain closed until accessed)
        try:
            # First try backed mode for efficiency
            adata = ad.read_h5ad(data_path, backed="r")
            logger.info("✓ Loaded with backed='r' (memory-mapped mode)")
        except Exception as e:
            logger.warning(f"Could not load with backed mode: {e}. Trying standard mode...")
            adata = ad.read_h5ad(data_path)
            logger.info("✓ Loaded with standard mode (full memory load)")

        # Store metadata
        self.n_samples = adata.n_obs
        self.n_features = adata.n_vars

        # Load labels into memory (small, needed for all operations)
        if "label" not in adata.obs:
            raise ValueError("Column 'label' not found in observations")

        self.y = torch.from_numpy(adata.obs["label"].values.astype(np.int64))

        # Store gene names for reference
        self.gene_names = adata.var_names.tolist() if hasattr(adata.var_names, 'tolist') else list(adata.var_names)

        # Close the backed file to save memory
        if hasattr(adata, 'file'):
            adata.file.close() if hasattr(adata.file, 'close') else None

        logger.info(f"Dataset metadata:")
        logger.info(f"  Samples: {self.n_samples:,}")
        logger.info(f"  Features: {self.n_features:,}")
        logger.info(f"  Labels shape: {self.y.shape}")
        logger.info(f"  Label distribution: {torch.bincount(self.y).numpy()}")
        logger.info(f"  Lazy loading: {'Enabled' if not use_cache else 'Enabled with caching'}")
        if use_cache:
            logger.info(f"  Cache size: {cache_size} samples")

    def _load_sample(self, idx: int) -> torch.Tensor:
        """
        Load single sample from disk.

        Args:
            idx: Sample index

        Returns:
            Expression tensor for the sample
        """
        # Try backed mode first (more efficient)
        try:
            adata = ad.read_h5ad(self.data_path, backed="r")
        except Exception:
            adata = ad.read_h5ad(self.data_path)

        # Extract single row
        X_row = adata.X[idx]

        # Close file
        if hasattr(adata, 'file'):
            adata.file.close() if hasattr(adata.file, 'close') else None

        # Convert sparse to dense if needed
        if sparse.issparse(X_row):
            X_row = X_row.toarray().squeeze()
        elif isinstance(X_row, np.matrix):
            X_row = np.asarray(X_row).squeeze()

        # Convert to tensor
        X_tensor = torch.from_numpy(X_row.astype(np.float32)).to(self.dtype)

        return X_tensor

    def _add_to_cache(self, idx: int, sample: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Add sample to LRU cache with automatic eviction.

        Args:
            idx: Sample index
            sample: Tuple of (expression, label)
        """
        if not self.use_cache:
            return

        # Add to cache
        self.cache[idx] = sample
        if idx in self.cache_order:
            self.cache_order.remove(idx)
        self.cache_order.append(idx)

        # Evict oldest if cache full
        if len(self.cache) > self.cache_size:
            oldest_idx = self.cache_order.pop(0)
            del self.cache[oldest_idx]

    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single sample with lazy loading and optional caching.

        Args:
            idx: Sample index

        Returns:
            Tuple of (expression, label)
        """
        # Check cache first
        if self.use_cache and idx in self.cache:
            # Update access order for LRU
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            return self.cache[idx]

        # Load from disk
        X = self._load_sample(idx)
        y = self.y[idx]

        sample = (X, y)

        # Add to cache if enabled
        if self.use_cache:
            self._add_to_cache(idx, sample)

        return sample


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_cache: bool = False,
    cache_size: int = 128,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders with lazy loading.

    Args:
        train_path: Path to training data H5AD file
        val_path: Path to validation data H5AD file
        test_path: Path to test data H5AD file
        batch_size: Batch size (default: 32)
        num_workers: Number of data loading workers (default: 0)
        pin_memory: Pin memory for faster transfer (default: True)
        use_cache: Whether to cache samples in Dataset (default: False)
        cache_size: Maximum samples to cache (default: 128)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info(f"Creating lazy-loading DataLoaders (batch_size={batch_size})")

    # Create datasets with lazy loading
    train_dataset = SEAADDataset(train_path, use_cache=use_cache, cache_size=cache_size)
    val_dataset = SEAADDataset(val_path, use_cache=use_cache, cache_size=cache_size)
    test_dataset = SEAADDataset(test_path, use_cache=use_cache, cache_size=cache_size)

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

    logger.info(f"Created lazy-loading DataLoaders:")
    logger.info(f"  Train: {len(train_loader)} batches ({train_dataset.n_samples:,} samples)")
    logger.info(f"  Val: {len(val_loader)} batches ({val_dataset.n_samples:,} samples)")
    logger.info(f"  Test: {len(test_loader)} batches ({test_dataset.n_samples:,} samples)")
    logger.info(f"  Memory usage: Only single batches + optional cache ({cache_size} samples max)")

    return train_loader, val_loader, test_loader
