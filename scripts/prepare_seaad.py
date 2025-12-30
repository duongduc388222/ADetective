#!/usr/bin/env python3
"""
SEAAD Data Preparation Pipeline for Oligodendrocyte AD Classification.

This script prepares the SEAAD A9 RNAseq dataset for binary classification:
  - Label 1 (High): Cells from donors with High AD Neuropathologic Change (ADNC)
  - Label 0 (Not AD): Cells from donors with Not AD status

Pipeline steps:
  1. Load raw SEAAD h5ad file (memory-efficient backed mode)
  2. Filter for Oligodendrocyte cells only
  3. Filter for High and Not AD categories (exclude Low, Intermediate)
  4. Create binary labels
  5. Perform donor-level stratified train/val/test split (prevents data leakage)
  6. Normalize each split independently (library-size + log1p)
  7. Select highly variable genes from training data only
  8. Save processed datasets to output directory

Output format compatible with train_mlp.py and train_transformer.py:
  - train.h5ad, val.h5ad, test.h5ad
  - obs["label"]: Binary labels (0 or 1)
  - Metadata columns preserved for --use-metadata option

Usage:
    python scripts/prepare_seaad.py \\
        --input-path /path/to/SEAAD_A9_RNAseq_DREAM.h5ad \\
        --output-dir ./data

    # With custom parameters
    python scripts/prepare_seaad.py \\
        --input-path /path/to/SEAAD_A9_RNAseq_DREAM.h5ad \\
        --output-dir ./data \\
        --n-hvgs 2000 \\
        --train-ratio 0.7 \\
        --val-ratio 0.15 \\
        --test-ratio 0.15

    # With genemap vocabulary filtering (ensures 100% gene coverage for finetuning)
    python scripts/prepare_seaad.py \\
        --input-path /path/to/SEAAD_A9_RNAseq_DREAM.h5ad \\
        --output-dir ./data \\
        --genemap ./data/genemap.csv \\
        --n-hvgs 2000
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare SEAAD dataset for Oligodendrocyte AD classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/prepare_seaad.py \\
    --input-path /path/to/SEAAD_A9_RNAseq_DREAM.h5ad \\
    --output-dir ./data

  # Custom split ratios
  python scripts/prepare_seaad.py \\
    --input-path /path/to/SEAAD_A9_RNAseq_DREAM.h5ad \\
    --output-dir ./data \\
    --train-ratio 0.7 \\
    --val-ratio 0.15 \\
    --test-ratio 0.15

  # More HVGs
  python scripts/prepare_seaad.py \\
    --input-path /path/to/SEAAD_A9_RNAseq_DREAM.h5ad \\
    --output-dir ./data \\
    --n-hvgs 3000
        """,
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the raw SEAAD H5AD file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save processed data (default: ./data)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of donors for training set (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of donors for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of donors for test set (default: 0.15)",
    )
    parser.add_argument(
        "--n-hvgs",
        type=int,
        default=2000,
        help="Number of highly variable genes to select (default: 2000)",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=100,
        help="Minimum number of cells a gene must be expressed in (default: 100)",
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=1e4,
        help="Target sum for library-size normalization (default: 10000)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--cell-type",
        type=str,
        default="Oligodendrocyte",
        help="Cell type to filter for (default: Oligodendrocyte)",
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip normalization (if data is already normalized)",
    )
    parser.add_argument(
        "--genemap",
        type=str,
        default=None,
        help="Path to genemap.csv for vocabulary filtering (ensures 100%% coverage for finetuning)",
    )
    parser.add_argument(
        "--genemap-min-coverage",
        type=float,
        default=0.3,
        help="Minimum vocabulary coverage threshold (default: 0.3 = 30%%)",
    )
    parser.add_argument(
        "--foundation-model-mode",
        action="store_true",
        default=False,
        help="Enable Foundation Model Mode: expand to full genemap vocab (zero-padding) instead of HVG selection",
    )

    return parser.parse_args()


def load_filtered_data(
    input_path: str,
    cell_type: str = "Oligodendrocyte",
    cell_type_column: str = "Subclass",
    adnc_column: str = "ADNC",
    exclude_categories: Optional[List[str]] = None,
):
    """
    Load SEAAD dataset with pre-filtering to minimize memory usage.

    This function:
    1. Loads metadata (obs, var) first
    2. Determines which cells to keep based on cell type and ADNC status
    3. Only loads the relevant rows of X

    This approach is much more memory-efficient for large datasets.

    Args:
        input_path: Path to the h5ad file
        cell_type: Cell type to filter for
        cell_type_column: Column name for cell type
        adnc_column: Column name for ADNC status
        exclude_categories: ADNC categories to exclude

    Returns:
        Tuple of (AnnData object, label_mapping)
    """
    import anndata as ad
    from scipy import sparse

    if exclude_categories is None:
        exclude_categories = ["Low", "Intermediate"]

    logger.info(f"Loading SEAAD dataset from {input_path}")
    logger.info(f"Pre-filtering for: {cell_type} cells with High/Not AD status")

    with h5py.File(input_path, 'r') as f:
        # Step 1: Read metadata
        logger.info("Reading obs (cell metadata)...")
        try:
            from anndata._io.specs import read_elem
        except ImportError:
            from anndata.experimental import read_elem

        obs = read_elem(f['obs'])
        logger.info(f"  Total cells: {len(obs):,}")

        logger.info("Reading var (gene metadata)...")
        var = read_elem(f['var'])
        logger.info(f"  Total genes: {len(var):,}")

        # Step 2: Determine which cells to keep
        logger.info("Determining cells to keep...")

        # Cell type filter
        cell_type_mask = obs[cell_type_column].str.lower() == cell_type.lower()
        n_cell_type = cell_type_mask.sum()
        logger.info(f"  {cell_type} cells: {n_cell_type:,}")

        # ADNC filter
        adnc_mask = ~obs[adnc_column].isin(exclude_categories)
        n_adnc = adnc_mask.sum()
        logger.info(f"  High/Not AD cells: {n_adnc:,}")

        # Combined filter
        keep_mask = cell_type_mask & adnc_mask
        keep_indices = np.where(keep_mask)[0]
        n_keep = len(keep_indices)
        logger.info(f"  Cells to load: {n_keep:,}")

        if n_keep == 0:
            raise ValueError(f"No cells match the filter criteria!")

        # Step 3: Load only the relevant rows of X
        logger.info(f"Loading filtered X matrix ({n_keep:,} cells)...")
        X_grp = f['X']

        if 'indptr' not in X_grp:
            raise ValueError("Expected sparse CSR matrix in X")

        # Read full indptr (small)
        indptr = X_grp['indptr'][:]
        n_vars = var.shape[0]

        # Build new sparse matrix by reading only needed rows
        new_data = []
        new_indices = []
        new_indptr = [0]
        valid_cell_indices = []  # Track which cells were successfully read
        n_errors = 0

        chunk_size = 1000  # Smaller chunks for better error handling
        n_chunks = (n_keep + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, n_keep)
            chunk_keep = keep_indices[start:end]

            for cell_idx in chunk_keep:
                row_start = indptr[cell_idx]
                row_end = indptr[cell_idx + 1]

                if row_end > row_start:
                    try:
                        row_data = X_grp['data'][row_start:row_end]
                        row_indices = X_grp['indices'][row_start:row_end]
                        new_data.append(row_data)
                        new_indices.append(row_indices)
                        new_indptr.append(new_indptr[-1] + (row_end - row_start))
                        valid_cell_indices.append(cell_idx)
                    except OSError:
                        n_errors += 1
                        # Skip corrupted row
                        continue
                else:
                    # Empty row (no expression data)
                    new_indptr.append(new_indptr[-1])
                    valid_cell_indices.append(cell_idx)

            # Progress update
            if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
                progress = (end / n_keep) * 100
                logger.info(f"  Progress: {progress:.1f}% ({end:,}/{n_keep:,} cells, {n_errors} errors)")

        if n_errors > 0:
            logger.warning(f"  Skipped {n_errors} corrupted cells")

        # Update n_keep to actual number of valid cells
        n_valid = len(valid_cell_indices)
        logger.info(f"  Valid cells loaded: {n_valid:,}")

        # Concatenate arrays
        if new_data:
            all_data = np.concatenate(new_data)
            all_indices = np.concatenate(new_indices)
        else:
            all_data = np.array([], dtype=np.float32)
            all_indices = np.array([], dtype=np.int64)

        X = sparse.csr_matrix(
            (all_data, all_indices, np.array(new_indptr)),
            shape=(n_valid, n_vars)
        )
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  Non-zero elements: {X.nnz:,}")

    # Step 4: Filter obs to kept cells (using valid_cell_indices which excludes corrupted rows)
    obs_filtered = obs.iloc[valid_cell_indices].copy().reset_index(drop=True)

    # Step 5: Create labels
    labels = (obs_filtered[adnc_column] == "High").astype(int)
    obs_filtered["label"] = labels
    obs_filtered["label_name"] = labels.map({0: "Not AD", 1: "High"})

    label_mapping = {"Not AD": 0, "High": 1}

    # Create AnnData
    adata = ad.AnnData(X=X, obs=obs_filtered, var=var)
    logger.info(f"Created AnnData: {adata.shape}")
    logger.info(f"Label distribution:")
    logger.info(adata.obs["label_name"].value_counts().to_string())

    return adata, label_mapping


def filter_cell_type(adata, cell_type: str = "Oligodendrocyte", cell_type_column: str = "Subclass"):
    """
    Filter data for specific cell type.

    Args:
        adata: AnnData object (backed or in-memory)
        cell_type: Cell type to filter for
        cell_type_column: Column name for cell type

    Returns:
        Filtered AnnData object (in-memory)
    """
    logger.info(f"Filtering for {cell_type} cells using column '{cell_type_column}'")
    before_count = adata.n_obs

    # Verify column exists
    if cell_type_column not in adata.obs:
        raise ValueError(
            f"Column '{cell_type_column}' not found in observations.\n"
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Filter (case-insensitive)
    logger.info("Filtering...")
    mask = adata.obs[cell_type_column].str.lower() == cell_type.lower()

    # Handle both backed and in-memory AnnData
    if hasattr(adata, 'isbacked') and adata.isbacked:
        adata_filtered = adata[mask].to_memory()
    else:
        adata_filtered = adata[mask].copy()

    logger.info(f"Cells before filtering: {before_count:,}")
    logger.info(f"Cells after filtering:  {adata_filtered.n_obs:,}")

    return adata_filtered


def filter_adnc_and_create_labels(
    adata,
    adnc_column: str = "ADNC",
    exclude_categories: Optional[List[str]] = None,
) -> Tuple:
    """
    Filter for High and Not AD categories and create binary labels.

    Args:
        adata: AnnData object
        adnc_column: Column name for ADNC status
        exclude_categories: Categories to exclude (default: ['Low', 'Intermediate'])

    Returns:
        Tuple of (filtered AnnData, label mapping)
    """
    if exclude_categories is None:
        exclude_categories = ["Low", "Intermediate"]

    logger.info(f"Creating labels from column '{adnc_column}'")
    logger.info(f"Excluding categories: {exclude_categories}")

    # Verify column exists
    if adnc_column not in adata.obs:
        raise ValueError(
            f"Column '{adnc_column}' not found in observations.\n"
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Show ADNC value distribution before filtering
    logger.info("ADNC value distribution before filtering:")
    logger.info(adata.obs[adnc_column].value_counts().to_string())

    # Filter out excluded categories
    logger.info("Filtering by ADNC status...")
    mask = ~adata.obs[adnc_column].isin(exclude_categories)
    adata_filtered = adata[mask].copy()

    logger.info(f"Cells retained: {adata_filtered.n_obs:,} / {adata.n_obs:,}")

    # Create binary labels: High=1, Not AD=0
    label_mapping = {"Not AD": 0, "High": 1}
    labels = (adata_filtered.obs[adnc_column] == "High").astype(int)

    adata_filtered.obs["label"] = labels
    adata_filtered.obs["label_name"] = labels.map({0: "Not AD", 1: "High"})

    logger.info("\nLabel distribution:")
    logger.info(adata_filtered.obs["label_name"].value_counts().to_string())

    return adata_filtered, label_mapping


def stratified_donor_split(
    adata,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    donor_column: str = "Donor ID",
) -> Tuple:
    """
    Create donor-level stratified train/val/test split.

    Ensures:
    - No donor appears in multiple splits
    - Each split maintains balanced label distribution

    Args:
        adata: AnnData object with labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
        donor_column: Column name for donor ID

    Returns:
        Tuple of (train_adata, val_adata, test_adata)
    """
    if "label" not in adata.obs:
        raise ValueError("Labels not created yet. Call filter_adnc_and_create_labels() first.")

    if donor_column not in adata.obs:
        raise ValueError(
            f"Column '{donor_column}' not found in observations.\n"
            f"Available columns: {list(adata.obs.columns)}"
        )

    logger.info("=" * 60)
    logger.info("DONOR-LEVEL STRATIFIED SPLIT")
    logger.info("=" * 60)
    logger.info(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    logger.info(f"Using donor column: {donor_column}")

    # Get donor-level label (majority vote per donor)
    donor_labels = (
        adata.obs.groupby(donor_column)["label"]
        .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
        .reset_index()
    )
    donor_labels.columns = [donor_column, "donor_label"]

    # Ensure string types for stratification
    donor_labels[donor_column] = donor_labels[donor_column].astype(str)
    donor_labels["donor_label"] = donor_labels["donor_label"].astype(str)

    logger.info(f"\nTotal unique donors: {len(donor_labels)}")
    logger.info("Donor-level label distribution:")
    label_dist = donor_labels["donor_label"].value_counts()
    for label, count in label_dist.items():
        label_name = "High" if label == "1" else "Not AD"
        logger.info(f"  {label_name}: {count} donors")

    # Add cell count per donor for weighted stratification
    donor_cell_counts = adata.obs.groupby(donor_column).size()
    donor_labels["cell_count"] = donor_labels[donor_column].map(donor_cell_counts)

    # Categorize donors by cell count (low/medium/high)
    try:
        donor_labels["count_bucket"] = pd.qcut(
            donor_labels["cell_count"],
            q=3,
            labels=["low", "med", "high"],
            duplicates="drop",
        ).astype(str)

        # Create combined stratification key
        donor_labels["strat_key"] = (
            donor_labels["donor_label"].astype(str)
            + "_"
            + donor_labels["count_bucket"].astype(str)
        )
        use_cell_count_strat = True
    except ValueError:
        # Not enough unique values for qcut
        logger.warning("Not enough donors for cell-count stratification, using label-only")
        use_cell_count_strat = False
        donor_labels["strat_key"] = donor_labels["donor_label"].astype(str)

    logger.info(f"\nStratification key distribution:")
    logger.info(donor_labels["strat_key"].value_counts().to_string())

    # First split: train+val vs test
    try:
        donors_train_val, donors_test = train_test_split(
            donor_labels[donor_column],
            test_size=test_ratio,
            random_state=random_state,
            stratify=donor_labels["strat_key"],
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}), falling back to label-only stratification")
        donors_train_val, donors_test = train_test_split(
            donor_labels[donor_column],
            test_size=test_ratio,
            random_state=random_state,
            stratify=donor_labels["donor_label"],
        )

    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_val_labels = donor_labels[donor_labels[donor_column].isin(donors_train_val)]

    try:
        donors_train, donors_val = train_test_split(
            train_val_labels[donor_column],
            test_size=val_size,
            random_state=random_state,
            stratify=train_val_labels["strat_key"],
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}), falling back to label-only stratification")
        donors_train, donors_val = train_test_split(
            train_val_labels[donor_column],
            test_size=val_size,
            random_state=random_state,
            stratify=train_val_labels["donor_label"],
        )

    # Create splits
    logger.info("\nCreating train/val/test splits...")
    train_adata = adata[adata.obs[donor_column].isin(donors_train)].copy()
    val_adata = adata[adata.obs[donor_column].isin(donors_val)].copy()
    test_adata = adata[adata.obs[donor_column].isin(donors_test)].copy()

    logger.info(f"Train set: {train_adata.n_obs:,} cells from {donors_train.nunique()} donors")
    logger.info(f"Val set:   {val_adata.n_obs:,} cells from {donors_val.nunique()} donors")
    logger.info(f"Test set:  {test_adata.n_obs:,} cells from {donors_test.nunique()} donors")

    # Validate label distributions
    logger.info("\n" + "=" * 60)
    logger.info("LABEL DISTRIBUTION VALIDATION")
    logger.info("=" * 60)

    for split_name, split_data in [
        ("Train", train_adata),
        ("Val", val_adata),
        ("Test", test_adata),
    ]:
        label_dist = split_data.obs["label_name"].value_counts()
        label_ratios = label_dist / label_dist.sum()

        logger.info(f"\n{split_name} set:")
        logger.info(f"  Counts: {label_dist.to_dict()}")
        logger.info(f"  Ratios: {dict((k, f'{v:.1%}') for k, v in label_ratios.items())}")

        # Check for single-class splits
        if len(label_dist) < 2:
            raise ValueError(
                f"CRITICAL ERROR: {split_name} set has only one class!\n"
                f"Current distribution: {label_dist.to_dict()}\n"
                f"This will cause invalid evaluation."
            )

    return train_adata, val_adata, test_adata


def filter_genes(adata, min_cells: int = 100):
    """
    Filter genes expressed in fewer than min_cells.

    Args:
        adata: AnnData object
        min_cells: Minimum number of cells a gene must be expressed in

    Returns:
        Filtered AnnData object
    """
    if min_cells <= 0:
        logger.info(f"Skipping gene filtering (min_cells={min_cells})")
        return adata

    logger.info(f"Filtering genes (min_cells={min_cells})")
    before_count = adata.n_vars

    sc.pp.filter_genes(adata, min_cells=min_cells)

    logger.info(f"Genes before filtering: {before_count:,}")
    logger.info(f"Genes after filtering:  {adata.n_vars:,}")
    logger.info(f"Genes removed: {before_count - adata.n_vars:,}")

    return adata


def filter_to_genemap(
    adata,
    genemap_path: str,
    min_coverage: float = 0.3,
) -> Tuple:
    """
    Filter genes to only those in the genemap vocabulary.

    This ensures 100% coverage for finetuning models that use this vocabulary.

    Args:
        adata: AnnData object
        genemap_path: Path to genemap.csv file
        min_coverage: Minimum coverage threshold (default: 0.3 = 30%)

    Returns:
        Tuple of (filtered AnnData, coverage report dict)

    Raises:
        FileNotFoundError: If genemap file doesn't exist
        ValueError: If coverage is below min_coverage threshold
    """
    logger.info(f"Filtering genes to genemap vocabulary: {genemap_path}")

    # Load genemap vocabulary
    if not os.path.exists(genemap_path):
        raise FileNotFoundError(f"Genemap file not found: {genemap_path}")

    genemap_df = pd.read_csv(genemap_path)

    # The first column contains gene names (may have unnamed header)
    gene_col = genemap_df.columns[0]
    vocab_genes = set(genemap_df[gene_col].astype(str).str.strip())
    logger.info(f"  Vocabulary size: {len(vocab_genes):,} genes")

    # Get current genes in data
    data_genes = set(adata.var_names.astype(str))
    logger.info(f"  Data genes: {len(data_genes):,}")

    # Find intersection
    intersection = data_genes & vocab_genes
    logger.info(f"  Intersection: {len(intersection):,} genes")

    # Calculate coverage
    coverage = len(intersection) / len(vocab_genes) if vocab_genes else 0
    logger.info(f"  Coverage: {coverage:.1%} of vocabulary")

    # Check minimum coverage
    if coverage < min_coverage:
        raise ValueError(
            f"Vocabulary coverage ({coverage:.1%}) is below minimum threshold ({min_coverage:.0%}). "
            f"Only {len(intersection):,} / {len(vocab_genes):,} vocabulary genes found in data."
        )

    # Find genes to drop (in data but not in vocab)
    oov_genes = data_genes - vocab_genes
    logger.info(f"  OOV genes to drop: {len(oov_genes):,}")

    # Filter to intersection
    genes_to_keep = [g for g in adata.var_names if g in intersection]
    adata_filtered = adata[:, genes_to_keep].copy()

    logger.info(f"  Genes after filtering: {adata_filtered.n_vars:,}")

    # Create coverage report
    report = {
        "genemap_path": genemap_path,
        "vocab_size": len(vocab_genes),
        "data_genes": len(data_genes),
        "intersection": len(intersection),
        "coverage": coverage,
        "oov_genes_dropped": len(oov_genes),
        "genes_retained": adata_filtered.n_vars,
    }

    return adata_filtered, report


def expand_to_genemap_vocab(
    adata,
    genemap_path: str,
) -> Tuple:
    """
    Expand/reindex AnnData to match genemap vocabulary exactly.

    This function performs vocabulary alignment for foundation models:
    - Genes in Data ∩ Genemap → Keep values
    - Genes in Genemap but NOT in Data → Fill with 0 (zero-padding)
    - Genes in Data but NOT in Genemap → Drop

    Uses sparse operations for memory efficiency.

    Args:
        adata: AnnData object
        genemap_path: Path to genemap.csv file

    Returns:
        Tuple of (expanded AnnData, coverage report dict)

    Raises:
        FileNotFoundError: If genemap file doesn't exist
    """
    import anndata as ad
    from scipy import sparse

    logger.info(f"Expanding to genemap vocabulary: {genemap_path}")

    # 1. Load genemap and get ordered gene list
    if not os.path.exists(genemap_path):
        raise FileNotFoundError(f"Genemap file not found: {genemap_path}")

    genemap_df = pd.read_csv(genemap_path)
    gene_col = genemap_df.columns[0]
    genemap_genes = genemap_df[gene_col].astype(str).str.strip().tolist()
    vocab_size = len(genemap_genes)
    logger.info(f"  Vocabulary size: {vocab_size:,} genes")

    # 2. Find intersection and coverage
    data_genes = set(adata.var_names.astype(str))
    genemap_genes_set = set(genemap_genes)
    intersection = data_genes & genemap_genes_set
    coverage = len(intersection) / vocab_size if vocab_size > 0 else 0

    logger.info(f"  Data genes: {len(data_genes):,}")
    logger.info(f"  Intersection: {len(intersection):,} genes")
    logger.info(f"  Coverage: {coverage:.1%} of vocabulary")

    # 3. Create gene-to-index mappings
    genemap_idx = {g: i for i, g in enumerate(genemap_genes)}
    data_gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    # 4. Build new sparse matrix with genemap column order
    n_cells = adata.n_obs
    logger.info(f"  Building expanded matrix: ({n_cells:,}, {vocab_size:,})")

    # Use lil_matrix for efficient column assignment
    X_new = sparse.lil_matrix((n_cells, vocab_size), dtype=np.float32)

    # Get source matrix (ensure it's sparse for efficiency)
    if sparse.issparse(adata.X):
        X_src = adata.X.tocsc()  # CSC for efficient column access
    else:
        X_src = sparse.csc_matrix(adata.X)

    # Copy data for genes that exist in both
    logger.info(f"  Copying {len(intersection):,} genes from source data...")
    for gene in intersection:
        src_idx = data_gene_to_idx[gene]
        dst_idx = genemap_idx[gene]
        # Get column from source, assign to destination
        col_data = X_src.getcol(src_idx).toarray().flatten()
        X_new[:, dst_idx] = col_data.reshape(-1, 1)

    # Convert to CSR for efficient row access and storage
    logger.info("  Converting to CSR format for storage...")
    X_new = X_new.tocsr()

    # 5. Create new AnnData with genemap var_names
    var_new = pd.DataFrame(index=genemap_genes)
    adata_expanded = ad.AnnData(
        X=X_new,
        obs=adata.obs.copy(),
        var=var_new,
    )

    logger.info(f"  Expanded AnnData shape: {adata_expanded.shape}")
    logger.info(f"  Non-zero elements: {X_new.nnz:,}")

    # 6. Build coverage report
    report = {
        "genemap_path": genemap_path,
        "vocab_size": vocab_size,
        "data_genes": len(data_genes),
        "intersection": len(intersection),
        "coverage": coverage,
        "zero_padded_genes": vocab_size - len(intersection),
        "dropped_genes": len(data_genes) - len(intersection),
    }

    return adata_expanded, report


def normalize_data(adata, target_sum: float = 1e4):
    """
    Normalize RNA-seq data to library size and log-transform.

    Args:
        adata: AnnData object
        target_sum: Target sum for library size normalization

    Returns:
        Normalized and log-transformed AnnData object
    """
    logger.info(f"Normalizing data to target_sum={target_sum} and log-transforming...")

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # Track preprocessing history
    if "preprocessing_history" not in adata.uns:
        adata.uns["preprocessing_history"] = {}
    adata.uns["preprocessing_history"]["normalized"] = True
    adata.uns["preprocessing_history"]["log_transformed"] = True
    adata.uns["preprocessing_history"]["normalization_target"] = target_sum

    return adata


def select_hvgs(adata, n_hvgs: int = 2000):
    """
    Select highly variable genes.

    Args:
        adata: AnnData object
        n_hvgs: Number of HVGs to select

    Returns:
        AnnData with HVGs selected
    """
    logger.info(f"Selecting {n_hvgs} highly variable genes")

    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs)
    adata_hvg = adata[:, adata.var.highly_variable].copy()

    logger.info(f"Selected {adata_hvg.n_vars:,} highly variable genes")

    return adata_hvg


def save_processed(adata, output_path: str):
    """
    Save processed AnnData.

    Args:
        adata: AnnData object to save
        output_path: Path to save to
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    adata.write_h5ad(output_path)
    logger.info(f"Saved processed data to {output_path}")


def main():
    """Main entry point for SEAAD data preparation."""
    args = parse_arguments()

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (abs(total_ratio - 1.0) < 0.01):
        logger.error(f"Split ratios must sum to 1.0, got {total_ratio}")
        return False

    logger.info("=" * 80)
    logger.info("SEAAD DATA PREPARATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"\nInput path: {args.input_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Cell type: {args.cell_type}")
    logger.info(f"Binary target: High (1) vs Not AD (0)")
    logger.info(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    if args.foundation_model_mode:
        logger.info(f"Mode: FOUNDATION MODEL MODE (expand to full genemap vocabulary)")
        if not args.genemap:
            logger.error("--genemap is required when using --foundation-model-mode")
            return False
        logger.info(f"Genemap vocabulary: {args.genemap}")
    else:
        logger.info(f"Mode: HVG Selection ({args.n_hvgs} highly variable genes)")
        if args.genemap:
            logger.info(f"Genemap vocabulary: {args.genemap}")
            logger.info(f"  (Pre-filtering genes to genemap for 100% coverage)")
    logger.info(f"Random state: {args.random_state}")

    # Validate input file exists
    if not os.path.exists(args.input_path):
        logger.error(f"Input file not found: {args.input_path}")
        return False

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1-3 Combined: Load data with pre-filtering (memory-efficient)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1-3: Loading and Filtering Data (Memory-Efficient)")
    logger.info("=" * 80)
    logger.info(f"  Cell type filter: {args.cell_type}")
    logger.info(f"  ADNC filter: Keep High and Not AD only")
    try:
        adata, label_mapping = load_filtered_data(
            args.input_path,
            cell_type=args.cell_type,
            cell_type_column="Subclass",
            adnc_column="ADNC",
            exclude_categories=["Low", "Intermediate"],
        )
    except Exception as e:
        logger.error(f"Failed to load/filter data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Filter genes (min_cells)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Filtering Genes")
    logger.info("=" * 80)
    try:
        adata = filter_genes(adata, min_cells=args.min_cells)
    except Exception as e:
        logger.error(f"Failed to filter genes: {e}")
        return False

    # Step 4.5: Filter to genemap vocabulary (optional, for HVG mode only)
    # In foundation-model-mode, we skip this step and expand to full vocab in Step 7
    genemap_report = None
    if args.genemap and not args.foundation_model_mode:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4.5: Filtering to Genemap Vocabulary")
        logger.info("=" * 80)
        logger.info("This ensures 100% gene coverage for finetuning models")
        try:
            adata, genemap_report = filter_to_genemap(
                adata,
                genemap_path=args.genemap,
                min_coverage=args.genemap_min_coverage,
            )
            logger.info(f"Vocabulary coverage: {genemap_report['coverage']:.1%}")
            logger.info(f"Genes retained: {genemap_report['genes_retained']:,}")
        except FileNotFoundError as e:
            logger.error(f"Genemap file not found: {e}")
            return False
        except ValueError as e:
            logger.error(f"Vocabulary coverage too low: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to filter to genemap: {e}")
            import traceback
            traceback.print_exc()
            return False
    elif args.foundation_model_mode:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4.5: Skipping Genemap Filter (Foundation Model Mode)")
        logger.info("=" * 80)
        logger.info("Vocabulary expansion will be performed in Step 7")

    # Step 5: Donor-level stratified split (BEFORE normalization)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Donor-Level Stratified Split")
    logger.info("=" * 80)
    try:
        train_data, val_data, test_data = stratified_donor_split(
            adata,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_state,
        )
    except Exception as e:
        logger.error(f"Failed to create splits: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 6: Normalize each split independently (AFTER split, BEFORE HVG)
    if not args.skip_normalization:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Normalizing Data (Independent per Split)")
        logger.info("=" * 80)
        try:
            logger.info("Normalizing training set...")
            train_data = normalize_data(train_data, target_sum=args.target_sum)

            logger.info("Normalizing validation set...")
            val_data = normalize_data(val_data, target_sum=args.target_sum)

            logger.info("Normalizing test set...")
            test_data = normalize_data(test_data, target_sum=args.target_sum)

            logger.info("All splits normalized independently")
        except Exception as e:
            logger.error(f"Failed to normalize data: {e}")
            return False
    else:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Skipping Normalization (--skip-normalization flag set)")
        logger.info("=" * 80)

    # Step 7: HVG Selection OR Vocabulary Expansion (Foundation Model Mode)
    vocab_expansion_report = None
    if args.foundation_model_mode:
        # Foundation Model Mode: Expand to full genemap vocabulary with zero-padding
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Expanding to Genemap Vocabulary (Foundation Model Mode)")
        logger.info("=" * 80)
        try:
            logger.info("Expanding training set to genemap vocabulary...")
            train_data, vocab_expansion_report = expand_to_genemap_vocab(train_data, args.genemap)

            logger.info("Expanding validation set to genemap vocabulary...")
            val_data, _ = expand_to_genemap_vocab(val_data, args.genemap)

            logger.info("Expanding test set to genemap vocabulary...")
            test_data, _ = expand_to_genemap_vocab(test_data, args.genemap)

            logger.info(f"\nVocabulary expansion complete:")
            logger.info(f"  Final shape: ({train_data.n_obs:,}, {train_data.n_vars:,})")
            logger.info(f"  Coverage: {vocab_expansion_report['coverage']:.1%}")
            logger.info(f"  Zero-padded genes: {vocab_expansion_report['zero_padded_genes']:,}")
            logger.info(f"  Dropped OOV genes: {vocab_expansion_report['dropped_genes']:,}")

            # Verify all splits have identical gene sets
            assert train_data.n_vars == val_data.n_vars == test_data.n_vars, \
                "Gene counts must be identical across splits!"
            logger.info(f"Verified: All splits have identical {train_data.n_vars:,} genes")

        except FileNotFoundError as e:
            logger.error(f"Genemap file not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to expand to genemap vocabulary: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        # Original HVG mode: Select highly variable genes
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Selecting Highly Variable Genes")
        logger.info("=" * 80)
        try:
            logger.info(f"Selecting {args.n_hvgs} HVGs from normalized training data...")
            train_data = select_hvgs(train_data, n_hvgs=args.n_hvgs)
            hvg_genes = train_data.var_names.copy()

            logger.info(f"Selected {len(hvg_genes)} HVGs from training set")
            logger.info("Subsetting validation and test sets to use the same genes...")

            # Subset val and test to use the SAME genes as training
            val_data = val_data[:, hvg_genes].copy()
            test_data = test_data[:, hvg_genes].copy()

            logger.info(f"Val data subset to {val_data.n_vars} genes")
            logger.info(f"Test data subset to {test_data.n_vars} genes")

            # Verify gene overlap
            assert set(train_data.var_names) == set(val_data.var_names) == set(test_data.var_names), \
                "Gene sets must be identical across splits!"
            logger.info(f"Verified: All splits use identical {len(hvg_genes)} genes")

        except Exception as e:
            logger.error(f"Failed to select HVGs: {e}")
            return False

    # Step 8: Save processed data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Saving Processed Data")
    logger.info("=" * 80)
    try:
        train_path = str(output_dir / "train.h5ad")
        val_path = str(output_dir / "val.h5ad")
        test_path = str(output_dir / "test.h5ad")

        save_processed(train_data, train_path)
        save_processed(val_data, val_path)
        save_processed(test_data, test_path)

    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        return False

    # Save preparation report
    report = {
        "input_path": args.input_path,
        "output_dir": str(output_dir),
        "cell_type": args.cell_type,
        "label_mapping": label_mapping,
        "foundation_model_mode": args.foundation_model_mode,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "random_state": args.random_state,
        "train": {
            "n_cells": train_data.n_obs,
            "n_genes": train_data.n_vars,
            "label_distribution": train_data.obs["label_name"].value_counts().to_dict(),
        },
        "val": {
            "n_cells": val_data.n_obs,
            "n_genes": val_data.n_vars,
            "label_distribution": val_data.obs["label_name"].value_counts().to_dict(),
        },
        "test": {
            "n_cells": test_data.n_obs,
            "n_genes": test_data.n_vars,
            "label_distribution": test_data.obs["label_name"].value_counts().to_dict(),
        },
    }

    # Add mode-specific info to report
    if args.foundation_model_mode:
        report["vocab_expansion"] = vocab_expansion_report
    else:
        report["n_hvgs"] = args.n_hvgs

    # Add genemap info to report if used (HVG mode only)
    if genemap_report is not None:
        report["genemap"] = genemap_report

    report_path = output_dir / "preparation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved preparation report to {report_path}")

    # Save genemap filter report separately if used
    if genemap_report is not None:
        genemap_report_path = output_dir / "genemap_filter_report.json"
        with open(genemap_report_path, "w") as f:
            json.dump(genemap_report, f, indent=2)
        logger.info(f"Saved genemap filter report to {genemap_report_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nSummary:")
    logger.info(f"  Cell type: {args.cell_type}")
    logger.info(f"  Binary classification: High (1) vs Not AD (0)")
    if args.foundation_model_mode:
        logger.info(f"\n  Foundation Model Mode: ENABLED")
        logger.info(f"    Vocabulary size: {vocab_expansion_report['vocab_size']:,}")
        logger.info(f"    Coverage: {vocab_expansion_report['coverage']:.1%}")
        logger.info(f"    Zero-padded genes: {vocab_expansion_report['zero_padded_genes']:,}")
        logger.info(f"    Dropped OOV genes: {vocab_expansion_report['dropped_genes']:,}")
    elif genemap_report is not None:
        logger.info(f"\n  Genemap vocabulary filtering:")
        logger.info(f"    Coverage: {genemap_report['coverage']:.1%}")
        logger.info(f"    Genes in vocabulary: {genemap_report['intersection']:,}")
        logger.info(f"    OOV genes dropped: {genemap_report['oov_genes_dropped']:,}")
    logger.info(f"\n  Train: {train_data.n_obs:,} cells x {train_data.n_vars:,} genes")
    logger.info(f"    Label distribution: {train_data.obs['label_name'].value_counts().to_dict()}")
    logger.info(f"\n  Val:   {val_data.n_obs:,} cells x {val_data.n_vars:,} genes")
    logger.info(f"    Label distribution: {val_data.obs['label_name'].value_counts().to_dict()}")
    logger.info(f"\n  Test:  {test_data.n_obs:,} cells x {test_data.n_vars:,} genes")
    logger.info(f"    Label distribution: {test_data.obs['label_name'].value_counts().to_dict()}")
    logger.info(f"\nProcessed data saved to: {output_dir}")
    logger.info(f"\nNext steps:")
    if args.foundation_model_mode:
        logger.info(f"  # Train CellFM model (foundation model finetuning)")
        logger.info(f"  python scripts/train_cellfm.py --data-dir {output_dir} --backbone-path /path/to/cellfm_weights.pt")
    else:
        logger.info(f"  # Train MLP model")
        logger.info(f"  python scripts/train_mlp.py --data-dir {output_dir}")
        logger.info(f"\n  # Train Transformer model")
        logger.info(f"  python scripts/train_transformer.py --data-dir {output_dir}")
        logger.info(f"\n  # With metadata features")
        logger.info(f"  python scripts/train_mlp.py --data-dir {output_dir} --use-metadata")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
