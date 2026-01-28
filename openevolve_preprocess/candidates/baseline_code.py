#!/usr/bin/env python3
"""
Baseline Preprocessing Script - Phase 2: Code Evolution.

This script is FULLY EVOLVABLE. OpenEvolve can modify any part
of the preprocessing logic to discover novel approaches.

Starting point: Standard scanpy preprocessing pipeline
Goal: Maximize AD classification performance on CellFM

The LLM can:
- Change normalization methods
- Add batch correction
- Add denoising
- Modify gene filtering
- Add feature engineering
- Create entirely new preprocessing strategies

Usage:
    from baseline_code import preprocess
    processed = preprocess(raw_adata, genemap_path)
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def expand_to_genemap(adata, genemap_path: str):
    """
    Expand AnnData to match genemap vocabulary (27,934 genes).

    Genes in both data and genemap: Keep values
    Genes only in genemap: Zero-pad
    Genes only in data: Drop

    Args:
        adata: Preprocessed AnnData object
        genemap_path: Path to genemap.csv file

    Returns:
        AnnData with shape (n_cells, 27934)
    """
    import anndata as ad

    # Load genemap vocabulary
    genemap_df = pd.read_csv(genemap_path)
    genemap_genes = genemap_df.iloc[:, 0].astype(str).str.strip().tolist()
    vocab_size = len(genemap_genes)

    # Build index mappings
    genemap_idx = {g: i for i, g in enumerate(genemap_genes)}
    data_gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    intersection = set(adata.var_names) & set(genemap_genes)

    # Create expanded sparse matrix
    n_cells = adata.n_obs
    X_new = sparse.lil_matrix((n_cells, vocab_size), dtype=np.float32)

    # Copy values for genes in intersection
    X_src = adata.X.tocsc() if sparse.issparse(adata.X) else sparse.csc_matrix(adata.X)
    for gene in intersection:
        src_idx = data_gene_to_idx[gene]
        dst_idx = genemap_idx[gene]
        X_new[:, dst_idx] = X_src.getcol(src_idx).toarray().flatten().reshape(-1, 1)

    # Create output AnnData
    var_new = pd.DataFrame(index=genemap_genes)
    return ad.AnnData(X=X_new.tocsr(), obs=adata.obs.copy(), var=var_new)


def preprocess(raw_adata, genemap_path: str):
    """
    Preprocess scRNA-seq data for CellFM classification.

    This function is FULLY EVOLVABLE. The entire implementation
    can be modified by OpenEvolve to discover optimal preprocessing.

    Current implementation:
    1. Gene filtering (min_cells=100)
    2. Library size normalization (target_sum=10000)
    3. Log transformation
    4. HVG selection (n_top_genes=2000)
    5. Expand to genemap vocabulary (27,934 genes)

    Args:
        raw_adata: Raw AnnData with counts
            - X: Sparse matrix of raw counts
            - obs['label']: Integer class labels (0-3)
        genemap_path: Path to genemap.csv (27,934 genes)

    Returns:
        AnnData with shape (n_cells, 27934), labels preserved

    Raises:
        ValueError: If input missing required 'label' column
    """
    # Validate input
    if "label" not in raw_adata.obs.columns:
        raise ValueError("Input must have 'label' column in obs")

    # Work on copy to avoid modifying input
    adata = raw_adata.copy()

    # Preserve labels before processing
    labels = adata.obs["label"].copy()
    label_names = adata.obs.get("label_name", labels.astype(str)).copy()

    # ================================================================
    # PREPROCESSING PIPELINE - This entire section is evolvable
    # ================================================================

    # Step 1: Gene filtering
    # Remove genes expressed in too few cells (noisy genes)
    sc.pp.filter_genes(adata, min_cells=100)

    # Step 2: Library size normalization
    # Normalize each cell to have the same total counts
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Step 3: Log transformation
    # Convert to log scale for better distribution
    sc.pp.log1p(adata)

    # Step 4: HVG selection
    # Select highly variable genes to focus on informative features
    if adata.n_vars > 2000:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
        adata = adata[:, adata.var.highly_variable].copy()

    # ================================================================
    # END EVOLVABLE SECTION
    # ================================================================

    # Expand to genemap vocabulary (REQUIRED - do not remove)
    adata = expand_to_genemap(adata, genemap_path)

    # Restore labels (REQUIRED - do not remove)
    adata.obs["label"] = labels.values
    adata.obs["label_name"] = label_names.values

    # Final validation - remove any NaN/Inf (REQUIRED - do not remove)
    if sparse.issparse(adata.X):
        adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=10.0, neginf=0.0)
    else:
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=10.0, neginf=0.0)

    return adata
