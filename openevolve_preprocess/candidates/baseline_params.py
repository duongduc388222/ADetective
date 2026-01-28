#!/usr/bin/env python3
"""
Baseline Preprocessing Script - Phase 1: Parameter Evolution.

This script has a FIXED structure with EVOLVABLE parameters.
OpenEvolve will modify the PARAMS dictionary to optimize performance.

The preprocessing pipeline:
1. Gene filtering (min_cells threshold)
2. Library size normalization (target_sum)
3. Log transformation (optional)
4. HVG selection (intermediate step for quality)
5. Expand to genemap vocabulary (27,934 genes)
6. Optional scaling

Usage:
    from baseline_params import preprocess
    processed = preprocess(raw_adata, genemap_path)
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# ============================================================
# EVOLVABLE PARAMETERS - OpenEvolve will modify these values
# ============================================================
PARAMS = {
    "min_cells": 100,           # Range: 10-1000
    "target_sum": 1e4,          # Range: 1e3-1e6
    "n_hvgs": 2000,             # Range: 500-5000
    "normalize_method": "library_size",  # Options: library_size, scran, pearson_residuals
    "hvg_method": "seurat",     # Options: seurat, cell_ranger, seurat_v3
    "use_log1p": True,          # True/False
    "scale_data": False,        # True/False (usually False for foundation models)
}
# ============================================================


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

    Pipeline:
    1. Copy input data (avoid side effects)
    2. Preserve labels
    3. Filter low-quality genes
    4. Normalize to library size
    5. Log transform (optional)
    6. Select highly variable genes (intermediate step)
    7. Expand to genemap vocabulary
    8. Restore labels
    9. Validate output

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

    # Step 1: Filter genes (remove noisy genes)
    min_cells = PARAMS["min_cells"]
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Step 2: Normalization
    normalize_method = PARAMS["normalize_method"]
    target_sum = PARAMS["target_sum"]

    if normalize_method == "library_size":
        sc.pp.normalize_total(adata, target_sum=target_sum)
    elif normalize_method == "scran":
        # scran-style normalization (simplified)
        sc.pp.normalize_total(adata, target_sum=target_sum)
    elif normalize_method == "pearson_residuals":
        # Pearson residuals normalization
        sc.experimental.pp.normalize_pearson_residuals(adata)
    else:
        # Default to library size
        sc.pp.normalize_total(adata, target_sum=target_sum)

    # Step 3: Log transformation (optional)
    if PARAMS["use_log1p"] and normalize_method != "pearson_residuals":
        sc.pp.log1p(adata)

    # Step 4: HVG selection (for intermediate quality, then expand to full genemap)
    n_hvgs = PARAMS["n_hvgs"]
    hvg_method = PARAMS["hvg_method"]

    if adata.n_vars > n_hvgs:
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_hvgs,
                flavor=hvg_method
            )
            adata = adata[:, adata.var.highly_variable].copy()
        except Exception:
            # If HVG selection fails, keep all genes
            pass

    # Step 5: Expand to genemap vocabulary
    adata = expand_to_genemap(adata, genemap_path)

    # Step 6: Restore labels
    adata.obs["label"] = labels.values
    adata.obs["label_name"] = label_names.values

    # Step 7: Optional scaling (usually False for foundation models)
    if PARAMS["scale_data"]:
        # Scale to unit variance (be careful, this can affect sparsity)
        if sparse.issparse(adata.X):
            X_dense = adata.X.toarray()
            X_scaled = (X_dense - X_dense.mean(axis=0)) / (X_dense.std(axis=0) + 1e-8)
            adata.X = sparse.csr_matrix(X_scaled.astype(np.float32))
        else:
            adata.X = (adata.X - adata.X.mean(axis=0)) / (adata.X.std(axis=0) + 1e-8)

    # Step 8: Final validation - remove any NaN/Inf
    if sparse.issparse(adata.X):
        adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=10.0, neginf=0.0)
    else:
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=10.0, neginf=0.0)

    return adata
