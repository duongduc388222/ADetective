#!/usr/bin/env python3
"""
SEAAD Data Preparation Pipeline for Oligodendrocyte AD Classification.

Prepares SEAAD RNAseq dataset for 4-class ADNC classification:
  - Label 0 (Not AD), 1 (Low), 2 (Intermediate), 3 (High)

Pipeline: Load -> Filter cells -> Filter genes -> Donor-level split ->
          Normalize -> HVG selection or vocab expansion -> Save
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAPPING = {"Not AD": 0, "Low": 1, "Intermediate": 2, "High": 3}
LABEL_NAMES = {v: k for k, v in LABEL_MAPPING.items()}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare SEAAD dataset for Oligodendrocyte AD classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-path", type=str, required=True, help="Path to raw SEAAD H5AD file")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--n-hvgs", type=int, default=2000, help="Number of HVGs to select")
    parser.add_argument("--min-cells", type=int, default=100, help="Min cells for gene filtering")
    parser.add_argument("--target-sum", type=float, default=1e4, help="Normalization target sum")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--cell-type", type=str, default="Oligodendrocyte", help="Cell type to filter")
    parser.add_argument("--skip-normalization", action="store_true", help="Skip normalization")
    parser.add_argument("--genemap", type=str, default=None, help="Path to genemap.csv for vocab filtering")
    parser.add_argument("--genemap-min-coverage", type=float, default=0.3, help="Min vocab coverage (0.3=30%)")
    parser.add_argument("--foundation-model-mode", action="store_true", help="Expand to full genemap vocab")
    return parser.parse_args()


# =============================================================================
# Data Loading
# =============================================================================

def load_filtered_data(
    input_path: str,
    cell_type: str = "Oligodendrocyte",
    cell_type_column: str = "Subclass",
    adnc_column: str = "ADNC",
    exclude_categories: Optional[List[str]] = None,
):
    """Load SEAAD dataset with pre-filtering for memory efficiency."""
    import anndata as ad

    exclude_categories = exclude_categories or []
    logger.info(f"Loading SEAAD from {input_path}, filtering for {cell_type}")

    with h5py.File(input_path, 'r') as f:
        try:
            from anndata._io.specs import read_elem
        except ImportError:
            from anndata.experimental import read_elem

        obs = read_elem(f['obs'])
        var = read_elem(f['var'])
        logger.info(f"Total: {len(obs):,} cells, {len(var):,} genes")

        # Build filter mask
        cell_mask = obs[cell_type_column].str.lower() == cell_type.lower()
        adnc_mask = ~obs[adnc_column].isin(exclude_categories)
        keep_indices = np.where(cell_mask & adnc_mask)[0]
        logger.info(f"Cells to load: {len(keep_indices):,}")

        if len(keep_indices) == 0:
            raise ValueError("No cells match filter criteria!")

        # Load sparse matrix rows
        X_grp, indptr = f['X'], f['X']['indptr'][:]
        new_data, new_indices, new_indptr = [], [], [0]
        valid_indices, n_errors = [], 0

        for i, cell_idx in enumerate(keep_indices):
            row_start, row_end = indptr[cell_idx], indptr[cell_idx + 1]
            if row_end > row_start:
                try:
                    new_data.append(X_grp['data'][row_start:row_end])
                    new_indices.append(X_grp['indices'][row_start:row_end])
                    new_indptr.append(new_indptr[-1] + (row_end - row_start))
                    valid_indices.append(cell_idx)
                except OSError:
                    n_errors += 1
            else:
                new_indptr.append(new_indptr[-1])
                valid_indices.append(cell_idx)

            if (i + 1) % 10000 == 0:
                logger.info(f"  Progress: {(i+1)/len(keep_indices)*100:.1f}%")

        if n_errors:
            logger.warning(f"Skipped {n_errors} corrupted cells")

        # Build sparse matrix
        all_data = np.concatenate(new_data) if new_data else np.array([], dtype=np.float32)
        all_indices = np.concatenate(new_indices) if new_indices else np.array([], dtype=np.int64)
        X = sparse.csr_matrix((all_data, all_indices, np.array(new_indptr)),
                              shape=(len(valid_indices), var.shape[0]))

    # Create AnnData with labels
    obs_filtered = obs.iloc[valid_indices].copy().reset_index(drop=True)
    obs_filtered["label"] = obs_filtered[adnc_column].map(LABEL_MAPPING)
    obs_filtered["label_name"] = obs_filtered[adnc_column]

    adata = ad.AnnData(X=X, obs=obs_filtered, var=var)
    logger.info(f"Created AnnData: {adata.shape}")
    logger.info(f"Label distribution:\n{adata.obs['label_name'].value_counts().to_string()}")
    return adata, LABEL_MAPPING


# =============================================================================
# Splitting
# =============================================================================

def _stratified_split_with_fallback(data, test_size, random_state, strat_key, fallback_key):
    """Stratified split with fallback to simpler stratification."""
    try:
        return train_test_split(data, test_size=test_size, random_state=random_state, stratify=strat_key)
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}), using fallback")
        return train_test_split(data, test_size=test_size, random_state=random_state, stratify=fallback_key)


def stratified_donor_split(
    adata,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    donor_column: str = "Donor ID",
) -> Tuple:
    """Create donor-level stratified train/val/test split."""
    if "label" not in adata.obs or donor_column not in adata.obs:
        raise ValueError(f"Required columns missing. Available: {list(adata.obs.columns)}")

    logger.info(f"Donor-level split: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    # Get donor-level labels (majority vote)
    donor_labels = (
        adata.obs.groupby(donor_column)["label"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        .reset_index()
    )
    donor_labels.columns = [donor_column, "donor_label"]
    donor_labels = donor_labels.astype(str)
    logger.info(f"Total donors: {len(donor_labels)}")

    # Create stratification key with cell count buckets
    cell_counts = adata.obs.groupby(donor_column).size()
    donor_labels["cell_count"] = donor_labels[donor_column].map(cell_counts)

    try:
        donor_labels["count_bucket"] = pd.qcut(donor_labels["cell_count"], q=3,
                                                labels=["low", "med", "high"], duplicates="drop").astype(str)
        donor_labels["strat_key"] = donor_labels["donor_label"] + "_" + donor_labels["count_bucket"]
    except ValueError:
        donor_labels["strat_key"] = donor_labels["donor_label"]

    # Perform splits
    if test_ratio == 0:  # Cross-region mode
        val_size = val_ratio / (train_ratio + val_ratio)
        donors_train, donors_val = _stratified_split_with_fallback(
            donor_labels[donor_column], val_size, random_state,
            donor_labels["strat_key"], donor_labels["donor_label"]
        )
        donors_test = pd.Series([], dtype=str)
    else:
        donors_train_val, donors_test = _stratified_split_with_fallback(
            donor_labels[donor_column], test_ratio, random_state,
            donor_labels["strat_key"], donor_labels["donor_label"]
        )
        val_size = val_ratio / (train_ratio + val_ratio)
        train_val_labels = donor_labels[donor_labels[donor_column].isin(donors_train_val)]
        donors_train, donors_val = _stratified_split_with_fallback(
            train_val_labels[donor_column], val_size, random_state,
            train_val_labels["strat_key"], train_val_labels["donor_label"]
        )

    # Create split datasets
    train_adata = adata[adata.obs[donor_column].isin(donors_train)].copy()
    val_adata = adata[adata.obs[donor_column].isin(donors_val)].copy()
    test_adata = adata[adata.obs[donor_column].isin(donors_test)].copy() if len(donors_test) > 0 else adata[0:0].copy()

    for name, data in [("Train", train_adata), ("Val", val_adata), ("Test", test_adata)]:
        if data.n_obs > 0:
            logger.info(f"{name}: {data.n_obs:,} cells, {data.obs['label_name'].value_counts().to_dict()}")

    return train_adata, val_adata, test_adata


# =============================================================================
# Gene Filtering & Genemap Operations
# =============================================================================

def filter_genes(adata, min_cells: int = 100):
    """Filter genes expressed in fewer than min_cells."""
    if min_cells <= 0:
        return adata
    before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    logger.info(f"Gene filtering: {before:,} -> {adata.n_vars:,} ({before - adata.n_vars:,} removed)")
    return adata


def _load_genemap(genemap_path: str) -> Tuple[List[str], Set[str]]:
    """Load genemap vocabulary from CSV."""
    if not os.path.exists(genemap_path):
        raise FileNotFoundError(f"Genemap not found: {genemap_path}")
    df = pd.read_csv(genemap_path)
    genes = df[df.columns[0]].astype(str).str.strip().tolist()
    return genes, set(genes)


def _genemap_coverage(data_genes: Set[str], genemap_set: Set[str]) -> Tuple[Set[str], float]:
    """Compute intersection and coverage between data genes and genemap."""
    intersection = data_genes & genemap_set
    coverage = len(intersection) / len(genemap_set) if genemap_set else 0
    return intersection, coverage


def filter_to_genemap(adata, genemap_path: str, min_coverage: float = 0.3) -> Tuple:
    """Filter genes to only those in genemap vocabulary."""
    logger.info(f"Filtering to genemap: {genemap_path}")
    genemap_genes, genemap_set = _load_genemap(genemap_path)
    data_genes = set(adata.var_names.astype(str))
    intersection, coverage = _genemap_coverage(data_genes, genemap_set)

    logger.info(f"  Vocab: {len(genemap_set):,}, Data: {len(data_genes):,}, "
                f"Intersection: {len(intersection):,}, Coverage: {coverage:.1%}")

    if coverage < min_coverage:
        raise ValueError(f"Coverage {coverage:.1%} < {min_coverage:.0%} threshold")

    adata_filtered = adata[:, [g for g in adata.var_names if g in intersection]].copy()
    report = {
        "genemap_path": genemap_path, "vocab_size": len(genemap_set),
        "data_genes": len(data_genes), "intersection": len(intersection),
        "coverage": coverage, "oov_genes_dropped": len(data_genes - genemap_set),
        "genes_retained": adata_filtered.n_vars,
    }
    return adata_filtered, report


def expand_to_genemap_vocab(adata, genemap_path: str) -> Tuple:
    """Expand AnnData to match genemap vocabulary exactly (zero-padding missing genes)."""
    import anndata as ad

    logger.info(f"Expanding to genemap vocab: {genemap_path}")
    genemap_genes, genemap_set = _load_genemap(genemap_path)
    data_genes = set(adata.var_names.astype(str))
    intersection, coverage = _genemap_coverage(data_genes, genemap_set)

    logger.info(f"  Vocab: {len(genemap_set):,}, Data: {len(data_genes):,}, "
                f"Intersection: {len(intersection):,}, Coverage: {coverage:.1%}")

    # Build expanded sparse matrix
    genemap_idx = {g: i for i, g in enumerate(genemap_genes)}
    data_idx = {g: i for i, g in enumerate(adata.var_names)}

    X_src = adata.X.tocsc() if sparse.issparse(adata.X) else sparse.csc_matrix(adata.X)
    X_new = sparse.lil_matrix((adata.n_obs, len(genemap_genes)), dtype=np.float32)

    for gene in intersection:
        X_new[:, genemap_idx[gene]] = X_src.getcol(data_idx[gene]).toarray().flatten().reshape(-1, 1)

    adata_expanded = ad.AnnData(X=X_new.tocsr(), obs=adata.obs.copy(),
                                 var=pd.DataFrame(index=genemap_genes))

    report = {
        "genemap_path": genemap_path, "vocab_size": len(genemap_genes),
        "data_genes": len(data_genes), "intersection": len(intersection),
        "coverage": coverage, "zero_padded_genes": len(genemap_genes) - len(intersection),
        "dropped_genes": len(data_genes) - len(intersection),
    }
    return adata_expanded, report


# =============================================================================
# Normalization & HVG Selection
# =============================================================================

def normalize_data(adata, target_sum: float = 1e4):
    """Library-size normalize and log-transform."""
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.uns.setdefault("preprocessing_history", {}).update({
        "normalized": True, "log_transformed": True, "normalization_target": target_sum
    })
    return adata


def select_hvgs(adata, n_hvgs: int = 2000):
    """Select highly variable genes."""
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs)
    return adata[:, adata.var.highly_variable].copy()


# =============================================================================
# Split Processing Helpers
# =============================================================================

def _apply_to_splits(train, val, test, func: Callable, desc: str, **kwargs):
    """Apply a function to train/val/test splits, handling empty test."""
    logger.info(f"{desc} (train)...")
    train = func(train, **kwargs)
    logger.info(f"{desc} (val)...")
    val = func(val, **kwargs)
    if test.n_obs > 0:
        logger.info(f"{desc} (test)...")
        test = func(test, **kwargs)
    return train, val, test


def normalize_splits(train, val, test, target_sum: float = 1e4):
    """Normalize each split independently."""
    return _apply_to_splits(train, val, test, normalize_data, "Normalizing", target_sum=target_sum)


def expand_splits_to_vocab(train, val, test, genemap_path: str):
    """Expand all splits to genemap vocabulary."""
    train, report = expand_to_genemap_vocab(train, genemap_path)
    val, _ = expand_to_genemap_vocab(val, genemap_path)
    if test.n_obs > 0:
        test, _ = expand_to_genemap_vocab(test, genemap_path)
    logger.info(f"Vocab expansion: coverage={report['coverage']:.1%}, "
                f"zero-padded={report['zero_padded_genes']:,}")
    return train, val, test, report


def subset_splits_to_hvgs(train, val, test, n_hvgs: int = 2000):
    """Select HVGs from training set and subset all splits."""
    train = select_hvgs(train, n_hvgs=n_hvgs)
    hvg_genes = train.var_names.copy()
    logger.info(f"Selected {len(hvg_genes)} HVGs from training set")
    val = val[:, hvg_genes].copy()
    test = test[:, hvg_genes].copy() if test.n_obs > 0 else test
    return train, val, test


# =============================================================================
# Programmatic API for Evaluator
# =============================================================================


def preprocess(raw_adata: "AnnData", genemap_path: str, **kwargs) -> "AnnData":
    """
    Programmatic API for preprocessing - called by evaluator.py.

    This function matches the evaluator's expected contract:
        preprocess(raw_adata: AnnData, genemap_path: str) -> AnnData

    Args:
        raw_adata: Raw AnnData with counts (already loaded by evaluator)
            - X: Sparse matrix of raw counts
            - obs['label']: Integer class labels (0-3 for ADNC)
        genemap_path: Path to genemap.csv (27,934 genes)
        **kwargs: Override preprocessing options
            - min_cells: int (default: from env or 100)
            - target_sum: float (default: from env or 1e4)
            - skip_normalization: bool (default: from env or False)
            - foundation_model_mode: bool (default: from env or True)

    Returns:
        Preprocessed AnnData with shape (n_cells, 27934) for foundation model mode
    """
    # Load defaults from environment
    min_cells = kwargs.get("min_cells", int(os.environ.get("MIN_CELLS", "100")))
    target_sum = kwargs.get("target_sum", float(os.environ.get("TARGET_SUM", "1e4")))
    skip_norm = kwargs.get("skip_normalization",
                           os.environ.get("SKIP_NORMALIZATION", "false").lower() == "true")
    fm_mode = kwargs.get("foundation_model_mode",
                         os.environ.get("FOUNDATION_MODEL_MODE", "true").lower() == "true")

    logger.info(f"preprocess() called: min_cells={min_cells}, target_sum={target_sum}, "
                f"skip_norm={skip_norm}, foundation_model_mode={fm_mode}")

    # Work on a copy
    adata = raw_adata.copy()

    # EVOLVE-BLOCK-START
    # This block is evolved by OpenEvolve for optimal preprocessing
    # Focus: gene filtering, normalization, feature selection
    # Goal: Maximize downstream classification performance (QWK score)

    # Step 1: Filter genes by min cells
    if min_cells > 0:
        adata = filter_genes(adata, min_cells=min_cells)

    # Step 2: Normalize (unless skipped)
    if not skip_norm:
        adata = normalize_data(adata, target_sum=target_sum)

    # Step 3: Expand to genemap vocabulary (foundation model mode)
    if fm_mode:
        adata, report = expand_to_genemap_vocab(adata, genemap_path)
        logger.info(f"Vocab expansion: {report['intersection']} genes matched, "
                    f"{report['zero_padded_genes']} zero-padded")

    # EVOLVE-BLOCK-END

    return adata


# =============================================================================
# Save & Report
# =============================================================================

def save_processed(adata, output_path: str):
    """Save processed AnnData."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    adata.write_h5ad(output_path)
    logger.info(f"Saved: {output_path}")


def _build_report(args, train, val, test, vocab_report=None, genemap_report=None):
    """Build preparation report dictionary."""
    report = {
        "input_path": args.input_path,
        "output_dir": str(args.output_dir),
        "cell_type": args.cell_type,
        "label_mapping": LABEL_MAPPING,
        "num_classes": 4,
        "foundation_model_mode": args.foundation_model_mode,
        "cross_region_mode": args.test_ratio == 0,
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "random_state": args.random_state,
    }

    for name, data in [("train", train), ("val", val)]:
        report[name] = {
            "n_cells": data.n_obs, "n_genes": data.n_vars,
            "label_distribution": data.obs["label_name"].value_counts().to_dict(),
        }

    report["test"] = (
        {"n_cells": test.n_obs, "n_genes": test.n_vars,
         "label_distribution": test.obs["label_name"].value_counts().to_dict()}
        if test.n_obs > 0 else {"note": "Cross-region mode"}
    )

    if vocab_report:
        report["vocab_expansion"] = vocab_report
    else:
        report["n_hvgs"] = args.n_hvgs
    if genemap_report:
        report["genemap"] = genemap_report

    return report


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()

    # Validate
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) >= 0.01:
        logger.error("Split ratios must sum to 1.0")
        return False
    if args.foundation_model_mode and not args.genemap:
        logger.error("--genemap required with --foundation-model-mode")
        return False
    if not os.path.exists(args.input_path):
        logger.error(f"Input not found: {args.input_path}")
        return False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SEAAD DATA PREPARATION")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Mode: {'Foundation Model' if args.foundation_model_mode else 'HVG Selection'}")

    # Step 1-3: Load and filter
    try:
        adata, _ = load_filtered_data(args.input_path, cell_type=args.cell_type)
    except Exception as e:
        logger.error(f"Load failed: {e}")
        return False

    # Step 4: Filter genes
    adata = filter_genes(adata, min_cells=args.min_cells)

    # Step 4.5: Genemap filtering (HVG mode only)
    genemap_report = None
    if args.genemap and not args.foundation_model_mode:
        try:
            adata, genemap_report = filter_to_genemap(adata, args.genemap, args.genemap_min_coverage)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Genemap filter failed: {e}")
            return False

    # Step 5: Donor-level split
    try:
        train, val, test = stratified_donor_split(
            adata, args.train_ratio, args.val_ratio, args.test_ratio, args.random_state
        )
    except Exception as e:
        logger.error(f"Split failed: {e}")
        return False

    # Step 6: Normalize
    if not args.skip_normalization:
        train, val, test = normalize_splits(train, val, test, args.target_sum)

    # Step 7: HVG or vocab expansion
    vocab_report = None
    if args.foundation_model_mode:
        try:
            train, val, test, vocab_report = expand_splits_to_vocab(train, val, test, args.genemap)
        except Exception as e:
            logger.error(f"Vocab expansion failed: {e}")
            return False
    else:
        train, val, test = subset_splits_to_hvgs(train, val, test, args.n_hvgs)

    # Step 8: Save
    save_processed(train, str(output_dir / "train.h5ad"))
    save_processed(val, str(output_dir / "val.h5ad"))
    if test.n_obs > 0:
        save_processed(test, str(output_dir / "test.h5ad"))

    # Save report
    report = _build_report(args, train, val, test, vocab_report, genemap_report)
    with open(output_dir / "preparation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Summary
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Train: {train.n_obs:,} x {train.n_vars:,}")
    logger.info(f"Val: {val.n_obs:,} x {val.n_vars:,}")
    logger.info(f"Test: {test.n_obs:,} x {test.n_vars:,}" if test.n_obs > 0 else "Test: cross-region mode")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
