#!/usr/bin/env python3
"""
Multi-Stage Cascade Evaluator for Preprocessing Scripts.

Evaluates preprocessing code quality through three stages:
1. Stage 0: Execution check - Can the script run without errors?
2. Stage 1: Statistical filters - Output format, sparsity, value range
3. Stage 2: Frozen backbone - CellFM embedding + linear probe classification

This evaluator is used by Phase 1 (parameter evolution) and Phase 2 (code evolution).

Usage (called by OpenEvolve):
    from evaluator import evaluate
    result = evaluate("path/to/preprocessing_script.py")
"""

import importlib.util
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Paths (can be overridden via environment variables)
PROXY_DATA_PATH = os.environ.get(
    "PROXY_DATA_PATH",
    str(SCRIPT_DIR / "data" / "proxy_raw.h5ad")
)
GENEMAP_PATH = os.environ.get(
    "GENEMAP_PATH",
    str(PROJECT_ROOT / "data" / "genemap.csv")
)
BACKBONE_PATH = os.environ.get(
    "BACKBONE_PATH",
    str(PROJECT_ROOT / "examples" / "save" / "cellFM" / "CellFM_80M_weight.ckpt")
)

# Evaluation thresholds
STAGE1_THRESHOLD = 0.3  # Minimum stats score to proceed to Stage 2
EXPECTED_GENE_COUNT = 27934  # Genemap vocabulary size
EXPECTED_SPARSITY_MIN = 0.5  # Minimum sparsity (50% zeros)
EXPECTED_SPARSITY_MAX = 0.99  # Maximum sparsity (99% zeros)
EXPECTED_VALUE_MAX = 20.0  # Maximum expected value for log-normalized data


def evaluate_statistics(adata) -> Tuple[float, Dict[str, Any]]:
    """
    Stage 1: Evaluate preprocessing output statistics.

    Checks:
    - Gene count matches genemap (27,934)
    - Sparsity is preserved (50-99% zeros)
    - Value range is appropriate (0-20 for log-normalized)
    - No NaN or Inf values
    - Labels are preserved

    Args:
        adata: Preprocessed AnnData object

    Returns:
        Tuple of (score, metrics_dict)
    """
    from scipy import sparse

    metrics = {}
    score = 0.0
    max_score = 5.0  # Total possible points

    # 1. Gene count check (1 point)
    gene_count = adata.n_vars
    metrics["gene_count"] = gene_count
    if gene_count == EXPECTED_GENE_COUNT:
        score += 1.0
        metrics["gene_count_ok"] = True
    else:
        metrics["gene_count_ok"] = False
        logger.warning(f"Gene count mismatch: {gene_count} != {EXPECTED_GENE_COUNT}")

    # 2. Sparsity check (1 point)
    if sparse.issparse(adata.X):
        nnz = adata.X.nnz
        total = adata.X.shape[0] * adata.X.shape[1]
        sparsity = 1.0 - (nnz / total)
    else:
        sparsity = 1.0 - (np.count_nonzero(adata.X) / adata.X.size)

    metrics["sparsity"] = sparsity
    if EXPECTED_SPARSITY_MIN <= sparsity <= EXPECTED_SPARSITY_MAX:
        score += 1.0
        metrics["sparsity_ok"] = True
    else:
        metrics["sparsity_ok"] = False
        logger.warning(f"Sparsity out of range: {sparsity:.2%}")

    # 3. Value range check (1 point)
    if sparse.issparse(adata.X):
        data = adata.X.data
    else:
        data = adata.X.flatten()

    if len(data) > 0:
        value_min = float(np.min(data))
        value_max = float(np.max(data))
        value_mean = float(np.mean(data))
    else:
        value_min, value_max, value_mean = 0.0, 0.0, 0.0

    metrics["value_min"] = value_min
    metrics["value_max"] = value_max
    metrics["value_mean"] = value_mean

    if value_min >= 0 and value_max <= EXPECTED_VALUE_MAX:
        score += 1.0
        metrics["value_range_ok"] = True
    else:
        metrics["value_range_ok"] = False
        logger.warning(f"Value range issue: [{value_min:.2f}, {value_max:.2f}]")

    # 4. NaN/Inf check (1 point)
    if sparse.issparse(adata.X):
        has_nan = np.any(np.isnan(adata.X.data))
        has_inf = np.any(np.isinf(adata.X.data))
    else:
        has_nan = np.any(np.isnan(adata.X))
        has_inf = np.any(np.isinf(adata.X))

    metrics["has_nan"] = has_nan
    metrics["has_inf"] = has_inf
    if not has_nan and not has_inf:
        score += 1.0
        metrics["no_nan_inf"] = True
    else:
        metrics["no_nan_inf"] = False
        logger.warning(f"Found NaN={has_nan}, Inf={has_inf}")

    # 5. Labels check (1 point)
    has_labels = "label" in adata.obs.columns
    metrics["has_labels"] = has_labels
    if has_labels:
        unique_labels = adata.obs["label"].nunique()
        metrics["n_unique_labels"] = unique_labels
        if unique_labels >= 2:
            score += 1.0
            metrics["labels_ok"] = True
        else:
            metrics["labels_ok"] = False
            logger.warning(f"Only {unique_labels} unique labels found")
    else:
        metrics["labels_ok"] = False
        logger.warning("No 'label' column in obs")

    # Normalize score to 0-1
    normalized_score = score / max_score
    metrics["raw_score"] = score
    metrics["max_score"] = max_score

    return normalized_score, metrics


def evaluate_frozen_backbone(adata) -> Tuple[float, Dict[str, Any]]:
    """
    Stage 2: Evaluate using frozen CellFM backbone + linear probe.

    Pipeline:
    1. Load frozen CellFM backbone
    2. Extract embeddings from processed data
    3. Train logistic regression classifier
    4. Evaluate macro F1 score

    Args:
        adata: Preprocessed AnnData object

    Returns:
        Tuple of (f1_score, metrics_dict)
    """
    import torch
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.model_selection import train_test_split

    metrics = {}

    # Add project root to path for imports
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

    try:
        from cellfm_backbone import load_cellfm_backbone
    finally:
        if str(PROJECT_ROOT) in sys.path:
            sys.path.remove(str(PROJECT_ROOT))
        if str(PROJECT_ROOT / "scripts") in sys.path:
            sys.path.remove(str(PROJECT_ROOT / "scripts"))

    # Check backbone exists
    if not os.path.exists(BACKBONE_PATH):
        logger.error(f"Backbone not found: {BACKBONE_PATH}")
        return 0.0, {"error": f"Backbone not found: {BACKBONE_PATH}"}

    # Load frozen backbone
    logger.info("Loading frozen CellFM backbone...")
    try:
        backbone, hidden_dim = load_cellfm_backbone(BACKBONE_PATH)
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
        metrics["hidden_dim"] = hidden_dim
    except Exception as e:
        logger.error(f"Failed to load backbone: {e}")
        return 0.0, {"error": f"Failed to load backbone: {e}"}

    # Prepare data
    logger.info("Preparing data for embedding extraction...")
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    y = adata.obs["label"].values
    if hasattr(y, "codes"):  # Handle categorical
        y = y.codes
    y = np.array(y, dtype=np.int64)

    metrics["n_samples"] = len(y)
    metrics["n_features"] = X.shape[1]
    metrics["class_distribution"] = dict(zip(*np.unique(y, return_counts=True)))

    # Train/val split
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError as e:
        # Handle case where stratification fails due to small class counts
        logger.warning(f"Stratified split failed: {e}, using random split")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    metrics["n_train"] = len(y_train)
    metrics["n_val"] = len(y_val)

    # Extract embeddings in batches
    logger.info("Extracting embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)

    batch_size = 64

    def get_embeddings(X_data):
        embeddings = []
        for i in range(0, len(X_data), batch_size):
            batch = torch.tensor(X_data[i:i + batch_size], dtype=torch.float32).to(device)
            with torch.no_grad():
                emb = backbone(batch)
            embeddings.append(emb.cpu().numpy())
        return np.vstack(embeddings)

    train_emb = get_embeddings(X_train)
    val_emb = get_embeddings(X_val)

    metrics["embedding_dim"] = train_emb.shape[1]

    # Train linear probe
    logger.info("Training linear probe...")
    clf = LogisticRegression(
        max_iter=200,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(train_emb, y_train)

    # Evaluate
    train_preds = clf.predict(train_emb)
    val_preds = clf.predict(val_emb)

    train_f1 = f1_score(y_train, train_preds, average="macro")
    val_f1 = f1_score(y_val, val_preds, average="macro")
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    metrics["train_f1"] = train_f1
    metrics["val_f1"] = val_f1
    metrics["train_accuracy"] = train_acc
    metrics["val_accuracy"] = val_acc

    logger.info(f"Training F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

    return val_f1, metrics


def extract_features(code: str, adata) -> Tuple[float, float]:
    """
    Extract feature dimensions for MAP-Elites diversity.

    Features:
    - preprocessing_complexity: Based on code structure (0-1)
    - normalization_strength: Based on output statistics (0-1)

    Args:
        code: Python code string
        adata: Processed AnnData object

    Returns:
        Tuple of (complexity, normalization_strength)
    """
    from scipy import sparse

    # Complexity: based on code length and operations
    complexity = 0.0
    code_lower = code.lower()

    # Count preprocessing operations
    operations = [
        "filter_genes", "filter_cells", "normalize_total",
        "log1p", "highly_variable_genes", "scale", "pca",
        "batch_correct", "combat", "harmony", "scran",
        "pearson_residuals",
    ]
    op_count = sum(1 for op in operations if op in code_lower)
    complexity = min(1.0, op_count / 6.0)  # Normalize to 0-1

    # Normalization strength: based on data statistics
    if sparse.issparse(adata.X):
        data = adata.X.data
    else:
        data = adata.X.flatten()

    if len(data) > 0:
        # Higher mean and lower variance suggests stronger normalization
        mean_val = np.mean(data[data > 0]) if np.any(data > 0) else 0
        std_val = np.std(data[data > 0]) if np.any(data > 0) else 1

        # Normalize to 0-1 (assuming log-normalized data typically has mean 1-3, std 1-2)
        norm_strength = min(1.0, mean_val / 5.0)
    else:
        norm_strength = 0.0

    return complexity, norm_strength


def evaluate(script_path: str) -> Dict[str, Any]:
    """
    Main evaluation function called by OpenEvolve.

    Performs multi-stage cascade evaluation:
    - Stage 0: Execute the preprocessing script
    - Stage 1: Check output statistics (if Stage 0 passes)
    - Stage 2: Frozen backbone evaluation (if Stage 1 passes threshold)

    Args:
        script_path: Path to the preprocessing Python script

    Returns:
        Evaluation result dictionary with:
        - combined_score: Weighted combination of stats and model scores
        - stage: Highest stage reached (0, 1, or 2)
        - stats_score: Stage 1 score (if reached)
        - model_score: Stage 2 score (if reached)
        - metrics: Detailed metrics from each stage
        - error: Error message (if any)
    """
    import anndata as ad

    logger.info("=" * 60)
    logger.info(f"Evaluating: {script_path}")
    logger.info("=" * 60)

    # Load code for feature extraction
    try:
        with open(script_path, "r") as f:
            code = f.read()
    except Exception as e:
        return {
            "combined_score": 0.0,
            "stage": 0,
            "error": f"Failed to read script: {e}",
        }

    # Stage 0: Execute preprocessing
    logger.info("\n[Stage 0] Executing preprocessing script...")
    try:
        # Load module dynamically
        spec = importlib.util.spec_from_file_location("preprocess_module", script_path)
        module = importlib.util.module_from_spec(spec)

        # Add paths for imports
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        sys.path.insert(0, str(SCRIPT_DIR))

        try:
            spec.loader.exec_module(module)
        finally:
            for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"), str(SCRIPT_DIR)]:
                if p in sys.path:
                    sys.path.remove(p)

        if not hasattr(module, "preprocess"):
            return {
                "combined_score": 0.0,
                "stage": 0,
                "error": "Script missing 'preprocess' function",
            }

        # Check proxy data exists
        if not os.path.exists(PROXY_DATA_PATH):
            return {
                "combined_score": 0.0,
                "stage": 0,
                "error": f"Proxy data not found: {PROXY_DATA_PATH}",
            }

        # Load proxy data and run preprocessing
        raw_adata = ad.read_h5ad(PROXY_DATA_PATH)
        processed_adata = module.preprocess(raw_adata.copy(), GENEMAP_PATH)

        if not isinstance(processed_adata, ad.AnnData):
            return {
                "combined_score": 0.0,
                "stage": 0,
                "error": f"preprocess() returned {type(processed_adata)}, expected AnnData",
            }

        logger.info(f"  Output shape: {processed_adata.shape}")

    except Exception as e:
        logger.error(f"Stage 0 failed: {e}")
        return {
            "combined_score": 0.0,
            "stage": 0,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    # Stage 1: Statistical evaluation
    logger.info("\n[Stage 1] Evaluating statistics...")
    try:
        stats_score, stats_metrics = evaluate_statistics(processed_adata)
        logger.info(f"  Stats score: {stats_score:.4f}")
    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        return {
            "combined_score": 0.0,
            "stage": 0,
            "error": f"Statistics evaluation failed: {e}",
            "traceback": traceback.format_exc(),
        }

    # Check if we should proceed to Stage 2
    if stats_score < STAGE1_THRESHOLD:
        logger.info(f"  Stats score below threshold ({STAGE1_THRESHOLD}), skipping Stage 2")
        return {
            "combined_score": stats_score * 0.3,
            "stage": 1,
            "stats_score": stats_score,
            "model_score": 0.0,
            "metrics": {"stage1": stats_metrics},
        }

    # Stage 2: Frozen backbone evaluation
    logger.info("\n[Stage 2] Evaluating with frozen backbone...")
    try:
        model_score, model_metrics = evaluate_frozen_backbone(processed_adata)
        logger.info(f"  Model score (F1): {model_score:.4f}")
    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")
        return {
            "combined_score": stats_score * 0.3,
            "stage": 1,
            "stats_score": stats_score,
            "model_score": 0.0,
            "metrics": {"stage1": stats_metrics},
            "error": f"Backbone evaluation failed: {e}",
            "traceback": traceback.format_exc(),
        }

    # Calculate combined score
    combined_score = 0.3 * stats_score + 0.7 * model_score

    # Extract features for MAP-Elites
    try:
        complexity, norm_strength = extract_features(code, processed_adata)
    except Exception:
        complexity, norm_strength = 0.5, 0.5

    # Final result
    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL SCORE: {combined_score:.4f}")
    logger.info(f"  Stats: {stats_score:.4f}, Model: {model_score:.4f}")
    logger.info("=" * 60)

    return {
        "combined_score": combined_score,
        "stage": 2,
        "stats_score": stats_score,
        "model_score": model_score,
        "preprocessing_complexity": complexity,
        "normalization_strength": norm_strength,
        "metrics": {
            "stage1": stats_metrics,
            "stage2": model_metrics,
        },
    }


# For standalone testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <script_path>")
        print("\nExample:")
        print("  python evaluator.py candidates/baseline_params.py")
        sys.exit(1)

    script_path = sys.argv[1]
    result = evaluate(script_path)

    print("\n" + "=" * 60)
    print("EVALUATION RESULT")
    print("=" * 60)
    for key, value in result.items():
        if key == "metrics":
            print(f"  {key}:")
            for stage, stage_metrics in value.items():
                print(f"    {stage}:")
                for k, v in stage_metrics.items():
                    print(f"      {k}: {v}")
        elif key == "traceback":
            print(f"  {key}: (see below)")
        else:
            print(f"  {key}: {value}")

    if "traceback" in result:
        print("\nTraceback:")
        print(result["traceback"])
