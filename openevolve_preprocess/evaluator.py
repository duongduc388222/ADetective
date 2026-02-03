#!/usr/bin/env python3
"""
================================================================================
MULTI-STAGE CASCADE EVALUATOR FOR PREPROCESSING CODE EVOLUTION
================================================================================

OVERVIEW
--------
This evaluator scores preprocessing scripts for single-cell RNA-seq data
(specifically SEA-AD Alzheimer's Disease classification). It uses a cheap but
predictive proxy suite to avoid expensive full model training on every iteration.

EVALUATION PHILOSOPHY
---------------------
You can't afford to fully fine-tune an 80M foundation model every iteration.
So we evaluate preprocessing with a *cheap but predictive proxy suite*:

A) HARD-GATE CORRECTNESS (must pass or score collapses to 0)
   1. Execution: Script runs without errors
   2. Schema: Output is valid AnnData with required fields
   3. Gene count: Exactly matches genemap vocabulary (27,934 genes)
   4. Data integrity: No NaN/Inf values, reasonable sparsity
   5. Label preservation: Multi-class labels present in output
   6. Donor integrity: If donor info available, no train/test leakage

B) DATA-QUALITY HEURISTICS (fast sanity checks, 0-1 score each)
   7. Sparsity sanity: scRNA data should be 50-99% zeros
   8. Value range sanity: Log-normalized data typically in [0, 20]
   9. Distribution sanity: No suspiciously uniform or degenerate distributions

C) TASK SIGNAL PROXY (cheap "does preprocessing preserve signal?")
   10. Frozen backbone + linear probe: CellFM embedding + logistic regression
       - This is the main scalar fitness signal (fast + correlated with
         downstream fine-tuning success)

STAGED EVALUATION (CASCADE)
---------------------------
Stage 0: EXECUTION
    - Can the script run without errors?
    - Returns AnnData object?
    - Has 'preprocess' function?
    Score: Binary pass/fail

Stage 1: STATISTICAL VALIDATION
    - Gene count matches genemap (27,934)?
    - Sparsity in expected range (50-99%)?
    - Value range appropriate (0-20 for log-normalized)?
    - No NaN/Inf values?
    - Labels preserved with >=2 classes?
    Score: [0, 1] weighted average of checks

Stage 2: FROZEN BACKBONE PROXY (expensive, only if Stage 1 passes threshold)
    - Load frozen CellFM backbone
    - Extract cell embeddings
    - Train logistic regression classifier
    - Evaluate macro F1 / QWK on validation split
    Score: [0, 1] based on classification performance

FINAL SCORE FORMULA
-------------------
    If Stage 0 fails: combined_score = 0
    If Stage 1 < threshold: combined_score = 0.3 * stage1_score
    Otherwise: combined_score = 0.3 * stage1_score + 0.7 * stage2_score

CONTRACT WITH GENERATED CODE
----------------------------
The generated preprocessing script MUST expose:

    def preprocess(raw_adata: AnnData, genemap_path: str) -> AnnData:
        '''
        Args:
            raw_adata: Raw AnnData with counts
                - X: Sparse matrix of raw counts
                - obs['label']: Integer class labels (0-3 for ADNC)
                - obs['donor_id']: (optional) Donor identifiers for split integrity
            genemap_path: Path to genemap.csv (27,934 genes)

        Returns:
            AnnData with:
                - X: Preprocessed expression matrix, shape (n_cells, 27934)
                - obs['label']: Preserved integer labels
                - obs['donor_id']: (optional) Preserved donor IDs
        '''

ENVIRONMENT VARIABLES
---------------------
    PROXY_DATA_PATH: Path to proxy raw data (.h5ad)
    GENEMAP_PATH: Path to genemap.csv
    BACKBONE_PATH: Path to CellFM checkpoint (.ckpt)
    STAGE1_THRESHOLD: Min Stage 1 score to proceed (default: 0.3)
    EVAL_TIMEOUT: Max evaluation time in seconds (default: 120)

USAGE
-----
    # Called by OpenEvolve
    from evaluator import evaluate
    result = evaluate("path/to/preprocessing_script.py")

    # Standalone testing
    python evaluator.py path/to/preprocessing_script.py
"""

import concurrent.futures
import importlib.util
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

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
STAGE1_THRESHOLD = float(os.environ.get("STAGE1_THRESHOLD", "0.3"))
EVAL_TIMEOUT = int(os.environ.get("EVAL_TIMEOUT", "120"))

# Expected data characteristics
EXPECTED_GENE_COUNT = 27934      # Genemap vocabulary size
EXPECTED_SPARSITY_MIN = 0.50    # Min sparsity (50% zeros)
EXPECTED_SPARSITY_MAX = 0.99    # Max sparsity (99% zeros)
EXPECTED_VALUE_MAX = 20.0       # Max for log-normalized data
MIN_CLASSES = 2                 # Minimum unique labels required

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def run_with_timeout(func, args=(), kwargs=None, timeout_seconds: int = 60):
    """Execute function with timeout, raising TimeoutError if exceeded."""
    kwargs = kwargs or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout_seconds)


def safe_float(x, default: float = 0.0) -> float:
    """Convert to float safely, returning default on failure."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def is_sparse(x: Any) -> bool:
    """Check if x is a scipy sparse matrix."""
    try:
        from scipy import sparse
        return sparse.issparse(x)
    except ImportError:
        return False


def get_matrix_data(X) -> np.ndarray:
    """Extract data array from dense or sparse matrix."""
    if is_sparse(X):
        return X.data if X.nnz > 0 else np.array([])
    return np.asarray(X).flatten()


def compute_sparsity(X) -> float:
    """Compute fraction of zero elements in matrix."""
    if is_sparse(X):
        total = X.shape[0] * X.shape[1]
        return 1.0 - (X.nnz / max(1, total))
    X_arr = np.asarray(X)
    return 1.0 - (np.count_nonzero(X_arr) / max(1, X_arr.size))


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: Optional[int] = None
) -> float:
    """
    Compute Quadratic Weighted Kappa for ordinal multi-class labels.

    QWK penalizes predictions proportionally to the squared distance
    from the true label. Useful for ordinal classification (ADNC stages).

    Args:
        y_true: Ground truth labels (integers 0..K-1)
        y_pred: Predicted labels (integers 0..K-1)
        n_classes: Number of classes (auto-detected if None)

    Returns:
        QWK score in [-1, 1], where 1 is perfect agreement
    """
    y_true = np.asarray(y_true, dtype=int).flatten()
    y_pred = np.asarray(y_pred, dtype=int).flatten()

    if n_classes is None:
        n_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)

    # Build confusion matrix O
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            O[t, p] += 1.0

    # Expected matrix E from marginal histograms
    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    N = O.sum()
    if N == 0:
        return 0.0
    E = np.outer(hist_true, hist_pred) / N

    # Quadratic weight matrix
    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    denom = (n_classes - 1) ** 2 if n_classes > 1 else 1.0
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / denom

    num = (W * O).sum()
    den = (W * E).sum()
    if den < 1e-12:
        return 0.0

    return float(1.0 - (num / den))


# =============================================================================
# STAGE 1: STATISTICAL VALIDATION
# =============================================================================


def check_gene_count(adata) -> Tuple[float, str]:
    """Check if gene count matches expected genemap vocabulary."""
    gene_count = adata.n_vars
    if gene_count == EXPECTED_GENE_COUNT:
        return 1.0, f"Gene count OK ({gene_count})"
    return 0.0, f"Gene count mismatch: {gene_count} != {EXPECTED_GENE_COUNT}"


def check_sparsity(adata) -> Tuple[float, str]:
    """Check if sparsity is within expected range for scRNA-seq."""
    sparsity = compute_sparsity(adata.X)
    if EXPECTED_SPARSITY_MIN <= sparsity <= EXPECTED_SPARSITY_MAX:
        return 1.0, f"Sparsity OK ({sparsity:.2%})"
    if sparsity < EXPECTED_SPARSITY_MIN:
        # Too dense - suspicious
        return 0.3, f"Suspiciously dense ({sparsity:.2%} < {EXPECTED_SPARSITY_MIN:.0%})"
    if sparsity > EXPECTED_SPARSITY_MAX:
        # Too sparse - possible data loss
        return 0.6, f"Extremely sparse ({sparsity:.2%} > {EXPECTED_SPARSITY_MAX:.0%})"
    return 0.5, f"Sparsity marginal ({sparsity:.2%})"


def check_value_range(adata) -> Tuple[float, str]:
    """Check if values are within expected range for log-normalized data."""
    data = get_matrix_data(adata.X)
    if len(data) == 0:
        return 0.0, "Empty matrix (no nonzero values)"

    val_min = float(np.min(data))
    val_max = float(np.max(data))
    val_mean = float(np.mean(data))

    issues = []
    if val_min < 0:
        issues.append(f"negative values (min={val_min:.2f})")
    if val_max > EXPECTED_VALUE_MAX:
        issues.append(f"too large (max={val_max:.2f} > {EXPECTED_VALUE_MAX})")

    if issues:
        return 0.5, f"Value range issues: {', '.join(issues)}"
    return 1.0, f"Value range OK ([{val_min:.2f}, {val_max:.2f}], mean={val_mean:.2f})"


def check_finite(adata) -> Tuple[float, str]:
    """Check for NaN/Inf values in expression matrix."""
    data = get_matrix_data(adata.X)
    if len(data) == 0:
        return 0.5, "No nonzero data to check"

    # Sample if matrix is large
    if len(data) > 100_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data), size=100_000, replace=False)
        sample = data[idx]
    else:
        sample = data

    has_nan = np.any(np.isnan(sample))
    has_inf = np.any(np.isinf(sample))

    if has_nan or has_inf:
        return 0.0, f"Found NaN={has_nan}, Inf={has_inf}"
    return 1.0, "All values finite"


def check_labels(adata) -> Tuple[float, str]:
    """Check that labels exist and have sufficient class coverage."""
    if "label" not in adata.obs.columns:
        return 0.0, "Missing 'label' column in obs"

    labels = adata.obs["label"]
    unique_labels = labels.nunique()

    if unique_labels < MIN_CLASSES:
        return 0.2, f"Insufficient classes: {unique_labels} < {MIN_CLASSES}"

    # Check for reasonable class distribution
    counts = labels.value_counts()
    min_count = counts.min()
    if min_count < 5:
        return 0.7, f"Imbalanced classes (smallest class has {min_count} samples)"

    return 1.0, f"Labels OK ({unique_labels} classes)"


def check_donor_leakage(adata) -> Tuple[float, str]:
    """
    Check for donor-level data leakage (if donor info available).

    Note: This check only applies if the data has train/test annotations
    AND donor information. For preprocessing scripts that don't split data,
    this returns a neutral score.
    """
    if "donor_id" not in adata.obs.columns:
        return 0.8, "No donor_id column (cannot verify donor integrity)"

    if "split" not in adata.obs.columns:
        return 0.8, "No split column (cannot check train/test leakage)"

    # Check for donor overlap between splits
    splits = adata.obs["split"].unique()
    if len(splits) < 2:
        return 0.8, f"Only one split present ({splits})"

    donors_by_split = {}
    for split in splits:
        mask = adata.obs["split"] == split
        donors_by_split[split] = set(adata.obs.loc[mask, "donor_id"].unique())

    # Check all pairs for overlap
    split_list = list(splits)
    for i, s1 in enumerate(split_list):
        for s2 in split_list[i + 1:]:
            overlap = donors_by_split[s1] & donors_by_split[s2]
            if overlap:
                return 0.0, f"LEAKAGE: {len(overlap)} donors in both {s1} and {s2}"

    return 1.0, "No donor leakage detected"


def evaluate_statistics(adata) -> Tuple[float, Dict[str, Any]]:
    """
    Stage 1: Evaluate preprocessing output statistics.

    Runs all statistical checks and combines into a weighted score.

    Args:
        adata: Preprocessed AnnData object

    Returns:
        Tuple of (normalized_score, metrics_dict)
    """
    metrics = {}
    checks = [
        ("gene_count", check_gene_count, 1.5),    # Higher weight - critical
        ("sparsity", check_sparsity, 1.0),
        ("value_range", check_value_range, 1.0),
        ("finite", check_finite, 1.5),             # Higher weight - critical
        ("labels", check_labels, 1.0),
        ("donor_leakage", check_donor_leakage, 1.0),
    ]

    total_score = 0.0
    total_weight = 0.0

    for name, check_fn, weight in checks:
        try:
            score, msg = check_fn(adata)
        except Exception as e:
            score, msg = 0.0, f"Check failed: {e}"

        metrics[f"{name}_score"] = score
        metrics[f"{name}_msg"] = msg
        total_score += score * weight
        total_weight += weight
        logger.info(f"  {name}: {score:.2f} - {msg}")

    # Normalize to [0, 1]
    normalized_score = total_score / total_weight if total_weight > 0 else 0.0
    metrics["raw_score"] = total_score
    metrics["max_score"] = total_weight
    metrics["n_cells"] = adata.n_obs
    metrics["n_genes"] = adata.n_vars

    return normalized_score, metrics


# =============================================================================
# STAGE 2: FROZEN BACKBONE EVALUATION
# =============================================================================


def evaluate_frozen_backbone(adata) -> Tuple[float, Dict[str, Any]]:
    """
    Stage 2: Evaluate using frozen CellFM backbone + linear probe.

    Pipeline:
    1. Load frozen CellFM backbone (no gradient updates)
    2. Extract cell embeddings in batches
    3. Train logistic regression classifier
    4. Evaluate macro F1 and QWK on held-out split

    Args:
        adata: Preprocessed AnnData object with labels

    Returns:
        Tuple of (f1_score, metrics_dict)
    """
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    metrics = {}

    # Check backbone exists
    if not os.path.exists(BACKBONE_PATH):
        logger.error(f"Backbone not found: {BACKBONE_PATH}")
        return 0.0, {"error": f"Backbone not found: {BACKBONE_PATH}"}

    # Load backbone
    logger.info("Loading frozen CellFM backbone...")
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from cellfm_backbone import load_cellfm_backbone
        backbone, hidden_dim = load_cellfm_backbone(BACKBONE_PATH)
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
        metrics["hidden_dim"] = hidden_dim
    except Exception as e:
        logger.error(f"Failed to load backbone: {e}")
        return 0.0, {"error": f"Failed to load backbone: {e}"}
    finally:
        for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")]:
            if p in sys.path:
                sys.path.remove(p)

    # Prepare data
    logger.info("Preparing data for embedding extraction...")
    X = adata.X
    if is_sparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # Get labels
    y = adata.obs["label"].values
    if hasattr(y, "codes"):  # Handle categorical
        y = y.codes
    y = np.asarray(y, dtype=np.int64)

    metrics["n_samples"] = len(y)
    metrics["n_features"] = X.shape[1]
    unique, counts = np.unique(y, return_counts=True)
    metrics["class_distribution"] = dict(zip(unique.tolist(), counts.tolist()))

    # Train/validation split (stratified if possible)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}, using random split")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    metrics["n_train"] = len(y_train)
    metrics["n_val"] = len(y_val)

    # Extract embeddings
    logger.info("Extracting embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    batch_size = 64

    def get_embeddings(X_data: np.ndarray) -> np.ndarray:
        embeddings = []
        for i in range(0, len(X_data), batch_size):
            batch = torch.tensor(
                X_data[i:i + batch_size],
                dtype=torch.float32,
                device=device
            )
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
        n_jobs=1,
    )
    clf.fit(train_emb, y_train)

    # Evaluate
    train_preds = clf.predict(train_emb)
    val_preds = clf.predict(val_emb)

    train_f1 = f1_score(y_train, train_preds, average="macro")
    val_f1 = f1_score(y_val, val_preds, average="macro")
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    # Compute QWK for ordinal labels
    n_classes = int(max(y.max(), val_preds.max()) + 1)
    train_qwk = quadratic_weighted_kappa(y_train, train_preds, n_classes)
    val_qwk = quadratic_weighted_kappa(y_val, val_preds, n_classes)

    metrics["train_f1"] = train_f1
    metrics["val_f1"] = val_f1
    metrics["train_accuracy"] = train_acc
    metrics["val_accuracy"] = val_acc
    metrics["train_qwk"] = train_qwk
    metrics["val_qwk"] = val_qwk

    logger.info(f"Training: F1={train_f1:.4f}, QWK={train_qwk:.4f}")
    logger.info(f"Validation: F1={val_f1:.4f}, QWK={val_qwk:.4f}")

    # Return F1 as main score (can switch to QWK if preferred)
    return val_f1, metrics


# =============================================================================
# FEATURE EXTRACTION FOR MAP-ELITES
# =============================================================================


def extract_features(code: str, adata) -> Tuple[float, float]:
    """
    Extract feature dimensions for MAP-Elites diversity.

    Features:
    - preprocessing_complexity: Based on code structure [0, 1]
    - normalization_strength: Based on output statistics [0, 1]

    Args:
        code: Python code string of the preprocessing script
        adata: Processed AnnData object

    Returns:
        Tuple of (complexity, normalization_strength)
    """
    # Complexity: count preprocessing operations in code
    code_lower = code.lower()
    operations = [
        "filter_genes", "filter_cells", "normalize_total",
        "log1p", "highly_variable_genes", "scale", "pca",
        "batch_correct", "combat", "harmony", "scran",
        "pearson_residuals", "regress_out", "normalize_per_cell",
    ]
    op_count = sum(1 for op in operations if op in code_lower)
    complexity = min(1.0, op_count / 6.0)

    # Normalization strength: based on data statistics
    data = get_matrix_data(adata.X)
    if len(data) == 0:
        return complexity, 0.0

    # Higher mean and lower variance suggests stronger normalization
    nonzero_data = data[data > 0] if np.any(data > 0) else data
    if len(nonzero_data) > 0:
        mean_val = np.mean(nonzero_data)
        # Normalize to [0, 1] assuming log-normalized data has mean in [1, 5]
        norm_strength = min(1.0, mean_val / 5.0)
    else:
        norm_strength = 0.0

    return complexity, norm_strength


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================


def evaluate(script_path: str) -> Dict[str, Any]:
    """
    Main evaluation function called by OpenEvolve.

    Performs multi-stage cascade evaluation:
    - Stage 0: Execute the preprocessing script
    - Stage 1: Statistical validation (if Stage 0 passes)
    - Stage 2: Frozen backbone evaluation (if Stage 1 >= threshold)

    Args:
        script_path: Path to the preprocessing Python script

    Returns:
        Evaluation result dictionary with:
        - combined_score: Final weighted score [0, 1]
        - stage: Highest stage reached (0, 1, or 2)
        - stats_score: Stage 1 score (if reached)
        - model_score: Stage 2 score (if reached)
        - metrics: Detailed metrics from each stage
        - error: Error message (if any)
    """
    import anndata as ad

    start_time = time.time()

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
            "elapsed_sec": time.time() - start_time,
        }

    # Stage 0: Execute preprocessing
    logger.info("\n[Stage 0] Executing preprocessing script...")
    try:
        # Load module dynamically
        spec = importlib.util.spec_from_file_location("preprocess_module", script_path)
        module = importlib.util.module_from_spec(spec)

        # Add paths for imports
        paths_to_add = [str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"), str(SCRIPT_DIR)]
        for p in paths_to_add:
            sys.path.insert(0, p)

        try:
            spec.loader.exec_module(module)
        finally:
            for p in paths_to_add:
                if p in sys.path:
                    sys.path.remove(p)

        # Check for preprocess function
        if not hasattr(module, "preprocess"):
            return {
                "combined_score": 0.0,
                "stage": 0,
                "error": "Script missing 'preprocess' function",
                "elapsed_sec": time.time() - start_time,
            }

        # Check proxy data exists
        if not os.path.exists(PROXY_DATA_PATH):
            return {
                "combined_score": 0.0,
                "stage": 0,
                "error": f"Proxy data not found: {PROXY_DATA_PATH}",
                "elapsed_sec": time.time() - start_time,
            }

        # Load proxy data and run preprocessing with timeout
        raw_adata = ad.read_h5ad(PROXY_DATA_PATH)

        def run_preprocess():
            return module.preprocess(raw_adata.copy(), GENEMAP_PATH)

        processed_adata = run_with_timeout(
            run_preprocess,
            timeout_seconds=EVAL_TIMEOUT
        )

        # Validate output type
        if not isinstance(processed_adata, ad.AnnData):
            return {
                "combined_score": 0.0,
                "stage": 0,
                "error": f"preprocess() returned {type(processed_adata)}, expected AnnData",
                "elapsed_sec": time.time() - start_time,
            }

        logger.info(f"  Output shape: {processed_adata.shape}")

    except concurrent.futures.TimeoutError:
        return {
            "combined_score": 0.0,
            "stage": 0,
            "error": f"Preprocessing timed out after {EVAL_TIMEOUT}s",
            "elapsed_sec": time.time() - start_time,
        }
    except Exception as e:
        logger.error(f"Stage 0 failed: {e}")
        return {
            "combined_score": 0.0,
            "stage": 0,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_sec": time.time() - start_time,
        }

    # Stage 1: Statistical validation
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
            "elapsed_sec": time.time() - start_time,
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
            "elapsed_sec": time.time() - start_time,
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
            "elapsed_sec": time.time() - start_time,
        }

    # Calculate combined score
    combined_score = 0.3 * stats_score + 0.7 * model_score

    # Extract features for MAP-Elites
    try:
        complexity, norm_strength = extract_features(code, processed_adata)
    except Exception:
        complexity, norm_strength = 0.5, 0.5

    # Final result
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL SCORE: {combined_score:.4f}")
    logger.info(f"  Stats: {stats_score:.4f}, Model: {model_score:.4f}")
    logger.info(f"  Elapsed: {elapsed:.2f}s")
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
        "elapsed_sec": elapsed,
    }


# =============================================================================
# STAGED EVALUATION ENTRY POINTS
# =============================================================================


def evaluate_stage1(script_path: str) -> Dict[str, Any]:
    """
    Fast gate: execution + statistical validation only.
    No expensive backbone evaluation.

    Use this for quick pre-filtering of candidate scripts.
    """
    import anndata as ad

    start_time = time.time()

    try:
        # Load and execute script
        spec = importlib.util.spec_from_file_location("preprocess_module", script_path)
        module = importlib.util.module_from_spec(spec)

        paths_to_add = [str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"), str(SCRIPT_DIR)]
        for p in paths_to_add:
            sys.path.insert(0, p)

        try:
            spec.loader.exec_module(module)
        finally:
            for p in paths_to_add:
                if p in sys.path:
                    sys.path.remove(p)

        if not hasattr(module, "preprocess"):
            return {"combined_score": 0.0, "error": "MissingFunction"}

        if not os.path.exists(PROXY_DATA_PATH):
            return {"combined_score": 0.0, "error": "MissingData"}

        raw_adata = ad.read_h5ad(PROXY_DATA_PATH)
        processed_adata = run_with_timeout(
            lambda: module.preprocess(raw_adata.copy(), GENEMAP_PATH),
            timeout_seconds=int(os.environ.get("STAGE1_TIMEOUT", "60"))
        )

        if not isinstance(processed_adata, ad.AnnData):
            return {"combined_score": 0.0, "error": "InvalidOutput"}

        stats_score, stats_metrics = evaluate_statistics(processed_adata)

        return {
            "combined_score": stats_score,
            "stage": 1,
            "stats_score": stats_score,
            "metrics": {"stage1": stats_metrics},
            "elapsed_sec": time.time() - start_time,
        }

    except concurrent.futures.TimeoutError:
        return {"combined_score": 0.0, "error": "Timeout"}
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }


def evaluate_stage2(script_path: str) -> Dict[str, Any]:
    """
    Full evaluation including backbone.
    Alias for evaluate() for API consistency.
    """
    return evaluate(script_path)


# =============================================================================
# CLI INTERFACE
# =============================================================================


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python evaluator.py <script_path> [--stage1-only]")
        print("\nExamples:")
        print("  python evaluator.py candidates/baseline_params.py")
        print("  python evaluator.py candidates/baseline_code.py --stage1-only")
        sys.exit(1)

    script_path = sys.argv[1]
    stage1_only = "--stage1-only" in sys.argv

    if stage1_only:
        result = evaluate_stage1(script_path)
    else:
        result = evaluate(script_path)

    print("\n" + "=" * 60)
    print("EVALUATION RESULT")
    print("=" * 60)

    # Pretty print result
    for key, value in result.items():
        if key == "metrics":
            print(f"  {key}:")
            for stage, stage_metrics in value.items():
                print(f"    {stage}:")
                for k, v in stage_metrics.items():
                    if isinstance(v, float):
                        print(f"      {k}: {v:.4f}")
                    else:
                        print(f"      {k}: {v}")
        elif key == "traceback":
            print(f"  {key}: (see below)")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    if "traceback" in result:
        print("\nTraceback:")
        print(result["traceback"])

    # Also output as JSON for programmatic use
    print("\n" + "-" * 60)
    print("JSON Output:")
    print(json.dumps(result, indent=2, default=str))
