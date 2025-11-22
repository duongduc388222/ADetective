#!/usr/bin/env python3
"""
Prepare clinical and pathology variables for MLP training.

This script:
1. Load pre-filtered oligodendrocyte data
2. Select specified clinical/pathology variables (no gene expression)
3. Encode categorical variables
4. Create donor-level stratified train/val/test splits
5. Save processed datasets

Variables included:
- Predictive: APOE genotype, Age at death, Sex, Race/ethnicity, Education,
              Cognitive status, pathology measures (%6E10+, %AT8+, %GFAP+, %pTDP43+, %aSyn+)
- Batch: Method, Library prep
- QC: PMI
"""

import logging
import sys
from pathlib import Path
from typing import Tuple
import json

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_clinical_data():
    """Prepare clinical and pathology data for training."""

    logger.info("=" * 80)
    logger.info("CLINICAL DATA PREPARATION FOR MLP TRAINING")
    logger.info("=" * 80)

    # Load config
    config = Config()
    data_path = config.get("data.path")
    output_dir = Path(config.get("output.results_dir")) / "clinical_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    logger.info(f"\n[1/5] Loading oligodendrocyte data from {data_path}")
    adata = ad.read_h5ad(data_path)
    logger.info(f"  Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Define variable groups
    predictive_vars = [
        "APOE Genotype",
        "Age at Death",
        "Sex",
        # Race/ethnicity variables (multiple boolean columns)
        "Race (choice=White)",
        "Race (choice=Black/ African American)",
        "Race (choice=Asian)",
        "Race (choice=American Indian/ Alaska Native)",
        "Race (choice=Native Hawaiian or Pacific Islander)",
        "Race (choice=Other)",
        "Race (choice=Unknown or unreported)",
        "Hispanic/Latino",
        "Years of education",
        "Cognitive Status",
        # Pathology measures
        "percent 6e10 positive area",
        "percent AT8 positive area",
        "percent GFAP positive area",
        "percent pTDP43 positive area",
        "percent aSyn positive area",
    ]

    batch_vars = [
        "Method",
        "library_prep",
    ]

    qc_vars = [
        "PMI",
    ]

    stratify_var = "Donor ID"
    label_var = "label"  # Created binary label: High (1) vs Not AD (0)

    # Check all variables exist
    logger.info(f"\n[2/5] Verifying variables in dataset")
    all_vars = predictive_vars + batch_vars + qc_vars + [stratify_var, label_var]
    missing_vars = [v for v in all_vars if v not in adata.obs.columns]
    if missing_vars:
        logger.error(f"  Missing variables: {missing_vars}")
        logger.error(f"  Available columns: {list(adata.obs.columns)}")
        return False

    logger.info(f"  ✓ All variables found")
    logger.info(f"    - Predictive: {len(predictive_vars)} variables")
    logger.info(f"    - Batch: {len(batch_vars)} variables")
    logger.info(f"    - QC: {len(qc_vars)} variables")

    # Extract relevant columns
    logger.info(f"\n[3/5] Extracting and encoding variables")
    df = adata.obs[[*predictive_vars, *batch_vars, *qc_vars, stratify_var, label_var]].copy()

    logger.info(f"  Initial dataset: {df.shape}")
    logger.info(f"  Missing values:\n{df.isnull().sum()}")

    # Handle missing values
    df = df.dropna()
    logger.info(f"  After removing NaNs: {df.shape}")

    # Encode categorical variables
    label_encoders = {}
    categorical_vars = [
        "APOE Genotype",
        "Sex",
        "Cognitive Status",
        "Method",
        "library_prep",
        "Hispanic/Latino",
    ]

    for var in categorical_vars:
        if var in df.columns:
            le = LabelEncoder()
            df[var] = le.fit_transform(df[var].astype(str))
            label_encoders[var] = le
            logger.info(f"  Encoded {var}: {len(le.classes_)} classes")

    # Separate features and labels
    X = df[predictive_vars + batch_vars + qc_vars].values.astype(np.float32)
    y = df[label_var].values.astype(np.int64)
    donors = df[stratify_var].values

    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  y distribution: {np.bincount(y)}")
    logger.info(f"  Unique donors: {len(np.unique(donors))}")

    # Create donor-level stratified splits
    logger.info(f"\n[4/5] Creating donor-level stratified splits")

    unique_donors = np.unique(donors)
    np.random.seed(42)
    np.random.shuffle(unique_donors)

    # Split by donor
    n_donors = len(unique_donors)
    train_idx = int(0.7 * n_donors)
    val_idx = int(0.1 * n_donors) + train_idx

    train_donors = set(unique_donors[:train_idx])
    val_donors = set(unique_donors[train_idx:val_idx])
    test_donors = set(unique_donors[val_idx:])

    train_mask = np.array([d in train_donors for d in donors])
    val_mask = np.array([d in val_donors for d in donors])
    test_mask = np.array([d in test_donors for d in donors])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info(f"  Train: {X_train.shape[0]} samples from {len(train_donors)} donors")
    logger.info(f"  Val:   {X_val.shape[0]} samples from {len(val_donors)} donors")
    logger.info(f"  Test:  {X_test.shape[0]} samples from {len(test_donors)} donors")
    logger.info(f"  Train label dist: {np.bincount(y_train)}")
    logger.info(f"  Val label dist: {np.bincount(y_val)}")
    logger.info(f"  Test label dist: {np.bincount(y_test)}")

    # Fit scaler on train, apply to all
    logger.info(f"\n[5/5] Standardizing features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    logger.info(f"  Features standardized (mean=0, std=1)")

    # Save as numpy arrays
    logger.info(f"\nSaving processed data to {output_dir}")

    np.save(output_dir / "X_train.npy", X_train_scaled)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val_scaled)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test_scaled)
    np.save(output_dir / "y_test.npy", y_test)

    # Save feature names and metadata
    feature_names = predictive_vars + batch_vars + qc_vars
    metadata = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "predictive_vars": predictive_vars,
        "batch_vars": batch_vars,
        "qc_vars": qc_vars,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "label_encoders": {k: v.classes_.tolist() for k, v in label_encoders.items()},
        "train_donors": list(train_donors),
        "val_donors": list(val_donors),
        "test_donors": list(test_donors),
        "data_shapes": {
            "train": X_train_scaled.shape,
            "val": X_val_scaled.shape,
            "test": X_test_scaled.shape,
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  ✓ Saved X_train.npy: {X_train_scaled.shape}")
    logger.info(f"  ✓ Saved X_val.npy: {X_val_scaled.shape}")
    logger.info(f"  ✓ Saved X_test.npy: {X_test_scaled.shape}")
    logger.info(f"  ✓ Saved metadata.json")

    logger.info(f"\n" + "=" * 80)
    logger.info(f"Clinical Data Preparation Complete!")
    logger.info(f"=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Features: {len(feature_names)} clinical/pathology variables")
    logger.info(f"  - Predictive: {len(predictive_vars)}")
    logger.info(f"  - Batch: {len(batch_vars)}")
    logger.info(f"  - QC: {len(qc_vars)}")

    return True


if __name__ == "__main__":
    success = prepare_clinical_data()
    sys.exit(0 if success else 1)
