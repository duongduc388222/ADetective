#!/usr/bin/env python3
"""
Phase 2: Load, explore, and prepare cleaned dataset for model training.

This script handles:
1. Loading the cleaned SEAAD dataset
2. Exploring metadata and verifying data structure
3. Filtering for oligodendrocytes and creating binary labels
4. Creating donor-level stratified train/val/test splits
5. Saving preprocessed datasets
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.data.loaders import SEAADDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Load and preprocess cleaned dataset."""
    logger.info("=" * 80)
    logger.info("PHASE 2: DATA LOADING AND PREPROCESSING")
    logger.info("=" * 80)

    # Load configuration
    config = Config(env="local")
    data_path = config.get("data.path")
    output_dir = Path(config.get("output.results_dir")) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nData path: {data_path}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize data loader
    try:
        loader = SEAADDataLoader(data_path)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return False

    # Step 1: Load raw data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Raw Data")
    logger.info("=" * 80)
    try:
        adata = loader.load_raw_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False

    # Step 2: Explore metadata
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Exploring Metadata")
    logger.info("=" * 80)
    try:
        metadata_info = loader.explore_metadata()
    except Exception as e:
        logger.error(f"Failed to explore metadata: {e}")
        return False

    # Step 3: Filter for oligodendrocytes
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Filtering for Oligodendrocytes")
    logger.info("=" * 80)
    try:
        adata = loader.filter_cell_type("Oligodendrocyte")
    except Exception as e:
        logger.error(f"Failed to filter cell type: {e}")
        logger.error(f"Error details: {e}")
        return False

    # Step 4: Create binary labels (High vs Not AD)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Creating Binary Labels")
    logger.info("=" * 80)
    try:
        adata, label_mapping = loader.create_labels()
    except Exception as e:
        logger.error(f"Failed to create labels: {e}")
        logger.error(f"Error details: {e}")
        return False

    # Step 5: Get donor distribution
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Donor Distribution")
    logger.info("=" * 80)
    try:
        donor_info = loader.get_donor_info()
        logger.info(f"Donor distribution: {donor_info}")
    except Exception as e:
        logger.warning(f"Could not retrieve donor info: {e}")

    # Step 6: Create stratified splits
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Creating Train/Val/Test Splits")
    logger.info("=" * 80)
    try:
        train_data, val_data, test_data = loader.stratified_split(
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
        )
    except Exception as e:
        logger.error(f"Failed to create splits: {e}")
        return False

    # Step 7: Select HVGs
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Selecting Highly Variable Genes")
    logger.info("=" * 80)
    n_hvgs = 2000
    try:
        train_data = loader.select_hvgs(train_data, n_hvgs)
        val_data = loader.select_hvgs(val_data, n_hvgs)
        test_data = loader.select_hvgs(test_data, n_hvgs)
    except Exception as e:
        logger.warning(f"Could not select HVGs: {e}")

    # Step 8: Check normalization
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Checking Normalization")
    logger.info("=" * 80)
    try:
        norm_info = loader.check_normalization(train_data)
        logger.info(f"Normalization status: {norm_info}")
    except Exception as e:
        logger.warning(f"Could not check normalization: {e}")

    # Step 9: Save processed data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 9: Saving Processed Data")
    logger.info("=" * 80)
    try:
        train_path = str(output_dir / "train.h5ad")
        val_path = str(output_dir / "val.h5ad")
        test_path = str(output_dir / "test.h5ad")

        loader.save_processed(train_data, train_path)
        loader.save_processed(val_data, val_path)
        loader.save_processed(test_data, test_path)

        logger.info(f"✓ Saved train data: {train_path}")
        logger.info(f"✓ Saved val data: {val_path}")
        logger.info(f"✓ Saved test data: {test_path}")

    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        return False

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DATA LOADING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nSummary:")
    logger.info(f"  Train: {train_data.n_obs} cells × {train_data.n_vars} genes")
    logger.info(f"  Val:   {val_data.n_obs} cells × {val_data.n_vars} genes")
    logger.info(f"  Test:  {test_data.n_obs} cells × {test_data.n_vars} genes")
    logger.info(f"\nProcessed data saved to: {output_dir}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
