#!/usr/bin/env python3
"""
Phase 2: Explore SEAAD dataset and understand its structure.

This script loads the 35GB dataset and explores metadata without intensive processing.
⚠️ WARNING: This will take several minutes to load the data on first run.
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
    """Explore SEAAD dataset."""
    logger.info("=" * 80)
    logger.info("SEAAD Dataset Exploration")
    logger.info("=" * 80)

    # Load configuration
    config = Config(env="local")
    data_path = config.get("data.path")

    logger.info(f"Data path: {data_path}")
    logger.info(f"Project root: {config.project_root}")

    # Initialize data loader
    try:
        loader = SEAADDataLoader(data_path)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info("Please ensure the data file exists at the configured path")
        return

    # Load raw data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Raw Data")
    logger.info("=" * 80)
    adata = loader.load_raw_data()

    # Explore metadata
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Metadata Exploration")
    logger.info("=" * 80)
    metadata_info = loader.explore_metadata()

    # Explore key columns
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Key Column Information")
    logger.info("=" * 80)

    # Check for common column names
    possible_cell_type_cols = [
        col for col in adata.obs.columns if "cell" in col.lower() or "type" in col.lower()
    ]
    possible_donor_cols = [
        col for col in adata.obs.columns if "donor" in col.lower() or "individual" in col.lower()
    ]
    possible_adnc_cols = [
        col for col in adata.obs.columns
        if "adnc" in col.lower() or "pathology" in col.lower() or "ad" in col.lower()
    ]

    logger.info(f"Possible cell type columns: {possible_cell_type_cols}")
    logger.info(f"Possible donor columns: {possible_donor_cols}")
    logger.info(f"Possible ADNC columns: {possible_adnc_cols}")

    # Sample actual values from each possible column
    if possible_cell_type_cols:
        col = possible_cell_type_cols[0]
        logger.info(f"\nCell types in '{col}':")
        logger.info(loader.get_column_info(col))

    if possible_donor_cols:
        col = possible_donor_cols[0]
        logger.info(f"\nDonors in '{col}':")
        print(f"  Unique donors: {adata.obs[col].nunique()}")

    if possible_adnc_cols:
        col = possible_adnc_cols[0]
        logger.info(f"\nADNC status in '{col}':")
        logger.info(loader.get_column_info(col))

    logger.info("\n" + "=" * 80)
    logger.info("Exploration Complete!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Verify column names match expected names in prepare_data.py")
    logger.info("2. Update prepare_data.py with correct column names if needed")
    logger.info("3. Run prepare_data.py to filter and preprocess data")


if __name__ == "__main__":
    main()
