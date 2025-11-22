#!/usr/bin/env python3
"""
Phase 2: Data Preparation and Preprocessing.

This script handles the complete data preprocessing pipeline:
1. Load raw SEAAD data
2. Explore and validate metadata
3. Filter for oligodendrocytes
4. Create binary labels (High vs Not AD)
5. Create donor-level stratified train/val/test splits
6. Select highly variable genes
7. Save processed datasets

⚠️ WARNING: Processing the 35GB dataset may take 30+ minutes
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.data.loaders import SEAADDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_data(
    config: Config,
    cell_type_col: str = "cell_type",
    cell_type_filter: str = "Oligodendrocyte",
    donor_col: str = "donor_id",
    adnc_col: str = "ADNC",
    n_hvgs: int = 2000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
):
    """
    Run complete data preparation pipeline.

    Args:
        config: Configuration object
        cell_type_col: Column name for cell type
        cell_type_filter: Cell type to filter for
        donor_col: Column name for donor ID
        adnc_col: Column name for ADNC status
        n_hvgs: Number of highly variable genes
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    logger.info("=" * 80)
    logger.info("SEAAD Data Preprocessing Pipeline")
    logger.info("=" * 80)

    data_path = config.get("data.path")
    output_dir = Path(config.get("output.results_dir")) / "processed_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loader
    logger.info(f"\n[1/6] Initializing data loader")
    try:
        loader = SEAADDataLoader(data_path)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return False

    # Load raw data
    logger.info(f"\n[2/6] Loading raw SEAAD data")
    logger.warning("⚠️  35GB file - this will take several minutes...")
    try:
        adata = loader.load_raw_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False

    # Explore metadata
    logger.info(f"\n[3/6] Exploring metadata")
    try:
        loader.explore_metadata()
    except Exception as e:
        logger.error(f"Failed to explore metadata: {e}")
        return False

    # Filter for oligodendrocytes
    logger.info(f"\n[4/6] Filtering for {cell_type_filter} cells")
    try:
        adata = loader.filter_cell_type(cell_type_filter, cell_type_col)
    except Exception as e:
        logger.error(f"Failed to filter cell type: {e}")
        logger.info(f"Available columns: {list(adata.obs.columns)}")
        return False

    # Create labels
    logger.info(f"\n[5/6] Creating binary labels")
    try:
        adata, label_mapping = loader.create_labels(adnc_col)
    except Exception as e:
        logger.error(f"Failed to create labels: {e}")
        logger.info(f"Available columns: {list(adata.obs.columns)}")
        return False

    # Get donor info
    try:
        donor_info = loader.get_donor_info(donor_col)
        logger.info(f"Donor distribution: {donor_info}")
    except Exception as e:
        logger.warning(f"Could not retrieve donor info: {e}")

    # Create stratified splits
    logger.info(f"\n[6/6] Creating train/val/test splits")
    try:
        train_data, val_data, test_data = loader.stratified_split(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            donor_column=donor_col,
        )
    except Exception as e:
        logger.error(f"Failed to create splits: {e}")
        return False

    # Process HVGs for each split
    logger.info(f"\nSelecting {n_hvgs} highly variable genes for each split")
    try:
        train_data = loader.select_hvgs(train_data, n_hvgs)
        val_data = loader.select_hvgs(val_data, n_hvgs)
        test_data = loader.select_hvgs(test_data, n_hvgs)
    except Exception as e:
        logger.warning(f"Could not select HVGs: {e}")

    # Check normalization
    logger.info(f"\nChecking normalization status")
    try:
        norm_info = loader.check_normalization(train_data)
        logger.info(f"Normalization info: {norm_info}")
    except Exception as e:
        logger.warning(f"Could not check normalization: {e}")

    # Save processed data
    logger.info(f"\nSaving processed data to {output_dir}")
    try:
        loader.save_processed(train_data, str(output_dir / "train_data.h5ad"))
        loader.save_processed(val_data, str(output_dir / "val_data.h5ad"))
        loader.save_processed(test_data, str(output_dir / "test_data.h5ad"))

        # Save metadata
        metadata_summary = {
            "train_samples": train_data.n_obs,
            "train_genes": train_data.n_vars,
            "val_samples": val_data.n_obs,
            "val_genes": val_data.n_vars,
            "test_samples": test_data.n_obs,
            "test_genes": test_data.n_vars,
            "label_mapping": label_mapping,
            "n_hvgs": n_hvgs,
        }

        import json

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata_summary, f, indent=2)
        logger.info(f"Saved metadata to {output_dir / 'metadata.json'}")

    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        return False

    logger.info("\n" + "=" * 80)
    logger.info("Data Preprocessing Complete!")
    logger.info("=" * 80)
    logger.info(f"\nProcessed data saved to: {output_dir}")
    logger.info(f"\nSummary:")
    logger.info(f"  Train: {train_data.n_obs} cells × {train_data.n_vars} genes")
    logger.info(f"  Val:   {val_data.n_obs} cells × {val_data.n_vars} genes")
    logger.info(f"  Test:  {test_data.n_obs} cells × {test_data.n_vars} genes")

    return True


@click.command()
@click.option(
    "--cell-type-col",
    default="cell_type",
    help="Column name for cell type",
)
@click.option(
    "--cell-type-filter",
    default="Oligodendrocyte",
    help="Cell type to filter for",
)
@click.option(
    "--donor-col",
    default="donor_id",
    help="Column name for donor ID",
)
@click.option(
    "--adnc-col",
    default="ADNC",
    help="Column name for ADNC status",
)
@click.option(
    "--n-hvgs",
    type=int,
    default=2000,
    help="Number of highly variable genes",
)
def main(cell_type_col, cell_type_filter, donor_col, adnc_col, n_hvgs):
    """Run data preparation pipeline."""
    config = Config(env="local")
    config.ensure_dirs()

    success = prepare_data(
        config,
        cell_type_col=cell_type_col,
        cell_type_filter=cell_type_filter,
        donor_col=donor_col,
        adnc_col=adnc_col,
        n_hvgs=n_hvgs,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
