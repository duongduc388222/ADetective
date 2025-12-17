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

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import SEAADDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load, explore, and preprocess SEAAD dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local usage (with normalization enabled by default)
  python scripts/load.py \\
    --data-path ./data/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad \\
    --output-dir ./results

  # Skip normalization
  python scripts/load.py \\
    --data-path ./data/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad \\
    --output-dir ./results \\
    --no-normalize

  # Custom parameters
  python scripts/load.py \\
    --data-path ./data/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad \\
    --output-dir ./results \\
    --target-sum 10000 \\
    --n-hvgs 2000 \\
    --min-label-ratio 0.3

  # Google Colab
  python scripts/load.py \\
    --data-path /content/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad \\
    --output-dir ./results \\
    --n-hvgs 2000

  # With scGPT vocabulary pre-filtering (guarantees 100% vocab coverage)
  python scripts/load.py \\
    --data-path ./data/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad \\
    --output-dir ./results \\
    --scgpt-vocab ./examples/save/scGPT_bc/vocab.json \\
    --n-hvgs 2000
        """,
    )

    parser.add_argument("--data-path", type=str, required=True, help="Path to the SEAAD H5AD file (e.g., /content/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad)")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save processed data (default: ./results)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of donors for training set (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of donors for validation set (default: 0.15, increased from 0.1 for better stratification)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Fraction of donors for test set (default: 0.15, decreased from 0.2 for better stratification)")
    parser.add_argument("--n-hvgs", type=int, default=2000, help="Number of highly variable genes to select (default: 2000)")
    parser.add_argument("--min-cells", type=int, default=100, help="Minimum number of cells a gene must be expressed in (default: 100, set to 0 to skip)")
    parser.add_argument("--no-normalize", action="store_true", help="Skip normalization (library-size normalization + log1p transform)")
    parser.add_argument("--target-sum", type=float, default=1e4, help="Target sum for library-size normalization (default: 10000)")
    parser.add_argument("--min-label-ratio", type=float, default=0.3, help="Minimum ratio for minority class in each split (default: 0.3)")
    parser.add_argument("--scgpt-vocab", type=str, default=None, help="Path to scGPT vocab.json to pre-filter genes (ensures 100%% vocab coverage for scGPT training)")
    parser.add_argument("--vocab-min-coverage", type=float, default=0.3, help="Minimum vocab coverage threshold (default: 0.3 = 30%%)")

    return parser.parse_args()


def main():
    """Load and preprocess cleaned dataset."""
    args = parse_arguments()

    # Validate split ratios
    if not (abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 0.01):
        logger.error(
            f"Split ratios must sum to 1.0, got {args.train_ratio + args.val_ratio + args.test_ratio}"
        )
        return False

    logger.info("=" * 80)
    logger.info("PHASE 2: DATA LOADING AND PREPROCESSING")
    logger.info("=" * 80)

    # Setup paths
    data_path = args.data_path
    output_dir = Path(args.output_dir) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nData path: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Train/Val/Test split: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    logger.info(f"Number of HVGs: {args.n_hvgs}")
    if args.scgpt_vocab:
        logger.info(f"scGPT vocabulary: {args.scgpt_vocab}")
        logger.info(f"  (Pre-filtering genes to vocab for 100% coverage)")

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

    # Step 4: Filter genes (min_cells)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Filtering Genes (min_cells)")
    logger.info("=" * 80)
    try:
        adata = loader.filter_genes(min_cells=args.min_cells)
    except Exception as e:
        logger.error(f"Failed to filter genes: {e}")
        logger.error(f"Error details: {e}")
        return False

    # Step 4.5: Filter to vocabulary (optional, for scGPT)
    vocab_report = None
    if args.scgpt_vocab:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4.5: Filtering to scGPT Vocabulary")
        logger.info("=" * 80)
        logger.info("This ensures 100% vocab coverage during scGPT training")
        try:
            adata, vocab_report = loader.filter_to_vocab(
                vocab_path=args.scgpt_vocab,
                min_coverage=args.vocab_min_coverage,
            )
            logger.info(f"Vocabulary coverage: {vocab_report['coverage']:.1%}")
            logger.info(f"Genes retained: {vocab_report['intersection']:,}")
            logger.info(f"OOV genes dropped: {vocab_report['oov_genes_dropped']:,}")
        except FileNotFoundError as e:
            logger.error(f"Vocabulary file not found: {e}")
            return False
        except ValueError as e:
            logger.error(f"Vocabulary coverage too low: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to filter to vocabulary: {e}")
            return False

    # Step 5: Create binary labels (High vs Not AD)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Creating Binary Labels")
    logger.info("=" * 80)
    try:
        adata, label_mapping = loader.create_labels()
    except Exception as e:
        logger.error(f"Failed to create labels: {e}")
        logger.error(f"Error details: {e}")
        return False

    # Step 6: Check preprocessing state
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Checking Preprocessing State")
    logger.info("=" * 80)
    try:
        preprocess_info = loader.check_preprocessing_state(adata)
        logger.info(f"Preprocessing state: {preprocess_info}")
    except Exception as e:
        logger.warning(f"Could not check preprocessing state: {e}")

    # Step 7: Get donor distribution
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Donor Distribution")
    logger.info("=" * 80)
    try:
        donor_info = loader.get_donor_info()
        logger.info(f"Donor distribution: {donor_info}")
    except Exception as e:
        logger.warning(f"Could not retrieve donor info: {e}")

    # Step 8: Create stratified splits (BEFORE normalization to prevent data leakage)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Creating Train/Val/Test Splits")
    logger.info("=" * 80)
    try:
        train_data, val_data, test_data = loader.stratified_split(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            min_label_ratio=args.min_label_ratio,
        )
    except Exception as e:
        logger.error(f"Failed to create splits: {e}")
        return False

    # Step 9: Normalize data (AFTER splitting, BEFORE HVG selection to prevent data leakage)
    if not args.no_normalize:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 9: Normalizing Data (After Split, Before HVG)")
        logger.info("=" * 80)
        logger.info("ℹ️  Normalizing each split independently to prevent data leakage")
        try:
            logger.info("Normalizing training set...")
            train_data = loader.normalize_data(train_data, target_sum=args.target_sum)

            logger.info("Normalizing validation set...")
            val_data = loader.normalize_data(val_data, target_sum=args.target_sum)

            logger.info("Normalizing test set...")
            test_data = loader.normalize_data(test_data, target_sum=args.target_sum)

            logger.info("✓ All splits normalized independently")
        except Exception as e:
            logger.error(f"Failed to normalize data: {e}")
            return False
    else:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 9: Skipping Normalization (--no-normalize flag set)")
        logger.info("=" * 80)
        logger.info("Note: Normalization is enabled by default. Use --no-normalize to skip.")

    # Step 10: Select HVGs (on NORMALIZED data, only on training data to prevent data leakage)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 10: Selecting Highly Variable Genes")
    logger.info("=" * 80)
    try:
        # Select HVGs on normalized training data only
        logger.info(f"Selecting {args.n_hvgs} HVGs from normalized training data...")
        train_data = loader.select_hvgs(train_data, args.n_hvgs)
        hvg_genes = train_data.var_names.copy()

        logger.info(f"✓ Selected {len(hvg_genes)} HVGs from training set")
        logger.info(f"Subsetting validation and test sets to use the same genes...")

        # Subset val and test to use the SAME genes as training (prevent data leakage)
        val_data = val_data[:, hvg_genes].copy()
        test_data = test_data[:, hvg_genes].copy()

        logger.info(f"✓ Val data subset to {val_data.n_vars} genes (same as train)")
        logger.info(f"✓ Test data subset to {test_data.n_vars} genes (same as train)")

        # Verify gene overlap
        train_genes_set = set(train_data.var_names)
        val_genes_set = set(val_data.var_names)
        test_genes_set = set(test_data.var_names)

        assert train_genes_set == val_genes_set == test_genes_set, \
            "Gene sets must be identical across splits!"
        logger.info(f"✓ Verified: All splits use identical {len(train_genes_set)} genes")

    except Exception as e:
        logger.warning(f"Could not select HVGs: {e}")

    # Step 11: Verify final preprocessing state
    logger.info("\n" + "=" * 80)
    logger.info("STEP 11: Verifying Final Preprocessing State")
    logger.info("=" * 80)
    try:
        preprocess_info = loader.check_preprocessing_state(train_data)
        logger.info(f"Final preprocessing state: {preprocess_info}")
    except Exception as e:
        logger.warning(f"Could not check preprocessing state: {e}")

    # Step 12: Save processed data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 12: Saving Processed Data")
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

    # Save vocab report if vocabulary filtering was done
    if vocab_report is not None:
        import json
        vocab_report_path = output_dir / "vocab_filter_report.json"
        with open(vocab_report_path, 'w') as f:
            json.dump(vocab_report, f, indent=2)
        logger.info(f"✓ Saved vocab filter report: {vocab_report_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DATA LOADING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nSummary:")
    logger.info(f"  Train: {train_data.n_obs} cells × {train_data.n_vars} genes")
    logger.info(f"  Val:   {val_data.n_obs} cells × {val_data.n_vars} genes")
    logger.info(f"  Test:  {test_data.n_obs} cells × {test_data.n_vars} genes")
    if vocab_report is not None:
        logger.info(f"\nVocabulary filtering:")
        logger.info(f"  Coverage: {vocab_report['coverage']:.1%}")
        logger.info(f"  Genes in vocab: {vocab_report['intersection']:,}")
        logger.info(f"  OOV genes dropped: {vocab_report['oov_genes_dropped']:,}")
    logger.info(f"\nProcessed data saved to: {output_dir}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
