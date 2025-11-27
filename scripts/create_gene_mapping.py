#!/usr/bin/env python3
"""
Create Gene Vocabulary Mapping for scGPT Fine-tuning.

This script:
1. Loads processed train/val/test datasets
2. Aligns dataset genes with scGPT's pretrained vocabulary
3. Generates alignment report with statistics
4. Saves aligned datasets for scGPT training

Task 3: Foundation Model Gene Vocabulary Alignment
"""

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.gene_vocab_alignment import create_gene_mapping_for_scgpt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create gene vocabulary mapping for scGPT fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: align processed data from results/processed/
  python scripts/create_gene_mapping.py

  # Custom paths
  python scripts/create_gene_mapping.py \\
    --data-dir ./results/processed \\
    --scgpt-vocab ./examples/save/scGPT_bc/vocab.json \\
    --output-dir ./results/scgpt_aligned

  # Use intersection strategy with 30% minimum coverage
  python scripts/create_gene_mapping.py \\
    --strategy intersect \\
    --min-coverage 0.3
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./results/processed",
        help="Directory with processed data (default: ./results/processed)",
    )

    parser.add_argument(
        "--scgpt-vocab",
        type=str,
        default="./examples/save/scGPT_bc/vocab.json",
        help="Path to scGPT vocab.json file (default: ./examples/save/scGPT_bc/vocab.json)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/scgpt_aligned",
        help="Directory to save aligned data (default: ./results/scgpt_aligned)",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["intersect", "pad_oov", "drop_oov"],
        default="intersect",
        help="Alignment strategy (default: intersect)",
    )

    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.3,
        help="Minimum coverage threshold (default: 0.3 = 30%%)",
    )

    return parser.parse_args()


def main():
    """Main function to create gene mapping."""
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("GENE VOCABULARY MAPPING FOR scGPT")
    logger.info("=" * 80)

    # Setup paths
    data_dir = Path(args.data_dir)
    scgpt_vocab_path = Path(args.scgpt_vocab)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify paths
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Run scripts/load.py first to create processed data")
        return False

    if not scgpt_vocab_path.exists():
        logger.error(f"scGPT vocabulary not found: {scgpt_vocab_path}")
        logger.error("Expected location: examples/save/scGPT_bc/vocab.json")
        return False

    logger.info(f"\nConfiguration:")
    logger.info(f"  Data directory: {data_dir}")
    logger.info(f"  scGPT vocab: {scgpt_vocab_path}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Min coverage: {args.min_coverage:.1%}")

    # Process train set (use for alignment)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Processing Training Data")
    logger.info("=" * 80)

    train_path = data_dir / "train.h5ad"
    if not train_path.exists():
        logger.error(f"Train data not found: {train_path}")
        return False

    try:
        logger.info(f"Loading train data from {train_path}")
        train_data = ad.read_h5ad(train_path)
        logger.info(f"Train data shape: {train_data.shape}")

        # Create gene mapping
        train_aligned, alignment_report = create_gene_mapping_for_scgpt(
            adata=train_data,
            scgpt_vocab_path=str(scgpt_vocab_path),
            output_path=str(output_dir / "gene_vocab_mapping.json"),
            strategy=args.strategy,
            min_coverage=args.min_coverage,
        )

        # Save aligned train data
        train_output_path = output_dir / "train_aligned.h5ad"
        train_aligned.write_h5ad(train_output_path)
        logger.info(f"✓ Saved aligned train data to {train_output_path}")

    except Exception as e:
        logger.error(f"Failed to process train data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Process val set (use same gene list as train)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Processing Validation Data")
    logger.info("=" * 80)

    val_path = data_dir / "val.h5ad"
    if val_path.exists():
        try:
            logger.info(f"Loading val data from {val_path}")
            val_data = ad.read_h5ad(val_path)
            logger.info(f"Val data shape: {val_data.shape}")

            # Align to same genes as train
            aligned_genes = train_aligned.var_names
            val_aligned = val_data[:, val_data.var_names.isin(aligned_genes)].copy()

            # Reorder to match train gene order
            val_aligned = val_aligned[:, aligned_genes]

            logger.info(f"Val data aligned: {val_aligned.shape}")

            # Save aligned val data
            val_output_path = output_dir / "val_aligned.h5ad"
            val_aligned.write_h5ad(val_output_path)
            logger.info(f"✓ Saved aligned val data to {val_output_path}")

        except Exception as e:
            logger.error(f"Failed to process val data: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        logger.warning(f"Val data not found: {val_path}, skipping")

    # Process test set (use same gene list as train)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Processing Test Data")
    logger.info("=" * 80)

    test_path = data_dir / "test.h5ad"
    if test_path.exists():
        try:
            logger.info(f"Loading test data from {test_path}")
            test_data = ad.read_h5ad(test_path)
            logger.info(f"Test data shape: {test_data.shape}")

            # Align to same genes as train
            aligned_genes = train_aligned.var_names
            test_aligned = test_data[:, test_data.var_names.isin(aligned_genes)].copy()

            # Reorder to match train gene order
            test_aligned = test_aligned[:, aligned_genes]

            logger.info(f"Test data aligned: {test_aligned.shape}")

            # Save aligned test data
            test_output_path = output_dir / "test_aligned.h5ad"
            test_aligned.write_h5ad(test_output_path)
            logger.info(f"✓ Saved aligned test data to {test_output_path}")

        except Exception as e:
            logger.error(f"Failed to process test data: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        logger.warning(f"Test data not found: {test_path}, skipping")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("GENE MAPPING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info(f"  - gene_vocab_mapping.json (alignment report)")
    logger.info(f"  - train_aligned.h5ad ({train_aligned.n_obs} cells × {train_aligned.n_vars} genes)")
    if val_path.exists():
        logger.info(f"  - val_aligned.h5ad ({val_aligned.n_obs} cells × {val_aligned.n_vars} genes)")
    if test_path.exists():
        logger.info(f"  - test_aligned.h5ad ({test_aligned.n_obs} cells × {test_aligned.n_vars} genes)")

    logger.info(f"\nAlignment Summary:")
    logger.info(f"  Dataset coverage: {alignment_report['coverage']['dataset_coverage']:.1%}")
    logger.info(f"  Genes retained: {alignment_report['intersection']:,} / {alignment_report['dataset_genes']:,}")
    logger.info(f"  Genes dropped (OOV): {alignment_report['oov_genes']:,}")

    if alignment_report['coverage']['dataset_coverage'] >= 0.5:
        logger.info(f"\n✅ Good coverage! scGPT fine-tuning can proceed.")
    else:
        logger.warning(f"\n⚠️  Low coverage! Consider checking gene ID format.")

    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review alignment report: {output_dir}/gene_vocab_mapping.json")
    logger.info(f"  2. Train scGPT with aligned data: python scripts/train_scgpt.py --data-dir {output_dir}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
