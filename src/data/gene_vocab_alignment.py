"""
Gene Vocabulary Alignment for scGPT Foundation Model Fine-tuning.

This module provides functionality to align dataset gene vocabularies with
scGPT's pretrained gene vocabulary, handling mismatches and documenting
out-of-vocabulary (OOV) genes.

Task 3 Implementation: Foundation Model Gene Vocabulary Alignment
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GeneVocabAligner:
    """
    Align dataset gene vocabulary with scGPT pretrained vocabulary.

    Handles:
    - Loading scGPT vocab from checkpoint
    - Mapping dataset genes to vocab indices
    - Identifying out-of-vocabulary (OOV) genes
    - Generating alignment statistics and reports
    """

    def __init__(
        self,
        scgpt_vocab_path: str,
        dataset_gene_id_column: str = "gene_ids",
        dataset_gene_symbol_column: Optional[str] = None,
    ):
        """
        Initialize gene vocabulary aligner.

        Args:
            scgpt_vocab_path: Path to scGPT vocab.json file
            dataset_gene_id_column: Column in adata.var with gene IDs
            dataset_gene_symbol_column: Column in adata.var with gene symbols (optional)
        """
        self.scgpt_vocab_path = Path(scgpt_vocab_path)
        self.dataset_gene_id_column = dataset_gene_id_column
        self.dataset_gene_symbol_column = dataset_gene_symbol_column

        if not self.scgpt_vocab_path.exists():
            raise FileNotFoundError(
                f"scGPT vocabulary file not found: {scgpt_vocab_path}\n"
                f"Expected location: examples/save/scGPT_bc/vocab.json"
            )

        # Load scGPT vocabulary
        self.scgpt_vocab = self._load_scgpt_vocab()
        logger.info(f"Loaded scGPT vocabulary with {len(self.scgpt_vocab)} genes")

    def _load_scgpt_vocab(self) -> Dict[str, int]:
        """
        Load scGPT vocabulary from JSON file.

        Returns:
            Dictionary mapping gene names to vocab indices
        """
        with open(self.scgpt_vocab_path, 'r') as f:
            vocab = json.load(f)

        logger.info(f"scGPT vocab loaded from {self.scgpt_vocab_path}")
        logger.info(f"  Vocab size: {len(vocab)} genes")

        # Show sample genes
        sample_genes = list(vocab.keys())[:5]
        logger.info(f"  Sample genes: {sample_genes}")

        return vocab

    def align_dataset(
        self,
        adata: ad.AnnData,
        strategy: str = "intersect",
        min_coverage: float = 0.3,
    ) -> Tuple[ad.AnnData, Dict[str, Any]]:
        """
        Align dataset genes with scGPT vocabulary.

        Args:
            adata: AnnData object with gene expression data
            strategy: Alignment strategy:
                - "intersect": Keep only genes in both dataset and vocab
                - "pad_oov": Keep all dataset genes, pad OOV with special token
                - "drop_oov": Drop OOV genes (same as intersect)
            min_coverage: Minimum required coverage (intersection / dataset genes)
                         Raises error if coverage is below this threshold

        Returns:
            Tuple of (aligned AnnData, alignment report dict)
        """
        logger.info(f"Aligning dataset genes with scGPT vocabulary")
        logger.info(f"  Strategy: {strategy}")
        logger.info(f"  Min coverage threshold: {min_coverage:.1%}")

        # Extract dataset genes
        dataset_genes = self._extract_dataset_genes(adata)
        logger.info(f"  Dataset genes: {len(dataset_genes)}")

        # Compute intersection
        scgpt_gene_set = set(self.scgpt_vocab.keys())
        dataset_gene_set = set(dataset_genes)

        intersection = scgpt_gene_set & dataset_gene_set
        oov_genes = dataset_gene_set - scgpt_gene_set
        scgpt_not_in_dataset = scgpt_gene_set - dataset_gene_set

        # Calculate coverage
        coverage = len(intersection) / len(dataset_genes) if len(dataset_genes) > 0 else 0.0

        logger.info(f"  Intersection: {len(intersection)} genes ({coverage:.1%})")
        logger.info(f"  OOV genes (dataset only): {len(oov_genes)}")
        logger.info(f"  Vocab genes not in dataset: {len(scgpt_not_in_dataset)}")

        # Check minimum coverage
        if coverage < min_coverage:
            raise ValueError(
                f"Gene vocabulary alignment failed: coverage {coverage:.1%} "
                f"is below minimum threshold {min_coverage:.1%}.\n"
                f"Dataset has {len(dataset_genes)} genes, but only {len(intersection)} "
                f"match scGPT's vocabulary.\n"
                f"This may indicate:\n"
                f"  1. Dataset uses different gene IDs (Ensembl vs Symbol)\n"
                f"  2. Dataset is from different species\n"
                f"  3. Wrong scGPT checkpoint loaded"
            )

        # Apply alignment strategy
        if strategy == "intersect" or strategy == "drop_oov":
            adata_aligned = self._align_by_intersection(adata, intersection, dataset_genes)
        elif strategy == "pad_oov":
            adata_aligned = self._align_with_padding(adata, intersection, oov_genes, dataset_genes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Create alignment report
        alignment_report = self._create_alignment_report(
            dataset_genes=dataset_genes,
            intersection=intersection,
            oov_genes=oov_genes,
            scgpt_not_in_dataset=scgpt_not_in_dataset,
            coverage=coverage,
            strategy=strategy,
        )

        logger.info(f"✓ Alignment complete: {adata_aligned.n_vars} genes retained")

        return adata_aligned, alignment_report

    def _extract_dataset_genes(self, adata: ad.AnnData) -> List[str]:
        """Extract gene names from dataset."""
        # Try gene symbol column first (preferred for scGPT)
        if self.dataset_gene_symbol_column and self.dataset_gene_symbol_column in adata.var:
            genes = adata.var[self.dataset_gene_symbol_column].tolist()
            logger.info(f"  Using gene symbols from '{self.dataset_gene_symbol_column}' column")
        # Fallback to gene ID column
        elif self.dataset_gene_id_column in adata.var:
            genes = adata.var[self.dataset_gene_id_column].tolist()
            logger.info(f"  Using gene IDs from '{self.dataset_gene_id_column}' column")
        # Fallback to var_names (index)
        else:
            genes = adata.var_names.tolist()
            logger.info(f"  Using adata.var_names (index) as gene names")

        return genes

    def _align_by_intersection(
        self,
        adata: ad.AnnData,
        intersection: set,
        dataset_genes: List[str],
    ) -> ad.AnnData:
        """Align by keeping only genes in intersection."""
        # Create mask for genes in intersection
        gene_mask = np.array([gene in intersection for gene in dataset_genes])

        # Filter adata
        adata_aligned = adata[:, gene_mask].copy()

        logger.info(f"  Filtered {adata.n_vars} → {adata_aligned.n_vars} genes (intersection)")

        return adata_aligned

    def _align_with_padding(
        self,
        adata: ad.AnnData,
        intersection: set,
        oov_genes: set,
        dataset_genes: List[str],
    ) -> ad.AnnData:
        """Align by keeping all genes, marking OOV with special handling."""
        # Add OOV marker to var
        adata_aligned = adata.copy()
        adata_aligned.var['in_scgpt_vocab'] = [
            gene in intersection for gene in dataset_genes
        ]
        adata_aligned.var['is_oov'] = [
            gene in oov_genes for gene in dataset_genes
        ]

        logger.info(f"  Kept all {adata_aligned.n_vars} genes, marked {len(oov_genes)} as OOV")

        return adata_aligned

    def _create_alignment_report(
        self,
        dataset_genes: List[str],
        intersection: set,
        oov_genes: set,
        scgpt_not_in_dataset: set,
        coverage: float,
        strategy: str,
    ) -> Dict[str, Any]:
        """Create detailed alignment report."""
        report = {
            "dataset_genes": len(dataset_genes),
            "scgpt_vocab_size": len(self.scgpt_vocab),
            "intersection": len(intersection),
            "oov_genes": len(oov_genes),
            "scgpt_genes_not_in_dataset": len(scgpt_not_in_dataset),
            "coverage": {
                "dataset_coverage": coverage,  # % dataset genes in scGPT vocab
                "vocab_coverage": len(intersection) / len(self.scgpt_vocab) if len(self.scgpt_vocab) > 0 else 0.0,  # % scGPT vocab in dataset
            },
            "strategy": strategy,
            "oov_gene_list": sorted(list(oov_genes))[:500],  # First 500 OOV genes
            "intersection_gene_list": sorted(list(intersection))[:100],  # First 100 matched genes
            "gene_id_column_used": self.dataset_gene_id_column,
            "gene_symbol_column_used": self.dataset_gene_symbol_column,
        }

        return report

    def save_alignment_report(
        self,
        report: Dict[str, Any],
        output_path: str,
    ) -> None:
        """
        Save alignment report to JSON file.

        Args:
            report: Alignment report dictionary
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"✓ Alignment report saved to {output_path}")

        # Print summary
        logger.info("=" * 80)
        logger.info("GENE VOCABULARY ALIGNMENT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Dataset genes:                {report['dataset_genes']:,}")
        logger.info(f"scGPT vocab size:            {report['scgpt_vocab_size']:,}")
        logger.info(f"Intersection:                {report['intersection']:,} genes")
        logger.info(f"OOV genes (dataset only):    {report['oov_genes']:,}")
        logger.info(f"Dataset coverage:            {report['coverage']['dataset_coverage']:.1%}")
        logger.info(f"Vocab coverage:              {report['coverage']['vocab_coverage']:.1%}")
        logger.info(f"Strategy:                    {report['strategy']}")
        logger.info("=" * 80)

        # Warnings
        if report['coverage']['dataset_coverage'] < 0.5:
            logger.warning(
                f"⚠️  Low coverage ({report['coverage']['dataset_coverage']:.1%})! "
                f"More than half of dataset genes are OOV."
            )
            logger.warning(
                "   This may significantly impact scGPT performance. Consider:\n"
                "   1. Verify gene ID format (Ensembl vs Symbol)\n"
                "   2. Check if dataset uses human genes (scGPT is human-trained)\n"
                "   3. Use alternative model that doesn't require vocab alignment"
            )


def create_gene_mapping_for_scgpt(
    adata: ad.AnnData,
    scgpt_vocab_path: str,
    output_path: str,
    strategy: str = "intersect",
    min_coverage: float = 0.3,
) -> Tuple[ad.AnnData, Dict[str, Any]]:
    """
    Convenience function to create gene mapping for scGPT fine-tuning.

    Args:
        adata: AnnData object with gene expression data
        scgpt_vocab_path: Path to scGPT vocab.json file
        output_path: Path to save alignment report
        strategy: Alignment strategy ("intersect", "pad_oov", "drop_oov")
        min_coverage: Minimum required coverage

    Returns:
        Tuple of (aligned AnnData, alignment report)
    """
    aligner = GeneVocabAligner(
        scgpt_vocab_path=scgpt_vocab_path,
        dataset_gene_id_column="gene_ids",
        dataset_gene_symbol_column=None,  # Will use var_names if not found
    )

    adata_aligned, report = aligner.align_dataset(
        adata=adata,
        strategy=strategy,
        min_coverage=min_coverage,
    )

    aligner.save_alignment_report(report, output_path)

    return adata_aligned, report
