import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SEAADDataLoader:
    """Load and preprocess SEAAD dataset for oligodendrocyte AD classification."""

    def __init__(
        self,
        data_path: str,
        cache_processed: bool = True,
        cell_type_column: str = "Subclass",
        donor_column: str = "Donor ID",
        adnc_column: str = "ADNC",
        cell_supertype_column: Optional[str] = "Supertype",
    ):
        """
        Initialize SEAAD data loader.

        Args:
            data_path: Path to SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad file
            cache_processed: Whether to cache processed data
            cell_type_column: Column name for cell type (default: "Subclass" - fine-grained)
            donor_column: Column name for donor ID (default: "Donor ID")
            adnc_column: Column name for ADNC status (default: "ADNC")
            cell_supertype_column: Column name for cell supertype (default: "Supertype")
        """
        self.data_path = Path(data_path)
        self.cache_processed = cache_processed
        self.adata = None
        self.metadata = None

        # Column name configuration
        self.cell_type_column = cell_type_column
        self.donor_column = donor_column
        self.adnc_column = adnc_column
        self.cell_supertype_column = cell_supertype_column

        logger.info(f"Configured columns for SEAAD analysis:")
        logger.info(f"  Cell type (primary): {self.cell_type_column}")
        logger.info(f"  Cell supertype (optional): {self.cell_supertype_column}")
        logger.info(f"  Donor ID: {self.donor_column}")
        logger.info(f"  ADNC status: {self.adnc_column}")
        logger.info(f"  Focus: Oligodendrocyte cells with High vs Not AD classification")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

    def load_raw_data(self) -> ad.AnnData:
        """
        Load raw SEAAD dataset.

        Returns:
            AnnData object with raw data
        """
        logger.info(f"Loading SEAAD dataset from {self.data_path}")
        logger.warning("⚠️ Loading dataset into memory...")

        # Load into memory (not backed mode) to allow filtering and manipulation
        self.adata = ad.read_h5ad(self.data_path)
        logger.info(f"Loaded data shape: {self.adata.shape}")
        logger.info(f"Observations (cells): {self.adata.n_obs}")
        logger.info(f"Variables (genes): {self.adata.n_vars}")

        return self.adata

    def explore_metadata(self) -> Dict[str, Any]:
        """
        Explore and document dataset metadata.

        Returns:
            Dictionary with metadata information
        """
        if self.adata is None:
            self.load_raw_data()

        obs_cols = self.adata.obs.columns.tolist()
        var_cols = self.adata.var.columns.tolist()

        metadata_info = {
            "obs_columns": obs_cols,
            "var_columns": var_cols,
            "n_obs": self.adata.n_obs,
            "n_vars": self.adata.n_vars,
        }

        logger.info("Observation columns:")
        for col in obs_cols:
            unique_vals = self.adata.obs[col].nunique() if col in self.adata.obs else 0
            logger.info(f"  - {col}: {unique_vals} unique values")

        logger.info("Variable columns:")
        for col in var_cols:
            unique_vals = self.adata.var[col].nunique() if col in self.adata.var else 0
            logger.info(f"  - {col}: {unique_vals} unique values")

        return metadata_info

    def get_column_info(self, column: str) -> pd.Series:
        """Get information about a specific column."""
        if self.adata is None:
            self.load_raw_data()

        if column in self.adata.obs:
            return self.adata.obs[column].value_counts()
        else:
            raise ValueError(f"Column '{column}' not found in observations")

    def filter_cell_type(self, cell_type: str = "Oligodendrocyte") -> ad.AnnData:
        """
        Filter data for specific cell type using configured column.

        Args:
            cell_type: Cell type to filter for (default: "Oligodendrocyte")

        Returns:
            Filtered AnnData object
        """
        if self.adata is None:
            self.load_raw_data()

        logger.info(f"Filtering for {cell_type} cells using column '{self.cell_type_column}'")
        before_count = self.adata.n_obs

        # Verify column exists
        if self.cell_type_column not in self.adata.obs:
            raise ValueError(
                f"Column '{self.cell_type_column}' not found in observations.\n"
                f"Available columns: {list(self.adata.obs.columns)}"
            )

        # Create a copy to avoid modifying backed dataset
        adata_filtered = self.adata[
            self.adata.obs[self.cell_type_column].str.lower() == cell_type.lower()
        ].copy()

        logger.info(f"Cells before filtering: {before_count}")
        logger.info(f"Cells after filtering: {adata_filtered.n_obs}")

        self.adata = adata_filtered
        return self.adata

    def create_labels(
        self, exclude_categories: Optional[list] = None
    ) -> Tuple[ad.AnnData, Dict[str, int]]:
        """
        Create binary labels: High AD (1) vs Not AD (0).

        Excludes 'Low' and 'Intermediate' categories by default.

        Args:
            exclude_categories: Categories to exclude (default: ['Low', 'Intermediate'])

        Returns:
            Tuple of (filtered AnnData, label mapping)
        """
        if exclude_categories is None:
            exclude_categories = ["Low", "Intermediate"]

        if self.adata is None:
            self.load_raw_data()

        logger.info(f"Creating labels from column '{self.adnc_column}'")
        logger.info(f"Excluding categories: {exclude_categories}")

        # Verify column exists
        if self.adnc_column not in self.adata.obs:
            raise ValueError(
                f"Column '{self.adnc_column}' not found in observations.\n"
                f"Available columns: {list(self.adata.obs.columns)}"
            )

        # Get ADNC values
        adnc_values = self.adata.obs[self.adnc_column].copy()
        logger.info(f"ADNC value distribution before filtering:")
        logger.info(self.adata.obs[self.adnc_column].value_counts())

        # Filter out excluded categories
        mask = ~self.adata.obs[self.adnc_column].isin(exclude_categories)
        adata_filtered = self.adata[mask].copy()

        logger.info(f"Cells retained: {adata_filtered.n_obs} / {self.adata.n_obs}")

        # Create binary labels
        label_mapping = {"Not AD": 0, "High": 1}
        labels = (adata_filtered.obs[self.adnc_column] == "High").astype(int)

        adata_filtered.obs["label"] = labels
        adata_filtered.obs["label_name"] = labels.map({0: "Not AD", 1: "High"})

        logger.info("Label distribution:")
        logger.info(adata_filtered.obs["label_name"].value_counts())

        self.adata = adata_filtered
        return self.adata, label_mapping

    def get_donor_info(self) -> Dict[str, int]:
        """
        Get donor distribution per class using configured donor column.

        Returns:
            Dictionary with donor counts
        """
        if self.adata is None:
            self.load_raw_data()

        if self.donor_column not in self.adata.obs:
            raise ValueError(
                f"Column '{self.donor_column}' not found in observations.\n"
                f"Available columns: {list(self.adata.obs.columns)}"
            )

        donor_class_info = {}
        for label in [0, 1]:
            label_subset = self.adata[self.adata.obs["label"] == label]
            donors = label_subset.obs[self.donor_column].nunique()
            donor_class_info[f"label_{label}"] = donors
            logger.info(f"Label {label}: {donors} unique donors")

        return donor_class_info

    def stratified_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
        """
        Create donor-level stratified train/val/test split.

        Ensures no donor appears in multiple splits using configured donor column.

        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed

        Returns:
            Tuple of (train_adata, val_adata, test_adata)
        """
        if self.adata is None:
            self.load_raw_data()

        if "label" not in self.adata.obs:
            raise ValueError("Labels not created yet. Call create_labels() first.")

        if self.donor_column not in self.adata.obs:
            raise ValueError(
                f"Column '{self.donor_column}' not found in observations.\n"
                f"Available columns: {list(self.adata.obs.columns)}"
            )

        logger.info("Creating donor-level stratified split")
        logger.info(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        logger.info(f"Using donor column: {self.donor_column}")

        # Get unique donors per class
        unique_donors = self.adata.obs[[self.donor_column, "label"]].drop_duplicates()

        # First split: train+val vs test
        donors_train_val, donors_test, _, _ = train_test_split(
            unique_donors[self.donor_column],
            unique_donors["label"],
            test_size=test_ratio,
            random_state=random_state,
            stratify=unique_donors["label"],
        )

        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        donors_train, donors_val, _, _ = train_test_split(
            donors_train_val,
            unique_donors[unique_donors[self.donor_column].isin(donors_train_val)]["label"],
            test_size=val_size,
            random_state=random_state,
            stratify=unique_donors[unique_donors[self.donor_column].isin(donors_train_val)]["label"],
        )

        # Create splits
        train_adata = self.adata[
            self.adata.obs[self.donor_column].isin(donors_train)
        ].copy()
        val_adata = self.adata[self.adata.obs[self.donor_column].isin(donors_val)].copy()
        test_adata = self.adata[self.adata.obs[self.donor_column].isin(donors_test)].copy()

        logger.info(f"Train set: {train_adata.n_obs} cells from {donors_train.nunique()} donors")
        logger.info(f"Val set:   {val_adata.n_obs} cells from {donors_val.nunique()} donors")
        logger.info(f"Test set:  {test_adata.n_obs} cells from {donors_test.nunique()} donors")

        # Log label distributions
        for split_name, split_data in [
            ("Train", train_adata),
            ("Val", val_adata),
            ("Test", test_adata),
        ]:
            label_dist = split_data.obs["label_name"].value_counts()
            logger.info(f"{split_name} label distribution: {label_dist.to_dict()}")

        return train_adata, val_adata, test_adata

    def select_hvgs(self, adata: ad.AnnData, n_hvgs: int = 2000) -> ad.AnnData:
        """
        Select highly variable genes.

        Args:
            adata: AnnData object
            n_hvgs: Number of HVGs to select

        Returns:
            AnnData with HVGs selected
        """
        logger.info(f"Selecting {n_hvgs} highly variable genes")

        # Check if HVGs already computed
        if "highly_variable" not in adata.var:
            import scanpy as sc

            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs)
        else:
            logger.info("HVGs already computed")

        adata_hvg = adata[:, adata.var.highly_variable].copy()
        logger.info(f"Selected {adata_hvg.n_vars} highly variable genes")

        return adata_hvg

    def check_normalization(self, adata: ad.AnnData) -> Dict[str, Any]:
        """
        Check if data is normalized.

        Args:
            adata: AnnData object

        Returns:
            Dictionary with normalization status
        """
        # Check for log-normalization indicators
        is_normalized = "log1p" in adata.obs_names or np.allclose(
            adata.X.sum(axis=1), 1.0, rtol=0.1
        )

        logger.info(f"Normalization status: {'Normalized' if is_normalized else 'Not normalized'}")

        return {"is_normalized": is_normalized, "dtype": str(adata.X.dtype)}

    def save_processed(self, adata: ad.AnnData, output_path: str) -> None:
        """
        Save processed AnnData.

        Args:
            adata: AnnData object to save
            output_path: Path to save to
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        adata.write_h5ad(output_path)
        logger.info(f"Saved processed data to {output_path}")
