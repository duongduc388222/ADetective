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
            data_path: Path to SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad file
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
        Load raw SEAAD dataset using memory-efficient backed mode.

        Uses backed="r" (read-only memory-mapped access) to avoid loading
        the entire 18GB file into RAM. Filtering operations will only load
        the subset of interest into memory.

        Returns:
            AnnData object with backed (memory-mapped) access
        """
        logger.info(f"Loading SEAAD dataset from {self.data_path}")
        logger.info("💾 Using backed='r' mode for memory-efficient loading (memory-mapped access)")
        logger.info("   Data will only be loaded into memory when filtered or accessed")

        # Load with backed='r' for memory-efficient access (read-only, memory-mapped)
        self.adata = ad.read_h5ad(self.data_path, backed="r")
        logger.info(f"Loaded data shape (backed): {self.adata.shape}")
        logger.info(f"Observations (cells): {self.adata.n_obs}")
        logger.info(f"Variables (genes): {self.adata.n_vars}")
        logger.info("✓ Metadata loaded successfully. Expression matrix remains on disk.")

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

        Loads only the filtered subset into memory (not the full dataset).

        Args:
            cell_type: Cell type to filter for (default: "Oligodendrocyte")

        Returns:
            Filtered AnnData object (loaded into memory)
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

        # Create a copy to load filtered subset into memory (more efficient than full load)
        logger.info("🔍 Filtering... (loading filtered subset into memory)")
        adata_filtered = self.adata[
            self.adata.obs[self.cell_type_column].str.lower() == cell_type.lower()
        ].to_memory()

        logger.info(f"Cells before filtering: {before_count:,}")
        logger.info(f"Cells after filtering:  {adata_filtered.n_obs:,}")
        logger.info(f"✓ Filtered subset loaded ({adata_filtered.n_obs} cells)")

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

        # Show ADNC value distribution before filtering
        logger.info(f"ADNC value distribution before filtering:")
        logger.info(self.adata.obs[self.adnc_column].value_counts())

        # Filter out excluded categories (loads filtered subset into memory)
        logger.info("🔍 Filtering by ADNC status... (loading filtered subset into memory)")
        mask = ~self.adata.obs[self.adnc_column].isin(exclude_categories)
        adata_filtered = self.adata[mask].to_memory()

        logger.info(f"Cells retained: {adata_filtered.n_obs:,} / {self.adata.n_obs:,}")
        logger.info(f"✓ Filtered subset loaded")

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
        min_label_ratio: float = 0.3,
    ) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
        """
        Create donor-level stratified train/val/test split with balanced label distribution.

        Ensures:
        - No donor appears in multiple splits
        - Each split maintains balanced label distribution (both High and Not AD)
        - Validates that minority class is at least min_label_ratio in each split

        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed
            min_label_ratio: Minimum ratio for minority class in each split (default: 0.3)

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

        logger.info("Creating donor-level stratified split with label balance validation")
        logger.info(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        logger.info(f"Using donor column: {self.donor_column}")
        logger.info(f"Minimum label ratio per split: {min_label_ratio:.1%}")

        # Get donor-level label (majority vote per donor to ensure stratification works properly)
        donor_labels = (
            self.adata.obs.groupby(self.donor_column)["label"]
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index()
        )
        donor_labels.columns = [self.donor_column, "donor_label"]

        logger.info(f"Total unique donors: {len(donor_labels)}")
        logger.info(f"Donor-level label distribution:")
        logger.info(donor_labels["donor_label"].value_counts().to_dict())

        # First split: train+val vs test
        donors_train_val, donors_test = train_test_split(
            donor_labels[self.donor_column],
            test_size=test_ratio,
            random_state=random_state,
            stratify=donor_labels["donor_label"],
        )

        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        train_val_labels = donor_labels[donor_labels[self.donor_column].isin(donors_train_val)]
        donors_train, donors_val = train_test_split(
            train_val_labels[self.donor_column],
            test_size=val_size,
            random_state=random_state,
            stratify=train_val_labels["donor_label"],
        )

        # Create splits (loads each split into memory separately)
        logger.info("🔍 Creating train/val/test splits and loading into memory...")
        train_adata = self.adata[
            self.adata.obs[self.donor_column].isin(donors_train)
        ].to_memory()
        val_adata = self.adata[self.adata.obs[self.donor_column].isin(donors_val)].to_memory()
        test_adata = self.adata[self.adata.obs[self.donor_column].isin(donors_test)].to_memory()

        logger.info(f"Train set: {train_adata.n_obs:,} cells from {donors_train.nunique()} donors")
        logger.info(f"Val set:   {val_adata.n_obs:,} cells from {donors_val.nunique()} donors")
        logger.info(f"Test set:  {test_adata.n_obs:,} cells from {donors_test.nunique()} donors")
        logger.info(f"✓ Splits created and loaded into memory")

        # CRITICAL: Check for single-class splits (causes overfitting)
        for split_name, split_data in [("Validation", val_adata), ("Test", test_adata)]:
            label_counts = split_data.obs["label"].value_counts()
            if len(label_counts) < 2:
                error_msg = (
                    f"❌ CRITICAL ERROR: {split_name} set has only one class!\n"
                    f"  Current distribution: {label_counts.to_dict()}\n"
                    f"  This will cause severe overfitting and invalid evaluation.\n\n"
                    f"  Solutions:\n"
                    f"  1. Increase --val-ratio and --test-ratio (try 0.15/0.15 instead of 0.1/0.2)\n"
                    f"  2. Try a different random_state\n"
                    f"  3. Use more donors if possible"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Validate label distributions
        logger.info("\n" + "="*60)
        logger.info("LABEL DISTRIBUTION VALIDATION")
        logger.info("="*60)

        all_balanced = True
        for split_name, split_data in [
            ("Train", train_adata),
            ("Val", val_adata),
            ("Test", test_adata),
        ]:
            label_dist = split_data.obs["label_name"].value_counts()
            label_ratios = label_dist / label_dist.sum()

            logger.info(f"\n{split_name} set:")
            logger.info(f"  Counts: {label_dist.to_dict()}")
            logger.info(f"  Ratios: {label_ratios.to_dict()}")

            # Check if minority class meets minimum threshold
            min_ratio = label_ratios.min()
            if min_ratio < min_label_ratio:
                logger.warning(
                    f"  ⚠️  WARNING: Minority class ratio ({min_ratio:.1%}) is below "
                    f"minimum threshold ({min_label_ratio:.1%})"
                )
                all_balanced = False
            else:
                logger.info(f"  ✓ Label distribution is balanced (min ratio: {min_ratio:.1%})")

        logger.info("="*60)

        if all_balanced:
            logger.info("✓ All splits have balanced label distributions")
        else:
            logger.warning(
                "⚠️  Some splits have imbalanced label distributions. "
                "Consider adjusting split ratios or using different random_state."
            )

        return train_adata, val_adata, test_adata

    def select_hvgs(self, adata: ad.AnnData, n_hvgs: int = 2000) -> ad.AnnData:
        """
        Select highly variable genes.

        For backed AnnData, loads data into memory after HVG selection
        (more efficient than loading full data first).

        Args:
            adata: AnnData object (backed or in-memory)
            n_hvgs: Number of HVGs to select

        Returns:
            AnnData with HVGs selected and loaded into memory
        """
        logger.info(f"Selecting {n_hvgs} highly variable genes")

        # Check if HVGs already computed
        if "highly_variable" not in adata.var:
            import scanpy as sc

            logger.info("📊 Computing highly variable genes...")
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs)
            logger.info(f"✓ HVGs computed")
        else:
            logger.info("HVGs already computed")

        # Select HVGs and load into memory
        logger.info(f"🔍 Selecting {n_hvgs} HVGs and loading into memory...")
        adata_hvg = adata[:, adata.var.highly_variable].to_memory()
        logger.info(f"✓ Selected {adata_hvg.n_vars:,} highly variable genes and loaded into memory")

        return adata_hvg

    def normalize_data(self, adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
        """
        Normalize RNA-seq data to library size and log-transform.

        Performs library-size normalization (each cell to same total count)
        followed by log1p transformation.

        Args:
            adata: AnnData object
            target_sum: Target sum for library size normalization (default: 10,000)

        Returns:
            Normalized and log-transformed AnnData object
        """
        import scanpy as sc

        # Check current preprocessing state
        preproc_state = self.check_preprocessing_state(adata)

        if preproc_state["is_normalized"] and preproc_state["is_log_transformed"]:
            logger.info("Data already normalized and log-transformed, skipping...")
            return adata

        logger.info(f"Normalizing data to target_sum={target_sum} and log-transforming...")

        if not preproc_state["is_normalized"]:
            logger.info("📊 Performing library-size normalization...")
            sc.pp.normalize_total(adata, target_sum=target_sum)
            logger.info("✓ Normalization complete")
        else:
            logger.info("Data already normalized, skipping normalization step")

        if not preproc_state["is_log_transformed"]:
            logger.info("📊 Performing log1p transformation...")
            sc.pp.log1p(adata)
            logger.info("✓ Log transformation complete")
        else:
            logger.info("Data already log-transformed, skipping log transformation step")

        # Verify preprocessing
        final_state = self.check_preprocessing_state(adata)
        logger.info(f"Final preprocessing state: normalized={final_state['is_normalized']}, "
                   f"log_transformed={final_state['is_log_transformed']}")

        return adata

    def check_preprocessing_state(self, adata: ad.AnnData) -> Dict[str, Any]:
        """
        Check preprocessing state of data (log transformation and normalization).

        Args:
            adata: AnnData object

        Returns:
            Dictionary with preprocessing status: is_log_transformed, is_normalized, dtype
        """
        # Check for log transformation
        # Log-transformed data typically has:
        # - Maximum values < 20 (since log1p(large_number) is relatively small)
        # - Minimum values >= 0 (log1p never produces negative values)
        X_data = adata.X
        if hasattr(X_data, 'toarray'):
            # Handle sparse matrices
            max_val = X_data.max()
            min_val = X_data.min()
        else:
            max_val = np.max(X_data)
            min_val = np.min(X_data)

        is_log_transformed = (max_val < 20) and (min_val >= 0)

        # Check for normalization (library-size normalization)
        # Normalized data has row sums approximately equal to a constant
        row_sums = np.array(X_data.sum(axis=1)).flatten()
        is_normalized = np.std(row_sums) / np.mean(row_sums) < 0.1  # CV < 10%

        logger.info(f"Log transformation status: {'Yes' if is_log_transformed else 'No'}")
        logger.info(f"Normalization status: {'Yes' if is_normalized else 'No'}")

        return {
            "is_log_transformed": is_log_transformed,
            "is_normalized": is_normalized,
            "dtype": str(X_data.dtype)
        }

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
