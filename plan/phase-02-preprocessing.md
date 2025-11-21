# Phase 02: Data Exploration & Preprocessing

## Objective
Load the SEA-AD dataset, explore metadata, filter for Oligodendrocytes, create labels based on ADNC status, and prepare train/test splits at the donor level.

## Duration
2-3 hours

## Tasks

### 2.1 Data Loader Implementation
Create `src/data/loader.py`:
```python
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SEAADDataLoader:
    def __init__(self, data_path):
        """
        Initialize data loader for SEA-AD dataset.

        Args:
            data_path: Path to SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad file
        """
        self.data_path = Path(data_path)
        self.adata = None
        self.metadata_columns = None

    def load_data(self):
        """Load AnnData object from h5ad file."""
        logger.info(f"Loading data from {self.data_path}")
        self.adata = sc.read_h5ad(self.data_path)
        logger.info(f"Loaded {self.adata.shape[0]} cells × {self.adata.shape[1]} genes")
        return self.adata

    def explore_metadata(self):
        """Explore and document metadata columns."""
        metadata = {
            'n_cells': self.adata.shape[0],
            'n_genes': self.adata.shape[1],
            'obs_columns': list(self.adata.obs.columns),
            'var_columns': list(self.adata.var.columns)
        }

        # Key columns to identify
        key_columns = ['Donor ID', 'Donor', 'ADNC', 'cell_type', 'cell_class',
                       'cell_subclass', 'Age at Death', 'Sex', 'APOE Genotype',
                       'Braak', 'Thal', 'CERAD']

        for col in key_columns:
            if col in self.adata.obs.columns:
                unique_vals = self.adata.obs[col].unique()
                metadata[f'{col}_unique'] = len(unique_vals)
                if len(unique_vals) < 20:
                    metadata[f'{col}_values'] = list(unique_vals)

        return metadata

    def get_cell_type_column(self):
        """Identify the correct cell type column."""
        possible_columns = ['cell_type', 'cell_class', 'cell_subclass',
                           'Cell Type', 'CellType', 'celltype']

        for col in possible_columns:
            if col in self.adata.obs.columns:
                if 'Oligodendrocyte' in self.adata.obs[col].values or \
                   'Oligo' in self.adata.obs[col].values:
                    logger.info(f"Found cell type column: {col}")
                    return col

        raise ValueError("Could not identify cell type column with Oligodendrocytes")

    def get_donor_column(self):
        """Identify the correct donor ID column."""
        possible_columns = ['Donor ID', 'Donor', 'donor_id', 'DonorID']

        for col in possible_columns:
            if col in self.adata.obs.columns:
                logger.info(f"Found donor column: {col}")
                return col

        raise ValueError("Could not identify donor ID column")
```

### 2.2 Preprocessing Pipeline
Create `src/data/preprocessor.py`:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scanpy as sc
import logging

logger = logging.getLogger(__name__)

class OligodendrocytePreprocessor:
    def __init__(self, adata, donor_col='Donor ID', cell_type_col='cell_type',
                 adnc_col='ADNC'):
        """
        Preprocess Oligodendrocyte cells for AD pathology classification.

        Args:
            adata: AnnData object
            donor_col: Column name for donor ID
            cell_type_col: Column name for cell type
            adnc_col: Column name for ADNC status
        """
        self.adata = adata
        self.donor_col = donor_col
        self.cell_type_col = cell_type_col
        self.adnc_col = adnc_col
        self.adata_filtered = None
        self.donor_to_label = {}

    def filter_oligodendrocytes(self):
        """Filter for Oligodendrocyte cells only."""
        # Find the exact string for Oligodendrocytes
        cell_types = self.adata.obs[self.cell_type_col].unique()
        oligo_types = [ct for ct in cell_types if 'Oligo' in ct or 'oligo' in ct]

        if not oligo_types:
            raise ValueError("No Oligodendrocyte cells found")

        logger.info(f"Found Oligodendrocyte types: {oligo_types}")

        # Filter for Oligodendrocytes
        mask = self.adata.obs[self.cell_type_col].isin(oligo_types)
        self.adata_filtered = self.adata[mask, :].copy()

        logger.info(f"Filtered to {self.adata_filtered.shape[0]} Oligodendrocyte cells")
        return self.adata_filtered

    def create_donor_labels(self):
        """Create binary labels for donors based on ADNC status."""
        # Get unique donors and their ADNC status
        donor_df = self.adata_filtered.obs[[self.donor_col, self.adnc_col]].drop_duplicates()

        # Filter for High vs Not AD only
        high_mask = donor_df[self.adnc_col] == 'High'
        not_ad_mask = donor_df[self.adnc_col] == 'Not AD'
        valid_mask = high_mask | not_ad_mask

        donor_df_filtered = donor_df[valid_mask].copy()

        # Create labels: High=1, Not AD=0
        donor_df_filtered['label'] = (donor_df_filtered[self.adnc_col] == 'High').astype(int)

        # Create mapping
        self.donor_to_label = dict(zip(donor_df_filtered[self.donor_col],
                                       donor_df_filtered['label']))

        logger.info(f"Created labels for {len(self.donor_to_label)} donors:")
        logger.info(f"  High AD: {sum(v == 1 for v in self.donor_to_label.values())} donors")
        logger.info(f"  Not AD: {sum(v == 0 for v in self.donor_to_label.values())} donors")

        return self.donor_to_label

    def filter_by_valid_donors(self):
        """Keep only cells from donors with valid labels."""
        valid_donors = set(self.donor_to_label.keys())
        mask = self.adata_filtered.obs[self.donor_col].isin(valid_donors)
        self.adata_filtered = self.adata_filtered[mask, :].copy()

        # Add label column to cells
        self.adata_filtered.obs['label'] = self.adata_filtered.obs[self.donor_col].map(
            self.donor_to_label)

        logger.info(f"Filtered to {self.adata_filtered.shape[0]} cells from valid donors")

        # Report cell distribution
        label_counts = self.adata_filtered.obs['label'].value_counts()
        logger.info(f"Cell distribution:")
        logger.info(f"  High AD cells: {label_counts[1]}")
        logger.info(f"  Not AD cells: {label_counts[0]}")

        return self.adata_filtered

    def create_donor_split(self, test_size=0.2, val_size=0.1, random_state=42):
        """Create train/val/test split at donor level."""
        # Get unique donors with labels
        donors = list(self.donor_to_label.keys())
        labels = list(self.donor_to_label.values())

        # First split: train+val vs test
        train_val_donors, test_donors = train_test_split(
            donors, test_size=test_size, stratify=labels, random_state=random_state
        )

        # Second split: train vs val
        train_val_labels = [self.donor_to_label[d] for d in train_val_donors]
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val size

        train_donors, val_donors = train_test_split(
            train_val_donors, test_size=val_size_adjusted,
            stratify=train_val_labels, random_state=random_state
        )

        # Create masks for cells
        train_mask = self.adata_filtered.obs[self.donor_col].isin(train_donors)
        val_mask = self.adata_filtered.obs[self.donor_col].isin(val_donors)
        test_mask = self.adata_filtered.obs[self.donor_col].isin(test_donors)

        # Add split information
        self.adata_filtered.obs['split'] = 'none'
        self.adata_filtered.obs.loc[train_mask, 'split'] = 'train'
        self.adata_filtered.obs.loc[val_mask, 'split'] = 'val'
        self.adata_filtered.obs.loc[test_mask, 'split'] = 'test'

        # Report statistics
        split_stats = {
            'train': {
                'donors': len(train_donors),
                'cells': train_mask.sum(),
                'high_donors': sum(self.donor_to_label[d] == 1 for d in train_donors),
                'not_ad_donors': sum(self.donor_to_label[d] == 0 for d in train_donors)
            },
            'val': {
                'donors': len(val_donors),
                'cells': val_mask.sum(),
                'high_donors': sum(self.donor_to_label[d] == 1 for d in val_donors),
                'not_ad_donors': sum(self.donor_to_label[d] == 0 for d in val_donors)
            },
            'test': {
                'donors': len(test_donors),
                'cells': test_mask.sum(),
                'high_donors': sum(self.donor_to_label[d] == 1 for d in test_donors),
                'not_ad_donors': sum(self.donor_to_label[d] == 0 for d in test_donors)
            }
        }

        for split, stats in split_stats.items():
            logger.info(f"{split.upper()} Split:")
            logger.info(f"  Donors: {stats['donors']} ({stats['high_donors']} High, {stats['not_ad_donors']} Not AD)")
            logger.info(f"  Cells: {stats['cells']}")

        return split_stats

    def select_highly_variable_genes(self, n_top_genes=2000, use_existing=True):
        """Select highly variable genes for feature reduction."""
        if use_existing and 'highly_variable' in self.adata_filtered.var.columns:
            # Use existing HVG selection
            logger.info("Using existing highly variable gene selection")
            hvg_mask = self.adata_filtered.var['highly_variable']
        else:
            # Calculate HVGs on training data only
            train_data = self.adata_filtered[self.adata_filtered.obs['split'] == 'train']

            # Save original data
            adata_full = self.adata_filtered.copy()

            # Calculate HVGs
            sc.pp.highly_variable_genes(train_data, n_top_genes=n_top_genes,
                                       subset=False, inplace=False)

            # Apply to full dataset
            hvg_mask = train_data.var['highly_variable']
            self.adata_filtered.var['highly_variable'] = hvg_mask

        # Subset to HVGs
        self.adata_filtered = self.adata_filtered[:, hvg_mask].copy()
        logger.info(f"Selected {self.adata_filtered.shape[1]} highly variable genes")

        return self.adata_filtered

    def normalize_expression(self):
        """Check and apply normalization if needed."""
        # Check if already normalized
        if 'log1p' in self.adata_filtered.uns:
            logger.info("Data appears to be already log-normalized")
        else:
            # Check data range
            data_max = self.adata_filtered.X.max()
            if data_max > 100:
                logger.info("Data appears to be raw counts, applying normalization")
                # Normalize to 10,000 reads per cell
                sc.pp.normalize_total(self.adata_filtered, target_sum=1e4)
                # Log transform
                sc.pp.log1p(self.adata_filtered)
            else:
                logger.info("Data appears to be already normalized (max value: {:.2f})".format(data_max))

        return self.adata_filtered

    def scale_features(self):
        """Scale features for neural network input."""
        # Store unscaled data
        self.adata_filtered.raw = self.adata_filtered.copy()

        # Scale to zero mean and unit variance
        sc.pp.scale(self.adata_filtered, zero_center=True, max_value=None)

        logger.info("Scaled features to zero mean and unit variance")
        return self.adata_filtered
```

### 2.3 Data Exploration Script
Create `scripts/explore_data.py`:
```python
#!/usr/bin/env python3
"""
Script to explore SEA-AD dataset and prepare for model training.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import SEAADDataLoader
from src.data.preprocessor import OligodendrocytePreprocessor
from src.utils.config import Config
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Explore SEA-AD dataset')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad')
    parser.add_argument('--output-dir', type=str, default='./results/exploration',
                       help='Output directory for exploration results')
    args = parser.parse_args()

    # Load data
    loader = SEAADDataLoader(args.data_path)
    adata = loader.load_data()

    # Explore metadata
    metadata = loader.explore_metadata()
    print("\n=== Dataset Overview ===")
    for key, value in metadata.items():
        if not key.endswith('_values'):
            print(f"{key}: {value}")

    # Find correct columns
    cell_type_col = loader.get_cell_type_column()
    donor_col = loader.get_donor_column()

    # Initialize preprocessor
    preprocessor = OligodendrocytePreprocessor(
        adata,
        donor_col=donor_col,
        cell_type_col=cell_type_col
    )

    # Filter Oligodendrocytes
    adata_oligo = preprocessor.filter_oligodendrocytes()

    # Create labels
    donor_labels = preprocessor.create_donor_labels()

    # Filter by valid donors
    adata_filtered = preprocessor.filter_by_valid_donors()

    # Create splits
    split_stats = preprocessor.create_donor_split()

    # Select HVGs
    adata_hvg = preprocessor.select_highly_variable_genes()

    # Normalize and scale
    adata_norm = preprocessor.normalize_expression()
    adata_scaled = preprocessor.scale_features()

    # Save processed data
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'processed_oligodendrocytes.h5ad')
    adata_scaled.write(output_path)
    print(f"\nSaved processed data to: {output_path}")

    # Save split information
    split_info = pd.DataFrame(split_stats).T
    split_info.to_csv(os.path.join(args.output_dir, 'split_statistics.csv'))
    print(f"Saved split statistics to: {os.path.join(args.output_dir, 'split_statistics.csv')}")

if __name__ == '__main__':
    main()
```

## Validation Checklist
- [ ] Data successfully loaded from h5ad file
- [ ] Metadata columns identified (Donor ID, ADNC, cell_type)
- [ ] Oligodendrocyte cells filtered correctly
- [ ] Binary labels created (High=1, Not AD=0)
- [ ] Donors with Low/Intermediate ADNC excluded
- [ ] Donor-level train/val/test split created
- [ ] No data leakage between splits
- [ ] HVGs selected based on training data only
- [ ] Data normalized and scaled appropriately
- [ ] Processed data saved for model training

## Expected Outputs
- Processed AnnData object with:
  - Only Oligodendrocyte cells
  - Binary labels (0/1)
  - Split assignments (train/val/test)
  - Selected HVGs
  - Scaled expression values
- Split statistics showing:
  - Number of donors per split
  - Number of cells per split
  - Class balance in each split

## Next Steps
- Verify data preprocessing pipeline works correctly
- Confirm no data leakage between splits
- Move to Phase 03 for MLP baseline implementation