# Data Pipeline Guide

This document describes the data preprocessing pipeline for the SEAAD oligodendrocyte AD classification project.

## Overview

The data pipeline is implemented in three files:

1. **`src/data/loaders.py`**: Core `SEAADDataLoader` class with all preprocessing methods
2. **`scripts/explore_data.py`**: Lightweight script to explore dataset structure
3. **`scripts/prepare_data.py`**: Full preprocessing pipeline (data loading → feature processing)

## Key Parameters

### Data Locations

- **Raw data**: `/Users/duonghongduc/Downloads/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad` (35GB)
- **Processed output**: `results/processed_data/`
  - `train_data.h5ad`
  - `val_data.h5ad`
  - `test_data.h5ad`
  - `metadata.json`

### Column Names (Confirmed from Dataset)

These are the actual column names in the SEAAD dataset:

- **Cell type column**: `Class` (contains cell type information)
  - Also available: `Subclass` for more detailed cell type classification
- **Donor ID column**: `Donor ID` (unique donor identifier)
- **ADNC status column**: `ADNC` (Alzheimer's Disease Neuropathologic Change status)

**Example values**:
- ADNC: "High", "Not AD", "Low", "Intermediate"
- Class: Cell type names (includes "Oligodendrocyte")
- Donor ID: Unique donor identifiers

If you need to verify or adjust column names:
```bash
python scripts/explore_data.py
```

## Pipeline Steps

### Step 1: Data Loading
- Loads the 35GB H5AD file using AnnData's backed reading for memory efficiency
- Reports data dimensions: cells × genes

### Step 2: Metadata Exploration
- Lists all observation (cell-level) columns
- Lists all variable (gene-level) columns
- Helps identify correct column names

### Step 3: Cell Type Filtering
- Filters for Oligodendrocyte cells only
- Reports cells retained

### Step 4: Label Creation
- Binary classification: "High" AD (label=1) vs "Not AD" (label=0)
- Excludes "Low" and "Intermediate" categories
- Creates two columns: `label` (0/1) and `label_name` (readable)

### Step 5: Donor-Level Stratified Splitting
- **Why donor-level?** Prevents data leakage - cells from same donor may be highly correlated
- Default split: 70% train / 10% validation / 20% test
- Ensures each donor only appears in one split
- Stratifies by label to balance class distribution

### Step 6: Feature Processing
- **HVG Selection**: Selects 2000 most variable genes (default)
- **Normalization Check**: Verifies if data is already normalized
- **HVG Computation**: Done per split to avoid information leakage

## Running the Pipeline

### Quick Exploration (No Heavy Processing)

```bash
python scripts/explore_data.py
```

Output:
- Dataset dimensions
- All available columns
- Value distributions for key columns
- Recommended next steps

### Full Data Preparation

Using default column names (recommended):
```bash
python scripts/prepare_data.py
```

With custom parameters (if column names differ):
```bash
python scripts/prepare_data.py \
  --cell-type-col Class \
  --cell-type-filter Oligodendrocyte \
  --donor-col "Donor ID" \
  --adnc-col ADNC \
  --n-hvgs 2000
```

To use Subclass instead of Class:
```bash
python scripts/prepare_data.py --cell-type-col Subclass
```

### Expected Runtime

⚠️ **First load**: ~20-30 minutes (reading 35GB file)
- Subsequent loads from H5AD files: ~5-10 minutes each

## Output Structure

After running `prepare_data.py`, you'll have:

```
results/processed_data/
├── train_data.h5ad          # 70% of cells
├── val_data.h5ad            # 10% of cells
├── test_data.h5ad           # 20% of cells
└── metadata.json            # Summary statistics
```

### Metadata JSON Format

```json
{
  "train_samples": 12345,
  "train_genes": 2000,
  "val_samples": 1234,
  "val_genes": 2000,
  "test_samples": 2468,
  "test_genes": 2000,
  "label_mapping": {
    "Not AD": 0,
    "High": 1
  },
  "n_hvgs": 2000
}
```

## Key Classes & Methods

### SEAADDataLoader

**Initialize with correct column names**:
```python
loader = SEAADDataLoader(
    data_path,
    cell_type_column="Class",      # Column with cell types
    donor_column="Donor ID",        # Column with donor IDs
    adnc_column="ADNC",            # Column with ADNC status
)

# Load raw data
adata = loader.load_raw_data()

# Explore
metadata = loader.explore_metadata()
info = loader.get_column_info("column_name")

# Filter & process
adata = loader.filter_cell_type("Oligodendrocyte")  # Uses configured column
adata, mapping = loader.create_labels()             # Uses configured column
train, val, test = loader.stratified_split()        # Uses configured column

# Feature processing
adata = loader.select_hvgs(adata, n_hvgs=2000)
info = loader.check_normalization(adata)

# Save
loader.save_processed(adata, "output.h5ad")
```

## Troubleshooting

### Column Name Issues

If you see errors like "Column 'X' not found in observations":

1. Run `python scripts/explore_data.py`
2. Note the actual column names
3. Update your prepare_data.py call with correct column names

### Memory Issues

If you run out of memory:

1. The initial load uses `backed="r"` for reading-only access
2. Data is only fully loaded into memory when needed
3. If issues persist, consider:
   - Processing on a machine with more RAM
   - Using Google Colab with GPU (connected to Drive with data)

### Large File I/O

The 35GB file is large, but H5AD format is optimized for efficient access:

- Backed reading (`.read_h5ad(..., backed="r")`) streams data
- Metadata operations don't require full load
- Filtering operations work efficiently

## Next Steps

1. **Run exploration**: `python scripts/explore_data.py`
2. **Verify column names** against your dataset
3. **Run full preprocessing**: `python scripts/prepare_data.py`
4. **Verify output**: Check `results/processed_data/metadata.json`
5. **Continue to Phase 3**: Train MLP baseline model

---

*For questions about specific methods, see docstrings in `src/data/loaders.py`*
