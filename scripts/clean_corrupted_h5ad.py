#!/usr/bin/env python3
"""
Clean Corrupted H5AD Files via Streaming Extraction.

This script handles corrupted h5ad files by:
1. Reading data in chunks to identify corrupted rows
2. Streaming only valid rows to output file
3. Ensuring alignment across X, obs, obsm, obsp, and layers
4. Generating cleaning report with statistics

Usage:
    python scripts/clean_corrupted_h5ad.py \
        --input-path /path/to/corrupted.h5ad \
        --output-path /path/to/cleaned.h5ad \
        --chunk-size 10000
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Set, Tuple, Optional

import h5py
import numpy as np

# Try importing read/write helpers
try:
    from anndata.io import read_elem, write_elem
except ImportError:
    from anndata.experimental import read_elem, write_elem

# Optional: scanpy for verification
try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean corrupted H5AD files via streaming extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/clean_corrupted_h5ad.py \\
    --input-path ./data/corrupted.h5ad \\
    --output-path ./data/cleaned.h5ad

  # Custom chunk size for memory optimization
  python scripts/clean_corrupted_h5ad.py \\
    --input-path ./data/corrupted.h5ad \\
    --output-path ./data/cleaned.h5ad \\
    --chunk-size 5000

  # Skip verification (faster)
  python scripts/clean_corrupted_h5ad.py \\
    --input-path ./data/corrupted.h5ad \\
    --output-path ./data/cleaned.h5ad \\
    --skip-verification

  # Filter for specific cell type (default: Oligodendrocyte)
  python scripts/clean_corrupted_h5ad.py \\
    --input-path ./data/corrupted.h5ad \\
    --output-path ./data/cleaned.h5ad \\
    --filter-cell-type "Oligodendrocyte"

  # Filter using different cell type column
  python scripts/clean_corrupted_h5ad.py \\
    --input-path ./data/corrupted.h5ad \\
    --output-path ./data/cleaned.h5ad \\
    --filter-cell-type "Astrocyte" \\
    --cell-type-column "Subclass"
        """,
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the corrupted H5AD file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path for the cleaned output H5AD file",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of rows to process per chunk (default: 10000)",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip final verification with scanpy",
    )
    parser.add_argument(
        "--filter-cell-type",
        type=str,
        default="Oligodendrocyte",
        help="Filter cells by cell type (default: Oligodendrocyte). Use 'none' to disable filtering.",
    )
    parser.add_argument(
        "--cell-type-column",
        type=str,
        default="Subclass",
        help="Column name in obs containing cell type labels (default: Subclass)",
    )

    return parser.parse_args()


def process_sparse_matrix(
    src_grp: h5py.Group,
    dst_grp: h5py.Group,
    valid_indices: Set[int],
    n_total: int,
    n_vars: int,
    chunk_size: int,
    matrix_name: str = "X",
) -> Tuple[Set[int], int]:
    """
    Process a sparse CSR matrix, extracting only valid rows.

    Args:
        src_grp: Source HDF5 group containing the sparse matrix
        dst_grp: Destination HDF5 group to write to
        valid_indices: Set of row indices to include
        n_total: Total number of rows in source
        n_vars: Number of columns (genes)
        chunk_size: Rows per chunk
        matrix_name: Name for logging

    Returns:
        Tuple of (updated valid_indices, number of errors)
    """
    logger.info(f"Processing '{matrix_name}' (Sparse Matrix)...")

    # Create resizable datasets
    dset_data = dst_grp.create_dataset(
        'data', shape=(0,), maxshape=(None,), dtype='float32', compression='gzip'
    )
    dset_indices = dst_grp.create_dataset(
        'indices', shape=(0,), maxshape=(None,), dtype='int64', compression='gzip'
    )
    dset_indptr = dst_grp.create_dataset(
        'indptr', shape=(1,), maxshape=(None,), dtype='int64', compression='gzip'
    )
    dset_indptr[0] = 0

    # Source data
    src_indptr = src_grp['indptr'][:]
    src_data = src_grp['data']
    src_indices = src_grp['indices']

    current_ptr = 0
    local_valid_indices = set()
    errors = 0

    for i in range(0, n_total, chunk_size):
        start = i
        end = min(i + chunk_size, n_total)

        # Only process rows that are in valid_indices
        chunk_target_indices = [idx for idx in range(start, end) if idx in valid_indices]
        if not chunk_target_indices:
            continue

        p_start = src_indptr[start]
        p_end = src_indptr[end]

        chunk_write_data = None
        chunk_write_indices = None
        chunk_write_lens = None

        try:
            # Optimistic read of entire chunk
            chunk_data = src_data[p_start:p_end]
            chunk_indices = src_indices[p_start:p_end]
            chunk_indptr = src_indptr[start:end+1]

            # Extract only the rows we want
            write_data = []
            write_indices = []
            write_lens = []

            for idx in chunk_target_indices:
                local_row = idx - start
                row_start = chunk_indptr[local_row] - p_start
                row_end = chunk_indptr[local_row + 1] - p_start
                write_data.append(chunk_data[row_start:row_end])
                write_indices.append(chunk_indices[row_start:row_end])
                write_lens.append(row_end - row_start)
                local_valid_indices.add(idx)

            if write_data:
                chunk_write_data = np.concatenate(write_data) if write_data else np.array([], dtype='float32')
                chunk_write_indices = np.concatenate(write_indices) if write_indices else np.array([], dtype='int64')
                chunk_write_lens = np.array(write_lens)

        except OSError:
            # Fallback: Row-by-row recovery
            logger.warning(f"  OSError in '{matrix_name}' for chunk {start}-{end}. Attempting row-by-row recovery.")
            write_data = []
            write_indices = []
            write_lens = []

            for idx in chunk_target_indices:
                r_p_start = src_indptr[idx]
                r_p_end = src_indptr[idx + 1]
                try:
                    r_data = src_data[r_p_start:r_p_end]
                    r_indices = src_indices[r_p_start:r_p_end]
                    write_data.append(r_data)
                    write_indices.append(r_indices)
                    write_lens.append(len(r_data))
                    local_valid_indices.add(idx)
                except OSError:
                    errors += 1
                    logger.warning(f"    Failed to read row {idx} in '{matrix_name}'. Skipping.")

            if write_data:
                chunk_write_data = np.concatenate(write_data)
                chunk_write_indices = np.concatenate(write_indices)
                chunk_write_lens = np.array(write_lens)

        # Write chunk to output
        if chunk_write_data is not None and len(chunk_write_data) > 0:
            n_new_items = len(chunk_write_data)
            n_new_rows = len(chunk_write_lens)

            dset_data.resize(dset_data.shape[0] + n_new_items, axis=0)
            dset_data[-n_new_items:] = chunk_write_data

            dset_indices.resize(dset_indices.shape[0] + n_new_items, axis=0)
            dset_indices[-n_new_items:] = chunk_write_indices

            new_ptrs = current_ptr + np.cumsum(chunk_write_lens)
            dset_indptr.resize(dset_indptr.shape[0] + n_new_rows, axis=0)
            dset_indptr[-n_new_rows:] = new_ptrs

            current_ptr = new_ptrs[-1]

        # Progress
        if end % 50000 == 0 or end == n_total:
            print(f"  Processed {matrix_name}: {end}/{n_total} (Errors: {errors})", end='\r')

    print()  # Newline after progress

    # Finalize attributes
    dst_grp.attrs["encoding-type"] = "csr_matrix"
    dst_grp.attrs["encoding-version"] = "0.1.0"
    dst_grp.attrs["shape"] = np.array([len(local_valid_indices), n_vars], dtype=np.int64)

    logger.info(f"  {matrix_name} done. Valid rows: {len(local_valid_indices)}, Errors: {errors}")

    return local_valid_indices, errors


def clean_h5ad_file(
    input_path: str,
    output_path: str,
    chunk_size: int = 10000,
    skip_verification: bool = False,
    filter_cell_type: Optional[str] = "Oligodendrocyte",
    cell_type_column: str = "Subclass",
) -> dict:
    """
    Clean a corrupted H5AD file by streaming extraction.

    Args:
        input_path: Path to corrupted input file
        output_path: Path for cleaned output file
        chunk_size: Rows per chunk for processing
        skip_verification: Skip final verification
        filter_cell_type: Cell type to filter for (e.g., "Oligodendrocyte"). Use None or "none" to disable.
        cell_type_column: Column name in obs containing cell type labels

    Returns:
        Dictionary with cleaning statistics
    """
    # Normalize filter_cell_type
    if filter_cell_type and filter_cell_type.lower() == "none":
        filter_cell_type = None

    logger.info("=" * 80)
    logger.info("H5AD FILE CLEANING - STREAMING EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Chunk size: {chunk_size:,}")
    if filter_cell_type:
        logger.info(f"Cell type filter: '{filter_cell_type}' (column: '{cell_type_column}')")
    else:
        logger.info("Cell type filter: DISABLED (processing all cells)")

    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "chunk_size": chunk_size,
        "filter_cell_type": filter_cell_type,
        "cell_type_column": cell_type_column,
        "n_total_rows": 0,
        "n_filtered_rows": 0,
        "n_valid_rows": 0,
        "n_dropped_rows": 0,
        "errors_by_matrix": {},
    }

    with h5py.File(input_path, 'r') as f_src, h5py.File(output_path, 'w') as f_dst:

        # --- A. Copy variable-aligned and unstructured metadata ---
        logger.info("\nStep 1: Copying 'var', 'uns', 'varm', 'varp'...")
        for key in ['var', 'uns', 'varm', 'varp']:
            if key in f_src:
                logger.info(f"  - Copying {key}")
                try:
                    f_src.copy(key, f_dst)
                except Exception as e:
                    logger.warning(f"    Direct copy of {key} failed ({e}). Trying read/write...")
                    try:
                        data = read_elem(f_src[key])
                        write_elem(f_dst, key, data)
                    except Exception as e2:
                        logger.error(f"    Could not copy {key}: {e2}")

        # Get dimensions
        src_grp_x = f_src['X']
        src_indptr = src_grp_x['indptr'][:]
        n_total = len(src_indptr) - 1
        stats["n_total_rows"] = n_total

        # Try to get n_vars
        if 'shape' in f_src['var'].attrs:
            n_vars = f_src['var'].attrs['shape'][0]
        else:
            try:
                n_vars = read_elem(f_src['var']).shape[0]
            except Exception:
                n_vars = len(f_src['var/_index'])

        logger.info(f"\nDataset dimensions: {n_total:,} cells x {n_vars:,} genes")

        # --- B. Cell type filtering (if enabled) ---
        if filter_cell_type:
            logger.info(f"\nStep 1.5: Filtering by cell type '{filter_cell_type}'...")
            try:
                obs_data = read_elem(f_src['obs'])
                if cell_type_column not in obs_data.columns:
                    available_cols = list(obs_data.columns)
                    logger.error(f"  Column '{cell_type_column}' not found in obs!")
                    logger.error(f"  Available columns: {available_cols}")
                    raise ValueError(f"Cell type column '{cell_type_column}' not found")

                # Find matching cells (case-insensitive partial match)
                cell_types = obs_data[cell_type_column].astype(str)
                mask = cell_types.str.contains(filter_cell_type, case=False, na=False)
                filtered_indices = set(np.where(mask)[0])

                stats["n_filtered_rows"] = len(filtered_indices)
                logger.info(f"  Found {len(filtered_indices):,} cells matching '{filter_cell_type}'")
                logger.info(f"  (out of {n_total:,} total cells)")

                if len(filtered_indices) == 0:
                    # Show available cell types
                    unique_types = cell_types.unique()[:20]  # Show first 20
                    logger.error(f"  No cells found matching '{filter_cell_type}'!")
                    logger.error(f"  Available cell types (first 20): {list(unique_types)}")
                    raise ValueError(f"No cells found matching '{filter_cell_type}'")

                # Start with filtered indices
                global_valid_indices = filtered_indices
                del obs_data  # Free memory

            except Exception as e:
                if "not found" in str(e) or "No cells found" in str(e):
                    raise
                logger.error(f"  Error during cell type filtering: {e}")
                raise
        else:
            # Start with all rows as potentially valid
            global_valid_indices = set(range(n_total))
            stats["n_filtered_rows"] = n_total

        # --- C. Process X matrix (first pass to identify valid rows) ---
        logger.info("\nStep 2: Processing X matrix...")
        grp_x = f_dst.create_group('X')

        x_valid_indices, x_errors = process_sparse_matrix(
            src_grp=src_grp_x,
            dst_grp=grp_x,
            valid_indices=global_valid_indices,
            n_total=n_total,
            n_vars=n_vars,
            chunk_size=chunk_size,
            matrix_name="X",
        )
        stats["errors_by_matrix"]["X"] = x_errors

        # Update global valid indices based on X
        global_valid_indices = x_valid_indices.copy()
        logger.info(f"  Valid rows after X: {len(global_valid_indices):,}")

        # --- C. Process layers (second pass, may further reduce valid indices) ---
        if 'layers' in f_src:
            logger.info("\nStep 3: Processing layers...")
            dst_layers = f_dst.create_group('layers')
            src_layers = f_src['layers']

            for key in src_layers.keys():
                logger.info(f"  - Layer: {key}")
                obj = src_layers[key]
                is_sparse = isinstance(obj, h5py.Group) and 'indptr' in obj

                if is_sparse:
                    # Sparse layer
                    l_dst = dst_layers.create_group(key)
                    layer_valid_indices, layer_errors = process_sparse_matrix(
                        src_grp=obj,
                        dst_grp=l_dst,
                        valid_indices=global_valid_indices,
                        n_total=n_total,
                        n_vars=n_vars,
                        chunk_size=chunk_size,
                        matrix_name=f"layers/{key}",
                    )
                    stats["errors_by_matrix"][f"layers/{key}"] = layer_errors

                    # Update global valid indices (intersection)
                    global_valid_indices = global_valid_indices & layer_valid_indices
                    logger.info(f"    Valid rows after {key}: {len(global_valid_indices):,}")

                else:
                    # Dense layer - subset and copy
                    logger.info(f"    (Dense layer, slicing...)")
                    l_src = obj
                    valid_idx_list = sorted(global_valid_indices)

                    l_dst_dataset = dst_layers.create_dataset(
                        key,
                        shape=(len(valid_idx_list),) + l_src.shape[1:],
                        dtype=l_src.dtype,
                        compression='gzip'
                    )

                    # Chunked read/write for dense
                    current_row = 0
                    for i in range(0, n_total, chunk_size):
                        start = i
                        end = min(i + chunk_size, n_total)

                        chunk_indices = [idx for idx in valid_idx_list if start <= idx < end]
                        if not chunk_indices:
                            continue

                        try:
                            chunk_data = l_src[start:end]
                            local_indices = [idx - start for idx in chunk_indices]
                            data_subset = chunk_data[local_indices]

                            l_dst_dataset[current_row:current_row + len(data_subset)] = data_subset
                            current_row += len(data_subset)
                        except OSError as e:
                            logger.warning(f"    OSError reading dense layer chunk {start}-{end}: {e}")
        else:
            logger.info("\nStep 3: No layers found, skipping...")

        # --- D. Now we have final global_valid_indices. Rebuild X if needed ---
        # If layers removed additional rows, we need to rebuild X with only those rows
        if len(global_valid_indices) < len(x_valid_indices):
            logger.info(f"\nStep 4: Rebuilding X (layers removed {len(x_valid_indices) - len(global_valid_indices)} additional rows)...")
            del f_dst['X']
            grp_x = f_dst.create_group('X')

            x_final_indices, x_final_errors = process_sparse_matrix(
                src_grp=src_grp_x,
                dst_grp=grp_x,
                valid_indices=global_valid_indices,
                n_total=n_total,
                n_vars=n_vars,
                chunk_size=chunk_size,
                matrix_name="X (rebuild)",
            )
            global_valid_indices = x_final_indices
        else:
            logger.info("\nStep 4: X rebuild not needed (all X rows valid in layers)")

        # Convert to sorted list for subsetting obs/obsm/obsp
        successful_indices = np.array(sorted(global_valid_indices))
        stats["n_valid_rows"] = len(successful_indices)
        stats["n_dropped_rows"] = n_total - len(successful_indices)

        logger.info(f"\nFinal valid rows: {len(successful_indices):,} / {n_total:,}")
        logger.info(f"Dropped rows: {stats['n_dropped_rows']:,}")

        # --- E. Process 'obs' (subset to valid rows) ---
        logger.info("\nStep 5: Processing 'obs'...")
        obs = read_elem(f_src['obs'])
        obs_subset = obs.iloc[successful_indices]
        write_elem(f_dst, 'obs', obs_subset)
        logger.info(f"  obs: {len(obs_subset):,} rows")

        # --- F. Process 'obsm' (subset) ---
        if 'obsm' in f_src:
            logger.info("\nStep 6: Processing 'obsm'...")
            try:
                obsm_data = read_elem(f_src['obsm'])
                if isinstance(obsm_data, dict) or hasattr(obsm_data, 'keys'):
                    f_dst.create_group('obsm')
                    for k, v in obsm_data.items():
                        if hasattr(v, 'shape') and v.shape[0] == n_total:
                            v_subset = v[successful_indices]
                            write_elem(f_dst, f'obsm/{k}', v_subset)
                            logger.info(f"  - {k}: {v_subset.shape}")
                        else:
                            logger.warning(f"  - Skipping obsm/{k}: shape mismatch")
            except Exception as e:
                logger.error(f"  Error processing obsm: {e}")
        else:
            logger.info("\nStep 6: No obsm found, skipping...")

        # --- G. Process 'obsp' (subset both dimensions) ---
        if 'obsp' in f_src:
            logger.info("\nStep 7: Processing 'obsp'...")
            try:
                obsp_data = read_elem(f_src['obsp'])
                f_dst.create_group('obsp')
                for k, v in obsp_data.items():
                    if hasattr(v, 'shape') and v.shape[0] == n_total and v.shape[1] == n_total:
                        v_subset = v[successful_indices, :][:, successful_indices]
                        write_elem(f_dst, f'obsp/{k}', v_subset)
                        logger.info(f"  - {k}: {v_subset.shape}")
                    else:
                        logger.warning(f"  - Skipping obsp/{k}: shape mismatch")
            except Exception as e:
                logger.error(f"  Error processing obsp: {e}")
        else:
            logger.info("\nStep 7: No obsp found, skipping...")

    # Verification
    logger.info("\n" + "=" * 80)
    logger.info("CLEANING COMPLETE")
    logger.info("=" * 80)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        stats["output_size_mb"] = size_mb
        logger.info(f"Output file created: {size_mb:.2f} MB")

        if not skip_verification and HAS_SCANPY:
            logger.info("\nVerifying with scanpy...")
            try:
                adata_check = sc.read_h5ad(output_path, backed='r')
                logger.info(f"  Verification SUCCESS: {adata_check}")
                stats["verification"] = "SUCCESS"
            except Exception as e:
                logger.error(f"  Verification FAILED: {e}")
                stats["verification"] = f"FAILED: {e}"
        elif skip_verification:
            logger.info("Verification skipped (--skip-verification)")
            stats["verification"] = "SKIPPED"
        else:
            logger.warning("Verification skipped (scanpy not installed)")
            stats["verification"] = "SKIPPED (no scanpy)"

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total rows:      {stats['n_total_rows']:,}")
    if filter_cell_type:
        logger.info(f"Filtered rows:   {stats['n_filtered_rows']:,} (matching '{filter_cell_type}')")
    logger.info(f"Valid rows:      {stats['n_valid_rows']:,}")
    logger.info(f"Dropped rows:    {stats['n_dropped_rows']:,}")
    logger.info(f"Errors by matrix:")
    for matrix, errors in stats['errors_by_matrix'].items():
        logger.info(f"  - {matrix}: {errors} errors")

    return stats


def main():
    """Main entry point."""
    args = parse_arguments()

    # Validate input
    if not os.path.exists(args.input_path):
        logger.error(f"Input file not found: {args.input_path}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run cleaning
    stats = clean_h5ad_file(
        input_path=args.input_path,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
        skip_verification=args.skip_verification,
        filter_cell_type=args.filter_cell_type,
        cell_type_column=args.cell_type_column,
    )

    # Exit code based on success
    if stats["n_valid_rows"] > 0:
        logger.info("\nCleaning completed successfully!")
        sys.exit(0)
    else:
        logger.error("\nCleaning failed - no valid rows!")
        sys.exit(1)


if __name__ == "__main__":
    main()
