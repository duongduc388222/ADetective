"""
Unit tests for data loading and preprocessing functionality.

Note: These tests require the actual SEAAD data file.
Run with: pytest tests/test_data_loaders.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import SEAADDataLoader
from src.utils.config import Config


class TestSEAADDataLoaderInit:
    """Test SEAADDataLoader initialization."""

    def test_init_with_missing_file(self):
        """Test that error is raised for missing data file."""
        with pytest.raises(FileNotFoundError):
            SEAADDataLoader("/nonexistent/path/to/data.h5ad")

    def test_init_with_valid_path(self, tmp_path):
        """Test initialization with valid path."""
        # Create a dummy h5ad file
        dummy_file = tmp_path / "dummy.h5ad"
        dummy_file.touch()

        loader = SEAADDataLoader(str(dummy_file))
        assert loader.data_path == dummy_file
        assert loader.adata is None


class TestConfig:
    """Test Config class functionality."""

    def test_config_local_env(self):
        """Test Config initialization in local environment."""
        config = Config(env="local")
        assert config.env == "local"
        assert config.project_root is not None

    def test_config_get_method(self):
        """Test Config get method with dotted keys."""
        config = Config(env="local")

        # Test existing key
        batch_size = config.get("training.batch_size")
        assert isinstance(batch_size, int)
        assert batch_size > 0

        # Test non-existing key with default
        value = config.get("nonexistent.key", default=42)
        assert value == 42

    def test_config_set_method(self):
        """Test Config set method."""
        config = Config(env="local")

        config.set("training.batch_size", 64)
        assert config.get("training.batch_size") == 64

    def test_config_ensure_dirs(self, tmp_path, monkeypatch):
        """Test that ensure_dirs creates necessary directories."""
        config = Config(env="local")

        # Override output dir to temp directory
        config.set("output.results_dir", str(tmp_path / "results"))
        config.set("output.checkpoint_dir", str(tmp_path / "checkpoints"))
        config.set("output.log_dir", str(tmp_path / "logs"))

        config.ensure_dirs()

        assert (tmp_path / "results").exists()
        assert (tmp_path / "checkpoints").exists()
        assert (tmp_path / "logs").exists()


class TestDataPreprocessing:
    """Tests for data preprocessing operations (requires actual data)."""

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return Config(env="local")

    def test_data_file_exists(self, config):
        """Test that data file exists at configured path."""
        data_path = config.get("data.path")
        assert Path(data_path).exists(), f"Data file not found at {data_path}"

    @pytest.mark.skipif(
        not Path("/Users/duonghongduc/Downloads/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad").exists(),
        reason="SEAAD data file not available",
    )
    def test_load_raw_data(self, config):
        """Test loading raw SEAAD data."""
        loader = SEAADDataLoader(config.get("data.path"))
        adata = loader.load_raw_data()

        assert adata is not None
        assert adata.n_obs > 0
        assert adata.n_vars > 0

    @pytest.mark.skipif(
        not Path("/Users/duonghongduc/Downloads/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad").exists(),
        reason="SEAAD data file not available",
    )
    def test_metadata_exploration(self, config):
        """Test metadata exploration."""
        loader = SEAADDataLoader(config.get("data.path"))
        loader.load_raw_data()

        metadata = loader.explore_metadata()
        assert "obs_columns" in metadata
        assert "var_columns" in metadata
        assert metadata["n_obs"] > 0
        assert metadata["n_vars"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
