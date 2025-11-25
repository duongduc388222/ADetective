import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration management for both local and Colab environments."""

    def __init__(self, config_path: Optional[str] = None, env: str = "local"):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file
            env: Environment type ('local' or 'colab')
        """
        self.env = env
        self.is_colab = self._check_colab()
        self.project_root = self._get_project_root()
        self.config_dict = {}

        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
        else:
            self._set_defaults()

    def _check_colab(self) -> bool:
        """Check if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False

    def _get_project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    def _set_defaults(self) -> None:
        """Set default configuration."""
        if self.is_colab:
            self._set_colab_defaults()
        else:
            self._set_local_defaults()

    def _set_local_defaults(self) -> None:
        """Set defaults for local development."""
        self.config_dict = {
            "environment": "local",
            "data": {
                "path": "/Users/duonghongduc/GrinnellCollege/MLAI/Data/SEAAD_A9_RNAseq_DREAM_Cleaned.h5ad",
                "cache_dir": self.project_root / "data",
            },
            "output": {
                "results_dir": self.project_root / "results",
                "checkpoint_dir": self.project_root / "results" / "checkpoints",
                "log_dir": self.project_root / "results" / "logs",
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-3,
                "epochs": 50,
                "device": "cuda",
                "mixed_precision": "bf16",
            },
            "preprocessing": {
                "n_hvgs": 2000,
                "split_ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
            },
        }

    def _set_colab_defaults(self) -> None:
        """Set defaults for Google Colab environment."""
        self.config_dict = {
            "environment": "colab",
            "data": {
                "path": "/content/drive/MyDrive/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad",
                "cache_dir": "/content/drive/MyDrive/adetective_cache",
            },
            "output": {
                "results_dir": "/content/drive/MyDrive/adetective_results",
                "checkpoint_dir": "/content/drive/MyDrive/adetective_results/checkpoints",
                "log_dir": "/content/drive/MyDrive/adetective_results/logs",
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 1e-3,
                "epochs": 50,
                "device": "cuda",
                "mixed_precision": "fp16",
            },
            "preprocessing": {
                "n_hvgs": 2000,
                "split_ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
            },
        }

    def load_from_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            self.config_dict = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")

    def save_to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(self.config_dict, f, default_flow_style=False)
        logger.info(f"Saved config to {config_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dotted key (e.g., 'training.batch_size')."""
        keys = key.split(".")
        value = self.config_dict
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dotted key."""
        keys = key.split(".")
        target = self.config_dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary."""
        return self.config_dict

    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        dirs = [
            self.get("output.results_dir"),
            self.get("output.checkpoint_dir"),
            self.get("output.log_dir"),
            self.get("data.cache_dir"),
        ]
        for dir_path in dirs:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")

    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config(env={self.env}, is_colab={self.is_colab})"
