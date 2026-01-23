"""
Metadata Preprocessing Module

Handles extraction, encoding, and normalization of donor metadata from AnnData objects.
Ensures no data leakage by fitting encoders on training data only.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MetadataProcessor:
    """
    Processes donor metadata with proper data leakage prevention.

    Features:
    - Fits encoders on training data only
    - Handles missing values gracefully
    - One-hot encodes categorical variables
    - Z-score normalizes continuous variables
    - Saves/loads fitted state for reproducibility
    """

    def __init__(self):
        self.categorical_features = None
        self.continuous_features = None
        self.encoders = {}
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, train_obs: pd.DataFrame) -> "MetadataProcessor":
        """
        Fit encoders on training data.

        Args:
            train_obs: Training data metadata (from .h5ad.obs)

        Returns:
            self
        """
        logger.info("Fitting metadata processor on training data...")

        # Define feature groups based on actual h5ad column names
        #
        # IMPORTANT: Avoid data leakage by NOT using features that directly
        # measure or encode the ADNC label (Alzheimer's Disease pathology).
        #
        # REMOVED (cause data leakage):
        # - "Cognitive Status" - directly related to AD diagnosis
        # - "Amyloid (6E10+)" - directly measures amyloid plaques (AD pathology)
        # - "Tau (AT8+)" - directly measures tau tangles (AD pathology)
        #
        # KEPT (acceptable predictors known before disease):
        # - APOE Genotype - genetic risk factor (determined at birth)
        # - Sex - biological variable
        # - Age at Death - demographic
        # - Years of Education - demographic (potential protective factor)
        # - PMI - quality control metric (not related to AD)
        # - GFAP, Alpha-Synuclein - borderline but kept for research purposes
        #
        self.categorical_features = {
            "APOE Genotype": "APOE Genotype",  # Genetic risk factor for AD
            "Sex": "Sex",  # Sex-specific AD patterns
            # REMOVED: "Cognitive Status" - causes data leakage
        }

        self.continuous_features = {
            "Age at Death": "Age at Death",
            "Years of Education": "Years of education",
            # REMOVED: "Amyloid (6E10+)" - directly measures AD pathology (data leakage)
            # REMOVED: "Tau (AT8+)" - directly measures AD pathology (data leakage)
            "GFAP (Astrocytes)": "percent GFAP positive area",  # Gliosis marker
            "Alpha-Synuclein": "percent aSyn positive area",  # Lewy body marker
            "PMI (QC)": "PMI",  # Quality control metric
        }

        # Fit categorical encoders (one-hot)
        encoded_frames = []
        for feature_name, col_name in self.categorical_features.items():
            if col_name in train_obs.columns:
                # Get unique values and filter by frequency
                values = train_obs[col_name].dropna()
                if len(values) > 0:
                    # One-hot encode
                    one_hot = pd.get_dummies(values, prefix=feature_name, drop_first=False)
                    self.encoders[col_name] = one_hot.columns.tolist()
                    encoded_frames.append(one_hot)
                    logger.info(f"  {feature_name}: {len(one_hot.columns)} features (categories: {one_hot.columns.tolist()})")
            else:
                logger.warning(f"  {feature_name} ({col_name}): NOT FOUND in metadata")

        # Fit continuous scaler (z-score)
        continuous_data = []
        for feature_name, col_name in self.continuous_features.items():
            if col_name in train_obs.columns:
                values = train_obs[col_name].dropna().values.reshape(-1, 1)
                if len(values) > 0:
                    continuous_data.append(values)
                    logger.info(f"  {feature_name}: range [{values.min():.2f}, {values.max():.2f}]")
            else:
                logger.warning(f"  {feature_name} ({col_name}): NOT FOUND in metadata")

        # Stack and fit scaler (stack columns, not rows)
        if continuous_data:
            all_continuous = np.hstack(continuous_data)
            self.scaler = StandardScaler()
            self.scaler.fit(all_continuous)

        # Build feature names list
        feature_names = []

        # Categorical feature names
        for col_name in self.categorical_features.values():
            if col_name in self.encoders:
                feature_names.extend(self.encoders[col_name])

        # Continuous feature names
        for feature_name, col_name in self.continuous_features.items():
            if col_name in train_obs.columns:
                feature_names.append(feature_name)

        self.feature_names = feature_names
        self.is_fitted = True

        logger.info(f"\nMetadata Processor Fit Complete:")
        logger.info(f"  Categorical features: {sum(len(v) for v in self.encoders.values())}")
        logger.info(f"  Continuous features: {len(self.continuous_features)}")
        logger.info(f"  Total metadata features: {len(self.feature_names)}")

        return self

    def transform(self, obs: pd.DataFrame) -> np.ndarray:
        """
        Apply learned transformations to metadata.

        Args:
            obs: Metadata dataframe (from .h5ad.obs)

        Returns:
            Processed metadata array (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("MetadataProcessor must be fitted before transform")

        encoded_frames = []

        # Encode categorical features
        for col_name, categories in self.encoders.items():
            if col_name in obs.columns:
                # One-hot encode with fitted categories
                one_hot = pd.get_dummies(obs[col_name], prefix=col_name, drop_first=False)

                # Add missing categories as zeros
                for cat in categories:
                    if cat not in one_hot.columns:
                        one_hot[cat] = 0

                # Reorder to match fitted order
                one_hot = one_hot[categories]
                encoded_frames.append(one_hot.values)

        # Normalize continuous features
        continuous_arrays = []
        for col_name in self.continuous_features.values():
            if col_name in obs.columns:
                values = obs[col_name].fillna(obs[col_name].mean()).values.reshape(-1, 1)
                continuous_arrays.append(values)

        # Transform with fitted scaler
        if continuous_arrays and self.scaler is not None:
            all_continuous = np.hstack(continuous_arrays)
            normalized = self.scaler.transform(all_continuous)
            # Stack all features
            if encoded_frames:
                metadata = np.hstack(encoded_frames + [normalized])
            else:
                metadata = normalized
        elif encoded_frames:
            metadata = np.hstack(encoded_frames)
        else:
            # No features found, return empty array
            metadata = np.zeros((len(obs), 0), dtype=np.float32)

        return metadata.astype(np.float32)

    def fit_transform(self, train_obs: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            train_obs: Training metadata

        Returns:
            Processed metadata array
        """
        self.fit(train_obs)
        return self.transform(train_obs)

    def save(self, path: Path) -> None:
        """
        Save fitted processor state.

        Args:
            path: Path to save pickle file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted processor")

        state = {
            'categorical_features': self.categorical_features,
            'continuous_features': self.continuous_features,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved metadata processor to {path}")

    def load(self, path: Path) -> "MetadataProcessor":
        """
        Load fitted processor state.

        Args:
            path: Path to saved pickle file

        Returns:
            self
        """
        path = Path(path)

        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.categorical_features = state['categorical_features']
        self.continuous_features = state['continuous_features']
        self.encoders = state['encoders']
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.is_fitted = True

        logger.info(f"Loaded metadata processor from {path}")
        return self

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        if not self.is_fitted:
            raise ValueError("Processor not fitted")
        return self.feature_names.copy()

    def get_metadata_dim(self) -> int:
        """Return total number of metadata features."""
        if not self.is_fitted:
            raise ValueError("Processor not fitted")
        return len(self.feature_names)
