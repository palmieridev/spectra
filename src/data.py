"""Data loading and preprocessing for stellar spectra."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class SpectraPreprocessor:
    """Preprocessor for stellar spectroscopic data."""

    def __init__(self):
        self.flux_scaler = StandardScaler()
        self.label_scalers = {}
        self.is_fitted = False

    def fit(self, spectra: np.ndarray, labels: dict):
        """Fit preprocessing scalers on training data.

        Args:
            spectra: Array of shape (n_samples, n_wavelengths) with flux values
            labels: Dict of label arrays with keys: 'temperature', 'composition', 'radial_velocity'
        """
        if spectra.ndim != 2:
            raise ValueError(f"Expected 2D spectra, got shape {spectra.shape}")

        # Fit flux scaler
        self.flux_scaler.fit(spectra)

        # Fit individual scalers for each label
        for label_name, label_values in labels.items():
            scaler = StandardScaler()
            scaler.fit(label_values.reshape(-1, 1))
            self.label_scalers[label_name] = scaler

        self.is_fitted = True
        return self

    def transform(self, spectra: np.ndarray, labels: Optional[dict] = None) -> Tuple:
        """Transform spectra and labels using fitted scalers.

        Args:
            spectra: Array of shape (n_samples, n_wavelengths)
            labels: Optional dict of label arrays

        Returns:
            Tuple of (normalized_spectra, normalized_labels_dict or None)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform()")

        # Normalize spectra
        normalized_spectra = self.flux_scaler.transform(spectra)

        # Normalize labels if provided
        normalized_labels = None
        if labels is not None:
            normalized_labels = {}
            for label_name, label_values in labels.items():
                if label_name in self.label_scalers:
                    normalized = self.label_scalers[label_name].transform(
                        label_values.reshape(-1, 1)
                    ).flatten()
                    normalized_labels[label_name] = normalized

        return normalized_spectra, normalized_labels

    def fit_transform(self, spectra: np.ndarray, labels: dict) -> Tuple:
        """Fit and transform in one step."""
        self.fit(spectra, labels)
        return self.transform(spectra, labels)


def load_sdss_spectra(
    cache_dir: str = 'data/raw',
    n_stars: int = 1000,
    min_snr: float = 10.0,
) -> Tuple[np.ndarray, dict]:
    """Load SDSS stellar spectra and SSPP labels for training.

    Delegates to :func:`src.sdss_loader.load_sdss_spectra`, which queries
    the SDSS CAS for SEGUE/SSPP stars, downloads their BOSS spectra, and
    runs the full spectral preprocessing pipeline.

    Args:
        cache_dir: Local directory for raw data / astroquery cache.
        n_stars:   Maximum number of spectra to download.
        min_snr:   Minimum median S/N per pixel (SSPP quality cut).

    Returns:
        Tuple of (spectra array of shape (n, n_wavelengths), labels dict).
        Labels dict has keys: ``'temperature'`` (Teff K),
        ``'composition'`` ([Fe/H] dex), ``'radial_velocity'`` (km/s).
    """
    from src.sdss_loader import load_sdss_spectra as _load
    return _load(cache_dir=cache_dir, n_stars=n_stars, min_snr=min_snr)


def create_dataset(spectra: np.ndarray, labels: dict, batch_size: int = 32) -> tuple:
    """Create TensorFlow dataset for training.

    Args:
        spectra: Normalized spectra array (n_samples, n_wavelengths)
        labels: Dict of normalized label arrays
        batch_size: Batch size for training

    Returns:
        tf.data.Dataset with (spectra, labels_tuple) structure
    """
    import tensorflow as tf

    # Convert labels dict to tuple for model compatibility
    label_keys = sorted(labels.keys())
    label_values = tuple(labels[k] for k in label_keys)

    dataset = tf.data.Dataset.from_tensor_slices((spectra, label_values))
    dataset = dataset.shuffle(len(spectra))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, label_keys
