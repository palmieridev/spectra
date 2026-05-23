"""Tests for data loading and preprocessing."""

import pytest
import numpy as np

# SpectraPreprocessor has no TensorFlow dependency — always importable.
from src.data import SpectraPreprocessor, create_dataset

# Check actual TF availability (needed for create_dataset which uses tf.data).
try:
    import tensorflow as _tf  # noqa: F401
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class TestSpectraPreprocessor:
    """Test data preprocessing pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra and labels for testing."""
        n_samples = 100
        n_wavelengths = 3000
        spectra = np.random.randn(n_samples, n_wavelengths).astype(np.float32)
        labels = {
            'temperature': np.random.uniform(3000, 10000, n_samples),
            'composition': np.random.uniform(0, 1, n_samples),
            'radial_velocity': np.random.uniform(-300, 300, n_samples)
        }
        return spectra, labels

    def test_preprocessor_fit(self, sample_data):
        """Test preprocessor fitting on training data."""
        spectra, labels = sample_data
        preprocessor = SpectraPreprocessor()
        preprocessor.fit(spectra, labels)
        assert preprocessor.is_fitted
        assert 'temperature' in preprocessor.label_scalers
        assert 'composition' in preprocessor.label_scalers
        assert 'radial_velocity' in preprocessor.label_scalers

    def test_preprocessor_transform(self, sample_data):
        """Test normalization transformation."""
        spectra, labels = sample_data
        preprocessor = SpectraPreprocessor()
        preprocessor.fit(spectra, labels)

        norm_spectra, norm_labels = preprocessor.transform(spectra, labels)

        # Check shapes preserved
        assert norm_spectra.shape == spectra.shape
        assert norm_labels['temperature'].shape == labels['temperature'].shape

        # Check values are normalized (approximate mean=0, std=1)
        assert np.abs(norm_spectra.mean()) < 0.1
        assert np.abs(norm_spectra.std() - 1.0) < 0.1

    def test_preprocessor_fit_transform(self, sample_data):
        """Test fit and transform in one call."""
        spectra, labels = sample_data
        preprocessor = SpectraPreprocessor()
        norm_spectra, norm_labels = preprocessor.fit_transform(spectra, labels)

        assert norm_spectra.shape == spectra.shape
        assert norm_labels is not None

    def test_preprocessor_transform_without_fit_raises(self, sample_data):
        """Test that transform without fit raises error."""
        spectra, labels = sample_data
        preprocessor = SpectraPreprocessor()

        with pytest.raises(RuntimeError):
            preprocessor.transform(spectra, labels)

    def test_invalid_spectra_shape_raises(self, sample_data):
        """Test that invalid input shape raises error."""
        spectra, labels = sample_data
        preprocessor = SpectraPreprocessor()

        # 1D array instead of 2D
        with pytest.raises(ValueError):
            preprocessor.fit(spectra.flatten(), labels)


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
class TestDatasetCreation:
    """Test TensorFlow dataset creation."""

    @pytest.fixture
    def normalized_data(self):
        """Create normalized data for dataset tests."""
        n_samples = 50
        n_wavelengths = 3000
        spectra = np.random.randn(n_samples, n_wavelengths).astype(np.float32)
        labels = {
            'temperature': np.random.uniform(3000, 10000, n_samples),
            'composition': np.random.uniform(0, 1, n_samples),
            'radial_velocity': np.random.uniform(-300, 300, n_samples)
        }
        return spectra, labels

    def test_dataset_creation(self, normalized_data):
        """Test dataset creation with proper batching."""
        spectra, labels = normalized_data
        dataset, label_keys = create_dataset(spectra, labels, batch_size=32)

        assert label_keys == ['composition', 'radial_velocity', 'temperature']

        # Verify first batch
        for batch_x, batch_y in dataset.take(1):
            assert batch_x.shape[0] <= 32
            assert batch_x.shape[1] == spectra.shape[1]
            assert len(batch_y) == 3

    def test_dataset_prefetch(self, normalized_data):
        """Test that dataset has proper prefetch."""
        spectra, labels = normalized_data
        dataset, _ = create_dataset(spectra, labels, batch_size=32)

        # Should be able to iterate without errors
        count = 0
        for _ in dataset:
            count += 1
        assert count > 0
