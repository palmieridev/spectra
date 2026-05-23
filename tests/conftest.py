"""Pytest configuration and fixtures."""

import os
import pytest

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


@pytest.fixture(scope='session', autouse=True)
def setup_tf():
    """Configure TensorFlow for testing."""
    if TENSORFLOW_AVAILABLE:
        # Disable GPU for tests (use CPU for consistency)
        tf.config.set_visible_devices([], 'GPU')
        # Set random seeds for reproducibility
        tf.random.set_seed(42)


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create temporary directory for model artifacts."""
    return str(tmp_path / 'models')
