"""Tests for neural network models."""

import pytest
import numpy as np

try:
    import tensorflow as tf
    from src.models import (
        create_spectra_model,
        compile_model,
        SpectraAnalyzer,
        create_conv1d_spectra_model,
    )
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TENSORFLOW_AVAILABLE,
    reason="TensorFlow not installed"
)


class TestSpectraModel:
    """Test neural network model architecture."""

    @pytest.fixture
    def input_shape(self):
        """Standard spectral dimension."""
        return 3000

    def test_model_creation(self, input_shape):
        """Test that model is created with correct architecture."""
        model = create_spectra_model(input_shape)
        assert model is not None
        assert len(model.outputs) == 3
        assert model.output_names == ['temperature', 'composition', 'radial_velocity']

    def test_model_input_shape(self, input_shape):
        """Test model accepts correct input shape."""
        model = create_spectra_model(input_shape)
        assert model.input_shape == (None, input_shape)

    def test_model_output_shapes(self, input_shape):
        """Test model outputs correct shapes."""
        model = create_spectra_model(input_shape)
        batch_size = 10
        inputs = np.random.randn(batch_size, input_shape).astype(np.float32)

        outputs = model(inputs, training=False)

        assert outputs[0].shape == (batch_size, 1)  # temperature
        assert outputs[1].shape == (batch_size, 1)  # composition
        assert outputs[2].shape == (batch_size, 1)  # radial_velocity

    def test_model_compilation(self, input_shape):
        """Test model compiles successfully."""
        model = create_spectra_model(input_shape)
        compile_model(model)
        assert model.optimizer is not None
        assert len(model.loss) == 3

    def test_model_training_step(self, input_shape):
        """Test model can train on dummy data."""
        model = create_spectra_model(input_shape)
        compile_model(model)

        # Create dummy data
        x_train = np.random.randn(32, input_shape).astype(np.float32)
        y_train = (
            np.random.randn(32, 1),
            np.random.randn(32, 1),
            np.random.randn(32, 1)
        )

        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        assert history.epoch == [0]
        assert 'loss' in history.history

    def test_model_inference(self, input_shape):
        """Test model inference on new data."""
        model = create_spectra_model(input_shape)
        compile_model(model)

        x_test = np.random.randn(5, input_shape).astype(np.float32)
        predictions = model.predict(x_test, verbose=0)

        assert len(predictions) == 3
        assert all(p.shape == (5, 1) for p in predictions)


class TestSpectraAnalyzerModel:
    """Test custom SpectraAnalyzer model."""

    @pytest.fixture
    def input_shape(self):
        return 3000

    def test_custom_model_creation(self, input_shape):
        """Test custom model creation."""
        model = SpectraAnalyzer(input_shape)
        assert model is not None

    def test_custom_model_call(self, input_shape):
        """Test custom model forward pass."""
        model = SpectraAnalyzer(input_shape)
        x = np.random.randn(10, input_shape).astype(np.float32)

        outputs = model(x, training=False)

        assert isinstance(outputs, dict)
        assert 'temperature' in outputs
        assert 'composition' in outputs
        assert 'radial_velocity' in outputs
        assert outputs['temperature'].shape == (10, 1)


class TestConv1DSpectraModel:
    """Test StarNet-inspired Conv1D model architecture."""

    # Use a smaller wavelength grid in tests so they run fast
    N_WAVELENGTHS = 512

    @pytest.fixture
    def input_shape(self):
        """Spectral dimension for Conv1D tests (smaller than production for speed)."""
        return self.N_WAVELENGTHS

    def test_conv1d_model_creation(self, input_shape):
        """Conv1D model is created with 3 output branches."""
        model = create_conv1d_spectra_model(input_shape)
        assert model is not None
        assert len(model.outputs) == 3

    def test_conv1d_model_output_names(self, input_shape):
        """Output layers are named temperature, composition, radial_velocity."""
        model = create_conv1d_spectra_model(input_shape)
        assert model.output_names == ['temperature', 'composition', 'radial_velocity']

    def test_conv1d_model_input_shape(self, input_shape):
        """Model input shape includes channel dimension: (None, n_wavelengths, 1)."""
        model = create_conv1d_spectra_model(input_shape)
        assert model.input_shape == (None, input_shape, 1)

    def test_conv1d_model_output_shapes(self, input_shape):
        """Each output head produces (batch, 1) tensor."""
        model = create_conv1d_spectra_model(input_shape)
        batch_size = 8
        # Conv1D expects (batch, timesteps, channels)
        inputs = np.random.randn(batch_size, input_shape, 1).astype(np.float32)

        outputs = model(inputs, training=False)

        assert outputs[0].shape == (batch_size, 1)  # temperature
        assert outputs[1].shape == (batch_size, 1)  # composition
        assert outputs[2].shape == (batch_size, 1)  # radial_velocity

    def test_conv1d_model_compilation(self, input_shape):
        """Conv1D model compiles with multi-task MSE losses."""
        model = create_conv1d_spectra_model(input_shape)
        compile_model(model)
        assert model.optimizer is not None
        assert len(model.loss) == 3

    def test_conv1d_model_training_step(self, input_shape):
        """Model completes a single training epoch without errors."""
        model = create_conv1d_spectra_model(input_shape)
        compile_model(model)

        x_train = np.random.randn(16, input_shape, 1).astype(np.float32)
        y_train = (
            np.random.randn(16, 1).astype(np.float32),
            np.random.randn(16, 1).astype(np.float32),
            np.random.randn(16, 1).astype(np.float32),
        )

        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        assert history.epoch == [0]
        assert 'loss' in history.history

    def test_conv1d_model_inference(self, input_shape):
        """Model performs inference and returns 3 output arrays."""
        model = create_conv1d_spectra_model(input_shape)

        x_test = np.random.randn(5, input_shape, 1).astype(np.float32)
        predictions = model(x_test, training=False)

        assert len(predictions) == 3
        assert all(p.shape == (5, 1) for p in predictions)

    def test_conv1d_model_has_conv_layers(self, input_shape):
        """Model architecture contains at least one Conv1D layer."""
        model = create_conv1d_spectra_model(input_shape)
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert 'Conv1D' in layer_types

    def test_conv1d_model_parameter_count(self, input_shape):
        """Model has a reasonable number of trainable parameters (< 5M)."""
        model = create_conv1d_spectra_model(input_shape)
        n_params = model.count_params()
        assert n_params < 5_000_000, f"Model too large: {n_params:,} params"
