"""Neural network models for stellar spectra analysis."""

import tensorflow as tf
from tensorflow import keras

# Canonical label names used as output-layer names throughout the project.
OUTPUT_LABELS = ['temperature', 'composition', 'radial_velocity']


def _build_output_heads(x: tf.Tensor) -> list[tf.Tensor]:
    """Build one Dense(64)→Dense(1, linear) head per output label.

    Shared by both the Dense and Conv1D model factories to avoid duplicating
    six layers every time a new architecture is added.
    """
    return [
        keras.layers.Dense(1, activation='linear', name=label)(
            keras.layers.Dense(64, activation='relu')(x)
        )
        for label in OUTPUT_LABELS
    ]


def create_spectra_model(
    input_shape: int,
    hidden_units: list = [512, 256, 128],
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Dense multi-task regression model for stellar property prediction.

    Args:
        input_shape:  Number of wavelength bins.
        hidden_units: Hidden layer sizes for the shared trunk.
        dropout_rate: Dropout probability.

    Returns:
        Uncompiled Keras functional model with outputs for each
        label in :data:`OUTPUT_LABELS`.
    """
    inputs = keras.layers.Input(shape=(input_shape,), name='spectra')

    x = keras.layers.Dense(hidden_units[0], activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    for units in hidden_units[1:]:
        x = keras.layers.Dense(units, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    return keras.Model(
        inputs=inputs,
        outputs=_build_output_heads(x),
        name='spectra_analyzer',
    )


def compile_model(model: keras.Model, learning_rate: float = 0.001) -> keras.Model:
    """Compile model with per-output MSE loss and MAE metric.

    Args:
        model:         Keras model to compile.
        learning_rate: Adam learning rate.

    Returns:
        The same model instance, now compiled.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={label: 'mse' for label in OUTPUT_LABELS},
        loss_weights={label: 1.0 for label in OUTPUT_LABELS},
        metrics={label: 'mae' for label in OUTPUT_LABELS},
    )
    return model


class SpectraAnalyzer(keras.Model):
    """Subclassed Keras model for spectral analysis.

    Useful when a custom training loop is needed.  Functionally equivalent
    to :func:`create_spectra_model` but uses the subclassing API.
    """

    def __init__(self, input_shape: int, **kwargs):
        super().__init__(**kwargs)
        self.input_layer = keras.layers.InputLayer(input_shape=(input_shape,))
        self._shared = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
        ])
        # One head per label, stored in insertion order.
        self._heads = {
            label: keras.Sequential([
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(1, activation='linear'),
            ])
            for label in OUTPUT_LABELS
        }

    def call(self, inputs, training=False):
        x = self._shared(inputs, training=training)
        return {label: head(x) for label, head in self._heads.items()}


def create_conv1d_spectra_model(
    input_shape: int,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """StarNet-inspired 1-D CNN for stellar spectra regression.

    Stacked Conv1D layers capture local absorption-line features before
    the fully-connected trunk projects to global stellar parameters.
    Input tensors must have shape ``(batch, input_shape, 1)``.

    Args:
        input_shape:  Number of wavelength bins.
        dropout_rate: Dropout probability for the FC trunk.

    Returns:
        Uncompiled Keras functional model with outputs for each label in
        :data:`OUTPUT_LABELS`.  Call :func:`compile_model` before training.
    """
    inputs = keras.layers.Input(shape=(input_shape, 1), name='spectra')

    x = keras.layers.Conv1D(32, kernel_size=8, activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPooling1D(pool_size=4)(x)
    x = keras.layers.Conv1D(64, kernel_size=8, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling1D(pool_size=4)(x)
    x = keras.layers.Conv1D(128, kernel_size=8, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling1D(pool_size=4)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(256, activation='relu')(x)

    return keras.Model(
        inputs=inputs,
        outputs=_build_output_heads(x),
        name='conv1d_spectra_analyzer',
    )
