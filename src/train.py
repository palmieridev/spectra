"""Training script for stellar spectra analyzer.

Supports two model architectures and two data sources via CLI flags::

    # Dense model, dummy data (default — no internet required)
    python -m src.train --model-type dense --data-source dummy

    # Conv1D model, dummy data
    python -m src.train --model-type conv1d --data-source dummy

    # Conv1D model, real SDSS data (requires internet)
    python -m src.train --model-type conv1d --data-source sdss --n-stars 2000

GPU training on the RTX 4060 is configured automatically via
:func:`src.gpu_config.configure_gpu`.
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.gpu_config import configure_gpu, get_device_info
from src.models import create_spectra_model, create_conv1d_spectra_model, compile_model, OUTPUT_LABELS
from src.data import SpectraPreprocessor, create_dataset


def _make_dummy_data(n_samples: int, n_wavelengths: int):
    """Return synthetic (spectra, labels) arrays for smoke-testing."""
    spectra = np.random.randn(n_samples, n_wavelengths).astype(np.float32)
    labels = {
        'temperature':    np.random.uniform(4000, 10000, n_samples),
        'composition':    np.random.uniform(-2.5, 0.5, n_samples),
        'radial_velocity': np.random.uniform(-300, 300, n_samples),
    }
    return spectra, labels


def _split_labels(labels: dict, idx) -> dict:
    """Index every label array by *idx* — works for both ndarray and list indices."""
    return {k: v[idx] for k, v in labels.items()}


def _make_dataset(spectra: np.ndarray, labels: dict, batch_size: int, model_type: str):
    """Create a batched tf.data.Dataset for *model_type*.

    Conv1D expects an extra trailing channel dimension ``(n, wavelengths, 1)``.
    Both branches delegate to :func:`~src.data.create_dataset` for consistent
    shuffling / prefetch behaviour.
    """
    if model_type == 'conv1d':
        spectra = spectra[..., np.newaxis]
    return create_dataset(spectra, labels, batch_size=batch_size)


def train(
    spectra_train: np.ndarray,
    labels_train: dict,
    spectra_val: np.ndarray,
    labels_val: dict,
    epochs: int = 50,
    batch_size: int = 32,
    output_dir: str = './models',
    model_type: str = 'dense',
    learning_rate: float = 0.001,
):
    """Train the stellar spectra analyzer model.

    Args:
        spectra_train: Training spectra ``(n_train, n_wavelengths)``.
        labels_train:  Training labels dict.
        spectra_val:   Validation spectra.
        labels_val:    Validation labels dict.
        epochs:        Number of training epochs.
        batch_size:    Mini-batch size.
        output_dir:    Directory for model checkpoints.
        model_type:    ``'dense'`` or ``'conv1d'``.
        learning_rate: Adam learning rate.

    Returns:
        Tuple ``(model, preprocessor, history)``.
    """
    configure_gpu()
    print("Device info:", get_device_info())

    print("\n[1/4] Preprocessing data…")
    preprocessor = SpectraPreprocessor()
    spectra_train_norm, labels_train_norm = preprocessor.fit_transform(spectra_train, labels_train)
    spectra_val_norm, labels_val_norm = preprocessor.transform(spectra_val, labels_val)

    print("[2/4] Creating datasets…")
    train_dataset, _ = _make_dataset(spectra_train_norm, labels_train_norm, batch_size, model_type)
    val_dataset, _   = _make_dataset(spectra_val_norm,   labels_val_norm,   batch_size, model_type)

    print("[3/4] Building model…")
    input_shape = spectra_train.shape[1]
    if model_type == 'conv1d':
        model = create_conv1d_spectra_model(input_shape)
        print(f"  Architecture: Conv1D (StarNet-inspired), input ({input_shape}, 1)")
    else:
        model = create_spectra_model(input_shape)
        print(f"  Architecture: Dense multi-task, input ({input_shape},)")

    compile_model(model, learning_rate=learning_rate)
    model.summary()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_path  = f'{output_dir}/best_{model_type}_model.keras'
    final_path = f'{output_dir}/final_{model_type}_model.keras'

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_path, monitor='val_loss',
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'{output_dir}/logs/{model_type}', histogram_freq=1,
        ),
    ]

    print("[4/4] Training model…")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_path)
    print(f"\n✓ Training complete.\n  Best  → {best_path}\n  Final → {final_path}")
    return model, preprocessor, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train stellar spectra analyzer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--batch-size',    type=int,   default=32)
    parser.add_argument('--output-dir',    default='./models')
    parser.add_argument('--model-type',    choices=['dense', 'conv1d'], default='dense')
    parser.add_argument('--data-source',   choices=['dummy', 'sdss'],   default='dummy')
    parser.add_argument('--n-stars',       type=int,   default=1000)
    parser.add_argument('--min-snr',       type=float, default=10.0)
    parser.add_argument('--cache-dir',     default='data/raw')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--val-fraction',  type=float, default=0.15)
    args = parser.parse_args()

    if args.data_source == 'sdss':
        print(f"Loading {args.n_stars} SDSS spectra (S/N ≥ {args.min_snr})…")
        from src.data import load_sdss_spectra
        spectra_all, labels_all = load_sdss_spectra(
            cache_dir=args.cache_dir, n_stars=args.n_stars, min_snr=args.min_snr,
        )
        print(f"  Loaded {spectra_all.shape[0]} spectra × {spectra_all.shape[1]} bins")
    else:
        n_wavelengths = 3748
        print(f"Generating {args.n_stars} synthetic spectra ({n_wavelengths} bins)…")
        spectra_all, labels_all = _make_dummy_data(args.n_stars, n_wavelengths)

    # Train / validation split via sklearn for reproducible shuffling.
    label_arrays = [labels_all[k] for k in OUTPUT_LABELS]
    splits = train_test_split(
        spectra_all, *label_arrays,
        test_size=args.val_fraction,
        random_state=42,
    )
    # train_test_split returns [X_train, X_test, y0_train, y0_test, y1_train, y1_test, ...]
    spectra_train, spectra_val = splits[0], splits[1]
    labels_train = {k: splits[2 + i * 2]     for i, k in enumerate(OUTPUT_LABELS)}
    labels_val   = {k: splits[2 + i * 2 + 1] for i, k in enumerate(OUTPUT_LABELS)}

    train(
        spectra_train, labels_train,
        spectra_val,   labels_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        model_type=args.model_type,
        learning_rate=args.learning_rate,
    )
