"""GPU configuration and utilities for TensorFlow."""

import tensorflow as tf

# Queried once at import time; reused by both configure_gpu() and get_device_info().
_GPUS = tf.config.list_physical_devices('GPU')
_CPUS = tf.config.list_physical_devices('CPU')


def configure_gpu() -> None:
    """Enable dynamic memory growth on all available GPUs.

    Prevents TensorFlow from pre-allocating the entire GPU VRAM, which
    avoids OOM errors when sharing the device with other processes.
    """
    if _GPUS:
        try:
            for gpu in _GPUS:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU configured: {len(_GPUS)} device(s) found")
            for gpu in _GPUS:
                print(f"  - {gpu}")
        except RuntimeError as e:
            print(f"✗ GPU configuration error: {e}")
    else:
        print("⚠ No GPU detected, falling back to CPU")


def get_device_info() -> dict:
    """Return a snapshot of the current TF device environment."""
    return {
        'gpus':            len(_GPUS),
        'cpus':            len(_CPUS),
        'gpu_devices':     [str(g) for g in _GPUS],
        'tensorflow_version': tf.__version__,
    }
