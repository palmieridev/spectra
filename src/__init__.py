"""Stellar spectra analyzer package.

TensorFlow-dependent symbols (models, GPU config, training) are imported
lazily so that the package can be imported in environments where TensorFlow
is not installed (e.g. during pure-Python unit tests for the data pipeline).
"""

# Always-available symbols (no TensorFlow dependency)
from src.preprocessing import (
    preprocess_spectrum,
    normalize_continuum,
    interpolate_to_grid,
    mask_bad_pixels,
    SDSS_WAVE_GRID,
)
from src.sdss_loader import SDSSLoader
from src.data import SpectraPreprocessor, load_sdss_spectra

__all__ = [
    # Spectral preprocessing (no TF)
    'preprocess_spectrum',
    'normalize_continuum',
    'interpolate_to_grid',
    'mask_bad_pixels',
    'SDSS_WAVE_GRID',
    # SDSS data loader (no TF)
    'SDSSLoader',
    # Data pipeline (TF only used inside create_dataset at call time)
    'SpectraPreprocessor',
    'load_sdss_spectra',
    'create_dataset',
    # TF-dependent symbols exported below (when TF is available)
    'configure_gpu',
    'get_device_info',
    'create_spectra_model',
    'create_conv1d_spectra_model',
    'compile_model',
    'SpectraAnalyzer',
    'train',
]

# TensorFlow-dependent imports — guarded so the package stays importable
# in environments without TF (pure-Python preprocessing / SDSS tests).
try:
    from src.gpu_config import configure_gpu, get_device_info
    from src.models import (
        create_spectra_model,
        create_conv1d_spectra_model,
        compile_model,
        SpectraAnalyzer,
    )
    from src.data import create_dataset
    from src.train import train
except ImportError:
    pass
