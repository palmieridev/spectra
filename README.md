# Stellar Spectra Analyzer

A TensorFlow-based neural network for predicting stellar properties (temperature, composition, radial velocity) from spectroscopic data using SDSS observations.

## Features

- **Multi-task regression**: Simultaneously predicts temperature, composition, and radial velocity
- **GPU optimized**: Configured for NVIDIA 4060 GPU with dynamic memory growth
- **Test-driven development**: Comprehensive test suite using pytest
- **Data preprocessing**: StandardScaler normalization for stable training
- **Production-ready**: Model checkpointing, early stopping, learning rate reduction

## Project Structure

```
spectra/
├── src/
│   ├── __init__.py           # Package exports
│   ├── gpu_config.py         # GPU configuration & utilities
│   ├── data.py               # Data loading & preprocessing
│   ├── models.py             # Neural network architectures
│   └── train.py              # Training pipeline
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Pytest configuration
│   ├── test_data.py          # Data preprocessing tests
│   └── test_models.py        # Model architecture tests
├── data/
│   ├── raw/                  # Original SDSS data
│   └── processed/            # Preprocessed data
├── models/                   # Saved model artifacts
├── requirements.txt          # Python dependencies
└── pytest.ini                # Pytest configuration
```

## Installation

1. Clone the repository:
```bash
cd spectra
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify GPU detection:
```bash
python -c "from src.gpu_config import get_device_info; print(get_device_info())"
```

## Quick Start

### 1. Run Tests (TDD Approach)

Run the full test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_models.py -v
```

### 2. Train Model

```bash
python -m src.train --epochs 100 --batch-size 32 --output-dir ./models
```

### 3. Use Trained Model

```python
from src.gpu_config import configure_gpu
from src.models import create_spectra_model, compile_model
from src.data import SpectraPreprocessor
import tensorflow as tf
import numpy as np

# Configure GPU
configure_gpu()

# Load preprocessor and model
preprocessor = ...  # Load your fitted preprocessor
model = tf.keras.models.load_model('models/best_model.h5')

# Prepare new spectra
new_spectra = np.random.randn(10, 3000)  # 10 spectra, 3000 wavelength bins
normalized, _ = preprocessor.transform(new_spectra)

# Predict
predictions = model.predict(normalized)
temperatures, compositions, velocities = predictions
```

## Model Architecture

### Shared Feature Extraction
- Dense(512) → BatchNorm → Dropout(0.3)
- Dense(256) → BatchNorm → Dropout(0.3)
- Dense(128) → BatchNorm

### Task-Specific Output Heads
- Temperature: Dense(64) → Dense(1, linear)
- Composition: Dense(64) → Dense(1, linear)
- Radial Velocity: Dense(64) → Dense(1, linear)

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Mean Squared Error (MSE) for each task
- **Callbacks**:
  - EarlyStopping (patience=5)
  - ModelCheckpoint (best model)
  - ReduceLROnPlateau (factor=0.5, patience=3)

## Loading SDSS Data

The `data.py` module provides a template for loading SDSS data. SDSS spectra are typically distributed in FITS format. Implement the `load_sdss_spectra()` function:

```python
def load_sdss_spectra(filepath: str) -> Tuple[np.ndarray, dict]:
    """Load SDSS spectra from FITS file."""
    from astropy.io import fits
    
    with fits.open(filepath) as hdul:
        flux = hdul['FLUX'].data  # Shape: (n_spectra, n_wavelengths)
        wavelength = hdul['WAVE'].data
        
        labels = {
            'temperature': hdul['TEFF'].data,
            'composition': hdul['[FE/H]'].data,  # Metallicity proxy
            'radial_velocity': hdul['Z'].data * 299792  # Redshift to km/s
        }
    
    return flux, labels
```

## Best Practices Applied

### TensorFlow & Keras
✓ Normalization before training
✓ Batch normalization for stability
✓ Dropout for regularization
✓ Appropriate activation functions (ReLU for hidden, linear for regression)
✓ Proper loss functions (MSE for regression)
✓ Model checkpointing and early stopping

### Test-Driven Development
✓ Test data preprocessing (shapes, normalization)
✓ Test model architecture (input/output shapes)
✓ Test training loop (gradient updates)
✓ Test inference (predictions)
✓ Fixtures for reusable test data

### GPU Optimization
✓ Dynamic memory growth (prevent OOM)
✓ Data prefetching (tf.data.AUTOTUNE)
✓ Batch processing for GPU efficiency
✓ Automatic mixed precision capable (add `tf.keras.mixed_precision`)

## Configuration for NVIDIA 4060

The 4060 has 24GB VRAM. Current settings:
- Batch size: 32 (adjust if memory errors occur)
- Hidden units: [512, 256, 128]
- Dropout: 0.3

If you encounter OOM errors:
1. Reduce batch size: `--batch-size 16`
2. Reduce hidden units in models.py
3. Increase data prefetch parallelism

If you have memory to spare:
1. Increase batch size: `--batch-size 64`
2. Increase hidden layer units for better capacity

## Development Workflow

1. **Write tests first**: Define test cases in `tests/`
2. **Implement features**: Add code to `src/`
3. **Run tests**: `pytest` to verify
4. **Train & evaluate**: Use `src/train.py`
5. **Commit**: Git commit with passing tests

Example:
```bash
# 1. Add new test for preprocessing
echo "def test_new_feature(): ..." >> tests/test_data.py

# 2. Implement feature
# ... edit src/data.py

# 3. Run tests
pytest tests/test_data.py -v

# 4. Train and evaluate
python -m src.train --epochs 50
```

## Next Steps

1. **Integrate real SDSS data**: Implement `load_sdss_spectra()` for your data source
2. **Add inference pipeline**: Create `src/inference.py` for model serving
3. **Expand evaluation**: Add metrics for spectral synthesis accuracy
4. **Hyperparameter tuning**: Use Keras Tuner for AutoML
5. **Model export**: Convert to TensorFlow Lite for mobile deployment

## References

- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [SDSS Data Release](https://www.sdss.org/)
- [Spectroscopic Analysis](https://arxiv.org/abs/astro-ph/0502026)

## License

MIT
