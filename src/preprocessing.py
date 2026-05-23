"""Spectral preprocessing pipeline for SDSS stellar spectra.

Pipeline order per spectrum:
  1. mask_bad_pixels  — replace ivar=0 (and NaN) pixels with local interpolation
  2. interpolate_to_grid — resample onto a common log-spaced wavelength grid
  3. normalize_continuum — running-median continuum normalization → flux ≈ 1
"""

import numpy as np
from scipy import interpolate, ndimage

# Common wavelength grid (SDSS BOSS optical range, log-spaced, 3748 pixels).
SDSS_WAVE_GRID: np.ndarray = np.logspace(
    np.log10(3800.0), np.log10(9200.0), 3748
).astype(np.float32)


def normalize_continuum(flux: np.ndarray, window_size: int = 150) -> np.ndarray:
    """Running-median continuum normalization.

    Estimates the stellar continuum via a median filter then divides the flux
    by that envelope so the continuum sits near unity.  Continuum values below
    1e-10 are clamped to prevent division by zero.

    Args:
        flux:        1-D flux array (any length).
        window_size: Running-median window width in pixels.

    Returns:
        Continuum-normalized flux as float32 with the same shape as *flux*.
    """
    flux = np.asarray(flux, dtype=np.float32)

    if np.any(np.isnan(flux)):
        flux = flux.copy()
        flux[np.isnan(flux)] = np.nanmedian(flux)

    continuum = ndimage.median_filter(flux, size=window_size, mode='reflect')
    continuum = np.where(continuum > 1e-10, continuum, 1e-10)
    return (flux / continuum).astype(np.float32)


def interpolate_to_grid(
    wavelength: np.ndarray,
    flux: np.ndarray,
    target_grid: np.ndarray = None,
) -> np.ndarray:
    """Resample a spectrum onto a common wavelength grid.

    Out-of-range pixels are filled with the nearest edge value to keep the
    output NaN-free.  All arithmetic stays in float32 to avoid unnecessary
    dtype promotion.

    Args:
        wavelength:  Source wavelength array (Å, monotonically increasing).
        flux:        Corresponding flux array.
        target_grid: Target wavelength grid (Å).  Defaults to
                     :data:`SDSS_WAVE_GRID`.

    Returns:
        Resampled flux as float32 with shape ``(len(target_grid),)``.
    """
    if target_grid is None:
        target_grid = SDSS_WAVE_GRID

    wavelength = np.asarray(wavelength, dtype=np.float32)
    flux = np.asarray(flux, dtype=np.float32)
    target_grid = np.asarray(target_grid, dtype=np.float32)

    interp_fn = interpolate.interp1d(
        wavelength,
        flux,
        kind='linear',
        bounds_error=False,
        fill_value=(flux[0], flux[-1]),
    )
    return interp_fn(target_grid).astype(np.float32)


def mask_bad_pixels(flux: np.ndarray, ivar: np.ndarray) -> np.ndarray:
    """Replace bad pixels (ivar ≤ 0 or non-finite flux) with local interpolation.

    SDSS sets ivar=0 for pixels flagged as unreliable.  Those pixels, along
    with any NaN/Inf values, are replaced by linearly interpolating the
    surrounding good pixels.

    Args:
        flux: 1-D flux array.
        ivar: 1-D inverse-variance array (same length as *flux*).

    Returns:
        Cleaned flux as float32 with the same shape as *flux*.
    """
    flux = np.asarray(flux, dtype=np.float32).copy()
    ivar = np.asarray(ivar, dtype=np.float32)

    bad_mask = (ivar <= 0) | ~np.isfinite(flux)

    if not np.any(bad_mask):
        return flux

    if np.all(bad_mask):
        flux[:] = 0.0
        return flux

    indices = np.arange(len(flux), dtype=np.float32)
    good = ~bad_mask

    fill_fn = interpolate.interp1d(
        indices[good],
        flux[good],
        kind='linear',
        bounds_error=False,
        fill_value=(flux[good][0], flux[good][-1]),
    )
    flux[bad_mask] = fill_fn(indices[bad_mask]).astype(np.float32)
    return flux


def preprocess_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    target_grid: np.ndarray = None,
) -> np.ndarray:
    """End-to-end preprocessing pipeline for a single stellar spectrum.

    Runs :func:`mask_bad_pixels` → :func:`interpolate_to_grid` →
    :func:`normalize_continuum` and returns a float32 array ready for the
    neural network.

    Args:
        wavelength:  1-D wavelength array (Å).
        flux:        1-D flux array.
        ivar:        1-D inverse-variance array.
        target_grid: Optional custom wavelength grid.  Defaults to
                     :data:`SDSS_WAVE_GRID`.

    Returns:
        Preprocessed, continuum-normalized spectrum as float32 with shape
        ``(len(target_grid),)``.
    """
    if target_grid is None:
        target_grid = SDSS_WAVE_GRID

    return normalize_continuum(
        interpolate_to_grid(
            wavelength,
            mask_bad_pixels(flux, ivar),
            target_grid=target_grid,
        )
    )
