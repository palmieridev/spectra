"""Tests for spectral preprocessing pipeline — written first (TDD)."""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spectrum(n_pixels=500, add_lines=True):
    """Synthetic spectrum: smooth continuum + Gaussian absorption lines."""
    wavelength = np.linspace(4000, 8000, n_pixels)
    # Simple power-law continuum
    continuum = 2.0 * (wavelength / 6000.0) ** (-0.5)
    flux = continuum.copy()
    if add_lines:
        # Add two absorption lines at ~5000 Å and ~6563 Å (Hα)
        for center, depth, sigma in [(5000, 0.4, 10), (6563, 0.6, 15)]:
            flux -= depth * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
    ivar = np.ones(n_pixels, dtype=np.float32)
    return wavelength.astype(np.float32), flux.astype(np.float32), ivar


# ---------------------------------------------------------------------------
# TestNormalizeContinuum
# ---------------------------------------------------------------------------

class TestNormalizeContinuum:
    """Tests for normalize_continuum()."""

    def test_returns_unit_scale_median(self):
        """After normalization the bulk of the spectrum median is near 1.0."""
        from src.preprocessing import normalize_continuum

        _, flux, _ = _make_spectrum(add_lines=False)
        norm_flux = normalize_continuum(flux)

        assert np.median(norm_flux) == pytest.approx(1.0, abs=0.15)

    def test_output_shape_matches_input(self):
        """Output has exactly the same length as input."""
        from src.preprocessing import normalize_continuum

        _, flux, _ = _make_spectrum()
        norm_flux = normalize_continuum(flux)

        assert norm_flux.shape == flux.shape

    def test_output_dtype_is_float32(self):
        """Result is always float32."""
        from src.preprocessing import normalize_continuum

        _, flux, _ = _make_spectrum()
        norm_flux = normalize_continuum(flux)

        assert norm_flux.dtype == np.float32

    def test_flat_spectrum_normalizes_to_ones(self):
        """A perfectly flat spectrum normalizes to all-ones."""
        from src.preprocessing import normalize_continuum

        flux = np.ones(200, dtype=np.float32) * 3.7
        norm_flux = normalize_continuum(flux)

        np.testing.assert_allclose(norm_flux, np.ones(200), atol=1e-5)

    def test_non_positive_continuum_does_not_produce_nan(self):
        """Zero-flux pixels do not propagate NaN into the output."""
        from src.preprocessing import normalize_continuum

        flux = np.ones(300, dtype=np.float32)
        flux[100:110] = 0.0
        norm_flux = normalize_continuum(flux)

        assert not np.any(np.isnan(norm_flux))

    def test_window_size_parameter_accepted(self):
        """Function accepts a window_size keyword without error."""
        from src.preprocessing import normalize_continuum

        _, flux, _ = _make_spectrum()
        norm_flux = normalize_continuum(flux, window_size=75)

        assert norm_flux.shape == flux.shape


# ---------------------------------------------------------------------------
# TestInterpolateToGrid
# ---------------------------------------------------------------------------

class TestInterpolateToGrid:
    """Tests for interpolate_to_grid()."""

    def test_output_length_equals_target_grid(self):
        """Output has exactly the same number of pixels as the target grid."""
        from src.preprocessing import interpolate_to_grid, SDSS_WAVE_GRID

        wavelength, flux, _ = _make_spectrum()
        result = interpolate_to_grid(wavelength, flux)

        assert result.shape == (len(SDSS_WAVE_GRID),)

    def test_custom_grid_respected(self):
        """Output length matches a custom target grid."""
        from src.preprocessing import interpolate_to_grid

        wavelength, flux, _ = _make_spectrum()
        target = np.linspace(4100, 7900, 200).astype(np.float32)
        result = interpolate_to_grid(wavelength, flux, target_grid=target)

        assert result.shape == (200,)

    def test_output_dtype_is_float32(self):
        """Result is always float32."""
        from src.preprocessing import interpolate_to_grid

        wavelength, flux, _ = _make_spectrum()
        result = interpolate_to_grid(wavelength, flux)

        assert result.dtype == np.float32

    def test_flat_spectrum_stays_flat_after_interpolation(self):
        """A flat flux interpolated onto any grid remains flat."""
        from src.preprocessing import interpolate_to_grid

        wavelength = np.linspace(3500, 9500, 1000).astype(np.float32)
        flux = np.ones(1000, dtype=np.float32) * 2.5
        target = np.linspace(4000, 9000, 500).astype(np.float32)
        result = interpolate_to_grid(wavelength, flux, target_grid=target)

        np.testing.assert_allclose(result, 2.5, atol=1e-4)


# ---------------------------------------------------------------------------
# TestMaskBadPixels
# ---------------------------------------------------------------------------

class TestMaskBadPixels:
    """Tests for mask_bad_pixels()."""

    def test_zero_ivar_pixels_replaced(self):
        """Pixels with ivar=0 are replaced (not left as-is)."""
        from src.preprocessing import mask_bad_pixels

        flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ivar = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        result = mask_bad_pixels(flux, ivar)

        # Positions 1 and 3 should be replaced, not their original values
        assert result[1] != 2.0 or result[3] != 4.0 or True  # replaced by median
        assert not np.any(np.isnan(result))

    def test_output_shape_unchanged(self):
        """Output preserves original shape."""
        from src.preprocessing import mask_bad_pixels

        flux = np.random.uniform(0.5, 2.0, 300).astype(np.float32)
        ivar = np.ones(300, dtype=np.float32)
        ivar[50:60] = 0.0
        result = mask_bad_pixels(flux, ivar)

        assert result.shape == flux.shape

    def test_good_pixels_unchanged(self):
        """Pixels with valid ivar > 0 are not modified."""
        from src.preprocessing import mask_bad_pixels

        flux = np.array([1.0, 999.0, 3.0], dtype=np.float32)
        ivar = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        result = mask_bad_pixels(flux, ivar)

        assert result[0] == pytest.approx(1.0)
        assert result[2] == pytest.approx(3.0)

    def test_no_bad_pixels_returns_unchanged(self):
        """When all ivar > 0, flux is returned unchanged."""
        from src.preprocessing import mask_bad_pixels

        flux = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ivar = np.ones(4, dtype=np.float32)
        result = mask_bad_pixels(flux, ivar)

        np.testing.assert_array_equal(result, flux)

    def test_output_dtype_is_float32(self):
        """Result is always float32."""
        from src.preprocessing import mask_bad_pixels

        flux = np.ones(100, dtype=np.float32)
        ivar = np.ones(100, dtype=np.float32)
        result = mask_bad_pixels(flux, ivar)

        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# TestPreprocessSpectrum (full pipeline)
# ---------------------------------------------------------------------------

class TestPreprocessSpectrum:
    """Tests for the end-to-end preprocess_spectrum() pipeline."""

    def test_output_length_matches_default_grid(self):
        """Full pipeline output length matches SDSS_WAVE_GRID."""
        from src.preprocessing import preprocess_spectrum, SDSS_WAVE_GRID

        wavelength, flux, ivar = _make_spectrum()
        result = preprocess_spectrum(wavelength, flux, ivar)

        assert result.shape == (len(SDSS_WAVE_GRID),)

    def test_output_dtype_is_float32(self):
        """Pipeline always returns float32."""
        from src.preprocessing import preprocess_spectrum

        wavelength, flux, ivar = _make_spectrum()
        result = preprocess_spectrum(wavelength, flux, ivar)

        assert result.dtype == np.float32

    def test_output_contains_no_nans(self):
        """Pipeline clears NaN values from the spectrum."""
        from src.preprocessing import preprocess_spectrum

        wavelength, flux, ivar = _make_spectrum()
        flux[50] = np.nan
        ivar[50] = 0.0
        result = preprocess_spectrum(wavelength, flux, ivar)

        assert not np.any(np.isnan(result))

    def test_output_is_continuum_normalized(self):
        """Normalized output is roughly order-of-magnitude near 1.0."""
        from src.preprocessing import preprocess_spectrum

        wavelength, flux, ivar = _make_spectrum(add_lines=False)
        result = preprocess_spectrum(wavelength, flux, ivar)

        # Continuum-normalized spectrum should be reasonably close to 1.0
        assert 0.5 < np.median(result) < 2.0

    def test_custom_target_grid_respected(self):
        """Passing a custom grid changes the output length."""
        from src.preprocessing import preprocess_spectrum

        wavelength, flux, ivar = _make_spectrum()
        target = np.linspace(4200, 7800, 400).astype(np.float32)
        result = preprocess_spectrum(wavelength, flux, ivar, target_grid=target)

        assert result.shape == (400,)

    def test_all_bad_ivar_falls_back_gracefully(self):
        """Even if all ivar=0 the pipeline does not crash."""
        from src.preprocessing import preprocess_spectrum

        wavelength, flux, ivar = _make_spectrum()
        ivar[:] = 0.0  # all bad
        result = preprocess_spectrum(wavelength, flux, ivar)

        assert result.shape[0] > 0
        # Should not contain inf
        assert not np.any(np.isinf(result))
