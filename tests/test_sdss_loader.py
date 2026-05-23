"""Tests for SDSS data loader — written first (TDD)."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, call
from astropy.table import Table
from astropy.io import fits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sspp_table(n=5):
    """Return a minimal astropy Table that looks like an SSPP CAS result."""
    return Table({
        'plate':   np.arange(n, dtype=np.int32),
        'mjd':     np.full(n, 55000, dtype=np.int32),
        'fiberID': np.arange(1, n + 1, dtype=np.int32),
        'Teff':    np.linspace(5000, 7000, n),
        'feh':     np.linspace(-1.0, 0.0, n),
        'logg':    np.linspace(3.5, 4.5, n),
        'elodiervFinal': np.linspace(-50, 50, n),
        'snMedian': np.full(n, 25.0),
    })


def _make_fits_hdul(n_pixels=3748):
    """Return a mock FITS HDUList matching SDSS BOSS spPlate format."""
    loglam = np.linspace(np.log10(3800), np.log10(9200), n_pixels).astype(np.float32)
    flux   = np.random.uniform(0.5, 2.0, n_pixels).astype(np.float32)
    ivar   = np.ones(n_pixels, dtype=np.float32)

    col_loglam = fits.Column(name='loglam', format='E', array=loglam)
    col_flux   = fits.Column(name='flux',   format='E', array=flux)
    col_ivar   = fits.Column(name='ivar',   format='E', array=ivar)

    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.BinTableHDU.from_columns([col_loglam, col_flux, col_ivar])
    hdul = fits.HDUList([hdu0, hdu1])
    return hdul, loglam, flux, ivar


# ---------------------------------------------------------------------------
# TestSDSSLoader
# ---------------------------------------------------------------------------

class TestSDSSLoader:
    """Tests for the SDSSLoader class."""

    # ------------------------------------------------------------------
    # query_sspp_stars
    # ------------------------------------------------------------------

    def test_query_sspp_stars_returns_dataframe(self):
        """query_sspp_stars() returns a non-empty DataFrame."""
        from src.sdss_loader import SDSSLoader

        mock_table = _make_sspp_table(n=5)
        with patch('src.sdss_loader.SDSS.query_sql', return_value=mock_table):
            loader = SDSSLoader()
            result = loader.query_sspp_stars(n_stars=5)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_query_sspp_stars_has_required_columns(self):
        """Result DataFrame contains all expected stellar parameter columns."""
        from src.sdss_loader import SDSSLoader

        mock_table = _make_sspp_table(n=3)
        with patch('src.sdss_loader.SDSS.query_sql', return_value=mock_table):
            loader = SDSSLoader()
            df = loader.query_sspp_stars(n_stars=3)

        for col in ('plate', 'mjd', 'fiberID', 'Teff', 'feh', 'elodiervFinal'):
            assert col in df.columns, f"Missing column: {col}"

    def test_query_sspp_stars_filters_low_snr(self):
        """Stars with S/N below min_snr threshold are excluded."""
        from src.sdss_loader import SDSSLoader

        table = _make_sspp_table(n=6)
        # Force 3 rows below S/N threshold
        table['snMedian'][:3] = 5.0

        with patch('src.sdss_loader.SDSS.query_sql', return_value=table):
            loader = SDSSLoader()
            df = loader.query_sspp_stars(n_stars=6, min_snr=10)

        assert len(df) == 3
        assert (df['snMedian'] >= 10).all()

    def test_query_returns_empty_df_when_no_results(self):
        """Returns an empty DataFrame gracefully when CAS returns None."""
        from src.sdss_loader import SDSSLoader

        with patch('src.sdss_loader.SDSS.query_sql', return_value=None):
            loader = SDSSLoader()
            df = loader.query_sspp_stars(n_stars=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    # ------------------------------------------------------------------
    # get_spectrum_from_fits
    # ------------------------------------------------------------------

    def test_get_spectrum_returns_three_arrays(self):
        """get_spectrum_from_fits() returns (wavelength, flux, ivar) arrays."""
        from src.sdss_loader import SDSSLoader

        hdul, loglam, flux, ivar = _make_fits_hdul()
        mock_speclist = [hdul]

        with patch('src.sdss_loader.SDSS.get_spectra', return_value=mock_speclist):
            loader = SDSSLoader()
            wavelength_out, flux_out, ivar_out = loader.get_spectrum_from_fits(
                plate=3586, mjd=55181, fiberid=1
            )

        assert wavelength_out.shape == flux_out.shape == ivar_out.shape
        assert wavelength_out.ndim == 1

    def test_get_spectrum_wavelength_is_linear_angstroms(self):
        """Returned wavelength array is in linear Angstroms (not log)."""
        from src.sdss_loader import SDSSLoader

        hdul, loglam, _, _ = _make_fits_hdul()
        with patch('src.sdss_loader.SDSS.get_spectra', return_value=[hdul]):
            loader = SDSSLoader()
            wavelength_out, _, _ = loader.get_spectrum_from_fits(3586, 55181, 1)

        expected_min = 10 ** loglam.min()
        expected_max = 10 ** loglam.max()
        assert wavelength_out.min() == pytest.approx(expected_min, rel=1e-3)
        assert wavelength_out.max() == pytest.approx(expected_max, rel=1e-3)

    def test_get_spectrum_returns_float32(self):
        """Flux and ivar arrays are float32."""
        from src.sdss_loader import SDSSLoader

        hdul, _, _, _ = _make_fits_hdul()
        with patch('src.sdss_loader.SDSS.get_spectra', return_value=[hdul]):
            loader = SDSSLoader()
            _, flux_out, ivar_out = loader.get_spectrum_from_fits(3586, 55181, 1)

        assert flux_out.dtype == np.float32
        assert ivar_out.dtype == np.float32

    def test_get_spectrum_raises_on_empty_result(self):
        """Raises ValueError when SDSS returns no spectra for given IDs."""
        from src.sdss_loader import SDSSLoader

        with patch('src.sdss_loader.SDSS.get_spectra', return_value=None):
            loader = SDSSLoader()
            with pytest.raises(ValueError, match="No spectrum"):
                loader.get_spectrum_from_fits(plate=9999, mjd=99999, fiberid=1)

    # ------------------------------------------------------------------
    # build_dataset
    # ------------------------------------------------------------------

    def test_build_dataset_returns_spectra_and_labels(self, tmp_path):
        """build_dataset() returns (spectra_array, labels_dict)."""
        from src.sdss_loader import SDSSLoader
        from src.preprocessing import SDSS_WAVE_GRID

        n = 4
        mock_table = _make_sspp_table(n=n)
        hdul, _, _, _ = _make_fits_hdul(n_pixels=len(SDSS_WAVE_GRID))

        with patch('src.sdss_loader.SDSS.query_sql', return_value=mock_table), \
             patch('src.sdss_loader.SDSS.get_spectra', return_value=[hdul]):
            loader = SDSSLoader()
            spectra, labels = loader.build_dataset(n_stars=n, cache_dir=str(tmp_path))

        assert isinstance(spectra, np.ndarray)
        assert spectra.ndim == 2
        assert spectra.shape[1] == len(SDSS_WAVE_GRID)
        assert 'temperature' in labels
        assert 'composition' in labels
        assert 'radial_velocity' in labels

    def test_build_dataset_label_lengths_match_spectra(self, tmp_path):
        """Labels have same number of samples as spectra rows."""
        from src.sdss_loader import SDSSLoader
        from src.preprocessing import SDSS_WAVE_GRID

        n = 4
        mock_table = _make_sspp_table(n=n)
        hdul, _, _, _ = _make_fits_hdul(n_pixels=len(SDSS_WAVE_GRID))

        with patch('src.sdss_loader.SDSS.query_sql', return_value=mock_table), \
             patch('src.sdss_loader.SDSS.get_spectra', return_value=[hdul]):
            loader = SDSSLoader()
            spectra, labels = loader.build_dataset(n_stars=n, cache_dir=str(tmp_path))

        n_samples = spectra.shape[0]
        for key, arr in labels.items():
            assert len(arr) == n_samples, f"Label '{key}' length mismatch"

    def test_build_dataset_spectra_dtype_is_float32(self, tmp_path):
        """Spectra returned by build_dataset are float32."""
        from src.sdss_loader import SDSSLoader
        from src.preprocessing import SDSS_WAVE_GRID

        n = 3
        mock_table = _make_sspp_table(n=n)
        hdul, _, _, _ = _make_fits_hdul(n_pixels=len(SDSS_WAVE_GRID))

        with patch('src.sdss_loader.SDSS.query_sql', return_value=mock_table), \
             patch('src.sdss_loader.SDSS.get_spectra', return_value=[hdul]):
            loader = SDSSLoader()
            spectra, _ = loader.build_dataset(n_stars=n, cache_dir=str(tmp_path))

        assert spectra.dtype == np.float32


# ---------------------------------------------------------------------------
# TestLoadSdssSpectra (integration shim in data.py)
# ---------------------------------------------------------------------------

class TestLoadSdssSpectra:
    """Tests for the updated load_sdss_spectra() function in data.py."""

    def test_load_sdss_spectra_delegates_to_loader(self, tmp_path):
        """load_sdss_spectra() returns (spectra, labels) without raising."""
        from src.data import load_sdss_spectra
        from src.preprocessing import SDSS_WAVE_GRID

        n = 3
        mock_table = _make_sspp_table(n=n)
        hdul, _, _, _ = _make_fits_hdul(n_pixels=len(SDSS_WAVE_GRID))

        with patch('src.sdss_loader.SDSS.query_sql', return_value=mock_table), \
             patch('src.sdss_loader.SDSS.get_spectra', return_value=[hdul]):
            spectra, labels = load_sdss_spectra(
                cache_dir=str(tmp_path), n_stars=n
            )

        assert spectra.shape[0] == n
        assert isinstance(labels, dict)

    def test_load_sdss_spectra_no_longer_raises(self, tmp_path):
        """The old NotImplementedError stub is gone."""
        from src.data import load_sdss_spectra
        from src.preprocessing import SDSS_WAVE_GRID

        n = 2
        mock_table = _make_sspp_table(n=n)
        hdul, _, _, _ = _make_fits_hdul(n_pixels=len(SDSS_WAVE_GRID))

        with patch('src.sdss_loader.SDSS.query_sql', return_value=mock_table), \
             patch('src.sdss_loader.SDSS.get_spectra', return_value=[hdul]):
            # Must NOT raise NotImplementedError
            spectra, labels = load_sdss_spectra(
                cache_dir=str(tmp_path), n_stars=n
            )
        assert spectra is not None
