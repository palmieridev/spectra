"""SDSS spectroscopic data loader.

Provides :class:`SDSSLoader`, which wraps ``astroquery.sdss.SDSS`` to:

1. Query the SDSS CAS for SEGUE/SSPP stellar parameters
   (Teff, [Fe/H], log g, radial velocity).
2. Download BOSS spectra in parallel using a thread pool.
3. Run the full preprocessing pipeline on each spectrum.
4. Return arrays ready for TensorFlow training.

Usage::

    from src.sdss_loader import SDSSLoader

    loader = SDSSLoader()
    spectra, labels = loader.build_dataset(n_stars=500, cache_dir='data/raw')
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from astropy.table import Table
from astroquery.sdss import SDSS

from src.preprocessing import preprocess_spectrum, SDSS_WAVE_GRID

logger = logging.getLogger(__name__)

# Joins SpecObj with sppParams (SSPP stellar parameters).
# NEWID() produces a random ordering so repeated calls return different subsets.
_SSPP_QUERY_TEMPLATE = """
SELECT TOP {n_stars}
    sp.plate,
    sp.mjd,
    sp.fiberID,
    pp.Teff,
    pp.feh,
    pp.logg,
    pp.elodiervFinal,
    sp.snMedian
FROM SpecObj AS sp
JOIN sppParams AS pp
    ON sp.specobjid = pp.specobjid
WHERE
    pp.Teff     BETWEEN 4000 AND 10000
    AND pp.feh  BETWEEN -4.0  AND  1.0
    AND pp.logg BETWEEN 0.0   AND  5.5
    AND sp.snMedian >= {min_snr}
    AND sp.zWarning = 0
ORDER BY NEWID()
""".strip()

# Sentinel returned when no spectra could be loaded.
_EMPTY_LABELS: dict[str, np.ndarray] = {
    'temperature':    np.array([], dtype=np.float64),
    'composition':    np.array([], dtype=np.float64),
    'radial_velocity': np.array([], dtype=np.float64),
}


class SDSSLoader:
    """Download and preprocess SDSS stellar spectra.

    Args:
        wave_grid:   Target wavelength grid (Å).  Defaults to
                     :data:`~src.preprocessing.SDSS_WAVE_GRID`.
        max_workers: Thread-pool size for parallel spectrum downloads.
    """

    def __init__(
        self,
        wave_grid: np.ndarray = None,
        max_workers: int = 8,
    ):
        self.wave_grid = wave_grid if wave_grid is not None else SDSS_WAVE_GRID
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query_sspp_stars(
        self,
        n_stars: int = 1000,
        min_snr: float = 10.0,
    ) -> pd.DataFrame:
        """Query the SDSS CAS for SEGUE stars with SSPP stellar parameters.

        Args:
            n_stars: Maximum number of stars to retrieve.
            min_snr: Minimum median S/N per pixel.

        Returns:
            DataFrame with columns ``plate``, ``mjd``, ``fiberID``, ``Teff``,
            ``feh``, ``logg``, ``elodiervFinal``, ``snMedian``.
            Empty DataFrame when the CAS returns no results.
        """
        sql = _SSPP_QUERY_TEMPLATE.format(n_stars=n_stars, min_snr=min_snr)
        logger.info("Querying SDSS CAS for %d SSPP stars (S/N ≥ %.1f)…", n_stars, min_snr)

        result: Table | None = SDSS.query_sql(sql)

        if result is None or len(result) == 0:
            logger.warning("SDSS query returned no results.")
            return pd.DataFrame()

        df = result.to_pandas()
        # Secondary Python-side filter guards against mock tables that skip the SQL WHERE.
        df = df[df['snMedian'] >= min_snr].reset_index(drop=True)
        logger.info("Retrieved %d stars after S/N filter.", len(df))
        return df

    def get_spectrum_from_fits(
        self,
        plate: int,
        mjd: int,
        fiberid: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fetch a single SDSS spectrum and return (wavelength, flux, ivar).

        Args:
            plate:   SDSS plate number.
            mjd:     Modified Julian Date of the observation.
            fiberid: Fiber identifier (1-based).

        Returns:
            Tuple of float32 arrays ``(wavelength_Å, flux, ivar)``.

        Raises:
            ValueError: When astroquery returns no spectra for this triplet.
        """
        speclist = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiberid)

        if speclist is None or len(speclist) == 0:
            raise ValueError(
                f"No spectrum found for plate={plate}, mjd={mjd}, fiberID={fiberid}"
            )

        data = speclist[0][1].data
        # loglam is log₁₀(wavelength / Å); convert to linear Angstroms.
        wavelength = (10.0 ** data['loglam'].astype(np.float64)).astype(np.float32)
        return wavelength, data['flux'].astype(np.float32), data['ivar'].astype(np.float32)

    def build_dataset(
        self,
        n_stars: int = 1000,
        min_snr: float = 10.0,
        cache_dir: str = 'data/raw',
    ) -> Tuple[np.ndarray, dict]:
        """Orchestrate the full pipeline: query → parallel download → preprocess.

        Spectra are downloaded in parallel using :attr:`max_workers` threads.
        Rows that fail to download are logged and skipped.

        Args:
            n_stars:   Target number of spectra.
            min_snr:   Minimum median S/N forwarded to :meth:`query_sspp_stars`.
            cache_dir: Directory for raw data / astroquery cache (created if absent).

        Returns:
            Tuple ``(spectra, labels)`` where *spectra* is float32
            ``(n_loaded, n_wavelengths)`` and *labels* is a dict with keys
            ``'temperature'``, ``'composition'``, ``'radial_velocity'``.
        """
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        catalog = self.query_sspp_stars(n_stars=n_stars, min_snr=min_snr)
        if catalog.empty:
            return np.empty((0, len(self.wave_grid)), dtype=np.float32), _EMPTY_LABELS.copy()

        def _fetch_and_preprocess(row: pd.Series):
            """Download + preprocess one spectrum; return (preprocessed, row) or raise."""
            wavelength, flux, ivar = self.get_spectrum_from_fits(
                plate=int(row['plate']),
                mjd=int(row['mjd']),
                fiberid=int(row['fiberID']),
            )
            return preprocess_spectrum(wavelength, flux, ivar, target_grid=self.wave_grid), row

        results: list[tuple[np.ndarray, pd.Series]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(_fetch_and_preprocess, row): row
                for _, row in catalog.iterrows()
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Skipping plate=%s mjd=%s fiberID=%s — %s",
                        row['plate'], row['mjd'], row['fiberID'], exc,
                    )

        if not results:
            return np.empty((0, len(self.wave_grid)), dtype=np.float32), _EMPTY_LABELS.copy()

        spectra_arrays, rows = zip(*results)
        spectra = np.stack(spectra_arrays, axis=0)
        labels = {
            'temperature':    np.array([float(r['Teff'])            for r in rows]),
            'composition':    np.array([float(r['feh'])             for r in rows]),
            'radial_velocity': np.array([float(r['elodiervFinal'])  for r in rows]),
        }

        logger.info("Dataset built: %d spectra × %d bins.", spectra.shape[0], spectra.shape[1])
        return spectra, labels


def load_sdss_spectra(
    cache_dir: str = 'data/raw',
    n_stars: int = 1000,
    min_snr: float = 10.0,
) -> Tuple[np.ndarray, dict]:
    """Load SDSS spectra and SSPP labels ready for training.

    Args:
        cache_dir: Directory for raw data / astroquery cache.
        n_stars:   Number of spectra to download.
        min_snr:   Minimum median S/N per pixel.

    Returns:
        ``(spectra, labels)`` — see :meth:`SDSSLoader.build_dataset`.
    """
    return SDSSLoader().build_dataset(n_stars=n_stars, min_snr=min_snr, cache_dir=cache_dir)
