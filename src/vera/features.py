"""Hand-crafted features used by the PLSR/RandomForest baselines.

These are *not* fed to the CNN — the CNN consumes the full preprocessed
spectrum directly. They exist so the baseline has a fighting chance with
small synthetic datasets.

Each feature is computed from one or more of:
  - the full 288-channel spectrum (after preprocessing)
  - the 12 LED narrowband reflectances
  - the LIF photodiode value

Features:
  * broad_albedo            mean reflectance over 500–800 nm
  * vis_red_slope           slope (per-nm) of a 1st-order fit on 500–800 nm
  * uv_blue_drop            mean(340–420) / mean(500–600)
  * band_depth_~700         continuum-removed depth around 700 nm
  * band_depth_~620         continuum-removed depth around 620 nm
  * led_red_blue_ratio      led_660 / led_450
  * led_nir_vis_ratio       mean(led_730, led_780, led_850) / mean(led_450, led_525)
  * lif_normalised          lif_450lp / (broad_albedo + eps)
"""

from __future__ import annotations

import numpy as np

from vera.preprocess import continuum_removal_batch
from vera.schema import LED_WAVELENGTHS_NM, WAVELENGTHS

EPS = 1e-6

# Pre-compute index masks once.
_VIS_BAND = (WAVELENGTHS >= 500) & (WAVELENGTHS <= 800)
_UV_BAND = (WAVELENGTHS >= 340) & (WAVELENGTHS <= 420)
_GREEN_BAND = (WAVELENGTHS >= 500) & (WAVELENGTHS <= 600)
_BAND_700 = (WAVELENGTHS >= 680) & (WAVELENGTHS <= 720)
_BAND_620 = (WAVELENGTHS >= 600) & (WAVELENGTHS <= 640)

LED_INDEX = {w: i for i, w in enumerate(LED_WAVELENGTHS_NM)}

FEATURE_NAMES: tuple[str, ...] = (
    "broad_albedo",
    "vis_red_slope",
    "uv_blue_drop",
    "band_depth_700",
    "band_depth_620",
    "led_red_blue_ratio",
    "led_nir_vis_ratio",
    "lif_normalised",
)
N_FEATURES: int = len(FEATURE_NAMES)


def _broad_albedo(spectra: np.ndarray) -> np.ndarray:
    return spectra[:, _VIS_BAND].mean(axis=1)


def _vis_red_slope(spectra: np.ndarray) -> np.ndarray:
    lam = WAVELENGTHS[_VIS_BAND]
    block = spectra[:, _VIS_BAND]
    lam_c = lam - lam.mean()
    denom = (lam_c**2).sum()
    return (block - block.mean(axis=1, keepdims=True)) @ lam_c / denom


def _uv_blue_drop(spectra: np.ndarray) -> np.ndarray:
    uv = spectra[:, _UV_BAND].mean(axis=1)
    green = spectra[:, _GREEN_BAND].mean(axis=1)
    return uv / (green + EPS)


def _band_depth(spectra_cr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Depth = 1 - min(continuum_removed) inside ``mask``."""
    return 1.0 - spectra_cr[:, mask].min(axis=1)


def compute_features(
    spectra: np.ndarray,
    leds: np.ndarray,
    lif: np.ndarray,
) -> np.ndarray:
    """Stack all hand-crafted features into a single (N, F) matrix.

    ``spectra`` is the (N, 288) reflectance block (already dark/white
    normalised). ``leds`` is (N, 12), ``lif`` is (N,).
    """
    spectra = np.asarray(spectra, dtype=np.float64)
    leds = np.asarray(leds, dtype=np.float64)
    lif = np.asarray(lif, dtype=np.float64).ravel()

    if spectra.ndim != 2 or spectra.shape[1] != WAVELENGTHS.size:
        raise ValueError(f"spectra must be (N, {WAVELENGTHS.size}); got {spectra.shape}")
    if leds.shape != (spectra.shape[0], len(LED_WAVELENGTHS_NM)):
        raise ValueError(f"leds must be (N, {len(LED_WAVELENGTHS_NM)})")
    if lif.shape != (spectra.shape[0],):
        raise ValueError(f"lif must be (N,) matching spectra")

    albedo = _broad_albedo(spectra)
    slope = _vis_red_slope(spectra)
    uv = _uv_blue_drop(spectra)

    cr = continuum_removal_batch(spectra, WAVELENGTHS)
    bd700 = _band_depth(cr, _BAND_700)
    bd620 = _band_depth(cr, _BAND_620)

    led_red = leds[:, LED_INDEX[660]]
    led_blue = leds[:, LED_INDEX[450]]
    led_red_blue = led_red / (led_blue + EPS)

    nir_idx = [LED_INDEX[w] for w in (730, 780, 850)]
    vis_idx = [LED_INDEX[w] for w in (450, 525)]
    led_nir_vis = leds[:, nir_idx].mean(axis=1) / (leds[:, vis_idx].mean(axis=1) + EPS)

    lif_norm = lif / (albedo + EPS)

    feats = np.stack(
        [albedo, slope, uv, bd700, bd620, led_red_blue, led_nir_vis, lif_norm],
        axis=1,
    )
    return feats


__all__ = [
    "FEATURE_NAMES",
    "N_FEATURES",
    "compute_features",
]
