"""Physically motivated augmentations for spectra.

The hardware will see effects we cannot model with infinite fidelity in
:mod:`regoscan.synth`, so we additionally augment training spectra at load
time. Each augmentation is a function ``(spectrum, rng) -> spectrum`` that
preserves shape ``(K,)`` and clips to a sane range.

Augmentations are deliberately *small* and *physical*. Aggressive
augmentation tends to wash out the absorption features the model needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Atomic augmentations
# ---------------------------------------------------------------------------


def add_gaussian_noise(
    spectrum: np.ndarray, rng: np.random.Generator, sigma: float = 0.005
) -> np.ndarray:
    """Per-channel Gaussian noise — represents residual readout noise."""
    return spectrum + rng.normal(0.0, sigma, size=spectrum.shape)


def intensity_jitter(
    spectrum: np.ndarray, rng: np.random.Generator, sigma: float = 0.05
) -> np.ndarray:
    """Multiplicative global intensity scaling — packing/standoff variation."""
    scale = 1.0 + rng.normal(0.0, sigma)
    return spectrum * scale


def baseline_shift(
    spectrum: np.ndarray, rng: np.random.Generator, amp: float = 0.02
) -> np.ndarray:
    """Smooth additive baseline drift (low-order polynomial)."""
    K = spectrum.shape[-1]
    x = np.linspace(-1.0, 1.0, K)
    coeffs = rng.uniform(-1.0, 1.0, size=4)
    poly = coeffs[0] + coeffs[1] * x + coeffs[2] * x**2 + coeffs[3] * x**3
    poly = poly / (np.max(np.abs(poly)) + 1e-12)
    return spectrum + amp * poly


def channel_dropout(
    spectrum: np.ndarray, rng: np.random.Generator, p: float = 0.01
) -> np.ndarray:
    """Zero out a small fraction of channels.

    Models a flaky pixel or a single LED outage. ``p`` is per-channel.
    """
    if p <= 0:
        return spectrum.copy()
    mask = rng.uniform(0.0, 1.0, size=spectrum.shape) >= p
    return spectrum * mask


def spectral_mix(
    spectra: np.ndarray,
    alpha_range: tuple[float, float] = (0.2, 0.8),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mix each spectrum with a randomly chosen partner from the batch.

    For each row *i*, draw ``alpha ~ Uniform(alpha_range)`` and a random
    partner index *j != i* (or *j* if ``N == 1``).  The output row is::

        mixed_i = alpha * spectra[i] + (1 - alpha) * spectra[j]

    This simulates mixed-mineral measurement spots and helps the model
    generalise across mineral transitions.

    Parameters
    ----------
    spectra:
        2-D array of shape ``(N, K)`` — a batch of spectra.
    alpha_range:
        Inclusive bounds for the mixing coefficient.
    rng:
        Numpy random generator; one is created if *None*.

    Returns
    -------
    np.ndarray
        Mixed batch with the same shape as *spectra*.
    """
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be 2-D, got shape {spectra.shape}")
    rng = rng or np.random.default_rng()
    N, K = spectra.shape
    out = np.empty_like(spectra, dtype=np.float64)
    for i in range(N):
        alpha = rng.uniform(alpha_range[0], alpha_range[1])
        if N > 1:
            j = rng.integers(0, N - 1)
            if j >= i:
                j += 1
        else:
            j = 0
        out[i] = alpha * spectra[i] + (1.0 - alpha) * spectra[j]
    return out


def wavelength_shift(
    spectrum: np.ndarray, rng: np.random.Generator, max_shift_px: int = 1
) -> np.ndarray:
    """Tiny integer-pixel shift along the spectral axis.

    Represents a small wavelength-calibration error. Edge values are
    repeated rather than wrapped.
    """
    if max_shift_px <= 0:
        return spectrum.copy()
    shift = int(rng.integers(-max_shift_px, max_shift_px + 1))
    if shift == 0:
        return spectrum.copy()
    out = np.roll(spectrum, shift, axis=-1)
    if shift > 0:
        out[..., :shift] = spectrum[..., :1]
    else:
        out[..., shift:] = spectrum[..., -1:]
    return out


# ---------------------------------------------------------------------------
# Composite pipeline
# ---------------------------------------------------------------------------


@dataclass
class AugmentConfig:
    """Knobs for the default training-time augmentation pipeline."""

    p_apply: float = 0.7           # probability the whole pipeline runs
    gaussian_sigma: float = 0.005
    intensity_sigma: float = 0.04
    baseline_amp: float = 0.015
    dropout_p: float = 0.005
    max_shift_px: int = 1
    p_spectral_mix: float = 0.3    # fraction of samples mixed (batch-level)
    mix_alpha_range: tuple[float, float] = (0.2, 0.8)
    clip_lo: float = 0.0
    clip_hi: float = 1.5


def augment_spectrum(
    spectrum: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig | None = None,
) -> np.ndarray:
    """Apply the full augmentation pipeline to a single spectrum.

    With probability ``1 - cfg.p_apply`` the spectrum is returned unchanged
    (so the model still sees clean examples).
    """
    cfg = cfg or AugmentConfig()
    if rng.uniform(0.0, 1.0) >= cfg.p_apply:
        return np.clip(spectrum.copy(), cfg.clip_lo, cfg.clip_hi)
    out = spectrum.astype(np.float64, copy=True)
    out = intensity_jitter(out, rng, sigma=cfg.intensity_sigma)
    out = baseline_shift(out, rng, amp=cfg.baseline_amp)
    out = wavelength_shift(out, rng, max_shift_px=cfg.max_shift_px)
    out = add_gaussian_noise(out, rng, sigma=cfg.gaussian_sigma)
    out = channel_dropout(out, rng, p=cfg.dropout_p)
    return np.clip(out, cfg.clip_lo, cfg.clip_hi)


def augment_batch(
    spectra: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig | None = None,
) -> np.ndarray:
    """Apply :func:`augment_spectrum` independently to each row of a (N, K) batch.

    When :attr:`AugmentConfig.p_spectral_mix` is > 0, a random subset of
    rows is replaced by a linear mix with another randomly chosen row
    *before* the per-spectrum augmentation pipeline runs.
    """
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be 2-D, got shape {spectra.shape}")
    cfg = cfg or AugmentConfig()

    # --- batch-level: spectral mixing on a random subset ----------------
    work = spectra.astype(np.float64, copy=True)
    if cfg.p_spectral_mix > 0 and work.shape[0] > 1:
        n_mix = int(np.round(cfg.p_spectral_mix * work.shape[0]))
        n_mix = max(0, min(n_mix, work.shape[0]))
        if n_mix > 0:
            idx = rng.choice(work.shape[0], size=n_mix, replace=False)
            subset = work[idx]
            mixed = spectral_mix(subset, alpha_range=cfg.mix_alpha_range, rng=rng)
            work[idx] = mixed

    # --- per-spectrum augmentation pipeline ------------------------------
    out = np.empty_like(work)
    for i in range(work.shape[0]):
        out[i] = augment_spectrum(work[i], rng, cfg)
    return out


__all__ = [
    "add_gaussian_noise",
    "intensity_jitter",
    "baseline_shift",
    "channel_dropout",
    "spectral_mix",
    "wavelength_shift",
    "AugmentConfig",
    "augment_spectrum",
    "augment_batch",
]
