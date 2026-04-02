"""Tests for the augmentation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from regoscan.augment import (
    AugmentConfig,
    add_gaussian_noise,
    augment_batch,
    augment_spectrum,
    baseline_shift,
    channel_dropout,
    intensity_jitter,
    wavelength_shift,
)
from regoscan.schema import N_SPEC


@pytest.fixture
def base_spectrum() -> np.ndarray:
    return np.linspace(0.3, 0.7, N_SPEC)


def test_gaussian_noise_changes_values_but_keeps_shape(base_spectrum):
    rng = np.random.default_rng(0)
    out = add_gaussian_noise(base_spectrum, rng, sigma=0.01)
    assert out.shape == base_spectrum.shape
    assert not np.allclose(out, base_spectrum)


def test_intensity_jitter_is_multiplicative(base_spectrum):
    rng = np.random.default_rng(0)
    out = intensity_jitter(base_spectrum, rng, sigma=0.05)
    ratio = out / base_spectrum
    # ratio should be (almost) constant across channels
    assert np.std(ratio) < 1e-12


def test_baseline_shift_is_additive_smooth(base_spectrum):
    rng = np.random.default_rng(1)
    out = baseline_shift(base_spectrum, rng, amp=0.05)
    diff = out - base_spectrum
    # diff is a smooth polynomial — second-order differences should be tiny
    assert np.max(np.abs(np.diff(diff, n=2))) < 0.01


def test_channel_dropout_zeros_some_channels():
    rng = np.random.default_rng(2)
    spec = np.ones(2000)
    out = channel_dropout(spec, rng, p=0.05)
    n_zero = int(np.sum(out == 0.0))
    # roughly 5% of channels zeroed; allow generous slack
    assert 30 <= n_zero <= 200


def test_channel_dropout_p_zero_is_identity(base_spectrum):
    rng = np.random.default_rng(0)
    out = channel_dropout(base_spectrum, rng, p=0.0)
    np.testing.assert_array_equal(out, base_spectrum)


def test_wavelength_shift_one_pixel(base_spectrum):
    rng = np.random.default_rng(3)
    out = wavelength_shift(base_spectrum, rng, max_shift_px=1)
    assert out.shape == base_spectrum.shape
    # shift is in {-1, 0, 1}; either way values are nearly the same elsewhere
    assert np.max(np.abs(out - base_spectrum)) < 0.05


def test_augment_spectrum_full_pipeline(base_spectrum):
    rng = np.random.default_rng(7)
    cfg = AugmentConfig(p_apply=1.0)  # always apply
    out = augment_spectrum(base_spectrum, rng, cfg)
    assert out.shape == base_spectrum.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0) and np.all(out <= 1.5)


def test_augment_spectrum_p_zero_is_clip_only(base_spectrum):
    rng = np.random.default_rng(7)
    cfg = AugmentConfig(p_apply=0.0)
    out = augment_spectrum(base_spectrum, rng, cfg)
    np.testing.assert_allclose(out, np.clip(base_spectrum, 0.0, 1.5))


def test_augment_batch_shape():
    rng = np.random.default_rng(0)
    x = np.tile(np.linspace(0.3, 0.7, N_SPEC), (5, 1))
    out = augment_batch(x, rng, AugmentConfig(p_apply=1.0))
    assert out.shape == x.shape
    # at least one row must differ from the input
    assert not np.allclose(out, x)


def test_augment_batch_rejects_wrong_dims():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        augment_batch(np.zeros(N_SPEC), rng, AugmentConfig())
