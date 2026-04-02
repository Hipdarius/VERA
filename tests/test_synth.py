"""Tests for the synthetic measurement generator.

These tests do NOT require the cached endmember .npz on disk — they build a
small in-memory Endmembers fixture that mirrors what
``scripts/download_usgs.py`` produces. This keeps the test suite hermetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from regoscan.io_csv import validate_dataframe, write_measurements_csv
from regoscan.schema import (
    LIF_COL,
    MINERAL_CLASSES,
    N_LED,
    N_SPEC,
    WAVELENGTHS,
    Measurement,
)
from regoscan.synth import (
    ENDMEMBER_INDEX,
    ENDMEMBER_NAMES,
    Endmembers,
    NoiseConfig,
    fractions_for_class,
    mixture_spectrum,
    synth_dataset,
    synth_measurement,
    synth_sample,
)


# ---------------------------------------------------------------------------
# Fixture: in-memory endmembers
# ---------------------------------------------------------------------------


def _toy_endmembers() -> Endmembers:
    """Build endmembers with the qualitative right shape but cheap to compute."""
    lam = WAVELENGTHS
    x = (lam - lam.min()) / (lam.max() - lam.min())
    olivine = 0.20 + 0.60 * x
    pyroxene = 0.15 + 0.50 * x
    anorthite = 0.55 + 0.30 * x
    ilmenite = 0.05 + 0.05 * x
    spectra = np.stack([olivine, pyroxene, anorthite, ilmenite], axis=0)
    return Endmembers(wavelengths_nm=lam, spectra=spectra, source="toy")


@pytest.fixture
def endmembers() -> Endmembers:
    return _toy_endmembers()


# ---------------------------------------------------------------------------
# fractions_for_class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("klass", MINERAL_CLASSES)
def test_fractions_sum_to_one(klass: str):
    rng = np.random.default_rng(42)
    f = fractions_for_class(klass, rng)
    assert f.shape == (4,)
    assert (f >= 0).all()
    assert f.sum() == pytest.approx(1.0)


def test_ilmenite_rich_has_high_ilmenite():
    rng = np.random.default_rng(0)
    fs = [fractions_for_class("ilmenite_rich", rng) for _ in range(20)]
    arr = np.stack(fs, axis=0)
    assert (arr[:, ENDMEMBER_INDEX["ilmenite"]] >= 0.30).all()


def test_anorthositic_has_low_ilmenite():
    rng = np.random.default_rng(1)
    fs = [fractions_for_class("anorthositic", rng) for _ in range(20)]
    arr = np.stack(fs, axis=0)
    assert (arr[:, ENDMEMBER_INDEX["ilmenite"]] <= 0.05).all()


def test_unknown_class_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        fractions_for_class("garnet_rich", rng)


# ---------------------------------------------------------------------------
# mixture_spectrum
# ---------------------------------------------------------------------------


def test_mixture_at_a_pure_endmember(endmembers):
    f = np.array([0.0, 0.0, 1.0, 0.0])  # pure anorthite
    mix = mixture_spectrum(f, endmembers)
    np.testing.assert_allclose(mix, endmembers.spectra[ENDMEMBER_INDEX["anorthite"]])


def test_mixture_is_linear(endmembers):
    f1 = np.array([0.5, 0.5, 0.0, 0.0])
    f2 = np.array([0.0, 0.0, 1.0, 0.0])
    half = 0.5 * mixture_spectrum(f1, endmembers) + 0.5 * mixture_spectrum(f2, endmembers)
    blend = mixture_spectrum(0.5 * f1 + 0.5 * f2, endmembers)
    np.testing.assert_allclose(half, blend)


# ---------------------------------------------------------------------------
# synth_measurement
# ---------------------------------------------------------------------------


def test_single_measurement_validates(endmembers):
    rng = np.random.default_rng(7)
    f = fractions_for_class("olivine_rich", rng)
    m = synth_measurement(
        sample_id="S0",
        mineral_class="olivine_rich",
        fractions=f,
        endmembers=endmembers,
        rng=rng,
    )
    assert isinstance(m, Measurement)
    assert len(m.spec) == N_SPEC
    assert len(m.led) == N_LED
    assert 0.0 <= m.ilmenite_fraction <= 1.0


def test_lif_low_when_ilmenite_high(endmembers):
    rng = np.random.default_rng(11)
    # build directly: pure ilmenite vs pure anorthite
    f_ilm = np.array([0.0, 0.0, 0.0, 1.0])
    f_an = np.array([0.0, 0.0, 1.0, 0.0])
    lifs_ilm = []
    lifs_an = []
    noise = NoiseConfig()
    for _ in range(30):
        lifs_ilm.append(
            synth_measurement(
                "S_i", "ilmenite_rich", f_ilm, endmembers, rng, noise=noise
            ).lif_450lp
        )
        lifs_an.append(
            synth_measurement(
                "S_a", "anorthositic", f_an, endmembers, rng, noise=noise
            ).lif_450lp
        )
    assert np.mean(lifs_ilm) < np.mean(lifs_an), \
        "ilmenite-pure samples must have lower LIF than anorthite-pure"
    # ilmenite LIF should be very small in absolute terms
    assert np.mean(lifs_ilm) < 0.1


def test_measurements_for_same_sample_share_ilmenite_fraction(endmembers):
    rng = np.random.default_rng(3)
    measurements = synth_sample(
        "S001",
        "pyroxene_rich",
        endmembers,
        n_measurements=5,
        rng=rng,
    )
    fractions = {m.ilmenite_fraction for m in measurements}
    # Composition is fixed per sample, so ilmenite_fraction is identical.
    assert len(fractions) == 1


def test_repeated_seed_is_deterministic(endmembers):
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    m_a = synth_sample("S", "mixed", endmembers, n_measurements=2, rng=rng_a)
    m_b = synth_sample("S", "mixed", endmembers, n_measurements=2, rng=rng_b)
    for a, b in zip(m_a, m_b):
        assert a.spec == b.spec
        assert a.led == b.led
        assert a.lif_450lp == b.lif_450lp


# ---------------------------------------------------------------------------
# synth_dataset
# ---------------------------------------------------------------------------


def test_dataset_balanced_class_round_robin(endmembers):
    out = synth_dataset(
        endmembers,
        n_samples=10,
        measurements_per_sample=3,
        seed=0,
    )
    assert len(out) == 30
    classes = [m.mineral_class for m in out]
    # 5 classes, 10 samples => each class appears in exactly 2 samples => 6 measurements each
    from collections import Counter
    counts = Counter(classes)
    assert set(counts.values()) == {6}


def test_dataset_round_trips_through_csv(tmp_path, endmembers):
    out = synth_dataset(
        endmembers,
        n_samples=5,
        measurements_per_sample=2,
        seed=1,
    )
    # write/read with the canonical CSV path; this also runs validation
    p = tmp_path / "synth_smoke.csv"
    write_measurements_csv(out, p)
    import pandas as pd
    df = pd.read_csv(p)
    validate_dataframe(df)
    assert len(df) == 10
    # the LIF column must be present and finite
    assert df[LIF_COL].notna().all()
