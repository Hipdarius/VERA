"""Synthetic Regoscan measurement generator.

Builds physically motivated synthetic spectra from a small set of mineral
endmembers (cached by ``scripts/download_usgs.py``). The pipeline only
exercises model wiring with these — once real hardware data arrives it
flows through the same canonical CSV with no code changes.

Physical model
--------------
A *sample* has a fixed mineral mass-fraction vector (olivine, pyroxene,
anorthite, ilmenite). The full-spectrum reflectance is the linear mixture
of endmembers weighted by mass fraction. We then generate multiple
*measurements* per sample by adding:

1. Per-channel multiplicative gain variation (small) — represents pixel
   responsivity and small temperature drifts.
2. Slow additive baseline drift modelled as a low-order polynomial — stray
   light, ambient IR, integration-window leakage.
3. Illumination intensity scaling — sample packing density and probe-to-
   surface standoff change overall brightness.
4. Shot noise (Poisson-like, scaled with sqrt(signal * integration_time)).
5. LED narrowband channels are derived by integrating the full spectrum
   under each LED's emission band, then perturbed with the same noise
   model at coarser resolution.
6. The LIF channel scales with ``(1 - ilmenite_fraction) ** 1.5`` plus a
   per-mineral fluorescence efficiency, plus noise. Ilmenite quenches
   fluorescence under 405 nm excitation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from regoscan.schema import (
    LED_WAVELENGTHS_NM,
    MINERAL_CLASSES,
    Measurement,
    N_LED,
    N_SPEC,
    PACKING_DENSITIES,
    WAVELENGTHS,
)

ENDMEMBER_NAMES: tuple[str, ...] = ("olivine", "pyroxene", "anorthite", "ilmenite")
ENDMEMBER_INDEX: dict[str, int] = {n: i for i, n in enumerate(ENDMEMBER_NAMES)}

# Per-mineral fluorescence efficiency at 450 nm under 405 nm excitation.
# Anorthite (Ca-plagioclase) is the brightest in the lunar context; ilmenite
# fully quenches.
LIF_EFFICIENCY: dict[str, float] = {
    "olivine":  0.30,
    "pyroxene": 0.40,
    "anorthite": 0.85,
    "ilmenite": 0.00,
}

# LED full-width estimates (nm) — typical narrowband LED FWHM ~20–30 nm.
LED_FWHM_NM: float = 25.0


# ---------------------------------------------------------------------------
# Endmember loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Endmembers:
    """Reference reflectance spectra on the C12880MA grid."""

    wavelengths_nm: np.ndarray
    spectra: np.ndarray  # (4, 288), rows in ENDMEMBER_NAMES order
    source: str

    @property
    def n_endmembers(self) -> int:
        return self.spectra.shape[0]


def load_endmembers(path: str | Path) -> Endmembers:
    """Load the cached endmember .npz produced by ``scripts/download_usgs.py``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"endmember cache not found at {path}; "
            f"run `python scripts/download_usgs.py` first"
        )
    data = np.load(path, allow_pickle=False)
    lam = data["wavelengths_nm"].astype(np.float64)
    if lam.shape != (N_SPEC,):
        raise ValueError(f"endmember wavelength grid has wrong shape {lam.shape}")
    if not np.allclose(lam, WAVELENGTHS):
        raise ValueError("endmember wavelength grid does not match WAVELENGTHS")
    rows = np.stack([data[n].astype(np.float64) for n in ENDMEMBER_NAMES], axis=0)
    source = str(data["source"]) if "source" in data.files else "unknown"
    return Endmembers(wavelengths_nm=lam, spectra=rows, source=source)


# ---------------------------------------------------------------------------
# Sample composition / class assignment
# ---------------------------------------------------------------------------


def fractions_for_class(klass: str, rng: np.random.Generator) -> np.ndarray:
    """Draw a 4-vector of mineral mass fractions for a target class.

    The vector is normalised to sum to 1 and is ordered as
    ``[olivine, pyroxene, anorthite, ilmenite]``.
    """
    f = np.zeros(4, dtype=np.float64)
    if klass == "olivine_rich":
        f[ENDMEMBER_INDEX["olivine"]] = rng.uniform(0.55, 0.85)
        f[ENDMEMBER_INDEX["pyroxene"]] = rng.uniform(0.05, 0.20)
        f[ENDMEMBER_INDEX["anorthite"]] = rng.uniform(0.05, 0.20)
        f[ENDMEMBER_INDEX["ilmenite"]] = rng.uniform(0.00, 0.05)
    elif klass == "pyroxene_rich":
        f[ENDMEMBER_INDEX["pyroxene"]] = rng.uniform(0.55, 0.85)
        f[ENDMEMBER_INDEX["olivine"]] = rng.uniform(0.05, 0.20)
        f[ENDMEMBER_INDEX["anorthite"]] = rng.uniform(0.05, 0.20)
        f[ENDMEMBER_INDEX["ilmenite"]] = rng.uniform(0.00, 0.05)
    elif klass == "anorthositic":
        f[ENDMEMBER_INDEX["anorthite"]] = rng.uniform(0.65, 0.92)
        f[ENDMEMBER_INDEX["olivine"]] = rng.uniform(0.02, 0.15)
        f[ENDMEMBER_INDEX["pyroxene"]] = rng.uniform(0.02, 0.15)
        f[ENDMEMBER_INDEX["ilmenite"]] = rng.uniform(0.00, 0.03)
    elif klass == "ilmenite_rich":
        f[ENDMEMBER_INDEX["ilmenite"]] = rng.uniform(0.35, 0.65)
        f[ENDMEMBER_INDEX["pyroxene"]] = rng.uniform(0.10, 0.30)
        f[ENDMEMBER_INDEX["olivine"]] = rng.uniform(0.05, 0.20)
        f[ENDMEMBER_INDEX["anorthite"]] = rng.uniform(0.05, 0.25)
    elif klass == "mixed":
        f = rng.dirichlet(alpha=np.array([1.0, 1.0, 1.0, 0.6]))
        # cap ilmenite for "mixed" so it doesn't overlap "ilmenite_rich"
        f[ENDMEMBER_INDEX["ilmenite"]] = min(f[ENDMEMBER_INDEX["ilmenite"]], 0.20)
    else:
        raise ValueError(f"unknown mineral class: {klass}")
    f = np.clip(f, 0.0, None)
    f /= f.sum()
    return f


# ---------------------------------------------------------------------------
# Per-measurement noise model
# ---------------------------------------------------------------------------


@dataclass
class NoiseConfig:
    gain_sigma: float = 0.015           # per-channel multiplicative gain
    baseline_amp: float = 0.020         # peak |baseline drift|
    intensity_sigma: float = 0.05       # global intensity scale jitter (relative)
    shot_noise_scale: float = 1.0       # multiplier on Poisson-like shot noise
    led_noise_sigma: float = 0.010      # extra LED-channel noise
    lif_noise_sigma: float = 0.020      # LIF channel noise
    intensity_by_packing: dict[str, float] = field(
        default_factory=lambda: {"loose": 0.85, "medium": 1.00, "packed": 1.10}
    )
    # --- realistic hardware degradation ---
    # Per-measurement probability that a cosmic-ray-like spike is present.
    # When it fires, 1-3 random channels get a large transient spike on top
    # of the underlying spectrum. The probe will see these at LEO/lunar
    # cadence because there is no atmosphere attenuation.
    cosmic_ray_prob: float = 0.02
    cosmic_ray_amp_range: tuple[float, float] = (0.15, 0.45)
    # Slow sensor degradation: a per-pixel responsivity mask that gets
    # baked in once per *sample* (so every measurement of the same sample
    # shares the same degradation pattern — matches reality, where pixels
    # age faster than the measurement cadence). A small fraction of pixels
    # degrade heavily ("hot"/"dead" pixels), the rest drift mildly.
    degradation_pct_hot: float = 0.005
    degradation_hot_range: tuple[float, float] = (0.40, 0.85)
    degradation_mild_sigma: float = 0.008


def _polynomial_baseline(rng: np.random.Generator, amp: float) -> np.ndarray:
    """Smooth degree-3 baseline drift with peak amplitude ``amp``."""
    x = np.linspace(-1.0, 1.0, N_SPEC)
    coeffs = rng.uniform(-1.0, 1.0, size=4)  # a + b x + c x^2 + d x^3
    poly = coeffs[0] + coeffs[1] * x + coeffs[2] * x**2 + coeffs[3] * x**3
    poly = poly / np.max(np.abs(poly) + 1e-12)
    return amp * poly


def _shot_noise(spectrum: np.ndarray, rng: np.random.Generator,
                integration_time_ms: int, scale: float) -> np.ndarray:
    """Poisson-like noise whose stddev grows with sqrt(signal/integration)."""
    # Use signal*const as photon-count surrogate so std ~ sqrt(signal)/sqrt(t).
    const = 5_000.0 * (integration_time_ms / 200.0)
    counts = np.maximum(spectrum, 1e-6) * const
    noisy_counts = rng.poisson(counts).astype(np.float64)
    noisy = noisy_counts / const
    return spectrum + scale * (noisy - spectrum)


def sensor_degradation_mask(
    rng: np.random.Generator, cfg: "NoiseConfig"
) -> np.ndarray:
    """Build a per-pixel responsivity mask in [0, 1].

    Most channels have a small multiplicative drift centred near 1.0; a
    tiny fraction are "hot" pixels with significantly reduced response,
    modelling radiation damage over time. This is generated *per sample*
    and shared across all of that sample's measurements (pixels age on a
    slower timescale than the measurement cadence).
    """
    mask = 1.0 + rng.normal(0.0, cfg.degradation_mild_sigma, size=N_SPEC)
    n_hot = max(0, int(round(cfg.degradation_pct_hot * N_SPEC)))
    if n_hot > 0:
        hot_idx = rng.choice(N_SPEC, size=n_hot, replace=False)
        lo, hi = cfg.degradation_hot_range
        mask[hot_idx] = 1.0 - rng.uniform(lo, hi, size=n_hot)
    return np.clip(mask, 0.0, 1.2)


def _apply_cosmic_rays(
    spectrum: np.ndarray, rng: np.random.Generator, cfg: "NoiseConfig"
) -> np.ndarray:
    """With probability ``cfg.cosmic_ray_prob``, deposit 1-3 narrow spikes."""
    if rng.uniform(0.0, 1.0) >= cfg.cosmic_ray_prob:
        return spectrum
    n_hits = int(rng.integers(1, 4))
    out = spectrum.copy()
    lo, hi = cfg.cosmic_ray_amp_range
    for _ in range(n_hits):
        idx = int(rng.integers(0, N_SPEC))
        amp = float(rng.uniform(lo, hi))
        # Triangle-like spike over a 3-pixel window.
        out[idx] = min(1.5, out[idx] + amp)
        if idx > 0:
            out[idx - 1] = min(1.5, out[idx - 1] + 0.4 * amp)
        if idx + 1 < N_SPEC:
            out[idx + 1] = min(1.5, out[idx + 1] + 0.4 * amp)
    return out


def _led_response(spectrum: np.ndarray) -> np.ndarray:
    """Integrate the full-resolution spectrum under each LED's Gaussian band."""
    sigma = LED_FWHM_NM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    out = np.empty(N_LED, dtype=np.float64)
    lam = WAVELENGTHS
    for i, center in enumerate(LED_WAVELENGTHS_NM):
        w = np.exp(-0.5 * ((lam - center) / sigma) ** 2)
        wsum = w.sum()
        if wsum < 1e-9:
            # LED outside spectrometer range (e.g. 940 nm) — fall back to a
            # smooth extrapolation: use the last bin scaled by a
            # mineral-independent rolloff factor so the channel is still
            # informative-ish.
            edge = float(spectrum[-1])
            out[i] = edge * 0.85
        else:
            out[i] = float((w * spectrum).sum() / wsum)
    return out


# ---------------------------------------------------------------------------
# High-level generation
# ---------------------------------------------------------------------------


def mixture_spectrum(fractions: np.ndarray, endmembers: Endmembers) -> np.ndarray:
    if fractions.shape != (endmembers.n_endmembers,):
        raise ValueError(f"fractions has wrong shape {fractions.shape}")
    return fractions @ endmembers.spectra  # (288,)


def synth_measurement(
    sample_id: str,
    mineral_class: str,
    fractions: np.ndarray,
    endmembers: Endmembers,
    rng: np.random.Generator,
    *,
    integration_time_ms: int | None = None,
    ambient_temp_c: float | None = None,
    packing_density: str | None = None,
    noise: NoiseConfig | None = None,
    timestamp: str | None = None,
    degradation_mask: np.ndarray | None = None,
) -> Measurement:
    """Generate one Measurement instance for the given sample composition."""
    if noise is None:
        noise = NoiseConfig()
    if integration_time_ms is None:
        integration_time_ms = int(rng.integers(80, 400))
    if ambient_temp_c is None:
        ambient_temp_c = float(rng.normal(20.0, 3.0))
    if packing_density is None:
        packing_density = str(rng.choice(PACKING_DENSITIES))
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    # 1) noise-free mixture
    base = mixture_spectrum(fractions, endmembers)

    # 2) per-channel gain
    gain = 1.0 + rng.normal(0.0, noise.gain_sigma, size=N_SPEC)
    spec = base * gain

    # 3) baseline drift
    spec = spec + _polynomial_baseline(rng, noise.baseline_amp)

    # 4) global intensity scaling (packing + jitter)
    intensity = noise.intensity_by_packing.get(packing_density, 1.0)
    intensity *= 1.0 + rng.normal(0.0, noise.intensity_sigma)
    spec = spec * intensity

    # 5) sensor degradation mask (per-sample responsivity, dead/hot pixels)
    if degradation_mask is not None:
        spec = spec * degradation_mask

    # 6) shot noise
    spec = _shot_noise(spec, rng, integration_time_ms, noise.shot_noise_scale)

    # 7) cosmic ray hits (rare, additive, large local amplitude)
    spec = _apply_cosmic_rays(spec, rng, noise)

    # clip to physically valid reflectance range, allowing a small overshoot
    # so down-stream preprocess steps can still see the noisy structure.
    spec = np.clip(spec, 0.0, 1.5)

    # 6) LED narrowband channels — derived from the *clean* mixture so the
    # LED measurement is approximately independent noise from the spectro.
    led_clean = _led_response(base * intensity)
    led = led_clean + rng.normal(0.0, noise.led_noise_sigma, size=N_LED)
    led = np.clip(led, 0.0, 1.5)

    # 7) LIF: weighted sum of per-mineral efficiencies, suppressed by
    # ilmenite. The 1.5 power makes ilmenite quenching strongly nonlinear.
    ilm_frac = float(fractions[ENDMEMBER_INDEX["ilmenite"]])
    eff = sum(
        float(fractions[ENDMEMBER_INDEX[name]]) * LIF_EFFICIENCY[name]
        for name in ENDMEMBER_NAMES
    )
    quench = (1.0 - ilm_frac) ** 1.5
    lif = eff * quench * intensity
    lif += rng.normal(0.0, noise.lif_noise_sigma)
    lif = float(max(lif, 0.0))

    return Measurement(
        sample_id=sample_id,
        measurement_id=str(uuid.UUID(bytes=bytes(rng.bytes(16)))),
        timestamp=timestamp,
        mineral_class=mineral_class,  # type: ignore[arg-type]
        ilmenite_fraction=ilm_frac,
        integration_time_ms=int(integration_time_ms),
        ambient_temp_c=float(ambient_temp_c),
        packing_density=packing_density,  # type: ignore[arg-type]
        spec=[float(x) for x in spec],
        led=[float(x) for x in led],
        lif_450lp=lif,
    )


def synth_sample(
    sample_id: str,
    mineral_class: str,
    endmembers: Endmembers,
    *,
    n_measurements: int,
    rng: np.random.Generator,
    noise: NoiseConfig | None = None,
) -> list[Measurement]:
    """Generate ``n_measurements`` measurements for one (sample, class) pair.

    Composition is fixed once per sample so that all measurements of the
    same sample share the same ilmenite fraction (modulo measurement noise).
    A sensor degradation mask is also fixed per sample, so the same dead/
    hot pixels persist across a sample's measurements — matching real
    hardware where pixel aging is slow compared to measurement cadence.
    """
    cfg = noise or NoiseConfig()
    fractions = fractions_for_class(mineral_class, rng)
    degradation_mask = sensor_degradation_mask(rng, cfg)
    out: list[Measurement] = []
    for _ in range(n_measurements):
        out.append(
            synth_measurement(
                sample_id=sample_id,
                mineral_class=mineral_class,
                fractions=fractions,
                endmembers=endmembers,
                rng=rng,
                noise=cfg,
                degradation_mask=degradation_mask,
            )
        )
    return out


def synth_dataset(
    endmembers: Endmembers,
    *,
    n_samples: int,
    measurements_per_sample: int,
    seed: int = 0,
    classes: Iterable[str] = MINERAL_CLASSES,
    noise: NoiseConfig | None = None,
) -> list[Measurement]:
    """Generate a complete synthetic dataset.

    Classes are dealt round-robin so the dataset is approximately balanced.
    """
    rng = np.random.default_rng(seed)
    classes = tuple(classes)
    out: list[Measurement] = []
    for i in range(n_samples):
        klass = classes[i % len(classes)]
        sample_id = f"S{i:04d}_{klass}"
        out.extend(
            synth_sample(
                sample_id=sample_id,
                mineral_class=klass,
                endmembers=endmembers,
                n_measurements=measurements_per_sample,
                rng=rng,
                noise=noise,
            )
        )
    return out


__all__ = [
    "ENDMEMBER_NAMES",
    "ENDMEMBER_INDEX",
    "LIF_EFFICIENCY",
    "Endmembers",
    "NoiseConfig",
    "load_endmembers",
    "fractions_for_class",
    "mixture_spectrum",
    "synth_measurement",
    "synth_sample",
    "synth_dataset",
]
