# Regoscan

Compact VIS/NIR + 405 nm laser-induced fluorescence (LIF) probe for estimating
lunar regolith mineral composition.

This repository contains the **software stack only** — no hardware drivers,
no firmware. The pipeline is validated end-to-end on synthetic spectra
derived from USGS Spectral Library endmembers so that when the real C12880MA +
LED + LIF hardware comes online, real measurements drop into the same canonical
CSV schema and flow through unchanged.

## What it does

- 5-way mineral classification: `ilmenite_rich`, `olivine_rich`,
  `pyroxene_rich`, `anorthositic`, `mixed`
- Continuous regression of `ilmenite_fraction` in `[0, 1]`
- Inputs per measurement: 288 reflectance channels (340–850 nm), 12 narrowband
  LED reflectances, 1 LIF photodiode value, plus sample metadata
- Two model paths:
  - **Baseline:** PLSR + RandomForest (sklearn)
  - **CNN:** small 1D ConvNet (~50k params, PyTorch)

The canonical measurement schema is locked in
[`src/regoscan/schema.py`](src/regoscan/schema.py) and is the contract between
hardware and software.

## Acceptance test

```bash
uv sync
python scripts/download_usgs.py
python scripts/generate_synth_dataset.py --n-samples 50 --measurements-per-sample 8 --out data/synth_v1.csv
python -m regoscan.train --model plsr --data data/synth_v1.csv --out runs/plsr/
python -m regoscan.train --model cnn  --data data/synth_v1.csv --epochs 20 --out runs/cnn/
python -m regoscan.evaluate --run runs/cnn/ --data data/synth_v1.csv
python -m regoscan.quantize --run runs/cnn/ --out runs/cnn/model_int8.tflite
streamlit run apps/dashboard.py
pytest -q
```

All steps must complete without errors. The CNN must beat random-guess on
synthetic data (>50% top-1 on 5 classes); the goal of this scaffolding session
is to prove the wiring, not to chase accuracy.

## Layout

```
regoscan/
  src/regoscan/      # library code (schema, synth, preprocess, models, ...)
  apps/dashboard.py  # Streamlit viewer + inference
  scripts/           # CLI utilities (USGS download, dataset generation)
  tests/             # unit tests, including the sample_id leak test
```

## Design rules (do not violate)

1. **Single canonical CSV schema** in `schema.py`. All other modules go
   through it.
2. **Sample-level splits.** Train/val/test split by `sample_id`, never by
   individual `measurement_id`. Enforced by `tests/test_datasets.py`.
3. **Synthetic spectra are physically motivated**, not random noise. Built by
   linear-mixing USGS endmember spectra and adding shot noise, gain
   variation, baseline drift, and an ilmenite-suppressed LIF response.
4. **Baseline first.** PLSR / RandomForest must work end-to-end before the
   CNN is trained.
5. **No hardware code.** Serial / firmware lives in a future session.
6. **Deterministic.** Seeds are pinned everywhere; the same `train` invocation
   produces bit-identical results across runs.

## Wavelength grid note

The C12880MA covers 340–850 nm in 288 pixels (~1.78 nm/px). Spectrometer
columns are named `spec_000 .. spec_287` (the literal nm value would be a
non-integer). The actual nanometer grid is exposed as
`regoscan.schema.WAVELENGTHS` (`np.linspace(340, 850, 288)`).
