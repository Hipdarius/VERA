<div align="center">

# REGOSCAN

**Compact VIS/NIR + 405 nm LIF Probe for Real-Time Lunar Regolith Mineralogy**

[![CI](https://github.com/Hipdarius/RegoScan/actions/workflows/ci.yml/badge.svg)](https://github.com/Hipdarius/RegoScan/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab.svg)](https://python.org)
[![Tests: 136 passed](https://img.shields.io/badge/Tests-136%20passed-brightgreen.svg)](#testing)

A low-power optical probe that identifies lunar minerals and estimates ilmenite mass fraction in real time — designed for the next generation of ISRU rovers.

[Quick Start](#quick-start) &middot; [Architecture](#architecture) &middot; [API Reference](#api-reference) &middot; [Contributing](CONTRIBUTING.md)

</div>

---

## Why REGOSCAN

Lunar oxygen extraction depends on finding ilmenite (FeTiO3) — the best feedstock for hydrogen reduction. Current prospecting instruments weigh kilograms and cost six figures. REGOSCAN targets **< 300 g**, **< 2 W**, and **< 1,500 EUR** in parts, classifying regolith composition in milliseconds on an ESP32.

The probe combines three sensing modalities into a single 301-feature measurement:

| Modality | Sensor | Channels | What it measures |
|----------|--------|----------|-----------------|
| VIS/NIR reflectance | Hamamatsu C12880MA | 288 (340--850 nm) | Mineral absorption fingerprints |
| Narrowband LED | 12 discrete LEDs | 12 (385--940 nm) | Targeted spectral features |
| Laser-induced fluorescence | 405 nm diode + 450 nm LP filter | 1 | Ilmenite suppresses fluorescence |

---

## Performance

Current benchmarks on held-out synthetic lunar samples:

| Model | Task | Metric | Value |
|:------|:-----|:-------|------:|
| 1D ResNet (670k params) | Classification | Top-1 accuracy | **93.2 %** |
| PLSR baseline | Regression | Ilmenite R^2 | **0.967** |
| PLSR baseline | Regression | Ilmenite RMSE | **0.037** |

The CNN classifies; the PLSR provides precise ilmenite mass fractions. Both are reported in the paper.

---

## Quick Start

### Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- Node.js 20+ (web dashboard)
- PlatformIO (firmware, optional)

### Setup

```bash
git clone https://github.com/Hipdarius/RegoScan.git && cd RegoScan

uv sync --all-extras            # Python environment
cd web && npm install && cd ..   # Web frontend
```

### Run

```bash
make data-gen                    # Generate 4,000 synthetic spectra
make train                       # Train CNN with 5-fold CV
make test                        # 136 tests, all passing
```

```bash
make serve-api                   # Terminal 1: FastAPI on :8000
make serve-web                   # Terminal 2: Next.js on :3000
```

Open **http://localhost:3000** and click **Initiate Scan**.

---

## Architecture

```
                    ┌───────────────────────────────────────────────────┐
                    │                 ESP32-S3 Probe                    │
                    │  ┌───────────┐ ┌──────────┐ ┌──────────────────┐ │
                    │  │ C12880MA  │ │ 12x LEDs │ │ 405 nm LIF Laser │ │
                    │  │ 288 ch    │ │ 385-940  │ │ + photodiode     │ │
                    │  └─────┬─────┘ └────┬─────┘ └────────┬─────────┘ │
                    │        └─────────┬──┘                │           │
                    │                  ▼                    ▼           │
                    │        Non-blocking state machine (main.cpp)      │
                    │        5x averaging per measurement               │
                    └──────────────────┬────────────────────────────────┘
                                       │ USB Serial (JSON)
                    ┌──────────────────▼────────────────────────────────┐
                    │              bridge.py                             │
                    │  Validate (schema.py) → ONNX infer → CSV log      │
                    └──────────────────┬────────────────────────────────┘
                                       │
                    ┌──────────────────▼────────────────────────────────┐
                    │           FastAPI  (port 8000)                     │
                    │  /api/predict  /api/predict/demo  /api/endmembers │
                    └──────────────────┬────────────────────────────────┘
                                       │
                    ┌──────────────────▼────────────────────────────────┐
                    │        Next.js Mission Console (port 3000)        │
                    │  Spectrum chart ─ Endmember overlays ─ Ilmenite   │
                    │  gauge ─ Probability bars ─ Scan history ─ CSV    │
                    │  upload ─ Dark/light theme                        │
                    └───────────────────────────────────────────────────┘
```

---

## Project Structure

```
src/regoscan/               Core ML library
  schema.py                 309-column data contract (locked v1.0.0)
  synth.py                  Physics-based spectral mixing engine
  preprocess.py             SG smoothing, baseline correction, continuum removal
  augment.py                6 augmentation types incl. spectral mixing
  features.py               8 hand-crafted spectral features
  datasets.py               Sample-level splitting (no leakage)
  train.py                  CNN + PLSR training with k-fold CV
  evaluate.py               Confusion matrices, ROC/PR curves, Bland-Altman, bootstrap CI
  inference.py              ONNX-only engine (torch-free, <5 ms)
  quantize.py               ONNX → TFLite INT8
  models/cnn.py             1D ResNet (670k params, dual heads)
  models/plsr.py            Random Forest + PLS + feature importance

apps/api.py                 FastAPI backend (5 endpoints)
scripts/bridge.py           Serial listener: ESP32 → validate → infer → log
scripts/mock_esp32.py       Synthetic JSON emitter for testing

firmware/src/               ESP32-S3 C++ (PlatformIO)
  Config.h                  Pin assignments, timing, temp compensation
  C12880MA.h/.cpp           Spectrometer driver (bit-banged 100 kHz clock)
  Illumination.h/.cpp       12 LED + 405 nm laser control
  Protocol.h/.cpp           JSON serialization + thermistor
  AS7265x.h                 Secondary 18-band sensor (interface)
  main.cpp                  Non-blocking state machine, 5x averaging

web/                        Next.js 14 mission console
  components/               10 React components (spectrum, gauges, history, upload)
  lib/                      API client + TypeScript types
  api/predict.py            Vercel serverless handler

tests/                      136 tests across 8 files
```

---

## Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Schema contract** | `schema.py` defines the exact 309-column format. Locked at v1.0.0. Every module reads/writes through this contract. |
| **Sample-level integrity** | Train/val/test splits partition by `sample_id`, never by measurement. A canary test enforces no leakage. |
| **Physical fidelity** | Synthetic spectra use linear mixing, shot noise, baseline drift, gain variation, packing density, and LIF quenching. |
| **Software-first** | The pipeline works on synthetic data today. When hardware arrives, swap the data source — nothing else changes. |
| **No heap on embedded** | All firmware buffers are `constexpr` sized. Zero `new`, zero `std::vector`, zero `String` concatenation. |

---

## API Reference

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/healthz` | Liveness probe, model SHA-256, schema version |
| `GET` | `/api/meta` | Wavelength grid, class names, feature count |
| `POST` | `/api/predict` | Classify from 301-feature JSON payload |
| `POST` | `/api/predict/demo` | Synthesize random spectrum and classify |
| `GET` | `/api/endmembers` | Pure mineral reference spectra (4 endmembers) |

---

## Testing

```bash
make test   # or: uv run pytest tests/ -v
```

| Suite | Tests | Coverage |
|-------|------:|----------|
| `test_schema.py` | 21 | Pydantic model, CSV validation, column names |
| `test_synth.py` | 15 | Endmember mixing, determinism, LIF physics |
| `test_preprocess.py` | 13 | Smoothing, baseline, continuum removal |
| `test_datasets.py` | 10 | Sample splits, no-leakage canary |
| `test_augment.py` | 10 | All 6 augmentation types |
| `test_train_smoke.py` | 4 | CNN forward, PLSR fit, determinism |
| `test_inference.py` | 29 | ONNX engine, softmax, batch predict |
| `test_api.py` | 34 | All 5 endpoints, validation, error handling |
| **Total** | **136** | |

---

## Makefile Targets

```bash
make test             # Run pytest (136 tests)
make lint             # Ruff check + format
make train            # CNN training with 5-fold CV
make data-gen         # Generate synthetic dataset (4,000 spectra)
make serve-api        # FastAPI on :8000
make serve-web        # Next.js on :3000
make firmware-build   # PlatformIO compile
make clean            # Remove caches
```

---

## License

[MIT](LICENSE)

---

<div align="center">

*Developed for the [Jonk Fuerscher](https://www.fjsl.lu) 2027 competition.*

*Luxembourg Space Agency AFAD Ambassador Project*

</div>
