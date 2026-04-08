<div align="center">

# VERA

### Visible & Emission Regolith Assessment

**Compact VIS/NIR + 405 nm LIF Probe for Real-Time Lunar Regolith Mineralogy**

[![CI](https://github.com/Hipdarius/RegoScan/actions/workflows/ci.yml/badge.svg)](https://github.com/Hipdarius/RegoScan/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-22d3ee)](LICENSE)
[![Tests: 72](https://img.shields.io/badge/tests-72_passing-34d399)]()
[![ONNX Runtime](https://img.shields.io/badge/inference-ONNX_Runtime-8b5cf6)]()

---

*A full-stack lunar prospecting instrument — from embedded firmware through*
*machine learning to real-time web telemetry — designed and built by*
**Darius Ferent** *for the Jonk Fuerscher 2027 competition.*

</div>

---

## Why VERA?

Lunar In-Situ Resource Utilization (ISRU) depends on knowing **what minerals are beneath the regolith**. Oxygen extraction from ilmenite (FeTiO3) is the most promising near-term ISRU pathway, but current prospecting tools are either too heavy, too slow, or require sample return.

VERA solves this with a **handheld-class optical probe** that combines three sensing modalities into a single 301-feature measurement, classifies the mineral phase in <5 ms on CPU, and estimates ilmenite mass fraction continuously.

---

## Sensor Architecture

| Modality | Hardware | Channels | Range |
|:---------|:---------|:--------:|:------|
| **VIS/NIR Reflectance** | Hamamatsu C12880MA | 288 | 340 -- 850 nm |
| **Narrowband LED** | 12x discrete LEDs | 12 | 385 -- 940 nm |
| **405 nm LIF** | Laser diode + 450 LP filter | 1 | Fluorescence intensity |

> **Total input vector: 301 features** (288 spectral + 12 LED + 1 LIF)

---

## System Architecture

```
+---------------------------------------------------------------------+
|                        VERA SYSTEM ARCHITECTURE                     |
+---------------------------------------------------------------------+
|                                                                     |
|  HARDWARE LAYER              INFERENCE LAYER        INTERFACE LAYER |
|  +-----------------+    +--------------------+    +---------------+ |
|  | ESP32-S3 MCU    |    | ONNX Runtime       |    | Next.js 14    | |
|  | C12880MA driver  |--->| 1D ResNet (~670k)  |--->| Recharts      | |
|  | 12x LED array   |    | Softmax + Regress  |    | Framer Motion | |
|  | 405 nm LIF laser|    | <5 ms / inference  |    | Tailwind CSS  | |
|  | AS7265x (18-ch) |    +--------------------+    +---------------+ |
|  +-----------------+             |                        |         |
|          |                       v                        v         |
|          |               +--------------------+    +---------------+|
|          +-------------->| FastAPI REST API    |--->| Theme system  ||
|           JSON/Serial    | 5 endpoints         |    | Scan history  ||
|                          | ONNX session pool   |    | CSV upload    ||
|                          +--------------------+    +---------------+|
|                                                                     |
|  ML TRAINING PIPELINE                                               |
|  +---------------------------------------------------------------+  |
|  | Synthetic spectra (USGS/RELAB endmembers)                     |  |
|  | StratifiedGroupKFold CV | Spectral mixing augmentation        |  |
|  | 1D ResNet (classification) + PLSR (regression) dual heads     |  |
|  | Bootstrap CI | Permutation feature importance | ONNX export   |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
```

---

## Performance

Benchmarked on a held-out test set of 3,000 synthetic lunar regolith samples:

| Model | Task | Metric | Score |
|:------|:-----|:-------|------:|
| **1D ResNet** | 5-class mineral ID | Top-1 Accuracy | **93.2 %** |
| **1D ResNet** | 5-class mineral ID | Macro F1 | **0.91** |
| **PLSR** | Ilmenite regression | R2 | **0.967** |
| **PLSR** | Ilmenite regression | RMSE | **0.037** |

The hybrid approach uses the CNN for robust classification and PLSR for precise mass-fraction estimation.

---

## Project Structure

```
vera/
├── src/vera/               Core Python library
│   ├── schema.py               Hardware-software data contract (locked)
│   ├── synth.py                Physically-motivated spectral mixing engine
│   ├── datasets.py             PyTorch dataset + StratifiedGroupKFold splits
│   ├── augment.py              Spectral mixing, noise injection, baseline drift
│   ├── train.py                CNN + PLSR training with k-fold cross-validation
│   ├── evaluate.py             Confusion matrices, ROC/PR curves, CI intervals
│   ├── inference.py            ONNX Runtime engine (torch-free, <5 ms)
│   ├── quantize.py             ONNX export + INT8 quantization
│   ├── features.py             Feature extraction and preprocessing pipeline
│   ├── io_csv.py               Schema-validated CSV I/O
│   ├── preprocess.py           SNV normalization, Savitzky-Golay smoothing
│   └── models/
│       ├── cnn.py                  1D ResNet (~670k params, dual head)
│       └── plsr.py                 PLS Regression + Random Forest baseline
│
├── apps/
│   ├── api.py                  FastAPI REST backend (5 endpoints)
│   └── dashboard.py            Streamlit exploration dashboard
│
├── web/                    Next.js 14 mission console
│   ├── app/                    App router, layout, global styles
│   ├── components/             14 React components (charts, gauges, panels)
│   └── lib/                    API client, TypeScript types
│
├── firmware/               ESP32-S3 embedded firmware
│   └── src/
│       ├── main.cpp                Non-blocking state machine
│       ├── C12880MA.cpp/h          Bit-banged spectrometer driver
│       ├── AS7265x.h               Secondary 18-band sensor interface
│       ├── Illumination.cpp/h      LED array + laser GPIO control
│       ├── Protocol.cpp/h          ArduinoJson serialization + thermistor
│       └── Config.h                Pin assignments, calibration constants
│
├── tests/                  72 tests across 6 test modules
│   ├── test_schema.py          22 tests — data contract validation
│   ├── test_preprocess.py      14 tests — SNV, Savitzky-Golay, edge cases
│   ├── test_synth.py           12 tests — spectral synthesis fidelity
│   ├── test_augment.py         10 tests — augmentation pipeline
│   ├── test_datasets.py        10 tests — split integrity, no sample leakage
│   └── test_train_smoke.py      4 tests — CNN forward pass, PLSR fit
│
├── scripts/                CLI utilities
│   ├── generate_synth_dataset.py   Synthetic data generation
│   ├── download_usgs.py            USGS Spectral Library downloader
│   ├── download_relab.py           RELAB lunar sample downloader
│   ├── bridge.py                   ESP32 serial -> inference -> CSV logger
│   └── mock_esp32.py              Synthetic JSON frame emitter for testing
│
├── .github/workflows/ci.yml   CI: pytest + TypeScript typecheck + PlatformIO
├── pyproject.toml              Project config (uv / hatch)
├── Makefile                    8 automation targets
└── CONTRIBUTING.md             Developer guide and conventions
```

---

## Design Principles

| Principle | Implementation |
|:----------|:---------------|
| **Hardware-Software Contract** | `schema.py` defines every column name, dtype, and valid range. All layers read/write through this single source of truth. |
| **No Sample Leakage** | Train/val/test splits use `StratifiedGroupKFold` on `sample_id`, never on individual measurements. |
| **Physical Fidelity** | Synthetic spectra use linear mixing of real USGS/RELAB endmembers with shot noise, gain drift, and baseline variation. |
| **Deterministic Training** | Same CLI invocation produces bit-identical model weights (seeded RNG, deterministic cuDNN). |
| **Torch-Free Inference** | Production path uses ONNX Runtime only — no PyTorch dependency at serve time. |
| **Transparency** | Every training run logs confusion matrices, ROC/PR curves, bootstrap 95% CI, and feature importance. |

---

## API Reference

The FastAPI backend exposes 5 endpoints:

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/healthz` | Liveness check |
| `GET` | `/api/meta` | Model version, schema, feature count, ONNX hash |
| `POST` | `/api/predict` | Classify a 301-feature input vector |
| `POST` | `/api/predict/demo` | Fire a synthetic acquisition for demo |
| `GET` | `/api/endmembers` | Return reference mineral spectra for overlay |

```bash
# Start the API server
make serve-api

# Start the web dashboard
make serve-web
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/Hipdarius/RegoScan.git
cd RegoScan
uv sync

# Generate synthetic training data
make data-gen

# Train the CNN with 5-fold cross-validation
make train

# Run the full test suite
make test

# Launch API + web dashboard
make serve-api   # terminal 1 — port 8000
make serve-web   # terminal 2 — port 3000
```

---

## Makefile Targets

| Target | Description |
|:-------|:------------|
| `make test` | Run 72 pytest tests with verbose output |
| `make lint` | Ruff check + format verification |
| `make train` | Train CNN with 5-fold CV (50 epochs) |
| `make data-gen` | Generate 4,000 synthetic spectral samples |
| `make serve-api` | FastAPI on port 8000 with hot reload |
| `make serve-web` | Next.js dev server on port 3000 |
| `make firmware-build` | Compile ESP32-S3 firmware via PlatformIO |
| `make clean` | Remove caches and build artifacts |

---

## CI / CD

GitHub Actions runs three jobs on every push to `main`:

| Job | What it does |
|:----|:-------------|
| **test** | `uv sync` + `pytest tests/ -v` across all 72 tests |
| **typecheck** | `npm ci` + `tsc --noEmit` on the Next.js frontend |
| **firmware-build** | PlatformIO compilation of the ESP32-S3 firmware |

---

## Test Breakdown

| Module | Tests | Coverage |
|:-------|------:|:---------|
| `test_schema.py` | 22 | Data contract, column names, dtypes, ranges |
| `test_preprocess.py` | 14 | SNV normalization, SG smoothing, edge cases |
| `test_synth.py` | 12 | Endmember mixing, noise models, output shapes |
| `test_augment.py` | 10 | Spectral mix augmentation, seed determinism |
| `test_datasets.py` | 10 | Split integrity, group isolation, augmentation |
| `test_train_smoke.py` | 4 | CNN forward pass, parameter count, PLSR fit |
| **Total** | **72** | |

---

## Tech Stack

| Layer | Technologies |
|:------|:-------------|
| **ML / Training** | PyTorch, scikit-learn, NumPy, pandas, SciPy |
| **Inference** | ONNX Runtime (CPU, <5 ms per prediction) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Next.js 14, React 18, Recharts, Framer Motion, Tailwind CSS |
| **Firmware** | C++ (ESP32-S3), PlatformIO, ArduinoJson |
| **CI / CD** | GitHub Actions (pytest + tsc + PlatformIO) |
| **Package Mgmt** | uv (Python), npm (Node.js) |

---

<div align="center">

**Developed by Darius Ferent for the Jonk Fuerscher 2027 competition.**

</div>
