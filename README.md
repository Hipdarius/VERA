# VERA

**Compact VIS/NIR + 405 nm LIF Probe for Real-Time Lunar Regolith Mineralogy**

VERA (Visible & Emission Regolith Assessment) is a lightweight, low-power optical probe designed for lunar In-Situ Resource Utilization (ISRU) prospecting. By combining 288-channel visible/NIR diffuse reflectance spectroscopy with 405 nm Laser-Induced Fluorescence (LIF), it identifies key mineral phases and estimates ilmenite (FeTiO₃) mass fraction in real-time.

---

## 🚀 Overview

The project provides a complete software-hardware co-design stack, validated on high-fidelity synthetic spectra derived from USGS and RELAB endmembers. It targets the next generation of lunar rovers and ISRU pilot plants, where identifying oxygen-rich minerals (like ilmenite) is a mission-critical bottleneck.

### Key Capabilities:
- **5-Way Mineral Classification**: Identifies `ilmenite_rich`, `olivine_rich`, `pyroxene_rich`, `anorthositic`, and `mixed` regolith.
- **Continuous Regression**: Estimates `ilmenite_fraction` (0–100%) with high precision.
- **Multi-Modal Input**: Integrates 288 reflectance channels (340–850 nm), 12 narrowband LED reflectances, and 1 LIF photodiode channel.
- **Embedded-Ready ML**: Includes a lightweight 1D ResNet (PyTorch/ONNX) and a statistical PLSR baseline.

---

## 📊 Performance (v2 Dataset)

Current benchmarks on a held-out test set of 3,000 synthetic lunar samples:

| Model | Task | Metric | Performance |
|-------|------|--------|-------------|
| **1D ResNet** | Classification | Top-1 Accuracy | **93.2%** |
| **PLSR** | Regression | Ilmenite R² | **0.967** |
| **PLSR** | Regression | Ilmenite RMSE | **0.037** |

The hybrid approach uses the CNN for robust class identification and the PLSR baseline for precise mass-fraction estimation, offering a "best-of-both-worlds" analytical tool.

---

## 🛠️ Installation

This project uses `uv` for lightning-fast dependency management.

```bash
# Clone the repository
git clone https://github.com/Hipdarius/VERA.git
cd VERA

# Install dependencies and sync environment
uv sync
```

---

## 🏃 Quick Start (The "Acceptance Test")

To verify the entire pipeline from data generation to quantized inference:

```bash
# 1. Download spectral endmembers (USGS)
python scripts/download_usgs.py

# 2. Generate a synthetic dataset (v2)
python scripts/generate_synth_dataset.py --n-samples 50 --measurements-per-sample 8 --out data/synth_v1.csv

# 3. Train models
python -m vera.train --model plsr --data data/synth_v1.csv --out runs/plsr/
python -m vera.train --model cnn  --data data/synth_v1.csv --epochs 20 --out runs/cnn/

# 4. Evaluate and Quantize
python -m vera.evaluate --run runs/cnn/ --data data/synth_v1.csv
python -m vera.quantize --run runs/cnn/ --out runs/cnn/model_int8.tflite

# 5. Launch the Dashboard
streamlit run apps/dashboard.py
```

---

## 📂 Project Structure

```text
vera/
├── apps/               # Interactive UIs (Streamlit, FastAPI)
├── data/               # Local cache for USGS/RELAB and generated datasets
├── runs/               # Trained models, metrics, and evaluation plots
├── scripts/            # CLI utilities for data ingestion and generation
├── src/vera/       # Core library:
│   ├── models/         # CNN (PyTorch) and PLSR architectures
│   ├── schema.py       # The hardware-software data contract (Locked)
│   ├── synth.py        # Physically-motivated spectral mixing engine
│   └── ...
└── tests/              # Unit tests (PyTest)
```

---

## 🧠 Design Principles

1. **Hardware-Software Contract**: All communication is enforced by the canonical schema in `schema.py`, ensuring seamless transition from synthetic data to real hardware.
2. **Sample-Level Integrity**: Train/Val/Test splits are strictly partitioned by `sample_id` (not individual measurements) to ensure the model generalizes to new mineral compositions.
3. **Physical Fidelity**: Synthetic spectra aren't random noise; they are built using linear-mixing models, shot noise, gain variation, and baseline drift.
4. **Transparency**: Every model run produces a full evaluation report with confusion matrices and 95% confidence intervals.

---

## 🔗 Related Resources
- **USGS Spectral Library**: Base endmembers for terrestrial minerals.
- **RELAB (Brown University)**: Actual lunar sample spectra for future validation.
- **ESRIC (Luxembourg)**: Contextual framework for lunar ISRU prospecting.

---
*Developed for the Jonk Fuerscher competition 2027.*
