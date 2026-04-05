#!/usr/bin/env python3
"""
Simulate the ESP32-S3 hardware output for the Regoscan project.

This script uses the project's ``synth.py`` engine to generate realistic
spectral data and outputs a JSON string to stdout every 2 seconds. This 
allows testing the ingestion bridge and web UI without physical hardware.
"""

import json
import time
import sys
from pathlib import Path

import numpy as np

# Ensure the in-repo src/ is importable
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from regoscan.schema import MINERAL_CLASSES, N_SPEC, N_LED
from regoscan.synth import (
    load_endmembers, 
    fractions_for_class, 
    synth_measurement, 
    NoiseConfig,
    Endmembers,
    WAVELENGTHS
)

def _get_toy_endmembers() -> Endmembers:
    """Fallback if USGS data isn't downloaded yet."""
    lam = WAVELENGTHS
    x = (lam - lam.min()) / (lam.max() - lam.min())
    # Simple linear spectra for the four endmembers
    olivine = 0.20 + 0.60 * x
    pyroxene = 0.15 + 0.50 * x
    anorthite = 0.55 + 0.30 * x
    ilmenite = 0.05 + 0.05 * x
    spectra = np.stack([olivine, pyroxene, anorthite, ilmenite], axis=0)
    return Endmembers(wavelengths_nm=lam, spectra=spectra, source="toy")

def main():
    print("Regoscan Mock ESP32 Started. Outputting JSON to stdout...", file=sys.stderr)
    
    # Setup generator
    rng = np.random.default_rng()
    
    # Try to load real endmembers, otherwise use toy ones
    em_path = ROOT / "data" / "cache" / "usgs_endmembers.npz"
    if em_path.exists():
        endmembers = load_endmembers(em_path)
    else:
        print("[warn] USGS endmembers not found. Using toy models.", file=sys.stderr)
        endmembers = _get_toy_endmembers()

    interval = 2.0  # seconds
    
    try:
        while True:
            # Pick a random class and generate fractions
            klass = rng.choice(MINERAL_CLASSES)
            fractions = fractions_for_class(klass, rng)
            
            # Generate one synthetic measurement
            # We use synth_measurement to get the final "clean" spec and noise
            measurement = synth_measurement(
                sample_id="mock_live",
                mineral_class=klass,
                fractions=fractions,
                endmembers=endmembers,
                rng=rng,
                noise=NoiseConfig()
            )
            
            # Pack into the ESP32 "Raw" format
            # In real hardware, broadband would be raw sensor, dark would be pedestal.
            # Here we just put the final spec in broadband and 0 in dark.
            payload = {
                "type": "measurement",
                "status": "ok",
                "integration_time_ms": int(measurement.integration_time_ms),
                "dark_spectrum": [0.0] * N_SPEC,
                "broadband_spectrum": [float(v) for v in measurement.spec],
                "narrowband_leds": [float(v) for v in measurement.led],
                "lif_value": float(measurement.lif_450lp)
            }
            
            # Output to stdout
            print(json.dumps(payload))
            sys.stdout.flush()
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMock terminated by user.", file=sys.stderr)

if __name__ == "__main__":
    main()
