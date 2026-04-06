// ─────────────────────────────────────────────────────────────────
// Hamamatsu C12880MA Micro-Spectrometer Driver — Implementation
//
// Bit-bangs CLK/ST lines to drive the C12880MA linear image sensor.
// All timing uses delayMicroseconds(); the full readout cycle
// completes in ≤ integration_ms_ + 4 ms and must not be interrupted.
// ─────────────────────────────────────────────────────────────────

#include "C12880MA.h"

namespace regoscan {

void C12880MA::init() {
    pinMode(PIN_SPEC_CLK,   OUTPUT);
    pinMode(PIN_SPEC_ST,    OUTPUT);
    pinMode(PIN_SPEC_TRG,   INPUT);
    pinMode(PIN_SPEC_VIDEO, INPUT);

    digitalWrite(PIN_SPEC_CLK, LOW);
    digitalWrite(PIN_SPEC_ST,  LOW);

    analogReadResolution(12);  // 12-bit ADC: 0–4095
}

void C12880MA::setIntegrationTime(uint16_t ms) {
    if (ms < MIN_INTEGRATION_MS) {
        ms = MIN_INTEGRATION_MS;
    } else if (ms > MAX_INTEGRATION_MS) {
        ms = MAX_INTEGRATION_MS;
    }
    integration_ms_ = ms;
}

inline void C12880MA::clockPulse() {
    digitalWrite(PIN_SPEC_CLK, HIGH);
    delayMicroseconds(CLK_HALF_PERIOD_US);
    digitalWrite(PIN_SPEC_CLK, LOW);
    delayMicroseconds(CLK_HALF_PERIOD_US);
}

void C12880MA::readSpectrum(uint16_t* buffer) {
    // ── 1. Begin integration: raise ST while clocking ───────────
    // Per datasheet, ST must go HIGH during a CLK rising edge.
    digitalWrite(PIN_SPEC_CLK, LOW);
    delayMicroseconds(CLK_HALF_PERIOD_US);

    digitalWrite(PIN_SPEC_CLK, HIGH);
    delayMicroseconds(CLK_HALF_PERIOD_US);
    digitalWrite(PIN_SPEC_ST, HIGH);  // ST rises while CLK is HIGH

    digitalWrite(PIN_SPEC_CLK, LOW);
    delayMicroseconds(CLK_HALF_PERIOD_US);

    // ── 2. Continue clocking for the integration window ─────────
    // Convert integration_ms_ to microseconds for the delay loop.
    // Each clock cycle takes 2 * CLK_HALF_PERIOD_US µs.
    const uint32_t integration_us = static_cast<uint32_t>(integration_ms_) * 1000u;
    const uint32_t us_per_clock   = 2u * CLK_HALF_PERIOD_US;
    const uint32_t integration_clocks = integration_us / us_per_clock;

    for (uint32_t c = 0; c < integration_clocks; c++) {
        clockPulse();
    }

    // ── 3. Lower ST to end integration ──────────────────────────
    digitalWrite(PIN_SPEC_ST, LOW);

    // ── 4. Shift out leading dummy pixels ───────────────────────
    for (uint16_t i = 0; i < N_LEADING_DUMMY; i++) {
        clockPulse();
    }

    // ── 5. Read N_SPEC_PIXELS valid pixels ──────────────────────
    // Sample the analog VIDEO output on each clock cycle.
    for (uint16_t i = 0; i < N_SPEC_PIXELS; i++) {
        clockPulse();
        buffer[i] = static_cast<uint16_t>(analogRead(PIN_SPEC_VIDEO));
    }

    // ── 6. Shift out trailing dummy pixel(s) ────────────────────
    // Total clocks = N_LEADING_DUMMY + N_SPEC_PIXELS + trailing.
    const uint16_t trailing = N_TOTAL_CLOCKS - N_LEADING_DUMMY - N_SPEC_PIXELS;
    for (uint16_t i = 0; i < trailing; i++) {
        clockPulse();
    }
}

}  // namespace regoscan
