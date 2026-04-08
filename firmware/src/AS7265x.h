#pragma once
// ─────────────────────────────────────────────────────────────────
// AS7265x Triad Spectral Sensor Driver
//
// The AMS AS7265x is a 3-chip multi-spectral sensor covering 18
// discrete bands from 410 nm to 940 nm over I2C.  It serves as a
// low-cost secondary arm for cross-validating the Hamamatsu
// C12880MA continuous spectrometer readings.
//
// This header declares the driver interface.  The implementation
// (.cpp) will be added once the dev board arrives and the I2C bus
// is verified on the logic analyzer.
//
// Usage:
//   vera::AS7265x sensor;
//   sensor.init();
//   if (sensor.isConnected()) {
//       float bands[vera::N_AS7265X_BANDS];
//       sensor.readAllBands(bands);
//   }
//
// Wire connection:
//   SDA -> PIN_AS7265X_SDA (GPIO 47)
//   SCL -> PIN_AS7265X_SCL (GPIO 48)
//   I2C address: 0x49
// ─────────────────────────────────────────────────────────────────

#include <cstdint>
#include "Config.h"

namespace vera {

/// Driver for the AMS AS7265x 18-channel spectral triad sensor.
///
/// The sensor is composed of three dies (AS72651 / AS72652 / AS72653),
/// each measuring 6 bands.  This class abstracts the virtual register
/// interface and returns calibrated floating-point micro-watts per
/// square centimetre for each band.
class AS7265x {
public:
    /// Initialize the I2C bus and configure the sensor with default
    /// gain and integration settings.  Must be called once in setup().
    void init();

    /// Return true if the sensor responds to an I2C address probe.
    /// Use this to gracefully degrade when the sensor is absent
    /// (e.g. during bench testing without the full sensor stack).
    bool isConnected();

    /// Read all 18 calibrated spectral bands into @p buffer.
    ///
    /// @param buffer  Caller-allocated array of at least N_AS7265X_BANDS
    ///                floats.  On return, buffer[0] corresponds to the
    ///                lowest wavelength band (~410 nm) and buffer[17]
    ///                to the highest (~940 nm).
    ///
    /// @note This performs a blocking read that takes approximately
    ///       (integration_time * 3) milliseconds because each die is
    ///       read sequentially.
    void readAllBands(float* buffer);
};

}  // namespace vera
