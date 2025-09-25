# Calculation #005: Polarization Pattern Prediction

## Overview

Prediction of unique polarization patterns arising from 4D torus magnetic field structure.

## Key Predictions

- **Helical polarization**: 19.29° pitch angle
- **Rotation period**: 111.8 hours (M87)
- **Pattern evolution**: Continuous rotation in polarization plane
- **Observational signature**: Periodic modulation in linear polarization

## Magnetic Field Structure

- Helical field lines with 6283 turns
- Right-handed helical configuration
- Toroidal + poloidal + W-axis components
- Field strength: 100 Gauss (reference)

## Observable Features

1. **Polarization angle rotation**: Δφ = 360° per W-axis period
2. **Intensity modulation**: I(t) = I₀(1 + A·cos(ωt))
3. **Wavelength dependence**: λ-dependent Faraday rotation

## Structure Parameters

- Major radius: R = 1000 Rs
- Minor radius: r = 1 Rs
- Helical turns: N = 2πR/λ_helix = 6283

## How to Run

```bash
python calculation.py
```

## Output Files

- `results.json`: Complete calculation results
- `polarization_pattern_005.html`: Interactive polarization visualization

## Observational Test

The predicted polarization pattern can be tested using:
- ALMA polarimetric observations
- EHT multi-epoch monitoring
- Optical/IR polarimetry for jet regions

## Dependencies

- numpy
- plotly
- json

## Author

Yoshiyuki Matsuyama