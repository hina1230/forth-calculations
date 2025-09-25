# Calculation #002: Magnetic Field Structure in 4D Torus

## Overview

Numerical simulation of magnetic field configuration in 4D torus structure with 3D projection visualization.

## Key Results

- **Jet velocity**: v/c = 0.999999500 (R/r=1000)
- **Helical pitch angle**: 19.29°
- **Number of helical turns**: 6283
- **W-axis period**: 111.8 hours (4.66 days)
- **Magnetic field strength**: 100 Gauss (reference)

## Structure Parameters

- Major radius: R = 1000 Rs
- Minor radius: r = 1 Rs
- Grid resolution: 50×50×8

## How to Run

```bash
python calculation.py
```

## Output Files

- `results.json`: Complete calculation results
- `magnetic_field_3d.html`: Interactive 3D visualization

## Physical Interpretation

The W-axis component creates a right-handed helical magnetic field structure that:
1. Provides stable confinement
2. Generates helical field lines
3. Facilitates jet acceleration

## Dependencies

- numpy
- plotly
- json

## Author

Yoshiyuki Matsuyama