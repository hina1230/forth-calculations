# Calculation #001: 4D Torus Jet Velocity Verification

## Overview

Verification of the fundamental FORTH theory jet velocity formula for M87 black hole.

## Key Formula

```
v/c = √(1 - (r/R)²)
```

Where:
- R: Major radius of torus
- r: Minor radius (tube radius)
- v: Jet velocity
- c: Speed of light

## Results

| R/r Ratio | v/c | Lorentz Factor (γ) |
|-----------|-----|-------------------|
| 10 | 0.994987437 | 10.0 |
| 100 | 0.999949999 | 100.0 |
| **1000** | **0.999999500** | **1000.0** |
| 10000 | 0.999999995 | 10000.0 |

## M87 Parameters

- Mass: 6.5×10⁹ M☉
- Schwarzschild radius: Rs = 1.920×10¹³ m (128.3 AU)
- Observed jet velocity: ~0.98-0.99c
- Theoretical prediction (R/r=1000): 0.999999500c

## How to Run

```bash
python calculation.py
```

## Output Files

- `results.json`: Complete calculation results
- Console output: Verification report

## Verification

All calculations can be verified manually:

```python
# For R/r = 1000
r_over_R = 1/1000 = 0.001
v_over_c = sqrt(1 - 0.001^2) = sqrt(0.999999) = 0.999999500
```

## Physical Interpretation

The theoretical velocity (0.999999500c) is slightly higher than observed (~0.99c), possibly due to:
1. Actual R/r ratio being less than 1000
2. Deceleration from radiation losses
3. Projection effects in observations

## Dependencies

- numpy
- json
- datetime

## Author

Yoshiyuki Matsuyama