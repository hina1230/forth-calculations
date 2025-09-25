# Calculation #004: Sgr A* vs M87 Comparison

## Overview

Comparative analysis of jet predictions for Sgr A* (Milky Way) and M87 black holes using FORTH theory.

## Key Comparisons

| Parameter | M87 | Sgr A* | Ratio |
|-----------|-----|---------|-------|
| Mass | 6.5×10⁹ M☉ | 4.3×10⁶ M☉ | 1512:1 |
| Schwarzschild radius | 1.92×10¹³ m | 1.27×10¹⁰ m | 1512:1 |
| Jet velocity (R/r=1000) | 0.999999500c | 0.999999500c | 1:1 |
| W-axis period | 111.8 hours | 0.074 hours | 1512:1 |
| Disappearance time | 4.03 ms | 0.0027 ms | 1512:1 |

## Key Findings

- **Jet velocity invariance**: Both black holes produce identical jet velocities for the same R/r ratio
- **Scale relationship**: All timescales scale linearly with mass
- **W-axis period**: Sgr A* has much shorter period (4.44 minutes vs 4.66 days)

## Structure Parameters

- R/r ratio: 1000 (for both)
- Grid resolution: 50×50×8

## How to Run

```bash
python calculation.py
```

## Output Files

- `results.json`: Complete calculation results
- `sgr_a_comparison.html`: Interactive comparison visualization

## Physical Implications

1. The velocity invariance suggests a universal jet mechanism
2. Sgr A*'s rapid W-axis rotation could produce detectable periodic signals
3. The shorter timescales make Sgr A* ideal for testing FORTH predictions

## Dependencies

- numpy
- plotly
- json

## Author

Yoshiyuki Matsuyama