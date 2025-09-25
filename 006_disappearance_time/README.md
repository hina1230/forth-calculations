# Calculation #006: W-Axis Disappearance Time

## Overview

Calculation of the characteristic disappearance time as matter transitions through the 4D torus structure.

## Key Formula

```
Δt = 2πr/c
```

Where:
- r: Minor radius of torus
- c: Speed of light
- Factor 2π: Complete circuit around minor circumference

## Results

### M87 Black Hole
- Minor radius: r = 1 Rs = 1.92×10¹³ m
- **Disappearance time: 4.03 ms**
- Observable as brief intensity drops

### Sgr A* (Milky Way)
- Minor radius: r = 1 Rs = 1.27×10¹⁰ m
- **Disappearance time: 0.0027 ms (2.7 μs)**
- Requires high-speed detectors

## Physical Interpretation

The disappearance represents:
1. Matter transitioning into W-axis dimension
2. Temporary invisibility from 3D perspective
3. Reappearance after completing W-axis circuit

## Observable Signatures

- Periodic intensity drops in accretion disk
- Quasi-periodic oscillations (QPOs)
- Correlation with jet launching events

## How to Run

```bash
python calculation.py
```

## Output Files

- `results.json`: Complete calculation results
- `disappearance_time_006.html`: Interactive visualization

## Detection Requirements

- Time resolution: < 1 ms for M87
- Time resolution: < 1 μs for Sgr A*
- Multi-wavelength monitoring recommended

## Dependencies

- numpy
- plotly
- json

## Author

Yoshiyuki Matsuyama