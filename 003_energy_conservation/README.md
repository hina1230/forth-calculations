# Calculation #003: Energy Conservation with Radiation

## Overview

Analysis of energy conservation in the 4D torus jet formation process, including realistic radiation losses.

## Key Results

- **Energy input**: Gravitational potential at accretion
- **Energy output**: Jet kinetic energy
- **Radiation losses**: 10% (realistic model)
- **Conservation check**: 90% efficiency achieved

## Energy Balance

```
Input Energy = Kinetic Energy + Radiation Losses
E_in = E_jet + E_rad
E_in = 0.9 E_jet + 0.1 E_jet
```

## Structure Parameters

- Major radius: R = 1000 Rs
- Minor radius: r = 1 Rs
- Accretion radius: R + r = 1001 Rs
- Jet velocity: v/c = 0.999999500

## How to Run

```bash
python calculation.py
```

## Output Files

- `results.json`: Complete calculation results
- `energy_flow_003.png`: Energy flow diagram

## Physical Interpretation

The 10% radiation loss represents:
1. Synchrotron radiation from accelerated particles
2. Thermal radiation from accretion disk
3. Energy dissipation through magnetic reconnection

This realistic efficiency is consistent with observed AGN jets.

## Dependencies

- numpy
- matplotlib
- json

## Author

Yoshiyuki Matsuyama