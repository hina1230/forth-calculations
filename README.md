# FORTH Theory Calculations

Four-dimensional Orthogonal Rotating Torus Hypothesis (FORTH) theory calculations for black hole jet physics.

## Overview

This repository contains verified numerical calculations supporting FORTH theory, which proposes a 4D torus structure around black holes with a spatial fourth dimension (W-axis). All calculations are transparent and reproducible.

## Key Predictions

- **Jet Velocity**: v/c = √(1 - (r/R)²)
- **W-axis Disappearance Time**: Δt = 2πr/c
- **Energy Conversion Efficiency**: Up to 40%

## Calculations

| # | Title | Key Result |
|---|-------|------------|
| 001 | [4D Torus Jet Velocity Verification](./001_torus_verification/) | v/c = 0.999999500 (R/r=1000) |
| 002 | [Magnetic Field Structure in 4D Torus](./002_magnetic_field/) | Helical structure with 19.29° pitch |
| 003 | [Relativistic Energy Conservation](./003_energy_conservation/) | 100% jet efficiency with 10% radiation |
| 004 | [Sgr A* and M87 Comparison](./004_sgr_a_comparison/) | Consistent across 1500× mass range |
| 005 | [Polarization Pattern Prediction](./005_polarization_prediction/) | 20-30% polarization, 16-day period |
| 006 | [W-axis Disappearance Time](./006_disappearance_time/) | 111.8 hours for M87 (r=1Rs) |
| 007 | [Theory Overview](./007_theory_overview/) | Complete theoretical framework |

## Requirements

```bash
pip install numpy scipy matplotlib plotly
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/hina1230/forth-calculations.git
cd forth-calculations

# Run example calculation
cd 001_torus_verification
python calculation.py
```

## Target Black Holes

- **M87**: Mass = 6.5×10⁹ M☉, Active jet
- **Sgr A***: Mass = 4.3×10⁶ M☉, No visible jet (explained by low accretion)

## Physical Constants

All calculations use CODATA 2018 values:
- G = 6.67430×10⁻¹¹ m³ kg⁻¹ s⁻²
- c = 299,792,458 m/s
- M☉ = 1.98847×10³⁰ kg

## Author

Yoshiyuki Matsuyama

## Citation

If you use these calculations in your research, please cite:
```
Matsuyama, Y. (2025). FORTH Theory Calculations for Black Hole Jet Physics.
GitHub: https://github.com/hina1230/forth-calculations
```

## License

MIT License - See [LICENSE](./LICENSE) for details

## Related Links

- [Theory Articles (Japanese)](https://pink-rose.info/forth/)
- [Original Theory Paper](in preparation)

## Contact

For questions or collaborations, please open an issue on GitHub.