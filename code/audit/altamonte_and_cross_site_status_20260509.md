# Altamonte and Cross-Site Status

Date: 2026-05-09

## What Completed

### Altamonte Single-Site Experiment

Completed run:

- `code/runs/altamonte_14pt_h1_seed42_noirr_models/metrics_field.csv`

Important data handling:

- Altamonte was added as a fourth FPV site.
- `NEPTMB` was excluded from Altamonte's single-site target because it has
  insufficient chronological validation/test coverage.
- Altamonte single-site target therefore uses 14 module-temperature points.
- The irradiance threshold was disabled for Altamonte because `FPOAI1 >= 200`
  leaves no validation/test split; the later period still has valid temperature
  and weather measurements.

Results, h=1 min, seed=42:

| Model | Field MAE (degC) | Tmean MAE | Tstd MAE | Spread95 MAE |
|---|---:|---:|---:|---:|
| DLinear | 2.474 | 1.479 | 2.029 | 3.430 |
| Autoformer-lite | 2.337 | 1.471 | 2.334 | 4.005 |
| Crossformer-lite | 2.438 | 1.400 | 2.349 | 3.712 |
| TimeXer-lite | 2.532 | 1.457 | 2.663 | 4.137 |

Best field MAE: Autoformer-lite.

### Interrupted Full Cross-Site Run

The first full four-site cross-site run was interrupted when the workstation
froze.

Partial metrics exist:

- `code/runs/cross_site_common8_four_sites_seed42/metrics_cross_site.csv`

Completed before interruption:

| Held-out site | Model | Field MAE (degC) | Tmean MAE | Tstd MAE | Spread95 MAE |
|---|---|---:|---:|---:|---:|
| Altamonte | DLinear | 7.705 | 6.936 | 2.735 | 6.125 |
| Altamonte | Autoformer-lite | 7.201 | 5.989 | 3.048 | 6.700 |

This run did not complete Oakville, Orlando, or Windsor held-out testing.

## Low-Occupancy GPU Mode Added

Added script options:

- `--torch-num-threads`: limits PyTorch CPU thread use.
- `--cuda-memory-fraction`: caps PyTorch CUDA memory fraction.
- `--epoch-pause-sec`: adds a short pause between epochs.

These do not guarantee a fixed GPU-utilization percentage, but they reduce
memory pressure, CPU contention, and continuous kernel pressure.

## Completed Lightweight Cross-Site Screening

Completed run:

- `code/runs/cross_site_common8_four_sites_seed42_light_gpu/metrics_cross_site.csv`

Configuration:

- Target layout: `common8`
- Sites: Altamonte, Oakville, Orlando, Windsor
- Target points: `NWPTMA`, `NWPTMB`, `NWPTMC`, `NEPTMA`, `NEPTMC`,
  `MDPTMA`, `MDPTMB`, `MDPTMC`
- Held-out evaluation: leave-one-site-out
- Training samples: 45,000 combined per held-out split
- Validation samples: 9,000 combined per held-out split
- Test samples: 12,000 held-out samples
- GPU-friendly settings: batch size 512, d_model 48, one layer, two CPU
  threads, CUDA memory fraction 0.65, 0.2 s epoch pause.

Best valid model by held-out site:

| Held-out site | Best model | Field MAE (degC) | Tmean MAE | Tstd MAE | Spread95 MAE |
|---|---|---:|---:|---:|---:|
| Altamonte | Crossformer-lite | 5.033 | 4.311 | 2.359 | 5.264 |
| Oakville | DLinear | 4.936 | 3.610 | 3.467 | 8.034 |
| Orlando | TimeXer-lite | 3.744 | 3.384 | 1.218 | 3.326 |
| Windsor | TimeXer-lite | 3.489 | 1.773 | 3.392 | 8.016 |

One invalid result:

- Held-out Oakville with Crossformer-lite produced NaN and should be excluded
  from valid comparisons unless rerun with a smaller learning rate or gradient
  stability changes.

## Interpretation for the Paper

The cross-site task is substantially harder than site-specific testing, which
is expected and important for the paper. The result supports emphasizing:

1. Site-specific sensorless temperature-field forecasting is feasible.
2. Cross-region generalization is possible but shows clear domain shift.
3. Different architectures transfer differently across held-out climates and
   layouts.
4. Future full-scale results should rerun the common8 leave-one-site-out matrix
   with conservative GPU settings and multiple seeds.

