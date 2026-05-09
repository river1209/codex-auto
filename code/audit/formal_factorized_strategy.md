# Formal Factorized Strategy

Date: 2026-05-09

## Decision

The formal modeling route is the factorized temperature-field strategy.

Instead of independently predicting scalar `Tmean` and `DeltaT`, the task is to reconstruct the future 15-point FPV module-temperature field from sensorless exogenous inputs.

## Input

The model input does not use historical module-temperature sensor values.

Input tensor:

```text
X: (batch, lookback_steps, n_features)
```

Default:

```text
lookback_steps = 60
lookback_minutes = 60
```

Feature groups:

- meteorological and environmental channels
- periodic time features
- dynamic feature differences
- operating/electrical channels only when available and not high-missing
- feature-screened variants such as XGBoost top-k

## Output And Reconstruction

Two independent models are trained for the same temperature-field task:

1. Mean model:

```text
f_mean(X) -> Tmean
```

2. Residual model:

```text
f_residual(X) -> residual_field, shape (batch, 15)
```

The final 15-point temperature field is reconstructed as:

```text
T_field = Tmean + residual_field
```

The residual field is zero-centered per sample before reconstruction.

## Evaluation

Primary evaluation:

- full-field MAE/RMSE/Bias
- per-sensor MAE
- Tmean MAE/RMSE/R2
- Tstd MAE/RMSE/R2
- P95-P5 spread MAE/RMSE/R2
- MaxDev MAE/RMSE/R2

Secondary diagnostic only:

- Range / DeltaT

## Experimental Role Of Direct Strategy

Direct full-field prediction remains available in the script but is now an ablation baseline, not the formal route.

Direct scalar `DeltaT` prediction should not be used as a formal task.
