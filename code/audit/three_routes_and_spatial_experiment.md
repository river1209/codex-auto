# Three Routes and Spatial Interpolation Experiment

Date: 2026-05-09

## Three Routes

### Route 1: Data Expansion and Site Coverage

Purpose: make the study less site-specific.

Actions:

- Add `FPV_Altamonte_FL_data.csv` once available.
- Keep Oakville, Orlando, Windsor, and Altamonte as floating-PV sites.
- Use FSEC land baseline data only as an auxiliary contrast, not as the main
  FPV field-forecasting dataset, unless its module-temperature layout is
  explicitly comparable.

Expected evidence:

- Same sensorless input rule across sites.
- Same 15-point field target where available.
- Report site-specific and pooled results.

### Route 2: Cross-Site Generalization

Purpose: evaluate whether the model learns transferable FPV thermal behavior
rather than only fitting one site.

Actions:

- Train on multiple FPV sites and test on a held-out site.
- Compare with site-specific training.
- Add site metadata or site embeddings only if cross-site performance is weak.

Expected evidence:

- Leave-one-site-out results.
- Error breakdown by site and weather/irradiance regime.
- Discussion of domain shift caused by module type, climate, water body, and
  sensor layout.

### Route 3: Spatial Field and Thermal Non-Uniformity

Purpose: turn the work from scalar module-temperature prediction into
sensorless temperature-field reconstruction.

Actions:

- Use relative sensor locations from `FPV_metafile_mm.xlsx`.
- Plot observed/predicted/error temperature fields.
- Evaluate Tmean, Tstd, P95-P5 spread, max deviation, and per-location errors.
- Avoid using raw DeltaT/range as the main instability target because it is
  too sensitive to single-sensor extremes.

Expected evidence:

- Interpolated relative field figures.
- Spatial error maps.
- Case studies for typical, high-spread, and high-error samples.

## Spatial Interpolation Experiment Completed

Script:

- `code/scripts/plot_temperature_field_maps.py`
- `code/scripts/run_spatial_field_visual_experiment.py`

Input predictions:

- Oakville: TimeXer-lite, h=1 min.
- Orlando: Crossformer-lite, h=1 min.
- Windsor: Autoformer-lite, h=1 min.

Generated output root:

- `code/figures/temperature_fields/spatial_experiment_20260509`

For each site, the batch script generated:

- `*_median_spread_interpolated_field.png`: representative thermal field.
- `*_max_spread_interpolated_field.png`: strongest thermal non-uniformity
  among saved prediction rows.
- `*_max_field_mae_interpolated_field.png`: largest field prediction error
  among saved prediction rows.
- `*_per_sensor_mae_map.png`: relative-position MAE map.
- `*_per_sensor_relative_error.csv`: per-sensor MAE/RMSE/Bias with relative
  coordinates.
- `*_selected_samples.csv`: selected timestamp and field-level statistics.

## Preliminary Spatial Error Summary

These results are computed from saved prediction CSV samples. The current
training script saves the first 2000 test predictions by default, so the
selected `max_spread` and `max_field_mae` samples are local to those saved rows,
not necessarily global over the whole test set.

| Site | Model | Mean sensor MAE (degC) | Max sensor MAE (degC) | Max-error channel |
|---|---|---:|---:|---|
| Oakville | TimeXer-lite | 4.017 | 8.163 | NEPTMC |
| Orlando | Crossformer-lite | 4.994 | 6.472 | SEPTMC |
| Windsor | Autoformer-lite | 2.155 | 3.974 | NEPTMA |

## Interpretation

The interpolated figures are useful for paper figures and diagnostics, but the
relative coordinate system should be explicitly described as metadata-derived.
The metadata provides location codes and A/B/C replicate numbers, not measured
physical coordinates.

The maps also reveal that the model can under-represent extreme local thermal
gradients, especially in high-spread cases. This supports the next method
improvement: keep the current best temporal models, but add a spatial
regularization or sensor-location-aware residual head and evaluate it on
Tstd/P95-P5/MaxDev rather than only field MAE.

