# Initial Sensorless Result Assessment

Date: 2026-05-08

## Inputs Reviewed

- Old script: `../FPV_CNN_LSTM_Attention/train_fpv_sensorless_reduced_sensor_1min (1).py`
- Result directory: `../FPV_CNN_LSTM_Attention/results_sensorless_reduced_sensor/SensorlessReduced_sensor_1min_seed42_20260508_151119`
- Key tables: `metrics_sci.csv`, `metrics_summary_mean_std.csv`, `best_by_task.csv`, `best_by_category_task.csv`

## What The Current Results Support

- The current Oakville-only scalar experiment supports the feasibility of sensorless mean temperature prediction.
- Sensorless Tmean at 1 min ahead:
  - Sensorless-Met: MAE 1.873 degC, RMSE 2.392 degC, R2 0.965
  - Sensorless-MetOP: MAE 1.999 degC, RMSE 2.542 degC, R2 0.961
- These are competitive enough for a preliminary baseline, but not yet formal SCI evidence because they are single-site, single-seed, one-horizon results.

## What The Current Results Do Not Support

- The current results do not support direct DeltaT prediction as a main contribution.
- Sensorless DeltaT at 1 min ahead has negative R2:
  - Sensorless-Met: MAE 10.914 degC, RMSE 13.525 degC, R2 -1.389
  - Sensorless-MetOP: MAE 10.858 degC, RMSE 13.558 degC, R2 -1.401
- Warning metrics for DeltaT thresholds are weak under sensorless inputs, with AUC near or below 0.5.

## Plan Change

- Main task should become sensorless multi-point temperature-field forecasting.
- The formal target should be the 15 temperature sensor vector at the forecast horizon.
- Tmean, Tstd, P95-P5 spread, and max deviation from mean should be derived from the predicted field.
- Range / DeltaT can be reported as a secondary diagnostic, not as the primary instability definition.

## New Script

- Added: `code/scripts/train_fpv_sensorless_field_models.py`
- Supports:
  - direct strategy: one model predicts the full field
  - factorized strategy: independent mean and residual models reconstruct the field
  - dlinear, mlp, lstm_attn, mscnn_lstm_attn, transformer, patch_transformer, itransformer

## Smoke Verification

Command used:

```powershell
E:\anaconda3\envs\patchtst\python.exe code\scripts\train_fpv_sensorless_field_models.py --smoke --models dlinear,mscnn_lstm_attn --strategies direct,factorized --epochs 1 --batch-size 256 --run-name smoke_field_test_noirr2 --max-rows 200000 --min-irradiance 0 --input-min-mean-irradiance 0
```

Smoke output exists at:

- `code/runs/smoke_field_test_noirr2/metrics_field.csv`

The smoke run only validates code paths. It is not a paper result.
