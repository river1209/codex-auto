# Revised Plan After Field-Forecasting Validation

Date: 2026-05-08

## Plan Changes

1. Make sensorless multi-point temperature-field forecasting the core task.
2. Use `Tmean`, `Tstd`, `P95-P5 spread`, and max deviation from mean as derived evaluation metrics.
3. Keep range / DeltaT only as a secondary diagnostic, not as the main "thermal instability" definition.
4. Use `factorized` as the formal field reconstruction strategy:
   - one model predicts field mean.
   - one model predicts spatial residuals.
   - `direct` remains only as an ablation baseline.
5. Use XGBoost top-k feature screening as a main feature-combination branch.
6. Keep NMF/nonnegative factorized features as an ablation branch only; current Oakville results do not justify making it a main method.

## Evidence From Current Runs

- Oakville 1 min full matrix: DLinear with XGBoost top-k and factorized reconstruction currently gives the best field MAE.
- Oakville multi-horizon focused run: DLinear remains strongest for field metrics across 1/5/10/30 min.
- Orlando and Windsor 1 min focused runs: iTransformer with XGBoost top-k and factorized reconstruction performs best for field MAE.
- Site behavior is therefore heterogeneous; the final paper should not claim that one architecture dominates universally until multi-seed and Altamonte confirmation are complete.

## Updated Formal Experiment Matrix

- Sites:
  - Current formal set: Oakville, Orlando, Windsor.
  - Add Altamonte when `FPV_Altamonte_FL_data.csv` is available.
  - Use `FSEC_Regional_Test_BaseLine_data.csv` later only for LPV comparison, not for the FPV core task.
- Horizons:
  - Oakville: 1, 5, 10, 30 min.
  - Other sites: start with 1 min; extend if the 1 min model is stable.
- Models:
  - Baselines: DLinear, MLP, LSTM-Attention.
  - Main candidates: multi-scale CNN-LSTM-Attention, Transformer, PatchTransformer/PatchTST-like, iTransformer-like.
- Feature modes:
  - raw meteorological/time features.
  - XGBoost top-k screened features.
  - NMF/nonnegative latent features as ablation.
- Reconstruction:
  - formal route: factorized mean + residual prediction.
  - ablation route: direct full-field prediction.

## Manuscript Implications

- The paper should emphasize temperature-field reconstruction and derived spatial thermal-distribution metrics.
- Current results suggest a useful story: factorized reconstruction is a defensible formal route, with simple linear temporal models strong at Oakville and fomer-style models more useful at Orlando/Windsor after feature screening.
- The method section should explain site-dependent missing-data handling, especially Orlando's meteorological availability.
- Claims must remain preliminary until multi-seed runs and Altamonte validation are complete.

## Required Next Confirmation

- Multi-seed runs for the current best candidates:
  - Oakville: `xgb_topk + dlinear + factorized/direct`.
  - Orlando/Windsor: `xgb_topk + itransformer + factorized/direct`.
- Add Altamonte once the file is provided.
- Compare against scalar Tmean-only results only as a baseline; do not mix them with field-forecasting metrics.
