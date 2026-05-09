# Final Factorized Training Validation Results

Date: 2026-05-09

## Scope Completed
- Formal route: factorized Tmean + residual reconstruction.
- Multi-seed validation: current three sites, 1 min horizon, seeds 42/3407/2026.
- Multi-horizon validation: current three sites, 1/5/10/30 min horizons, seed42.
- Feature mode: XGBoost top-k, k=12.

## Multi-Seed 1 Min Best Models
| site | model | n_seeds | field_MAE_mean | field_MAE_std | Tmean_MAE_mean | Tstd_MAE_mean | Spread95_MAE_mean | Range_MAE_secondary_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oakville | dlinear | 3 | 5.409 | 0.153 | 2.141 | 2.189 | 4.758 | 10.391 |
| oakville | autoformer_lite | 3 | 5.413 | 0.050 | 2.288 | 2.318 | 5.174 | 10.942 |
| oakville | timexer_lite | 3 | 5.486 | 0.552 | 1.907 | 2.701 | 5.123 | 12.171 |
| orlando | crossformer_lite | 3 | 3.961 | 0.330 | 3.528 | 0.844 | 2.676 | 3.010 |
| orlando | itransformer | 3 | 4.155 | 0.165 | 3.771 | 0.919 | 2.668 | 2.963 |
| orlando | autoformer_lite | 3 | 4.162 | 0.424 | 3.769 | 0.860 | 2.536 | 2.793 |
| windsor | autoformer_lite | 3 | 2.760 | 0.397 | 1.874 | 1.204 | 3.338 | 4.842 |
| windsor | timexer_lite | 3 | 2.911 | 0.709 | 2.096 | 1.696 | 4.646 | 6.796 |
| windsor | crossformer_lite | 3 | 2.951 | 0.763 | 2.508 | 0.860 | 2.115 | 3.631 |

## Multi-Horizon Best Rows
| site | horizon_min | model | field_MAE | Tmean_MAE | Tstd_MAE | Spread95_MAE | Range_MAE_secondary | test_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oakville | 1 | timexer_lite | 4.854 | 2.136 | 2.384 | 4.664 | 10.267 | 61640 |
| oakville | 1 | autoformer_lite | 5.555 | 2.161 | 2.483 | 5.712 | 11.620 | 61640 |
| oakville | 1 | dlinear | 5.731 | 2.265 | 2.102 | 4.443 | 9.832 | 61640 |
| oakville | 5 | timexer_lite | 4.889 | 1.932 | 2.539 | 5.391 | 10.623 | 61103 |
| oakville | 5 | crossformer_lite | 5.491 | 2.108 | 2.459 | 5.264 | 11.422 | 61103 |
| oakville | 5 | dlinear | 5.575 | 2.327 | 2.153 | 4.657 | 10.196 | 61103 |
| oakville | 10 | autoformer_lite | 5.712 | 3.082 | 2.649 | 6.575 | 11.869 | 60571 |
| oakville | 10 | crossformer_lite | 6.030 | 2.339 | 3.433 | 6.674 | 19.090 | 60571 |
| oakville | 10 | dlinear | 6.213 | 3.373 | 2.218 | 4.830 | 10.713 | 60571 |
| oakville | 30 | autoformer_lite | 5.806 | 3.220 | 2.246 | 4.922 | 10.779 | 58446 |
| oakville | 30 | timexer_lite | 5.868 | 2.751 | 2.504 | 4.634 | 11.935 | 58446 |
| oakville | 30 | crossformer_lite | 6.029 | 2.772 | 2.371 | 5.145 | 11.477 | 58446 |
| orlando | 1 | autoformer_lite | 3.850 | 3.368 | 0.911 | 2.560 | 2.764 | 69556 |
| orlando | 1 | crossformer_lite | 4.053 | 3.571 | 0.972 | 3.115 | 3.437 | 69556 |
| orlando | 1 | dlinear | 5.524 | 5.027 | 1.058 | 3.172 | 3.464 | 69556 |
| orlando | 5 | crossformer_lite | 4.227 | 3.708 | 1.070 | 3.196 | 3.408 | 67745 |
| orlando | 5 | autoformer_lite | 4.371 | 3.863 | 0.958 | 2.899 | 3.187 | 67745 |
| orlando | 5 | dlinear | 5.567 | 5.085 | 1.001 | 3.069 | 3.491 | 67745 |
| orlando | 10 | autoformer_lite | 3.946 | 3.368 | 0.938 | 2.823 | 3.196 | 66710 |
| orlando | 10 | crossformer_lite | 4.516 | 4.068 | 0.849 | 2.723 | 2.967 | 66710 |
| orlando | 10 | dlinear | 5.503 | 4.995 | 1.051 | 3.412 | 3.848 | 66710 |
| orlando | 30 | autoformer_lite | 4.392 | 3.892 | 0.934 | 2.845 | 3.063 | 63978 |
| orlando | 30 | crossformer_lite | 4.788 | 4.290 | 1.048 | 3.123 | 3.478 | 63978 |
| orlando | 30 | timexer_lite | 5.957 | 5.682 | 1.011 | 3.158 | 3.538 | 63978 |
| windsor | 1 | autoformer_lite | 2.465 | 1.780 | 0.923 | 2.398 | 4.303 | 61079 |
| windsor | 1 | timexer_lite | 2.644 | 1.640 | 1.648 | 4.405 | 6.868 | 61079 |
| windsor | 1 | crossformer_lite | 2.850 | 2.327 | 0.861 | 2.097 | 3.694 | 61079 |
| windsor | 5 | autoformer_lite | 2.465 | 1.779 | 0.934 | 2.478 | 4.333 | 60620 |
| windsor | 5 | timexer_lite | 2.649 | 1.708 | 1.629 | 4.275 | 6.648 | 60620 |
| windsor | 5 | crossformer_lite | 2.666 | 2.196 | 0.946 | 2.211 | 3.964 | 60620 |
| windsor | 10 | autoformer_lite | 3.319 | 2.358 | 1.482 | 4.155 | 5.253 | 60096 |
| windsor | 10 | timexer_lite | 3.757 | 3.162 | 1.631 | 4.316 | 6.666 | 60096 |
| windsor | 10 | crossformer_lite | 3.810 | 3.325 | 0.955 | 2.164 | 4.101 | 60096 |
| windsor | 30 | autoformer_lite | 3.431 | 2.545 | 1.163 | 3.320 | 4.371 | 58323 |
| windsor | 30 | timexer_lite | 4.588 | 3.352 | 1.835 | 4.930 | 7.401 | 58323 |
| windsor | 30 | crossformer_lite | 5.332 | 5.041 | 1.172 | 2.825 | 4.866 | 58323 |

## Recommended Models By Site And Horizon
| site | horizon_min | best_model_seed42 | field_MAE | Tmean_MAE | Tstd_MAE | Spread95_MAE |
| --- | --- | --- | --- | --- | --- | --- |
| oakville | 1 | timexer_lite | 4.854 | 2.136 | 2.384 | 4.664 |
| oakville | 5 | timexer_lite | 4.889 | 1.932 | 2.539 | 5.391 |
| oakville | 10 | autoformer_lite | 5.712 | 3.082 | 2.649 | 6.575 |
| oakville | 30 | autoformer_lite | 5.806 | 3.220 | 2.246 | 4.922 |
| orlando | 1 | autoformer_lite | 3.850 | 3.368 | 0.911 | 2.560 |
| orlando | 5 | crossformer_lite | 4.227 | 3.708 | 1.070 | 3.196 |
| orlando | 10 | autoformer_lite | 3.946 | 3.368 | 0.938 | 2.823 |
| orlando | 30 | autoformer_lite | 4.392 | 3.892 | 0.934 | 2.845 |
| windsor | 1 | autoformer_lite | 2.465 | 1.780 | 0.923 | 2.398 |
| windsor | 5 | autoformer_lite | 2.465 | 1.779 | 0.934 | 2.478 |
| windsor | 10 | autoformer_lite | 3.319 | 2.358 | 1.482 | 4.155 |
| windsor | 30 | autoformer_lite | 3.431 | 2.545 | 1.163 | 3.320 |

## Main Findings
- Oakville: DLinear and Autoformer-lite are the most stable in multi-seed H1. In the seed42 horizon sweep, TimeXer-lite is strongest at 1/5 min, while Autoformer-lite is strongest at 10/30 min.
- Orlando: Crossformer-lite and Autoformer-lite dominate. Autoformer-lite is best or near-best across horizons and has strong spatial-spread metrics.
- Windsor: Autoformer-lite is the most robust choice across horizons; Crossformer-lite is competitive at short horizons but degrades at 30 min.
- DLinear remains a required baseline, but it fails badly on Windsor longer horizons, so it should not be the main model.
- Range/DeltaT remains less stable than Tstd and P95-P5 spread; keep it secondary.

## Updated Model Plan
- Main manuscript candidates: Autoformer-lite and Crossformer-lite under factorized reconstruction.
- Site-specific comparison: include TimeXer-lite because it performs well on Oakville/Windsor short-horizon cases.
- Required baseline: DLinear.
- Keep iTransformer-lite as a comparison model for 1 min only unless later results improve.

## Output Files
- Multi-seed raw metrics: `code\runs\formal_factorized_h1_all_seed_metrics.csv`
- Multi-seed mean/std: `code\runs\formal_factorized_h1_summary_mean_std.csv`
- Multi-horizon raw metrics: `code\runs\formal_factorized_horizons_seed42_metrics.csv`
- Multi-horizon summary: `code\runs\formal_factorized_horizons_seed42_summary.csv`
- Recommended models: `code\runs\formal_factorized_recommended_models_by_horizon.csv`