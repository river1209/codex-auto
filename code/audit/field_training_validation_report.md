# Sensorless Temperature-Field Training Validation Report

Date: 2026-05-08

## Runs Included
- `field_oakville_h1_seed42_matrix`
- `field_oakville_horizons_seed42_focus`
- `field_orlando_h1_seed42_focus_miss60`
- `field_windsor_h1_seed42_focus`

## Best Field MAE Rows
| run_name | site | horizon_min | feature_mode | strategy | model | field_MAE | Tmean_MAE | Tstd_MAE | Spread95_MAE | Range_MAE_secondary | n_features | test_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| field_oakville_h1_seed42_matrix | oakville | 1 | xgb_topk | factorized | dlinear | 5.142 | 2.416 | 2.124 | 4.567 | 9.759 | 12 | 61640 |
| field_oakville_h1_seed42_matrix | oakville | 1 | raw | factorized | dlinear | 5.205 | 2.295 | 2.215 | 5.030 | 10.236 | 20 | 61640 |
| field_oakville_horizons_seed42_focus | oakville | 1 | xgb_topk | direct | dlinear | 5.280 | 2.101 | 2.143 | 4.717 | 9.976 | 12 | 61640 |
| field_oakville_h1_seed42_matrix | oakville | 1 | xgb_topk | direct | dlinear | 5.349 | 2.028 | 2.142 | 4.658 | 10.111 | 12 | 61640 |
| field_oakville_horizons_seed42_focus | oakville | 1 | xgb_topk | factorized | dlinear | 5.443 | 1.998 | 2.093 | 4.351 | 9.941 | 12 | 61640 |
| field_oakville_horizons_seed42_focus | oakville | 5 | raw | factorized | dlinear | 5.232 | 2.523 | 2.214 | 4.874 | 10.682 | 20 | 61103 |
| field_oakville_horizons_seed42_focus | oakville | 5 | raw | direct | dlinear | 5.347 | 2.271 | 2.171 | 4.858 | 10.602 | 20 | 61103 |
| field_oakville_horizons_seed42_focus | oakville | 5 | xgb_topk | factorized | dlinear | 5.494 | 2.298 | 2.073 | 4.433 | 9.923 | 12 | 61103 |
| field_oakville_horizons_seed42_focus | oakville | 5 | xgb_topk | direct | dlinear | 5.603 | 2.183 | 2.224 | 5.362 | 10.210 | 12 | 61103 |
| field_oakville_horizons_seed42_focus | oakville | 5 | xgb_topk | factorized | itransformer | 6.227 | 2.019 | 3.844 | 8.365 | 20.957 | 12 | 61103 |
| field_oakville_horizons_seed42_focus | oakville | 10 | xgb_topk | factorized | dlinear | 5.663 | 2.560 | 2.238 | 4.683 | 10.930 | 12 | 60571 |
| field_oakville_horizons_seed42_focus | oakville | 10 | xgb_topk | direct | dlinear | 5.817 | 2.762 | 2.267 | 5.192 | 10.984 | 12 | 60571 |
| field_oakville_horizons_seed42_focus | oakville | 10 | raw | factorized | dlinear | 5.868 | 3.345 | 2.207 | 4.977 | 10.170 | 20 | 60571 |
| field_oakville_horizons_seed42_focus | oakville | 10 | xgb_topk | factorized | itransformer | 6.154 | 2.227 | 3.584 | 7.788 | 19.686 | 12 | 60571 |
| field_oakville_horizons_seed42_focus | oakville | 10 | raw | direct | dlinear | 6.215 | 3.078 | 2.258 | 5.383 | 10.455 | 20 | 60571 |
| field_oakville_horizons_seed42_focus | oakville | 30 | xgb_topk | direct | dlinear | 5.843 | 3.142 | 2.321 | 5.455 | 10.989 | 12 | 58446 |
| field_oakville_horizons_seed42_focus | oakville | 30 | raw | factorized | dlinear | 6.009 | 3.131 | 2.251 | 5.015 | 11.089 | 20 | 58446 |
| field_oakville_horizons_seed42_focus | oakville | 30 | xgb_topk | factorized | dlinear | 6.274 | 3.081 | 2.384 | 5.461 | 11.549 | 12 | 58446 |
| field_oakville_horizons_seed42_focus | oakville | 30 | raw | factorized | itransformer | 6.344 | 3.157 | 3.339 | 7.487 | 17.548 | 20 | 58446 |
| field_oakville_horizons_seed42_focus | oakville | 30 | raw | direct | dlinear | 6.377 | 3.185 | 2.470 | 6.325 | 11.332 | 20 | 58446 |
| field_orlando_h1_seed42_focus_miss60 | orlando | 1 | xgb_topk | factorized | itransformer | 3.473 | 2.860 | 1.000 | 2.978 | 3.375 | 12 | 69556 |
| field_orlando_h1_seed42_focus_miss60 | orlando | 1 | xgb_topk | direct | itransformer | 3.697 | 3.057 | 1.156 | 3.362 | 3.704 | 12 | 69556 |
| field_orlando_h1_seed42_focus_miss60 | orlando | 1 | raw | factorized | itransformer | 3.816 | 3.499 | 0.914 | 2.559 | 2.606 | 24 | 69556 |
| field_orlando_h1_seed42_focus_miss60 | orlando | 1 | raw | direct | itransformer | 4.075 | 3.685 | 1.236 | 3.480 | 3.704 | 24 | 69556 |
| field_orlando_h1_seed42_focus_miss60 | orlando | 1 | xgb_topk | direct | dlinear | 4.382 | 3.724 | 1.241 | 3.656 | 4.276 | 12 | 69556 |
| field_windsor_h1_seed42_focus | windsor | 1 | xgb_topk | factorized | itransformer | 3.009 | 2.708 | 1.032 | 2.596 | 4.522 | 12 | 61079 |
| field_windsor_h1_seed42_focus | windsor | 1 | xgb_topk | direct | itransformer | 3.405 | 1.737 | 2.278 | 7.175 | 8.974 | 12 | 61079 |
| field_windsor_h1_seed42_focus | windsor | 1 | raw | direct | itransformer | 4.047 | 2.363 | 2.942 | 8.666 | 11.193 | 20 | 61079 |
| field_windsor_h1_seed42_focus | windsor | 1 | xgb_topk | factorized | dlinear | 5.441 | 5.212 | 0.893 | 2.301 | 3.892 | 12 | 61079 |
| field_windsor_h1_seed42_focus | windsor | 1 | xgb_topk | direct | dlinear | 5.860 | 5.625 | 1.140 | 3.273 | 4.613 | 12 | 61079 |

## Best Tmean MAE Rows
| run_name | site | horizon_min | feature_mode | strategy | model | field_MAE | Tmean_MAE | Tstd_MAE | Spread95_MAE | Range_MAE_secondary | n_features | test_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| field_oakville_h1_seed42_matrix | oakville | 1 | xgb_topk | factorized | patch_transformer | 6.578 | 1.727 | 5.358 | 11.566 | 29.155 | 12 | 61640 |
| field_oakville_horizons_seed42_focus | oakville | 1 | xgb_topk | factorized | itransformer | 7.107 | 1.748 | 5.318 | 12.132 | 29.391 | 12 | 61640 |
| field_oakville_horizons_seed42_focus | oakville | 1 | raw | direct | itransformer | 5.599 | 1.749 | 2.795 | 6.023 | 14.778 | 20 | 61640 |
| field_oakville_horizons_seed42_focus | oakville | 5 | raw | direct | patch_transformer | 6.605 | 2.002 | 4.132 | 7.953 | 22.933 | 20 | 61103 |
| field_oakville_horizons_seed42_focus | oakville | 5 | xgb_topk | factorized | itransformer | 6.227 | 2.019 | 3.844 | 8.365 | 20.957 | 12 | 61103 |
| field_oakville_horizons_seed42_focus | oakville | 5 | xgb_topk | direct | patch_transformer | 7.198 | 2.026 | 5.505 | 11.179 | 30.119 | 12 | 61103 |
| field_oakville_horizons_seed42_focus | oakville | 10 | raw | factorized | patch_transformer | 7.034 | 2.168 | 5.743 | 12.640 | 31.563 | 20 | 60571 |
| field_oakville_horizons_seed42_focus | oakville | 10 | xgb_topk | factorized | itransformer | 6.154 | 2.227 | 3.584 | 7.788 | 19.686 | 12 | 60571 |
| field_oakville_horizons_seed42_focus | oakville | 10 | xgb_topk | direct | itransformer | 7.260 | 2.304 | 5.474 | 11.655 | 29.939 | 12 | 60571 |
| field_oakville_horizons_seed42_focus | oakville | 30 | raw | direct | patch_transformer | 7.672 | 2.390 | 6.300 | 14.243 | 34.373 | 20 | 58446 |
| field_oakville_horizons_seed42_focus | oakville | 30 | xgb_topk | direct | itransformer | 6.822 | 2.408 | 4.730 | 10.198 | 26.173 | 12 | 58446 |
| field_oakville_horizons_seed42_focus | oakville | 30 | raw | direct | itransformer | 6.773 | 2.564 | 4.191 | 9.467 | 22.643 | 20 | 58446 |
| field_orlando_h1_seed42_focus_miss60 | orlando | 1 | xgb_topk | factorized | itransformer | 3.473 | 2.860 | 1.000 | 2.978 | 3.375 | 12 | 69556 |
| field_orlando_h1_seed42_focus_miss60 | orlando | 1 | xgb_topk | direct | itransformer | 3.697 | 3.057 | 1.156 | 3.362 | 3.704 | 12 | 69556 |
| field_orlando_h1_seed42_focus_miss60 | orlando | 1 | raw | factorized | itransformer | 3.816 | 3.499 | 0.914 | 2.559 | 2.606 | 24 | 69556 |
| field_windsor_h1_seed42_focus | windsor | 1 | xgb_topk | direct | itransformer | 3.405 | 1.737 | 2.278 | 7.175 | 8.974 | 12 | 61079 |
| field_windsor_h1_seed42_focus | windsor | 1 | raw | direct | itransformer | 4.047 | 2.363 | 2.942 | 8.666 | 11.193 | 20 | 61079 |
| field_windsor_h1_seed42_focus | windsor | 1 | xgb_topk | factorized | itransformer | 3.009 | 2.708 | 1.032 | 2.596 | 4.522 | 12 | 61079 |

## Best Tstd MAE Rows
| run_name | site | horizon_min | feature_mode | strategy | model | field_MAE | Tmean_MAE | Tstd_MAE | Spread95_MAE | Range_MAE_secondary | n_features | test_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| field_oakville_horizons_seed42_focus | oakville | 1 | xgb_topk | factorized | dlinear | 5.443 | 1.998 | 2.093 | 4.351 | 9.941 | 12 | 61640 |
| field_oakville_h1_seed42_matrix | oakville | 1 | xgb_topk | factorized | dlinear | 5.142 | 2.416 | 2.124 | 4.567 | 9.759 | 12 | 61640 |
| field_oakville_h1_seed42_matrix | oakville | 1 | nmf | factorized | dlinear | 6.060 | 3.155 | 2.140 | 4.335 | 9.967 | 8 | 61640 |
| field_oakville_horizons_seed42_focus | oakville | 5 | xgb_topk | factorized | dlinear | 5.494 | 2.298 | 2.073 | 4.433 | 9.923 | 12 | 61103 |
| field_oakville_horizons_seed42_focus | oakville | 5 | raw | direct | dlinear | 5.347 | 2.271 | 2.171 | 4.858 | 10.602 | 20 | 61103 |
| field_oakville_horizons_seed42_focus | oakville | 5 | raw | factorized | dlinear | 5.232 | 2.523 | 2.214 | 4.874 | 10.682 | 20 | 61103 |
| field_oakville_horizons_seed42_focus | oakville | 10 | raw | factorized | dlinear | 5.868 | 3.345 | 2.207 | 4.977 | 10.170 | 20 | 60571 |
| field_oakville_horizons_seed42_focus | oakville | 10 | xgb_topk | factorized | dlinear | 5.663 | 2.560 | 2.238 | 4.683 | 10.930 | 12 | 60571 |
| field_oakville_horizons_seed42_focus | oakville | 10 | raw | direct | dlinear | 6.215 | 3.078 | 2.258 | 5.383 | 10.455 | 20 | 60571 |
| field_oakville_horizons_seed42_focus | oakville | 30 | raw | factorized | dlinear | 6.009 | 3.131 | 2.251 | 5.015 | 11.089 | 20 | 58446 |
| field_oakville_horizons_seed42_focus | oakville | 30 | xgb_topk | direct | dlinear | 5.843 | 3.142 | 2.321 | 5.455 | 10.989 | 12 | 58446 |
| field_oakville_horizons_seed42_focus | oakville | 30 | xgb_topk | factorized | dlinear | 6.274 | 3.081 | 2.384 | 5.461 | 11.549 | 12 | 58446 |
| field_orlando_h1_seed42_focus_miss60 | orlando | 1 | raw | factorized | itransformer | 3.816 | 3.499 | 0.914 | 2.559 | 2.606 | 24 | 69556 |
| field_orlando_h1_seed42_focus_miss60 | orlando | 1 | xgb_topk | factorized | itransformer | 3.473 | 2.860 | 1.000 | 2.978 | 3.375 | 12 | 69556 |
| field_orlando_h1_seed42_focus_miss60 | orlando | 1 | xgb_topk | factorized | dlinear | 5.403 | 4.872 | 1.067 | 3.306 | 3.648 | 12 | 69556 |
| field_windsor_h1_seed42_focus | windsor | 1 | xgb_topk | factorized | dlinear | 5.441 | 5.212 | 0.893 | 2.301 | 3.892 | 12 | 61079 |
| field_windsor_h1_seed42_focus | windsor | 1 | xgb_topk | factorized | itransformer | 3.009 | 2.708 | 1.032 | 2.596 | 4.522 | 12 | 61079 |
| field_windsor_h1_seed42_focus | windsor | 1 | xgb_topk | direct | dlinear | 5.860 | 5.625 | 1.140 | 3.273 | 4.613 | 12 | 61079 |

## Notes
- These are training/validation outputs from the new field-forecasting script, not manuscript-ready final results.
- Current runs use seed 42 only. Multi-seed confirmation is still required before formal claims.
- Oakville operating columns were dropped because they were 100% missing in the loaded main site CSV, so the current sensorless input is meteorological/time only after high-missing filtering.
- Orlando requires a relaxed missing-feature threshold (0.60) to keep meteorological features; otherwise most meteorological channels are removed and the model becomes time-feature dominated.
- Range/DeltaT is retained only as a secondary diagnostic; Tstd and P95-P5 spread are more stable spatial-distribution metrics.