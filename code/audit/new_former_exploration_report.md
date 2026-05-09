# New Former Exploration Report

Date: 2026-05-09

## Runs Included
- `field_oakville_new_formers_h1_seed42`
- `field_orlando_new_formers_h1_seed42`
- `field_windsor_new_formers_h1_seed42`

## Best Rows By Site
| run_name | site | horizon_min | feature_mode | strategy | model | field_MAE | Tmean_MAE | Tstd_MAE | Spread95_MAE | Range_MAE_secondary | n_features | test_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| field_oakville_new_formers_h1_seed42 | oakville | 1 | xgb_topk | factorized | crossformer_lite | 5.157 | 1.982 | 2.279 | 4.691 | 11.003 | 12 | 61640 |
| field_oakville_new_formers_h1_seed42 | oakville | 1 | xgb_topk | direct | crossformer_lite | 5.261 | 2.609 | 1.965 | 3.980 | 9.197 | 12 | 61640 |
| field_oakville_new_formers_h1_seed42 | oakville | 1 | xgb_topk | direct | autoformer_lite | 5.397 | 2.072 | 2.256 | 5.332 | 10.219 | 12 | 61640 |
| field_oakville_new_formers_h1_seed42 | oakville | 1 | xgb_topk | factorized | autoformer_lite | 5.422 | 2.120 | 2.394 | 5.360 | 11.135 | 12 | 61640 |
| field_oakville_new_formers_h1_seed42 | oakville | 1 | xgb_topk | direct | dlinear | 5.544 | 2.395 | 2.164 | 4.809 | 10.175 | 12 | 61640 |
| field_oakville_new_formers_h1_seed42 | oakville | 1 | xgb_topk | factorized | dlinear | 5.612 | 2.233 | 2.160 | 4.552 | 10.271 | 12 | 61640 |
| field_oakville_new_formers_h1_seed42 | oakville | 1 | xgb_topk | direct | timexer_lite | 5.925 | 2.118 | 2.269 | 5.303 | 10.202 | 12 | 61640 |
| field_oakville_new_formers_h1_seed42 | oakville | 1 | xgb_topk | factorized | timexer_lite | 6.787 | 1.891 | 5.089 | 9.652 | 28.302 | 12 | 61640 |
| field_orlando_new_formers_h1_seed42 | orlando | 1 | xgb_topk | factorized | autoformer_lite | 3.551 | 2.907 | 0.905 | 2.808 | 3.174 | 12 | 69556 |
| field_orlando_new_formers_h1_seed42 | orlando | 1 | xgb_topk | direct | crossformer_lite | 3.697 | 2.962 | 1.076 | 3.229 | 3.839 | 12 | 69556 |
| field_orlando_new_formers_h1_seed42 | orlando | 1 | xgb_topk | direct | autoformer_lite | 3.769 | 3.168 | 0.932 | 2.847 | 3.140 | 12 | 69556 |
| field_orlando_new_formers_h1_seed42 | orlando | 1 | xgb_topk | factorized | crossformer_lite | 4.018 | 3.525 | 1.006 | 2.967 | 3.327 | 12 | 69556 |
| field_orlando_new_formers_h1_seed42 | orlando | 1 | xgb_topk | direct | pathformer_lite | 4.048 | 3.376 | 1.222 | 3.497 | 3.800 | 12 | 69556 |
| field_orlando_new_formers_h1_seed42 | orlando | 1 | xgb_topk | direct | itransformer | 4.338 | 3.957 | 1.005 | 2.822 | 3.043 | 12 | 69556 |
| field_orlando_new_formers_h1_seed42 | orlando | 1 | xgb_topk | direct | dlinear | 4.497 | 3.864 | 1.210 | 3.640 | 4.425 | 12 | 69556 |
| field_orlando_new_formers_h1_seed42 | orlando | 1 | xgb_topk | direct | timexer_lite | 4.572 | 4.366 | 0.778 | 2.439 | 2.944 | 12 | 69556 |
| field_windsor_new_formers_h1_seed42 | windsor | 1 | xgb_topk | direct | timexer_lite | 2.247 | 1.459 | 0.931 | 2.453 | 4.376 | 12 | 61079 |
| field_windsor_new_formers_h1_seed42 | windsor | 1 | xgb_topk | factorized | itransformer | 2.286 | 1.777 | 1.027 | 2.609 | 4.423 | 12 | 61079 |
| field_windsor_new_formers_h1_seed42 | windsor | 1 | xgb_topk | factorized | autoformer_lite | 2.319 | 1.615 | 0.907 | 2.318 | 4.338 | 12 | 61079 |
| field_windsor_new_formers_h1_seed42 | windsor | 1 | xgb_topk | factorized | crossformer_lite | 2.390 | 1.839 | 0.989 | 2.313 | 4.128 | 12 | 61079 |
| field_windsor_new_formers_h1_seed42 | windsor | 1 | xgb_topk | direct | pathformer_lite | 2.643 | 1.917 | 0.880 | 2.339 | 3.902 | 12 | 61079 |
| field_windsor_new_formers_h1_seed42 | windsor | 1 | xgb_topk | factorized | timexer_lite | 2.663 | 1.825 | 1.539 | 4.128 | 6.561 | 12 | 61079 |
| field_windsor_new_formers_h1_seed42 | windsor | 1 | xgb_topk | direct | autoformer_lite | 2.699 | 1.792 | 0.994 | 2.801 | 4.096 | 12 | 61079 |
| field_windsor_new_formers_h1_seed42 | windsor | 1 | xgb_topk | direct | crossformer_lite | 2.812 | 2.171 | 0.949 | 2.387 | 4.286 | 12 | 61079 |

## Best Metric Per Model
| site | model | field_MAE | Tmean_MAE | Tstd_MAE | Spread95_MAE |
| --- | --- | --- | --- | --- | --- |
| oakville | crossformer_lite | 5.157 | 1.982 | 1.965 | 3.980 |
| oakville | autoformer_lite | 5.397 | 2.072 | 2.256 | 5.332 |
| oakville | dlinear | 5.544 | 2.233 | 2.160 | 4.552 |
| oakville | timexer_lite | 5.925 | 1.891 | 2.269 | 5.303 |
| oakville | itransformer | 7.183 | 1.764 | 5.539 | 12.275 |
| oakville | pathformer_lite | 7.482 | 1.810 | 5.878 | 12.013 |
| orlando | autoformer_lite | 3.551 | 2.907 | 0.905 | 2.808 |
| orlando | crossformer_lite | 3.697 | 2.962 | 1.006 | 2.967 |
| orlando | pathformer_lite | 4.048 | 3.376 | 1.145 | 3.369 |
| orlando | itransformer | 4.338 | 3.957 | 1.005 | 2.822 |
| orlando | dlinear | 4.497 | 3.864 | 1.140 | 3.512 |
| orlando | timexer_lite | 4.572 | 4.366 | 0.778 | 2.439 |
| windsor | timexer_lite | 2.247 | 1.459 | 0.931 | 2.453 |
| windsor | itransformer | 2.286 | 1.777 | 0.939 | 2.411 |
| windsor | autoformer_lite | 2.319 | 1.615 | 0.907 | 2.318 |
| windsor | crossformer_lite | 2.390 | 1.839 | 0.949 | 2.313 |
| windsor | pathformer_lite | 2.643 | 1.684 | 0.880 | 2.339 |
| windsor | dlinear | 5.878 | 5.692 | 0.918 | 2.380 |

## Current Interpretation
- Crossformer-style dimension-segment tokens improved Oakville field MAE relative to the previous DLinear/iTransformer baselines in this seed42 run.
- Autoformer-style decomposition was strongest on Orlando among the new-former batch, especially with factorized mean+residual reconstruction.
- TimeXer-style target-query cross attention was strongest on Windsor direct field MAE, suggesting explicit target/sensor queries may help when station-level spatial patterns are more stable; because the formal route is now factorized, this should be rechecked under factorized multi-seed runs.
- Pathformer-style multi-scale routing looked promising in smoke tests but did not dominate full-data seed42 runs; it remains an ablation candidate, not the main model yet.
- The best architecture is site dependent. The next formal stage should use factorized-only multi-seed confirmation and, once provided, Altamonte validation.

## Literature Mapping
- Autoformer: decomposition plus auto-correlation for long-term series forecasting. We adapted only the decomposition idea to external meteorological inputs.
- Crossformer: dimension-segment-wise representation and cross-dimension dependency. We adapted it to exogenous feature variables over time.
- Pathformer: multi-scale Transformer with adaptive pathways. We adapted it as gated multi-scale patch branches.
- TimeXer: Transformer forecasting with exogenous variables. We adapted it as target/sensor query cross-attention over exogenous patch tokens.

## Caution
- All runs here are seed42 only and should be treated as model exploration, not final manuscript evidence.
- The script implements task-adapted lightweight variants, not exact reproductions of the original papers.
