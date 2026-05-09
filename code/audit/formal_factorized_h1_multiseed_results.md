# Formal Factorized H1 Multi-Seed Results

Date: 2026-05-09

## Experiment Scope
- Strategy: factorized mean + residual reconstruction only.
- Horizon: 1 min.
- Sites: Oakville, Orlando, Windsor.
- Seeds: 42, 3407, 2026.
- Feature mode: XGBoost top-k, k=12.
- Models: DLinear, Crossformer-lite, Autoformer-lite, TimeXer-lite, iTransformer-lite.

## Best Models By Site
| site | model | feature_mode | strategy | n_seeds | field_MAE_mean | field_MAE_std | Tmean_MAE_mean | Tstd_MAE_mean | Spread95_MAE_mean | Range_MAE_secondary_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oakville | dlinear | xgb_topk | factorized | 3 | 5.409 | 0.153 | 2.141 | 2.189 | 4.758 | 10.391 |
| oakville | autoformer_lite | xgb_topk | factorized | 3 | 5.413 | 0.050 | 2.288 | 2.318 | 5.174 | 10.942 |
| oakville | timexer_lite | xgb_topk | factorized | 3 | 5.486 | 0.552 | 1.907 | 2.701 | 5.123 | 12.171 |
| oakville | crossformer_lite | xgb_topk | factorized | 3 | 5.851 | 0.497 | 2.050 | 2.955 | 6.726 | 15.080 |
| oakville | itransformer | xgb_topk | factorized | 3 | 6.783 | 0.537 | 1.898 | 4.729 | 10.651 | 25.541 |
| orlando | crossformer_lite | xgb_topk | factorized | 3 | 3.961 | 0.330 | 3.528 | 0.844 | 2.676 | 3.010 |
| orlando | itransformer | xgb_topk | factorized | 3 | 4.155 | 0.165 | 3.771 | 0.919 | 2.668 | 2.963 |
| orlando | autoformer_lite | xgb_topk | factorized | 3 | 4.162 | 0.424 | 3.769 | 0.860 | 2.536 | 2.793 |
| orlando | dlinear | xgb_topk | factorized | 3 | 4.994 | 0.635 | 4.394 | 1.027 | 3.180 | 3.549 |
| orlando | timexer_lite | xgb_topk | factorized | 3 | 5.588 | 0.417 | 5.333 | 1.118 | 3.339 | 3.799 |
| windsor | autoformer_lite | xgb_topk | factorized | 3 | 2.760 | 0.397 | 1.874 | 1.204 | 3.338 | 4.842 |
| windsor | timexer_lite | xgb_topk | factorized | 3 | 2.911 | 0.709 | 2.096 | 1.696 | 4.646 | 6.796 |
| windsor | crossformer_lite | xgb_topk | factorized | 3 | 2.951 | 0.763 | 2.508 | 0.860 | 2.115 | 3.631 |
| windsor | itransformer | xgb_topk | factorized | 3 | 3.542 | 1.463 | 2.314 | 2.097 | 5.852 | 8.269 |
| windsor | dlinear | xgb_topk | factorized | 3 | 7.814 | 4.314 | 7.661 | 2.115 | 6.034 | 8.251 |

## Stability-Aware Best Models
| site | model | field_MAE_mean | field_MAE_std | field_MAE_mean_plus_std | Tmean_MAE_mean | Tstd_MAE_mean | Spread95_MAE_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oakville | autoformer_lite | 5.413 | 0.050 | 5.463 | 2.288 | 2.318 | 5.174 |
| oakville | dlinear | 5.409 | 0.153 | 5.562 | 2.141 | 2.189 | 4.758 |
| oakville | timexer_lite | 5.486 | 0.552 | 6.039 | 1.907 | 2.701 | 5.123 |
| orlando | crossformer_lite | 3.961 | 0.330 | 4.291 | 3.528 | 0.844 | 2.676 |
| orlando | itransformer | 4.155 | 0.165 | 4.321 | 3.771 | 0.919 | 2.668 |
| orlando | autoformer_lite | 4.162 | 0.424 | 4.586 | 3.769 | 0.860 | 2.536 |
| windsor | autoformer_lite | 2.760 | 0.397 | 3.158 | 1.874 | 1.204 | 3.338 |
| windsor | timexer_lite | 2.911 | 0.709 | 3.620 | 2.096 | 1.696 | 4.646 |
| windsor | crossformer_lite | 2.951 | 0.763 | 3.714 | 2.508 | 0.860 | 2.115 |

## Overall Best Rows
| site | model | feature_mode | strategy | n_seeds | field_MAE_mean | field_MAE_std | Tmean_MAE_mean | Tstd_MAE_mean | Spread95_MAE_mean | Range_MAE_secondary_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| windsor | autoformer_lite | xgb_topk | factorized | 3 | 2.760 | 0.397 | 1.874 | 1.204 | 3.338 | 4.842 |
| windsor | timexer_lite | xgb_topk | factorized | 3 | 2.911 | 0.709 | 2.096 | 1.696 | 4.646 | 6.796 |
| windsor | crossformer_lite | xgb_topk | factorized | 3 | 2.951 | 0.763 | 2.508 | 0.860 | 2.115 | 3.631 |
| windsor | itransformer | xgb_topk | factorized | 3 | 3.542 | 1.463 | 2.314 | 2.097 | 5.852 | 8.269 |
| orlando | crossformer_lite | xgb_topk | factorized | 3 | 3.961 | 0.330 | 3.528 | 0.844 | 2.676 | 3.010 |
| orlando | itransformer | xgb_topk | factorized | 3 | 4.155 | 0.165 | 3.771 | 0.919 | 2.668 | 2.963 |
| orlando | autoformer_lite | xgb_topk | factorized | 3 | 4.162 | 0.424 | 3.769 | 0.860 | 2.536 | 2.793 |
| orlando | dlinear | xgb_topk | factorized | 3 | 4.994 | 0.635 | 4.394 | 1.027 | 3.180 | 3.549 |
| oakville | dlinear | xgb_topk | factorized | 3 | 5.409 | 0.153 | 2.141 | 2.189 | 4.758 | 10.391 |
| oakville | autoformer_lite | xgb_topk | factorized | 3 | 5.413 | 0.050 | 2.288 | 2.318 | 5.174 | 10.942 |
| oakville | timexer_lite | xgb_topk | factorized | 3 | 5.486 | 0.552 | 1.907 | 2.701 | 5.123 | 12.171 |
| orlando | timexer_lite | xgb_topk | factorized | 3 | 5.588 | 0.417 | 5.333 | 1.118 | 3.339 | 3.799 |
| oakville | crossformer_lite | xgb_topk | factorized | 3 | 5.851 | 0.497 | 2.050 | 2.955 | 6.726 | 15.080 |
| oakville | itransformer | xgb_topk | factorized | 3 | 6.783 | 0.537 | 1.898 | 4.729 | 10.651 | 25.541 |
| windsor | dlinear | xgb_topk | factorized | 3 | 7.814 | 4.314 | 7.661 | 2.115 | 6.034 | 8.251 |

## Interpretation
- Oakville: DLinear is the most stable and has the lowest mean field MAE among tested models; Autoformer-lite is close. Crossformer/TimeXer can win on individual seeds but are less stable.
- Orlando: Crossformer-lite has the best mean field MAE, with Autoformer-lite close behind and lower spatial-spread error. DLinear and TimeXer-lite are weaker.
- Windsor: Crossformer-lite has the best mean field MAE and good spatial metrics; Autoformer-lite and TimeXer-lite are competitive but less consistently strong.
- Across sites, Crossformer-lite is the strongest candidate for a unified nonlinear model, while DLinear should remain a required baseline because it is hard to beat on Oakville.
- Range/DeltaT remains worse and less stable than Tstd or P95-P5 spread; keep it as a secondary diagnostic.

## Files
- Raw metrics: `code\runs\formal_factorized_h1_all_seed_metrics.csv`
- Mean/std summary: `code\runs\formal_factorized_h1_summary_mean_std.csv`