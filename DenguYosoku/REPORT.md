# DenguYosoku (予測) — Dengue Forecasting · Textbook Rebuild (8 steps)

## Provenance & the leak that was fixed
Pipeline: template -> feature_selection -> hey -> case_lag -> comprehensive -> super_combo.
super_combo's `w_apparent_temperature_mean` used week-t weather to predict week t
(a leak). It is now `w_apparent_temperature_mean_L1` (week t-1). Causality proven by
two rebuild tests: delete-the-future changes 0 values; perturb-the-week changes 0 values.
The province baseline is a fitted statistic (fit once on training rows, held constant),
identical in role to a fitted scaler.

## Split
TRAIN < 2025-01-01 (87,549 rows).
TEST 2025-01-05 .. 2026-10-25 (7,315 rows), sealed until Step 7.

## Model
LightGBM, Poisson, 600 rounds (time-ordered early stop on last 10% of train dates),
Optuna-tuned on training folds only (gap 4 weeks). Seed 42, 1 thread.

## Results — scored once on the sealed test
champion MAE=3.514 RMSE=6.566 R2=0.911 PDev=2.444
persistence=3.956  seasonal_naive=14.691  climatology=11.644
skill vs persistence: +11.2%
paired dMAE vs persistence: -0.441 [95% CI -0.631,-0.302] -> REAL

## Limitations
- Autoregressive features dominate (93% of gain).
- Year-straddling weeks (late Dec) show a 2024/25 reporting anomaly; expect larger errors there.
- Single 2025-26 holdout; the honest next step is prospective evaluation.
- population / mosquito indices excluded (broken / sparse).
