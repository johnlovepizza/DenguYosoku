# DenguYosoku (デング予測)

Leak-free weekly dengue nowcasting for 77 Thai provinces, built to the 8-step
methodology. LightGBM with a loss matched to the pre-registered metric,
two rebuild leakage proofs, three naive baselines, and a paired bootstrap CI.

**Headline result** — on a sealed 94-week holdout (2025-01-05 → 2026-10-18,
7,238 province-weeks), MAE **3.518** cases vs persistence **3.964**
→ **+11.3 % skill**, paired ΔMAE **−0.446** [95 % CI −0.650, −0.301].

## The 8 steps, and where each lives

| Step | What it does | Where in `DenguYosoku.py` |
|---|---|---|
| 1 FRAME | Five written questions; metric fixed **before** modelling | `FRAME` dict |
| 2 AUDIT | Duplicates, 7-day grid, negatives, leaky/broken columns | audit block |
| 3 SPLIT | Time-ordered train/test, test sealed | `TRAIN_END`, `EVAL_CUTOFF` |
| 4 PREPARE | 12 causal features + **two rebuild leakage proofs** | `build_features`, `n_diff` |
| 5 BASELINE | Persistence, seasonal naive, train-only climatology | `base_persist` / `base_season` / `base_clim` |
| 6 MODEL | Optional Optuna search + **loss-to-metric matching** on CV | `objective`, `cv_mae`, `CANDIDATES` |
| 7 EVALUATE | Sealed test scored once; paired week-bootstrap CI | `boot_ci` |
| 8 REPORT | Importance, predictions, `metrics.json` | Step 8 block |

## Install

```bash
git clone https://github.com/<you>/DenguYosoku.git
cd DenguYosoku
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.14.9 was used for the reported run.

## Quickstart

```bash
# 1. Put the data next to the script, then set CSV in DenguYosoku.py
#    CSV = Path("dengue_cut_allweeks_2003-18_Oct_2026.csv")

# 2. Train + evaluate (≈4 min as shipped, RUN_OPTUNA=False;
#    ≈100 min if you set RUN_OPTUNA=True with OPTUNA_TRIALS=100)
python DenguYosoku.py

# 3. Blind deployment rehearsal (a week with the answer removed)
python make_blind_holdout.py        # writes predict_week/predict_2026-10-25.csv
python predict_2026-10-25.py        # forecast it without ever seeing 'cases'
```

## Repository layout

```
.
├── DenguYosoku.py               # main 8-step pipeline (train → evaluate → report)
├── make_blind_holdout.py        # cut one week out WITH lags, WITHOUT the answer
├── predict_2026-10-25.py        # predict the blind week
├── dengue_cut_allweeks_2003-18_Oct_2026.csv   # not committed (see Data)
├── requirements.txt
└── README.md
```

## Outputs

```
DenguYosoku/
 ├── model.txt                    # LightGBM booster
 ├── holdout_predictions.csv      # province, week, cases, pred, persist, seasonal_naive, climatology, residual
 ├── feature_importance.csv       # gain, split, gain_pct
 └── metrics.json                 # every number, config, and library versions

predict_week/
 ├── answer_key.csv               # Actually cases of blind week
 ├── features.json                # Features requirements
 ├── predict_2026-10-25.csv       # Blind csv file an output from make_blind_holdout.py
 └── predictions.csv              # Output of predict_2026-10-25.py (predicted blind cases)
```

## Data

One row = one province in one epidemiological week (Sunday start).

| Column | Role |
|---|---|
| `province`, `week_start`, `week`, `year` | identifiers / calendar |
| `cases` | **target** — weekly reported dengue count |
| `w_apparent_temperature_mean` | weather source (used **lagged only**) |
| `population`, `cases_per_100k` | **excluded** — see below |

The reported run used 95,634 rows × 1,242 weeks, 2003-01-05 → 2026-10-18,
77 provinces, zero duplicates, zero grid gaps, zero negative counts.

**Never used as features:** `cases_per_100k` (contains the target),
`population` (national total ranges 62 M–385 M vs Thailand's ~66 M — a broken
unit/normalisation column), `prov_lat`, `prov_lon`, `year_ce`, `week_start`.

## Methodology highlights

### 1 · Two rebuild leakage proofs (not just name assertions)

```
LEAK TEST 1  delete-after-2019-09-29 : changed = 0
LEAK TEST 2  perturb-week-2019-09-29 : moved   = 0
```

The cut week is the 70th-percentile week of the modelling panel, not a fixed
date. Delete-the-future rebuilds every feature after truncating the panel;
nothing before the cut may move. Perturb-the-week changes only the cut week's
raw `cases` and weather; that week's features must not move. `NaN == NaN` is
treated as equal, and `prov_baseline` is held fixed as a fitted statistic (like
a scaler), so the tests compare against a stable target.

### 2 · Matching the training loss to the decision metric

MAE is minimised by the conditional **median**. Poisson fits the conditional
**mean**, which on overdispersed counts sits above the median and therefore
over-predicts quiet weeks. The loss was chosen on CV folds only:

**Quantile won at every round count.** The shipped booster is
`objective=quantile, alpha=0.5` at **600 rounds**. The switch also redistributed
what the model learned: `cases_L1` fell from 83 % → **49.8 %** of gain while the
multi-week slope (`cases_L2..L4`, now 33.7 %) and the seasonal harmonic gained
weight.

RMSE and Poisson deviance are higher for the median fit **by design** — they
are mean-forecast metrics, traded away for the pre-registered MAE.

## Known limitations

Read these before quoting any number.

1. **The shipped model uses LightGBM defaults, not the Step 6 tree shape.**
   Step 6 builds a Poisson model at `BEST` — with `RUN_OPTUNA=False` (the shipped
   setting) `BEST` is a *hand-set* config, not a searched one — then Step 6b
   re-fits at `BASE_P` with the loss-matched objective, overwriting both `model`
   and `iters`. `model.txt` confirms it: `num_leaves=31`, `learning_rate=0.1`,
   `max_depth=-1`, no bagging, no column subsampling, no L1/L2. Any Optuna search
   is therefore exploratory only. To ship the Step 6 shape, change the final
   `lgb.train` to `dict(BEST, **CANDIDATES[best_obj], metric="None")` — this
   changes results and requires a fresh disclosure.

2. **The holdout informed more than one configuration** (Poisson, then
   quantile α=0.5). The loss choice was made on CV alone, so selection is honest,
   but the final estimate carries mild optimism from repeated inspection.

3. **Year-straddle reporting anomaly.** The first holdout week (2025-01-05) has
   national actual 821 vs predicted 1,551 (residual −730) — the largest national
   residual in the holdout: the late-December reporting spike propagates through
   `cases_L1`. Isolated from all turning-point statistics.

4. **`prov_baseline` is fitted on the full training period**, so earlier CV folds
   see a denominator built partly from their own future. Test-set evaluation is
   unaffected; CV numbers are mildly optimistic.

5. **This is a nowcast, not a multi-week forecast.** Autoregressive features carry
   93.7 % of gain and the output trails turning points by one week.

6. **Excluded data:** `population` (broken), mosquito indices (sparse, with
   implausible Breteau values).

## Reproducibility

| Item | Value |
|---|---|
| Seed | **42** (NumPy, LightGBM) |
| Threads | **1** — LightGBM histogram binning is non-deterministic across threads |
| Split | `TRAIN_END = 2025-01-01`, `EVAL_CUTOFF = 2026-10-25` |
| CV | 5 expanding folds, 4-week purge gap, `MIN_TRAIN_FRAC = 0.45` |
| Search | `RUN_OPTUNA = False` as shipped |
| Shipped model | `quantile α=0.5`, 600 rounds, LightGBM default tree params |
| Metric | MAE vs persistence, fixed in Step 1 before any result |
| Versions | recorded in `metrics.json` |

Re-running with the same config reproduces every number bit-for-bit.