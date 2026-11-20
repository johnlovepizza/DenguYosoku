# DenguYosoku (デング予測)

Leak-free weekly dengue nowcasting for 77 Thai provinces, built to the 8-step
methodology. LightGBM with a loss matched to the pre-registered metric,
two rebuild leakage proofs, three naive baselines, and a paired bootstrap CI.

**Headline result** — on a sealed 94-week holdout (2025-01-05 → 2026-10-18,
7,238 province-weeks), MAE **3.518** cases vs persistence **3.964**
→ **+11.3 % skill**, paired ΔMAE −0.446 [95 % CI −0.650, −0.301].

## The 8 steps, and where each lives

| Step | What it does | Where in `DenguYosoku.py` |
| --- | --- | --- |
| 1 FRAME | Five written questions; metric fixed *before* modelling | `FRAME` dict |
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
pip install -r requirements.txt    # CORE ONLY — runs the whole pipeline
```

Core is five packages: `numpy`, `pandas`, `lightgbm`, `matplotlib`, `seaborn`.
Everything else is optional and commented out in `requirements.txt`:

```bash
pip install optuna tqdm            # only if RUN_OPTUNA=True (False as shipped)
pip install plotly                 # only for the two interactive HTML heatmaps
pip install "plotly>=6" "kaleido>=1.0"   # only for optuna_*.png (needs Chrome)
```

Every optional import is wrapped in `try/except`, so a missing package prints a
notice and skips that one output instead of crashing the run.

Python 3.14.9 was used for the reported run.

## Quickstart

```bash
# 1. Put both CSVs next to the scripts; CSV is set in DenguYosoku.py:
#    CSV = Path("dengue_cut_allweeks_2003-18_Oct_2026.csv")

python DenguYosoku.py            # train + evaluate  -> DenguYosoku/  (KEEP this folder)
python make_blind_holdout.py     # blind rehearsal   -> predict_week/predict_2026-10-25.csv
python predict_2026-10-25.py     # forecast it       -> predict_week/predictions.csv
python plot_all.py               # all 8 plotting/prediction scripts, one command
```

`DenguYosoku.py` takes ≈4 min as shipped (`RUN_OPTUNA=False`); ≈100 min with
`RUN_OPTUNA=True, OPTUNA_TRIALS=100`.

`plot_all.py` runs each embedded script in its **own fresh subprocess**, so no
matplotlib/seaborn/font state leaks between them and every PNG/CSV is exactly
what the standalone script writes.

```bash
python plot_all.py --list                  # show the 8 embedded scripts
python plot_all.py skew_report skew_split  # run a subset
python plot_all.py --stop-on-error         # halt on first failure
```

## Repository layout

```
.
├── DenguYosoku.py                  # main 8-step pipeline (train → evaluate → report)
├── make_blind_holdout.py           # cut one week out WITH lags, WITHOUT the answer
├── plot_all.py                     # runner: all 8 scripts below, embedded verbatim
├── predict_2026-10-25.py           # predict the blind week
├── plot_DenguYosoku.py             # main figure deck (11 plots + 77 province panels)
├── plot_top5_worst_mae_actual.py   # 5 largest blind-week MAE, actual curves
├── plot_top5_best_mae_actual.py    # 5 smallest blind-week MAE, actual curves
├── blind_compare_hist.py           # actual vs predicted bars, top/least 5 gaps
├── raw_cases_hist_swapped.py       # raw-cases vertical histogram
├── skew_report.py                  # skewness diagnostics
├── skew_split.py                   # TRAIN vs TEST skewness
├── requirements.txt
├── README.md
├── dengue_cut_allweeks_2003-18_Oct_2026.csv   # INPUT, not committed (see Data)
├── dengue_full.csv                            # INPUT, not committed (see Data)
└── DenguYosoku/                    # ← THE ONLY OUTPUT FOLDER YOU KEEP
```

`plot_all.py` already embeds all eight plotting/prediction scripts, so the
standalone copies of those eight are optional. `DenguYosoku.py` and
`make_blind_holdout.py` are **not** embedded and must stay as files.

## Keeping only `DenguYosoku/`

`DenguYosoku/` is the single artefact folder. It holds the model and everything
needed to reproduce every number and re-draw the main figure deck. All other
output directories are scratch and can be deleted at any time.

| Path | Keep? | Regenerate with |
| --- | --- | --- |
| `DenguYosoku/` | **KEEP** | `python DenguYosoku.py` |
| `predict_week/` | delete | `make_blind_holdout.py` → `predict_2026-10-25.py` |
| `DenguYosoku_plots/` | delete | `python plot_all.py` |
| `blind_plots/` | delete | `python plot_all.py blind_compare_hist` |

What each kept file is for:

| File in `DenguYosoku/` | Consumed by |
| --- | --- |
| `model.txt` | `predict_2026-10-25.py`, `blind_compare_hist.py --recompute` |
| `metrics.json` | `plot_DenguYosoku.py` headline numbers; **authority for version pins** |
| `feature_importance.csv` | `plot_DenguYosoku.py` plot 10 |
| `holdout_predictions.csv` | `plot_DenguYosoku.py` plots 01–09 and 11 |
| `test_pred_vs_actual.png` | Step 7 figure (standalone, not read by anything) |

**One dependency to remember:** the blind-week scripts read `predict_week/`, not
`DenguYosoku/`. If you delete `predict_week/`, re-run `make_blind_holdout.py` and
`predict_2026-10-25.py` before `plot_all.py`. With only `DenguYosoku/` + the raw
CSV present, this subset still runs:

```bash
python plot_all.py plot_DenguYosoku raw_cases_hist_swapped skew_report skew_split
```

Suggested `.gitignore` — note it deliberately does **not** ignore `DenguYosoku/`:

```gitignore
.venv/
__pycache__/
predict_week/          # scratch — regenerate
DenguYosoku_plots/     # scratch — regenerate
blind_plots/           # scratch — regenerate
# DenguYosoku/ is KEPT: model.txt + metrics.json + predictions + importance
```

## Outputs

Kept — `DenguYosoku/`:

```
DenguYosoku/
 ├── model.txt                    # LightGBM booster (quantile α=0.5, 600 rounds)
 ├── holdout_predictions.csv      # province, week_start, year, cases, pred,
 │                                #   persist, seasonal_naive, climatology, residual
 ├── feature_importance.csv       # feature, gain, split, gain_pct
 ├── metrics.json                 # every number, config, and library versions
 ├── test_pred_vs_actual.png      # Step 7 scatter + residual panel
 ├── optuna_history.html / .png   # only when RUN_OPTUNA=True (.png needs kaleido)
 └── optuna_params.html / .png    # only when RUN_OPTUNA=True (.png needs kaleido)
```

Regenerable — not kept:

```
predict_week/
 ├── answer_key.csv               # actual cases of the blind week (PRIVATE)
 ├── features.json                # the 12 feature names, in model column order
 ├── predict_2026-10-25.csv       # blind week: 77 rows, 12 features, NO cases
 └── predictions.csv              # output of predict_2026-10-25.py

DenguYosoku_plots/
 ├── 01…11_*.png                  # national, grid pages, heatmaps, scatter, subgroups
 ├── province_individual/         # 77 per-province panels
 ├── subgroup_summary.csv
 ├── interactive_*.html           # only when plotly is installed
 ├── top5_mae_blind_<date>/       # worst-5 curves + bar + ranking CSV
 ├── best5_mae_blind_<date>/      # best-5 curves + bar + ranking CSV
 ├── raw_cases_hist_vertical.png
 └── skew_*.png, skew_table.csv, skew_split_summary.csv, skew_by_province_split.csv

blind_plots/
 ├── blind_top5_gap.png
 ├── blind_least5_gap.png
 └── blind_comparison_77.csv
```

## Data

One row = one province in one epidemiological week (Sunday start).

| Column | Role |
| --- | --- |
| `province`, `week_start`, `week`, `year` | identifiers / calendar |
| `cases` | **target** — weekly reported dengue count |
| `w_apparent_temperature_mean` | weather source (used **lagged only**) |
| `population`, `cases_per_100k` | **excluded** — see below |

Two input files, both required for a full run:

| File | Read by | Must contain |
| --- | --- | --- |
| `dengue_cut_allweeks_2003-18_Oct_2026.csv` | `DenguYosoku.py` (hardcoded, no fallback), `skew_report.py`, `skew_split.py`, `raw_cases_hist_swapped.py` | full grid through 2026-10-18 |
| `dengue_full.csv` | `make_blind_holdout.py` (hardcoded), both top-5 scripts | the same history **plus the blind week 2026-10-25** |

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

```
Quantile won at every round count.
Shipped booster: objective=quantile, alpha=0.5, 600 rounds.
```

The switch also redistributed what the model learned: `cases_L1` fell from
83 % → 49.8 % of gain while the multi-week slope (`cases_L2..L4`, now 33.7 %)
and the seasonal harmonic gained weight.

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

6. **The blind rehearsal reads a different file than training.**
   `make_blind_holdout.py` has `SRC = Path("dengue_full.csv")` while
   `DenguYosoku.py` trains on the `allweeks` cut. If the two disagree on the last
   12 weeks per province, the blind week's lags differ from training-time
   features. Verify the tail matches, or point `SRC` at the training file.

7. **The blind filename is dynamic on one side, hardcoded on the other.**
   `make_blind_holdout.py` writes `predict_{LAST.date()}.csv` from
   `dengue_full.csv`'s max week; `predict_2026-10-25.py` hardcodes
   `predict_2026-10-25.csv`. `dengue_full.csv` must therefore end exactly at
   **2026-10-25** — the week `DenguYosoku.py` drops via `EVAL_CUTOFF`.

8. **Excluded data:** `population` (broken), mosquito indices (sparse, with
   implausible Breteau values).

## Reproducibility

| Item | Value |
| --- | --- |
| Seed | 42 (NumPy, LightGBM) |
| Threads | 1 — LightGBM histogram binning is non-deterministic across threads |
| Split | `TRAIN_END = 2025-01-01`, `EVAL_CUTOFF = 2026-10-25` |
| CV | 5 expanding folds, 4-week purge gap, `MIN_TRAIN_FRAC = 0.45` |
| Search | `RUN_OPTUNA = False` as shipped |
| Shipped model | quantile α=0.5, 600 rounds, LightGBM default tree params |
| Metric | MAE vs persistence, fixed in Step 1 before any result |
| Versions | recorded in `DenguYosoku/metrics.json` → `versions` |

Re-running with the same config reproduces every number bit-for-bit.

`metrics.json` records python / numpy / pandas / lightgbm — but **not**
matplotlib or seaborn, which is what figures depend on. For byte-identical
PNGs, freeze the whole stack after a verified run:

```bash
pip freeze > requirements.lock.txt
```
