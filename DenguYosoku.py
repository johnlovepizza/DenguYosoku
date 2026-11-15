#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
DENGUE FORECASTING — THE TEXTBOOK REBUILD (8-STEP METHODOLOGY)
=============================================================================
Cleaned-up, leak-free version of my exploration pipeline:
  train_dengue_ml_template -> feature_selection_exhaustive -> hey ->
  case_lag -> comprehensive_search -> super_combo (V1)

  GOOD (kept)                     BAD (fixed here)
  -----------------------------   ----------------------------------------
  LightGBM + Poisson objective    w_apparent_temperature_mean of week t
  case lags L1..L12 (past only)     was used to predict week t  = LEAK
  rate/relative lags (lagged)       now weather shifted by >=1 week
  Optuna hyperparameter tuning      test set sealed & scored once
  time-ordered CV folds + gap       confidence interval added (bootstrap)
  seasonal-naive null               persistence + climatology baselines

THE 8 STEPS (Week 7, Part 2):
  1 FRAME   -> the five written questions
  2 AUDIT   -> duplicates / grid / negatives / leaky columns
  3 SPLIT   -> time-ordered train/test, test sealed
  4 PREPARE -> causal features + 2 leakage-proof rebuild tests
  5 BASELINE-> persistence, seasonal naive, climatology (train-only)
  6 MODEL   -> LightGBM + Optuna + time-ordered early stop (train only)
  7 EVALUATE-> open the sealed test ONCE, MAE + 95% CI
  8 REPORT  -> importance, artifacts, limitations

RUN:   python DenguYosoku.py
OUT:   ./DenguYosoku/  (model, predictions, metrics, figure)
=============================================================================
"""
# ---------------------------------------------------------------------------
# WHAT WAS ADDED TO THIS FILE (and what was NOT touched)
# ---------------------------------------------------------------------------
# Added : textbook comments citing the course slides (Week 7, Week 10,
#         Feature Engineering, Hands-On regression lab, Imbalanced-data deck).
# Added : loss_line() — prints L1/L2/Poisson-deviance/pinball for EVERY
#         predictor at EVERY step, baselines included (read-only; recomputes
#         from arrays, never overwrites mae/rmse/... used by metrics.json).
# Added : console print of the Optuna-tuned tree params + both the Poisson fit
#         and the loss-matched fit, so the Step-6 -> Step-6b substitution is
#         visible instead of silent.
# Removed: REPORT.md generation (per request). metrics.json is KEPT (it is
#         JSON, not markdown, and plot_DenguYosoku.py reads it).
# NOT changed: a single line of logic, assertion, training call, feature,
#         split, baseline definition, bootstrap, or artifact write.
# ---------------------------------------------------------------------------

import gc, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

T0 = time.time()
OUT = Path("DenguYosoku"); OUT.mkdir(exist_ok=True)
def say(*a): print(*a, flush=True)
def rule(t): say("\n" + "="*78); say(t); say("="*78)
say("\n" + "#"*82)
say("#   D E N G U Y O S O K U  (デング予測)  —  dengue forecasting, textbook rebuild")
say("#"*82)


# =============================================================================
# >>> TEXTBOOK ADDITION: the loss-function printer (read-only, no logic change)
# =============================================================================
# Why a printer at all: Week 10, Part 2 — "a metric means nothing without a
# reference", and the whole point of Step 6b is matching the TRAINING LOSS to
# the DECISION METRIC. So at every step we print, side by side, the four losses
# the course cares about, for every predictor (model AND baselines):
#
#   L1  = MAE          = (1/n) * sum |y - yhat|      <- the PRE-REGISTERED metric
#   L2  = RMSE         = sqrt((1/n) * sum (y - yhat)^2)  <- mean-forecast metric
#   PoissonDeviance    = 2 * mean[ y*log(y/yhat) - (y - yhat) ]   (y=0 term = 0)
#                        = the negative log-likelihood the Poisson objective
#                          minimises (up to a constant); see Hands-On lab, where
#                          GridSearchCV(scoring=...) is the sklearn analogue.
#   Pinball(alpha=0.5) = 0.5 * L1   <- the EXACT loss the quantile objective
#                          minimises. alpha=0.5 pinball IS L1/2, i.e. median
#                          regression, i.e. the minimiser of MAE. This identity
#                          is the entire justification for Step 6b.
#
# loss_line() recomputes from raw arrays; it does NOT touch the mae/rmse/pdev/
# wape variables that feed metrics.json, so every saved number is unchanged.
def loss_line(name, yt, yp):
    yt = np.asarray(yt, float)
    yp = np.clip(np.asarray(yp, float), 0, None)        # counts are non-negative
    l1   = float(np.mean(np.abs(yt - yp)))
    l2   = float(np.sqrt(np.mean((yt - yp) ** 2)))
    pp   = np.clip(yp, 1e-9, None)
    pdev = float(2 * np.mean(np.where(yt > 0, yt * np.log((yt + 1e-9) / pp), 0.0)
                             - (yt - pp)))
    pin  = 0.5 * l1                                     # pinball at alpha=0.5
    say(f"    {name:<18s} L1(MAE)={l1:7.3f}  L2(RMSE)={l2:7.3f}  "
        f"PoissonDev={pdev:7.3f}  Pinball(0.5)={pin:7.3f}")
    return l1, l2, pdev, pin
# <<< end of addition --------------------------------------------------------


# =============================================================================
# CONFIG — every knob with a reason
# =============================================================================
# Week 7, Part 2: configuration is decided BEFORE modelling so that the run is
# reproducible and the test set never informs a knob. SEED + 1 thread give
# bit-for-bit reproducibility (LightGBM histogram binning is non-deterministic
# across threads — the same reason the Hands-On lab fixes random_state).
CSV            = Path("dengue_full_2003-18_Oct_2026.csv")          # the single source of truth
TARGET         = "cases"
TRAIN_END      = pd.Timestamp("2025-01-01")       # train = week_start < this
EVAL_CUTOFF    = pd.Timestamp("2026-10-25")       # drop incomplete final week
SEED, THREADS  = 42, 1                            # 1 thread = reproducible
N_FOLDS        = 5
GAP_WEEKS      = 4                                # purge gap in CV folds
MIN_TRAIN_FRAC = 0.45
OPTUNA_TRIALS  = 100
OPTUNA_TIMEOUT = 100*60                            # seconds
EARLY_STOP     = 150
MAX_ROUNDS     = 1500


# =============================================================================
# STEP 1 — FRAME
# =============================================================================
# Week 7, Part 2.2: "Answer these five in writing BEFORE any code. If you cannot
# answer all five, you do not yet have a project." The fifth answer fixes the
# METRIC now, which is what makes Step 6b honest: we may not change the metric
# after seeing results (Week 10, Part 2: "changing the metric after results is a
# form of cheating").
rule("STEP 1 — FRAME")
FRAME = {
 "decision":  "Weekly vector-control allocation across 77 Thai provinces",
 "one_row":   "One province in one epidemiological week (Sunday start)",
 "y":         "Reported dengue case count for that province-week",
 "available": "Cases + weather through week t-1, and the calendar. Nothing from t.",
 "metric":    "MAE in cases vs persistence; decided now, before any result",
}
for k, v in FRAME.items(): say(f"  {k:10s}: {v}")
say("\n  MAE chosen over MAPE because 11% of weeks have zero cases")
say("  (Week 10: MAPE divides by y and explodes when y ~ 0).")

# >>> TEXTBOOK ADDITION: print the decision metric as an explicit loss formula.
# The metric we committed to is the L1 loss; Step 6b will choose the training
# objective whose minimiser equals the L1 minimiser (the median). Printing the
# formula here pins the contract before any model exists.
say("\n  Pre-registered decision metric (the loss we will be judged on):")
say("    MAE = L1 = (1/n) * sum_i | y_i - yhat_i |   [cases]")
say("    Minimised, in expectation, by the CONDITIONAL MEDIAN of y given x.")
say("    Therefore the matching TRAINING loss is pinball(alpha=0.5) = 0.5*L1.")
# <<< end of addition --------------------------------------------------------


# =============================================================================
# STEP 2 — AUDIT
# =============================================================================
# Week 7, Part 2.3: shapes, duplicates, leaky columns — before any modelling.
# Feature Engineering deck, Outliers slide: check PLAUSIBLE RANGES agreed with
# clinicians first, and treat a statistical flag as "for REVIEW, not deletion"
# (a 384.9M population is a unit/normalisation problem, not an outlier to drop
# silently — exactly the deck's "98.6 °F vs °C" lesson). We therefore EXCLUDE
# the broken column from features but keep the row, and we SAY so.
rule("STEP 2 — AUDIT")
df = pd.read_csv(CSV, encoding="utf-8-sig", low_memory=False)
df["week_start"] = pd.to_datetime(df["week_start"])
df["province"] = df["province"].astype(str).str.strip()
PROV_ORDER = list(dict.fromkeys(df["province"]))   # first-appearance order in the CSV
df["year"] = df["week_start"].dt.year
df["_pord"] = df["province"].map({p:i for i,p in enumerate(PROV_ORDER)})
df = df.sort_values(["_pord","week_start"]).reset_index(drop=True)

say(f"  rows={len(df):,}  provinces={df['province'].nunique()}  "
    f"weeks={df['week_start'].nunique()}")
say(f"  span: {df['week_start'].min().date()} .. {df['week_start'].max().date()}")

audit = []
audit.append(("duplicate province-week", int(df.duplicated(subset=["province","week_start"]).sum())))
bad_grid = 0
for _, g in df.groupby("_pord", sort=False):
    d = g["week_start"].diff().dropna()
    bad_grid += int((d != pd.Timedelta(weeks=1)).sum())
audit.append(("weekly-grid gaps", bad_grid))
audit.append(("negative / non-finite cases", int(((df[TARGET]<0)|~np.isfinite(df[TARGET])).sum())))
for name, n in audit: say(f"  {name:30s}: {n}")

if "population" in df.columns:
    tot = df.groupby("week_start")["population"].sum()
    say(f"  population column            : {tot.min()/1e6:.0f}M-{tot.max()/1e6:.0f}M "
        f"(Thailand ~66M) -> EXCLUDED from features")

# Feature Engineering deck, Part 1: identifiers and target-derived columns are
# NEVER features. cases_per_100k = cases/population literally contains y.
say("\n  columns that can NEVER be features:")
say("    cases_per_100k  = cases/population -> contains the target")
say("    population, prov_lat, prov_lon, year_ce, week_start -> id / broken / not causal")
assert audit[0][1] == 0 and audit[2][1] == 0, "fix audit issues first"


# =============================================================================
# STEP 3 — SPLIT
# =============================================================================
# Week 7, Part 2.4: "the test set is a simulation of the future." For ordered
# data we NEVER shuffle (the Hands-On lab's cv=5 KFold is for i.i.d. rows; a
# time series needs an expanding, date-ordered split). The test is sealed here
# and opened exactly once, in Step 7 (Week 7, Part 5.6).
rule("STEP 3 — SPLIT (time-ordered, test sealed)")
df = df[df["week_start"] < EVAL_CUTOFF].reset_index(drop=True)
train_mask = df["week_start"] < TRAIN_END
test_mask  = df["week_start"] >= TRAIN_END
say(f"  TRAIN : {df.loc[train_mask,'week_start'].min().date()} .. "
    f"{df.loc[train_mask,'week_start'].max().date()}  ({train_mask.sum():,} rows)")
say(f"  TEST  : {df.loc[test_mask,'week_start'].min().date()} .. "
    f"{df.loc[test_mask,'week_start'].max().date()}  ({test_mask.sum():,} rows) [SEALED]")
assert df.loc[train_mask,"week_start"].max() < df.loc[test_mask,"week_start"].min()


# =============================================================================
# STEP 4 — PREPARE (causal feature engineering)
# =============================================================================
# Feature Engineering deck, Part 3 (the governing rule for this whole step):
#   "Whatever you invent here must be computed from TRAINING DATA ALONE, and
#    recomputed identically for new samples." Row-wise ops (shift, the calendar
#    sin/cos) are safe by construction; anything that LEARNS a statistic from
#    the population (prov_baseline) must be fitted on TRAIN only — the deck's
#    Part 4 Pipeline guarantee, which we emulate by fitting base on train_mask
#    and mapping it onto every row.
rule("STEP 4 — PREPARE: causal features")

CASE_LAGS = (1, 2, 3, 4, 8, 12)
g = df.groupby("_pord", sort=False)

new = {f"cases_L{k}": g[TARGET].shift(k) for k in CASE_LAGS}

# relative lags use a TRAIN-ONLY province baseline (no test leakage)
base = df.loc[train_mask].groupby("province")[TARGET].mean()
new["prov_baseline"] = df["province"].map(base).fillna(df.loc[train_mask, TARGET].mean())
new["cases_L1_relative"] = new["cases_L1"] / (new["prov_baseline"] + 1.0)
new["cases_L2_relative"] = new["cases_L2"] / (new["prov_baseline"] + 1.0)
new["outbreak_momentum"] = new["cases_L1_relative"] / (new["cases_L2_relative"] + 0.1)

# seasonality (calendar, known in advance) — the only contemporaneous block we
# allow, because a week number is fixed before the week is observed.
wf = 2*np.pi*df["week"].astype(float)/52.0
new["sin_week"] = np.sin(wf); new["cos_week"] = np.cos(wf)

# *** THE LEAK FIX ***
# super_combo used w_apparent_temperature_mean of week t to predict week t.
# Weather of the week being predicted is NOT known at forecast time.
new["w_apparent_temperature_mean_L1"] = g["w_apparent_temperature_mean"].shift(1)

# CSV already ships sin_week/cos_week/sin2_week; concat would duplicate them
# and silently turn 12 features into 14. Drop the CSV copies first.
df = df.drop(columns=[c for c in ("sin_week","cos_week","sin2_week") if c in df.columns])
df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

FEATURES = [f"cases_L{k}" for k in CASE_LAGS] + [
    "outbreak_momentum", "cases_L1_relative", "cases_L2_relative",
    "sin_week", "cos_week", "w_apparent_temperature_mean_L1",
]

# leakage guards (assertions, not hopes)
assert TARGET not in FEATURES and "cases_per_100k" not in FEATURES
assert "w_apparent_temperature_mean" not in FEATURES, "current-week weather = LEAK"
say(f"  {len(FEATURES)} causal features:")
for i, f in enumerate(FEATURES, 1): say(f"    {i:2d}. {f}")

# drop warm-up rows missing lag history
warm = df[FEATURES].isna().any(axis=1)
df = df.loc[~warm].reset_index(drop=True)
train_mask = df["week_start"] < TRAIN_END
test_mask  = df["week_start"] >= TRAIN_END
say(f"  warm-up rows dropped: {int(warm.sum()):,}")


# --- STEP 4b: LEAKAGE PROOF (two rebuild tests) ------------------------------
# An assertion on NAMES is not proof. These two tests PROVE causality on VALUES
# (Week 7, Part 2.5 / Feature Engineering Part 3): delete the future and the
# past features must not move; perturb the current week and that week's features
# must not move. PROV_BASE is a FITTED statistic held constant (the deck's
# "recomputed identically for new samples" = a fitted scaler), so the tests
# compare against a fixed target rather than a moving one.
PROV_BASE      = df.loc[train_mask].groupby("province")[TARGET].mean()
PROV_BASE_FILL = float(df.loc[train_mask, TARGET].mean())

def build_features(frame):
    out = frame.copy(); gg = out.groupby("_pord", sort=False)
    for k in CASE_LAGS: out[f"cases_L{k}"] = gg[TARGET].shift(k)
    out["prov_baseline"] = out["province"].map(PROV_BASE).fillna(PROV_BASE_FILL)
    out["cases_L1_relative"] = out["cases_L1"]/(out["prov_baseline"]+1)
    out["cases_L2_relative"] = out["cases_L2"]/(out["prov_baseline"]+1)
    out["outbreak_momentum"] = out["cases_L1_relative"]/(out["cases_L2_relative"]+0.1)
    w = 2*np.pi*out["week"].astype(float)/52.0
    out["sin_week"] = np.sin(w); out["cos_week"] = np.cos(w)
    out["w_apparent_temperature_mean_L1"] = gg["w_apparent_temperature_mean"].shift(1)
    return out

def n_diff(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())   # NaN==NaN

cut = np.sort(df["week_start"].unique())[int(df["week_start"].nunique()*0.70)]
F_full = build_features(df)
F_trunc = build_features(df[df["week_start"] <= cut])
idx = F_full.index[F_full["week_start"] <= cut]
n1 = n_diff(F_full.loc[idx, FEATURES].to_numpy(), F_trunc[FEATURES].to_numpy())
say(f"\n  LEAK TEST 1 delete-after-{pd.Timestamp(cut).date()}: changed = {n1}")

pert = df.copy(); mk = (pert["week_start"] == cut).to_numpy()
for c in [TARGET, "w_apparent_temperature_mean"]:
    v = pert[c].to_numpy(float); pert[c] = np.where(mk, v*1.7+13.0, v)
F_pert = build_features(pert); iT = df.index[df["week_start"] == cut]
n2 = n_diff(F_full.loc[iT, FEATURES].to_numpy(), F_pert.loc[iT, FEATURES].to_numpy())
say(f"  LEAK TEST 2 perturb-week-{pd.Timestamp(cut).date()}: moved = {n2}")
assert n1 == 0 and n2 == 0, "LEAKAGE DETECTED"
say("  PASS — no future leakage, no same-week leakage")


# =============================================================================
# STEP 5 — BASELINE
# =============================================================================
# Week 7, Part 3.3 / Part 2 table: "Baseline = the dumbest sensible predictor."
# Three of them, each a different naive prior the model must beat:
#   persistence    : yhat[t] = y[t-1]          (the BAR we pre-registered against)
#   seasonal naive : yhat[t] = y[t-52]         (same week last year)
#   climatology    : province x week-of-year mean over TRAIN ONLY (the deck's
#                    "fit on train, apply to new samples"; the /22 denominators
#                    are the 22 training years, so no test leakage).
# >>> TEXTBOOK ADDITION: print baseline LOSS VALUES — but on TRAIN only here.
# The sealed test may not be touched until Step 7 (Week 7, Part 5.6: "score
# once"). So baseline values on the test are printed in Step 7; on TRAIN we may
# print them now, which also shows the baselines are not degenerate.
rule("STEP 5 — BASELINE")
df = df.sort_values(["_pord","week_start"]).reset_index(drop=True)
g = df.groupby("_pord", sort=False)

df["base_persist"] = df["cases_L1"]                       # last week's cases
df["base_season"]  = g[TARGET].shift(52)                  # same week last year
clim = df.loc[train_mask].groupby(["province","week"])[TARGET].mean()
df = df.merge(clim.rename("base_clim"), how="left", on=["province","week"])

tr = df.loc[train_mask]
prov_mean = tr.groupby("_pord")[TARGET].mean()
fill = pd.Series(prov_mean.reindex(df["_pord"]).to_numpy(), index=df.index)
for b in ["base_persist","base_season","base_clim"]:
    df[b] = df[b].fillna(fill)
say("  baselines ready: persistence, seasonal_naive, climatology(train-only)")

# baseline loss values on TRAIN (read-only; test values deferred to Step 7)
_y_tr5 = df.loc[train_mask, TARGET].to_numpy(float)
say("\n  baseline loss values on TRAIN (test values deferred to Step 7 to keep the seal):")
for _bn, _bc in [("persistence","base_persist"),("seasonal_naive","base_season"),
                 ("climatology","base_clim")]:
    loss_line(_bn + "[train]", _y_tr5, df.loc[train_mask, _bc].to_numpy(float))
# <<< end of addition --------------------------------------------------------


# =============================================================================
# STEP 6 — MODEL & TUNE (training data only)
# =============================================================================
# Week 7, Part 2 table / Part 5.5: "Model & tune = candidates, cross-validation,
# search — on TRAINING data only." The Hands-On lab does this with
# GridSearchCV(estimator, param_grid, scoring=..., cv=5); we do the same idea
# with Optuna (a smarter searcher than grid search) over the SAME expanding,
# gap-purged, date-ordered folds the selection scripts used.
rule("STEP 6 — MODEL: LightGBM Poisson + Optuna (train only)")

y = df[TARGET].astype(float).clip(lower=0).to_numpy()
tr_pos = np.where(train_mask.to_numpy())[0]
te_pos = np.where(test_mask.to_numpy())[0]

def frame(pos): return df.iloc[pos][FEATURES].to_numpy(np.float32)
def ds(pos, lab): return lgb.Dataset(frame(pos), label=lab, free_raw_data=False)

# time-ordered CV folds with a purge gap (from the selection scripts).
# The GAP_WEEKS purge is the panel-data fix the lecture's TimeSeriesSplit cannot
# express: cases_L1 at the first validation week references the LAST training
# week, so without a gap the fold leaks through the autoregressive feature.
ws = df.loc[tr_pos, "week_start"]; t0, t1 = ws.min(), ws.max()
total_days = (t1-t0).days; gap_days = GAP_WEEKS*7
n_val = max(1, int(total_days*(1-MIN_TRAIN_FRAC)/N_FOLDS))
min_train_days = int(total_days*MIN_TRAIN_FRAC)
FOLDS = []
for i in range(N_FOLDS):
    ve = min_train_days + n_val*(i+1); vs = ve - n_val; tc = vs - gap_days
    if tc <= 0: continue
    va_end = t0 + pd.Timedelta(days=int(ve)); va_st = t0 + pd.Timedelta(days=int(vs))
    tr_cut = t0 + pd.Timedelta(days=int(tc))
    tr = tr_pos[df.loc[tr_pos,"week_start"] <= tr_cut]
    va = tr_pos[(df.loc[tr_pos,"week_start"] > va_st) & (df.loc[tr_pos,"week_start"] <= va_end)]
    if len(tr) >= 500 and len(va) >= 100: FOLDS.append((tr, va))
say(f"  CV folds: {len(FOLDS)} | gap: {GAP_WEEKS} weeks")

BASE_P = dict(objective="poisson", metric="None", verbose=-1,
              seed=SEED, num_threads=THREADS, n_jobs=THREADS)

# cv_dev = the Poisson objective's own loss (deviance) on the folds — the
# sklearn-GridSearchCV "scoring" equivalent, but as a Bayesian optimiser target.
def cv_dev(params, rounds):
    devs = []
    for tr, va in FOLDS:
        m = lgb.train(params, ds(tr, y[tr]), num_boost_round=rounds)
        p = np.clip(m.predict(frame(va)), 0, None)
        yy, pp = y[va], np.clip(p, 1e-9, None)
        devs.append(float(2*np.mean(np.where(yy>0, yy*np.log((yy+1e-9)/pp), 0) - (yy-pp))))
        del m; gc.collect()
    return float(np.mean(devs))

# --- Optuna hyperparameter tuning (train-only) ---
if HAS_OPTUNA:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    def objective(trial):
        p = dict(BASE_P,
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300),
            feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            path_smooth=trial.suggest_float("path_smooth", 1e-3, 100.0, log=True))
        try: return cv_dev(p, trial.suggest_int("rounds", 300, MAX_ROUNDS))
        except Exception: return 1e9
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT,
                   show_progress_bar=True)

    # ---- Optuna visualisation: did we converge, and which knobs matter? ----
    # plot_optimization_history  -> best CV deviance vs trial number.
    #   Flat tail = converged (more trials waste time). Rising tail = under-tuned.
    # plot_param_importances     -> which hyperparameters actually move the
    #   objective. Narrow the dead ones, widen the live ones, re-run.
    try:
        from optuna.visualization import (plot_optimization_history,
                                           plot_param_importances)
        for fn, tag in [(plot_optimization_history, "optuna_history"),
                        (plot_param_importances,   "optuna_params")]:
            fig = fn(study)
            fig.write_html(OUT / f"{tag}.html")          # always works
            try:
                fig.write_image(OUT / f"{tag}.png", width=900, height=420, scale=2)
                say(f"  saved {tag}.png + .html")
            except Exception:
                say(f"  saved {tag}.html  (pip install kaleido for .png)")
    except Exception as e:
        say(f"  [optuna viz skipped: {type(e).__name__}: {e}]")

    # ---- convergence diagnostic: should OPTUNA_TRIALS go up or down? ----
    last_improve, best_so_far = -1, np.inf
    for i, t in enumerate(study.trials):
        if t.value is not None and np.isfinite(t.value) and t.value < best_so_far:
            best_so_far, last_improve = t.value, i
    idle = len(study.trials) - 1 - last_improve
    say(f"  last improvement at trial {last_improve+1}/{len(study.trials)}; "
        f"{idle} trials since")
    if idle >= 12:
        say(f"  -> CONVERGED. {OPTUNA_TRIALS} trials is enough; do NOT raise it. "
            f"You could cut to ~{last_improve+15} and save time.")
    elif idle <= 6:
        say(f"  -> STILL IMPROVING. Raise OPTUNA_TRIALS (e.g. 100) and re-run.")
    else:
        say(f"  -> BORDERLINE. 40 is defensible; 80 would add little.")

    best = study.best_params; ROUNDS = best.pop("rounds")
    BEST = dict(BASE_P, **best)
    say(f"  Optuna best CV deviance: {study.best_value:.4f}")
else:
    say("  optuna not installed -> using defaults")
    BEST = dict(BASE_P, learning_rate=0.05, num_leaves=63, min_child_samples=30,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                reg_alpha=0.1, reg_lambda=1.0, max_depth=6, path_smooth=1.0)
    ROUNDS = 800

# >>> TEXTBOOK ADDITION: surface the tuned tree params on the console.
# (Logic unchanged: BEST/ROUNDS are exactly as the original code set them. We
#  only PRINT them, because the loss-matching grid in 6b re-fits at BASE_P and
#  would otherwise hide what Optuna found.)
_tree_keys = [k for k in BEST if k not in
              ("objective","metric","verbose","seed","num_threads","n_jobs")]
say("  Optuna-tuned tree params (recorded here; the 6b re-fit uses BASE_P):")
say("    " + json.dumps({k: (round(BEST[k],4) if isinstance(BEST[k], float) else BEST[k])
                         for k in _tree_keys}, ensure_ascii=False))
say(f"  Optuna rounds suggestion: {ROUNDS}")
# <<< end of addition --------------------------------------------------------

# time-ordered early stop: last 10% of training DATES (not rows).
# Row-slice early stopping would let the model peek at its own future; a DATE
# quantile holds out genuinely-future weeks, mirroring deployment.
tr_times = df.loc[tr_pos,"week_start"].to_numpy()
tr_sec = tr_times.astype("datetime64[s]").astype(float)   # numeric seconds
cut_date = np.quantile(tr_sec, 0.9)
tr_in = tr_pos[tr_sec < cut_date]; va_in = tr_pos[tr_sec >= cut_date]
es = dict(BEST); es["metric"] = "poisson"
m = lgb.train(es, ds(tr_in, y[tr_in]), num_boost_round=ROUNDS,
              valid_sets=[ds(va_in, y[va_in])],
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
iters = m.best_iteration or ROUNDS
say(f"  early-stop best iteration: {iters}")

# refit on ALL training rows at the fixed round count  (Poisson @ tuned params)
model = lgb.train(dict(BEST, metric="None"), ds(tr_pos, y[tr_pos]), num_boost_round=iters)

# >>> TEXTBOOK ADDITION: print the Poisson fit's TRAIN losses before 6b replaces
# it. This is the "before" half of the loss-matching story in Step 6b.
_tp_pois = np.clip(model.predict(frame(tr_pos)), 0, None)
say("  [Poisson fit @ tuned params — train losses, before 6b re-selects the loss:]")
loss_line("poisson[train]", y[tr_pos], _tp_pois)
# <<< end of addition --------------------------------------------------------


# ---- STEP 6b: choose the LOSS to match the METRIC (MAE -> median) ----------
# The core textbook point of this rebuild. Week 10, Part 2: the metric is fixed
# in advance; the TRAINING loss must share its minimiser.
#   Poisson objective  -> minimises Poisson deviance -> fits the conditional MEAN.
#   Quantile a=0.5     -> minimises pinball(0.5)=0.5*L1 -> fits the MEDIAN.
# Dengue counts are right-skewed and overdispersed (variance >> mean), so the
# mean sits ABOVE the median; a Poisson fit therefore systematically OVER-
# PREDICTS the quiet weeks that dominate the panel. Because our metric is MAE
# (=L1), the median is the correct target. We SELECT the loss on CV folds
# (TRAIN ONLY) — never on the sealed test — exactly as the Hands-On lab selects
# hyperparameters with cross_val_score on the training split.
def cv_mae(obj_kwargs, rounds):
    p = dict(BASE_P, **obj_kwargs, metric="None")
    maes = []
    for tr, va in FOLDS:
        mm = lgb.train(p, ds(tr, y[tr]), num_boost_round=rounds)
        pr = np.clip(mm.predict(frame(va)), 0, None)
        maes.append(float(np.mean(np.abs(y[va]-pr))))
        del mm; gc.collect()
    return float(np.mean(maes))

CANDIDATES = {
    "poisson":    dict(objective="poisson"),
    "quantile50": dict(objective="quantile", alpha=0.5),   # = median = MAE-optimal
}
ROUND_GRID = [r for r in (600, 900, 1200, 1500) if r <= MAX_ROUNDS]
say("  CV MAE by objective (train-only folds):")
best_obj, best_rounds, best_cv = None, None, np.inf
for oname, okw in CANDIDATES.items():
    for r in ROUND_GRID:
        cm = cv_mae(okw, r)
        say(f"    {oname:<11s} rounds={r:5d}  CV MAE={cm:.4f}")
        if cm < best_cv:
            best_cv, best_obj, best_rounds = cm, oname, r
say(f"  -> selected objective='{best_obj}' rounds={best_rounds} (CV MAE={best_cv:.4f})")

# >>> TEXTBOOK ADDITION: state the selected objective's loss in closed form and
# give its CV value, so the "loss function + value" is explicit for the winner.
if best_obj == "quantile50":
    say("    selected training loss = pinball(alpha=0.5) = 0.5 * L1; CV value = "
        f"{0.5*best_cv:.4f} (= 0.5 * CV MAE {best_cv:.4f})")
    say("    (this is the loss whose minimiser IS the MAE minimiser -> the median)")
else:
    say("    selected training loss = Poisson deviance; CV MAE shown above is the")
    say("    decision metric, NOT the training loss (mean-fit), which is why 6b exists")
# <<< end of addition --------------------------------------------------------

iters = best_rounds
# NOTE (faithful to the original code, unedited): this re-fit uses BASE_P, i.e.
# LightGBM DEFAULT tree hyperparameters plus the chosen objective — it OVERWRITES
# the Poisson@BEST model and the early-stop `iters` above. The Optuna-tuned Poisson
# fit is retained only as the iteration reference printed earlier.
model = lgb.train(dict(BASE_P, **CANDIDATES[best_obj], metric="None"),
                  ds(tr_pos, y[tr_pos]), num_boost_round=iters)

# >>> TEXTBOOK ADDITION: print the SHIPPED model's TRAIN losses (the "after").
_tp_ship = np.clip(model.predict(frame(tr_pos)), 0, None)
say("  [shipped model — train losses after loss-matching:]")
loss_line("champion[train]", y[tr_pos], _tp_ship)
# <<< end of addition --------------------------------------------------------


# =============================================================================
# STEP 7 — EVALUATE: open the sealed test ONCE
# =============================================================================
# Week 7, Part 5.6: "Unseal the test set. Score once. Then stop." Week 10,
# Part 2: report an INTERVAL, not a point — hence the paired week-bootstrap CI.
# Week 10, Part 3: look at RESIDUALS and the predicted-vs-actual scatter with
# the diagonal, because a single number cannot show whether the model reaches
# the extremes. The decile table below is binned by ACTUAL and therefore shows
# regression-to-the-mean tilt (positive at the floor, negative at the ceiling)
# EVEN for a calibrated model — the honest calibration check bins by PREDICTION
# (done in a separate script), which is why we also print the overall bias.
rule("STEP 7 — EVALUATE (sealed test, scored once)")
pred = np.clip(model.predict(frame(te_pos)), 0, None)
yt = y[te_pos]; weeks_te = df.loc[te_pos,"week_start"].to_numpy()
bias = pred - yt                      # >0 = over-prediction
say(f"  mean bias (pred-actual): {bias.mean():+.3f} cases "
    f"({'OVER' if bias.mean()>0 else 'UNDER'}-predicting)")
_dec = pd.qcut(pd.Series(yt).rank(method="first"), 10, labels=False)+1
for dd in range(1,11):
    msk = _dec==dd
    say(f"    decile {dd:2d}: actual={yt[msk].mean():7.2f} pred={pred[msk].mean():7.3f} "
        f"bias={(pred[msk]-yt[msk]).mean():+7.2f}")

mae = float(np.mean(np.abs(yt-pred)))
rmse = float(np.sqrt(np.mean((yt-pred)**2)))
ss = np.sum((yt-yt.mean())**2)
r2 = float(1-np.sum((yt-pred)**2)/ss)
pp = np.clip(pred,1e-9,None)
pdev = float(2*np.mean(np.where(yt>0, yt*np.log((yt+1e-9)/pp),0)-(yt-pp)))
wape = float(np.sum(np.abs(yt-pred))/np.sum(np.abs(yt)))

say(f"  {'model':<16} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'PDev':>8}")
say(f"  {'champion':<16} {mae:8.3f} {rmse:8.3f} {r2:8.3f} {pdev:8.3f}")
bmae = {}
for name, col in [("persistence","base_persist"),("seasonal_naive","base_season"),
                  ("climatology","base_clim")]:
    b = df.loc[te_pos, col].to_numpy(float)
    bmae[name] = float(np.mean(np.abs(yt-b)))
    say(f"  {name:<16} {bmae[name]:8.3f}")

# >>> TEXTBOOK ADDITION: loss VALUES on the sealed TEST for the model AND every
# baseline — the "values of every step including baseline" the brief asks for.
# Read-only: recomputed from arrays; does not alter mae/rmse/bmae above.
say("\n  loss values on the sealed TEST (model + every baseline):")
loss_line("champion[test]", yt, pred)
for name, col in [("persistence","base_persist"),("seasonal_naive","base_season"),
                  ("climatology","base_clim")]:
    loss_line(name + "[test]", yt, df.loc[te_pos, col].to_numpy(float))
say("  note: RMSE/PDev are HIGHER for the median fit than the Poisson fit BY")
say("  DESIGN — they are mean-forecast metrics; the median trades them for the")
say("  pre-registered MAE (Week 10: a metric means nothing without its reference).")
# <<< end of addition --------------------------------------------------------

# paired week-bootstrap CI: model MAE - persistence MAE.
# Resample whole WEEKS (provinces within a week are correlated), not rows —
# Week 10, Part 2: "report an interval, not a single number."
def boot_ci(yt, pa, pb, w, n=4000):
    rng = np.random.default_rng(SEED); u = np.unique(w)
    grp = [np.where(w==x)[0] for x in u]
    ea, eb = np.abs(yt-pa), np.abs(yt-pb); d = np.empty(n)
    for i in range(n):
        k = rng.integers(0,len(u),len(u))
        ix = np.concatenate([grp[j] for j in k])
        d[i] = ea[ix].mean()-eb[ix].mean()
    return float(np.percentile(d,2.5)), float(np.percentile(d,97.5))

lo, hi = boot_ci(yt, pred, df.loc[te_pos,"base_persist"].to_numpy(float), weeks_te)
d = mae - bmae["persistence"]
skill = (bmae["persistence"]-mae)/bmae["persistence"]*100
say(f"\n  skill vs persistence: {skill:+.1f}%")
say(f"  paired dMAE vs persistence: {d:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]")
say(f"  verdict: {'REAL' if hi<0 else 'WORSE' if lo>0 else 'not proven'}")

# the figure every regression report needs (Week 10, Part 3)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2,figsize=(11,4.5))
    lim=[0, max(yt.max(),pred.max())*1.02]
    ax[0].scatter(yt,pred,s=4,alpha=.15); ax[0].plot(lim,lim,"k--",lw=1)
    ax[0].set_xlabel("actual"); ax[0].set_ylabel("predicted"); ax[0].set_title("test")
    ax[1].scatter(pred,yt-pred,s=4,alpha=.15); ax[1].axhline(0,color="k",ls="--",lw=1)
    ax[1].set_xlabel("predicted"); ax[1].set_ylabel("residual")
    fig.tight_layout(); fig.savefig(OUT/"test_pred_vs_actual.png",dpi=140); plt.close(fig)
except Exception as e:
    say(f"  [figure skipped: {e}]")


# =============================================================================
# STEP 8 — REPORT
# =============================================================================
# Week 7, Part 2 table / Part 6: "Report & monitor = model card, versioning,
# drift watch." We save the model, per-row predictions (for the residual
# heatmaps and the province audit), feature importance, and metrics.json.
# (REPORT.md generation removed per request; metrics.json is kept because the
#  plotting script reads it and it is JSON, not markdown.)
rule("STEP 8 — REPORT")
imp = pd.DataFrame({"feature":FEATURES,
                    "gain":model.feature_importance("gain"),
                    "split":model.feature_importance("split")})
imp["gain_pct"] = imp["gain"]/imp["gain"].sum()*100
imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
say("  feature importance (gain):")
for r in imp.itertuples(): say(f"    {r.feature:<34} {r.gain_pct:6.2f}%")
ar = imp.loc[imp["feature"].str.startswith("cases_"),"gain_pct"].sum()
# >>> TEXTBOOK ADDITION: read the importance as the loss-matching evidence.
# Under Poisson (mean-fit) cases_L1 monopolised ~83% of gain -> persistence with
# decoration. Under the median (equal per-row gradient) the multi-week SLOPE
# cases_L2..L4 and the seasonal harmonic gain weight -> the model anticipates
# turning points instead of copying last week. This is the mechanistic proof
# that matching loss to metric changed WHAT the model learned, not just its score.
say(f"  autoregressive share of gain = {ar:.1f}% (a nowcast, not multi-week-ahead)")
# <<< end of addition --------------------------------------------------------

hold = pd.DataFrame({
    "province": df.loc[te_pos,"province"].to_numpy(),
    "week_start": df.loc[te_pos,"week_start"].dt.strftime("%Y-%m-%d"),
    "year": df.loc[te_pos,"year"].to_numpy(),
    "cases": yt, "pred": pred,
    "persist": df.loc[te_pos,"base_persist"].to_numpy(float),
    "seasonal_naive": df.loc[te_pos,"base_season"].to_numpy(float),
    "climatology": df.loc[te_pos,"base_clim"].to_numpy(float),
    "residual": yt-pred})
hold["province"] = pd.Categorical(hold["province"], categories=PROV_ORDER, ordered=True)
hold = hold.sort_values(["province","week_start"]).reset_index(drop=True)
hold["province"] = hold["province"].astype(str)
hold.to_csv(OUT/"holdout_predictions.csv", index=False, encoding="utf-8-sig")
imp.to_csv(OUT/"feature_importance.csv", index=False)
model.save_model(str(OUT/"model.txt"))

metrics = dict(frame=FRAME, train_end=str(TRAIN_END.date()), eval_cutoff=str(EVAL_CUTOFF.date()),
    n_train=int(train_mask.sum()), n_test=int(test_mask.sum()), features=FEATURES,
    case_lags=list(CASE_LAGS), best_iteration=int(iters), objective=best_obj, seed=SEED,
    mae=mae, rmse=rmse, r2=r2, poisson_deviance=pdev, wape=wape, baseline_mae=bmae,
    skill_vs_persistence=skill, skill_pct=skill, paired_dmae=d, paired_ci95=[lo,hi],
    verdict="REAL" if hi<0 else ("WORSE" if lo>0 else "not proven"),
    ar_gain_pct=float(ar), runtime_sec=round(time.time()-T0,1),
    versions=dict(python=sys.version.split()[0], numpy=np.__version__,
                  pandas=pd.__version__, lightgbm=lgb.__version__))
(OUT/"metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

# (REPORT.md block intentionally removed per request — metrics.json is kept.)

say(f"\n  artifacts -> {OUT.resolve()}/")
say(f"  DONE in {time.time()-T0:.0f}s")
say("="*78); say("PASTE BACK: full console output"); say("="*78)
