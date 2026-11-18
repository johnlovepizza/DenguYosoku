#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skew_report.py — skewness diagnostics.
Feature Engineering deck, Part 3 "SKEW": |skew| > 1 is strongly skewed;
log1p handles zeros (counts). Deck caveat: "Trees do not need this" — which is
why DenguYosoku models RAW counts with a quantile objective, not log1p(y).

The raw-cases panel also shows mean vs median. That gap IS the reason the
Poisson (mean) fit over-predicted quiet weeks and the median fit fixed it.

RUN : python skew_report.py [path/to/dengue.csv]
OUT : DenguYosoku_plots/skew_*.png + skew_table.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
import matplotlib.pyplot as plt

CANDIDATES = ["dengue_cut_allweeks_2003-18_Oct_2026.csv"]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else next(
    (Path(c) for c in CANDIDATES if Path(c).exists()), None)
if SRC is None: sys.exit("no CSV found — pass the path as an argument")
OUT = Path("DenguYosoku_plots"); OUT.mkdir(exist_ok=True)
TARGET, CASE_LAGS = "cases", (1, 2, 3, 4, 8, 12)
WSRC = "w_apparent_temperature_mean"
TRAIN_END = pd.Timestamp("2025-01-01")

df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
df["week_start"] = pd.to_datetime(df["week_start"])
df["province"] = df["province"].astype(str).str.strip()
order = list(dict.fromkeys(df["province"]))
df["_pord"] = df["province"].map({p: i for i, p in enumerate(order)})
df = df.sort_values(["_pord", "week_start"]).reset_index(drop=True)
print(f"{SRC.name}: {len(df):,} rows, {df.province.nunique()} provinces")

# rebuild the 12 model features so the feature panel always works
g = df.groupby("_pord", sort=False)
for k in CASE_LAGS: df[f"cases_L{k}"] = g[TARGET].shift(k)
df[f"{WSRC}_L1"] = g[WSRC].shift(1)
tr = df["week_start"] < TRAIN_END
base = df.loc[tr].groupby("province")[TARGET].mean()
df["prov_baseline"] = df["province"].map(base).fillna(df.loc[tr, TARGET].mean())
df["cases_L1_relative"] = df["cases_L1"] / (df["prov_baseline"] + 1.0)
df["cases_L2_relative"] = df["cases_L2"] / (df["prov_baseline"] + 1.0)
df["outbreak_momentum"] = df["cases_L1_relative"] / (df["cases_L2_relative"] + 0.1)
wf = 2*np.pi*df["week"].astype(float)/52.0
df["sin_week"], df["cos_week"] = np.sin(wf), np.cos(wf)
FEATURES = [f"cases_L{k}" for k in CASE_LAGS] + [
    "outbreak_momentum","cases_L1_relative","cases_L2_relative",
    "sin_week","cos_week", f"{WSRC}_L1"]

def skew(s): return float(pd.Series(s).dropna().skew())

# ---- FIG 1: target histogram, raw vs log1p ---------------------------------
y = df[TARGET].dropna().to_numpy(float)
mu, md, sk = y.mean(), np.median(y), skew(y)
yl = np.log1p(y); skl = skew(yl)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].hist(y, bins=120, color="#d62728", alpha=.85)
ax[0].axvline(mu, color="k", ls="--", lw=1.6, label=f"mean = {mu:.2f}")
ax[0].axvline(md, color="#1f77b4", ls="-", lw=1.6, label=f"median = {md:.0f}")
ax[0].set_title(f"RAW {TARGET}   skew = {sk:+.2f}  (strongly right-skewed)")
ax[0].set_xlabel("cases / province-week"); ax[0].set_ylabel("count")
ax[0].set_xlim(0, np.percentile(y, 99.5)); ax[0].legend()
ax[1].hist(yl, bins=80, color="#2ca02c", alpha=.85)
ax[1].axvline(yl.mean(), color="k", ls="--", lw=1.6, label=f"mean = {yl.mean():.2f}")
ax[1].axvline(np.median(yl), color="#1f77b4", ls="-", lw=1.6,
              label=f"median = {np.median(yl):.2f}")
ax[1].set_title(f"log1p({TARGET})   skew = {skl:+.2f}  (near-symmetric)")
ax[1].set_xlabel("log1p(cases)"); ax[1].legend()
fig.suptitle(f"Target skewness — mean {mu:.2f} > median {md:.0f} is WHY a Poisson "
             "(mean) fit over-predicts quiet weeks", fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"skew_01_target_hist.png", dpi=150); plt.close(fig)
print(f"raw skew={sk:+.3f}  log1p skew={skl:+.3f}  mean={mu:.2f} median={md:.0f}")
print("Saved skew_01_target_hist.png")

# ---- FIG 2: skewness of the 12 model features ------------------------------
fs = pd.Series({f: skew(df[f]) for f in FEATURES}).sort_values()
fig, ax = plt.subplots(figsize=(10, 6.5))
cols = ["#d62728" if abs(v) > 1 else "#7f7f7f" for v in fs.values]
ax.barh(fs.index, fs.values, color=cols, alpha=.88)
for v in (-1, 1): ax.axvline(v, color="k", ls=":", lw=1)
ax.axvline(0, color="k", lw=.8)
for i, v in enumerate(fs.values):
    ax.text(v + (0.15 if v >= 0 else -0.15), i, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=8)
ax.set_xlabel("skewness (adjusted Fisher–Pearson)")
ax.set_title("Model-feature skewness — red = |skew| > 1 (deck threshold)\n"
             "Trees are rank-based, so no transform is applied", fontsize=12)
ax.margins(x=0.18); fig.tight_layout()
fig.savefig(OUT/"skew_02_feature_skew.png", dpi=150); plt.close(fig)
print("Saved skew_02_feature_skew.png")

# ---- FIG 3: histogram of skewness across ALL numeric columns ---------------
SKIP = {"year_ce","week","year","prov_lat","prov_lon","_pord","population",
        "log_population","cases_per_100k"}
num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in SKIP]
sk_all = pd.Series({c: skew(df[c]) for c in num}).replace(
    [np.inf, -np.inf], np.nan).dropna()
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.hist(sk_all.values, bins=40, color="#1f77b4", alpha=.85, edgecolor="white")
for v in (-1, 1): ax.axvline(v, color="#d62728", ls="--", lw=1.5)
ax.axvline(0, color="k", lw=.8)
n_strong = int((sk_all.abs() > 1).sum())
ax.set_title(f"Distribution of skewness across {len(sk_all)} numeric columns — "
             f"{n_strong} ({100*n_strong/len(sk_all):.0f}%) have |skew| > 1", fontsize=12)
ax.set_xlabel("skewness"); ax.set_ylabel("number of columns")
fig.tight_layout(); fig.savefig(OUT/"skew_03_skew_histogram.png", dpi=150); plt.close(fig)
print(f"Saved skew_03_skew_histogram.png  ({n_strong}/{len(sk_all)} strongly skewed)")

sk_all.sort_values(ascending=False).rename("skew").to_csv(
    OUT/"skew_table.csv", header=True, encoding="utf-8-sig")
print(f"Saved skew_table.csv\n\nTop 10 most skewed:\n"
      f"{sk_all.sort_values(ascending=False).head(10).round(3).to_string()}")
