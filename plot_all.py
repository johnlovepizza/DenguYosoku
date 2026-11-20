#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_all.py — ALL 7 SCRIPTS MERGED INTO ONE FILE, NOTHING EDITED.

Embedded verbatim (original logic, strings, colours, filenames, defaults):
  1. predict_2026-10-25.py          -> predict_week/predictions.csv   (needed by 3 + 4)
  2. plot_DenguYosoku.py            -> DenguYosoku_plots/* + province_individual/*
  3. plot_top5_worst_mae_actual.py  -> DenguYosoku_plots/top5_mae_blind_<date>/*
  4. plot_top5_best_mae_actual.py   -> DenguYosoku_plots/best5_mae_blind_<date>/*
  5. raw_cases_hist_swapped.py      -> DenguYosoku_plots/raw_cases_hist_vertical.png
  6. skew_report.py                 -> DenguYosoku_plots/skew_0*.png + skew_table.csv
  7. skew_split.py                  -> DenguYosoku_plots/skew_0*.png + skew_*split*.csv

HOW IT STAYS EXACT
  Each script is written to a temp .py NEXT TO this file (so
  Path(__file__).resolve().parent is unchanged) and executed in its OWN
  fresh Python subprocess: same interpreter, same working directory,
  same environment, same argv as `python <original>.py`.
  Fresh process = fresh matplotlib / seaborn / font state, so no rcParams,
  grids, palettes or fonts leak between scripts. Every PNG / CSV / HTML is
  exactly what the standalone script writes. Temp files are deleted after.

RUN
  python plot_all.py                      # all 7, in dependency order
  python plot_all.py --list
  python plot_all.py skew_report skew_split
  python plot_all.py --stop-on-error
  (run it from the same folder you normally run the scripts from)

Per-script CLI args: edit ARGS below (empty list = plain standalone run).
"""
import argparse, os, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent

ORDER = [
    "predict_2026-10-25",
    "plot_DenguYosoku",
    "plot_top5_worst_mae_actual",
    "plot_top5_best_mae_actual",
    "blind_compare_hist",
    "raw_cases_hist_swapped",
    "skew_report",
    "skew_split",
]

# Extra command-line args per script. Empty = exactly `python <script>.py`.
# Examples:
#   ARGS["raw_cases_hist_swapped"] = ["--bins", "80", "--clip", "99"]
#   ARGS["skew_report"]            = [r"D:\data\dengue_full.csv"]
ARGS = {name: [] for name in ORDER}

SOURCES = {}

# ============================================================
# 1) predict_2026-10-25.py
# ============================================================
SOURCES["predict_2026-10-25"] = r'''
import json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
D = Path("predict_week")
d = pd.read_csv(D/"predict_2026-10-25.csv", encoding="utf-8-sig")
F = json.loads((D/"features.json").read_text())
assert "cases" not in d.columns, "prediction file must not contain the answer"
m = lgb.Booster(model_file="DenguYosoku/model.txt")
assert m.num_feature() == len(F), f"model wants {m.num_feature()} features, file has {len(F)}"
d["pred"] = np.clip(m.predict(d[F].to_numpy(np.float32)), 0, None)   # column ORDER matters
d[["province","week_start","pred"]].to_csv(D/"predictions.csv", index=False, encoding="utf-8-sig")
print(d[["province","week_start","pred"]].round(2).to_string(index=False))
print(f"\nnational total = {d.pred.sum():.0f} cases")
'''

# ============================================================
# 2) plot_DenguYosoku.py
# ============================================================
SOURCES["plot_DenguYosoku"] = r'''
#!/usr/bin/env python3
"""Plot all graphs for the DenguYosogu run.
Input : DenguYosoku/holdout_predictions.csv  (master order, with pred + naive + residual)
DenguYosoku/metrics.json             (headline numbers)
DenguYosoku/feature_importance.csv   (gain per feature)
Output: DenguYosoku_plots/  (all PNGs + interactive HTML)"""
import warnings; warnings.filterwarnings("ignore")
import json, sys, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False       # ASCII '-' not U+2212
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"  # log-axis digits
import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.font_manager as fm
import matplotlib.ticker as mticker                     # FIX: plain-text symlog ticks
import seaborn as sns
# ============================================================
# PATHS
# ============================================================
BASE = Path(__file__).resolve().parent
PRED = BASE / "DenguYosoku" / "holdout_predictions.csv"
META = BASE / "DenguYosoku" / "metrics.json"
IMP  = BASE / "DenguYosoku" / "feature_importance.csv"
OUT  = BASE / "DenguYosoku_plots"; OUT.mkdir(exist_ok=True)
IND  = OUT / "province_individual"; IND.mkdir(exist_ok=True)
if not PRED.exists():
    sys.exit(f"ERROR: run DenguYosoku.py first.\nMissing: {PRED}")
# ============================================================
# THAI FONTS
# ============================================================
sns.set_theme(style="whitegrid")
avail = {f.name for f in fm.fontManager.ttflist}
thai = [f for f in ["Noto Sans Thai","Noto Serif Thai","Sarabun","TH Sarabun New","Loma","Garuda"] if f in avail] or ["DejaVu Sans"]
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":200,"font.size":10,
                     "font.family":thai+["DejaVu Sans"],"axes.unicode_minus":False,
                     "axes.grid":True,"grid.alpha":0.25,
                     # FIX: re-assert AFTER seaborn so mathtext keeps a font with digits
                     "mathtext.fontset":"dejavusans","mathtext.default":"regular"})
# ============================================================
# LOAD
# ============================================================
df = pd.read_csv(PRED, encoding="utf-8-sig", low_memory=False)
df["province"]   = df["province"].astype(str).str.strip()
df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
for c in ["cases","pred","naive"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["province","week_start","cases","pred"]).copy()
# derive residual if missing
if "residual" not in df.columns:
    df["residual"] = df["cases"] - df["pred"]
df["abs_err"]   = df["residual"].abs()
df["log_ratio"] = np.log((df["cases"]+1.0)/(df["pred"]+1.0))
df["month_label"] = df["week_start"].dt.to_period("M").astype(str)
# plots expect 'naive'; alias persistence (the pre-registered bar), not seasonal naive
# FIX: guard missing columns, coerce dtype, and record what the baseline actually IS
NAIVE_LABEL = "Naive baseline"
if "naive" not in df.columns:
    if "persist" in df.columns:
        df["naive"] = df["persist"];         NAIVE_LABEL = "Persistence (t-1)"
    elif "seasonal_naive" in df.columns:
        df["naive"] = df["seasonal_naive"];  NAIVE_LABEL = "Seasonal naive (t-52)"
    else:
        df["naive"] = np.nan;                NAIVE_LABEL = "Naive (missing)"
df["naive"] = pd.to_numeric(df["naive"], errors="coerce")
# master province order (already น่าน-first from the CSV, but lock it)
PROVINCE_ORDER = list(dict.fromkeys(df["province"]))
print(f"Rows: {len(df)} | Provinces: {len(PROVINCE_ORDER)} | "
      f"{df['week_start'].min().date()} -> {df['week_start'].max().date()}")
print(f"First 3 provinces: {PROVINCE_ORDER[:3]}")
print(f"Baseline plotted as: {NAIVE_LABEL}")
# headline metrics from JSON
meta = {}
if META.exists():
    meta = json.loads(META.read_text())
    print(f"\nDenguYosogu: MAE={meta.get('mae',0):.4f} | R2={meta.get('r2',0):.4f} | "
          f"WAPE={meta.get('wape',0):.4f} | Skill={meta.get('skill_pct',0):.1f}%")
# ============================================================
# METRIC HELPERS
# ============================================================
EPS = 1e-9
def mae(y,p):  return float(np.mean(np.abs(np.asarray(y,float)-np.clip(np.asarray(p,float),0,None))))
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y,float)-np.clip(np.asarray(p,float),0,None))**2)))
def r2(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),0,None)
    ss=np.sum((y-y.mean())**2)
    return float(1-np.sum((y-p)**2)/ss) if ss>0 else np.nan
def wape(y,p):
    # FIX: clip predictions like mae/rmse/r2 do, and guard zero denominator
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),0,None)
    d=np.sum(np.abs(y))
    return float(np.sum(np.abs(y-p))/d) if d>0 else np.nan
def pdev(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),EPS,None)
    return float(2*np.mean(np.where(y>0,y*np.log((y+EPS)/p),0.0)-(y-p)))
has_naive = "naive" in df.columns and df["naive"].notna().any()
# ============================================================
# PLOT 1: NATIONAL WEEKLY + RESIDUAL
# ============================================================
nat = df.groupby("week_start",as_index=False).agg(
    actual=("cases","sum"), predicted=("pred","sum"))
if has_naive:
    # FIX: align by week label instead of relying on positional .values
    nat["naive_sum"] = (df.groupby("week_start")["naive"].sum()
                        .reindex(nat["week_start"]).to_numpy())
nat["residual"] = nat["actual"] - nat["predicted"]
nat["r4a"] = nat["actual"].rolling(4,min_periods=1).mean()
nat["r4p"] = nat["predicted"].rolling(4,min_periods=1).mean()
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(16,8.5),sharex=True,
                             gridspec_kw={"height_ratios":[3,1]})
ax1.plot(nat.week_start, nat.actual, "-o", ms=3, lw=1.8, color="#1f77b4", label="Actual")
ax1.plot(nat.week_start, nat.predicted, "-s", ms=3, lw=1.8, alpha=.85, color="#d62728", label="Predicted")
if has_naive:
    # FIX: was hardcoded "Seasonal naive" while the column is persistence
    ax1.plot(nat.week_start, nat.naive_sum, "--", lw=1.0, color="#7f7f7f", alpha=.6, label=NAIVE_LABEL)
ax1.plot(nat.week_start, nat.r4a, lw=2.2, color="#1f77b4", alpha=.45, ls="--", label="Actual 4wk roll")
ax1.plot(nat.week_start, nat.r4p, lw=2.2, color="#d62728", alpha=.45, ls="--", label="Pred 4wk roll")
ax1.fill_between(nat.week_start, nat.actual, nat.predicted, color="gray", alpha=.12, label="Gap")
title_extra = ""
if meta:
    title_extra = (f"\nMAE={meta['mae']:.4f} | RMSE={meta['rmse']:.4f} | "
                   f"R2={meta['r2']:.4f} | WAPE={meta['wape']:.4f} | Skill={meta['skill_pct']:.1f}%")
ax1.set_title(f"DenguYosogu — National Weekly Dengue Forecast{title_extra}", fontsize=13)
ax1.set_ylabel("Total cases / week"); ax1.set_ylim(bottom=0)
ax1.legend(loc="upper left", fontsize=9)
colors = np.where(nat.residual >= 0, "#2ca02c", "#d62728")
ax2.bar(nat.week_start, nat.residual, width=5, color=colors, alpha=.8)
ax2.axhline(0, color="k", lw=1)
ax2.set_ylabel("Actual − Predicted"); ax2.set_xlabel("Week start")
ax2.set_title("National residual: green = under-predicted, red = over-predicted", fontsize=11)
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(OUT/"01_national_weekly.png", bbox_inches="tight"); plt.close(fig)
print("Saved 01_national_weekly.png")
# ============================================================
# PLOT 2: PER-PROVINCE INDIVIDUAL (master order)
# ============================================================
def safe(p): return "".join(c if c.isalnum() or c in "-_" else "_" for c in p)
for i, prov in enumerate(PROVINCE_ORDER, start=1):
    g = df[df.province == prov].sort_values("week_start")
    pm = mae(g.cases, g.pred); pr = rmse(g.cases, g.pred)
    pp = r2(g.cases, g.pred); pb = float(g.residual.mean())
    pw = wape(g.cases.values, np.clip(g.pred.values,0,None))
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(13,5.8),sharex=True,
                                 gridspec_kw={"height_ratios":[3,1]})
    ax1.plot(g.week_start, g.cases, "-o", ms=2.8, lw=1.5, color="#1f77b4", label="Actual")
    ax1.plot(g.week_start, g.pred, "-s", ms=2.8, lw=1.5, alpha=.85, color="#d62728", label="Predicted")
    if has_naive:
        ax1.plot(g.week_start, g.naive, "--", lw=0.9, color="#7f7f7f", alpha=.6, label="Naive")
    ax1.fill_between(g.week_start, g.cases, g.pred, color="gray", alpha=.12)
    ax1.set_title(f"#{i} {prov}\nMAE={pm:.3f} | RMSE={pr:.3f} | R2={pp:.3f} | WAPE={pw:.3f} | Bias={pb:.3f}",
                  fontsize=12)
    ax1.set_ylabel("Cases / week"); ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left", fontsize=8)
    rc = np.where(g.residual.values >= 0, "#2ca02c", "#d62728")
    ax2.bar(g.week_start, g.residual, width=5, color=rc, alpha=.8)
    ax2.axhline(0, color="k", lw=1)
    ax2.set_ylabel("Error"); ax2.set_xlabel("Week start")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(IND/f"{i:02d}_{safe(prov)}.png", bbox_inches="tight"); plt.close(fig)
print(f"Saved {len(PROVINCE_ORDER)} individual plots -> {IND}")
# ============================================================
# PLOT 3: MULTI-PAGE GRID (master order)
# ============================================================
cols, rpp = 4, 8; per = cols*rpp; pages = math.ceil(len(PROVINCE_ORDER)/per)
for page in range(pages):
    sub = PROVINCE_ORDER[page*per:min((page+1)*per, len(PROVINCE_ORDER))]
    fig, axes = plt.subplots(rpp, cols, figsize=(22, 3.2*rpp), squeeze=False)
    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(sub): ax.axis("off"); continue
        prov = sub[idx]
        g = df[df.province == prov].sort_values("week_start")
        ax.plot(g.week_start, g.cases, "-o", ms=1.8, lw=1.0, color="#1f77b4", label="Actual")
        ax.plot(g.week_start, g.pred, "-s", ms=1.8, lw=1.0, alpha=.85, color="#d62728", label="Predicted")
        ax.fill_between(g.week_start, g.cases, g.pred, color="gray", alpha=.10)
        ax.set_title(f"{prov}\nMAE={mae(g.cases,g.pred):.2f} R2={r2(g.cases,g.pred):.3f}", fontsize=9)
        ax.set_ylim(bottom=0); ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if idx == 0: ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"DenguYosogu — Province Grid Page {page+1}/{pages} (master order)", fontsize=14)
    fig.tight_layout(rect=[0,0,1,.985])
    fig.savefig(OUT/f"03_grid_page_{page+1}.png", bbox_inches="tight"); plt.close(fig)
    print(f"Saved 03_grid_page_{page+1}.png")
# ============================================================
# PLOT 4: RESIDUAL HEATMAP (province × week)
# ============================================================
res_pv = df.pivot_table(index="province", columns="week_start", values="residual", aggfunc="mean")
res_pv = res_pv.reindex(PROVINCE_ORDER)
res_pv.columns = res_pv.columns.strftime("%Y-%m-%d")
fh = max(10, 0.17*len(PROVINCE_ORDER)+3)
fig, ax = plt.subplots(figsize=(26, fh))
sns.heatmap(res_pv, center=0, cmap="RdBu_r", robust=True,
            cbar_kws={"label":"Actual − Predicted"}, ax=ax)
ax.set_title("DenguYosogu — Weekly Residual Heatmap\nRed = over-predicted, Blue = under-predicted", fontsize=14)
ax.set_xlabel("Week start"); ax.set_ylabel("Province")
plt.xticks(rotation=90, fontsize=5.5); plt.yticks(fontsize=8)
fig.tight_layout()
fig.savefig(OUT/"04_heatmap_residual.png", bbox_inches="tight"); plt.close(fig)
print("Saved 04_heatmap_residual.png")
# ============================================================
# PLOT 5: LOG-RATIO HEATMAP
# ============================================================
lr_pv = df.pivot_table(index="province", columns="week_start", values="log_ratio", aggfunc="mean")
lr_pv = lr_pv.reindex(PROVINCE_ORDER)
lr_pv.columns = lr_pv.columns.strftime("%Y-%m-%d")
fig, ax = plt.subplots(figsize=(26, fh))
sns.heatmap(lr_pv, center=0, cmap="coolwarm", robust=True,
            cbar_kws={"label":"log((actual+1)/(pred+1))"}, ax=ax)
ax.set_title("DenguYosogu — Log-Ratio Heatmap (scale-free error)", fontsize=14)
ax.set_xlabel("Week start"); ax.set_ylabel("Province")
plt.xticks(rotation=90, fontsize=5.5); plt.yticks(fontsize=8)
fig.tight_layout()
fig.savefig(OUT/"05_heatmap_logratio.png", bbox_inches="tight"); plt.close(fig)
print("Saved 05_heatmap_logratio.png")
# ============================================================
# PLOT 6: ACTUAL vs PREDICTED INTENSITY (side by side)
# ============================================================
act_pv = df.pivot_table(index="province", columns="week_start", values="cases", aggfunc="mean")
act_pv = act_pv.reindex(PROVINCE_ORDER); act_pv.columns = act_pv.columns.strftime("%Y-%m-%d")
prd_pv = df.pivot_table(index="province", columns="week_start", values="pred", aggfunc="mean")
prd_pv = prd_pv.reindex(PROVINCE_ORDER); prd_pv.columns = prd_pv.columns.strftime("%Y-%m-%d")
fig, axes = plt.subplots(1, 2, figsize=(34, fh))
sns.heatmap(np.log1p(act_pv), cmap="viridis", ax=axes[0], cbar_kws={"label":"log1p(actual)"})
axes[0].set_title("Actual weekly cases (log1p)", fontsize=13)
axes[0].set_xlabel("Week start"); axes[0].set_ylabel("Province")
sns.heatmap(np.log1p(prd_pv), cmap="viridis", ax=axes[1], yticklabels=False, cbar=False)
axes[1].set_title("Predicted weekly cases (log1p)", fontsize=13)
axes[1].set_xlabel("Week start")
for ax in axes:
    plt.sca(ax); plt.xticks(rotation=90, fontsize=5.0); plt.yticks(fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT/"06_heatmap_intensity.png", bbox_inches="tight"); plt.close(fig)
print("Saved 06_heatmap_intensity.png")
# ============================================================
# PLOT 7: MONTHLY OVERVIEW
# ============================================================
mon = df.groupby(["province","month_label"])["residual"].mean().unstack().reindex(PROVINCE_ORDER)
fig, ax = plt.subplots(figsize=(20, max(8, 0.20*len(PROVINCE_ORDER)+3)))
sns.heatmap(mon, center=0, cmap="RdBu_r", robust=True,
            cbar_kws={"label":"mean actual − predicted"}, ax=ax)
ax.set_title("DenguYosogu — Monthly Mean Error by Province", fontsize=14)
ax.set_xlabel("Month"); ax.set_ylabel("Province")
plt.xticks(rotation=90, fontsize=7); plt.yticks(fontsize=8)
fig.tight_layout()
fig.savefig(OUT/"07_heatmap_monthly.png", bbox_inches="tight"); plt.close(fig)
print("Saved 07_heatmap_monthly.png")
# ============================================================
# PLOT 8: SCATTER + BINNED FIT
# ============================================================
fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(df.cases, df.pred, s=10, alpha=.18, color="#1f77b4", edgecolors="none", label="province-week")
try:
    b = pd.qcut(df.cases, q=25, duplicates="drop")
    bb = df.groupby(b, observed=True).agg(x=("cases","mean"), y=("pred","mean")).reset_index(drop=True)
    ax.plot(bb.x, bb.y, "o-", color="#d62728", lw=2.2, ms=5, label="binned mean fit")
except Exception: pass
lim = float(max(df.cases.max(), df.pred.max())*1.05)
ax.plot([0,lim],[0,lim], "k--", lw=1.5, label="perfect")
ax.set_xlim(0,lim); ax.set_ylim(0,lim)
ax.set_xlabel("Actual cases"); ax.set_ylabel("Predicted cases")
sc = f" | Skill={meta['skill_pct']:.1f}%" if meta else ""
ax.set_title(f"DenguYosogu — Regression Fit\nMAE={mae(df.cases,df.pred):.4f} | RMSE={rmse(df.cases,df.pred):.4f} | R2={r2(df.cases,df.pred):.4f}{sc}", fontsize=13)
ax.legend(loc="upper left"); fig.tight_layout()
fig.savefig(OUT/"08_scatter_fit.png", bbox_inches="tight"); plt.close(fig)
print("Saved 08_scatter_fit.png")
# ============================================================
# PLOT 9: RESIDUAL DISTRIBUTION + ERROR BY BIN
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.histplot(df.residual, bins=100, kde=True, color="#8e44ad", ax=axes[0])
axes[0].axvline(0, color="k", ls="--", lw=1.2)
bias = float(df.residual.mean())
axes[0].axvline(bias, color="red", ls="--", lw=1.5, label=f"bias={bias:.3f}")
axes[0].set_title("Distribution of Weekly Errors"); axes[0].set_xlabel("Actual − Predicted")
axes[0].set_ylabel("Count"); axes[0].legend()
try:
    cb = pd.qcut(df.cases, q=10, duplicates="drop")
    eb = df.groupby(cb, observed=True).agg(mr=("residual","mean"), mae_=("abs_err","mean")).reset_index()
    eb["mid"] = eb.cases.apply(lambda x: float(x.mid))
    axes[1].plot(eb.mid, eb.mr, "o-", color="#d62728", label="mean error")
    axes[1].plot(eb.mid, eb.mae_, "s-", color="#1f77b4", label="mean |error|")
    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].set_xscale("symlog", linthresh=1)
    # FIX: symlog's default formatter emits mathtext ($\mathdefault{...}$), which was
    # rendering dummy glyphs. Plain numbers remove the mathtext dependency entirely.
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:,.0f}"))
    axes[1].xaxis.set_minor_formatter(mticker.NullFormatter())
    axes[1].set_title("Error by Actual Case Level"); axes[1].set_xlabel("bin midpoint")
    axes[1].set_ylabel("Error"); axes[1].legend()
except Exception as e:
    axes[1].text(.05,.5,f"failed:\n{e}", transform=axes[1].transAxes); axes[1].axis("off")
fig.tight_layout()
fig.savefig(OUT/"09_residual_bins.png", bbox_inches="tight"); plt.close(fig)
print("Saved 09_residual_bins.png")
# ============================================================
# PLOT 10: FEATURE IMPORTANCE (horizontal bar)
# ============================================================
if IMP.exists():
    imp = pd.read_csv(IMP)
    imp = imp.sort_values("gain", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp.feature, imp.gain_pct, color="#2ca02c", alpha=.85)
    ax.set_xlabel("Split gain (%)"); ax.set_ylabel("Feature")
    ax.set_title("DenguYosogu — Feature Importance (11 features)", fontsize=13)
    for i, (g, pct) in enumerate(zip(imp.gain, imp.gain_pct)):
        ax.text(pct+.3, i, f"{pct:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT/"10_feature_importance.png", bbox_inches="tight"); plt.close(fig)
    print("Saved 10_feature_importance.png")
# ============================================================
# PLOT 11: SUBGROUP BAR (small / medium / large)
# ============================================================
size = df.groupby("province")["cases"].mean()
def bucket(x): return "small (<5/wk)" if x<5 else ("medium (5-20)" if x<20 else "large (>20/wk)")
grp = pd.Series({p: bucket(v) for p, v in size.items()})
df["group"] = df["province"].map(grp)
sub_rows = []
for g, sg in df.groupby("group"):
    y = sg.cases.values; p = np.clip(sg.pred.values, 0, None)
    # FIX: score model vs naive on the SAME rows; a single NaN naive used to
    # poison the whole mean and silently turn skill into NaN.
    nv = sg.naive.to_numpy(dtype=float) if has_naive else np.full(len(sg), np.nan)
    ok = np.isfinite(nv)
    nm = mae(y[ok], nv[ok]) if ok.any() else np.nan
    pm_paired = mae(y[ok], p[ok]) if ok.any() else np.nan
    sub_rows.append({"group":g, "n_prov":sg.province.nunique(),
                     "MAE":mae(y,p), "WAPE":wape(y,p), "R2":r2(y,p),
                     "skill_%":(nm-pm_paired)/nm*100 if np.isfinite(nm) and nm>0 else np.nan})
sub_df = pd.DataFrame(sub_rows).set_index("group").reindex(["small (<5/wk)","medium (5-20)","large (>20/wk)"])
sub_df.to_csv(OUT/"subgroup_summary.csv", encoding="utf-8-sig")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
x = np.arange(len(sub_df))
axes[0].bar(x, sub_df.MAE, color="#d62728", alpha=.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(sub_df.index, fontsize=8)
axes[0].set_ylabel("MAE"); axes[0].set_title("MAE by province size")
for i,v in enumerate(sub_df.MAE): axes[0].text(i,v+.05,f"{v:.2f}",ha="center",fontsize=9)
axes[1].bar(x, sub_df.WAPE*100, color="#f39c12", alpha=.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(sub_df.index, fontsize=8)
axes[1].set_ylabel("WAPE (%)"); axes[1].set_title("WAPE by province size")
for i,v in enumerate(sub_df.WAPE): axes[1].text(i,v*100+1,f"{v*100:.1f}%",ha="center",fontsize=9)
axes[2].bar(x, sub_df["skill_%"], color="#2ca02c", alpha=.85)
axes[2].set_xticks(x); axes[2].set_xticklabels(sub_df.index, fontsize=8)
axes[2].set_ylabel("Skill vs naive (%)"); axes[2].set_title("Skill by province size")
axes[2].axhline(50, ls="--", c="gray", lw=1)
for i,v in enumerate(sub_df["skill_%"]): axes[2].text(i,v+1,f"{v:.1f}%",ha="center",fontsize=9)
fig.suptitle("DenguYosogu — Performance by Province Size (holdout 2025→Oct2026)", fontsize=13)
fig.tight_layout(rect=[0,0,1,.95])
fig.savefig(OUT/"11_subgroup_bars.png", bbox_inches="tight"); plt.close(fig)
print("Saved 11_subgroup_bars.png")
print("\nSubgroup table:")
print(sub_df.round(4).to_string())
# ============================================================
# INTERACTIVE (optional)
# ============================================================
def mpl_cmap_to_plotly(name, n=11):
    # FIX: plotly has no 'coolwarm'. Convert the matplotlib cmap so the HTML
    # uses the same colours as the PNG instead of a different scale.
    from matplotlib.colors import to_hex
    cm = matplotlib.colormaps[name]
    return [[i/(n-1), to_hex(cm(i/(n-1)))] for i in range(n)]
try:
    import plotly.express as px
    for pv, name, scale, lab in [
            (res_pv, "interactive_residual", "RdBu_r", "Actual − Predicted"),
            (lr_pv,  "interactive_logratio", "coolwarm", "log ratio")]:
        f = px.imshow(pv, labels=dict(x="Week",y="Province",color=lab),
                      color_continuous_scale=mpl_cmap_to_plotly(scale), aspect="auto",
                      title=f"DenguYosogu — {name.replace('_',' ').title()}")
        f.update_layout(height=max(900,18*len(PROVINCE_ORDER)), width=2400,
                        font=dict(family=thai[0], size=12))
        f.update_xaxes(tickangle=90, tickfont=dict(size=8))
        f.update_yaxes(tickfont=dict(size=10))
        f.write_html(OUT/f"{name}.html")
    print("Saved interactive HTML heatmaps.")
except ImportError:
    print("Plotly not installed -> skip interactive (pip install plotly)")
except Exception as e:
    # FIX: optional block must never kill the run after all PNGs are written
    print(f"Interactive skipped: {type(e).__name__}: {e}")
# ============================================================
# FINAL
# ============================================================
print("\n" + "="*60)
print("DenguYosogu PLOTS — DONE")
print("="*60)
if meta:
    print(f"  MAE    = {meta['mae']:.4f}")
    print(f"  RMSE   = {meta['rmse']:.4f}")
    print(f"  WAPE   = {meta['wape']:.4f}")
    print(f"  R2     = {meta['r2']:.4f}")
    print(f"  Skill  = {meta['skill_pct']:.1f}%")
    print(f"  Holdout: {df['week_start'].min().date()} -> {df['week_start'].max().date()} ({len(df)} rows)")
print(f"\nAll outputs -> {OUT}")
for f in sorted(OUT.glob("*.png")): print(f"  {f.name}")
print(f"  province_individual/ ({len(PROVINCE_ORDER)} files)")
'''

# ============================================================
# 3) plot_top5_worst_mae_actual.py
# ============================================================
SOURCES["plot_top5_worst_mae_actual"] = r'''
#!/usr/bin/env python3
"""Top-N MAE provinces in the BLIND TEST - ACTUAL CASES ONLY.
Inputs:
predict_week/predictions.csv   <- from predict_2026-10-25.py
predict_week/answer_key.csv    <- private truth
dengue_full.csv                <- actual history (REQUIRED for the lines)
Blind file = ONE week x 77 provinces, so per-province MAE = |actual - pred|.
Output: DenguYosoku_plots/top5_mae_blind_2026-10-25/
00_top5_error_bar.png       <- bar chart (kept unchanged)
00_top5_lines_combined.png  <- NEW: all 5 provinces, one linear plot
01_<province>.png ... 05_<province>.png
blind_mae_ranking_all_provinces.csv
top5_blind_summary.csv
No prediction line anywhere. X = week, Y = actual cases.
Colors taken from plot_DenguYosoku.py.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import matplotlib.font_manager as fm
import seaborn as sns
# ============================================================
# CONFIG
# ============================================================
TOP_N        = 5
YEAR         = 2026     # plot window; None -> blind week's year
TICK_MONTHS  = 1
MAKE_BAR     = True     # keep the bar chart
INDIVIDUAL   = True     # keep the 5 per-province line charts
HIST_OVERRIDE = None    # e.g. r"D:\data\dengue_full.csv" if it lives elsewhere
BASE  = Path(__file__).resolve().parent
BLIND = BASE / "predict_week"
PREDS = BLIND / "predictions.csv"
KEY   = BLIND / "answer_key.csv"
# old palette from plot_DenguYosoku.py
OLD_BLUE, OLD_RED, OLD_GREEN = "#1f77b4", "#d62728", "#2ca02c"
OLD_PURPLE, OLD_ORANGE, OLD_GRAY = "#8e44ad", "#f39c12", "#7f7f7f"
PALETTE = [OLD_BLUE, OLD_RED, OLD_GREEN, OLD_PURPLE, OLD_ORANGE]
for f in (PREDS, KEY):
    if not f.exists():
        sys.exit(f"ERROR: missing {f}\nRun make_blind_holdout.py then predict_2026-10-25.py first.")
# ============================================================
# THAI FONTS
# ============================================================
sns.set_theme(style="whitegrid")
avail = {f.name for f in fm.fontManager.ttflist}
thai = [f for f in ["Noto Sans Thai", "Noto Serif Thai", "Sarabun",
                    "TH Sarabun New", "Loma", "Garuda"] if f in avail] or ["DejaVu Sans"]
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
                     "font.family": thai + ["DejaVu Sans"], "axes.unicode_minus": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "mathtext.fontset": "dejavusans", "mathtext.default": "regular"})
# ============================================================
# LOAD BLIND PREDICTIONS + ANSWER KEY
# ============================================================
def load(path, valcol):
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    d.columns = d.columns.str.strip()
    miss = {"province", "week_start", valcol} - set(d.columns)
    if miss:
        sys.exit(f"ERROR: {path.name} missing {sorted(miss)} (has {list(d.columns)})")
    d = d[["province", "week_start", valcol]].copy()
    d["province"]   = d["province"].astype(str).str.strip()
    d["week_start"] = pd.to_datetime(d["week_start"], errors="coerce")
    d[valcol]       = pd.to_numeric(d[valcol], errors="coerce")
    d = d.dropna()
    dup = int(d.duplicated(["province", "week_start"]).sum())
    if dup:
        print(f"Note: {path.name} had {dup} duplicate rows -> kept last")
    d = d.drop_duplicates(["province", "week_start"], keep="last")
    return d
pr, ky = load(PREDS, "pred"), load(KEY, "cases")
b = pr.merge(ky, on=["province", "week_start"], how="inner", validate="one_to_one")
if b.empty:
    sys.exit("ERROR: predictions and answer key share no (province, week_start) rows.")
lost_p, lost_k = set(pr.province) - set(ky.province), set(ky.province) - set(pr.province)
if lost_p or lost_k:
    print(f"WARNING: unmatched provinces -> preds only {len(lost_p)}, key only {len(lost_k)}")
b["pred_clipped"] = b["pred"].clip(lower=0)
b["error"]        = b["cases"] - b["pred_clipped"]
b["abs_err"]      = b["error"].abs()
dates = sorted(b.week_start.dt.date.unique())
TAG   = str(dates[0]) if len(dates) == 1 else f"{dates[0]}_to_{dates[-1]}"
LABEL = str(dates[0]) if len(dates) == 1 else f"{dates[0]} -> {dates[-1]}"
if YEAR is None:
    YEAR = int(b.week_start.dt.year.max())
OUT = BASE / "DenguYosoku_plots" / f"top{TOP_N}_mae_blind_{TAG}"
OUT.mkdir(parents=True, exist_ok=True)
nat_mae  = float(b.abs_err.mean())
nat_wape = float(b.abs_err.sum() / b.cases.abs().sum()) if b.cases.abs().sum() > 0 else np.nan
print(f"Blind test: {LABEL} | {b.province.nunique()} provinces | {len(dates)} week(s)"
      f" -> per-province MAE = |error|")
print(f"National: MAE={nat_mae:.3f} | WAPE={nat_wape:.3f} | "
      f"actual={b.cases.sum():.0f} | pred={b.pred_clipped.sum():.0f} | bias={b.error.mean():+.3f}")
# ============================================================
# RANK BY BLIND MAE
# ============================================================
rows = []
for prov, g in b.groupby("province", sort=False):
    last = g.sort_values("week_start").iloc[-1]
    rows.append({"province": prov, "n_blind_weeks": int(len(g)),
                 "MAE": float(g.abs_err.mean()),
                 "RMSE": float(np.sqrt((g.error ** 2).mean())),
                 "blind_week": str(last.week_start.date()),
                 "actual": float(last.cases), "pred": float(last.pred_clipped),
                 "error": float(last.error), "abs_error": float(last.abs_err),
                 "abs_err_pct_of_actual": float(last.abs_err / max(float(last.cases), 1.0) * 100)})
rank = pd.DataFrame(rows).sort_values(["MAE", "abs_error"], ascending=False).reset_index(drop=True)
rank.insert(0, "rank", np.arange(1, len(rank) + 1))
rank.to_csv(OUT / "blind_mae_ranking_all_provinces.csv", index=False, encoding="utf-8-sig")
top = rank.head(TOP_N).copy()
print(f"\nTop {len(top)} provinces by BLIND-TEST MAE:")
print(top[["rank", "province", "MAE", "actual", "pred", "error"]].round(3).to_string(index=False))
# ============================================================
# ACTUAL HISTORY - REQUIRED, searched loudly
# ============================================================
cands = ([Path(HIST_OVERRIDE)] if HIST_OVERRIDE else []) + [
    BASE / "dengue_full.csv", BASE / "DenguYosoku" / "dengue_full.csv",
    BASE / "data" / "dengue_full.csv", Path.cwd() / "dengue_full.csv"]
HIST = next((p for p in cands if p.exists()), None)
if HIST is None:
    msg = "\n".join(str(p) for p in cands)
    sys.exit(f"ERROR: dengue_full.csv not found - the line plots need it.\nSearched:\n{msg}\n"
             f"Set HIST_OVERRIDE at the top of this file to its real path.")
h = pd.read_csv(HIST, encoding="utf-8-sig", low_memory=False)
h.columns = h.columns.str.strip()
miss = {"province", "week_start", "cases"} - set(h.columns)
if miss:
    sys.exit(f"ERROR: {HIST} missing columns {sorted(miss)} (has {list(h.columns)[:15]})")
h["province"]   = h["province"].astype(str).str.strip()
h["week_start"] = pd.to_datetime(h["week_start"], errors="coerce")
h["cases"]      = pd.to_numeric(h["cases"], errors="coerce")
h = h.dropna(subset=["province", "week_start", "cases"])
h = h[h.week_start.dt.year == YEAR]
if h.empty:
    sys.exit(f"ERROR: {HIST.name} has no {YEAR} rows.")
hist = (h.groupby(["province", "week_start"], as_index=False)["cases"].mean()
        .sort_values("week_start"))
print(f"\nHistory file : {HIST}")
print(f"{YEAR} window  : {hist.week_start.min().date()} -> {hist.week_start.max().date()} "
      f"| {hist.week_start.nunique()} weeks | {hist.province.nunique()} provinces")
# ============================================================
# BUILD CURVES + STATS
# ============================================================
def safe(p):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in p)
curves, summary = {}, []
for _, r in top.iterrows():
    prov = r["province"]
    bw = b[b.province == prov].sort_values("week_start")
    c  = hist[hist.province == prov][["week_start", "cases"]].copy()
    if c.empty:
        print(f"WARNING: '{prov}' not in {HIST.name} -> check spelling; skipped")
        continue
    extra = bw[~bw.week_start.isin(c.week_start)][["week_start", "cases"]]
    if len(extra):
        c = pd.concat([c, extra], ignore_index=True)
    c = c.sort_values("week_start").reset_index(drop=True)
    if len(c) < 2:
        print(f"WARNING: '{prov}' has only {len(c)} week in {YEAR} -> cannot draw a line")
    curves[prov] = c
    s = r.to_dict()
    s.update({"mean_cases": float(c.cases.mean()), "max_cases": float(c.cases.max()),
              "total_cases": float(c.cases.sum()), "n_weeks": int(len(c)), "year": YEAR})
    summary.append(s)
if not summary:
    sys.exit("ERROR: none of the top provinces matched the history file.")
color_of = {s["province"]: PALETTE[(s["rank"] - 1) % len(PALETTE)] for s in summary}
def dedup_legend(ax, loc="upper left", fs=9):
    hs, ls = ax.get_legend_handles_labels()
    u = dict(zip(ls, hs))
    ax.legend(u.values(), u.keys(), loc=loc, fontsize=fs)
def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=TICK_MONTHS))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylim(bottom=0)
# ============================================================
# PLOT A: ALL 5 PROVINCES ON ONE LINEAR PLOT  (NEW)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
for s in summary:
    prov, c, col = s["province"], curves[s["province"]], color_of[s["province"]]
    ax.plot(c.week_start, c.cases, "-o", ms=3.2, lw=1.8, color=col, alpha=.9,
            label=f"#{s['rank']} {prov} (MAE={s['MAE']:.1f})")
    bw = b[b.province == prov]
    for _, r in bw.iterrows():
        ax.scatter([r.week_start], [r.cases], marker="*", s=240, color=col,
                   edgecolor="k", linewidth=.7, zorder=6, label="Blind week (actual)")
    for w in sorted(b.week_start.unique()):
        ax.axvline(pd.Timestamp(w), color=OLD_GRAY, ls=":", lw=1.2, alpha=.8, label="Blind week")
ax.set_title(f"Top {len(summary)} blind-test MAE provinces - {YEAR} ACTUAL weekly cases\n"
             f"blind week {LABEL} | national MAE={nat_mae:.3f} | WAPE={nat_wape:.3f} "
             f"| actual only, no prediction", fontsize=13)
ax.set_xlabel("Week start"); ax.set_ylabel("Actual cases / week")
date_axis(ax); dedup_legend(ax, loc="upper left", fs=9)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(OUT / "00_top5_lines_combined.png", bbox_inches="tight"); plt.close(fig)
print("Saved 00_top5_lines_combined.png  (all 5 provinces, linear)")
# ============================================================
# PLOT B: INDIVIDUAL LINE CHARTS (same colors as combined)
# ============================================================
if INDIVIDUAL:
    for s in summary:
        prov, c, col = s["province"], curves[s["province"]], color_of[s["province"]]
        fig, ax = plt.subplots(figsize=(13, 4.8))
        ax.plot(c.week_start, c.cases, "-o", ms=3.2, lw=1.6, color=col,
                label=f"Actual cases {YEAR}")
        ax.fill_between(c.week_start, 0, c.cases, color=col, alpha=.10)
        for _, r in b[b.province == prov].iterrows():
            ax.axvline(r.week_start, color=OLD_GRAY, ls=":", lw=1.2, alpha=.8)
            ax.scatter([r.week_start], [r.cases], marker="*", s=230, color=col,
                       edgecolor="k", linewidth=.7, zorder=6, label="Blind week (actual)")
            ax.annotate(f"{r.cases:.0f}", (r.week_start, r.cases),
                        textcoords="offset points", xytext=(9, 11), fontsize=9)
        ax.set_title(f"#{s['rank']} {prov} - blind test {LABEL} | "
                     f"MAE = |error| = {s['MAE']:.2f} cases\n"
                     f"actual={s['actual']:.0f} | {YEAR} mean={s['mean_cases']:.2f}/wk | "
                     f"peak={s['max_cases']:.0f} | n={s['n_weeks']} weeks (actual only)", fontsize=12)
        ax.set_xlabel("Week start"); ax.set_ylabel("Actual cases / week")
        date_axis(ax); dedup_legend(ax)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(OUT / f"{s['rank']:02d}_{safe(prov)}.png", bbox_inches="tight"); plt.close(fig)
        print(f"Saved {s['rank']:02d}_{safe(prov)}.png  (MAE={s['MAE']:.2f})")
pd.DataFrame(summary).to_csv(OUT / f"top{len(summary)}_blind_summary.csv",
                             index=False, encoding="utf-8-sig")
# ============================================================
# PLOT C: BAR CHART - KEPT AS IS
# ============================================================
if MAKE_BAR:
    t = pd.DataFrame(summary).sort_values("MAE")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(t.province, t.MAE, color=OLD_RED, alpha=.85)
    for i, v in enumerate(t.MAE):
        ax.text(v + max(t.MAE) * .01, i, f"{v:.1f}", va="center", fontsize=9)
    ax.set_xlabel("Absolute error (cases) - blind week")
    ax.set_ylabel("Province")
    ax.set_title(f"Top {len(t)} blind-test MAE provinces - {LABEL}\n"
                 f"national MAE={nat_mae:.3f} | WAPE={nat_wape:.3f}", fontsize=12)
    ax.set_xlim(0, max(t.MAE) * 1.15)
    fig.tight_layout()
    fig.savefig(OUT / "00_top5_error_bar.png", bbox_inches="tight"); plt.close(fig)
    print("Saved 00_top5_error_bar.png")
# ============================================================
# FINAL
# ============================================================
print("\n" + "=" * 60)
print(f"TOP {len(summary)} BLIND-TEST MAE PROVINCES - DONE")
print("=" * 60)
print(f"Blind week : {LABEL}")
print(f"Curves     : {YEAR} actual cases, linear, X = week / Y = cases")
print(f"History    : {HIST}")
print(f"Outputs -> {OUT}")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}")
'''

# ============================================================
# 4) plot_top5_best_mae_actual.py
# ============================================================
SOURCES["plot_top5_best_mae_actual"] = r'''
#!/usr/bin/env python3
"""Top-N BEST / WORST MAE provinces in the BLIND TEST - ACTUAL CASES ONLY.
Inputs:
predict_week/predictions.csv   <- from predict_2026-10-25.py
predict_week/answer_key.csv    <- private truth
dengue_full.csv                <- actual history (REQUIRED for the lines)
Blind file = ONE week x 77 provinces, so per-province MAE = |actual - pred|.
MODE = "best"  -> 5 LOWEST MAE   -> DenguYosoku_plots/best5_mae_blind_<date>/
MODE = "worst" -> 5 HIGHEST MAE  -> DenguYosoku_plots/worst5_mae_blind_<date>/
00_top5_lines_combined.png   (all 5 provinces, one linear plot)
00_top5_error_bar.png        (bar chart)
01_<province>.png ... 05_<province>.png
blind_mae_ranking_all_provinces.csv
best5_blind_summary.csv
No prediction line anywhere. X = week, Y = actual cases.
Colors taken from plot_DenguYosoku.py.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import matplotlib.font_manager as fm
import seaborn as sns
# ============================================================
# CONFIG
# ============================================================
MODE         = "best"   # "best" = least MAE | "worst" = most MAE
TOP_N        = 5
RANK_METRIC  = "MAE"    # or "abs_err_pct_of_actual" for a scale-fair ranking
YEAR         = 2026     # plot window; None -> blind week's year
TICK_MONTHS  = 1
MAKE_BAR     = True
INDIVIDUAL   = True
HIST_OVERRIDE = None    # e.g. r"D:\data\dengue_full.csv"
if MODE not in ("best", "worst"):
    sys.exit('ERROR: MODE must be "best" or "worst"')
ASC  = (MODE == "best")
WORD = "LOWEST" if ASC else "HIGHEST"
BASE  = Path(__file__).resolve().parent
BLIND = BASE / "predict_week"
PREDS = BLIND / "predictions.csv"
KEY   = BLIND / "answer_key.csv"
# old palette from plot_DenguYosoku.py
OLD_BLUE, OLD_RED, OLD_GREEN = "#1f77b4", "#d62728", "#2ca02c"
OLD_PURPLE, OLD_ORANGE, OLD_GRAY = "#8e44ad", "#f39c12", "#7f7f7f"
PALETTE = [OLD_BLUE, OLD_RED, OLD_GREEN, OLD_PURPLE, OLD_ORANGE]
for f in (PREDS, KEY):
    if not f.exists():
        sys.exit(f"ERROR: missing {f}\nRun make_blind_holdout.py then predict_2026-10-25.py first.")
# ============================================================
# THAI FONTS
# ============================================================
sns.set_theme(style="whitegrid")
avail = {f.name for f in fm.fontManager.ttflist}
thai = [f for f in ["Noto Sans Thai", "Noto Serif Thai", "Sarabun",
                    "TH Sarabun New", "Loma", "Garuda"] if f in avail] or ["DejaVu Sans"]
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
                     "font.family": thai + ["DejaVu Sans"], "axes.unicode_minus": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "mathtext.fontset": "dejavusans", "mathtext.default": "regular"})
# ============================================================
# LOAD BLIND PREDICTIONS + ANSWER KEY
# ============================================================
def load(path, valcol):
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    d.columns = d.columns.str.strip()
    miss = {"province", "week_start", valcol} - set(d.columns)
    if miss:
        sys.exit(f"ERROR: {path.name} missing {sorted(miss)} (has {list(d.columns)})")
    d = d[["province", "week_start", valcol]].copy()
    d["province"]   = d["province"].astype(str).str.strip()
    d["week_start"] = pd.to_datetime(d["week_start"], errors="coerce")
    d[valcol]       = pd.to_numeric(d[valcol], errors="coerce")
    d = d.dropna()
    dup = int(d.duplicated(["province", "week_start"]).sum())
    if dup:
        print(f"Note: {path.name} had {dup} duplicate rows -> kept last")
    d = d.drop_duplicates(["province", "week_start"], keep="last")
    return d
pr, ky = load(PREDS, "pred"), load(KEY, "cases")
b = pr.merge(ky, on=["province", "week_start"], how="inner", validate="one_to_one")
if b.empty:
    sys.exit("ERROR: predictions and answer key share no (province, week_start) rows.")
lost_p, lost_k = set(pr.province) - set(ky.province), set(ky.province) - set(pr.province)
if lost_p or lost_k:
    print(f"WARNING: unmatched provinces -> preds only {len(lost_p)}, key only {len(lost_k)}")
b["pred_clipped"] = b["pred"].clip(lower=0)
b["error"]        = b["cases"] - b["pred_clipped"]
b["abs_err"]      = b["error"].abs()
dates = sorted(b.week_start.dt.date.unique())
TAG   = str(dates[0]) if len(dates) == 1 else f"{dates[0]}_to_{dates[-1]}"
LABEL = str(dates[0]) if len(dates) == 1 else f"{dates[0]} -> {dates[-1]}"
if YEAR is None:
    YEAR = int(b.week_start.dt.year.max())
OUT = BASE / "DenguYosoku_plots" / f"{MODE}{TOP_N}_mae_blind_{TAG}"
OUT.mkdir(parents=True, exist_ok=True)
nat_mae  = float(b.abs_err.mean())
nat_wape = float(b.abs_err.sum() / b.cases.abs().sum()) if b.cases.abs().sum() > 0 else np.nan
print(f"MODE: {MODE.upper()} ({WORD} MAE) | ranking metric: {RANK_METRIC}")
print(f"Blind test: {LABEL} | {b.province.nunique()} provinces | {len(dates)} week(s)"
      f" -> per-province MAE = |error|")
print(f"National: MAE={nat_mae:.3f} | WAPE={nat_wape:.3f} | "
      f"actual={b.cases.sum():.0f} | pred={b.pred_clipped.sum():.0f} | bias={b.error.mean():+.3f}")
# ============================================================
# RANK BY BLIND MAE  (ascending = best, descending = worst)
# ============================================================
rows = []
for prov, g in b.groupby("province", sort=False):
    last = g.sort_values("week_start").iloc[-1]
    rows.append({"province": prov, "n_blind_weeks": int(len(g)),
                 "MAE": float(g.abs_err.mean()),
                 "RMSE": float(np.sqrt((g.error ** 2).mean())),
                 "blind_week": str(last.week_start.date()),
                 "actual": float(last.cases), "pred": float(last.pred_clipped),
                 "error": float(last.error), "abs_error": float(last.abs_err),
                 "abs_err_pct_of_actual": float(last.abs_err / max(float(last.cases), 1.0) * 100)})
rank = pd.DataFrame(rows)
if RANK_METRIC not in rank.columns:
    sys.exit(f"ERROR: RANK_METRIC '{RANK_METRIC}' not in {list(rank.columns)}")
rank = rank.sort_values([RANK_METRIC, "MAE"], ascending=ASC).reset_index(drop=True)
rank.insert(0, "rank", np.arange(1, len(rank) + 1))
rank.to_csv(OUT / "blind_mae_ranking_all_provinces.csv", index=False, encoding="utf-8-sig")
top = rank.head(TOP_N).copy()
print(f"\nTop {len(top)} {WORD}-MAE provinces (blind test):")
print(top[["rank", "province", "MAE", "actual", "pred", "error", "abs_err_pct_of_actual"]]
      .round(3).to_string(index=False))
# ============================================================
# ACTUAL HISTORY - REQUIRED
# ============================================================
cands = ([Path(HIST_OVERRIDE)] if HIST_OVERRIDE else []) + [
    BASE / "dengue_full.csv", BASE / "DenguYosoku" / "dengue_full.csv",
    BASE / "data" / "dengue_full.csv", Path.cwd() / "dengue_full.csv"]
HIST = next((p for p in cands if p.exists()), None)
if HIST is None:
    msg = "\n".join(str(p) for p in cands)
    sys.exit(f"ERROR: dengue_full.csv not found - the line plots need it.\nSearched:\n{msg}\n"
             f"Set HIST_OVERRIDE at the top of this file.")
h = pd.read_csv(HIST, encoding="utf-8-sig", low_memory=False)
h.columns = h.columns.str.strip()
miss = {"province", "week_start", "cases"} - set(h.columns)
if miss:
    sys.exit(f"ERROR: {HIST} missing columns {sorted(miss)} (has {list(h.columns)[:15]})")
h["province"]   = h["province"].astype(str).str.strip()
h["week_start"] = pd.to_datetime(h["week_start"], errors="coerce")
h["cases"]      = pd.to_numeric(h["cases"], errors="coerce")
h = h.dropna(subset=["province", "week_start", "cases"])
h = h[h.week_start.dt.year == YEAR]
if h.empty:
    sys.exit(f"ERROR: {HIST.name} has no {YEAR} rows.")
hist = (h.groupby(["province", "week_start"], as_index=False)["cases"].mean()
        .sort_values("week_start"))
print(f"\nHistory file : {HIST}")
print(f"{YEAR} window  : {hist.week_start.min().date()} -> {hist.week_start.max().date()} "
      f"| {hist.week_start.nunique()} weeks | {hist.province.nunique()} provinces")
# ============================================================
# BUILD CURVES + STATS
# ============================================================
def safe(p):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in p)
curves, summary = {}, []
for _, r in top.iterrows():
    prov = r["province"]
    bw = b[b.province == prov].sort_values("week_start")
    c  = hist[hist.province == prov][["week_start", "cases"]].copy()
    if c.empty:
        print(f"WARNING: '{prov}' not in {HIST.name} -> check spelling; skipped")
        continue
    extra = bw[~bw.week_start.isin(c.week_start)][["week_start", "cases"]]
    if len(extra):
        c = pd.concat([c, extra], ignore_index=True)
    c = c.sort_values("week_start").reset_index(drop=True)
    if len(c) < 2:
        print(f"WARNING: '{prov}' has only {len(c)} week in {YEAR} -> cannot draw a line")
    curves[prov] = c
    s = r.to_dict()
    s.update({"mean_cases": float(c.cases.mean()), "max_cases": float(c.cases.max()),
              "total_cases": float(c.cases.sum()), "n_weeks": int(len(c)), "year": YEAR})
    summary.append(s)
if not summary:
    sys.exit("ERROR: none of the selected provinces matched the history file.")
color_of = {s["province"]: PALETTE[(s["rank"] - 1) % len(PALETTE)] for s in summary}
def dedup_legend(ax, loc="upper left", fs=9):
    hs, ls = ax.get_legend_handles_labels()
    u = dict(zip(ls, hs))
    ax.legend(u.values(), u.keys(), loc=loc, fontsize=fs)
def date_axis(ax, ymax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=TICK_MONTHS))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylim(0, max(1.0, float(ymax) * 1.08))   # safe for all-zero curves
# ============================================================
# PLOT A: ALL 5 PROVINCES ON ONE LINEAR PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
for s in summary:
    prov, c, col = s["province"], curves[s["province"]], color_of[s["province"]]
    ax.plot(c.week_start, c.cases, "-o", ms=3.2, lw=1.8, color=col, alpha=.9,
            label=f"#{s['rank']} {prov} (MAE={s['MAE']:.2f})")
    for _, r in b[b.province == prov].iterrows():
        ax.scatter([r.week_start], [r.cases], marker="*", s=240, color=col,
                   edgecolor="k", linewidth=.7, zorder=6, label="Blind week (actual)")
    for w in sorted(b.week_start.unique()):
        ax.axvline(pd.Timestamp(w), color=OLD_GRAY, ls=":", lw=1.2, alpha=.8, label="Blind week")
ax.set_title(f"Top {len(summary)} {WORD} blind-test MAE provinces - {YEAR} ACTUAL weekly cases\n"
             f"blind week {LABEL} | national MAE={nat_mae:.3f} | WAPE={nat_wape:.3f} "
             f"| actual only, no prediction", fontsize=13)
ax.set_xlabel("Week start"); ax.set_ylabel("Actual cases / week")
date_axis(ax, max(s["max_cases"] for s in summary))
dedup_legend(ax, loc="upper left", fs=9)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(OUT / f"00_top{len(summary)}_lines_combined.png", bbox_inches="tight"); plt.close(fig)
print(f"Saved 00_top{len(summary)}_lines_combined.png  (all {len(summary)} provinces, linear)")
# ============================================================
# PLOT B: INDIVIDUAL LINE CHARTS (same colors as combined)
# ============================================================
if INDIVIDUAL:
    for s in summary:
        prov, c, col = s["province"], curves[s["province"]], color_of[s["province"]]
        fig, ax = plt.subplots(figsize=(13, 4.8))
        ax.plot(c.week_start, c.cases, "-o", ms=3.2, lw=1.6, color=col,
                label=f"Actual cases {YEAR}")
        ax.fill_between(c.week_start, 0, c.cases, color=col, alpha=.10)
        for _, r in b[b.province == prov].iterrows():
            ax.axvline(r.week_start, color=OLD_GRAY, ls=":", lw=1.2, alpha=.8)
            ax.scatter([r.week_start], [r.cases], marker="*", s=230, color=col,
                       edgecolor="k", linewidth=.7, zorder=6, label="Blind week (actual)")
            ax.annotate(f"{r.cases:.0f}", (r.week_start, r.cases),
                        textcoords="offset points", xytext=(9, 11), fontsize=9)
        ax.set_title(f"#{s['rank']} {prov} - blind test {LABEL} | "
                     f"MAE = |error| = {s['MAE']:.2f} cases\n"
                     f"actual={s['actual']:.0f} | {YEAR} mean={s['mean_cases']:.2f}/wk | "
                     f"peak={s['max_cases']:.0f} | n={s['n_weeks']} weeks (actual only)", fontsize=12)
        ax.set_xlabel("Week start"); ax.set_ylabel("Actual cases / week")
        date_axis(ax, s["max_cases"]); dedup_legend(ax)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(OUT / f"{s['rank']:02d}_{safe(prov)}.png", bbox_inches="tight"); plt.close(fig)
        print(f"Saved {s['rank']:02d}_{safe(prov)}.png  (MAE={s['MAE']:.2f})")
pd.DataFrame(summary).to_csv(OUT / f"{MODE}{len(summary)}_blind_summary.csv",
                             index=False, encoding="utf-8-sig")
# ============================================================
# PLOT C: BAR CHART (kept - direction follows MODE)
# ============================================================
if MAKE_BAR:
    t = pd.DataFrame(summary).sort_values(RANK_METRIC, ascending=ASC)
    vals = t[RANK_METRIC].to_numpy(float)
    xmax = max(float(np.nanmax(vals)), 1.0) * 1.15
    lab = ("Absolute error (cases) - blind week" if RANK_METRIC == "MAE"
           else "Absolute error (% of actual) - blind week")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(t.province, vals, color=OLD_RED, alpha=.85)
    for i, v in enumerate(vals):
        ax.text(v + xmax * .01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel(lab); ax.set_ylabel("Province")
    ax.set_title(f"Top {len(t)} {WORD} blind-test MAE provinces - {LABEL}\n"
                 f"national MAE={nat_mae:.3f} | WAPE={nat_wape:.3f}", fontsize=12)
    ax.set_xlim(0, xmax)
    fig.tight_layout()
    fig.savefig(OUT / f"00_top{len(t)}_error_bar.png", bbox_inches="tight"); plt.close(fig)
    print(f"Saved 00_top{len(t)}_error_bar.png")
# ============================================================
# FINAL
# ============================================================
print("\n" + "=" * 60)
print(f"{MODE.upper()} {len(summary)} BLIND-TEST MAE PROVINCES - DONE")
print("=" * 60)
print(f"Blind week : {LABEL}")
print(f"Selected   : {WORD} {RANK_METRIC}")
print(f"Curves     : {YEAR} actual cases, linear, X = week / Y = cases")
print(f"History    : {HIST}")
print(f"Outputs -> {OUT}")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}")
'''

# ============================================================
# 4b) blind_compare_hist.py
# ============================================================
SOURCES["blind_compare_hist"] = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blind_compare_hist.py — blind week 2026-10-25: ACTUAL vs PREDICTED (Thai labels).
Two SEPARATE figures, 5 provinces each, vertical grouped bars:
1) blind_top5_gap.png    -> the 5 LARGEST  |actual - pred|
2) blind_least5_gap.png  -> the 5 SMALLEST |actual - pred|
Also writes blind_comparison_77.csv (full ranking).
Predictions come from predict_week/predictions.csv if present, otherwise
from DenguYosoku/model.txt (--recompute forces the model path).
RUN:
python blind_compare_hist.py
python blind_compare_hist.py --dpi 600
python blind_compare_hist.py --rank pct --min-actual 5
python blind_compare_hist.py --top 10
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from matplotlib import font_manager
from matplotlib.ticker import StrMethodFormatter
# ORDER MATTERS. DenguYosoku.py trains on numpy arrays, so model.txt only knows
# Column_0..Column_11. Prefer features.json / metrics.json; this is the fallback.
FEATURES_FALLBACK = [
    "cases_L1", "cases_L2", "cases_L3", "cases_L4", "cases_L8", "cases_L12",
    "outbreak_momentum", "cases_L1_relative", "cases_L2_relative",
    "sin_week", "cos_week", "w_apparent_temperature_mean_L1",
]
THAI_FONTS = ["Noto Sans Thai", "Noto Sans Thai Looped", "Noto Serif Thai", "Sarabun",
              "IBM Plex Sans Thai", "Loma", "Garuda", "Norasi", "Waree",
              "Tlwg Typist", "Tlwg Typo", "TH Sarabun New", "TH SarabunPSK",
              "Angsana New", "Leelawadee UI", "Thonburi"]
def first_existing(paths):
    return next((Path(p) for p in paths if Path(p).exists()), None)
def clean(d):
    d.columns = d.columns.str.strip()
    for c in d.columns:
        if not is_numeric_dtype(d[c]):
            d[c] = d[c].astype(str).str.strip()
    return d
ap = argparse.ArgumentParser(description="Blind-week actual vs predicted, top/least 5 gaps")
ap.add_argument("--answer", default=None)
ap.add_argument("--features", default=None)
ap.add_argument("--preds", default=None)
ap.add_argument("--model", default="DenguYosoku/model.txt")
ap.add_argument("--features-json", default=None)
ap.add_argument("--recompute", action="store_true")
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--rank", choices=["abs", "pct", "signed"], default="abs")
ap.add_argument("--min-actual", type=int, default=0)
ap.add_argument("--dpi", type=int, default=300)
ap.add_argument("--outdir", default="blind_plots")
A = ap.parse_args()
OUT = Path(A.outdir); OUT.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------- locate files
ans_p = Path(A.answer) if A.answer else first_existing(
    ["predict_week/answer_key.csv", "answer_key.csv"])
fea_p = Path(A.features) if A.features else first_existing(
    ["predict_week/predict_2026-10-25.csv", "predict_2026-10-25.csv"])
prd_p = Path(A.preds) if A.preds else first_existing(
    ["predict_week/predictions.csv", "predictions.csv"])
if ans_p is None or fea_p is None:
    sys.exit("need answer_key.csv AND predict_2026-10-25.csv (run make_blind_holdout.py)")
ans = clean(pd.read_csv(ans_p, encoding="utf-8-sig"))
fea = clean(pd.read_csv(fea_p, encoding="utf-8-sig"))
assert "cases" not in fea.columns, "feature file must NOT contain the answer"
for d in (ans, fea):
    d["week_start"] = pd.to_datetime(d["week_start"])
WEEK = ans["week_start"].iloc[0]
assert (ans["week_start"] == WEEK).all() and (fea["week_start"] == WEEK).all(), "mixed weeks"
print(f"blind week : {WEEK.date()}   answer rows={len(ans)}   feature rows={len(fea)}")
# ---------------------------------------------------------------- predictions
if prd_p is not None and not A.recompute:
    prd = clean(pd.read_csv(prd_p, encoding="utf-8-sig"))
    prd["week_start"] = pd.to_datetime(prd["week_start"])
    prd = prd.loc[prd["week_start"] == WEEK, ["province", "pred"]]
    src = f"{prd_p} (pre-computed)"
else:
    import lightgbm as lgb
    mp = Path(A.model)
    if not mp.exists():
        sys.exit(f"{mp} not found — run predict_2026-10-25.py first, or pass --preds")
    fj = Path(A.features_json) if A.features_json else first_existing(
        ["predict_week/features.json", "features.json", "DenguYosoku/metrics.json"])
    if fj is not None:
        blob = json.loads(fj.read_text(encoding="utf-8"))
        F = blob["features"] if isinstance(blob, dict) else blob
        fsrc = fj.name
    else:
        F, fsrc = FEATURES_FALLBACK, "hardcoded fallback"
    booster = lgb.Booster(model_file=str(mp))
    assert booster.num_feature() == len(F), \
        f"model wants {booster.num_feature()} features, list has {len(F)}"
    missing = [c for c in F if c not in fea.columns]
    assert not missing, f"feature columns missing from CSV: {missing}"
    fea["pred"] = np.clip(booster.predict(fea[F].to_numpy(np.float32)), 0, None)
    prd = fea[["province", "pred"]]
    src = f"{mp} (recomputed, feature order from {fsrc})"
print(f"predictions: {src}")
# ---------------------------------------------------------------- merge + rank
m = ans.merge(prd, on="province", how="inner", validate="one_to_one")
assert len(m) == len(ans) == len(fea), "province mismatch between answer / features / preds"
m["actual"] = m["cases"].astype(float)
m["pred"] = m["pred"].astype(float)
m["error"] = m["pred"] - m["actual"]                            # >0 = OVER-prediction
m["abs_error"] = m["error"].abs()
m["pct_error"] = 100.0 * m["abs_error"] / (m["actual"] + 1.0)   # +1 smoothed, no /0
m["rank_value"] = {"abs": m["abs_error"], "pct": m["pct_error"],
                   "signed": m["error"]}[A.rank]
m["_ord"] = np.arange(len(m))
nat_mae = float(m["abs_error"].mean())
print(f"national   : actual={m['actual'].sum():.0f}  pred={m['pred'].sum():.0f}  "
      f"blind MAE={nat_mae:.3f}  bias={m['error'].mean():+.3f}")
pool = m.loc[m["actual"] >= A.min_actual].copy()
if len(pool) < 2 * A.top:
    sys.exit(f"only {len(pool)} provinces pass --min-actual {A.min_actual}; need {2*A.top}")
pool = pool.sort_values(["rank_value", "actual", "_ord"], ascending=[False, False, True])
top = pool.head(A.top).reset_index(drop=True)
least = pool.tail(A.top).sort_values(
    ["rank_value", "actual", "_ord"], ascending=[True, False, True]).reset_index(drop=True)
m.drop(columns=["_ord", "rank_value"]).sort_values("abs_error", ascending=False).to_csv(
    OUT / "blind_comparison_77.csv", index=False, encoding="utf-8-sig")
# ---------------------------------------------------------------- font
installed = {f.name for f in font_manager.fontManager.ttflist}
thai_name = next((f for f in THAI_FONTS if f in installed), None)
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.unicode_minus": False,
})
THAI = None
if thai_name:
    _p = font_manager.findfont(font_manager.FontProperties(family=thai_name))
    THAI = font_manager.FontProperties(fname=_p, size=11)
    print(f"fonts      : DejaVu Sans + {thai_name}")
# ---------------------------------------------------------------- plot builder
def draw(tab, fname, headline):
    n = len(tab)
    x = np.arange(n)
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(9.5, 2.0 * n), 6.4))
    b1 = ax.bar(x - w / 2, tab["actual"], w, color="#1f77b4", alpha=.92,
                edgecolor="white", linewidth=.6, label="actual (answer key)")
    b2 = ax.bar(x + w / 2, tab["pred"], w, color="#d62728", alpha=.92,
                edgecolor="white", linewidth=.6, label="predicted (model)")
    ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=9)
    peak = float(np.maximum(tab["actual"], tab["pred"]).max())
    ax.set_ylim(0, max(peak * 1.30, 1.0))
    for i, r in tab.iterrows():
        hi = max(r["actual"], r["pred"])
        ax.text(x[i], hi + peak * 0.10,
                f"gap = {r['error']:+.1f}\n({r['pct_error']:.0f}% of actual+1)",
                ha="center", va="bottom", fontsize=9,
                color="#b22222" if r["error"] > 0 else "#1f5fb2")
    ax.set_xticks(x)
    if THAI is not None:
        ax.set_xticklabels(tab["province"].tolist(), fontproperties=THAI)
    else:
        ax.set_xticklabels(tab["province"].tolist(), fontsize=11)
    ax.set_ylabel("dengue cases in the week", fontsize=11)
    ax.set_xlabel("province", fontsize=11)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", ls=":", alpha=.30)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", fontsize=10, framealpha=.95)
    info = (f"shown = {n} of {len(m)} provinces\n"
            f"sum actual = {tab['actual'].sum():,.0f}\n"
            f"sum pred = {tab['pred'].sum():,.1f}\n"
            f"MAE of these {n} = {tab['abs_error'].mean():.2f}\n"
            f"MAE all {len(m)} = {nat_mae:.2f}")
    ax.text(0.975, 0.74, info, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bbbbbb", alpha=.93))
    ax.set_title(f"Blind week {WEEK.date()} - {headline}\n"
                 "ranked by " + A.rank + " error; red gap = over-prediction, "
                 "blue gap = under-prediction", fontsize=12, pad=12)
    fig.tight_layout()
    p = OUT / fname
    fig.savefig(p, dpi=A.dpi, facecolor="white")
    print(f"Saved {p}  ({int(fig.get_figwidth()*A.dpi)} x "
          f"{int(fig.get_figheight()*A.dpi)} px @ {A.dpi} dpi)")
    plt.close(fig)
draw(top, f"blind_top{A.top}_gap.png", f"TOP {A.top} LARGEST gaps")
draw(least, f"blind_least{A.top}_gap.png", f"LEAST {A.top} gaps")
cols = ["province", "actual", "pred", "error", "abs_error", "pct_error"]
print(f"\nTOP {A.top} (largest {A.rank} error):")
print(top[cols].round(2).to_string(index=False))
print(f"\nLEAST {A.top} (smallest {A.rank} error):")
print(least[cols].round(2).to_string(index=False))
print(f"\nSaved {OUT/'blind_comparison_77.csv'} (full 77-province ranking)")
'''

# ============================================================
# 5) raw_cases_hist_swapped.py
# ============================================================
SOURCES["raw_cases_hist_swapped"] = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raw_cases_hist_vertical.py — RAW cases histogram, VERTICAL bars.
x-axis = cases per province-week        (the variable)
y-axis = number of province-weeks       (the frequency)
Text rendering: single font (DejaVu Sans), NO monospace, NO space-padded
alignment, ASCII-only labels -> no random-looking gaps in legend / stats.
Standalone. Reads only the `cases` column. Touches nothing from
skew_report.py or DenguYosoku.py.
RUN:
python raw_cases_hist_vertical.py
python raw_cases_hist_vertical.py dengue_full.csv --dpi 600 --vector
python raw_cases_hist_vertical.py --bins 80 --clip 99
python raw_cases_hist_vertical.py --full-range --bin-width 25
OUT:
DenguYosoku_plots/raw_cases_hist_vertical.png   (+ .pdf with --vector)
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, MultipleLocator
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",      # one font for every text object
    "font.size": 10,
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
})
TARGET = "cases"
FIGW, FIGH = 11.0, 7.0
CANDIDATES = [
    "dengue_cut_allweeks_2003-18_Oct_2026.csv",
    "dengue_cut_2003-18_Oct_2026.csv",
    "dengue_cut_allweeks.csv",
    "dengue_full.csv",
    "/home/joseph/Documents/Donki Dengue/Real/dengue_full.csv",
]
ap = argparse.ArgumentParser(description="Vertical raw-cases histogram")
ap.add_argument("csv", nargs="?", default=None)
ap.add_argument("--dpi", type=int, default=300)
ap.add_argument("--bins", type=int, default=120, help="target bin count")
ap.add_argument("--bin-width", type=float, default=None, help="exact bin width in cases")
ap.add_argument("--clip", type=float, default=99.5, help="upper percentile shown")
ap.add_argument("--full-range", action="store_true")
ap.add_argument("--vector", action="store_true", help="also write a PDF")
ap.add_argument("--outdir", default="DenguYosoku_plots")
A = ap.parse_args()
# ---------------------------------------------------------------- locate file
if A.csv:
    SRC = Path(A.csv)
    if not SRC.exists():
        sys.exit(f"not found: {SRC}")
else:
    SRC = next((Path(c) for c in CANDIDATES if Path(c).exists()), None)
    if SRC is None:
        sys.exit("no CSV found — pass the path as an argument")
OUT = Path(A.outdir); OUT.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------- load counts
head = pd.read_csv(SRC, encoding="utf-8-sig", nrows=0).columns
if TARGET not in head:
    sys.exit(f"'{TARGET}' column missing from {SRC.name}")
y = pd.read_csv(SRC, encoding="utf-8-sig", usecols=[TARGET])[TARGET]
y = pd.to_numeric(y, errors="coerce").dropna().to_numpy(float)
if y.size == 0:
    sys.exit("no case values")
mu, md = float(y.mean()), float(np.median(y))
sk = float(pd.Series(y).skew())              # adjusted Fisher–Pearson
zero_pct = 100.0 * float((y == 0).mean())
pclip = float(np.percentile(y, A.clip))
# ---------------------------------------------------------------- bin edges
# Bin across the DISPLAYED range (not 0..max), snapped to an integer width so
# integer counts do not alias against fractional edges.
hi = float(y.max()) if A.full_range else pclip
w = float(A.bin_width) if A.bin_width else max(1.0, float(round(hi / max(1, A.bins))))
top = w * float(np.ceil(hi / w))
bin_edges = np.arange(0.0, top + w, w)
n_over = int((y > top).sum())
# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(FIGW, FIGH))
counts, _e, _p = ax.hist(bin_edges[:-1].size and y, bins=bin_edges,
                         color="#d62728", alpha=.88,
                         edgecolor="white", linewidth=.3)
# variable on x -> mean / median markers are VERTICAL lines
ax.axvline(mu, color="k", ls="--", lw=1.9, label=f"mean = {mu:.2f}")
ax.axvline(md, color="#1f77b4", ls="-", lw=1.9, label=f"median = {md:.0f}")
ax.set_xlabel("cases / province-week", fontsize=11)
ax.set_ylabel("frequency (number of province-weeks)", fontsize=11)
ax.set_xlim(0, top)
ax.set_ylim(0, counts.max() * 1.14)          # headroom for legend + stats
ax.xaxis.set_major_locator(MultipleLocator(max(w, round(top / 12 / w) * w)))
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.grid(axis="y", ls=":", alpha=.30)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper right", fontsize=10, framealpha=.95,
          borderpad=0.8, labelspacing=0.6)
# one label per line, single spaces, no padding, no monospace
lines = [
    f"n = {y.size:,} province-weeks",
    f"skew = {sk:+.2f}",
    f"mean = {mu:.2f}",
    f"median = {md:.0f}",
    f"mean / median = {mu / max(md, 1e-9):.2f}x",
    f"zero-case weeks = {zero_pct:.1f}%",
]
if n_over:
    lines.append(f"above {top:.0f} cases = {n_over:,} ({100*n_over/y.size:.2f}%)")
ax.text(0.975, 0.78, "\n".join(lines), transform=ax.transAxes,
        ha="right", va="top", fontsize=9.5, linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bbbbbb", alpha=.93))
ax.set_title("RAW cases - vertical histogram\n"
             f"skew = {sk:+.2f}; mean {mu:.2f} > median {md:.0f}: "
             "a Poisson (mean) fit over-predicts quiet weeks",
             fontsize=12, pad=12)
fig.tight_layout()
png = OUT / "raw_cases_hist_vertical.png"
fig.savefig(png, dpi=A.dpi, facecolor="white")
if A.vector:
    fig.savefig(OUT / "raw_cases_hist_vertical.pdf", facecolor="white")
plt.close(fig)
# ---------------------------------------------------------------- console
print(f"{SRC.name}: {y.size:,} rows")
print(f"raw skew={sk:+.3f}  mean={mu:.2f}  median={md:.0f}  zeros={zero_pct:.1f}%")
print(f"cases axis: 0 .. {top:.0f}  (p{A.clip} = {pclip:.1f}, max = {y.max():.0f})")
print(f"bin width = {w:g} cases, {len(bin_edges)-1} bins, {n_over:,} rows off-axis")
print(f"Saved {png}  ({int(FIGW*A.dpi)} x {int(FIGH*A.dpi)} px @ {A.dpi} dpi)")
if A.vector:
    print(f"Saved {OUT / 'raw_cases_hist_vertical.pdf'}  (vector = resolution-free)")
'''

# ============================================================
# 6) skew_report.py
# ============================================================
SOURCES["skew_report"] = r'''
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
'''

# ============================================================
# 7) skew_split.py
# ============================================================
SOURCES["skew_split"] = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skew_split.py — TRAIN vs TEST skewness, uncramped.

What changed:
  * TRAIN and TEST get SEPARATE panels with SEPARATE x-scales. Sharing one axis
    squeezed 94 test weeks into 7% of the width - that was the cramping.
  * Stats go in the panel title, not in boxes drawn over the data.
  * Heavy tail shown as log1p histogram + log-log CCDF, not 70 linear bins.
  * Province skew is horizontal and paged, so Thai names read normally.

RUN : python skew_split.py [path/to/dengue.csv]
OUT : DenguYosoku_plots/skew_01_split.png
      DenguYosoku_plots/skew_02_tail.png
      DenguYosoku_plots/skew_03_province_p*.png
      DenguYosoku_plots/skew_split_summary.csv
"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.font_manager as fm
import seaborn as sns

# ---- fonts: identical to plot_DenguYosoku.py (LIST + DejaVu per-glyph fallback)
sns.set_theme(style="whitegrid")
avail = {f.name for f in fm.fontManager.ttflist}
thai = [f for f in ["Noto Sans Thai","Noto Serif Thai","Sarabun","TH Sarabun New",
                    "Loma","Garuda"] if f in avail] or ["DejaVu Sans"]
plt.rcParams.update({"figure.dpi":130, "savefig.dpi":300, "font.size":10,
                     "font.family":thai+["DejaVu Sans"], "axes.unicode_minus":False,
                     "axes.grid":True, "grid.alpha":0.22,
                     "axes.formatter.use_mathtext":False,
                     "mathtext.fontset":"custom",
                     "mathtext.rm":"DejaVu Sans",
                     "mathtext.it":"DejaVu Sans:italic",
                     "mathtext.bf":"DejaVu Sans:bold"})
print(f"font: {thai[0]} + DejaVu Sans fallback")

CANDS = ["dengue_cut_allweeks_2003-18_Oct_2026.csv",
         "dengue_full_2003-18_Oct_2026.csv", "dengue_full.csv"]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else next(
    (Path(c) for c in CANDS if Path(c).exists()), None)
if SRC is None: sys.exit("no CSV found - pass the path as an argument")
OUT = Path("DenguYosoku_plots"); OUT.mkdir(exist_ok=True)

TRAIN_END, EVAL_CUTOFF = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-10-25")
BLUE, ORANGE, GREY = "#1f77b4", "#ff7f0e", "#9aa0a6"

df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
df["week_start"] = pd.to_datetime(df["week_start"])
df["province"] = df["province"].astype(str).str.strip()
ORDER = list(dict.fromkeys(df["province"]))
df = df[df.week_start < EVAL_CUTOFF].reset_index(drop=True)
df["split"] = np.where(df.week_start < TRAIN_END, "TRAIN", "TEST")

nat = (df.groupby("week_start", as_index=False)["cases"].sum()
         .sort_values("week_start").reset_index(drop=True))
NTR = nat[nat.week_start <  TRAIN_END].reset_index(drop=True)
NTE = nat[nat.week_start >= TRAIN_END].reset_index(drop=True)
PTR, PTE = df[df.split=="TRAIN"], df[df.split=="TEST"]

def st(a):
    a = np.asarray(a, float)
    return dict(n=len(a), mean=a.mean(), median=float(np.median(a)),
                skew=float(pd.Series(a).skew()), p95=float(np.percentile(a,95)),
                p99=float(np.percentile(a,99)), max=a.max(),
                pct_zero=float((a==0).mean()))
S = {"national_TRAIN":st(NTR.cases), "national_TEST":st(NTE.cases),
     "provweek_TRAIN":st(PTR.cases), "provweek_TEST":st(PTE.cases)}

# ============================================================
# FIG 1 — one row per split, own x-scale, distribution beside it
# ============================================================
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2, width_ratios=[4.4, 1], hspace=0.34, wspace=0.05)

for r, (seg, col, lab, key) in enumerate([
        (NTR, BLUE, "TRAIN", "national_TRAIN"), (NTE, ORANGE, "TEST (sealed)", "national_TEST")]):
    s = S[key]
    ax = fig.add_subplot(gs[r, 0])
    hx = fig.add_subplot(gs[r, 1], sharey=ax)
    x, y = seg.week_start, seg.cases.to_numpy(float)
    is_test = lab.startswith("TEST")
    roll = pd.Series(y).rolling(4, min_periods=1).mean()

    if r == 0:
        # OVERVIEW: whole history, TRAIN blue + TEST orange on one axis
        ax.axvspan(NTE.week_start.min(), NTE.week_start.max(),
                   color=ORANGE, alpha=.08, zorder=0)
        ax.plot(NTR.week_start, NTR.cases, color=BLUE, lw=1.6, zorder=4,
                label="TRAIN weekly actual")
        ax.plot(NTE.week_start, NTE.cases, color=ORANGE, lw=1.9, zorder=5,
                label="TEST weekly actual")
        ax.plot(x, roll, color="#4d4d4d", lw=1.1, alpha=.40, ls="--", zorder=2,
                label="TRAIN 4-week rolling mean")
        ax.axvline(TRAIN_END, color="k", ls="--", lw=1.4)
    else:
        # DETAIL: TEST zoom + last 12 TRAIN weeks as grey context
        ctx = NTR.tail(12)
        ax.plot(ctx.week_start, ctx.cases, color=GREY, lw=1.4, alpha=.8,
                label="last 12 TRAIN weeks (context)")
        ax.axvline(TRAIN_END, color="k", ls="--", lw=1.4)
        ax.plot(x, y, color=col, lw=2.0, zorder=4, marker="o", ms=3.4,
                label="TEST weekly actual")
        ax.plot(x, roll, color="#4d4d4d", lw=1.1, alpha=.40, ls="--", zorder=2,
                label="4-week rolling mean")

    # mean/median drawn only over their own period, not across the whole axis
    x0, x1 = x.min(), x.max()
    ax.hlines(s["median"], x0, x1, color="#2ca02c", ls=":", lw=1.3, alpha=.85,
              zorder=3, label=f"{lab.split()[0]} median {s['median']:,.0f}")
    ax.hlines(s["mean"], x0, x1, color="#d62728", ls="--", lw=1.3, alpha=.85,
              zorder=3, label=f"{lab.split()[0]} mean {s['mean']:,.0f}")

    ax.set_ylim(bottom=0)
    ax.set_ylabel("National cases / week")
    head = "FULL HISTORY (TRAIN + TEST)" if r == 0 else "TEST detail (zoom)"
    ax.set_title(f"{head}   |   {lab} {x.min().date()} to {x.max().date()}   "
                 f"{s['n']} weeks   mean-median = {s['mean']-s['median']:+,.0f}   "
                 f"skew = {s['skew']:+.2f}", fontsize=11.5, pad=8)
    ax.legend(loc="upper right" if is_test else "upper left",
              fontsize=8.5, ncol=2, framealpha=.92)
    if r == 0:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlabel("Week start")

    hx.hist(y, bins=30, orientation="horizontal", color=col, alpha=.85)
    hx.axhline(s["median"], color="#2ca02c", ls=":", lw=1.8)
    hx.axhline(s["mean"], color="#d62728", ls="--", lw=1.8)
    hx.tick_params(labelleft=False)
    hx.set_xlabel("weeks", fontsize=9)
    hx.set_title(f"{lab.split()[0]} distribution", fontsize=10)

fig.suptitle("Dengue weekly cases - top: full history with TEST highlighted, "
             "bottom: TEST zoom\n"
             "green dotted = median, red dashed = mean; the gap is the right skew",
             fontsize=13.5, y=0.985)
fig.savefig(OUT/"skew_01_split.png", bbox_inches="tight"); plt.close(fig)
print("Saved skew_01_split.png")

# ============================================================
# FIG 2 — the target the model fits: log1p hist + log-log CCDF + box
# ============================================================
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
for seg, col, lab in [(PTR, BLUE, "TRAIN"), (PTE, ORANGE, "TEST")]:
    s = S[f"provweek_{lab}"]
    v = seg.cases.to_numpy(float)
    ax[0].hist(np.log1p(v), bins=60, density=True, histtype="step", lw=2.2, color=col,
               label=f"{lab}  skew(log1p)={st(np.log1p(v))['skew']:+.2f}")
    sv = np.sort(v); sv = sv[sv > 0]
    ax[1].step(sv, 1 - np.arange(len(sv))/len(sv), where="post", lw=2.0, color=col,
               label=f"{lab}  raw skew={s['skew']:+.2f}")
ax[0].set_xlabel("log1p(cases per province-week)"); ax[0].set_ylabel("density")
ax[0].set_title("Target distribution (log1p)", fontsize=12); ax[0].legend(fontsize=9)
from matplotlib.ticker import FuncFormatter
ax[1].set_xscale("log"); ax[1].set_yscale("log")
_plain = FuncFormatter(lambda v, pos: f"{v:g}")   # plain text, bypasses mathtext
ax[1].xaxis.set_major_formatter(_plain)
ax[1].yaxis.set_major_formatter(_plain)
ax[1].xaxis.set_minor_formatter(_plain)
ax[1].yaxis.set_minor_formatter(_plain)
ax[1].set_xlabel("cases per province-week (log)"); ax[1].set_ylabel("P(Y >= y)  (log)")
ax[1].set_title("Tail survival curve - straight-ish on log-log = heavy tail", fontsize=12)
ax[1].legend(fontsize=9)
bp = ax[2].boxplot([np.log1p(PTR.cases), np.log1p(PTE.cases)],
                   tick_labels=["TRAIN","TEST"], showfliers=False,
                   patch_artist=True, widths=.55)
for patch, col in zip(bp["boxes"], [BLUE, ORANGE]):
    patch.set_facecolor(col); patch.set_alpha(.6)
ax[2].set_ylabel("log1p(cases)")
ax[2].set_title(f"Spread shift: median {S['provweek_TRAIN']['median']:.0f} -> "
                f"{S['provweek_TEST']['median']:.0f} cases", fontsize=12)
fig.suptitle("Province-week counts - the distribution the model actually fits", fontsize=13.5)
fig.tight_layout(); fig.savefig(OUT/"skew_02_tail.png", bbox_inches="tight"); plt.close(fig)
print("Saved skew_02_tail.png")

# ============================================================
# FIG 3 — province skew, horizontal, paged (Thai names read normally)
# ============================================================
sk_tr = PTR.groupby("province")["cases"].skew().reindex(ORDER)
sk_te = PTE.groupby("province")["cases"].skew().reindex(ORDER)
P = pd.DataFrame({"train":sk_tr, "test":sk_te}).sort_values("train", ascending=False)
per = 26; pages = math.ceil(len(P)/per)
for pg in range(pages):
    sub = P.iloc[pg*per:(pg+1)*per]
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(11, 0.42*len(sub)+2.2))
    ax.barh(y-0.20, sub.train.values, 0.38, color=BLUE,   alpha=.9, label="TRAIN skew")
    ax.barh(y+0.20, sub.test.values,  0.38, color=ORANGE, alpha=.9, label="TEST skew")
    ax.axvline(1, color="#d62728", ls="--", lw=1.3, label="|skew| = 1")
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(y); ax.set_yticklabels(sub.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("skewness of weekly cases"); ax.grid(axis="y", alpha=0)
    ax.set_title(f"Per-province skewness, TRAIN vs TEST  (page {pg+1}/{pages}, "
                 "sorted by TRAIN skew)", fontsize=12.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/f"skew_03_province_p{pg+1}.png",
                                    bbox_inches="tight"); plt.close(fig)
    print(f"Saved skew_03_province_p{pg+1}.png")

# ============================================================
T = pd.DataFrame(S).T; T.index.name = "series"
print("\n" + T.round(3).to_string())
T.round(6).to_csv(OUT/"skew_split_summary.csv", encoding="utf-8-sig")
P.assign(delta=P.test-P.train).round(4).to_csv(
    OUT/"skew_by_province_split.csv", encoding="utf-8-sig")
print("\nSaved skew_split_summary.csv, skew_by_province_split.csv")
'''

# ============================================================
# RUNNER — stdlib only; never imports matplotlib/pandas itself
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Run all 7 embedded scripts, each in its own untouched subprocess.")
    ap.add_argument("scripts", nargs="*",
                    help="subset to run, e.g. skew_report skew_split (default: all 7, in order)")
    ap.add_argument("--list", action="store_true", help="list embedded scripts and exit")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="stop at the first failing script (default: keep going)")
    A = ap.parse_args()

    if A.list:
        for i, n in enumerate(ORDER, 1):
            print(f"{i}. {n}.py")
        return 0

    chosen = ORDER
    if A.scripts:
        look = {}
        for n in ORDER:
            look[n] = n
            look[n + ".py"] = n
        bad = [s for s in A.scripts if s not in look]
        if bad:
            ap.error(f"unknown script(s): {', '.join(bad)}  (use --list)")
        chosen = []
        for s in A.scripts:
            n = look[s]
            if n not in chosen:
                chosen.append(n)

    if Path.cwd().resolve() != HERE:
        print(f"NOTE: cwd = {Path.cwd()} ; scripts using relative paths behave exactly "
              f"as they would standalone from this folder.")

    results, failed = [], False
    try:
        for i, name in enumerate(chosen, 1):
            print("\n" + "=" * 72, flush=True)
            print(f"[{i}/{len(chosen)}] {name}.py", flush=True)
            print("=" * 72, flush=True)
            tmp, rc = None, 1
            t0 = time.time()
            try:
                # temp file lives NEXT TO this file so Path(__file__).resolve().parent
                # inside the child is identical to the original script's BASE.
                fd, tmp = tempfile.mkstemp(prefix=f"__plot_all__{name}__",
                                           suffix=".py", dir=str(HERE))
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(SOURCES[name])
                # same interpreter, same cwd, same env, same stdio, argv = [script] + ARGS
                rc = subprocess.run([sys.executable, tmp] + ARGS.get(name, [])).returncode
            finally:
                if tmp is not None:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            dt = time.time() - t0
            ok = (rc == 0)
            results.append((name, "OK" if ok else f"FAILED (exit {rc})", dt))
            if not ok:
                failed = True
                if A.stop_on_error:
                    print(f"--stop-on-error: stopping after {name}.py", flush=True)
                    break
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        return 130

    print("\n" + "=" * 72)
    print("plot_all.py — SUMMARY")
    print("=" * 72)
    for name, status, dt in results:
        print(f"  {status:<18} {name}.py   ({dt:.1f}s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
