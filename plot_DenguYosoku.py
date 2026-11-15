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
                     "axes.grid":True,"grid.alpha":0.25})

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
if "naive" not in df.columns:
    df["naive"] = df["persist"] if "persist" in df.columns else df["seasonal_naive"]

# master province order (already น่าน-first from the CSV, but lock it)
PROVINCE_ORDER = list(dict.fromkeys(df["province"]))
print(f"Rows: {len(df)} | Provinces: {len(PROVINCE_ORDER)} | "
      f"{df['week_start'].min().date()} -> {df['week_start'].max().date()}")
print(f"First 3 provinces: {PROVINCE_ORDER[:3]}")

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
def wape(y,p): return float(np.sum(np.abs(y-p))/np.sum(np.abs(y)))
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
    nat["naive_sum"] = df.groupby("week_start")["naive"].sum().values
nat["residual"] = nat["actual"] - nat["predicted"]
nat["r4a"] = nat["actual"].rolling(4,min_periods=1).mean()
nat["r4p"] = nat["predicted"].rolling(4,min_periods=1).mean()

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(16,8.5),sharex=True,
                             gridspec_kw={"height_ratios":[3,1]})
ax1.plot(nat.week_start, nat.actual, "-o", ms=3, lw=1.8, color="#1f77b4", label="Actual")
ax1.plot(nat.week_start, nat.predicted, "-s", ms=3, lw=1.8, alpha=.85, color="#d62728", label="Predicted")
if has_naive:
    ax1.plot(nat.week_start, nat.naive_sum, "--", lw=1.0, color="#7f7f7f", alpha=.6, label="Seasonal naive")
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
    nm = mae(y, sg.naive.values) if has_naive else np.nan
    sub_rows.append({"group":g, "n_prov":sg.province.nunique(),
                     "MAE":mae(y,p), "WAPE":wape(y,p), "R2":r2(y,p),
                     "skill_%":(nm-mae(y,p))/nm*100 if nm and nm>0 else np.nan})
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
try:
    import plotly.express as px
    for pv, name, scale, lab in [
        (res_pv, "interactive_residual", "RdBu_r", "Actual − Predicted"),
        (lr_pv,  "interactive_logratio", "coolwarm", "log ratio")]:
        f = px.imshow(pv, labels=dict(x="Week",y="Province",color=lab),
                      color_continuous_scale=scale, aspect="auto",
                      title=f"DenguYosogu — {name.replace('_',' ').title()}")
        f.update_layout(height=max(900,18*len(PROVINCE_ORDER)), width=2400,
                        font=dict(family=thai[0], size=12))
        f.update_xaxes(tickangle=90, tickfont=dict(size=8))
        f.update_yaxes(tickfont=dict(size=10))
        f.write_html(OUT/f"{name}.html")
    print("Saved interactive HTML heatmaps.")
except ImportError:
    print("Plotly not installed -> skip interactive (pip install plotly)")

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
