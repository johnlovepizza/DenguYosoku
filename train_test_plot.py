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
