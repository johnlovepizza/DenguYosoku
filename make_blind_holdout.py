#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_blind_holdout.py — cut ONE week out as a standalone prediction file.

Why features are built BEFORE the cut:
  cases_L1 for week t is week t-1's cases. If you split first, the file has
  one row per province and no history -> every lag is NaN. So we compute on
  the full history, then slice the target week out.

Why this is not leakage:
  1. the answer is BLANKED before any lag is computed, so no feature can
     contain it (structurally impossible, not just asserted)
  2. lags read BACKWARD only -> the week's lags come from earlier weeks
  3. prov_baseline is frozen at TRAIN_END, exactly as the model was trained

RUN : python make_blind_holdout.py
OUT : predict_week/predict_2026-10-25.csv   (77 rows, 12 features, NO cases)
      predict_week/answer_key.csv           (private)
      predict_week/features.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

SRC       = Path("dengue_full.csv")        # <- set to the file the model was trained on
TARGET    = "cases"
TRAIN_END = pd.Timestamp("2025-01-01")     # MUST match DenguYosoku.py
CASE_LAGS = (1, 2, 3, 4, 8, 12)
WSRC      = "w_apparent_temperature_mean"
OUT       = Path("predict_week"); OUT.mkdir(exist_ok=True)

df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
df["week_start"] = pd.to_datetime(df["week_start"])
df["province"]   = df["province"].astype(str).str.strip()
df["year"]       = df["week_start"].dt.year          # <-- THE MISSING LINE
ORDER = list(dict.fromkeys(df["province"]))
df["_pord"] = df["province"].map({p: i for i, p in enumerate(ORDER)})
df = df.sort_values(["_pord", "week_start"]).reset_index(drop=True)

LAST = df["week_start"].max()
hold = df["week_start"] == LAST
print(f"prediction week : {LAST.date()}   provinces : {int(hold.sum())}")
print(f"TRAIN_END       : {TRAIN_END.date()}  ->  this week is "
      f"{'IN TRAINING (bad!)' if LAST < TRAIN_END else 'NOT in training (correct)'}")

# 1) answer aside, then BLANK it (float cast: pandas 3 refuses NaN into int64)
answer = df.loc[hold, ["province", "week_start", TARGET]].copy()
df[TARGET] = df[TARGET].astype("float64")
df.loc[hold, TARGET] = np.nan
df = df.drop(columns=[c for c in ("cases_per_100k",) if c in df.columns])

# 2+3) build EVERYTHING in a dict, then ONE concat -> no fragmentation warnings
g, new = df.groupby("_pord", sort=False), {}
for k in CASE_LAGS:
    new[f"cases_L{k}"] = g[TARGET].shift(k)
new[f"{WSRC}_L1"] = g[WSRC].shift(1)

tr   = df["week_start"] < TRAIN_END
base = df.loc[tr].groupby("province")[TARGET].mean()
new["prov_baseline"]      = df["province"].map(base).fillna(df.loc[tr, TARGET].mean())
new["cases_L1_relative"]  = new["cases_L1"] / (new["prov_baseline"] + 1.0)
new["cases_L2_relative"]  = new["cases_L2"] / (new["prov_baseline"] + 1.0)
new["outbreak_momentum"]  = new["cases_L1_relative"] / (new["cases_L2_relative"] + 0.1)
wf = 2*np.pi*df["week"].astype(float)/52.0
new["sin_week"] = np.sin(wf); new["cos_week"] = np.cos(wf)

df = df.drop(columns=[c for c in ("sin_week","cos_week","sin2_week") if c in df.columns])
df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

FEATURES = [f"cases_L{k}" for k in CASE_LAGS] + [
    "outbreak_momentum", "cases_L1_relative", "cases_L2_relative",
    "sin_week", "cos_week", f"{WSRC}_L1"]

# ---- THE check: is cases_L1 the PREVIOUS week, or the answer? ----
prev = df.loc[df["week_start"] == LAST - pd.Timedelta(weeks=1)].set_index("province")[TARGET]
b    = df.loc[hold].set_index("province")
assert np.allclose(b["cases_L1"].to_numpy(), prev.reindex(b.index).to_numpy()), \
    "LAG MISALIGNED — cases_L1 is not week t-1"
ans   = answer.set_index("province")[TARGET].reindex(b.index)
coinc = int((b["cases_L1"].to_numpy() == ans.to_numpy()).sum())
assert b[FEATURES].notna().all().all(), "prediction week has NaN features"
print(f"cases_L1 == previous week : PASS (all {len(b)})")
print(f"cases_L1 == the answer    : {coinc}/{len(b)}  (coincidence; 77 would mean leakage)")

# 4) write the standalone week — NO cases column
cols = ["province", "week_start", "week", "year", "prov_baseline"] + FEATURES
out  = df.loc[hold, cols].copy()
out["week_start"] = out["week_start"].dt.strftime("%Y-%m-%d")
out.to_csv(OUT/f"predict_{LAST.date()}.csv", index=False, encoding="utf-8-sig")
answer.to_csv(OUT/"answer_key.csv", index=False, encoding="utf-8-sig")
(OUT/"features.json").write_text(json.dumps(FEATURES, ensure_ascii=False, indent=2))

print(f"\nwrote {OUT}/predict_{LAST.date()}.csv   rows={len(out)}  cols={out.shape[1]}")
print("columns:", list(out.columns))
print("\nGIVE FRIEND : predict_2026-10-25.csv + features.json + DenguYosoku/model.txt")
print("KEEP PRIVATE: answer_key.csv")
