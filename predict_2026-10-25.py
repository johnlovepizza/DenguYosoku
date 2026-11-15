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
