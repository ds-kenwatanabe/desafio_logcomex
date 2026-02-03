from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ARTIFACTS = Path("artifacts")
OUT_DIR = ARTIFACTS / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# load artifacts
prep = joblib.load(ARTIFACTS / "preprocess.joblib")
cascade = joblib.load(ARTIFACTS / "model_cascade.joblib")

model = cascade.model_red_

if not hasattr(model, "coef_"):
    raise RuntimeError("Stage A model does not expose coef_ (not a linear model).")

# feature names
try:
    feature_names = prep.get_feature_names_out()
except Exception:
    feature_names = np.array([f"f_{i}" for i in range(model.coef_.shape[1])])

coefs = model.coef_.ravel()

df = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs,
    "abs_coef": np.abs(coefs),
    "direction": np.where(coefs > 0, "↑ increases RED risk", "↓ decreases RED risk"),
})

df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

# save full + top N
df.to_csv(OUT_DIR / "stageA_all_features.csv", index=False)
df.head(30).to_csv(OUT_DIR / "stageA_top_features.csv", index=False)

print("[OK] Saved:")
print(" - artifacts/analysis/stageA_all_features.csv")
print(" - artifacts/analysis/stageA_top_features.csv")
