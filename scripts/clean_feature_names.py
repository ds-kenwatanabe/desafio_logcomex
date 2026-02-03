import joblib
import pandas as pd
from pathlib import Path

ARTIFACTS = Path("artifacts")
OUT = ARTIFACTS / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

prep = joblib.load(ARTIFACTS / "preprocess.joblib")

# captura o último step da pipeline
final_step = prep.steps[-1][1]

if not hasattr(final_step, "get_feature_names_out"):
    raise RuntimeError(
        f"O step final ({type(final_step).__name__}) não expõe get_feature_names_out()"
    )

names = final_step.get_feature_names_out()

df = pd.DataFrame(
    {
        "feature_index": range(len(names)),
        "feature_name": names,
    }
)

df.to_csv(OUT / "feature_index_map.csv", index=False)
print("Saved artifacts/analysis/feature_index_map.csv")
