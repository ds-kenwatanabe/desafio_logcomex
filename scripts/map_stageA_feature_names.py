from pathlib import Path
import pandas as pd


ARTIFACTS = Path("artifacts")
ANALYSIS = ARTIFACTS / "analysis"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return path


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    p_map = _require(ANALYSIS / "feature_index_map.csv")
    p_coef = _require(ANALYSIS / "stageA_top_features.csv")
    p_perm = _require(ANALYSIS / "stageA_perm_importance.csv")

    df_map = pd.read_csv(p_map)
    if not {"feature_index", "feature_name"}.issubset(df_map.columns):
        raise ValueError("feature_index_map.csv must have columns: feature_index, feature_name")

    # build mapping dict
    idx2name = dict(zip(df_map["feature_index"].astype(int), df_map["feature_name"].astype(str)))

    # COEF
    df_coef = pd.read_csv(p_coef)
    if "feature" not in df_coef.columns:
        raise ValueError("stageA_top_features.csv must have column: feature")

    df_coef = df_coef.copy()
    df_coef["feature_index"] = df_coef["feature"].astype(str).str.replace("f_", "", regex=False).astype(int)
    df_coef["feature_name"] = df_coef["feature_index"].map(idx2name)
    df_coef.to_csv(ANALYSIS / "stageA_top_features_named.csv", index=False)

    # PERM
    df_perm = pd.read_csv(p_perm)

    # tolera nomes diferentes (feature ou feature_x)
    feat_col = None
    for c in ["feature", "feature_x"]:
        if c in df_perm.columns:
            feat_col = c
            break
    if feat_col is None:
        raise ValueError("stageA_perm_importance.csv must have a feature column (feature or feature_x)")

    df_perm = df_perm.copy()
    df_perm["feature_index"] = df_perm[feat_col].astype(str).str.replace("f_", "", regex=False).astype(int)
    df_perm["feature_name"] = df_perm["feature_index"].map(idx2name)
    df_perm.to_csv(ANALYSIS / "stageA_perm_importance_named.csv", index=False)

    # SUMMARY (coef + perm juntos)
    # mantém só colunas relevantes e faz join por feature_index
    keep_coef = [c for c in ["feature_index", "feature_name", "coef", "abs_coef", "direction"] if c in df_coef.columns]
    keep_perm = [c for c in ["feature_index", "importance_mean", "importance_std"] if c in df_perm.columns]

    df_sum = (
        df_coef[keep_coef]
        .merge(df_perm[keep_perm], on="feature_index", how="outer")
        .sort_values(["importance_mean", "abs_coef"], ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    df_sum.to_csv(ANALYSIS / "stageA_features_summary.csv", index=False)

    print("[OK] Saved:")
    print(" - artifacts/analysis/stageA_top_features_named.csv")
    print(" - artifacts/analysis/stageA_perm_importance_named.csv")
    print(" - artifacts/analysis/stageA_features_summary.csv")


if __name__ == "__main__":
    main()
