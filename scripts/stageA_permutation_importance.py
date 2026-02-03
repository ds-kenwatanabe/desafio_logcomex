from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


def recall_red(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, pos_label=1))


def _permute_column_dense(X: np.ndarray, j: int, rng: np.random.Generator) -> np.ndarray:
    Xp = X.copy()
    Xp[:, j] = rng.permutation(Xp[:, j])
    return Xp


def permutation_importance_manual(
    model,
    X,
    y_true: np.ndarray,
    *,
    metric_fn,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance manual (funciona com X dense OU sparse).
    Estratégia: para cada coluna j, permuta valores (em dense) e mede queda na métrica.
    Para sparse, converte apenas uma vez para dense.
    """
    rng = np.random.default_rng(random_state)

    # Base score
    y_base = model.predict(X)
    base = metric_fn(y_true, np.asarray(y_base))

    # Para evitar incompatibilidade do sklearn + CSR, convertemos para dense UMA vez.
    if not isinstance(X, np.ndarray):
        Xd = X.toarray()
    else:
        Xd = X

    n_features = Xd.shape[1]
    importances = np.zeros((n_repeats, n_features), dtype=float)

    for r in range(n_repeats):
        for j in range(n_features):
            Xp = _permute_column_dense(Xd, j, rng)
            y_pred = model.predict(Xp)
            score = metric_fn(y_true, np.asarray(y_pred))
            importances[r, j] = base - score  # queda na métrica (quanto maior, mais importante)

    mean_imp = importances.mean(axis=0)
    std_imp = importances.std(axis=0)

    out = pd.DataFrame(
        {
            "j": np.arange(n_features),  # índice real da coluna
            "importance_mean": mean_imp,
            "importance_std": std_imp,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    out.attrs["base_score"] = base
    return out



def main():
    ARTIFACTS = Path("artifacts")
    OUT_DIR = ARTIFACTS / "analysis"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prep = joblib.load(ARTIFACTS / "preprocess.joblib")
    cascade = joblib.load(ARTIFACTS / "model_cascade.joblib")
    model = cascade.model_red_

    df = pd.read_parquet("data/sample_data.parquet")
    df["registry_date"] = pd.to_datetime(df["registry_date"], errors="coerce")
    df = df.dropna(subset=["registry_date"]).sort_values("registry_date")

    cutoff = pd.Timestamp("2024-10-01")
    valid = df[df["registry_date"] >= cutoff].copy()

    X_valid = valid.drop(columns=["channel"])
    y_valid = valid["channel"].astype(str)

    Xv = prep.transform(X_valid)
    y_true = (y_valid == "VERMELHO").astype(int).to_numpy()

    df_imp = permutation_importance_manual(
        model=model,
        X=Xv,
        y_true=y_true,
        metric_fn=recall_red,
        n_repeats=5,
        random_state=42,
    )

    base_score = df_imp.attrs.get("base_score", None)

    # feature names (best-effort)
    try:
        feature_names = np.asarray(prep.get_feature_names_out(), dtype=object)
    except Exception:
        feature_names = np.asarray([f"f_{i}" for i in range(Xv.shape[1])], dtype=object)

    # mapeia pelo j real
    df_imp["feature"] = df_imp["j"].map(
        lambda k: feature_names[k] if 0 <= int(k) < len(feature_names) else f"f_{k}"
    )

    # organiza colunas finais
    df_imp = df_imp[["feature", "j", "importance_mean", "importance_std"]]

    df_imp.to_csv(OUT_DIR / "stageA_perm_importance.csv", index=False)
    print("[OK] Saved: artifacts/analysis/stageA_perm_importance.csv")
    if base_score is not None:
        print(f"[INFO] Base recall_red (stage A): {base_score:.4f}")



if __name__ == "__main__":
    main()
