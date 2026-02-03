import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from src.preprocess import PreprocessConfig, build_preprocess_pipeline
from src.cascade import CascadedChannelModel, CascadeConfig


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("tune_threshold")


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/cascade.yaml")
    parser.add_argument("--recall_target", type=float, default=0.80, help="Recall mínimo desejado para VERMELHO")
    parser.add_argument("--max_thresholds", type=int, default=200, help="Número de thresholds no grid")
    args = parser.parse_args()

    logger = setup_logging()
    cfg_all = load_yaml(args.config)

    data_path = cfg_all["data"]["path"]
    cutoff = pd.Timestamp(cfg_all["split"]["cutoff"])

    runtime = cfg_all.get("runtime", {})
    artifacts_dir = Path(runtime.get("artifacts_dir", "artifacts"))
    out_dir = artifacts_dir / "threshold"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data: %s", data_path)
    df = pd.read_parquet(data_path)
    df["registry_date"] = pd.to_datetime(df["registry_date"], errors="coerce")
    df = df.dropna(subset=["registry_date"]).sort_values("registry_date")

    train = df[df["registry_date"] < cutoff].copy()
    valid = df[df["registry_date"] >= cutoff].copy()

    X_train = train.drop(columns=["channel"])
    y_train = train["channel"].astype(str)

    X_valid = valid.drop(columns=["channel"])
    y_valid = valid["channel"].astype(str)

    # preprocess
    pp_cfg_dict = cfg_all.get("preprocess", {
        "date_col": "registry_date",
        "target_col": "channel",
        "positive_label": "VERMELHO",
    })
    pp_cfg = PreprocessConfig(**pp_cfg_dict)
    prep = build_preprocess_pipeline(pp_cfg)

    Xt_train = prep.fit_transform(X_train, y_train)
    Xt_valid = prep.transform(X_valid)

    # cascata (mas aqui o foco é só no stage A)
    cas_cfg_dict = cfg_all.get("cascade", {
        "oversample_red_gate": True,
        "red_gate_model": "logreg",
        "yg_model": "linearsvc",
        "random_state": 42,
    })
    cas_cfg = CascadeConfig(**cas_cfg_dict)
    cascade = CascadedChannelModel(cas_cfg).fit(Xt_train, y_train)

    # gate vermelho precisa ter predict_proba (LogReg)
    model_red = cascade.model_red_
    if not hasattr(model_red, "predict_proba"):
        raise RuntimeError("O red_gate_model precisa suportar predict_proba para tuning (use logreg).")

    proba = model_red.predict_proba(Xt_valid)[:, 1]
    y_bin = (y_valid == "VERMELHO").astype(int).to_numpy()

    # Grid de thresholds
    thresholds = np.linspace(0.0, 1.0, args.max_thresholds)

    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_bin == 1)).sum())
        fp = int(((pred == 1) & (y_bin == 0)).sum())
        fn = int(((pred == 0) & (y_bin == 1)).sum())
        tn = int(((pred == 0) & (y_bin == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0

        rows.append({
            "threshold": float(t),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": float(recall),
            "precision": float(precision),
            "pred_pos_rate": float((pred == 1).mean()),
        })

    tbl = pd.DataFrame(rows)

    # escolher threshold: recall >= alvo, minimizar FP; se empate, maior threshold
    feasible = tbl[tbl["recall"] >= args.recall_target].copy()
    if feasible.empty:
        # fallback: maximizar recall, e dentro disso minimizar FP
        best = tbl.sort_values(["recall", "fp", "threshold"], ascending=[False, True, False]).iloc[0]
        reason = f"fallback (nenhum threshold atingiu recall_target={args.recall_target})"
    else:
        best = feasible.sort_values(["fp", "threshold"], ascending=[True, False]).iloc[0]
        reason = f"recall>=target ({args.recall_target}) e menor FP"

    best_t = float(best["threshold"])
    logger.info("Recommended threshold=%.4f | recall=%.4f | precision=%.4f | fp=%d | fn=%d (%s)",
                best_t, best["recall"], best["precision"], int(best["fp"]), int(best["fn"]), reason)

    # salvar tabela
    tbl.to_csv(out_dir / "threshold_candidates.csv", index=False)

    # PR curve
    prec, rec, pr_th = precision_recall_curve(y_bin, proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (RED gate)")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=160)
    plt.close(fig)

    # salvar recomendado
    payload = {
        "recommended_threshold": best_t,
        "reason": reason,
        "metrics_at_threshold": {
            "recall": float(best["recall"]),
            "precision": float(best["precision"]),
            "fp": int(best["fp"]),
            "fn": int(best["fn"]),
            "tp": int(best["tp"]),
            "tn": int(best["tn"]),
        },
        "config_snapshot": {
            "preprocess": asdict(pp_cfg),
            "cascade": asdict(cas_cfg),
            "cutoff": str(cutoff.date()),
            "data_path": data_path,
        }
    }
    save_json(out_dir / "threshold_recommended.json", payload)

    logger.info("Saved: %s", str(out_dir / "threshold_candidates.csv"))
    logger.info("Saved: %s", str(out_dir / "pr_curve.png"))
    logger.info("Saved: %s", str(out_dir / "threshold_recommended.json"))


if __name__ == "__main__":
    main()
