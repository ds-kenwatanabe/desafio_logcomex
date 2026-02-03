from __future__ import annotations

import sys
import platform
import argparse
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix

from src.preprocess import PreprocessConfig, build_preprocess_pipeline
from src.cascade import CascadedChannelModel, CascadeConfig
from src.common.io import ensure_dir, save_joblib, save_json, fingerprint_df


def project_root() -> Path:
    # scripts/train_cascade.py
    return Path(__file__).resolve().parents[1]


def relpath(p: Path, base: Path) -> str:
    """Return path relative to base if possible; otherwise return just the name."""
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except Exception:
        # fallback: sem leak absolute path
        return str(p.name)


# Config loader (JSON or YAML)
def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    suf = p.suffix.lower()
    if suf in {".yaml", ".yml"}:
        try:
            import yaml  # pip install pyyaml
        except Exception as e:
            raise RuntimeError(
                "Para usar YAML, instale PyYAML: pip install pyyaml "
                "(ou use config .json para evitar essa dependência)."
            ) from e
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if suf == ".json":
        import json
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config extension: {suf} (use .yaml/.yml ou .json)")


def _maybe_load_threshold_recommended(artifacts_dir: Path) -> float | None:
    """
    Se existir artifacts/threshold/threshold_recommended.json, retorna recommended_threshold.
    Caso contrário, None.
    """
    p = artifacts_dir / "threshold" / "threshold_recommended.json"
    if not p.exists():
        return None
    import json
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    thr = obj.get("recommended_threshold", None)
    if thr is None:
        return None
    return float(thr)


# Logging
def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_cascade")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(Path(log_dir) / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# Helpers: reporting & saving
def print_class_balance(y: pd.Series, title: str):
    vc = y.value_counts()
    pct = y.value_counts(normalize=True) * 100
    out = pd.concat([vc, pct.rename("pct")], axis=1)
    print(f"\n=== Class balance: {title} ===")
    print(out)


def pretty_confusion_matrix(y_true, y_pred, labels, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    df_cm = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )

    df_cmn = pd.DataFrame(
        np.round(cmn * 100, 2),
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )

    print(f"\n--- {title} | Confusion Matrix (counts) ---")
    print(df_cm)

    print(f"\n--- {title} | Confusion Matrix (row-normalized %) ---")
    print(df_cmn)

    return {
        "counts": cm,
        "row_norm": cmn,
        "labels": labels,
        "df_counts": df_cm,
        "df_row_norm": df_cmn,
    }


def save_confusion_plots_and_csv(
    out_dir: Path,
    prefix: str,
    labels,
    df_counts: pd.DataFrame,
    df_row_norm: pd.DataFrame,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    df_counts.to_csv(out_dir / f"{prefix}_cm_counts.csv")
    df_row_norm.to_csv(out_dir / f"{prefix}_cm_row_norm_pct.csv")

    # PNG - counts
    _plot_matrix(
        df_counts.to_numpy(),
        labels,
        title=f"{prefix} - CM (counts)",
        path=out_dir / f"{prefix}_cm_counts.png",
        fmt="%.0f",
    )
    # PNG - row norm
    _plot_matrix(
        df_row_norm.to_numpy(),
        labels,
        title=f"{prefix} - CM (row-normalized %)",
        path=out_dir / f"{prefix}_cm_row_norm_pct.png",
        fmt="%.2f",
    )


def _plot_matrix(mat: np.ndarray, labels, title: str, path: Path, fmt: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(mat, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, fmt % mat[i, j], ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=False,
        help="Path to config (.json or .yaml). If omitted, uses defaults.",
    )
    parser.add_argument(
        "--red_threshold",
        type=float,
        default=None,
        help="Override do threshold do canal vermelho (ex: 0.23).",
    )
    args = parser.parse_args()

    # Defaults
    BASE_DIR = project_root()
    default_path = str(BASE_DIR / "data" / "sample_data.parquet")
    default_cutoff = "2024-10-01"

    cfg_all: dict = {}
    if args.config:
        cfg_all = load_config(args.config)

    data_path = cfg_all.get("data", {}).get("path", default_path)
    data_path = str(data_path)
    data_path_p = Path(data_path)

    # Se vier caminho relativo no YAML/CLI, resolve relativo ao root do projeto
    if not data_path_p.is_absolute():
        data_path_p = (BASE_DIR / data_path_p).resolve()

    cutoff = pd.Timestamp(cfg_all.get("split", {}).get("cutoff", default_cutoff))

    runtime = cfg_all.get("runtime", {})
    artifacts_dir = Path(runtime.get("artifacts_dir", "artifacts"))
    plots_dir = Path(runtime.get("plots_dir", str(artifacts_dir / "plots")))
    logs_dir = Path(runtime.get("logs_dir", "logs"))

    # resolve to project root (avoid current working dir ambiguity)
    if not artifacts_dir.is_absolute():
        artifacts_dir = (BASE_DIR / artifacts_dir).resolve()
    if not plots_dir.is_absolute():
        plots_dir = (BASE_DIR / plots_dir).resolve()
    if not logs_dir.is_absolute():
        logs_dir = (BASE_DIR / logs_dir).resolve()

    logger = setup_logging(str(logs_dir))
    logger.info("Starting train_cascade")
    if args.config:
        logger.info("Config file: %s", relpath(Path(args.config), BASE_DIR))
    else:
        logger.info("Config file: (none; using defaults)")
    logger.info("Data path: %s", relpath(data_path_p, BASE_DIR))
    logger.info("Data exists: %s", data_path_p.exists())
    logger.info("Cutoff: %s", str(cutoff.date()))
    logger.info("Artifacts dir: %s", relpath(artifacts_dir, BASE_DIR))
    logger.info("Plots dir: %s", relpath(plots_dir, BASE_DIR))
    logger.info("Logs dir: %s", relpath(logs_dir, BASE_DIR))

    # Load data
    df = pd.read_parquet(data_path_p)
    df["registry_date"] = pd.to_datetime(df["registry_date"], errors="coerce")
    df = df.dropna(subset=["registry_date"]).sort_values("registry_date")

    train = df[df["registry_date"] < cutoff].copy()
    valid = df[df["registry_date"] >= cutoff].copy()

    X_train = train.drop(columns=["channel"])
    y_train = train["channel"].astype(str)

    X_valid = valid.drop(columns=["channel"])
    y_valid = valid["channel"].astype(str)

    print_class_balance(y_train, "TRAIN")
    print_class_balance(y_valid, "VALID")

    logger.info("Train rows=%d | Valid rows=%d", len(train), len(valid))
    logger.info("Train class balance: %s", y_train.value_counts().to_dict())
    logger.info("Valid class balance: %s", y_valid.value_counts().to_dict())

    # Preprocess
    pp_cfg_dict = cfg_all.get(
        "preprocess",
        {
            "date_col": "registry_date",
            "target_col": "channel",
            "positive_label": "VERMELHO",
        },
    )
    pp_cfg = PreprocessConfig(**pp_cfg_dict)
    prep = build_preprocess_pipeline(pp_cfg)

    Xt_train = prep.fit_transform(X_train, y_train)
    Xt_valid = prep.transform(X_valid)

    print("\nTrain matrix:", Xt_train.shape)
    print("Valid matrix:", Xt_valid.shape)
    logger.info("Train matrix: %s | Valid matrix: %s", Xt_train.shape, Xt_valid.shape)

    # Cascade model
    cas_cfg_dict = cfg_all.get(
        "cascade",
        {
            "oversample_red_gate": True,
            "red_gate_model": "logreg",
            "yg_model": "linearsvc",
            "random_state": 42,
            # red_gate_threshold pode vir do YAML
            # "red_gate_threshold": 0.5,
        },
    )
    cas_cfg = CascadeConfig(**cas_cfg_dict)

    # ---- Aplicar threshold do gate vermelho (prioridade):
    # 1) --red_threshold (CLI)
    # 2) artifacts/threshold/threshold_recommended.json
    # 3) valor do config (cas_cfg.red_gate_threshold) ou default 0.5
    thr_used = None
    thr_source = None

    if args.red_threshold is not None:
        thr_used = float(args.red_threshold)
        thr_source = "cli"
    else:
        thr_rec = _maybe_load_threshold_recommended(artifacts_dir)
        if thr_rec is not None:
            thr_used = float(thr_rec)
            thr_source = "threshold_recommended.json"

    if thr_used is not None:
        cas_cfg.red_gate_threshold = float(thr_used)
        logger.info(
            "Using red_gate_threshold=%.4f (source=%s)",
            cas_cfg.red_gate_threshold,
            thr_source,
        )
    else:
        logger.info(
            "Using red_gate_threshold=%.4f (source=config/default)",
            float(cas_cfg.red_gate_threshold),
        )

    cascade = CascadedChannelModel(cas_cfg)
    cascade.fit(Xt_train, y_train)

    # Stage evaluations
    print("\n===== STAGE A: RED vs NOT_RED =====")
    s1 = cascade.evaluate_stage_red_gate(Xt_valid, y_valid)
    print(f"Accuracy: {s1['accuracy']:.4f}")
    print(s1["report_text"])
    print("Confusion matrix [0=not_red, 1=red]:")
    print(s1["confusion_matrix"])
    logger.info("Stage A accuracy=%.4f", s1["accuracy"])

    print("\n===== STAGE B: YELLOW vs GREEN (excluding RED) =====")
    s2 = cascade.evaluate_stage_yg(Xt_valid, y_valid)
    print(f"Accuracy: {s2['accuracy']:.4f}")
    print(s2["report_text"])
    print("Confusion matrix [0=green, 1=yellow]:")
    print(s2["confusion_matrix"])
    logger.info("Stage B accuracy=%.4f", s2["accuracy"])

    # End-to-end evaluation
    print("\n===== END-TO-END (GREEN/YELLOW/RED + GRAY present) =====")
    e2e = cascade.evaluate_end_to_end(Xt_valid, y_valid)
    print(f"Accuracy: {e2e['accuracy']:.4f}")
    print(e2e["report_text"])
    print("Labels order:", e2e["labels"])
    logger.info("E2E accuracy=%.4f", e2e["accuracy"])

    # Pretty confusion matrix (legível)
    y_pred = cascade.predict(Xt_valid)
    labels = e2e["labels"]
    cm_pack = pretty_confusion_matrix(y_valid, y_pred, labels, "END-TO-END")

    # Save artifacts
    ensure_dir(artifacts_dir)
    ensure_dir(plots_dir)

    save_joblib(prep, artifacts_dir / "preprocess.joblib")
    save_joblib(cascade, artifacts_dir / "model_cascade.joblib")

    # salvar threshold usado explicitamente (auditoria)
    threshold_payload = {
        "red_gate_threshold_used": float(cas_cfg.red_gate_threshold),
        "source": thr_source or "config/default",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(threshold_payload, artifacts_dir / "threshold_used.json")

    # Save metrics/config/split/metadata
    metrics = {"stage_red": s1, "stage_yg": s2, "end_to_end": e2e}
    save_json(metrics, artifacts_dir / "metrics_valid.json")
    save_json(asdict(pp_cfg), artifacts_dir / "preprocess_config.json")
    save_json(asdict(cas_cfg), artifacts_dir / "cascade_config.json")
    
    split = {
        "cutoff": str(cutoff.date()),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "train_class_balance": y_train.value_counts().to_dict(),
        "valid_class_balance": y_valid.value_counts().to_dict(),
    }
    save_json(split, artifacts_dir / "split.json")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": relpath(data_path_p, BASE_DIR),
        "train_matrix_shape": list(Xt_train.shape),
        "valid_matrix_shape": list(Xt_valid.shape),
        # fingerprints (rastreio do dataset)
        "data_fingerprint": {
            "train": fingerprint_df(train),
            "valid": fingerprint_df(valid),
        },
        # versões/ambiente (auditoria)
        "env": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "libs": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
    }
    save_json(metadata, artifacts_dir / "metadata.json")

    # Save confusion artifacts (csv + png)
    # END-TO-END
    save_confusion_plots_and_csv(
        out_dir=plots_dir,
        prefix="end2end",
        labels=labels,
        df_counts=cm_pack["df_counts"],
        df_row_norm=cm_pack["df_row_norm"],
    )

    # STAGE A (threshold-aware)
    y_bin_true = (y_valid.astype(str) == "VERMELHO").astype(int).to_numpy()
    y_bin_pred = np.asarray(cascade.predict_red_gate(Xt_valid)["pred"])

    df_stageA_counts = pd.DataFrame(
        confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1]),
        index=["true_not_red(0)", "true_red(1)"],
        columns=["pred_not_red(0)", "pred_red(1)"],
    )
    df_stageA_row = pd.DataFrame(
        np.round(
            confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1], normalize="true")
            * 100,
            2,
        ),
        index=["true_not_red(0)", "true_red(1)"],
        columns=["pred_not_red(0)", "pred_red(1)"],
    )
    save_confusion_plots_and_csv(
        out_dir=plots_dir,
        prefix="stageA_red_gate",
        labels=["not_red(0)", "red(1)"],
        df_counts=df_stageA_counts,
        df_row_norm=df_stageA_row,
    )

    # STAGE B
    # Avalia só amostras com true {VERDE, AMARELO}
    idx = np.flatnonzero(y_valid.astype(str).isin(["VERDE", "AMARELO"]).to_numpy())
    if idx.size > 0:
        y_yg_true = (y_valid.iloc[idx].astype(str) == "AMARELO").astype(int).to_numpy()

        # Se stage B não foi treinado, predict será baseline (sempre 0) via evaluate_stage_yg,
        # para plots usamos o próprio modelo se existir.
        if getattr(cascade, "model_yg_", None) is None:
            y_yg_pred = np.zeros_like(y_yg_true)
        else:
            y_yg_pred = np.asarray(cascade.model_yg_.predict(Xt_valid[idx]))

        df_stageB_counts = pd.DataFrame(
            confusion_matrix(y_yg_true, y_yg_pred, labels=[0, 1]),
            index=["true_green(0)", "true_yellow(1)"],
            columns=["pred_green(0)", "pred_yellow(1)"],
        )
        df_stageB_row = pd.DataFrame(
            np.round(
                confusion_matrix(y_yg_true, y_yg_pred, labels=[0, 1], normalize="true")
                * 100,
                2,
            ),
            index=["true_green(0)", "true_yellow(1)"],
            columns=["pred_green(0)", "pred_yellow(1)"],
        )
        save_confusion_plots_and_csv(
            out_dir=plots_dir,
            prefix="stageB_yellow_vs_green",
            labels=["green(0)", "yellow(1)"],
            df_counts=df_stageB_counts,
            df_row_norm=df_stageB_row,
        )

    logger.info("Saved: %s", relpath(artifacts_dir / "preprocess.joblib", BASE_DIR))
    logger.info("Saved: %s", relpath(artifacts_dir / "model_cascade.joblib", BASE_DIR))
    logger.info("Saved: %s", relpath(artifacts_dir / "metrics_valid.json", BASE_DIR))
    logger.info("Saved plots/csv in: %s", relpath(plots_dir, BASE_DIR))

    print(f"\nSaved artifacts in: {relpath(artifacts_dir, BASE_DIR)}")
    print(f"Saved plots/csv in: {relpath(plots_dir, BASE_DIR)}")
    print(f"Saved logs in: {relpath(logs_dir, BASE_DIR)}")


if __name__ == "__main__":
    main()
