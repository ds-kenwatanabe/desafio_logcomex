import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# = loader train_cascade.py
def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    suf = p.suffix.lower()
    if suf in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("Para usar YAML, instale PyYAML: pip install pyyaml") from e
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if suf == ".json":
        import json
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config extension: {suf} (use .yaml/.yml ou .json)")


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(subset=[col])


def _save_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    """Lower bound do intervalo de Wilson para proporção (robusto contra n pequeno)."""
    if n <= 0:
        return 0.0
    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)) / denom
    return float(max(0.0, center - margin))


def compute_risk_table(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "channel",
    positive_label: str = "VERMELHO",
    min_n: int = 50,
) -> pd.DataFrame:
    """Tabela de risco de vermelho por categoria com bound de Wilson."""
    g = (
        df.groupby(group_col, dropna=False)[target_col]
        .agg(
            n="size",
            red=lambda s: int((s.astype(str) == positive_label).sum()),
        )
        .reset_index()
    )
    g["red_rate"] = g["red"] / g["n"]
    g["wilson_lb"] = [wilson_lower_bound(int(k), int(n)) for k, n in zip(g["red"], g["n"])]

    # filtro por volume mínimo
    g = g[g["n"] >= min_n].copy()
    # rank por bound (mais “confiável”)
    g = g.sort_values(["wilson_lb", "red_rate", "n"], ascending=False).reset_index(drop=True)
    return g


def plot_top_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path):
    plt.figure(figsize=(10, 5))
    plt.bar(df[x].astype(str), df[y].astype(float))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=30, ha="right")
    _save_plot(path)


def plot_monthly_shares(df: pd.DataFrame, date_col: str, target_col: str, path: Path):
    tmp = df.copy()
    tmp["ym"] = pd.to_datetime(tmp[date_col]).dt.to_period("M").astype(str)

    share = (
        tmp.groupby(["ym", target_col], dropna=False)
        .size()
        .reset_index(name="n")
    )
    total = share.groupby("ym")["n"].sum().reset_index(name="total")
    share = share.merge(total, on="ym", how="left")
    share["pct"] = share["n"] / share["total"]

    piv = share.pivot(index="ym", columns=target_col, values="pct").fillna(0.0)
    piv = piv.sort_index()

    plt.figure(figsize=(12, 5))
    for col in piv.columns:
        plt.plot(piv.index, piv[col].values, marker="o", label=str(col))
    plt.title("Distribuição mensal dos canais (share por mês)")
    plt.xlabel("Mês (YYYY-MM)")
    plt.ylabel("Proporção")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    _save_plot(path)

    return share.sort_values(["ym", target_col])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/cascade.yaml ou .json")
    parser.add_argument("--min_n_ncm", type=int, default=50, help="volume mínimo por NCM para ranking")
    parser.add_argument("--min_n_transport", type=int, default=50)
    parser.add_argument("--min_n_size", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = cfg.get("data", {}).get("path", "data/sample_data.parquet")
    date_col = cfg.get("preprocess", {}).get("date_col", "registry_date")
    target_col = cfg.get("preprocess", {}).get("target_col", "channel")
    positive_label = cfg.get("preprocess", {}).get("positive_label", "VERMELHO")

    out_dir = Path(cfg.get("runtime", {}).get("artifacts_dir", "artifacts")) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    df = _ensure_datetime(df, date_col)

    # Q1) Top 5 NCM com maior risco de canal vermelho
    ncm_col = "ncm_code"
    if ncm_col not in df.columns:
        raise ValueError(f"Coluna {ncm_col} não encontrada no dataset. Ajuste o nome no script.")

    q1 = compute_risk_table(
        df=df,
        group_col=ncm_col,
        target_col=target_col,
        positive_label=positive_label,
        min_n=args.min_n_ncm,
    ).head(5)

    q1.to_csv(out_dir / "q1_top5_ncm_red_risk.csv", index=False)
    plot_top_bar(
        q1.assign(ncm=q1[ncm_col].astype(str)),
        x=ncm_col,
        y="wilson_lb",
        title="Top 5 NCMs com maior risco (Wilson lower bound) de VERMELHO",
        path=out_dir / "q1_top5_ncm_red_risk.png",
    )

    # Q2) Sazonalidade na distribuição dos canais
    q2 = plot_monthly_shares(
        df=df,
        date_col=date_col,
        target_col=target_col,
        path=out_dir / "q2_channel_share_by_month.png",
    )
    q2.to_csv(out_dir / "q2_channel_share_by_month.csv", index=False)

    # Q3) Impacto do modo de transporte
    transport_col = "transport_mode_pt"
    if transport_col in df.columns:
        q3 = compute_risk_table(
            df=df,
            group_col=transport_col,
            target_col=target_col,
            positive_label=positive_label,
            min_n=args.min_n_transport,
        )
        q3.to_csv(out_dir / "q3_transport_mode_red_risk.csv", index=False)

        # plot top categorias
        plot_top_bar(
            q3,
            x=transport_col,
            y="red_rate",
            title="Risco de VERMELHO por modo de transporte (red_rate)",
            path=out_dir / "q3_transport_mode_red_risk.png",
        )
    else:
        # não quebra o script
        pd.DataFrame({"warning": [f"coluna ausente: {transport_col}"]}).to_csv(
            out_dir / "q3_transport_mode_red_risk.csv", index=False
        )

    # Q4) Influência do porte da importadora
    size_col = "consignee_company_size"
    if size_col in df.columns:
        q4 = compute_risk_table(
            df=df,
            group_col=size_col,
            target_col=target_col,
            positive_label=positive_label,
            min_n=args.min_n_size,
        )
        q4.to_csv(out_dir / "q4_company_size_red_risk.csv", index=False)
        plot_top_bar(
            q4,
            x=size_col,
            y="red_rate",
            title="Risco de VERMELHO por porte da empresa (red_rate)",
            path=out_dir / "q4_company_size_red_risk.png",
        )
    else:
        pd.DataFrame({"warning": [f"coluna ausente: {size_col}"]}).to_csv(
            out_dir / "q4_company_size_red_risk.csv", index=False
        )

    print(f"[OK] Outputs gerados em: {out_dir}")


if __name__ == "__main__":
    main()
