from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _to_period_month(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.to_period("M")


def _infer_positive_label(values: Iterable) -> Optional[str]:
    # tenta achar "vermelho"/"red"
    vals = [str(v).strip().lower() for v in pd.Series(list(values)).dropna().unique()]
    for candidate in ["vermelho", "red"]:
        if candidate in vals:
            return candidate
    return None


@dataclass
class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Cria features temporais e derivadas (sem depender de y).
    """
    date_col: str = "registry_date"
    ncm_col: str = "ncm_code"

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # data principal
        X[self.date_col] = pd.to_datetime(X[self.date_col], errors="coerce")

        # features temporais
        X["year"] = X[self.date_col].dt.year.astype("Int64")
        X["month"] = X[self.date_col].dt.month.astype("Int64")
        X["dow"] = X[self.date_col].dt.dayofweek.astype("Int64")  # 0=segunda
        X["quarter"] = X[self.date_col].dt.quarter.astype("Int64")
        X["ym"] = X[self.date_col].dt.to_period("M").astype(str)

        # NCM derivadas
        if self.ncm_col in X.columns:
            ncm = X[self.ncm_col].astype("string")
            X["ncm_chapter"] = ncm.str.slice(0, 2)

        return X


@dataclass
class MonthlyHistoricalRiskEncoder(BaseEstimator, TransformerMixin):
    """
    Cria features de risco histórico (target encoding temporal):
    - Agrega por mês (Period M)
    - Para cada (grupo, mês): calcula taxa cumulativa até o mês anterior
    - Durante transform, usa a taxa do mês anterior ao registro

    Gera features:
      risk_<col>  (taxa histórica da classe positiva)
      cnt_<col>   (volume histórico)

    Observação:
    - Para multiclasses, usamos um binário (classe positiva vs resto)
    """
    date_col: str
    target_col: str
    group_cols: List[str]
    positive_label: Optional[str] = None
    smoothing: float = 20.0  # suaviza categorias raras
    min_count: int = 1       # opcional: threshold
    global_prior_: float = 0.0

    _tables: Dict[str, pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y=None):
        if y is not None:
            target = pd.Series(y, name=self.target_col)
            df = X.copy()
            df[self.target_col] = target.values
        else:
            df = X.copy()

        if self.target_col not in df.columns:
            raise ValueError("target_col não encontrado em X (ou forneça y).")

        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df = df.dropna(subset=[self.date_col, self.target_col])

        # inferir label positivo
        pos = self.positive_label
        if pos is None:
            pos = _infer_positive_label(df[self.target_col].unique())
        if pos is None:
            raise ValueError(
                "Positive_label não inferido. "
                "Defina explicitamente (ex.: 'vermelho' ou 'red')."
            )

        # normalização string
        y_str = df[self.target_col].astype("string").str.strip().str.lower()
        pos = str(pos).strip().lower()
        df["_is_pos"] = (y_str == pos).astype(int)

        # prior global
        self.global_prior_ = float(df["_is_pos"].mean())

        df["_period"] = _to_period_month(df[self.date_col])

        self._tables = {}

        for col in self.group_cols:
            if col not in df.columns:
                continue

            # agregação mensal por categoria
            monthly = (
                df.groupby(["_period", col], dropna=False)
                .agg(
                    pos=("_is_pos", "sum"),
                    n=("_is_pos", "size"),
                )
                .reset_index()
                .sort_values("_period")
            )

            # cumulativo por categoria (até mês anterior)
            monthly["cum_pos"] = monthly.groupby(col)["pos"].cumsum().shift(1).fillna(0.0)
            monthly["cum_n"]   = monthly.groupby(col)["n"].cumsum().shift(1).fillna(0.0)

            # smoothing bayesiano
            m = float(self.smoothing)
            monthly["risk"] = (monthly["cum_pos"] + m * self.global_prior_) / (monthly["cum_n"] + m)

            self._tables[col] = monthly[["_period", col, "risk", "cum_n"]].copy()


        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col], errors="coerce")
        X["_period"] = _to_period_month(X[self.date_col])

        for col in self.group_cols:
            if self._tables is None or col not in self._tables or col not in X.columns:
                continue

            t = self._tables[col]

            # merge por (period, categoria)
            out = X.merge(
                t,
                how="left",
                left_on=["_period", col],
                right_on=["_period", col],
                suffixes=("", "_hist"),
            )

            risk_name = f"risk_{col}"
            cnt_name = f"cnt_{col}"

            X[risk_name] = out["risk"].fillna(self.global_prior_).astype(float)
            X[cnt_name] = out["cum_n"].fillna(0.0).astype(float)

        X = X.drop(columns=["_period"])
        return X


@dataclass
class TemporalAggregations(BaseEstimator, TransformerMixin):
    """
    Agregações temporais seguras por mês:
      - contagem histórica cumulativa até mês anterior por chave(s)
      - opcionalmente número de meses ativos etc.

    Ex.: histórico de volume por consignee_code e ncm_chapter.
    """
    date_col: str
    group_cols: List[str]
    prefix: str = "hist"
    _tables: Dict[Tuple[str, ...], pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df = df.dropna(subset=[self.date_col])
        df["_period"] = _to_period_month(df[self.date_col])

        self._tables = {}

        # cada col individual vira uma agregação
        for col in self.group_cols:
            if col not in df.columns:
                continue

            monthly = (
                df.groupby(["_period", col], dropna=False)
                  .size()
                  .reset_index(name="n")
                  .sort_values("_period")
            )
            monthly["cum_n"] = monthly.groupby(col)["n"].cumsum().shift(1).fillna(0.0)
            self._tables[(col,)] = monthly[["_period", col, "cum_n"]].copy()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col], errors="coerce")
        X["_period"] = _to_period_month(X[self.date_col])

        if not self._tables:
            return X.drop(columns=["_period"], errors="ignore")

        for key, t in self._tables.items():
            col = key[0]
            if col not in X.columns:
                continue

            out = X.merge(
                t,
                how="left",
                left_on=["_period", col],
                right_on=["_period", col],
            )
            feat_name = f"{self.prefix}_cnt_{col}"
            X[feat_name] = out["cum_n"].fillna(0.0).astype(float)

        X = X.drop(columns=["_period"])
        return X
