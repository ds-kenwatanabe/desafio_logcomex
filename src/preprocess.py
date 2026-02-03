from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from src.features import FeatureBuilder, MonthlyHistoricalRiskEncoder, TemporalAggregations


class PreImputeTransformer:
    """
    Pré-imputação leve antes do pipeline sklearn.
    Precisa ser top-level para permitir pickle/joblib.
    """
    def __init__(self, date_col, cat_cols_low, cat_cols_mid, high_card_cols):
        self.date_col = date_col
        self.cat_cols_low = cat_cols_low
        self.cat_cols_mid = cat_cols_mid
        self.high_card_cols = high_card_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # remover linhas sem data
        X = X.dropna(subset=[self.date_col])

        # normalizações
        if "country_origin_code" in X.columns:
            X["country_origin_code"] = X["country_origin_code"].astype("Int64").astype("string")

        # imputar categóricas relevantes
        fill_unknown = self.cat_cols_low + self.cat_cols_mid + self.high_card_cols
        for c in fill_unknown:
            if c in X.columns:
                X[c] = X[c].astype(str)
                X[c] = X[c].where(X[c].notna(), "UNKNOWN").str.strip()
                # OBS: não fazer astype(str) antes do fillna, NaN vira "nan" (string) e deixa de ser imputado como UNKNOWN.
                X[c] = X[c].astype("string")
                X[c] = X[c].fillna("UNKNOWN").str.strip()

        return X


@dataclass
class PreprocessConfig:
    date_col: str = "registry_date"
    target_col: str = "channel"

    # colunas base do dataset
    cat_cols_low: List[str] = None
    cat_cols_mid: List[str] = None
    high_card_cols: List[str] = None

    # colunas para risco histórico
    risk_group_cols: List[str] = None

    # agregações temporais
    agg_group_cols: List[str] = None

    positive_label: Optional[str] = "VERMELHO"

    def __post_init__(self):
        if self.cat_cols_low is None:
            self.cat_cols_low = [
                "transport_mode_pt",
                "consignee_company_size",
            ]
        if self.cat_cols_mid is None:
            self.cat_cols_mid = [
                "clearance_place_entry",
                "clearance_place_dispatch",
                "country_origin_code",
                "ncm_chapter",
            ]
        if self.high_card_cols is None:
            # não usar OHE
            self.high_card_cols = [
                "consignee_name",
                "shipper_name",
                "document_number",
            ]

        if self.risk_group_cols is None:
            # risco histórico (feature engineering)
            self.risk_group_cols = [
                "consignee_name",
                "ncm_chapter",
                "country_origin_code",
            ]

        if self.agg_group_cols is None:
            self.agg_group_cols = [
                "consignee_name",
                "ncm_chapter",
            ]


class _EnforceDtypes:
    def __init__(self, cat_cols, num_cols):
        self.cat_cols = cat_cols
        self.num_cols = num_cols

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()

        # categóricas para OHE
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = X[c].astype(str)  # object
                X[c] = X[c].fillna("UNKNOWN")

        # temporais numéricas (evitar pandas Int64/NA)
        for c in self.num_cols:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)

        # grante risk_/cnt_/hist_ são float
        for c in X.columns:
            if c.startswith(("risk_", "cnt_", "hist_")):
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)

        return X


def build_preprocess_pipeline(cfg: PreprocessConfig) -> Pipeline:
    """
    Pipeline completo:
      1) FeatureBuilder: datas + ncm derivadas
      2) Missing: regras
      3) Risk encoder (mensal, cumulativo, sem leakage)
      4) Agregações temporais (mensal, cumulativo)
      5) ColumnTransformer: OHE para categóricas de baixa/média cardinalidade
    """

    # - registry_date é essencial para features temporais + validação temporal; linhas sem data devem ser removidas.
    # - country_origin_code é código, mas categórico -> converter para string + imputar "UNKNOWN".
    # - ncm_chapter é derivado de ncm_code e costuma generalizar melhor que o NCM completo.
    # - consignee_name pode ser forte, mas é high-card -> usar risco histórico/contagens e NÃO one-hot.

    feature_builder = FeatureBuilder(date_col=cfg.date_col, ncm_col="ncm_code")

    # imputação mínima antes dos encoders:
    # input dentro do ColumnTransformer,
    # e para colunas usadas nos encoders históricos, preenchemos com "UNKNOWN" antes.
    pre_imputer = PreImputeTransformer(
    date_col=cfg.date_col,
    cat_cols_low=cfg.cat_cols_low,
    cat_cols_mid=cfg.cat_cols_mid,
    high_card_cols=cfg.high_card_cols,
    )

    risk_encoder = MonthlyHistoricalRiskEncoder(
        date_col=cfg.date_col,
        target_col=cfg.target_col,
        group_cols=cfg.risk_group_cols,
        positive_label=cfg.positive_label,
        smoothing=20.0,
    )

    temporal_aggs = TemporalAggregations(
        date_col=cfg.date_col,
        group_cols=cfg.agg_group_cols,
        prefix="hist",
    )

    # Após risk/aggs, teremos novas colunas numéricas: risk_* e cnt_*, hist_cnt_*
    # categóricas OHE e numéricas passthrough (com imputação)
    # identificamos numéricas por prefixo após transform.

    # ColumnTransformer não vê colunas criadas depois automaticamente se passarmos lista fixa.
    # OHE apenas nas colunas fixas e manter o restante como passthrough.
    # Categóricas (somente strings/códigos)
    cat_for_ohe = cfg.cat_cols_low + cfg.cat_cols_mid
    num_cols_selector = selector(dtype_include=np.number)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            cat_for_ohe),

            ("num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            num_cols_selector),
        ],
        remainder="drop",  # dropa strings (ym, ncm_4d, consignee_name etc.)
    )

    num_cols = ["year", "month", "dow", "quarter"]

    pipe = Pipeline(steps=[
        ("build_features", feature_builder),
        ("pre_impute", pre_imputer),
        ("risk", risk_encoder),
        ("aggs", temporal_aggs),
        ("enforce_types", _EnforceDtypes(cat_cols=cat_for_ohe, num_cols=num_cols)),
        ("encode", preprocessor),
    ])



    return pipe
