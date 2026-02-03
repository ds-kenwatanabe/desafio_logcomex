from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    from imblearn.over_sampling import RandomOverSampler
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False


def _as_series(y) -> pd.Series:
    if isinstance(y, pd.Series):
        return y.astype(str)
    return pd.Series(y).astype(str)


def pretty_confusion_matrix(y_true, y_pred, labels, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    df_cm = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    df_cmn = pd.DataFrame(np.round(cmn * 100, 2), index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

    print(f"\n--- {title} | Confusion Matrix (counts) ---")
    print(df_cm)

    print(f"\n--- {title} | Confusion Matrix (row-normalized %) ---")
    print(df_cmn)

    return {"counts": cm, "row_norm": cmn, "labels": labels}


@dataclass
class CascadeConfig:
    label_green: str = "VERDE"
    label_yellow: str = "AMARELO"
    label_red: str = "VERMELHO"
    label_gray: str = "CINZA"

    # oversampling (gate vermelho)
    oversample_red_gate: bool = True
    red_gate_sampler: str = "random_over"   # futuro: "none"
    random_state: int = 42

    # modelos base (bons para sparse)
    red_gate_model: str = "logreg"  # "logreg" ou "linearsvc"
    yg_model: str = "linearsvc"  # "logreg" ou "linearsvc"

    # threshold operacional do canal vermelho
    red_gate_threshold: float = 0.5

    # hiperparâmetros
    logreg_C: float = 1.0
    logreg_max_iter: int = 8000
    logreg_tol: float = 1e-3


class CascadedChannelModel(BaseEstimator):
    """
    Cascata:
      A) Red gate: RED vs NOT_RED
      B) Yellow/Green: YELLOW vs GREEN (apenas para NOT_RED)
    CINZA fica como fallback (não modelado aqui) por baixa frequência.
    """

    def __init__(self, cfg: CascadeConfig):
        self.cfg = cfg

        self.model_red_: Optional[Any] = None
        self.model_yg_: Optional[Any] = None
        self.stage_yg_trained_: bool = False

    # helpers internos do gate vermelho
    def _red_proba(self, X) -> Optional[np.ndarray]:
        """
        Retorna probabilidade de RED (classe 1) se o modelo suportar predict_proba.
        Caso contrário, retorna None.
        """
        if self.model_red_ is None:
            return None
        if hasattr(self.model_red_, "predict_proba"):
            proba = self.model_red_.predict_proba(X)[:, 1]
            return np.asarray(proba)
        return None

    def _red_pred_bin(self, X) -> np.ndarray:
        """
        Predição binária do gate vermelho:
        - se houver proba: aplica threshold self.cfg.red_gate_threshold
        - senão: usa predict() do estimator (fallback)
        """
        if self.model_red_ is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        proba = self._red_proba(X)
        if proba is not None:
            return (proba >= float(self.cfg.red_gate_threshold)).astype(int)
        return np.asarray(self.model_red_.predict(X))

    def predict_red_gate(self, X) -> Dict[str, Any]:
        """
        Interface pública para análises/tuning:
        retorna pred_bin, proba (se disponível) e threshold usado.
        """
        proba = self._red_proba(X)
        pred = self._red_pred_bin(X)
        return {
            "pred": pred,
            "proba": proba,
            "threshold": float(self.cfg.red_gate_threshold),
            "has_proba": proba is not None,
        }

    def _make_logreg(self, class_weight=None) -> LogisticRegression:
        return LogisticRegression(
            solver="saga",
            C=self.cfg.logreg_C,
            max_iter=self.cfg.logreg_max_iter,
            tol=self.cfg.logreg_tol,
            class_weight=class_weight,
        )

    def _make_linearsvc(self, class_weight=None) -> LinearSVC:
        return LinearSVC(class_weight=class_weight)

    def _make_model(self, which: str, class_weight=None):
        if which == "logreg":
            return self._make_logreg(class_weight=class_weight)
        if which == "linearsvc":
            return self._make_linearsvc(class_weight=class_weight)
        raise ValueError(f"Unknown model type: {which}")

    def fit(self, X, y):
        y = _as_series(y)

        # A) RED vs NOT_RED
        y_red = (y == self.cfg.label_red).astype(int)  # 1=red
        self.model_red_ = self._make_model(self.cfg.red_gate_model, class_weight="balanced")

        X_red = X
        y_red_fit = y_red

        if self.cfg.oversample_red_gate:
            if not IMBLEARN_AVAILABLE:
                raise RuntimeError("imbalanced-learn não instalado, mas oversample_red_gate=True.")
            ros = RandomOverSampler(random_state=self.cfg.random_state)
            X_red, y_red_fit = ros.fit_resample(X_red, y_red_fit)

        self.model_red_.fit(X_red, y_red_fit)

        # B) YELLOW vs GREEN (somente não-vermelho)
        # OBS: para indexar sparse, use índices inteiros (não pd.Series)
        idx_not_red = np.flatnonzero((y != self.cfg.label_red).to_numpy())

        y_sub = y.iloc[idx_not_red]
        idx_yg_local = np.flatnonzero(y_sub.isin([self.cfg.label_green, self.cfg.label_yellow]).to_numpy())
        idx_yg = idx_not_red[idx_yg_local]

        X_yg = X[idx_yg]
        y_yg = y.iloc[idx_yg]

        # binário: yellow=1, green=0
        y_yg_bin = (y_yg == self.cfg.label_yellow).astype(int).to_numpy()

        # se não houver 2 classes no treino do stage B, não treina o modelo
        # (ex.: nenhum AMARELO no período). Nesse caso, fallback em predict = sempre VERDE para não-vermelhos.
        if np.unique(y_yg_bin).size < 2:
            self.model_yg_ = None
            self.stage_yg_trained_ = False
        else:
            self.model_yg_ = self._make_model(self.cfg.yg_model, class_weight="balanced")
            self.model_yg_.fit(X_yg, y_yg_bin)
            self.stage_yg_trained_ = True

        return self


    def predict(self, X):
        if self.model_red_ is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")

        pred_red = self._red_pred_bin(X)

        # inicializa tudo como VERDE
        out = np.full(X.shape[0], self.cfg.label_green, dtype=object)

        # marca vermelhos
        out[pred_red == 1] = self.cfg.label_red

        # B) para não-vermelhos, decide verde/amarelo
        # Se o stage B não foi treinado, fallback: todos os não-vermelhos ficam VERDE.
        if self.model_yg_ is None:
            return out

        idx_not_red = np.where(pred_red == 0)[0]
        if idx_not_red.size > 0:
            X_nr = X[idx_not_red]
            pred_y = self.model_yg_.predict(X_nr)
            pred_y = np.asarray(pred_y)

            out[idx_not_red[pred_y == 1]] = self.cfg.label_yellow
            out[idx_not_red[pred_y == 0]] = self.cfg.label_green

        return out

    # helpers de avaliação
    def evaluate_end_to_end(self, X, y_true) -> Dict[str, Any]:
        y_true = _as_series(y_true)
        y_pred = pd.Series(self.predict(X), index=y_true.index).astype(str)

        labels = [self.cfg.label_green, self.cfg.label_yellow, self.cfg.label_red, self.cfg.label_gray]
        # alguns labels podem não aparecer no y_true do holdout; ok.
        labels_present = [l for l in labels if l in set(y_true.unique())]

        acc = accuracy_score(y_true, y_pred)
        rep = classification_report(y_true, y_pred, labels=labels_present, zero_division=0, output_dict=False)
        cm = confusion_matrix(y_true, y_pred, labels=labels_present)

        return {
            "accuracy": acc,
            "report_text": rep,
            "confusion_matrix": cm,
            "labels": labels_present,
        }

    def evaluate_stage_red_gate(self, X, y_true) -> Dict[str, Any]:
        y_true = _as_series(y_true)
        y_bin = (y_true == self.cfg.label_red).astype(int).to_numpy()

        pred = self._red_pred_bin(X)
        acc = accuracy_score(y_bin, pred)
        rep = classification_report(y_bin, pred, labels=[0, 1], zero_division=0, output_dict=False)
        cm = confusion_matrix(y_bin, pred, labels=[0, 1])

        out = {"accuracy": acc, "report_text": rep, "confusion_matrix": cm, "labels": [0, 1]}
        # acrescenta informação de threshold/proba quando existir
        proba = self._red_proba(X)
        out["red_gate_threshold"] = float(self.cfg.red_gate_threshold)
        out["red_gate_has_proba"] = proba is not None
        return out


    def evaluate_stage_yg(self, X, y_true) -> Dict[str, Any]:
        y_true = _as_series(y_true)

        # avalia só onde true está em {verde, amarelo}
        idx = np.flatnonzero(y_true.isin([self.cfg.label_green, self.cfg.label_yellow]).to_numpy())

        X2 = X[idx]
        y2 = y_true.iloc[idx]
        y_bin = (y2 == self.cfg.label_yellow).astype(int).to_numpy()

        # Se stage B não foi treinado, baseline: sempre predizer VERDE (0)
        if self.model_yg_ is None:
            pred = np.zeros_like(y_bin)
            stage_trained = False
        else:
            pred = np.asarray(self.model_yg_.predict(X2))
            stage_trained = True

        acc = accuracy_score(y_bin, pred)
        rep = classification_report(y_bin, pred, labels=[0, 1], zero_division=0, output_dict=False)
        cm = confusion_matrix(y_bin, pred, labels=[0, 1])

        return {
            "accuracy": acc,
            "report_text": rep,
            "confusion_matrix": cm,
            "labels": [0, 1],
            "stage_trained": stage_trained,
        }

