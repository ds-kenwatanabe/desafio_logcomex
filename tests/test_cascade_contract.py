import numpy as np
import pandas as pd

from src.cascade import CascadedChannelModel, CascadeConfig


def test_cascade_contract_fit_predict_and_evaluate():
    # Dataset sintético, 3 classes, sem CINZA
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 5))

    y = pd.Series(
        ["VERDE"] * 30 + ["AMARELO"] * 5 + ["VERMELHO"] * 5,
        name="channel"
    )

    # Evitar dependência de imblearn nos testes
    cfg = CascadeConfig(
        oversample_red_gate=False,
        red_gate_model="logreg",
        yg_model="linearsvc",
        random_state=42,
    )
    m = CascadedChannelModel(cfg).fit(X, y)

    preds = m.predict(X)
    assert len(preds) == len(y)

    allowed = {"VERDE", "AMARELO", "VERMELHO", "CINZA"}
    assert set(map(str, preds)).issubset(allowed)

    # Avaliações devem retornar as chaves esperadas
    s1 = m.evaluate_stage_red_gate(X, y)
    assert "accuracy" in s1 and "report_text" in s1 and "confusion_matrix" in s1

    s2 = m.evaluate_stage_yg(X, y)
    assert "accuracy" in s2 and "report_text" in s2 and "confusion_matrix" in s2

    e2e = m.evaluate_end_to_end(X, y)
    assert "accuracy" in e2e and "report_text" in e2e and "confusion_matrix" in e2e and "labels" in e2e
