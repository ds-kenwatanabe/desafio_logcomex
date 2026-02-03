import joblib
import numpy as np
import pandas as pd

from src.cascade import CascadedChannelModel, CascadeConfig


def test_artifacts_roundtrip_joblib(tmp_path):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(60, 6))

    # OBS: Stage B precisa de pelo menos 2 classes entre {VERDE, AMARELO}
    # Então incluímos alguns AMARELOs.
    y = pd.Series(
        ["VERDE"] * 45 + ["AMARELO"] * 5 + ["VERMELHO"] * 10,
        name="channel"
    )

    # Evita dependência de imblearn no teste
    cfg = CascadeConfig(
        oversample_red_gate=False,
        red_gate_model="logreg",
        yg_model="linearsvc",
        random_state=123,
    )
    model = CascadedChannelModel(cfg).fit(X, y)

    p1 = model.predict(X)

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)

    loaded = joblib.load(path)
    p2 = loaded.predict(X)

    assert np.array_equal(p1, p2), "Predições mudaram após dump/load (roundtrip falhou)."
