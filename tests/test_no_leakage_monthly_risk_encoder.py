import numpy as np
import pandas as pd

from src.features import MonthlyHistoricalRiskEncoder


def test_monthly_risk_encoder_no_leakage():
    # Cenário sintético:
    # - Janeiro: 0% vermelho
    # - Fevereiro: 100% vermelho
    #
    # Se houver leakage, o risco em fevereiro para a categoria pode ficar ~1.0 (usando o próprio mês).
    # Se estiver correto, o risco em fevereiro deve refletir apenas janeiro (≈ 0, com smoothing puxando pro prior).

    df = pd.DataFrame({
        "registry_date": pd.to_datetime(
            ["2024-01-10", "2024-01-20", "2024-02-05", "2024-02-18"]
        ),
        "grp": ["A", "A", "A", "A"],
    })

    y = pd.Series(["VERDE", "VERDE", "VERMELHO", "VERMELHO"], name="channel")

    enc = MonthlyHistoricalRiskEncoder(
        date_col="registry_date",
        target_col="channel",
        group_cols=["grp"],
        positive_label="VERMELHO",
        smoothing=0.0,  # sem smoothing para tornar o teste mais rígido
    )

    enc.fit(df, y)
    out = enc.transform(df)

    # Risco em Janeiro para "A" deve cair no prior (não há histórico anterior): global_prior = 0.5 (2 vermelhos / 4)
    # Mas como smoothing=0, fórmula vira cum_pos/cum_n com cum_n=0 => NaN preenchido pelo prior no transform
    # Então o risco em janeiro = prior = 0.5
    jan_mask = out["registry_date"].dt.to_period("M").astype(str) == "2024-01"
    feb_mask = out["registry_date"].dt.to_period("M").astype(str) == "2024-02"

    # Em fevereiro, o histórico anterior é apenas janeiro (0 vermelhos em 2 obs) => risco = 0.0
    # (com smoothing=0)
    feb_risks = out.loc[feb_mask, "risk_grp"].to_numpy()

    assert np.allclose(feb_risks, 0.0), f"Leakage suspeito: riscos fevereiro={feb_risks} (esperado 0.0)"
