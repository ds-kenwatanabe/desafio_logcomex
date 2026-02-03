# desafio_logcomex — Pipeline de classificação em cascata (Stage A / Stage B)

Este repositório contém uma pipeline de **engenharia de features + treino + avaliação** para prever o *canal* (`VERDE`, `AMARELO`, `VERMELHO`, `CINZA`) usando uma **arquitetura em cascata**:

- **Stage A (gate)**: `VERMELHO` vs `NOT_RED` (binário)
- **Stage B**: `AMARELO` vs `VERDE` (avaliado apenas quando `NOT_RED`)
- O `CINZA` pode aparecer no dataset, mas o foco principal é o gate do vermelho e o segundo estágio verde/amarelo.

A pasta `artifacts/` concentra tudo que é gerado (modelos, métricas, plots, análises).

---

## Estrutura do projeto (alto nível)

- `scripts/`: entrypoints CLI (treino, tuning, análises)
- `src/`: código de biblioteca (pipeline, modelo em cascata, utilitários)
- `configs/`: configs (ex.: `configs/cascade.yaml`)
- `data/`: datasets locais (ex.: `data/sample_data.parquet`)
- `artifacts/`: outputs do pipeline (modelos/plots/métricas)
- `tests/`: testes de contrato e não-vazamento

---

## Requisitos

- **Python 3.12** (recomendado)
- Pacotes principais:
  - `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`

---

## Setup

### 1) Clonar o projeto

No diretório do projeto:

```
git clone https://github.com/ds-kenwatanabe/desafio_logcomex.git
```

### 2) Instalar dependências

```python
pip install -r requirements.txt
```

---

## Configuração

O pipeline aceita `--config` apontando para **JSON** ou **YAML**.

Exemplo: `configs/cascade.yaml`

Campos típicos:

- `data.path`: caminho do dataset (`.parquet`)
- `split.cutoff`: data de corte (holdout temporal)
- `runtime.artifacts_dir`, `runtime.plots_dir`, `runtime.logs_dir`
- `preprocess.*`: colunas e parâmetros de pré-processamento
- `cascade.*`: parâmetros da cascata (`red_gate_model`, `yg_model`, `random_state`, etc.)

---

## Executando o pipeline (fluxo recomendado)

### Passo 0 — (Opcional) gerar features
É possível usar um dataset bruto e realizar engenharia de features:

```python
python -m scripts.make_features --config configs/cascade.yaml
```

> Caso esteja usando `data/sample_data.parquet` como dataset final, pule.

### Passo 1 — Treinar e avaliar a cascata

```python
python -m scripts.train_cascade --config configs/cascade.yaml
```

Você verá no terminal:
- balanceamento de classes (train/valid)
- métricas do Stage A e Stage B
- avaliação End-to-end
- matrizes de confusão (tabelas) + arquivos em `artifacts/plots/`

Outputs principais em `artifacts/`:
- `preprocess.joblib`
- `model_cascade.joblib`
- `metrics_valid.json`
- `metadata.json`
- `split.json`
- `threshold_used.json`
- `plots/*.png` e `plots/*.csv`

### Passo 2 — Ajustar threshold do gate do vermelho (Stage A)

Para sugerir um threshold com base em curva PR / candidatos:

```python
python -m scripts.tune_threshold --config configs/cascade.yaml
```

Outputs em `artifacts/threshold/`:
- `threshold_candidates.csv`
- `threshold_recommended.json`
- `pr_curve.png`

O `train_cascade.py` tenta automaticamente usar (nesta ordem):
1. `--red_threshold` (CLI)
2. `artifacts/threshold/threshold_recommended.json`
3. `cascade.red_gate_threshold` no config (ou default)

Para rodar treino com override:

```python
python -m scripts.train_cascade --config configs/cascade.yaml --red_threshold 0.23
```

### Passo 3 — Interpretabilidade (Stage A)

#### 3.1 Coeficientes / Feature importance do LogReg (Stage A)

```python
python -m scripts.stageA_feature_importance --config configs/cascade.yaml
```

#### 3.2 Permutation importance em holdout temporal (mais robusto)

```py
python -m scripts.stageA_permutation_importance --config configs/cascade.yaml
```

Arquivos típicos em `artifacts/analysis/`:
- `stageA_top_features.csv`
- `stageA_top_features_named.csv`
- `stageA_perm_importance.csv`
- `stageA_perm_importance_named.csv`
- `feature_index_map.csv`

---

## Relatórios e EDA

### EDA e perguntas do desafio
Foi feito um notebook `notebooks/01_eda_overview.ipynb` com uma análise prévia para verificar informações gerais do dataset.
Além disso para as perguntas específicas, foi feito um script a fim de repondê-las.

```python
python -m scripts.eda_answers --config configs/cascade.yaml
```

Saídas em `artifacts/eda/` (`.csv` e `.png`).

### Report consolidado

- `reports/reports.md`
- `reports/Relatorio_Executivo.pdf`

---

## Testes

Rodar com `pytest`:

```python
pip install pytest
pytest -q
```

Testes relevantes:
- `tests/test_cascade_contract.py`
- `tests/test_no_leakage_monthly_risk_encoder.py`
- `tests/test_artifacts_roundtrip.py`

---

## Observações importantes

### 1) Holdout temporal
O split é por `registry_date` com `split.cutoff`, evitando vazamento temporal.

### 2) Ordem dos argumentos do `save_json`
O helper em `src/common/io.py`:

```python
save_json(path, obj)
```

### 3) Artefatos
Por padrão:
- `artifacts/` (modelos, métricas, configs)
- `artifacts/plots/` (matrizes de confusão em PNG/CSV)
- `artifacts/analysis/` (importance, mapeamentos)
- `artifacts/threshold/` (tuning do threshold)
- `logs/` (logs do treino)
