# Relatório técnico — Previsão de Canal Aduaneiro (Cascata / Risk Gate VERMELHO)

## 0) Objetivo
O objetivo do projeto é prever o canal aduaneiro (VERDE/AMARELO/VERMELHO/CINZA) em um cenário de **desbalanceamento extremo**, com foco operacional em **mitigação de risco**.  
O custo de errar um VERMELHO (falso negativo - FN) é muito maior do que o custo falsos positivos (FP), a métrica principal do sistema é:

- **Recall do VERMELHO (Stage A)**: maximizar identificação de risco (reduzir FN)
- Métricas auxiliares: FP (carga operacional), Precision do VERMELHO, e curvas PR/threshold

Acurácia global end-to-end é **não-objetivo**, pois é dominada pela classe VERDE (≈97%) e não reflete risco.


## 0.1) Testes iniciais
Primeiro foi feito um teste antes da adoção da arquitetura em cascata, foram avaliados modelos lineares multiclasse como baseline, com o objetivo de verificar se abordagens diretas seriam suficientes para o problema proposto. Pode ser visto no arquivo `scripts/train_and_evaluate.py`.

Foram testados:

Regressão Logística multiclasse com class_weight="balanced"

Linear SVM (LinearSVC) com balanceamento de classes

Regressão Logística combinada com oversampling da classe minoritária (RandomOverSampler)

Todos os modelos foram treinados utilizando o mesmo pipeline de pré-processamento, com divisão temporal dos dados, garantindo comparabilidade justa entre abordagens.

Os resultados mostraram que, apesar de razoável desempenho em termos de acurácia global, os modelos multiclasse diretos apresentaram baixa capacidade de recuperação da classe VERMELHO, que é a classe de maior interesse do ponto de vista de risco operacional.

Em um cenário de desbalanceamento extremo (~2% de VERMELHO), a acurácia global se mostrou inadequada como métrica de sucesso, sendo dominada pela classe VERDE. Mesmo com balanceamento e oversampling, os modelos apresentaram trade-offs desfavoráveis entre falsos negativos e falsos positivos, dificultando a calibração do risco.

A partir dessas observações, optou-se por reformular o problema de classificação multiclasse em uma arquitetura hierárquica (cascata), separando explicitamente:

1. canal de risco focado exclusivamente na detecção de VERMELHO (alta sensibilidade),

2. seguido de um estágio secundário para diferenciação operacional entre VERDE e AMARELO.

Essa reformulação permitiu alinhar diretamente o modelo com o objetivo de negócio, minimizando falsos negativos de VERMELHO. Possibilitou a  calibração explícita do trade-off entre risco e carga operacional por meio de threshold.

---

## 1) Dados, desbalanceamento e split temporal
As classes são altamente desbalanceadas:

- TRAIN: VERDE ~97.0%, VERMELHO ~2.1%, AMARELO ~0.87%, CINZA ~0.046%
- VALID: VERDE ~97.0%, VERMELHO ~2.13%, AMARELO ~0.82%, CINZA ~0.052%

O split foi temporal usando `registry_date` com cutoff em **2024-10-01** para simular cenário real de produção (treino passado -> validação futura).

---

## 2) Pipeline de features e prevenção de leakage
A pipeline de pré-processamento foi construída para suportar:
- features temporais a partir de `registry_date`
- derivação de NCM (ex.: capítulo)
- target encoding histórico mensal (sem leakage)
- agregações cumulativas / contagens históricas
- OHE controlado e compatível com matrizes esparsas (sparse)

Para evitar leakage temporal, o risco histórico mensal é calculado de modo que, para um evento em mês M, só se use informação **até M-1**.

---

## 3) Modelagem: cascata hierárquica
Modelos multiclasse “diretos” tiveram desempenho operacional fraco em cenário de desbalanceamento extremo. Verificados em `scripts/train_and_evaluate.py` 
Foi adotada uma arquitetura hierárquica (cascata), implementada em `scripts/train_cascade.py` orientada ao objetivo de negócio:

### Stage A (Canal de risco): **VERMELHO vs NÃO-VERMELHO**
- Modelo: LogisticRegression (sparse-friendly) + balanceamento (class_weight/oversampling)
- Threshold: **t = 0.4975** (selecionado via tuning por recall-alvo 0.80)
- Objetivo: alto recall do VERMELHO, aceitando FP (alerta operacional)

### Stage B (Diferenciação operacional): **AMARELO vs VERDE**
- Modelo: LinearSVC (sparse-friendly, robusto)
- Objetivo: separar “inspeção/atenção” (AMARELO) de “fluxo normal” (VERDE)
- Observação: AMARELO é difícil e pode ser tratado como “zona de incerteza” em abordagens futuras

### CINZA
- Frequência extremamente baixa. Tratado como exceção/fallback (não modelado explicitamente)

### Robustez
O cascata foi reforçado para não falhar caso o Stage B tenha apenas 1 classe no treino (ex.: ausência de AMARELO em uma janela), usando fallback seguro “sempre VERDE” para não-vermelhos nesse caso.

---

## 4) Resultados quantitativos (VALID)

### 4.1 Stage A — Canal vermelho (principal)
Threshold operacional: **0.4975** (source: `threshold_recommended.json`)

Métricas (VALID):
- Recall (VERMELHO=1): **0.82**
- Precision (VERMELHO=1): **0.02**
- Matriz de confusão (labels: 0=not_red, 1=red):
  - TN=2379, FP=12754
  - FN=58,   TP=272

Interpretação:
- O sistema recupera ~82% dos vermelhos (**bom para risco**).
- Há muitos FP (alertas), o que é esperado em um canal sensível com base-rate muito baixa (~2% vermelho).
- O trade-off FN vs FP é governado por threshold e deve ser calibrado conforme capacidade operacional.

Artifacts:
- `artifacts/plots/stageA_red_gate_cm_counts.png` e `.csv`
- `artifacts/threshold_used.json` (threshold aplicado)

---

### 4.2 Stage B — Amarelo vs Verde (secundário)
Métricas (VALID, apenas amostras true ∈ {VERDE, AMARELO}):
- Recall (AMARELO=1): **0.41**
- Precision (AMARELO=1): **0.01**
- Confusion Matrix:
  - VERDE: 8604 corretos, 6394 confundidos como AMARELO
  - AMARELO: 52 corretos, 75 perdidos (viraram VERDE)

Interpretação:
- AMARELO é um sinal fraco/difícil: recall moderado, precisão muito baixa.
- Isso reforça tratar AMARELO como “incerteza” ou melhorar com features específicas / modelos não lineares / regras híbridas.

Artifacts:
- `artifacts/plots/stageB_yellow_vs_green_cm_counts.png` e `.csv`

---

### 4.3 End-to-end (auditoria)
Accuracy end-to-end (VALID): **0.105**
Não é métrica de sucesso devido ao design do canal vermelho (sensível) e ao desbalanceamento extremo.

Observação: end-to-end mostra alta taxa de predição VERMELHO (muitos FP), coerente com Stage A.

Artifacts:
- `artifacts/plots/end2end_cm_counts.png` e `.csv`

---

## 5) Respostas às Perguntas Específicas

### 1) Quais são os 5 NCMs com maior risco de canal vermelho?
Foi gerado ranking robusto usando:
- taxa de vermelho por NCM
- filtro de volume mínimo (`min_n`)
- ranking por **Wilson lower bound** (evita que NCMs raros dominem por acaso)

Considerando isto, os top 5 NCMs são:
1. 38220090
2. 22072000
3. 22071000
4. 85444200
5. 40169990

Outputs:
- `artifacts/eda/q1_top5_ncm_red_risk.csv`
- `artifacts/eda/q1_top5_ncm_red_risk.png`

Interpretação:
- NCMs do topo combinam **alta taxa de vermelho** com **volume**.

Próximo aprimoramento:
- incluir também intervalo de confiança completo, e análise pelo capítulo NCM para generalização.

---

### 2) Existe sazonalidade na distribuição dos canais?
Foi gerada a série mensal da proporção de canais por mês (share mensal):
- no intervalo analisado, não houve variação relevante

Outputs:
- `artifacts/eda/q2_channel_share_by_month.csv`
- `artifacts/eda/q2_channel_share_by_month.png`

Como isso impacta o modelo, caso haja sazonalidade:
- se a proporção de VERMELHO variar ao longo do ano, o threshold fixo pode ficar “alto/baixo” demais em certas épocas
- drift sazonal afeta também categorias (NCM, transport_mode, porte)

Mitigação:
- monitoramento por mês (PSI/KS em features e base-rate)
- retreinamento periódico ou recalibração de threshold por janela móvel

---

### 3) Qual o impacto do modo de transporte na seleção do canal?
Foi medido risco de VERMELHO por `transport_mode_pt`, com volume mínimo e ranking (similar ao NCM)

Ordem de impacto do modo de transporte:
1. AEREA
2. MARITIMA
3. POSTAL
4. RODOVIARIA
5. FERROVIARIA

Justificativa:
- as taxas por modo (e Wilson LB) permitem identificar quais modais concentram maior risco histórico
- isso suporta tanto feature importance quanto regras híbridas se necessário

Outputs:
- `artifacts/eda/q3_transport_mode_red_risk.csv`
- `artifacts/eda/q3_transport_mode_red_risk.png`

---

### 4) Como o porte da empresa importadora influencia o canal selecionado?
Foi calculado risco de VERMELHO por `consignee_company_size`:

1. EMPRESA DE PEQUENO PORTE
2. DEMAIS
3. MICRO EMPRESA

Outputs:
- `artifacts/eda/q4_company_size_red_risk.csv`
- `artifacts/eda/q4_company_size_red_risk.png`

Interpretação:
- o porte pode atuar como proxy de maturidade/regularidade/capacidade documental
- porém também é fonte potencial de bias (ver seção 11)

---

## 6) Modelagem (perguntas 5–8)

### 5) Qual modelo apresentou melhor desempenho? (métricas de negócio)
O melhor desempenho operacional para o objetivo do projeto foi o **cascata**, porque:
- Stage A atinge **recall VERMELHO ~0.82** com threshold calibrado
- transforma um problema multiclasse desbalanceado em canal de risco + refinamento
- permite calibrar explicitamente o trade-off FN/FP por threshold

Métrica de negócio:
- minimizar FN de VERMELHO (alto custo)
- aceitar FP dentro da capacidade operacional

---

### 6) Como lidar com desbalanceamento extremo? (2 estratégias)
Foram aplicadas (e são recomendadas em conjunto):

**Estratégia A — Reenquadramento do problema**
- arquitetura em cascata: primeiro detectar VERMELHO vs resto
- reduz complexidade do multiclasse e melhora foco em risco

**Estratégia B — Reamostragem / pesos**
- `class_weight="balanced"` e/ou oversampling no Stage A (oversampling é utilizado por padrão)
- aumenta a sensibilidade para a classe rara
- tuning de threshold complementa para controlar FP

Estratégias adicionais (futuras):
- focal loss / modelos gradiente com class weights (LightGBM/CatBoost)
- hard negative mining - se concentra nas amostras negativas mais difíceis durante o treinamento
- threshold por segmento (modal, capítulo NCM) com cuidado de overfit

---

### 7) Quais features são mais importantes? (interpretabilidade)
Como os modelos usados são lineares (LogReg/LinearSVC), recomenda-se:

- **LogReg (Stage A)**: importância por coeficiente (abs(coef)) no espaço de features
- Importância baseada no valor absoluto do coeficiente do modelo de Regressão Logística.
```
| feature | coef | abs_coef | direction | feature_index | feature_name |
|--------|------|----------|-----------|---------------|--------------|
| f_0  | 0.010318 | 0.010318 | ↑ increases RED risk | 0  | cat__transport_mode_pt_AEREA |
| f_3  | -0.009326 | 0.009326 | ↓ decreases RED risk | 3  | cat__transport_mode_pt_MARITIMA |
| f_69 | 0.005430 | 0.005430 | ↑ increases RED risk | 69 | cat__ncm_chapter_22 |
| f_82 | 0.004745 | 0.004745 | ↑ increases RED risk | 82 | num__dow |
| f_81 | 0.004501 | 0.004501 | ↑ increases RED risk | 81 | num__month |
```

- **Permutation importance** em holdout temporal (mais robusto)
- Mede a queda de performance quando a feature é embaralhada no conjunto de validação.
```
| feature | j | importance_mean | importance_std | feature_index | feature_name |
|--------|---|------------------|----------------|---------------|--------------|
| f_89 | 89 | 0.01879 | 0.00844 | 89 | num__cnt_country_origin_code |
| f_83 | 83 | 0.01758 | 0.00227 | 83 | num__quarter |
| f_82 | 82 | 0.01636 | 0.00454 | 82 | num__dow |
| f_91 | 91 | 0.01394 | 0.00411 | 91 | num__hist_cnt_ncm_chapter |
| f_87 | 87 | 0.00848 | 0.00227 | 87 | num__cnt_ncm_chapter |
```

- Diferença entre as duas abordagens
- LogReg — Coeficientes (Stage A)

Mede a influência direta da feature na decisão do modelo

Calculada no espaço transformado (após encoding / scaling)

Como interpretar?

* coef > 0 → aumenta a probabilidade de RED

* coef < 0 → diminui a probabilidade de RED

* abs(coef) → força da influência

Vantagens:

1. Interpretabilidade direta

1. Direção causal clara (↑ / ↓ risco)

Limitações

1. Sensível a correlação entre features, escala, regularização

1. Mede influência interna, não impacto real na performance


- Permutation Importance (Holdout Temporal)

Mede o desempenho do modelo piora quando a feature perde informação

Avaliado fora do treino, em dados futuros

Como interpretar:

* importance_mean alto → feature importante para previsão

* Não indica direção (↑ ou ↓ risco)

* importance_std alto → instabilidade / dependência de contexto

Vantagens

1. Mais robusto

2. Captura interações e dependências reais

3. Reflete impacto prático no modelo

Limitações

1. Não informa sinal do efeito

2. Features correlacionadas podem dividir importância

- Futuramente: **SHAP** para modelo linear/sparse (custo maior)

Outputs:
- `artifacts/analysis/stageA_top_features.csv` com top-N coeficientes
- `stageA_perm_importance.csv`

---

### 8) Como garantir robustez a mudanças sazonais?
- Split temporal já reduz risco de leakage e simula produção.
Mas é possível:
- Monitorar drift por mês e por segmentos críticos (NCM/modal/porte).
- Recalibrar threshold periodicamente (ex.: mensal) usando PR curve recente.
- Retreinamento com janela móvel (ex.: últimos 6–12 meses).

---

## 7) Produção

### 9) Como implementar monitoramento em produção?
Recomendações mínimas:
- Monitorar **base-rate** de VERMELHO real e **taxa de alerta** (pred=vermelho)
- Monitorar FN/FP (quando rótulo chega) e métricas por segmento (NCM/modal/porte)
- Drift de features: PSI/KS em variáveis-chave + embeddings/encodings
- Auditoria: logar `red_proba`, `threshold_used`, top features (quando possível)

---

### 10) Estratégia de retreinamento recomendada
- Retreinamento periódico (mensal ou bimestral) com janela móvel
- Recalibração de threshold mais frequente que retreino (ex.: semanal/mensal)
- Gatilho por drift: se PSI/base-rate ultrapassar limites, antecipar retreino

---

### 11) Principais riscos de bias
Fontes potenciais:
- `consignee_company_size`: pode penalizar pequenos importadores
- modal e NCM podem refletir setores/indústrias com políticas de inspeção diferentes
- target encoders históricos podem perpetuar vieses do passado (feedback loop)

Mitigações:
- Avaliar métricas por subgrupos (porte/modal)
- Ajustar threshold por segmento com cautela (evitar discriminação)
- Documentar limitações e evitar uso automático punitivo sem revisão humana
