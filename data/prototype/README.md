O notebook contém a geração dos dados e a demonstração dos testes com uma regressão logística.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `scm_geracao_dados.ipynb` | geração dos dados com SCM, testes estatísticos e demonstração com regressão logística |
| `dados_scm/` | conjuntos gerados pelo notebook (CSV), criados ao executar a última célula |

## Como executar

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib jupyter
jupyter notebook scm_geracao_dados.ipynb
```

## O SCM usado

Grafo causal: `X1 → Y ← X2`, `Y → S`, e `Z` isolado.

As variáveis são geradas nesta ordem (causas antes dos efeitos):

| Variável | Papel | Equação |
|---|---|---|
| `X1` | causa real | `X1 = ε₁`, com `ε₁ ~ N(0,1)` |
| `X2` | causa real | `X2 = ε₂`, com `ε₂ ~ N(0,1)` |
| `Z` | ruído (controle negativo) | `Z = ε_Z`, com `ε_Z ~ N(0,1)` |
| `Y` | alvo binário | `Y = 1` se `1,5·X1 − 1,0·X2 + ε_Y > 0`, com `ε_Y ~ Logística(0,1)` |
| `S` | espúrio | `S = α·(2Y − 1) + ε_S`, com `ε_S ~ N(0,1)` |


## Ambientes gerados

| Ambiente | α | S em relação a Y | Correlação S–Y esperada |
|---|---|---|---|
| forte | 2,0 | quase entrega Y | +0,89 |
| médio | 1,0 | separação moderada | +0,71 |
| fraco | 0,5 | separação fraca | +0,45 |
| nulo | 0,0 | ruído puro | 0 |
| invertido fraco | −1,0 | relação invertida | −0,71 |
| invertido | −2,0 | totalmente invertida | −0,89 |

Cada ambiente tem **2000 amostras** e é gerado com **5 seeds** (30 conjuntos no total). Há ainda um conjunto de treino separado (α = 2, seed 999).
