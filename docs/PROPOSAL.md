# Proposta do Projeto: CausalFoundations

## 1. Integrantes
- **GUILHERME TORRESAN BAZZO** - 00287687 - Mestrado em Ciência da Computação
- **JOÃO ENRIQUE REGO CAIRUGA** - 00549998 - Mestrado em Ciência da Computação
- **KAMILLE KONARZEWSKI PIMENTEL** - 00342607 - Bacharelado em Ciência da Computação
- **LUCIANO LUZ BEYLOUNI FARIAS** - 00323903 - Mestrado em Ciência da Computação
- **SOFIA BATISTA SARTORI** - 00333201 - Mestrado em Ciência da Computação

---

## 2. Nome do Grupo
**CausalFoundations**

---

## 3. Tópicos Relacionados
- Análise de dados: correlação, dependência e seleção de atributos;
- Shortcut learning e correlações espúrias;
- Valores de Shapley e SHAP (SHapley Additive exPlanations);
- Análise contrafactual;
- Interpretabilidade para inferência causal e descoberta científica;
- Modelos Fundacionais Tabulares (ex: TabPFN, TabICL) vs. Baselines Clássicos (LightGBM, XGBoost, MLP, Regressão Logística).

---

## 4. Problema / Objeto de Estudo e Literatura

### Objeto de estudo
Investigar se os padrões utilizados por modelos fundacionais tabulares para realizar previsões são compatíveis com a estrutura causal que gera os dados e como esses padrões se comportam diante de mudanças nas relações entre as variáveis.

### Estado da arte e contextualização
Modelos fundacionais tabulares, como o TabPFN, utilizam o conhecimento adquirido durante o pré-treinamento para realizar previsões em novos conjuntos de dados tabulares, apresentando resultados competitivos em determinadas tarefas, especialmente em conjuntos de dados pequenos. Entretanto, compreender quais padrões esses modelos utilizam para generalizar ainda é uma questão relevante.

Uma possibilidade é que esses modelos aprendam regularidades estatísticas que se mantêm em diferentes conjuntos de dados. Outra é que parte de seu desempenho dependa de correlações específicas dos dados, que podem deixar de existir em novos cenários (shortcut learning / spurious signals). Investigar essa distinção é importante para compreender a robustez e os limites de generalização desses modelos.

Nesse contexto, métodos de interpretabilidade, como SHAP, e análises contrafactuais podem ajudar a investigar quais variáveis influenciam as previsões e como elas se alteram diante de mudanças controladas nos dados. Ao utilizar conjuntos de dados sintéticos ou semi-sintéticos com estruturas causais conhecidas injetadas sobre covariáveis reais do Censo Americano (**Folktables** / ACS PUMS), é possível comparar diretamente o comportamento observado do modelo com as relações definidas no processo de geração dos dados (Ground Truth DAG).

### Referências Principais
1. **Molnar, C. (2026).** *Tabular Foundation Models: A Short and Opinionated Guide.* [https://tabularfoundationmodels.com/](https://tabularfoundationmodels.com/)
2. **Hollmann, N. et al. (2023).** *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second.* ICLR 2023.
3. **Schölkopf, B. et al. (2021).** *Toward Causal Representation Learning.* Proceedings of the IEEE.
4. **Qu, J. et al. (2026).** *TabICLv2: A better, faster, scalable, and open tabular foundation model.* [arXiv:2602.11139](https://doi.org/10.48550/arXiv.2602.11139)
5. **Hollmann, N. et al. (2025).** *Accurate predictions on small data with a tabular foundation model.* Nature, 637, 319–326.
6. **Liu, Q. et al. (2026).** *Causal Pre-training Under the Fairness Lens: An Empirical Study of TabPFN.* Proceedings of the ACM Web Conference 2026 (WWW '26).
7. **Vlontzos, A. et al. (2026).** *Entangled by Design: Spurious Intra-Variable Signal Routing in Tabular In-Context Learners.* arXiv:2607.25532.
8. **Ding, F., Hardt, M., Miller, J., Schmidt, L. (2021).** *Retiring Adult: New Datasets for Fair Machine Learning (Folktables).* NeurIPS 2021.

---

## 5. Pergunta Fundamental
> **Por que os modelos fundacionais tabulares conseguem fazer boas previsões em conjuntos de dados que nunca viram?**

---

## 6. Hipótese Específica
> **Hipótese:** Modelos fundacionais tabulares generalizam quando aprendem regularidades estatísticas que se mantêm entre datasets, mas perdem desempenho quando dependem de correlações específicas que mudam em novos cenários.  
> **Estratégia de teste:** Utilizaremos dados com causalidade conhecida (injeção de SCMs em dados do Folktables), variando relações entre variáveis (intervenções e ruído espúrio), e aplicaremos interpretabilidade (SHAP, contrafactuais) para identificar se os atributos que guiam as previsões correspondem à estrutura causal real ou a atalhos estatísticos.

---

## 7. Limitações e Alternativas da Hipótese
- **Possível falha da suposição:** Assumir que estabilidade estatística implica aprendizado causal.
- **Hipótese alternativa:** O modelo pode ignorar relações causais e basear-se puramente no casamento de padrões (*in-context pattern matching*) vistos no pré-treino sintético bayesiano. A generalização dependeria estritamente da proximidade entre o novo dataset e o espaço de priors do modelo.

---

## 8. Conjunto de Dados e Metodologia
- **Base Real:** **Folktables (ACS PUMS)** — Dados reais do Censo Americano (renda, emprego, transporte, demografia, estados e anos múltiplos). Fornece covariáveis reais com ruído e multicolinearidade autênticos.
- **Injeção de SCM (Semi-sintético):**
  - Selecionam-se covariáveis reais como pais causais verdadeiros $Pa(Y)$.
  - O alvo $Y$ é gerado via mecanismo estrutural controlado $Y = f(Pa(Y), \epsilon)$.
  - Cria-se um atributo espúrio $S$ fortemente correlacionado com $Y$ no ambiente de treino (ex: Estado CA ou ano base), porém descorrelacionado ou invertido no ambiente de teste (ex: Estado TX, MI ou intervenção).
- **Avaliação:** Comparação de TabPFN com baselines (LightGBM, XGBoost, Regressão Logística). Avaliação de precisão preditiva (AUC/F1/LogLoss) e alinhamento causal via SHAP (Causal Precision@k, atribuição espúria vs. causal).

---

## 9. Riscos Metodológicos e Mitigações
- **Risco:** Dados puramente sintéticos podem ser simples demais ou descolados da realidade.
  - *Mitigação:* Usar abordagem **semi-sintética** sobre covariáveis autênticas do Folktables.
- **Risco:** SHAP mede atribuição preditiva, não efeito causal estrito.
  - *Mitigação:* Medir a concordância entre os atributos de maior relevância SHAP e o conjunto de pais verdadeiros $Pa(Y)$, complementado por testes contrafactuais diretos ($do(X_i = x)$ vs $do(S = s)$).

---

## 10. Desafios Antecipados
1. Variar intensidade de correlações e ruído sem colapsar a identificabilidade causal.
2. Garantir que perturbações estatísticas não alterem inadvertidamente as relações causais fundamentais.
3. Custo computacional de cálculo de SHAP exato / permutação sobre grandes amostras (mitigado por amostragem e TreeSHAP / KernelSHAP otimizado).

---

## 11. Originalidade do Projeto
Não há consenso sobre a robustez do TabPFN: estudos recentes apontam que ele ignora colunas aleatórias ou ruídos isolados, mas estudos de 2026 demonstram degradação quando sinais espúrios se emaranham ao sinal preditivo (*spurious intra-variable signal routing*). O artigo original do TabPFN usou SHAP apenas de forma ilustrativa. A proposta inova ao fundir:
1. **SCMs controlados** sobre covariáveis reais do censo;
2. **Avaliação quantitativa de alinhamento causal** das explicações de SHAP (se refletem o DAG real);
3. **Mapeamento do ponto de quebra** onde o TabPFN cede ao shortcut learning.

---

## 12. Impacto Esperado
Evidenciar em quais regimes os modelos fundacionais tabulares utilizam mecanismos robustos vs. atalhos espúrios, auxiliando na adoção responsável de modelos fundacionais em domínios críticos (Saúde, Políticas Públicas, Crédito) onde shifts populacionais ocorrem frequentemente.
