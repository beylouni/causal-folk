# CausalFoundations

Investigating causal alignment and shortcut learning in Tabular Foundation Models (TabPFN) using **Folktables (ACS PUMS)** and Structural Causal Models (SCMs).

---

## 👥 Integrantes
- **Guilherme Torresan Bazzo** — 00287687
- **João Enrique Rego Cairuga** — 00549998
- **Kamille Konarzewski Pimentel** — 00342607
- **Luciano Luz Beylouni Farias** — 00323903
- **Sofia Batista Sartori** — 00333201

---

## 🎯 Pergunta Fundamental & Hipótese

> **Pergunta:** Por que os modelos fundacionais tabulares conseguem fazer boas previsões em conjuntos de dados que nunca viram?  
> **Hipótese:** Modelos fundacionais tabulares generalizam quando aprendem regularidades estatísticas estáveis, mas perdem desempenho quando dependem de correlações específicas (atalhos/spurious correlations) que mudam em novos cenários.

Para ler a proposta completa do projeto com referências bibliográficas, consulte [docs/PROPOSAL.md](docs/PROPOSAL.md).

---

## 📁 Estrutura do Projeto

```text
causal-folk/
├── pyproject.toml              # Dependências e metadados gerenciados com uv
├── uv.lock                     # Lockfile reproduzível do uv
├── README.md                   # Instruções e visão geral
├── docs/
│   └── PROPOSAL.md             # Proposta submetida e referências
├── data/                       # Diretório local para os dados do Censo (ignorado no git)
│   ├── raw/
│   └── processed/
├── notebooks/                  # Notebooks Jupyter para prototipagem
│   └── 01_folktables_exploration.ipynb
├── src/
│   └── causal_foundations/     # Pacote Python com código reutilizável
│       ├── __init__.py
│       ├── data.py             # Funções de carga do Folktables (ACS PUMS)
│       ├── scm.py              # Helpers para injeção de SCM e atalhos espúrios
│       └── models.py           # Wrappers simples para TabPFN e baselines
└── tests/
    └── test_basic.py           # Testes básicos de sanidade
```

---

## 🚀 Como Começar

Este projeto utiliza o [uv](https://docs.astral.sh/uv/) como gerenciador de dependências e ambientes virtuais.

### 1. Instalar o uv (caso ainda não tenha)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# ou no macOS:
brew install uv
```

### 2. Sincronizar o ambiente virtual
Na raiz do projeto, execute:
```bash
uv sync
```
Isso criará o `.venv` e instalará automaticamente todas as dependências (`folktables`, `tabpfn`, `shap`, `scikit-learn`, `torch`, `jupyterlab`, etc.).

### 3. Rodar os testes básicos
```bash
uv run pytest
```

### 4. Abrir o Jupyter Lab
Para começar a explorar os dados e prototipar os experimentos em conjunto:
```bash
uv run jupyter lab
```
Abra o notebook `notebooks/01_folktables_exploration.ipynb`.
