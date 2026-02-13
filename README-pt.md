# Modelo de Previsão de Risco Acadêmico

[![English Version](https://img.shields.io/badge/Language-English-blue)](README.md)

Este projeto implementa uma solução completa de Machine Learning para prever o risco de defasagem escolar para alunos apoiados pela ONG **Passos Mágicos**.

## 🚀 Visão Geral do Projeto

O sistema prevê se um aluno corre risco de atraso escolar com base em indicadores socioemocionais (IAN, IDA, IEG, etc.) e desempenho acadêmico (INDE). Ele foi construído com foco nas melhores práticas de MLOps, modularidade e prontidão para produção.

### Principais Funcionalidades
- **Pré-processamento de Dados**: Limpeza robusta, imputação de valores ausentes e engenharia de features.
- **Treinamento de Modelo**: Pipeline de treinamento automatizado com rastreamento via MLflow.
- **Avaliação**: Métricas padronizadas (Recall, ROC-AUC) focadas no impacto social (minimizar Falsos Negativos).
- **API**: API REST baseada em Flask para inferência e orquestração de treinamento.
- **Docker**: Containerização pronta para produção.

---

## 🏗 Arquitetura

O projeto está estruturado em componentes modulares:

```
src/
├── api/             # Pontos de entrada da API e roteamento
├── evaluation/      # Lógica de cálculo de métricas
├── features/        # Lógica de engenharia de features
├── preprocessing/   # Limpeza de dados e pipelines scikit-learn
├── training/        # Orquestração do treinamento
└── utils/           # Utilitários auxiliares
```

---

## 🛠 Instalação e Configuração

### Pré-requisitos
- Python 3.10+
- Docker (opcional)

### Configuração Local

1.  **Clone o repositório**:
    ```bash
    git clone https://github.com/seu-usuario/academic-risk-model.git
    cd academic-risk-model
    ```

2.  **Crie um ambiente virtual**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a API**:
    ```bash
    python src/api/main.py
    ```

---

## 🐳 Implantação com Docker

### 1. Construir a Imagem
Certifique-se de ter um modelo treinado em `models/production/` (ou treine um dentro do container posteriormente).

```bash
docker build -t academic-risk-api .
```

### 2. Executar o Container
```bash
docker run -p 5000:5000 academic-risk-api
```

A API estará disponível em `http://localhost:5000`.

---

## 📡 Uso da API

### Verificação de Saúde (Health Check)
```http
GET /health
```

### Treinar Modelo
Aciona o pipeline de treinamento no conjunto de dados padrão.
```http
POST /train
```

### Prever Risco
Envie dados do aluno para obter uma avaliação de risco.
**Nota:** Os nomes das features são padronizados (ex: `INDE` em vez de `INDE_2021`).

```http
POST /predict
Content-Type: application/json

[
  {
    "INDE": 7.5,
    "IAA": 8.0,
    "IEG": 6.5,
    "IPS": 7.0,
    "IDA": 9.0,
    "IPP": 7.5,
    "IPV": 8.5,
    "IAN": 6.0,
    "DEFASAGEM": 0,
    "IDADE_ALUNO": 12,
    "ANOS_PM": 2,
    "PEDRA": "Ametista",
    "PONTO_VIRADA": "Não",
    "SINALIZADOR_INGRESSANTE": "Não"
  }
]
```

**Resposta:**
```json
{
  "status": "success",
  "count": 1,
  "predictions": [
    {
      "id": 0,
      "risk_prediction": 0,
      "risk_probability": 0.12,
      "risk_label": "Low Risk"
    }
  ]
}
```

---

## 📅 Fases de Desenvolvimento

Este projeto foi desenvolvido em fases sequenciais:

1.  **Entendimento do Projeto e EDA**: Análise do dataset e definição da variável alvo (`DEFASAGEM_2022`).
2.  **Engenharia de Features**: Criação de pipelines para limpeza e pré-processamento.
3.  **Treinamento de Modelo**: Seleção do Gradient Boosting (Recall ~93%) como o melhor modelo.
4.  **Persistência do Modelo**: Implementação de versionamento de artefatos e promoção para produção.
5.  **Modularização**: Refatoração do código em arquitetura limpa (`src/api`, `src/training`, etc.).
6.  **Desenvolvimento da API**: Implementação de endpoint `/predict` robusto com validação de entrada.
7.  **Dockerização**: Containerização da aplicação para implantação.

---

[![English Version](https://img.shields.io/badge/Language-English-blue)](README.md)
