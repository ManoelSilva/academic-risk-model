# Academic Risk Prediction Model

[![Portuguese Version](https://img.shields.io/badge/Language-Portuguese-green)](README-pt.md)

This project implements an end-to-end Machine Learning solution to predict the risk of school delay (educational lag) for students supported by the NGO **Passos Mágicos**.

## 🚀 Project Overview

The system predicts whether a student is at risk of falling behind in their education based on socio-emotional indicators (IAN, IDA, IEG, etc.) and academic performance (INDE). It is built with a focus on MLOps best practices, modularity, and production readiness.

### Key Features
- **Data Preprocessing**: Robust cleaning, missing value imputation, and feature engineering.
- **Model Training**: Automated training pipeline with MLflow tracking.
- **Evaluation**: Standardized metrics (Recall, ROC-AUC) focused on social impact (minimizing False Negatives).
- **API**: Flask-based REST API for inference and training orchestration.
- **Dockerized**: Production-ready containerization.

## 🚀 Deployment

### 1. Provision Infrastructure
Before deploying via GitHub Actions, you must provision the AWS infrastructure using Terraform.

```bash
cd terraform
terraform init
terraform apply
```

After the `apply` command finishes, it will output the `public_ip`. You need this IP for the GitHub Secrets.

### 2. Configure GitHub Secrets
Go to your repository settings -> Secrets and variables -> Actions, and add the following secrets:

- `EC2_HOST`: The `public_ip` output from Terraform.
- `EC2_USER`: `ec2-user`
- `EC2_SSH_KEY`: The private key content of your `~/.ssh/id_rsa` (or the key you used in Terraform).

### 3. Automatic Deployment
Pushing to the `main` branch will trigger the **Deploy to AWS** workflow, which:
1. SSHs into the EC2 instance.
2. Pulls the latest code.
3. Rebuilds and restarts the Docker containers.

---

## 🏗 Architecture

The project is structured into modular components:

```
src/
├── api/             # API entry points and routing
├── evaluation/      # Metric calculation logic
├── features/        # Feature engineering logic
├── preprocessing/   # Data cleaning and scikit-learn pipelines
├── training/        # Training orchestration
└── utils/           # Helper utilities
```

---

## 🛠 Installation & Setup

### Prerequisites
- Python 3.10+
- Docker (optional)

### Local Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/academic-risk-model.git
    cd academic-risk-model
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the API**:
    ```bash
    python src/api/main.py
    ```

---

## 🐳 Docker Deployment

### 1. Build the Image
Ensure you have a trained model in `models/production/` (or train one inside the container later).

```bash
docker build -t academic-risk-api .
```

### 2. Run the Container
```bash
docker run -p 5000:5000 academic-risk-api
```

The API will be available at `http://localhost:5000`.

---

## 📡 API Usage

### Interactive API Documentation (Swagger)
```http
GET /docs
```

OpenAPI specification file:
```http
GET /swagger.yml
```

### Health Check
```http
GET /health
```

### Train Model
Triggers the training pipeline on the default dataset, with optional custom training parameters.
```http
POST /train
```

Example with custom parameters:
```http
POST /train
Content-Type: application/json

{
  "data_path": "data/raw/PEDE_PASSOS_DATASET_FIAP.csv",
  "training_params": {
    "test_size": 0.3,
    "random_state": 77,
    "cv_folds": 3,
    "scoring": "roc_auc",
    "class_weight": "balanced",
    "models_to_run": ["Logistic_Regression", "Random_Forest"]
  }
}
```

Supported `training_params`:
- `test_size` (float, range: 0.1 to 0.5)
- `random_state` (int)
- `cv_folds` (int, range: 2 to 10)
- `scoring` (`recall`, `roc_auc`, `f1`)
- `class_weight` (`balanced` or `null`)
- `models_to_run` (non-empty list with `Logistic_Regression`, `Random_Forest`, `Gradient_Boosting`)

The training response now includes:
- `experiment_name`: generated using selected parameter names and values
- `training_config`: normalized configuration used in training

### Predict Risk
Send student data to get a risk assessment.
**Note:** Feature names are standardized (e.g., `INDE` instead of `INDE_2021`).

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

**Response:**
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

## 📅 Development Phases

This project was developed in sequential phases:

1.  **Project Understanding & EDA**: Analyzed dataset and defined target variable (`DEFASAGEM_2022`).
2.  **Feature Engineering**: Created pipelines for cleaning and preprocessing.
3.  **Model Training**: Selected Gradient Boosting (Recall ~93%) as the best model.
4.  **Model Persistence**: Implemented artifact versioning and production promotion.
5.  **Modularization**: Refactored code into clean architecture (`src/api`, `src/training`, etc.).
6.  **API Development**: Implemented robust `/predict` endpoint with input validation.
7.  **Dockerization**: Containerized the application for deployment.

---

[![Portuguese Version](https://img.shields.io/badge/Language-Portuguese-green)](README-pt.md)
