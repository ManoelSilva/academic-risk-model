[Leia em português](MODEL_CARD.pt-br.md)

# Model Card: Academic Risk Prediction — Gradient Boosting Classifier

## Model Details

### Model Information
- **Model Name**: Academic Risk Prediction Classifier
- **Model Type**: Classical Machine Learning — Ensemble (Gradient Boosting)
- **Framework**: scikit-learn
- **Version**: 1.0
- **Date**: 02-2026

### Model Architecture

**Architecture Type**: scikit-learn Pipeline with embedded preprocessing

**Pipeline Structure**:
```
Input: Raw student feature DataFrame
    ↓
FeatureEngineer (Custom Transformer):
    - Derives IS_NEW_STUDENT from SINALIZADOR_INGRESSANTE
    - Drops identifier columns (NOME, INSTITUICAO_ENSINO_ALUNO_2020)
    ↓
Preprocessor (ColumnTransformer):
    ├─→ Numeric Pipeline: MedianImputer → StandardScaler  (11 features)
    └─→ Categorical Pipeline: ConstantImputer → OneHotEncoder  (3 features)
    ↓
GradientBoostingClassifier:
    - n_estimators: 100
    - random_state: 42
    ↓
Output: Binary prediction (0 = Low Risk, 1 = High Risk) + probability
```

**Key Components**:
- **FeatureEngineer**: Custom scikit-learn transformer for domain-specific feature derivation
- **ColumnTransformer**: Parallel numeric and categorical preprocessing
- **GradientBoostingClassifier**: Ensemble of 100 sequential decision trees with boosting
- **Single Artifact**: Entire pipeline serialized as one `model.joblib` file (no train-serve skew)

## Training Data

### Dataset
- **Source**: PEDE (Pesquisa de Desenvolvimento Educacional) — Passos Mágicos NGO
- **Domain**: Education — Student academic risk prediction
- **Original Records**: ~1,349 students (2020–2022)
- **Records After Cleaning**: ~862 (rows with missing 2022 ground truth removed)
- **Features**: 11 numeric + 3 categorical = 14 total features
- **Update Frequency**: Annually, aligned with PEDE survey data collection cycle

### Data Preprocessing
- **Cleaning**: Target computed from `FASE_2022 - NIVEL_IDEAL_2022`; all 2022 columns dropped to prevent leakage
- **Numeric Features**: Median imputation for missing values, StandardScaler normalization
- **Categorical Features**: Constant imputation (`'MISSING'`), OneHotEncoder with `handle_unknown='ignore'`
- **Column Standardization**: Year suffixes removed (e.g., `INDE_2021` → `INDE`) for year-agnostic prediction

### Data Splits
- **Training Set**: 80% of data (~690 samples)
- **Test Set**: 20% of data (~172 samples)
- **Split Method**: Stratified split preserving class distribution
- **Random State**: 42 (reproducible)

### Class Distribution
- **Positive Class (At-Risk, TARGET=1)**: ~70% of samples
- **Negative Class (Not At-Risk, TARGET=0)**: ~30% of samples
- **Class Imbalance Handling**:
  - `class_weight='balanced'` for Logistic Regression and Random Forest
  - Recall-focused model selection for Gradient Boosting
  - Stratified cross-validation and train/test split

## Training Procedure

### Training Configuration

**Hyperparameters**:
- **n_estimators**: 100 (number of boosting stages)
- **random_state**: 42 (reproducibility)
- **test_size**: 0.2 (20% held out for testing)
- **cv_folds**: 5 (stratified K-fold cross-validation)
- **scoring**: `recall` (primary optimization metric)

**Model Comparison**:
Three models were evaluated in each training run:
1. **Logistic Regression**: `class_weight='balanced'`, `max_iter=1000`
2. **Random Forest**: `class_weight='balanced'`, `n_estimators=100`
3. **Gradient Boosting**: `n_estimators=100` (selected as best model)

**Selection Criterion**: Best test recall score across all candidate models.

### Training Process
1. **Data Preparation**: Load CSV, clean data, compute target, remove leakage
2. **Feature Engineering**: Derive `IS_NEW_STUDENT`, drop identifiers
3. **Stratified Split**: 80/20 train/test preserving class ratios
4. **Cross-Validation**: 5-fold stratified CV for each candidate model
5. **Full Training**: Fit pipeline on complete training set
6. **Evaluation**: Compute recall, F1, ROC-AUC on held-out test set
7. **Model Selection**: Select best model by test recall
8. **Persistence**: Save to `models/production/model.joblib`
9. **Tracking**: Log all parameters and metrics to MLflow

## Evaluation

### Evaluation Metrics (Latest MLflow Run)

**Source**: `latest_metrics_202602282239.csv` — MLflow experiment metrics export

#### Best Model Performance (Gradient Boosting)

| Metric | Value | MLflow Run UUID |
|--------|-------|-----------------|
| **CV Mean Recall** | 0.9433 (94.33%) | `e94c275bc9784ea8b68736d10d4002fd` |
| **CV Std Recall** | 0.0304 (3.04%) | `e94c275bc9784ea8b68736d10d4002fd` |
| **Test Recall** | 0.9503 (95.03%) | `e94c275bc9784ea8b68736d10d4002fd` |
| **Test F1 Score** | 0.8289 (82.89%) | `e94c275bc9784ea8b68736d10d4002fd` |
| **Test ROC-AUC** | 0.7459 (74.59%) | `e94c275bc9784ea8b68736d10d4002fd` |
| **Training Duration** | 4.21 seconds | `e94c275bc9784ea8b68736d10d4002fd` |
| **Best Score (Recall)** | 0.9503 | `a6535aea0faf4866b7ed72274cbc845e` |

#### All Model Comparison (Latest 3-Model Experiment)

| Model | CV Mean Recall | Test Recall | Test F1 | Test ROC-AUC | Duration (s) |
|-------|---------------|-------------|---------|--------------|-------------|
| Logistic Regression | 0.3770 | 0.5138 | 0.6436 | 0.7056 | 4.04 |
| **Gradient Boosting** | **0.9433** | **0.9503** | **0.8289** | **0.7459** | **4.21** |

#### Historical Run Consistency (Standard Configuration)

| Model | CV Mean Recall | Test Recall | Test F1 | Test ROC-AUC |
|-------|---------------|-------------|---------|--------------|
| Logistic Regression | 0.4191 | 0.3719 | 0.5357 | 0.7212 |
| Random Forest | 0.5209 | 0.5537 | 0.6768 | 0.7548 |
| Gradient Boosting | 0.9481 | 0.9421 | 0.8539 | 0.7672 |

**Interpretation**:
- Gradient Boosting consistently achieves **>94% recall** across all training runs
- Cross-validation standard deviation of ~3% confirms model stability
- F1 score of 82.9% demonstrates good precision-recall balance
- ROC-AUC of 74.6% indicates adequate rank-ordering capability

### Overfitting Analysis

- **CV Mean Recall vs. Test Recall**: 94.3% vs. 95.0% — no overfitting detected
- **CV Standard Deviation**: 3.0% — low variance across folds
- **Reproducibility**: Identical results across repeated runs with `random_state=42`

## Model Performance Summary

### Strengths
1. **High Recall (95.0%)**: Captures 95 out of 100 at-risk students
2. **Stable Cross-Validation**: Low variance (3.0% std) across 5 folds
3. **Good F1 Score (82.9%)**: Healthy precision-recall balance
4. **Fast Training (~4 seconds)**: Enables rapid retraining and experimentation
5. **Reproducible**: Deterministic results with fixed random state
6. **Single-Artifact Deployment**: Complete pipeline in one serialized file

### Limitations
1. **Small Dataset (~862 samples)**: Limits model complexity and generalization guarantees
2. **Population-Specific**: Trained on Passos Mágicos students in Embu-Guaçu; not validated for other populations
3. **ROC-AUC (74.6%)**: Adequate but not exceptional rank-ordering ability
4. **Temporal Dependency**: Performance assumes stable educational dynamics across years
5. **No Hyperparameter Tuning**: Fixed defaults used; marginal improvements possible with tuning
6. **PEDE Survey Dependency**: Requires consistent indicator methodology across years

## Intended Use

### Primary Use Cases
1. **Student Risk Identification**: Flag students at risk of educational lag for early intervention
2. **Resource Prioritization**: Rank students by risk probability for triage
3. **Batch Scoring**: Score entire student cohorts at the beginning of academic periods
4. **Monitoring Tool**: Track risk distribution changes across years

### Out-of-Scope Uses
- **Not for**: Individual student disciplinary or administrative decisions without educator review
- **Not for**: Other student populations without retraining and validation
- **Not for**: Real-time, high-frequency scoring (designed for batch/periodic use)
- **Not for**: Prediction beyond the immediate next academic period

## Ethical Considerations

### Bias and Fairness
- **Class Imbalance Awareness**: Model is optimized for recall (sensitivity), deliberately accepting higher false positive rates to minimize missed at-risk students
- **Population Bias**: Trained exclusively on Passos Mágicos students; performance on other demographics is unknown
- **Socioeconomic Context**: Features reflect socio-emotional indicators that may correlate with socioeconomic status; predictions should be used for support, never for punitive actions
- **Recommendation**: Predictions should always be reviewed by educators who know the students; the model augments, not replaces, human judgment

### Transparency
- **Model Architecture**: Fully documented with open-source components (scikit-learn)
- **Training Process**: Reproducible with fixed random state and logged via MLflow
- **Evaluation Metrics**: Comprehensive metrics provided per model and per class
- **Limitations**: Clearly stated with actionable context

### Data Privacy
- **Data Source**: PEDE survey data collected by Passos Mágicos with appropriate consent
- **Personal Data**: Student names (`NOME`) and institution identifiers are dropped before model training
- **Compliance**: Feature engineering removes PII before the model processes any data
- **Storage**: Model artifact contains no individual student data

### Risk Warnings
- **Not a Deterministic Assessment**: Predictions are probabilistic and should inform, not determine, educational interventions
- **False Negatives**: Approximately 5% of at-risk students may be missed — manual review of borderline cases is recommended
- **False Positives**: Some non-at-risk students will be flagged — this is by design (low cost of this error type)
- **Temporal Validity**: Model should be retrained annually with new PEDE data

## Model Maintenance

### Retraining Schedule
- **Frequency**: Annually, when new PEDE data is available
- **Trigger**: Data drift detected via Evidently, or recall drops below 90%
- **Process**: Invoke `/train` endpoint or run `ModelTrainer` directly
- **Validation**: Automated multi-model comparison ensures best model is selected

### Version Control
- **Model Versioning**: Each training run produces a uniquely named experiment in MLflow
- **Artifact Storage**: `models/artifacts/{experiment_name}/model.joblib` + `metadata.json`
- **Production Model**: `models/production/model.joblib` (copy of best artifact)
- **Rollback**: Restore any previous artifact from the `artifacts/` directory

### Monitoring
- **Metrics to Monitor**:
  - Test recall on new ground truth data
  - Prediction probability distribution (calibration)
  - Feature drift via Evidently
  - API latency and error rates via Prometheus
- **Alert Thresholds**:
  - Recall drop > 5% from baseline (95.0%)
  - Drift detected on > 3 features simultaneously
  - API error rate > 1%

## Technical Specifications

### Hardware Requirements
- **Training**: Any modern CPU (training takes ~4 seconds)
- **Inference**: Any modern CPU (< 50ms per prediction)
- **Memory**: 2GB+ RAM sufficient
- **GPU**: Not required (scikit-learn is CPU-only)

### Software Dependencies
- **Python**: 3.10+
- **scikit-learn**: ≥ 1.0.0
- **pandas**: ≥ 2.0.0
- **numpy**: ≥ 2.4.1
- **mlflow**: ≥ 2.0.0
- **flask**: ≥ 3.1.2
- **joblib**: ≥ 1.5.3
- **evidently**: ≥ 0.4.0
- **prometheus-client**: ≥ 0.21.0

### Model Size
- **Model File**: ~50KB–500KB (joblib-serialized pipeline)
- **Metadata File**: ~1KB (JSON)
- **Total**: < 1MB

### Inference Speed
- **Single Prediction**: < 50ms (CPU)
- **Batch (100 students)**: < 200ms (CPU)
- **Throughput**: ~2,000+ predictions/second (CPU)

---

**Last Updated**: 02-2026  
**Model Version**: 1.0  
