"""
Prometheus metrics definitions for the Academic Risk Model API.

Exposes inference, training, drift, and system health metrics via /metrics endpoint.
"""

import time
import functools
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
)


MODEL_INFO = Info(
    "academic_risk_model",
    "Model metadata",
)

# --- Inference Metrics ---

PREDICTION_REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["status", "risk_label"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of predicted risk probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

PREDICTIONS_HIGH_RISK = Counter(
    "predictions_high_risk_total",
    "Total high-risk predictions",
)

PREDICTIONS_LOW_RISK = Counter(
    "predictions_low_risk_total",
    "Total low-risk predictions",
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether a production model is currently loaded (1=yes, 0=no)",
)

# --- Training Metrics ---

TRAINING_DURATION = Histogram(
    "training_duration_seconds",
    "Total training job duration in seconds",
    ["model_type"],
    buckets=[10, 30, 60, 120, 300, 600, 1200, 3600],
)

TRAINING_REQUESTS = Counter(
    "training_requests_total",
    "Total training requests",
    ["status"],
)

TRAINING_BEST_SCORE = Gauge(
    "training_best_score",
    "Best model score from latest training run",
    ["metric_name", "model_type"],
)

TRAINING_CV_SCORE = Gauge(
    "training_cv_score",
    "Cross-validation score from latest training run",
    ["stat", "metric_name", "model_type"],
)

# --- Data Drift Metrics ---

DRIFT_CHECK_COUNT = Counter(
    "drift_check_total",
    "Total drift check requests",
    ["status"],
)

DRIFT_DETECTED = Gauge(
    "drift_detected",
    "Whether data drift was detected in the last check (1=yes, 0=no)",
)

DRIFT_CHECK_DURATION = Histogram(
    "drift_check_duration_seconds",
    "Duration of drift check in seconds",
    buckets=[1, 5, 10, 30, 60, 120],
)

# --- API Health Metrics ---

HTTP_REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)


def metrics_response():
    """Generate Prometheus metrics response for the /metrics endpoint."""
    return generate_latest(REGISTRY), 200, {"Content-Type": CONTENT_TYPE_LATEST}
