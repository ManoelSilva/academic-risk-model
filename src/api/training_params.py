DEFAULT_TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "scoring": "recall",
    "class_weight": "balanced",
    "models_to_run": ["Logistic_Regression", "Random_Forest", "Gradient_Boosting"],
}

ALLOWED_SCORING = {"recall", "roc_auc", "f1"}
ALLOWED_MODELS = {"Logistic_Regression", "Random_Forest", "Gradient_Boosting", "CatBoost"}


def parse_training_params(params_payload):
    """
    Validate and normalize custom training parameters received by /train.
    """
    if params_payload is None:
        params_payload = {}
    if not isinstance(params_payload, dict):
        return None, ["training_params must be an object"]

    config = DEFAULT_TRAINING_CONFIG.copy()
    errors = []

    if "test_size" in params_payload:
        try:
            test_size = float(params_payload["test_size"])
            if not 0.1 <= test_size <= 0.5:
                errors.append("test_size must be between 0.1 and 0.5")
            else:
                config["test_size"] = test_size
        except (TypeError, ValueError):
            errors.append("test_size must be a number")

    if "random_state" in params_payload:
        try:
            config["random_state"] = int(params_payload["random_state"])
        except (TypeError, ValueError):
            errors.append("random_state must be an integer")

    if "cv_folds" in params_payload:
        try:
            cv_folds = int(params_payload["cv_folds"])
            if not 2 <= cv_folds <= 10:
                errors.append("cv_folds must be between 2 and 10")
            else:
                config["cv_folds"] = cv_folds
        except (TypeError, ValueError):
            errors.append("cv_folds must be an integer")

    if "scoring" in params_payload:
        scoring = str(params_payload["scoring"]).lower()
        if scoring not in ALLOWED_SCORING:
            errors.append(f"scoring must be one of {sorted(ALLOWED_SCORING)}")
        else:
            config["scoring"] = scoring

    if "class_weight" in params_payload:
        class_weight = params_payload["class_weight"]
        if class_weight not in ["balanced", None]:
            errors.append("class_weight must be 'balanced' or null")
        else:
            config["class_weight"] = class_weight

    if "models_to_run" in params_payload:
        models = params_payload["models_to_run"]
        if not isinstance(models, list) or not models:
            errors.append("models_to_run must be a non-empty list")
        else:
            invalid_models = [m for m in models if m not in ALLOWED_MODELS]
            if invalid_models:
                errors.append(f"models_to_run has invalid models: {invalid_models}")
            else:
                config["models_to_run"] = models

    return config, errors
