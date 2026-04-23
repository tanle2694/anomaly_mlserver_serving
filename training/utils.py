import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn

from detector import AnomalyDetector
from explainer import ShapExplainer

# =========================
# CONFIG
# =========================
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# =========================
# DATA
# =========================
FEATURE_DISTRIBUTIONS = {
    "temperature": (70, 5,   100, 10),
    "pressure":    (30, 2,   50,  5),
    "vibration":   (0.5, 0.1, 1.5, 0.3),
    "humidity":    (40, 10,  10,  5),
}
_DEFAULT_NORMAL_MU, _DEFAULT_NORMAL_SIGMA = 0.0, 1.0
_DEFAULT_ANOMALY_MU, _DEFAULT_ANOMALY_SIGMA = 3.0, 1.0


def generate_sample_data(
    features: list[str],
    n_normal: int = 2000,
    n_anomalies: int = 50,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def dist(feat, kind):
        if feat in FEATURE_DISTRIBUTIONS:
            mu, sigma = FEATURE_DISTRIBUTIONS[feat][0:2] if kind == "normal" else FEATURE_DISTRIBUTIONS[feat][2:4]
        else:
            mu, sigma = (_DEFAULT_NORMAL_MU, _DEFAULT_NORMAL_SIGMA) if kind == "normal" else (_DEFAULT_ANOMALY_MU, _DEFAULT_ANOMALY_SIGMA)
        return mu, sigma

    normal = pd.DataFrame({f: rng.normal(*dist(f, "normal"), n_normal) for f in features})
    anomalies = pd.DataFrame({f: rng.normal(*dist(f, "anomaly"), n_anomalies) for f in features})

    df = pd.concat([normal, anomalies], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Generated {len(df)} rows ({n_normal} normal + {n_anomalies} anomalies) for features: {features}")
    return df


def load_data(csv_path: str, features: list[str]) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[features].dropna()
    print(f"Loaded {len(df)} rows from {path}")
    return df


# =========================
# TRAIN
# =========================
def train(pipeline, df, features: list[str]):
    X = df[features].values
    return AnomalyDetector(pipeline).fit(X)


# =========================
# EVALUATE
# =========================
def evaluate(detector, df, features: list[str]):
    X = df[features].values

    result = detector.predict(X)
    scores = result[:, 0]
    labels = result[:, 1].astype(np.int64)

    df["anomaly_score"] = scores
    df["is_anomaly"] = labels.astype(bool)

    print("\nAnomaly distribution:")
    print(df["is_anomaly"].value_counts())

    return df


# =========================
# SHAP EXPLAINER
# =========================
def build_shap_explainer(pipeline, df, features: list[str]):
    X = df[features].values
    idx = np.random.choice(len(X), size=min(100, len(X)), replace=False)
    background = X[idx]
    return ShapExplainer(pipeline, background, features)


# =========================
# SAVE
# =========================
def _to_snake_case(name: str) -> str:
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _rounded_float(value) -> float:
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, 4)


def build_metadata(
    model_name: str,
    features: list[str],
    df: pd.DataFrame,
    data_source: str,
    pipeline,
    anomaly_rule: dict,
) -> dict:
    model = pipeline.named_steps["model"]
    scores = df["anomaly_score"]
    anomalies = df["is_anomaly"].astype(bool)
    n_samples = int(len(df))
    n_anomalies = int(anomalies.sum())
    sample_row = df[features].iloc[0]

    return {
        "model_name": model_name,
        "version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "features": features,
        "sample_input": {feature: _rounded_float(sample_row[feature]) for feature in features},
        "training": {
            "n_samples": n_samples,
            "data_source": data_source,
            "algorithm": _to_snake_case(type(model).__name__),
            "hyperparameters": _json_safe(model.get_params()),
            "random_seed": RANDOM_SEED,
        },
        "evaluation": {
            "n_anomalies": n_anomalies,
            "n_normal": n_samples - n_anomalies,
            "anomaly_rate": _rounded_float(n_anomalies / n_samples) if n_samples else 0.0,
            "score_stats": {
                "min": _rounded_float(scores.min()),
                "max": _rounded_float(scores.max()),
                "mean": _rounded_float(scores.mean()),
                "std": _rounded_float(scores.std()),
                "p5": _rounded_float(np.percentile(scores, 5)),
                "p25": _rounded_float(np.percentile(scores, 25)),
                "p50": _rounded_float(np.percentile(scores, 50)),
                "p75": _rounded_float(np.percentile(scores, 75)),
                "p95": _rounded_float(np.percentile(scores, 95)),
            },
            "anomaly_rule": _json_safe(anomaly_rule),
        },
        "artifacts": {
            "predictor": {
                "model": "predictor/predictor.joblib",
                "model_settings": "predictor/model-settings.json",
            },
            "explainer": {
                "model": "explainer/explainer.joblib",
                "model_settings": "explainer/model-settings.json",
            },
        },
        "runtime": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "sklearn": sklearn.__version__,
            "shap": shap.__version__,
        },
    }


def save(detector, explainer, model_name, features: list[str], metadata: dict):
    predictor_dir = Path("out") / model_name / "predictor"
    explainer_dir = Path("out") / model_name / "explainer"
    predictor_dir.mkdir(parents=True, exist_ok=True)
    explainer_dir.mkdir(parents=True, exist_ok=True)

    predictor_path = predictor_dir / "predictor.joblib"
    explainer_path = explainer_dir / "explainer.joblib"

    joblib.dump(detector, predictor_path)
    joblib.dump(explainer, explainer_path)

    n_features = len(features)

    predictor_settings = {
        "name": f"{model_name}__predictor",
        "implementation": "predictor_runtime.PredictorRuntime",
        "parameters": {"uri": "./predictor.joblib", "version": "v1"},
        "inputs": [{"name": "input-0", "datatype": "FP64", "shape": [-1, n_features]}],
        "outputs": [
            {"name": "anomaly_score", "datatype": "FP64", "shape": [-1]},
            {"name": "is_anomaly", "datatype": "INT64", "shape": [-1]},
        ],
    }
    explainer_settings = {
        "name": f"{model_name}__explainer",
        "implementation": "shap_runtime.ShapExplainerRuntime",
        "parameters": {"uri": "./explainer.joblib", "version": "v1"},
        "inputs": [{"name": "input-0", "datatype": "FP64", "shape": [-1, n_features]}],
        "outputs": [{"name": "shap_values", "datatype": "FP64", "shape": [-1, n_features]}],
    }

    for settings, path in [
        (predictor_settings, predictor_dir / "model-settings.json"),
        (explainer_settings, explainer_dir / "model-settings.json"),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")

    metadata_path = Path("out") / model_name / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
        f.write("\n")

    print("\nSaved:")
    print(f"- {predictor_path}")
    print(f"- {predictor_dir / 'model-settings.json'}")
    print(f"- {explainer_path}")
    print(f"- {explainer_dir / 'model-settings.json'}")
    print(f"- {metadata_path}")


# =========================
# TEST
# =========================
def test(detector, explainer, df, features: list[str]):
    X_input = df[features].iloc[[0]].values

    print("\nTest prediction (first CSV row):")
    for f, v in zip(features, X_input[0]):
        print(f"  {f}: {v}")

    result = detector.predict(X_input)
    score = result[0, 0]
    is_anomaly = bool(result[0, 1])

    print("Score:", score)
    print("Is anomaly:", is_anomaly)

    contributions = explainer.explain(X_input)

    print("\nSHAP contributions:")
    for f, val in contributions[0].items():
        print(f"  {f}: {val:.4f}")
