from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.schemas.model import (
    BatchInferenceRequest,
    BatchInferenceResponse,
    BatchPrediction,
    ExplanationResponse,
    InferenceRequest,
    InferenceResponse,
)
from app.services.mlserver_client import MLServerClient
from app.services.registry import Registry

router = APIRouter(prefix="/infer", tags=["infer"])


def get_registry(settings: Settings = Depends(get_settings)) -> Registry:
    return Registry(settings.models_dir)


def get_mlserver(settings: Settings = Depends(get_settings)) -> MLServerClient:
    return MLServerClient(settings.mlserver_url)


def _feature_names(model_name: str, registry: Registry, settings: Settings) -> list[str]:
    schema = registry.get_feature_schema(model_name)
    if schema is not None:
        return [f["name"] for f in schema]
    return settings.feature_names


def ordered_values(features: dict[str, float], feature_names: list[str]) -> list[float]:
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {', '.join(missing)}")
    return [features[name] for name in feature_names]


def _named_outputs(payload: dict) -> dict[str, list]:
    return {output["name"]: output["data"] for output in payload.get("outputs", [])}


@router.post("/{model_name}/batch", response_model=BatchInferenceResponse)
async def infer_batch(
    model_name: str,
    request: BatchInferenceRequest,
    settings: Settings = Depends(get_settings),
    registry: Registry = Depends(get_registry),
    mlserver: MLServerClient = Depends(get_mlserver),
):
    if not request.datapoints:
        raise HTTPException(status_code=400, detail="datapoints list must not be empty")

    try:
        feature_names = _feature_names(model_name, registry, settings)
        timestamps = [dp.timestamp for dp in request.datapoints]
        batch_values = [ordered_values(dp.features, feature_names) for dp in request.datapoints]
        predictor_name, _ = registry.mlserver_names(model_name)
        payload = await mlserver.infer(predictor_name, batch_values)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"mlserver batch inference failed: {exc}") from exc

    outputs = _named_outputs(payload)
    try:
        scores = outputs["anomaly_score"]
        is_anomalies = outputs["is_anomaly"]
    except KeyError as exc:
        raise HTTPException(status_code=502, detail=f"mlserver response missing output: {exc.args[0]}") from exc

    if len(scores) != len(request.datapoints) or len(is_anomalies) != len(request.datapoints):
        raise HTTPException(
            status_code=502,
            detail=f"mlserver returned mismatched predictions for {len(request.datapoints)} datapoints",
        )

    contributions_per_sample: list[dict[str, float] | None] = [None] * len(request.datapoints)
    if request.include_contributions:
        _, explainer_name = registry.mlserver_names(model_name)
        if not (registry.model_dir(model_name) / "explainer" / "explainer.joblib").exists():
            raise HTTPException(
                status_code=400,
                detail=f"model '{model_name}' has no explainer; cannot include contributions",
            )

        try:
            explain_payload = await mlserver.infer(explainer_name, batch_values)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"mlserver batch explanation failed: {exc}") from exc

        try:
            flat_contributions = explain_payload["outputs"][0]["data"]
        except (KeyError, IndexError) as exc:
            raise HTTPException(status_code=502, detail="mlserver response missing contribution output") from exc

        n_features = len(feature_names)
        if len(flat_contributions) != len(request.datapoints) * n_features:
            raise HTTPException(
                status_code=502,
                detail="mlserver returned mismatched contribution shape",
            )

        contributions_per_sample = [
            dict(zip(feature_names, flat_contributions[i * n_features : (i + 1) * n_features]))
            for i in range(len(request.datapoints))
        ]

    predictions = [
        BatchPrediction(
            timestamp=timestamp,
            anomaly_score=score,
            is_anomaly=bool(is_anomaly),
            contributions=contributions,
        )
        for timestamp, score, is_anomaly, contributions in zip(
            timestamps, scores, is_anomalies, contributions_per_sample
        )
    ]
    return BatchInferenceResponse(model_name=model_name, predictions=predictions)


@router.post("/{model_name}", response_model=InferenceResponse)
async def infer(
    model_name: str,
    request: InferenceRequest,
    settings: Settings = Depends(get_settings),
    registry: Registry = Depends(get_registry),
    mlserver: MLServerClient = Depends(get_mlserver),
):
    try:
        feature_names = _feature_names(model_name, registry, settings)
        predictor_name, _ = registry.mlserver_names(model_name)
        payload = await mlserver.infer(predictor_name, [ordered_values(request.features, feature_names)])
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"mlserver inference failed: {exc}") from exc

    outputs = _named_outputs(payload)
    try:
        score = outputs["anomaly_score"][0]
        is_anomaly = outputs["is_anomaly"][0]
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=502, detail="mlserver response missing prediction output") from exc

    return InferenceResponse(
        model_name=model_name,
        raw_prediction=score,
        is_anomaly=bool(is_anomaly),
    )


@router.post("/{model_name}/explain", response_model=ExplanationResponse)
async def explain(
    model_name: str,
    request: InferenceRequest,
    settings: Settings = Depends(get_settings),
    registry: Registry = Depends(get_registry),
    mlserver: MLServerClient = Depends(get_mlserver),
):
    try:
        feature_names = _feature_names(model_name, registry, settings)
        _, explainer_name = registry.mlserver_names(model_name)
        payload = await mlserver.infer(explainer_name, [ordered_values(request.features, feature_names)])
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"mlserver explanation failed: {exc}") from exc

    values = payload["outputs"][0]["data"]
    return ExplanationResponse(
        model_name=model_name,
        shap_values=dict(zip(feature_names, values)),
    )
