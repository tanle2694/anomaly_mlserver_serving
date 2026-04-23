from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.schemas.model import FeatureDef, FeatureSchemaResponse, ModelsResponse
from app.services.mlserver_client import MLServerClient
from app.services.registry import Registry

router = APIRouter(prefix="/models", tags=["models"])


def get_registry(settings: Settings = Depends(get_settings)) -> Registry:
    return Registry(settings.models_dir)


def get_mlserver(settings: Settings = Depends(get_settings)) -> MLServerClient:
    return MLServerClient(settings.mlserver_url)


@router.get("", response_model=ModelsResponse)
async def list_models(
    registry: Registry = Depends(get_registry),
    mlserver: MLServerClient = Depends(get_mlserver),
):
    try:
        loaded_names = await mlserver.repository_index()
    except Exception:
        loaded_names = set()
    return ModelsResponse(models=registry.list_models(loaded_names))


@router.post("", response_model=ModelsResponse)
async def upload_model(
    model_name: str = Form(...),
    predictor_file: UploadFile = File(...),
    predictor_settings: UploadFile = File(...),
    explainer_file: UploadFile = File(...),
    explainer_settings: UploadFile = File(...),
    feature_schema: str | None = Form(None),
    registry: Registry = Depends(get_registry),
    mlserver: MLServerClient = Depends(get_mlserver),
):
    try:
        await registry.register_upload(model_name, predictor_file, predictor_settings, explainer_file, explainer_settings)
        registry.save_feature_schema(model_name, feature_schema)
        predictor_name, explainer_name = registry.mlserver_names(model_name)
        await mlserver.load(predictor_name)
        await mlserver.load(explainer_name)
        loaded_names = await mlserver.repository_index()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"mlserver load failed: {exc}") from exc
    return ModelsResponse(models=registry.list_models(loaded_names))


@router.get("/{model_name}/features", response_model=FeatureSchemaResponse)
async def get_model_features(
    model_name: str,
    settings: Settings = Depends(get_settings),
    registry: Registry = Depends(get_registry),
):
    try:
        schema = registry.get_feature_schema(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if schema is not None:
        return FeatureSchemaResponse(features=[FeatureDef(**f) for f in schema])
    return FeatureSchemaResponse(
        features=[FeatureDef(name=name, default=0.1) for name in settings.feature_names]
    )


@router.delete("/{model_name}", response_model=ModelsResponse)
async def delete_model(
    model_name: str,
    registry: Registry = Depends(get_registry),
    mlserver: MLServerClient = Depends(get_mlserver),
):
    try:
        predictor_name, explainer_name = registry.mlserver_names(model_name)
        await mlserver.unload(predictor_name)
        await mlserver.unload(explainer_name)
        registry.delete_model(model_name)
        loaded_names = await mlserver.repository_index()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        loaded_names = set()
    return ModelsResponse(models=registry.list_models(loaded_names))
