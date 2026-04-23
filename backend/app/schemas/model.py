from pydantic import BaseModel, Field


class FeatureDef(BaseModel):
    name: str
    default: float


class FeatureSchemaResponse(BaseModel):
    features: list[FeatureDef]


class LogicalModel(BaseModel):
    name: str
    has_predictor: bool
    has_explainer: bool
    predictor_loaded: bool = False
    explainer_loaded: bool = False


class ModelsResponse(BaseModel):
    models: list[LogicalModel]


class InferenceRequest(BaseModel):
    features: dict[str, float] = Field(
        examples=[
            {
                "temperature": 95.0,
                "pressure": 48.0,
                "vibration": 1.2,
                "humidity": 15.0,
            }
        ]
    )


class InferenceResponse(BaseModel):
    model_name: str
    raw_prediction: float | int
    is_anomaly: bool


class InferSample(BaseModel):
    timestamp: str
    features: dict[str, float]


class BatchInferenceRequest(BaseModel):
    datapoints: list[InferSample]
    include_contributions: bool = False


class BatchPrediction(BaseModel):
    timestamp: str
    anomaly_score: float | int
    is_anomaly: bool
    contributions: dict[str, float] | None = None


class BatchInferenceResponse(BaseModel):
    model_name: str
    predictions: list[BatchPrediction]


class ExplanationResponse(BaseModel):
    model_name: str
    shap_values: dict[str, float]
