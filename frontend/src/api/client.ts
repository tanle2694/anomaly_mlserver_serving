export type LogicalModel = {
  name: string;
  has_predictor: boolean;
  has_explainer: boolean;
  predictor_loaded: boolean;
  explainer_loaded: boolean;
};

export type InferenceResult = {
  model_name: string;
  raw_prediction: number;
  is_anomaly: boolean;
};

export type ExplanationResult = {
  model_name: string;
  shap_values: Record<string, number>;
};

export type Features = Record<string, number>;

export type FeatureDef = {
  name: string;
  default: number;
};

export type BatchSample = { timestamp: string; features: Features };

export type BatchPrediction = {
  timestamp: string;
  anomaly_score: number;
  is_anomaly: boolean;
};

export type BatchInferenceResult = {
  model_name: string;
  predictions: BatchPrediction[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json() as Promise<T>;
}

export async function listModels(): Promise<LogicalModel[]> {
  const payload = await request<{ models: LogicalModel[] }>("/models");
  return payload.models;
}

export async function uploadModel(
  modelName: string,
  predictorJoblib: File,
  predictorSettings: File,
  explainerJoblib: File,
  explainerSettings: File,
  featureSchema?: FeatureDef[],
): Promise<LogicalModel[]> {
  const form = new FormData();
  form.append("model_name", modelName);
  form.append("predictor_file", predictorJoblib);
  form.append("predictor_settings", predictorSettings);
  form.append("explainer_file", explainerJoblib);
  form.append("explainer_settings", explainerSettings);
  if (featureSchema && featureSchema.length > 0) {
    form.append("feature_schema", JSON.stringify(featureSchema));
  }
  const payload = await request<{ models: LogicalModel[] }>("/models", {
    method: "POST",
    body: form,
  });
  return payload.models;
}

export async function getModelFeatures(modelName: string): Promise<FeatureDef[]> {
  const payload = await request<{ features: FeatureDef[] }>(`/models/${modelName}/features`);
  return payload.features;
}

export async function deleteModel(modelName: string): Promise<LogicalModel[]> {
  const payload = await request<{ models: LogicalModel[] }>(`/models/${modelName}`, {
    method: "DELETE",
  });
  return payload.models;
}

export function infer(modelName: string, features: Features): Promise<InferenceResult> {
  return request<InferenceResult>(`/infer/${modelName}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
}

export function inferBatch(modelName: string, datapoints: BatchSample[]): Promise<BatchInferenceResult> {
  return request<BatchInferenceResult>(`/infer/${modelName}/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ datapoints }),
  });
}

export function explain(modelName: string, features: Features): Promise<ExplanationResult> {
  return request<ExplanationResult>(`/infer/${modelName}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
}
