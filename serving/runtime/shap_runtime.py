import json
import sys
from pathlib import Path

import joblib
import numpy as np

from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput

sys.path.insert(0, "/app")
import explainer  # noqa: F401


class ShapExplainerRuntime(MLModel):
    async def load(self):
        uri = self._settings.parameters.uri
        artifact_path = Path(uri)
        if not artifact_path.is_absolute():
            artifact_path = self._find_model_dir() / uri
        self._explainer = joblib.load(artifact_path)
        self.ready = True
        return self.ready

    def _find_model_dir(self) -> Path:
        for settings_path in Path("/models").rglob("model-settings.json"):
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            if settings.get("name") == self.name:
                return settings_path.parent
        raise FileNotFoundError(f"Could not find model directory for {self.name}")

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        inp = payload.inputs[0]
        x = np.array(inp.data, dtype=np.float64).reshape(inp.shape)
        contributions = self._explainer.explain(x)
        feature_names = self._explainer.feature_names
        shap_array = np.array(
            [[row[name] for name in feature_names] for row in contributions],
            dtype=np.float64,
        )

        return InferenceResponse(
            model_name=self.name,
            outputs=[
                ResponseOutput(
                    name="shap_values",
                    shape=list(shap_array.shape),
                    datatype="FP64",
                    data=shap_array.flatten().tolist(),
                    parameters={"content_type": "np"},
                )
            ],
        )
