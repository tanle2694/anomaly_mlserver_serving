import json
import sys
from pathlib import Path

import joblib
import numpy as np

from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput

sys.path.insert(0, "/app")
import detector  # noqa: F401


class PredictorRuntime(MLModel):
    async def load(self):
        uri = self._settings.parameters.uri
        artifact_path = Path(uri)
        if not artifact_path.is_absolute():
            artifact_path = self._find_model_dir() / uri
        self._detector = joblib.load(artifact_path)
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
        result = self._detector.predict(x)
        scores = result[:, 0].astype(np.float64)
        labels = result[:, 1].astype(np.int64)

        return InferenceResponse(
            model_name=self.name,
            outputs=[
                ResponseOutput(
                    name="anomaly_score",
                    shape=[len(scores)],
                    datatype="FP64",
                    data=scores.tolist(),
                ),
                ResponseOutput(
                    name="is_anomaly",
                    shape=[len(labels)],
                    datatype="INT64",
                    data=labels.tolist(),
                ),
            ],
        )
