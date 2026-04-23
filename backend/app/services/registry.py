import json
import re
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile

from app.schemas.model import LogicalModel

MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class Registry:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def validate_model_name(self, model_name: str) -> str:
        if not MODEL_RE.match(model_name):
            raise ValueError(
                "model_name must be 1-64 chars and contain only letters, numbers, dots, underscores, and hyphens"
            )
        return model_name

    def model_dir(self, model_name: str) -> Path:
        safe_name = self.validate_model_name(model_name)
        return self.models_dir / safe_name

    def mlserver_names(self, model_name: str) -> tuple[str, str]:
        safe_name = self.validate_model_name(model_name)
        return f"{safe_name}__predictor", f"{safe_name}__explainer"

    def list_models(self, loaded_names: set[str] | None = None) -> list[LogicalModel]:
        loaded_names = loaded_names or set()
        models = []
        for path in sorted(self.models_dir.iterdir()):
            if not path.is_dir():
                continue
            predictor_name, explainer_name = self.mlserver_names(path.name)
            models.append(
                LogicalModel(
                    name=path.name,
                    has_predictor=(path / "predictor" / "predictor.joblib").exists(),
                    has_explainer=(path / "explainer" / "explainer.joblib").exists(),
                    predictor_loaded=predictor_name in loaded_names,
                    explainer_loaded=explainer_name in loaded_names,
                )
            )
        return models

    async def register_upload(
        self,
        model_name: str,
        predictor_file: UploadFile,
        predictor_settings: UploadFile,
        explainer_file: UploadFile,
        explainer_settings: UploadFile,
    ) -> None:
        model_dir = self.model_dir(model_name)
        predictor_dir = model_dir / "predictor"
        explainer_dir = model_dir / "explainer"
        predictor_dir.mkdir(parents=True, exist_ok=True)
        explainer_dir.mkdir(parents=True, exist_ok=True)

        predictor_name, explainer_name = self.mlserver_names(model_name)
        await self._save_upload(predictor_file, predictor_dir / "predictor.joblib")
        await self._save_settings(predictor_settings, predictor_dir / "model-settings.json", predictor_name)
        await self._save_upload(explainer_file, explainer_dir / "explainer.joblib")
        await self._save_settings(explainer_settings, explainer_dir / "model-settings.json", explainer_name)

    def get_feature_schema(self, model_name: str) -> list[dict] | None:
        schema_path = self.model_dir(model_name) / "feature_schema.json"
        if not schema_path.exists():
            return None
        with open(schema_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("features") or None

    def save_feature_schema(self, model_name: str, feature_schema_json: str | None) -> None:
        if not feature_schema_json:
            return
        try:
            features = json.loads(feature_schema_json)
        except json.JSONDecodeError:
            return
        schema_path = self.model_dir(model_name) / "feature_schema.json"
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump({"features": features}, f, indent=2)
            f.write("\n")

    def delete_model(self, model_name: str) -> None:
        shutil.rmtree(self.model_dir(model_name), ignore_errors=True)

    async def _save_upload(self, upload: UploadFile, destination: Path) -> None:
        with NamedTemporaryFile(delete=False, dir=destination.parent) as tmp:
            tmp_path = Path(tmp.name)
            while chunk := await upload.read(1024 * 1024):
                tmp.write(chunk)
        tmp_path.replace(destination)

    async def _save_settings(self, upload: UploadFile, destination: Path, name_override: str) -> None:
        raw = await upload.read()
        try:
            settings = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {upload.filename}: {exc}") from exc
        settings["name"] = name_override
        if destination.parent.name == "predictor":
            settings["implementation"] = "predictor_runtime.PredictorRuntime"
            settings["outputs"] = [
                {"name": "anomaly_score", "datatype": "FP64", "shape": [-1]},
                {"name": "is_anomaly", "datatype": "INT64", "shape": [-1]},
            ]
        with NamedTemporaryFile(delete=False, dir=destination.parent, mode="w", encoding="utf-8") as tmp:
            tmp_path = Path(tmp.name)
            json.dump(settings, tmp, indent=2)
            tmp.write("\n")
        tmp_path.replace(destination)
