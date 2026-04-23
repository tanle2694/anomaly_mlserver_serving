# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An anomaly detection serving system with four components:
- **backend/** — FastAPI REST API (port 8000): model registry management, inference proxying, feature validation
- **serving/** — MLServer engine (ports 8080/8081): loads sklearn and SHAP models dynamically from shared `/models` volume
- **training/** — Offline training pipeline: generates synthetic sensor data, trains IsolationForest + SHAP explainer
- **frontend/** — React + Vite SPA (port 8089): model upload, listing, and inference UI

## Running the System

```bash
# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health
curl http://localhost:8080/v2/health/ready

# View logs
docker-compose logs -f mlserver
docker-compose logs -f backend

# Stop (keep volumes)
docker-compose down
# Stop and wipe volumes
docker-compose down -v
```

## Training a Model Locally

```bash
cd training
pip install -r requirements.txt
python training.py --model-name iforest_v2
# Outputs: training/out/iforest_v2/{predictor.joblib, explainer.joblib}
```

## Frontend Development

```bash
cd frontend
npm install
npm run dev    # Dev server on :5173 with /api proxy to backend:8000
npm run build  # Production build to dist/
```

## Testing Inference via Scripts

```bash
./scripts/infer.sh iforest_v1                        # default values
./scripts/infer.sh iforest_v1 70.0 30.0 0.5 40.0    # temp, pressure, vib, humidity
./scripts/infer_shap.sh iforest_v1                   # SHAP explanation
```

## Architecture: Model Registry Convention

Each logical model lives at `models/<model_name>/` and contains two MLServer sub-models:
- `models/<model_name>/predictor/` → loaded as `<model_name>__predictor` (SKLearnModel runtime)
- `models/<model_name>/explainer/` → loaded as `<model_name>__explainer` (ShapExplainerRuntime)

The double-underscore naming allows `registry.py` to split on `__` and group artifacts by logical model name.

## Architecture: Inference and Explanation Flows

**Inference:** Frontend → `POST /api/infer/{model_name}` → Backend validates 4 features → MLServerClient calls `/v2/models/{model_name}__predictor/infer` → IsolationForest `decision_function` → `is_anomaly = (score < 0)`

**Explanation:** Frontend → `POST /api/infer/{model_name}/explain` → MLServerClient calls `/v2/models/{model_name}__explainer/infer` → `ShapExplainerRuntime.predict()` → SHAP `KernelExplainer.shap_values()` → dict of feature → contribution

**Upload:** Frontend multipart form → `POST /api/models` → Backend writes joblib files + generates `model-settings.json` → Backend calls MLServer `/v2/repository/models/{name}/load` for both sub-models

## Architecture: Hot-Loading

The backend never restarts MLServer. It calls `/v2/repository/models/{name}/load` and `/unload` to dynamically add and remove models. MLServer discovers `model-settings.json` recursively under `/models`.

## Critical: ShapExplainer Class Duplication

`training/explainer.py` and `serving/explainer.py` define the **same `ShapExplainer` class**. This is intentional — joblib pickle requires the class to be importable at the same module path at unpickle time. If you change one, change both.

## Hardcoded Feature Schema

The four sensor features `["temperature", "pressure", "vibration", "humidity"]` are hardcoded in:
- `backend/app/config.py` (validation)
- `backend/app/routers/infer.py` (request parsing)
- `frontend/src/pages/Inference.tsx` (form fields)
- `training/training.py` (data generation)
- `serving/model_repo_seed/iforest_v1/*/model-settings.json` (input shape `[-1, 4]`)

Any schema change requires updating all five locations.

## Key Environment Variables

| Variable | Component | Default | Purpose |
|---|---|---|---|
| `MLSERVER_URL` | backend | `http://localhost:8080` | MLServer API base URL |
| `MODELS_DIR` | backend | `/models` | Shared model repository root |
| `MLSERVER_PARALLEL_WORKERS` | serving | `0` | Sequential processing (set in compose) |

## Key Files to Understand First

- `backend/app/services/registry.py` — filesystem layout and model-settings.json generation
- `backend/app/services/mlserver_client.py` — MLServer V2 API calls
- `serving/runtime/shap_runtime.py` — custom MLServer runtime for SHAP explainers
- `docker-compose.yml` — volume mounts, port bindings, service dependencies
