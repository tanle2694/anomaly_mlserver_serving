# Anomaly Detection Serving System

A full-stack system for training, serving, and explaining anomaly detection models. Supports **Isolation Forest** and **Local Outlier Factor** algorithms with SHAP-based explainability.

## Architecture Overview

```
training/          ← Train models offline → outputs joblib artifacts
backend/           ← FastAPI REST API (port 8000): model registry & inference proxy
serving/           ← MLServer engine (ports 8080/8081): loads models dynamically
frontend/          ← React SPA (port 5173 dev): upload models, run inference
models/            ← Shared volume between backend and MLServer
```

---

## Step 1 — Train a Model

### Prerequisites

```bash
cd training
pip install -r requirements.txt
```

### Option A: Isolation Forest

```bash
python training_isolate_forest.py \
  --model-name <model_name> \
  --data <path_to_csv> \
  --contamination auto \
  --features "col1" "col2" "col3"
```

**Arguments**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model-name` | No | `iforest_v1` | Name used to identify the model |
| `--data` | No | *(synthetic)* | Path to CSV file; omit to generate synthetic data |
| `--features` | No | 11 chiller columns | Space-separated list of CSV column names to use |
| `--contamination` | No | `auto` | Expected anomaly fraction: `auto` or a float `0.0–0.5` |

**Example — real CSV with chiller sensor data:**

```bash
python training_isolate_forest.py \
  --model-name Chiller_iforest_v1 \
  --data data/Chiller_Process_Data_processed.csv \
  --contamination auto \
  --features \
    "Chiller Power (kW)" \
    "Chilled Water(CHW) Supply Temp (C°)" \
    "Chilled Water(CHW) Return Temp (C°)" \
    "Condensed Water(CDW) Supply Temp (C°)" \
    "Condensed Water(CDW) Return Temp (C°)" \
    "Chilled Water(CDW) Return Flow (m3/hr)" \
    "Condensed Water(CDW) Return Flow (m3/hr)" \
    "Cooling Tower Power (kW)" \
    "Condensed Water(CDW) Pump Power (kW)" \
    "Chilled Water(CHW) Pump Power (kW)" \
    "Cooling Load (RT)"
```

**Example — synthetic data with simple features:**

```bash
python training_isolate_forest.py \
  --model-name sensor_iforest \
  --contamination 0.05 \
  --features temperature pressure vibration humidity
```

Or use the pre-filled convenience script:

```bash
bash train_isolate.sh
```

---

### Option B: Local Outlier Factor

```bash
python training_local_outlier.py \
  --model-name <model_name> \
  --data <path_to_csv> \
  --contamination auto \
  --n-neighbors 20 \
  --features "col1" "col2" "col3"
```

**Arguments**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model-name` | No | `lof_v1` | Name used to identify the model |
| `--data` | **Yes** | — | Path to CSV file (LOF does not support synthetic generation) |
| `--features` | No | 11 chiller columns | Space-separated list of CSV column names to use |
| `--contamination` | No | `auto` | Expected anomaly fraction: `auto` or a float `0.0–0.5` |
| `--n-neighbors` | No | `20` | Number of neighbors for LOF |

**Example — real CSV with chiller sensor data:**

```bash
python training_local_outlier.py \
  --model-name Chiller_lof_v1 \
  --data data/Chiller_Process_Data_processed.csv \
  --contamination auto \
  --n-neighbors 20 \
  --features \
    "Chiller Power (kW)" \
    "Chilled Water(CHW) Supply Temp (C°)" \
    "Chilled Water(CHW) Return Temp (C°)" \
    "Condensed Water(CDW) Supply Temp (C°)" \
    "Condensed Water(CDW) Return Temp (C°)" \
    "Chilled Water(CDW) Return Flow (m3/hr)" \
    "Condensed Water(CDW) Return Flow (m3/hr)" \
    "Cooling Tower Power (kW)" \
    "Condensed Water(CDW) Pump Power (kW)" \
    "Chilled Water(CHW) Pump Power (kW)" \
    "Cooling Load (RT)"
```

Or use the pre-filled convenience script:

```bash
bash train_lof.sh
```

---

### Training Output

Artifacts are saved under `training/out/<model_name>/`:

```
training/out/<model_name>/
├── metadata.json                    ← model info, features, evaluation stats
├── predictor/
│   ├── predictor.joblib             ← trained sklearn pipeline
│   └── model-settings.json          ← MLServer runtime config
└── explainer/
    ├── explainer.joblib             ← SHAP KernelExplainer
    └── model-settings.json          ← MLServer runtime config
```

---

## Step 2 — Start the Services

From the repository root:

```bash
docker-compose up -d
```

This starts two containers:

| Container | Ports | Role |
|---|---|---|
| `anomaly-mlserver` | 8080 (HTTP), 8081 (gRPC) | MLServer — loads and runs models |
| `anomaly-backend` | 8000 | FastAPI — model registry, inference proxy |

**Check health:**

```bash
curl http://localhost:8000/health
curl http://localhost:8080/v2/health/ready
```

**View logs:**

```bash
docker-compose logs -f mlserver
docker-compose logs -f backend
```

**Stop services:**

```bash
# Keep model volume
docker-compose down

# Wipe model volume (removes all loaded models)
docker-compose down -v
```

---

## Step 3 — Start the Frontend

The frontend runs locally (not inside Docker):

```bash
cd frontend
npm install
npm run dev    # http://localhost:5173, /api proxied → backend:8000
```

Open **http://localhost:5173** in your browser.

---

## Step 4 — Upload a Model from the UI

1. Open **http://localhost:5173** and go to the **Upload** tab.

2. **Quickest path — folder picker:**
   - Click **"Model folder"** and select the `training/out/<model_name>/` directory.
   - The UI reads `metadata.json` and auto-fills the model name, feature list, and all four artifact files.

3. **Manual path — upload files individually:**
   - Enter a **Model name**.
   - Under **Predictor**, select `predictor/predictor.joblib` and `predictor/model-settings.json`.
   - Under **Explainer**, select `explainer/explainer.joblib` and `explainer/model-settings.json`.
   - Optionally edit the **Input Features** table (names and default inference values).

4. Click **Upload**. The backend writes the files to the shared `/models` volume and hot-loads both sub-models into MLServer — no restart needed.

---

## Running Inference

### From the UI

1. Go to the **Inference** tab.
2. Select a registered model from the dropdown.
3. Fill in the feature values and click **Run**.
4. The result shows the anomaly score, a normal/anomaly label, and SHAP feature contributions.

### From the command line

```bash
# Anomaly prediction with default values
./scripts/infer.sh <model_name>

# With explicit feature values (temperature pressure vibration humidity)
./scripts/infer.sh <model_name> 70.0 30.0 0.5 40.0

# SHAP explanation
./scripts/infer_shap.sh <model_name>
```

---

## Key Environment Variables

| Variable | Component | Default | Purpose |
|---|---|---|---|
| `MLSERVER_URL` | backend | `http://localhost:8080` | MLServer API base URL |
| `MODELS_DIR` | backend | `/models` | Shared model repository root |
| `MLSERVER_PARALLEL_WORKERS` | serving | `0` | Sequential processing (avoids pickle conflicts) |
