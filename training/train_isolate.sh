#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="Chiller_anomaly_isolate"
DATA="data/Chiller_Process_Data_processed.csv"

python training_isolate_forest.py \
  --model-name "$MODEL_NAME" \
  --data "$DATA" \
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
