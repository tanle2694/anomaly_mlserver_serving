#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${1:-iforest_v1}
TEMPERATURE=${2:-70.0}
PRESSURE=${3:-30.0}
VIBRATION=${4:-0.5}
HUMIDITY=${5:-40.0}

curl -s -X POST "http://localhost:8000/api/infer/${MODEL_NAME}" \
  -H "Content-Type: application/json" \
  -d "{
    \"features\": {
      \"temperature\": ${TEMPERATURE},
      \"pressure\": ${PRESSURE},
      \"vibration\": ${VIBRATION},
      \"humidity\": ${HUMIDITY}
    }
  }" | python3 -m json.tool
