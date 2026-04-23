#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${1:-iforest_v1}
TEMPERATURE=${2:-95.0}
PRESSURE=${3:-48.0}
VIBRATION=${4:-1.2}
HUMIDITY=${5:-15.0}

curl -s -X POST "http://localhost:8000/api/infer/${MODEL_NAME}/explain" \
  -H "Content-Type: application/json" \
  -d "{
    \"features\": {
      \"temperature\": ${TEMPERATURE},
      \"pressure\": ${PRESSURE},
      \"vibration\": ${VIBRATION},
      \"humidity\": ${HUMIDITY}
    }
  }" | python3 -m json.tool
