#!/usr/bin/env sh
set -eu

if [ -d /app/model_repo_seed ] && [ -z "$(find /models -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
  cp -R /app/model_repo_seed/. /models/
fi

exec mlserver start /models
