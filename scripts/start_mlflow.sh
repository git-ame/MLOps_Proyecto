#!/usr/bin/env bash
# Servidor MLflow local (SQLite + artefactos servidos por proxy). Uso: scripts/start_mlflow.sh [puerto]
set -e
PORT="${1:-5000}"
mkdir -p mlflow_store/artifacts
exec mlflow server --host 0.0.0.0 --port "$PORT" \
  --backend-store-uri sqlite:///mlflow_store/mlflow.db \
  --artifacts-destination ./mlflow_store/artifacts \
  --serve-artifacts
