#!/usr/bin/env bash
# Despliegue completo en kind: clúster -> imágenes -> MLflow -> entrenamiento -> API (3 réplicas).
set -euo pipefail
cd "$(dirname "$0")/.."
kind create cluster --name desercion --config k8s/kind-config.yaml || true
docker build -t desercion-mlflow:1.0 -f Dockerfile.mlflow .
docker build -t desercion-api:1.0 -f Dockerfile .
kind load docker-image desercion-mlflow:1.0 desercion-api:1.0 --name desercion
kubectl apply -f k8s/mlflow.yaml
kubectl rollout status deployment/mlflow --timeout=180s
# Entrenar y registrar contra el MLflow DEL CLÚSTER (NodePort 30500 -> localhost:5001)
MLFLOW_TRACKING_URI=http://localhost:5001 python -m src.train
kubectl apply -f k8s/api.yaml
kubectl rollout status deployment/desercion-api --timeout=300s
kubectl get pods -o wide
echo "API: http://localhost:8080/info   MLflow UI: http://localhost:5001"
