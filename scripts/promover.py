"""Promueve una versión del registro a champion (mover el alias) y recarga las réplicas.
Uso: python -m scripts.promover 1     # rollback a v1 (baseline)"""
import sys, mlflow
from mlflow import MlflowClient
from src import config
mlflow.set_tracking_uri(config.TRACKING_URI)
v = sys.argv[1]
MlflowClient().set_registered_model_alias(config.MODEL_NAME, config.ALIAS_CHAMPION, v)
print(f"alias '{config.ALIAS_CHAMPION}' -> versión {v}. Reinicie las réplicas: kubectl rollout restart deployment/desercion-api")
