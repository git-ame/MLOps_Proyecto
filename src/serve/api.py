"""API de inferencia (FastAPI). Carga el modelo con alias ``champion`` desde el Model Registry de MLflow.

Cada réplica informa su nombre de pod (variable POD_NAME inyectada por Kubernetes) para poder
demostrar el balanceo de carga, y la versión/run del modelo servido para trazabilidad.
"""
from __future__ import annotations

import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from src import config

POD = os.getenv("POD_NAME", socket.gethostname())
UMBRAL = float(os.getenv("UMBRAL_DECISION", "0.5"))

_estado: dict = {"modelo": None, "version": None, "run_id": None, "cargado_en": None}


class Participante(BaseModel):
    precio_matricula: float = Field(ge=0)
    precio_colegiatura: float = Field(ge=0)
    nro_cuotas: int = Field(ge=1)
    monto_pagado_formativa: float = Field(ge=0)
    saldo_pendiente_formativa: float = Field(ge=0)
    porcentaje_pagado: float = Field(ge=0, le=1)
    descuento_pct: float = Field(ge=0, le=100)
    cuotas_vencidas: int = Field(ge=0)
    dias_desde_ultimo_pago: float = Field(ge=0)
    meses_desde_inscripcion: int = Field(ge=0)
    modulos_programados: int = Field(ge=1)
    modulos_reprobados: int = Field(ge=0)
    tipo_programa: Literal["Diplomado", "Especialidad", "Maestría"]
    sede: str
    tipo_plan_pago: Literal["Crédito", "Contado"]
    tiene_promocion: Literal["Sí", "No"]
    es_regularizado: Literal["Sí", "No"]


class Lote(BaseModel):
    participantes: list[Participante] = Field(min_length=1, max_length=5000)


def cargar_modelo() -> None:
    """Resuelve el alias en el registro y carga esa versión concreta."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    cliente = MlflowClient()
    mv = cliente.get_model_version_by_alias(config.MODEL_NAME, config.ALIAS_CHAMPION)
    # Los tipos confiables (skops) viajan dentro del propio modelo registrado.
    _estado["modelo"] = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}/{mv.version}")
    _estado.update(version=mv.version, run_id=mv.run_id, cargado_en=time.strftime("%Y-%m-%dT%H:%M:%S"))


def _cargar_con_reintentos() -> None:
    """Reintenta: en Kubernetes la API puede arrancar antes que MLflow o antes de registrar el modelo."""
    for intento in range(90):
        try:
            cargar_modelo()
            print(f"[{POD}] modelo v{_estado['version']} cargado", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[{POD}] modelo no disponible aún ({intento + 1}/90): {exc}", flush=True)
            time.sleep(2)


@asynccontextmanager
async def lifespan(_app):
    # Hilo aparte: /health responde enseguida y /ready se abre solo cuando hay modelo.
    threading.Thread(target=_cargar_con_reintentos, daemon=True).start()
    yield


app = FastAPI(title="API de predicción de deserción (caso X)", version="1.0.0", lifespan=lifespan)


def _modelo():
    if _estado["modelo"] is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return _estado["modelo"]


@app.get("/health")
def health() -> dict:
    """Liveness: el proceso responde."""
    return {"status": "ok", "pod": POD}


@app.get("/ready")
def ready() -> dict:
    """Readiness: el modelo está cargado y puede recibir tráfico."""
    _modelo()
    return {"status": "ready", "pod": POD}


@app.get("/info")
def info() -> dict:
    return {
        "pod": POD,
        "modelo": config.MODEL_NAME,
        "alias": config.ALIAS_CHAMPION,
        "version": _estado["version"],
        "run_id": _estado["run_id"],
        "cargado_en": _estado["cargado_en"],
        "umbral": UMBRAL,
    }


@app.post("/reload")
def reload_() -> dict:
    """Recarga el alias vigente (tras promover otra versión) sin reconstruir la imagen."""
    cargar_modelo()
    return info()


def _predecir(df: pd.DataFrame) -> list[dict]:
    probas = _modelo().predict_proba(df[config.FEATURES])[:, 1]
    return [
        {"probabilidad_desercion": round(float(p), 4), "desercion": bool(p >= UMBRAL)} for p in probas
    ]


@app.post("/predict")
def predict(p: Participante) -> dict:
    res = _predecir(pd.DataFrame([p.model_dump()]))[0]
    return {**res, "pod": POD, "version_modelo": _estado["version"]}


@app.post("/predict_batch")
def predict_batch(lote: Lote) -> dict:
    df = pd.DataFrame([p.model_dump() for p in lote.participantes])
    return {"predicciones": _predecir(df), "pod": POD, "version_modelo": _estado["version"]}
