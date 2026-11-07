import pytest
from fastapi.testclient import TestClient
from src import config, datos
from src.serve import api


def test_dataset_cumple_requisitos():
    df = datos.generar_dataset_base()
    assert len(df) >= 5000
    assert set(config.FEATURES + [config.TARGET]) <= set(df.columns)
    assert 0.05 < df[config.TARGET].mean() < 0.5


def test_generador_es_determinista():
    assert datos.huella(datos.generar(300, seed=7)) == datos.huella(datos.generar(300, seed=7))


@pytest.fixture()
def cliente(modelo):
    api._estado.update(modelo=modelo, version="2", run_id="run-test", cargado_en="t")
    return TestClient(api.app)  # sin `with`: no ejecuta el startup que busca MLflow


def _fila():
    f = datos.generar(1, seed=11)[config.FEATURES].iloc[0].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in f.items()}


def test_health_ready_info(cliente):
    assert cliente.get("/health").json()["status"] == "ok"
    assert cliente.get("/ready").status_code == 200
    assert cliente.get("/info").json()["version"] == "2"


def test_predict_devuelve_probabilidad_y_pod(cliente):
    r = cliente.post("/predict", json=_fila())
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["probabilidad_desercion"] <= 1 and "pod" in j and j["version_modelo"] == "2"


def test_predict_rechaza_entrada_invalida(cliente):
    f = _fila(); f["porcentaje_pagado"] = 7
    assert cliente.post("/predict", json=f).status_code == 422


def test_ready_503_sin_modelo():
    api._estado["modelo"] = None
    assert TestClient(api.app).get("/ready").status_code == 503
