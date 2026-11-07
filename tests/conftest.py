"""Fixtures: un modelo entrenado en memoria (sin servidor MLflow) para pruebas autocontenidas."""
import warnings
import pandas as pd
import pytest
from src import config, datos
from src.train import construir_pipeline
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def particion():
    return datos.dividir(datos.generar_dataset_base())


@pytest.fixture(scope="session")
def modelo(particion):
    X_tr, _, y_tr, _ = particion
    return construir_pipeline(GradientBoostingClassifier,
                              dict(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=config.SEED)).fit(X_tr, y_tr)


@pytest.fixture(scope="session")
def baseline(particion):
    X_tr, _, _, _ = particion
    return X_tr.reset_index(drop=True)
