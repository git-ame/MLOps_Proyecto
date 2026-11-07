"""Detección de concept drift: cambia la relación entradas -> etiqueta, aunque las entradas no cambien.

Se monitorea el F1 de la clase "deserta" sobre lotes sucesivos de producción (500 participantes).

Criterio de reentrenamiento
  Alerta cuando el F1 del lote queda por debajo de   F1_base - max(0.05, 3*sigma)   durante
  2 lotes consecutivos.
    * F1_base y sigma se estiman con lotes remuestreados del conjunto de prueba (la variabilidad
      natural de F1 con n=500 y ~18 % de positivos).
    * 3*sigma: una sola caída por azar es improbable (<0.3 %); el piso de 0.05 evita umbrales
      absurdamente estrechos si sigma sale pequeña.
    * 2 lotes consecutivos: distingue una caída sostenida de un lote atípico.

Retraso de etiquetas (label delay)
  La etiqueta "desertó" se conoce semanas/meses después de la predicción. Con retraso L lotes, el
  F1 del lote t solo es calculable en t+L: la alerta llega con L lotes de atraso (latencia de
  detección). `evaluar_con_retraso` lo modela. Mientras tanto se usan señales SIN etiqueta (drift
  de entradas y de la distribución de probabilidades), que NO detectan un concept drift puro; por
  eso se complementa con un modelo challenger/shadow y auditoría muestral de etiquetas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import config, datos

TAMANO_LOTE = 500
LOTES_CONSECUTIVOS = 2
PISO_CAIDA = 0.05


@dataclass
class Referencia:
    f1_base: float
    sigma: float
    umbral: float


def estimar_referencia(modelo, X_te, y_te, n_boot: int = 300, seed: int = config.SEED) -> Referencia:
    rng = np.random.default_rng(seed)
    pred = modelo.predict(X_te)
    y = y_te.to_numpy()
    f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), TAMANO_LOTE)
        f1s.append(f1_score(y[idx], pred[idx], zero_division=0))
    f1_base, sigma = float(np.mean(f1s)), float(np.std(f1s))
    return Referencia(f1_base, sigma, f1_base - max(PISO_CAIDA, 3 * sigma))


def generar_lotes(n_lotes: int = 12, inicio_drift: int = 6, seed: int = 1000) -> list[pd.DataFrame]:
    """Lotes sucesivos; a partir de `inicio_drift` la regla de abandono cambia (entradas iguales)."""
    return [
        datos.generar(TAMANO_LOTE, seed=seed + i,
                      concepto="original" if i < inicio_drift else "desplazado")
        for i in range(n_lotes)
    ]


def f1_por_lote(modelo, lotes: list[pd.DataFrame]) -> list[float]:
    return [float(f1_score(l[config.TARGET], modelo.predict(l[config.FEATURES]), zero_division=0))
            for l in lotes]


def primer_lote_alerta(f1s: list[float], umbral: float, consecutivos: int = LOTES_CONSECUTIVOS):
    """Índice del lote en que se cumple el criterio (caída sostenida), o None."""
    racha = 0
    for i, f in enumerate(f1s):
        racha = racha + 1 if f < umbral else 0
        if racha >= consecutivos:
            return i
    return None


def evaluar_con_retraso(f1s: list[float], umbral: float, retraso: int):
    """Lote en que se DETECTA el drift si las etiquetas llegan `retraso` lotes tarde."""
    idx = primer_lote_alerta(f1s, umbral)
    return None if idx is None else idx + retraso
