"""Detección de data drift: compara la distribución de las ENTRADAS de producción contra el baseline.

Prueba por tipo de variable
  * Numéricas   -> Kolmogórov-Smirnov de dos muestras (no paramétrica, no asume normalidad; las
                   variables de pago son asimétricas y con colas largas).
  * Categóricas -> chi-cuadrado de homogeneidad sobre la tabla de frecuencias baseline vs. lote.

Criterio de alerta por variable (ambas condiciones a la vez)
  1. Significancia: p < alpha/m  (corrección de Bonferroni; m = nº de variables, alpha = 0.01).
     Justificación: con 17 pruebas simultáneas y alpha=0.01 sin corregir, la probabilidad de al
     menos una falsa alarma sería ~16 %; con Bonferroni queda <= 1 %.
  2. Tamaño del efecto: KS D >= 0.10 (numéricas) o PSI >= 0.10 (categóricas).
     Justificación: con n grande, una diferencia irrelevante ya es "significativa"; el umbral de
     efecto evita reentrenar por cambios sin consecuencia práctica. 0.10 es la frontera
     convencional entre cambio "insignificante" y "moderado" del PSI.

Criterio de alerta del conjunto: >= 20 % de las variables en alerta (>= 4 de 17). Un cambio aislado
en una variable no justifica reentrenar; un cambio coordinado sí.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src import config

ALPHA = 0.01
UMBRAL_EFECTO = 0.10
FRACCION_ALERTA = 0.20


def psi(base: pd.Series, nuevo: pd.Series, categorias: list[str], eps: float = 1e-4) -> float:
    """Population Stability Index sobre categorías."""
    pb = base.value_counts(normalize=True).reindex(categorias).fillna(0).clip(lower=eps)
    pn = nuevo.value_counts(normalize=True).reindex(categorias).fillna(0).clip(lower=eps)
    return float(((pn - pb) * np.log(pn / pb)).sum())


@dataclass
class ResultadoVariable:
    variable: str
    tipo: str
    prueba: str
    estadistico: float
    p_valor: float
    efecto: float
    alerta: bool


def evaluar(baseline: pd.DataFrame, lote: pd.DataFrame) -> pd.DataFrame:
    m = len(config.FEATURES)
    alpha_ajustado = ALPHA / m
    filas = []
    for v in config.FEATURES_NUM:
        ks = stats.ks_2samp(baseline[v], lote[v])
        filas.append(ResultadoVariable(v, "numérica", "KS", ks.statistic, ks.pvalue, ks.statistic,
                                       bool(ks.pvalue < alpha_ajustado and ks.statistic >= UMBRAL_EFECTO)))
    for v in config.FEATURES_CAT:
        cats = sorted(set(baseline[v]) | set(lote[v]))
        tabla = np.array([[ (baseline[v] == c).sum() for c in cats], [ (lote[v] == c).sum() for c in cats]])
        chi2, p, *_ = stats.chi2_contingency(tabla)
        efecto = psi(baseline[v], lote[v], cats)
        filas.append(ResultadoVariable(v, "categórica", "chi2", chi2, p, efecto,
                                       bool(p < alpha_ajustado and efecto >= UMBRAL_EFECTO)))
    df = pd.DataFrame([f.__dict__ for f in filas])
    df.attrs["alpha_ajustado"] = alpha_ajustado
    return df


def hay_drift(resultado: pd.DataFrame) -> bool:
    return bool(resultado["alerta"].mean() >= FRACCION_ALERTA)
