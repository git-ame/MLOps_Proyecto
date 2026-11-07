"""Generador de datos sintéticos de participantes de posgrado (caso X) y utilidades de partición.

Por qué sintético: los registros reales de caso X son confidenciales (ver tesis, apartado 1.6). El
generador replica la *estructura* del instrumento "Reporte SXX v3.2" (precios, pagos, saldos, cuotas
vencidas, módulos reprobados) y permite *inyectar de forma controlada* los dos tipos de deriva que
exige el proyecto:

  * data drift     -> cambia la distribución de las entradas (``deriva_entradas=True``).
  * concept drift  -> cambia la regla que genera la etiqueta (``concepto="desplazado"``), con
                      entradas distribuidas igual que en el entrenamiento.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

PROGRAMAS = ["Diplomado", "Especialidad", "Maestría"]
SEDES = ["Sede A", "Sede B", "Sede C", "Sede D", "Sede E"]


def _sigmoide(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generar(
    n: int,
    seed: int,
    concepto: str = "original",
    deriva_entradas: bool = False,
) -> pd.DataFrame:
    """Genera ``n`` participantes sintéticos.

    concepto:         "original" (regla vigente al entrenar) o "desplazado" (regla distinta).
    deriva_entradas:  si es True, altera la distribución de las variables de entrada.
    """
    if concepto not in ("original", "desplazado"):
        raise ValueError("concepto debe ser 'original' o 'desplazado'")
    rng = np.random.default_rng(seed)

    # ---- Parámetros de las entradas (con o sin deriva) ---------------------------------------
    p_programa = [0.50, 0.15, 0.35]
    p_sede = [0.30, 0.25, 0.20, 0.15, 0.10]
    p_credito = 0.70
    factor_precio = 1.0
    factor_mora = 1.0
    if deriva_entradas:
        p_programa = [0.25, 0.15, 0.60]      # más maestrías
        p_sede = [0.10, 0.15, 0.20, 0.25, 0.30]  # el volumen migra a otras sedes
        p_credito = 0.92                     # casi todos pagan a crédito
        factor_precio = 1.30                 # ajuste de aranceles
        factor_mora = 1.80                   # cartera mucho más lenta

    tipo_programa = rng.choice(PROGRAMAS, size=n, p=p_programa)
    sede = rng.choice(SEDES, size=n, p=p_sede)
    tipo_plan_pago = np.where(rng.random(n) < p_credito, "Crédito", "Contado")
    tiene_promocion = np.where(rng.random(n) < 0.30, "Sí", "No")
    es_regularizado = np.where(rng.random(n) < 0.10, "Sí", "No")

    media_col = {"Diplomado": 3000, "Especialidad": 5500, "Maestría": 9000}
    desv_col = {"Diplomado": 300, "Especialidad": 500, "Maestría": 800}
    cuotas_base = {"Diplomado": 6, "Especialidad": 10, "Maestría": 18}
    modulos_base = {"Diplomado": 4, "Especialidad": 8, "Maestría": 12}

    precio_colegiatura = np.array(
        [rng.normal(media_col[p], desv_col[p]) for p in tipo_programa]
    ) * factor_precio
    precio_matricula = precio_colegiatura * rng.uniform(0.08, 0.14, n)
    nro_cuotas = np.where(
        tipo_plan_pago == "Contado", 1, [cuotas_base[p] for p in tipo_programa]
    ).astype(int)
    modulos_programados = np.array([modulos_base[p] for p in tipo_programa], dtype=int)
    meses_desde_inscripcion = rng.integers(1, 19, n)

    # Avance de pago: contado paga casi todo; crédito es más disperso y se rezaga con la mora.
    pct_contado = rng.beta(8, 1.5, n)
    pct_credito = np.clip(rng.beta(3.0, 2.2, n) / factor_mora ** 0.5, 0, 1)
    porcentaje_pagado = np.where(tipo_plan_pago == "Contado", pct_contado, pct_credito)

    total_programa = precio_matricula + precio_colegiatura
    descuento_pct = np.where(tiene_promocion == "Sí", rng.uniform(5, 30, n), 0.0)
    descuento_monto = total_programa * descuento_pct / 100.0
    monto_pagado_formativa = porcentaje_pagado * (total_programa - descuento_monto)
    saldo_pendiente_formativa = np.clip(
        total_programa - descuento_monto - monto_pagado_formativa, 0, None
    )

    lam_venc = np.where(
        tipo_plan_pago == "Crédito", 0.2 + 3.2 * (1 - porcentaje_pagado) * factor_mora, 0.0
    )
    cuotas_vencidas = np.minimum(rng.poisson(lam_venc), nro_cuotas).astype(int)
    dias_desde_ultimo_pago = np.clip(
        rng.exponential(30.0, n) + 12.0 * cuotas_vencidas * factor_mora, 0, 365
    )
    modulos_reprobados = rng.binomial(modulos_programados, 0.08)

    # ---- Regla que genera la etiqueta -------------------------------------------------------
    credito = (tipo_plan_pago == "Crédito").astype(float)
    promo = (tiene_promocion == "Sí").astype(float)
    regul = (es_regularizado == "Sí").astype(float)
    ruido = rng.normal(0, 0.30, n)

    if concepto == "original":
        # Antes: la mora de pagos es la señal dominante del abandono.
        z = (
            -2.7
            + 0.80 * cuotas_vencidas
            + 0.012 * (dias_desde_ultimo_pago - 30)
            - 2.0 * porcentaje_pagado
            + 0.40 * modulos_reprobados
            - 0.60 * promo
            + 0.60 * credito
            + 0.30 * regul
            + ruido
        )
    else:
        # Después (p. ej., plan de flexibilización de pagos + campaña de retención): la mora deja
        # de predecir el abandono; pesa el desempeño académico y la promoción se vuelve decisiva.
        z = (
            -1.9
            + 0.05 * cuotas_vencidas
            + 0.0005 * (dias_desde_ultimo_pago - 30)
            + 0.20 * porcentaje_pagado
            + 1.10 * modulos_reprobados
            - 1.60 * promo
            - 0.20 * credito
            + 0.30 * regul
            + ruido
        )
    desercion = (rng.random(n) < _sigmoide(z)).astype(int)

    df = pd.DataFrame(
        {
            "precio_matricula": precio_matricula.round(2),
            "precio_colegiatura": precio_colegiatura.round(2),
            "nro_cuotas": nro_cuotas,
            "monto_pagado_formativa": monto_pagado_formativa.round(2),
            "saldo_pendiente_formativa": saldo_pendiente_formativa.round(2),
            "porcentaje_pagado": porcentaje_pagado.round(4),
            "descuento_pct": descuento_pct.round(2),
            "cuotas_vencidas": cuotas_vencidas,
            "dias_desde_ultimo_pago": dias_desde_ultimo_pago.round(0),
            "meses_desde_inscripcion": meses_desde_inscripcion,
            "modulos_programados": modulos_programados,
            "modulos_reprobados": modulos_reprobados,
            "tipo_programa": tipo_programa,
            "sede": sede,
            "tipo_plan_pago": tipo_plan_pago,
            "tiene_promocion": tiene_promocion,
            "es_regularizado": es_regularizado,
            config.TARGET: desercion,
        }
    )
    return df


# ---- Utilidades de dataset base y partición --------------------------------------------------
def generar_dataset_base() -> pd.DataFrame:
    """Dataset de entrenamiento oficial del proyecto (determinista)."""
    return generar(config.N_REGISTROS, seed=config.SEED, concepto="original")


def dividir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partición train/test estratificada y determinista: 80/20, semilla fija."""
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )


def huella(df: pd.DataFrame) -> str:
    """SHA-256 del contenido del DataFrame (trazabilidad de datos en MLflow)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


if __name__ == "__main__":
    base = generar_dataset_base()
    base.to_csv(config.RUTA_DATASET, index=False)
    X_tr, _, y_tr, _ = dividir(base)
    pd.concat([X_tr, y_tr], axis=1).to_csv(config.RUTA_BASELINE, index=False)
    print(f"Dataset: {config.RUTA_DATASET} ({len(base)} filas, {len(config.FEATURES)} variables)")
    print(f"Prevalencia de deserción: {base[config.TARGET].mean():.3f}")
    print(f"Baseline de referencia (train): {config.RUTA_BASELINE} ({len(X_tr)} filas)")
