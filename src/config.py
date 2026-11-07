"""Configuración central del proyecto (única fuente de verdad para nombres y constantes)."""
import os

# --- Reproducibilidad -------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
N_REGISTROS = 6000

# --- Esquema de datos --------------------------------------------------------
TARGET = "desercion"  # 1 = el participante abandona el programa; 0 = continúa/concluye

FEATURES_NUM = [
    "precio_matricula",
    "precio_colegiatura",
    "nro_cuotas",
    "monto_pagado_formativa",
    "saldo_pendiente_formativa",
    "porcentaje_pagado",
    "descuento_pct",
    "cuotas_vencidas",
    "dias_desde_ultimo_pago",
    "meses_desde_inscripcion",
    "modulos_programados",
    "modulos_reprobados",
]
FEATURES_CAT = [
    "tipo_programa",
    "sede",
    "tipo_plan_pago",
    "tiene_promocion",
    "es_regularizado",
]
FEATURES = FEATURES_NUM + FEATURES_CAT

# --- MLflow ------------------------------------------------------------------
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "desercion_caso_x"
MODEL_NAME = os.getenv("MODEL_NAME", "desercion_caso_x")
ALIAS_CHAMPION = os.getenv("MODEL_ALIAS", "champion")
ALIAS_CHALLENGER = "challenger"
METRICA_PRINCIPAL = "cv_f1_mean"  # F1 de la clase "deserta", validación cruzada 5-fold sobre train

# --- Rutas locales -----------------------------------------------------------
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_DATASET = os.path.join(RAIZ, "data", "desercion_caso_x.csv")
RUTA_BASELINE = os.path.join(RAIZ, "data", "baseline_referencia.csv")
RUTA_EVIDENCIAS = os.path.join(RAIZ, "evidencias")
