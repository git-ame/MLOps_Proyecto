"""Genera las evidencias de deriva (PNG + resumen) usando el modelo champion del registro.
Uso: python -m scripts.run_drift_report"""
import json, os
import mlflow, pandas as pd
from src import config, datos
from src.drift import concept_drift as cd, data_drift as dd, graficas

RETRASO_ETIQUETAS = 3  # lotes (≈ semanas) hasta conocer si el participante desertó
os.makedirs(config.RUTA_EVIDENCIAS, exist_ok=True)
mlflow.set_tracking_uri(config.TRACKING_URI)
modelo = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@{config.ALIAS_CHAMPION}")
base = pd.read_csv(config.RUTA_BASELINE)
_, X_te, _, y_te = datos.dividir(datos.generar_dataset_base())

# --- Data drift
ok = dd.evaluar(base, datos.generar(1000, seed=2001))
dr = dd.evaluar(base, datos.generar(1000, seed=2002, deriva_entradas=True))
graficas.semaforo_data_drift(ok, dr, f"{config.RUTA_EVIDENCIAS}/data_drift_semaforo.png")
dr.round(4).to_csv(f"{config.RUTA_EVIDENCIAS}/data_drift_detalle_con_deriva.csv", index=False)

# --- Concept drift
ref = cd.estimar_referencia(modelo, X_te, y_te)
f1s = cd.f1_por_lote(modelo, cd.generar_lotes(n_lotes=12, inicio_drift=6))
alerta = cd.primer_lote_alerta(f1s, ref.umbral)
graficas.f1_temporal(f1s, ref, 6, alerta, f"{config.RUTA_EVIDENCIAS}/concept_drift_f1.png", RETRASO_ETIQUETAS)
resumen = {
    "data_drift": {"sin_deriva": {"alertas": int(ok.alerta.sum()), "hay_drift": dd.hay_drift(ok)},
                   "con_deriva": {"alertas": int(dr.alerta.sum()), "hay_drift": dd.hay_drift(dr)}},
    "concept_drift": {"f1_base": round(ref.f1_base, 3), "sigma": round(ref.sigma, 3), "umbral": round(ref.umbral, 3),
                      "f1_por_lote": [round(x, 3) for x in f1s], "cambio_en_lote": 7,
                      "lote_alerta": None if alerta is None else alerta + 1,
                      "lote_deteccion_con_retraso": None if alerta is None else alerta + 1 + RETRASO_ETIQUETAS},
}
json.dump(resumen, open(f"{config.RUTA_EVIDENCIAS}/drift_resumen.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(resumen, indent=2, ensure_ascii=False))
