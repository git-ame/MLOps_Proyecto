"""Entrenamiento con MLflow: 6 experimentos comparables y registro de versiones en el Model Registry.

Uso:  python -m src.train
Requiere un servidor MLflow en MLFLOW_TRACKING_URI (ver scripts/start_mlflow.sh).

Resultado:
  * 6 runs en el experimento ``desercion_caso_x`` (3 familias de modelos, 2 configuraciones c/u).
  * Registro de 2 versiones en el Model Registry:
      v1 = línea base (regresión logística del primer run)  -> alias ``baseline``
      v2 = mejor run por F1 (CV 5-fold sobre train)         -> alias ``champion`` (el desplegado)
"""
from __future__ import annotations

import warnings

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config, datos

warnings.filterwarnings("ignore")

# MLflow >= 3 serializa sklearn con skops; los árboles requieren declararse como tipo confiable.
# Es seguro aquí porque el artefacto lo produce este mismo script (origen controlado).
TIPOS_CONFIABLES = ["sklearn.tree._tree.Tree"]

EXPERIMENTOS = [
    ("logreg_C0.1", "LogisticRegression", LogisticRegression, dict(C=0.1, class_weight="balanced", max_iter=1000)),
    ("logreg_C1", "LogisticRegression", LogisticRegression, dict(C=1.0, class_weight="balanced", max_iter=1000)),
    ("rf_100_d5", "RandomForest", RandomForestClassifier,
     dict(n_estimators=100, max_depth=5, class_weight="balanced", random_state=config.SEED, n_jobs=-1)),
    ("rf_300_d10", "RandomForest", RandomForestClassifier,
     dict(n_estimators=300, max_depth=10, class_weight="balanced", random_state=config.SEED, n_jobs=-1)),
    ("gb_100_lr0.1", "GradientBoosting", GradientBoostingClassifier,
     dict(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=config.SEED)),
    ("gb_200_lr0.05", "GradientBoosting", GradientBoostingClassifier,
     dict(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=config.SEED)),
]


def construir_pipeline(clase, params) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), config.FEATURES_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.FEATURES_CAT),
        ]
    )
    return Pipeline([("pre", pre), ("clf", clase(**params))])


def main() -> None:
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    client = MlflowClient()

    df = datos.generar_dataset_base()
    X_tr, X_te, y_tr, y_te = datos.dividir(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    firma = infer_signature(X_tr, np.zeros(len(X_tr), dtype=int))
    huella = datos.huella(df)

    resultados = []  # (nombre, run_id, f1_cv)
    for nombre, familia, clase, params in EXPERIMENTOS:
        with mlflow.start_run(run_name=nombre) as run:
            modelo = construir_pipeline(clase, params)
            scores = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="f1", n_jobs=1)
            modelo.fit(X_tr, y_tr)
            pred = modelo.predict(X_te)
            proba = modelo.predict_proba(X_te)[:, 1]

            mlflow.set_tags({"familia": familia, "caso": "X", "objetivo": config.TARGET})
            mlflow.log_params({"modelo": familia, **{k: v for k, v in params.items() if k != "n_jobs"}})
            mlflow.log_params(
                {
                    "semilla": config.SEED,
                    "test_size": config.TEST_SIZE,
                    "n_train": len(X_tr),
                    "n_test": len(X_te),
                    "n_variables": len(config.FEATURES),
                    "dataset_sha256": huella[:16],
                }
            )
            mlflow.log_metrics(
                {
                    "cv_f1_mean": float(scores.mean()),
                    "cv_f1_std": float(scores.std()),
                    "test_f1": f1_score(y_te, pred),
                    "test_precision": precision_score(y_te, pred),
                    "test_recall": recall_score(y_te, pred),
                    "test_auc": roc_auc_score(y_te, proba),
                }
            )
            mlflow.sklearn.log_model(
                modelo, name="model", signature=firma, input_example=X_tr.head(3),
                skops_trusted_types=TIPOS_CONFIABLES,
            )
            resultados.append((nombre, run.info.run_id, float(scores.mean())))
            print(f"{nombre:16s} cv_f1={scores.mean():.3f}±{scores.std():.3f} test_f1={f1_score(y_te, pred):.3f} "
                  f"auc={roc_auc_score(y_te, proba):.3f}")

    # ---- Model Registry ----------------------------------------------------------------------
    baseline = resultados[0]                       # v1: regresión logística, la más simple
    mejor = max(resultados, key=lambda r: r[2])    # v2: mejor F1 en validación cruzada

    for etiqueta, (nombre, run_id, f1cv) in (("baseline", baseline), ("champion", mejor)):
        mv = mlflow.register_model(f"runs:/{run_id}/model", config.MODEL_NAME)
        client.set_model_version_tag(config.MODEL_NAME, mv.version, "run_name", nombre)
        client.set_model_version_tag(config.MODEL_NAME, mv.version, "cv_f1_mean", f"{f1cv:.4f}")
        client.update_model_version(
            config.MODEL_NAME, mv.version,
            description=f"{etiqueta}: {nombre} (run {run_id}); F1 CV={f1cv:.3f}",
        )
        client.set_registered_model_alias(config.MODEL_NAME, etiqueta, mv.version)
        print(f"Registrado {config.MODEL_NAME} v{mv.version} <- run {run_id} ({nombre}) alias={etiqueta}")

    # El alias ``champion`` es lo único que consulta la API: promover = mover el alias.
    pd.DataFrame(resultados, columns=["run_name", "run_id", "cv_f1_mean"]).to_csv(
        f"{config.RUTA_EVIDENCIAS}/runs_resumen.csv", index=False
    )


if __name__ == "__main__":
    main()
