# MLOps — Predicción de deserción (caso X)

Servicio de predicción de deserción de participantes de posgrado de una escuela de negocios (caso X, institución anonimizada), con trazabilidad en **MLflow**, empaquetado en **Docker**, desplegado en **Kubernetes** (3 réplicas) y monitoreado con pruebas de **data drift** y **concept drift**. Datos 100 % sintéticos (estructura inspirada en el instrumento "Reporte SXX v3.2" de la tesis).

```
src/                 config, datos sintéticos, train (MLflow), serve/api (FastAPI), drift/
tests/               pytest: datos, API, data drift, concept drift (15 pruebas)
k8s/                 kind-config, mlflow.yaml, api.yaml
scripts/             start_mlflow, despliegue_k8s, demo_k8s, run_drift_report, promover, info_despliegue
ui/app.py            interfaz Streamlit que consume la API del clúster (extra)
docs/                ARQUITECTURA, DRIFT, REPARTO_TRABAJO, EVIDENCIAS, GUION_DEFENSA, INFO_DESPLIEGUE
evidencias/          gráficas, capturas de MLflow y resultados (demo_k8s.txt se genera al ejecutar)
```

## Ejecución rápida (local, sin contenedores)
```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
python -m src.datos                      # genera data/*.csv
scripts/start_mlflow.sh 5000 &           # MLflow en http://localhost:5000
python -m src.train                      # 6 runs + registro v1 (baseline) y v2 (champion)
uvicorn src.serve.api:app --port 8000    # API
pytest -q                                # pruebas (no requieren MLflow)
python -m scripts.run_drift_report       # gráficas de deriva
```

## Despliegue en Kubernetes (kind)
Requisitos: Docker, kind, kubectl.
```bash
scripts/despliegue_k8s.sh      # clúster, imágenes, MLflow, entrenamiento, API x3
scripts/demo_k8s.sh            # las 4 demostraciones (guarda evidencias/demo_k8s.txt)
API_URL=http://localhost:8080 python -m scripts.info_despliegue   # versión y run desplegados
pip install -r requirements-ui.txt && API_URL=http://localhost:8080 streamlit run ui/app.py
```
Puertos: API `localhost:8080`, MLflow UI `localhost:5001`.

Promover/rollback: `MLFLOW_TRACKING_URI=http://localhost:5001 python -m scripts.promover 1` y `kubectl rollout restart deployment/desercion-api` (la API resuelve el alias `champion` al arrancar).
