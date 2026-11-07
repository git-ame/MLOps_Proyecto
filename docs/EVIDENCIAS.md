# Evidencias

## Ya incluidas (generadas y verificadas en esta entrega)
| Requisito | Evidencia |
|---|---|
| ≥ 5 runs comparables en MLflow | `mlflow_01_experimento_runs.png` (6 runs), `runs_resumen.csv` |
| Registry con > 1 versión y alias | `mlflow_02_registry_modelos.png`, `mlflow_03_registry_versiones_alias.png` (v1 `baseline`, v2 `champion`) |
| API operativa | `api_swagger.png`; `docs/INFO_DESPLIEGUE.md` |
| Data drift verde/rojo | `data_drift_semaforo.png`, `data_drift_detalle_con_deriva.csv` |
| Concept drift temporal | `concept_drift_f1.png`, `drift_resumen.json` |
| Pruebas automatizadas | `pytest -q` → 15 passed (`evidencias/pytest.txt`) |

## Por generar en la máquina con Docker + kind (no ejecutables en el entorno de desarrollo, que no tiene daemon de Docker)
Ejecutar `scripts/despliegue_k8s.sh` y luego `scripts/demo_k8s.sh`; la salida queda en `evidencias/demo_k8s.txt`. Añadir capturas de pantalla de la terminal en cada punto:

| # | Demostración | Qué debe verse |
|---|---|---|
| 1 | Réplicas Running | `kubectl get pods`: 3 pods `desercion-api-*` en `Running 1/1` |
| 2 | Balanceo | Conteo de respuestas por nombre de pod: 3 nombres distintos |
| 3 | Autorreparación | Pod eliminado y reemplazado por uno nuevo (nombre y AGE distintos) |
| 4 | Escalado | 3 → 5 → 3 réplicas |
| 5 | UI (extra) | Captura de Streamlit mostrando "Respondió el pod …" |
| 6 | Versión desplegada | `docs/INFO_DESPLIEGUE.md` regenerado desde el clúster |
