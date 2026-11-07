# Arquitectura

## Visión general
```
 Usuario/UI (Streamlit) ──HTTP──> Service desercion-api (NodePort 30080 → :8080 host)
                                     │ balancea
                      ┌──────────────┼──────────────┐
                  Pod api-1       Pod api-2       Pod api-3      (Deployment, 3 réplicas)
                      └──── carga models:/desercion_caso_x@champion ────┐
                                                                        ▼
                                   Service mlflow (:5000) ── Pod mlflow (SQLite + artefactos en PVC)
                                   (tracking + Model Registry + proxy de artefactos)
```

## Componentes y decisiones
| Componente | Decisión | Motivo |
|---|---|---|
| Datos | Generador sintético con semilla 42, 6 000 filas, 17 variables, prevalencia ≈ 17,6 % | Datos reales confidenciales; permite inyectar drift de forma controlada |
| Partición | 80/20 estratificada, `random_state=42` | Reproducible; CV 5-fold sobre train para elegir |
| Métrica principal | **F1 de la clase "deserta"** | Clase minoritaria: *accuracy* sería engañosa |
| Tracking | 6 runs (LogReg×2, RF×2, GB×2) con parámetros, métricas CV y test, hash del dataset, firma e input example | Comparabilidad y trazabilidad |
| Registry | Modelo `desercion_caso_x`: **v1** (baseline, alias `baseline`) y **v2** (alias `champion`) | La API consume el *alias*, no un número: promover/rollback = mover el alias |
| Servidor MLflow | `--serve-artifacts` (proxy): la API solo necesita HTTP al servidor, sin montar el disco de artefactos | Evita acoplar volúmenes entre pods |
| API | FastAPI; `/health` (liveness), `/ready` (readiness, 503 sin modelo), `/info`, `/predict`, `/predict_batch`, `/reload` | Probes de K8s distinguen "vivo" de "listo" |
| Carga del modelo | Hilo en segundo plano con reintentos | La API puede arrancar antes que MLflow |
| Imagen | `python:3.13-slim`, `requirements.txt` con **todas las versiones fijadas** (incl. `mlflow==3.18.0`, igual que `Dockerfile.mlflow`), usuario no root | Misma versión cliente/servidor evita fallos al deserializar |
| Kubernetes | Deployment `replicas: 3`, `POD_NAME` por downward API, requests/limits, probes; Service NodePort | Balanceo visible por nombre de pod |
| MLflow en K8s | 1 réplica, `strategy: Recreate`, PVC | SQLite no admite escritores concurrentes: la persistencia es de un solo pod por diseño |

## Versión del registro desplegada
**Modelo `desercion_caso_x`, versión v2 (alias `champion`)**, generada por el run `gb_100_lr0.1`.

- Verificación local (esta entrega): v2 ← run `928212d1cd2d454a9b51ced57256dcca` (ver `INFO_DESPLIEGUE.md`).
- En el clúster, `scripts/despliegue_k8s.sh` vuelve a entrenar contra el MLflow del clúster; el modelo es el mismo (semilla fija) pero el **run_id es distinto**. Tras desplegar, ejecutar `scripts.info_despliegue` y sustituir el valor de `INFO_DESPLIEGUE.md` (lo lee del endpoint `/info`, no se escribe a mano).

## Resultados de entrenamiento (verificados)
| Run | F1 CV (5-fold) | F1 test | Precisión | Recall | AUC |
|---|---|---|---|---|---|
| logreg_C0.1 (**v1 baseline**) | 0.601 | 0.593 | 0.474 | 0.791 | 0.873 |
| logreg_C1 | 0.598 | 0.595 | 0.475 | 0.796 | 0.873 |
| rf_100_d5 | 0.580 | 0.578 | 0.452 | 0.801 | 0.864 |
| rf_300_d10 | 0.602 | 0.595 | 0.505 | 0.725 | 0.859 |
| **gb_100_lr0.1 (v2 champion)** | **0.604** | 0.558 | 0.688 | 0.469 | 0.869 |
| gb_200_lr0.05 | 0.598 | 0.563 | 0.694 | 0.474 | 0.869 |

### Lectura honesta de la selección
La regla de selección se fijó antes de entrenar (mayor F1 en CV). Gana `gb_100_lr0.1` con 0.604, pero **la diferencia con la baseline (0.601) es muy inferior a la desviación de la CV (≈ 0.02): son prácticamente equivalentes**, y en test la regresión logística tiene mejor F1 (0.593 vs 0.558). El champion es más preciso (0.69) pero detecta menos casos (recall 0.47 vs 0.79). Para retención, donde omitir a un desertor suele costar más que contactar a uno que no lo era, la baseline podría ser preferible, o bien ajustar el umbral de decisión (`UMBRAL_DECISION`). Se mantiene v2 como desplegado para cumplir el flujo del módulo, y `scripts/promover.py 1` permite el rollback inmediato. Con datos reales esta decisión debe tomarse con el costo de cada error.

## Limitaciones
Datos sintéticos (las métricas no son estimaciones sobre caso X); un solo nodo MLflow; sin autenticación en la API ni en MLflow (apto para laboratorio, no para producción); sin autoescalado (HPA) porque el requisito pide escalado manual demostrable.
