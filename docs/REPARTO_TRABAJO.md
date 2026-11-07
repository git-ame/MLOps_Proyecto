# Reparto de trabajo

> Completar los nombres antes de entregar. Los entregables corresponden a carpetas concretas del repositorio, de modo que cada integrante pueda explicar y defender su parte.

| Integrante | Rol | Entregables | Archivos |
|---|---|---|---|
| _(nombre 1)_ | Datos y modelado | Generador sintético, entrenamiento, 6 runs, registro v1/v2 | `src/datos.py`, `src/train.py`, `src/config.py` |
| _(nombre 2)_ | Servicio y contenedores | API, Dockerfiles, compose | `src/serve/`, `Dockerfile*`, `docker-compose.yml` |
| _(nombre 3)_ | Kubernetes y operación | Manifiestos, 4 demostraciones, scripts | `k8s/`, `scripts/*.sh` |
| _(nombre 4)_ | Monitoreo y calidad | Data/concept drift, pytest, gráficas | `src/drift/`, `tests/`, `scripts/run_drift_report.py` |
| _(nombre 5)_ | Documentación e interfaz | Arquitectura, evidencias, guion, UI | `docs/`, `ui/`, `README.md` |

Con un equipo menor, se agrupan roles contiguos; todos revisan los PR de los demás (ver historial Git).
