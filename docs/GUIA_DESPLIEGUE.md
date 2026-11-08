# Guía paso a paso de despliegue

Tiempo estimado: 25–40 minutos la primera vez (la mayor parte es la descarga de imágenes). Al final de cada paso hay una **comprobación**: si no coincide, no avanzar y ver la sección *Problemas frecuentes*.

## 0. Requisitos previos

| Herramienta | Versión sugerida | Para qué |
|---|---|---|
| Python | **3.13** (las versiones de `requirements.txt` están fijadas para 3.13) | Entrenar, pruebas, UI |
| Docker (Desktop o Engine) | 24 o superior, **en ejecución** | Construir imágenes |
| kind | 0.23 o superior | Clúster Kubernetes local |
| kubectl | 1.29 o superior | Operar el clúster |
| Git | cualquiera | Clonar el repositorio |
| Bash | Linux/macOS: ya viene. **Windows: usar WSL2 (Ubuntu) o Git Bash** | Ejecutar `scripts/*.sh` |

Recursos recomendados para Docker: 4 CPU y 6 GB de RAM (Docker Desktop → Settings → Resources). Con menos, los pods de la API pueden quedar reiniciándose.

Instalación rápida de kind y kubectl:
- **macOS:** `brew install kind kubectl`
- **Windows (PowerShell):** `winget install Kubernetes.kind Kubernetes.kubectl`
- **Linux:** descargar los binarios desde las páginas oficiales de kind y kubectl y copiarlos a `/usr/local/bin`.

**Comprobación:**
```bash
python --version        # Python 3.13.x
docker info | head -3   # debe mostrar datos del servidor, no un error de conexión
kind version
kubectl version --client
```

## 1. Obtener el proyecto
```bash
git clone <URL-DEL-REPOSITORIO> mlops_desercion
cd mlops_desercion
```
(Si se recibió el `.zip`: descomprimirlo y entrar a la carpeta `mlops_desercion`.)

## 2. Entorno Python y pruebas
```bash
python -m venv .venv
source .venv/bin/activate          # Windows (WSL/Git Bash): igual. PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m src.datos                # genera data/desercion_caso_x.csv y data/baseline_referencia.csv
pytest -q
```
**Comprobación:** `15 passed`. Las pruebas no necesitan MLflow ni Docker; si fallan, el problema es del entorno Python.

> Mantener el entorno **activado** en todos los pasos siguientes: el paso 4 entrena desde esta terminal.

## 3. Prueba opcional sin Kubernetes (docker compose)
Sirve para aislar errores de imagen antes de pasar al clúster. Se puede omitir.
```bash
docker compose up --build -d
# esperar ~30 s
curl localhost:5000/health         # OK  (MLflow)
MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.train
docker compose restart api         # recarga el modelo recién registrado
curl localhost:8000/info           # debe mostrar version "2"
docker compose down -v             # limpiar antes del paso 4 (libera los puertos 5000/8000)
```

## 4. Despliegue completo en Kubernetes (un solo comando)
```bash
chmod +x scripts/*.sh
scripts/despliegue_k8s.sh
```
El script hace, en orden:
1. Crea el clúster `desercion` (1 control-plane + 2 workers) con los puertos 8080 (API) y 5001 (MLflow) abiertos hacia `localhost`.
2. Construye las imágenes `desercion-mlflow:1.0` y `desercion-api:1.0`.
3. Las carga dentro del clúster (`kind load`), sin necesidad de un registro de imágenes.
4. Despliega MLflow (1 réplica con volumen persistente) y espera a que esté listo.
5. **Entrena desde tu máquina contra el MLflow del clúster** (`http://localhost:5001`): 6 runs y registro de v1 (baseline) y v2 (champion).
6. Despliega la API con **3 réplicas** y espera a que estén listas.

**Comprobación:** al terminar, `kubectl get pods` muestra 1 pod `mlflow-*` y 3 pods `desercion-api-*`, todos `Running` y `1/1`.

Si se prefiere hacerlo a mano (para entender cada paso), los comandos equivalentes son:
```bash
kind create cluster --name desercion --config k8s/kind-config.yaml
docker build -t desercion-mlflow:1.0 -f Dockerfile.mlflow .
docker build -t desercion-api:1.0 -f Dockerfile .
kind load docker-image desercion-mlflow:1.0 desercion-api:1.0 --name desercion
kubectl apply -f k8s/mlflow.yaml && kubectl rollout status deployment/mlflow
MLFLOW_TRACKING_URI=http://localhost:5001 python -m src.train
kubectl apply -f k8s/api.yaml && kubectl rollout status deployment/desercion-api
```

## 5. Verificar el servicio
```bash
curl localhost:8080/info
```
Debe devolver `"alias":"champion"`, `"version":"2"` y un `run_id`. Abrir además:
- **MLflow UI:** http://localhost:5001 → experimento `desercion_caso_x` (6 runs) y pestaña *Models* → `desercion_caso_x` (v1 con alias `baseline`, v2 con alias `champion`).
- **Swagger de la API:** http://localhost:8080/docs → probar `POST /predict` con "Try it out".

Registrar la versión desplegada para la documentación:
```bash
API_URL=http://localhost:8080 python -m scripts.info_despliegue
```
Esto reescribe `docs/INFO_DESPLIEGUE.md` con la versión y el `run_id` **reales del clúster** (son distintos a los de la verificación local).

## 6. Las cuatro demostraciones
```bash
scripts/demo_k8s.sh
```
Guarda todo en `evidencias/demo_k8s.txt`. Tomar una captura de pantalla de la terminal en cada bloque:

| # | Demostración | Resultado esperado |
|---|---|---|
| 1 | Réplicas Running | 3 pods `desercion-api-*` en `Running 1/1` |
| 2 | Balanceo | 12 peticiones repartidas entre 3 nombres de pod distintos (conteo aproximadamente parejo) |
| 3 | Autorreparación | Se elimina un pod; segundos después aparece otro con nombre y AGE nuevos y vuelve a haber 3 |
| 4 | Escalado | 3 → 5 → 3 réplicas |

Para una demostración en vivo en dos terminales: en una, `kubectl get pods -l app=desercion-api -w` (muestra los cambios en tiempo real); en la otra, ejecutar el script.

## 7. Pruebas de deriva y gráficas
```bash
MLFLOW_TRACKING_URI=http://localhost:5001 python -m scripts.run_drift_report
```
Regenera `evidencias/data_drift_semaforo.png`, `evidencias/concept_drift_f1.png` y `evidencias/drift_resumen.json` usando el modelo `champion` del clúster. Las pruebas automatizadas están en `pytest -q` (paso 2).

## 8. Interfaz de usuario (punto extra)
En otra terminal:
```bash
source .venv/bin/activate
pip install -r requirements-ui.txt
API_URL=http://localhost:8080 streamlit run ui/app.py
```
Abre http://localhost:8501. Al calcular el riesgo, la UI muestra qué pod respondió: pulsar varias veces para evidenciar el balanceo. Capturar la pantalla.

## 9. Operaciones útiles

**Rollback o promoción de versión** (el alias `champion` es lo único que consulta la API):
```bash
MLFLOW_TRACKING_URI=http://localhost:5001 python -m scripts.promover 1     # volver a v1
kubectl rollout restart deployment/desercion-api
kubectl rollout status deployment/desercion-api
curl localhost:8080/info                                                   # ahora "version":"1"
```
Para volver a v2: `promover 2` y reiniciar de nuevo.

**Ver registros:** `kubectl logs deployment/desercion-api` · `kubectl logs deployment/mlflow`
**Diagnosticar un pod:** `kubectl describe pod <nombre>`

## 10. Limpieza
```bash
kind delete cluster --name desercion     # elimina clúster, pods y volumen
docker compose down -v                   # si se usó el paso 3
deactivate
```

## 11. Entrega
1. Subir el repositorio a GitHub/GitLab (`git remote add origin <URL> && git push -u origin main`) y verificar que se vea `README.md`.
2. Completar los nombres en `docs/REPARTO_TRABAJO.md`.
3. Confirmar que `docs/INFO_DESPLIEGUE.md` fue regenerado **desde el clúster** (paso 5) y que `evidencias/` contiene `demo_k8s.txt` y las capturas de los pasos 6 y 8.
4. Comprimir: `zip -r entrega_mlops.zip . -x ".venv/*" "mlflow_store/*" ".git/*"` y adjuntar el enlace del repositorio.

---

## Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker no está iniciado | Abrir Docker Desktop / `sudo systemctl start docker` |
| `bind: address already in use` al crear el clúster | Puerto 8080 o 5001 ocupado (p. ej. por el paso 3) | `docker compose down -v` o cerrar el proceso que usa el puerto |
| Pods de la API en `ImagePullBackOff` | La imagen no se cargó en kind | Repetir `kind load docker-image desercion-api:1.0 desercion-mlflow:1.0 --name desercion` |
| Pods de la API en `Running` pero `0/1` | Aún no hay modelo registrado (el paso de entrenamiento falló o no se ejecutó) | Ejecutar el entrenamiento del paso 4; las réplicas pasan a `1/1` solas, reintentan hasta 3 minutos. Si ya pasó ese tiempo: `kubectl rollout restart deployment/desercion-api` |
| Pods de la API en `OOMKilled` / reinicios | Poca memoria asignada a Docker | Subir Docker a ≥ 6 GB o el `limits.memory` en `k8s/api.yaml` |
| `403 Invalid Host header` al entrenar | Se usó una dirección distinta a las permitidas | Entrenar con `MLFLOW_TRACKING_URI=http://localhost:5001` (o `127.0.0.1:5001`); otras direcciones deben añadirse a `--allowed-hosts` en `Dockerfile.mlflow` y reconstruir |
| `Connection refused` a `localhost:5001` | MLflow aún no está listo o el mapeo de puertos no se creó | `kubectl get pods`; si MLflow está listo y sigue fallando, `kind delete cluster --name desercion` y repetir el paso 4 |
| `untrusted types` / error de skops al cargar | Versión de MLflow distinta entre cliente e imagen | Reinstalar con `pip install -r requirements.txt` (fija `mlflow==3.18.0`, igual que las imágenes) |
| `pip install` falla por versiones | Python distinto de 3.13 | Usar Python 3.13 (p. ej. `pyenv install 3.13`) |
| Error de formato (`\r`) al correr los `.sh` en Windows | Saltos de línea CRLF | `git config core.autocrlf input` y volver a clonar, o `dos2unix scripts/*.sh` |
| `Operation not permitted` / permisos al ejecutar scripts | Sin permiso de ejecución | `chmod +x scripts/*.sh` |
| Tras reiniciar el PC desaparece todo | kind vive dentro de Docker | `kind get clusters`; si no aparece, repetir el paso 4 (el entrenamiento se vuelve a registrar) |
