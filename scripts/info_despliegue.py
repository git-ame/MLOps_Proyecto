"""Lee /info de la API viva y escribe docs/INFO_DESPLIEGUE.md (versión del registro y run desplegados).
Uso: API_URL=http://localhost:8080 python -m scripts.info_despliegue [etiqueta]"""
import os, sys, time, requests
API = os.getenv("API_URL", "http://localhost:8080")
etiqueta = sys.argv[1] if len(sys.argv) > 1 else "clúster Kubernetes"
i = requests.get(f"{API}/info", timeout=10).json()
txt = f"""# Información del despliegue (generada automáticamente)

Generado: {time.strftime('%Y-%m-%d %H:%M:%S')} · entorno: {etiqueta} · endpoint: {API}

| Campo | Valor |
|---|---|
| Modelo registrado | {i['modelo']} |
| Alias servido | {i['alias']} |
| **Versión del registro desplegada** | **v{i['version']}** |
| **Run de MLflow de origen** | `{i['run_id']}` |
| Pod que respondió | {i['pod']} |
| Modelo cargado en | {i['cargado_en']} |
"""
open("docs/INFO_DESPLIEGUE.md", "w").write(txt); print(txt)
