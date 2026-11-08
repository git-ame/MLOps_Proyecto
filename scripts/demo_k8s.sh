#!/usr/bin/env bash
# Las 4 demostraciones obligatorias. Guarda la salida en evidencias/ (ejecutar y capturar pantalla).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=evidencias/demo_k8s.txt; : > "$OUT"
say() { echo -e "\n=== $* ===" | tee -a "$OUT"; }
BODY=$(python - <<'PY'
import json
from src import datos, config
f=datos.generar(1,seed=11)[config.FEATURES].iloc[0].to_dict()
print(json.dumps({k:(v.item() if hasattr(v,"item") else v) for k,v in f.items()}))
PY
)

say "1) Réplicas en estado Running"
kubectl get deployment desercion-api | tee -a "$OUT"; kubectl get pods -l app=desercion-api -o wide | tee -a "$OUT"

say "2) Balanceo de carga: 12 peticiones, nombre del pod que responde"
for i in $(seq 1 12); do curl -s -X POST localhost:8080/predict -H 'content-type: application/json' -d "$BODY" | python -c "import sys,json;print(json.load(sys.stdin)['pod'])"; done | sort | uniq -c | tee -a "$OUT"

say "3) Autorreparación: se elimina un pod y el Deployment lo recrea"
V=$(kubectl get pods -l app=desercion-api -o jsonpath='{.items[0].metadata.name}')
echo "Eliminando $V" | tee -a "$OUT"; kubectl delete pod "$V" --wait=false | tee -a "$OUT"
sleep 3; kubectl get pods -l app=desercion-api | tee -a "$OUT"
kubectl wait --for=condition=Ready pod -l app=desercion-api --timeout=120s | tee -a "$OUT"
kubectl get pods -l app=desercion-api | tee -a "$OUT"

say "4) Escalado: 3 -> 5 -> 3 réplicas"
kubectl scale deployment/desercion-api --replicas=5 | tee -a "$OUT"
kubectl rollout status deployment/desercion-api --timeout=180s | tee -a "$OUT"; kubectl get pods -l app=desercion-api | tee -a "$OUT"
kubectl scale deployment/desercion-api --replicas=3 | tee -a "$OUT"
sleep 5; kubectl get pods -l app=desercion-api | tee -a "$OUT"
echo -e "\nEvidencia guardada en $OUT"
