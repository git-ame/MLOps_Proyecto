# Guion de defensa (≈ 8 min)
1. **Problema y datos (1 min):** deserción en posgrado, datos sintéticos por confidencialidad; F1 de la clase minoritaria como métrica.
2. **MLflow (2 min):** UI con 6 runs; v1 baseline y v2 champion; la API pide el *alias*. Decir con franqueza que v2 y v1 son equivalentes en CV (ver ARQUITECTURA).
3. **Imagen y despliegue (1 min):** versiones fijadas, misma de MLflow en cliente y servidor; MLflow con 1 réplica por SQLite.
4. **Demos K8s (2 min):** `demo_k8s.sh`: Running → balanceo → borrar pod → escalar.
5. **Deriva (2 min):** semáforo verde/rojo; gráfica de F1 con umbral y alerta; el retraso de etiquetas y por qué data drift no ve un concept drift puro.

Preguntas probables: ¿por qué Bonferroni? ¿por qué 2 lotes consecutivos? ¿qué pasa si no hay etiquetas? ¿por qué MLflow no escala? ¿cómo se hace rollback?
