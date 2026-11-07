# Monitoreo de deriva

Pruebas automatizadas en `tests/test_data_drift.py` y `tests/test_concept_drift.py` (`pytest -q`); gráficas en `evidencias/` vía `python -m scripts.run_drift_report`.

## 1. Data drift (cambia la distribución de las entradas)
| Tipo de variable | Prueba | Por qué |
|---|---|---|
| Numéricas (12) | Kolmogórov-Smirnov 2 muestras | No paramétrica; los montos y días son asimétricos |
| Categóricas (5) | χ² de homogeneidad + PSI | Comparan frecuencias por categoría |

**Umbral (por variable):** `p < 0.01/17` (Bonferroni) **y** efecto ≥ 0.10 (KS D o PSI).
- Bonferroni: 17 pruebas con α=0.01 sin corregir darían ≈ 16 % de falsas alarmas por lote.
- Efecto ≥ 0.10: con n grande cualquier diferencia mínima es "significativa"; 0.10 es la frontera convencional del PSI entre cambio insignificante y moderado.

**Umbral (conjunto):** alerta si ≥ 20 % de las variables (≥ 4 de 17) están en alerta; un cambio aislado no justifica reentrenar.

**Resultados verificados:**
| Lote | Variables en alerta | Resultado |
|---|---|---|
| Sin deriva (n=1000) | 0 / 17 | 🟢 VERDE |
| Con deriva (más maestrías, +30 % precios, 92 % a crédito, mora ×1.8, cambio de sedes) | 12 / 17 | 🔴 ROJO |
| 30 lotes limpios adicionales | 0 falsas alarmas | — |

Gráfica: `evidencias/data_drift_semaforo.png`; detalle: `data_drift_detalle_con_deriva.csv`.

## 2. Concept drift (cambia la relación entradas → deserción)
Simulación: 12 lotes sucesivos de 500 participantes; desde el lote 7 la regla de abandono cambia (la mora deja de predecir; pesan el rendimiento académico y las promociones) **con entradas idénticas en distribución**.

**Criterio de reentrenamiento:** F1 del lote < `F1_base − max(0.05, 3σ)` durante **2 lotes consecutivos**.
- F1_base = 0.558 y σ = 0.052 (bootstrap de lotes de 500 sobre el test) → **umbral = 0.402**.
- 3σ: una caída aislada por azar es improbable; piso 0.05 por si σ resultara muy pequeña.
- 2 consecutivos: separa caída sostenida de un lote atípico (cubierto por una prueba).

**F1 por lote:** 0.56, 0.60, 0.54, 0.63, 0.65, 0.59 │ 0.17, 0.19, 0.26, 0.18, 0.20, 0.25
→ cambio en el lote 7; **alerta en el lote 8** (segundo lote bajo el umbral). Gráfica: `evidencias/concept_drift_f1.png`.

### Retraso de etiquetas
La etiqueta "desertó" se conoce semanas después de la predicción. Con retraso de 3 lotes, el F1 del lote 7 solo se calcula en el lote 10: la alerta llega en el **lote 11** (tres lotes de latencia; línea naranja de la gráfica). Mientras tanto:
1. Los tests de *data drift* **no detectan un concept drift puro** (aquí las entradas no cambian: es una limitación demostrada, no un detalle).
2. Se pueden vigilar proxies sin etiqueta: distribución de probabilidades predichas, tasa de alertas, y la mora a corto plazo (p. ej. "abandono en 30 días" como etiqueta anticipada).
3. Modelo *challenger* en sombra y auditoría muestral de etiquetas tempranas para acortar la latencia.
4. Una decisión de reentrenar se puede apoyar con la ventana más reciente de etiquetas ya disponibles.

### Qué hacer cuando salta la alerta
Reentrenar con datos recientes (`src.train`) → nueva versión en el registro → comparar contra `champion` → mover alias (`scripts/promover.py N`) → `kubectl rollout restart`. El rollback es mover el alias de vuelta.
