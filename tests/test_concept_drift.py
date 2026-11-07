"""Concept drift: lotes sucesivos, criterio de reentrenamiento y retraso de etiquetas."""
from src.drift import concept_drift as cd


def test_sin_drift_no_alerta(modelo, particion):
    _, X_te, _, y_te = particion
    ref = cd.estimar_referencia(modelo, X_te, y_te)
    f1s = cd.f1_por_lote(modelo, cd.generar_lotes(n_lotes=8, inicio_drift=99))
    assert cd.primer_lote_alerta(f1s, ref.umbral) is None


def test_con_drift_alerta_tras_el_cambio(modelo, particion):
    _, X_te, _, y_te = particion
    ref = cd.estimar_referencia(modelo, X_te, y_te)
    f1s = cd.f1_por_lote(modelo, cd.generar_lotes(n_lotes=12, inicio_drift=6))
    idx = cd.primer_lote_alerta(f1s, ref.umbral)
    assert idx is not None and 6 <= idx <= 8          # no antes del cambio; dentro de 2-3 lotes
    assert max(f1s[6:]) < ref.umbral                  # todos los lotes post-cambio por debajo


def test_un_lote_atipico_no_dispara_alerta():
    assert cd.primer_lote_alerta([0.56, 0.20, 0.57, 0.55], umbral=0.40) is None


def test_retraso_de_etiquetas_retrasa_la_deteccion():
    f1s = [0.56, 0.58, 0.2, 0.2, 0.2]
    sin = cd.evaluar_con_retraso(f1s, 0.4, retraso=0)
    con = cd.evaluar_con_retraso(f1s, 0.4, retraso=3)
    assert con == sin + 3
