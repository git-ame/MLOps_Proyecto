"""Data drift: el test debe quedar VERDE sin deriva y ROJO con deriva (prueba por tipo de variable)."""
from src import config, datos
from src.drift import data_drift as dd


def test_sin_drift_es_verde(baseline):
    res = dd.evaluar(baseline, datos.generar(1000, seed=2001))
    assert not dd.hay_drift(res)
    assert res["alerta"].sum() == 0


def test_pocos_falsos_positivos(baseline):
    """30 lotes limpios: la tasa de falsas alarmas del criterio compuesto debe ser ~0."""
    falsas = sum(dd.hay_drift(dd.evaluar(baseline, datos.generar(1000, seed=3000 + i))) for i in range(30))
    assert falsas == 0


def test_con_drift_es_rojo(baseline):
    res = dd.evaluar(baseline, datos.generar(1000, seed=2002, deriva_entradas=True))
    assert dd.hay_drift(res)
    assert res["alerta"].sum() >= 4


def test_se_usa_la_prueba_correcta_por_tipo(baseline):
    res = dd.evaluar(baseline, datos.generar(500, seed=5)).set_index("variable")
    assert all(res.loc[v, "prueba"] == "KS" for v in config.FEATURES_NUM)
    assert all(res.loc[v, "prueba"] == "chi2" for v in config.FEATURES_CAT)


def test_bonferroni_aplicado(baseline):
    res = dd.evaluar(baseline, datos.generar(500, seed=5))
    assert abs(res.attrs["alpha_ajustado"] - dd.ALPHA / len(config.FEATURES)) < 1e-12
