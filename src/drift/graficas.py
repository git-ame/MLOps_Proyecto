"""Gráficas de monitoreo (matplotlib, backend sin pantalla)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def f1_temporal(f1s, ref, inicio_drift, alerta, ruta, retraso=None):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    x = list(range(1, len(f1s) + 1))
    ax.plot(x, f1s, marker="o", color="#1f4e79", lw=2, label="F1 por lote (clase deserta)")
    ax.axhline(ref.f1_base, color="#2e7d32", ls="--", lw=1.2, label=f"F1 base = {ref.f1_base:.2f}")
    ax.axhline(ref.umbral, color="#c62828", ls="--", lw=1.2, label=f"Umbral reentrenar = {ref.umbral:.2f}")
    ax.axvline(inicio_drift + 0.5, color="#757575", ls=":", lw=1.2)
    ax.text(inicio_drift + 0.6, max(f1s) * 0.98, "cambio de regla\n(concept drift)", fontsize=8, va="top", color="#424242")
    if alerta is not None:
        ax.axvline(alerta + 1, color="#c62828", lw=1.6)
        ax.annotate("ALERTA: reentrenar", (alerta + 1, ref.umbral), xytext=(alerta + 1.2, ref.umbral - 0.12),
                    fontsize=8, color="#c62828", arrowprops=dict(arrowstyle="->", color="#c62828"))
    if retraso and alerta is not None:
        ax.axvline(alerta + 1 + retraso, color="#ef6c00", lw=1.2, ls="-.", label=f"Detección con etiquetas +{retraso} lotes")
    ax.set_xlabel("Lote de producción (n=500)"); ax.set_ylabel("F1")
    ax.set_title("Concept drift: F1 del modelo champion en lotes sucesivos")
    ax.set_xticks(x); ax.set_ylim(0, max(f1s) * 1.15); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)


def semaforo_data_drift(res_ok, res_drift, ruta):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=130, sharex=True)
    for ax, (res, titulo) in zip(axes, [(res_ok, "Lote sin drift (VERDE)"), (res_drift, "Lote con drift (ROJO)")]):
        r = res.sort_values("efecto")
        ax.barh(r["variable"], r["efecto"], color=["#c62828" if a else "#2e7d32" for a in r["alerta"]])
        ax.axvline(0.10, color="k", ls="--", lw=1); ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("Tamaño del efecto (KS D / PSI)")
    fig.suptitle("Data drift por variable (rojo = en alerta; línea = umbral de efecto 0.10)", fontsize=10)
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)
