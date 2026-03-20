"""
plot_speedup.py
Calcula speedup y genera gráficas del experimento K-Means.

Uso:
    python3 plot_speedup.py

Entrada:  tiempos.csv  (generado por run_experiment.sh)
Salida:   speedup_2d.png, speedup_3d.png, tiempo_2d.png, tiempo_3d.png

Instalar dependencias si es necesario:
    pip install pandas matplotlib numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import os

# ── Configuración de estilo ───────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "lines.linewidth":   2,
    "lines.markersize":  7,
})

COLORES = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
MARCADORES = ["o", "s", "^", "D", "v"]

# ── Cargar datos ──────────────────────────────────────────────────────────────

CSV_FILE = "tiempos.csv"

if not os.path.exists(CSV_FILE):
    print(f"Error: no se encontró '{CSV_FILE}'")
    print("Ejecuta primero: ./run_experiment.sh")
    sys.exit(1)

df = pd.read_csv(CSV_FILE)
df = df[df["tiempo_seg"] != "NA"]        # eliminar corridas fallidas
df["tiempo_seg"] = df["tiempo_seg"].astype(float)

print(f"Filas cargadas: {len(df)}")
print(df.head())
print()

# ── Calcular promedios de 10 repeticiones ─────────────────────────────────────

# Promedio por (dims, puntos, hilos)
promedios = (
    df.groupby(["dims", "puntos", "hilos"])["tiempo_seg"]
    .mean()
    .reset_index()
    .rename(columns={"tiempo_seg": "tiempo_promedio"})
)

# ── Calcular speedup ──────────────────────────────────────────────────────────

# Speedup = T(serial) / T(paralelo con N hilos)
# El serial tiene hilos=1, que es la línea base

speedup_rows = []

for dims in promedios["dims"].unique():
    for puntos in promedios["puntos"].unique():
        subset = promedios[(promedios["dims"] == dims) & (promedios["puntos"] == puntos)]

        # Tiempo serial (hilos == 1 es la versión serial)
        serial_row = subset[subset["hilos"] == 1]
        if serial_row.empty:
            continue
        t_serial = serial_row["tiempo_promedio"].values[0]

        for _, row in subset.iterrows():
            speedup = t_serial / row["tiempo_promedio"]
            speedup_rows.append({
                "dims":            dims,
                "puntos":          row["puntos"],
                "hilos":           int(row["hilos"]),
                "tiempo_promedio": row["tiempo_promedio"],
                "speedup":         speedup,
            })

speedup_df = pd.DataFrame(speedup_rows)
print("Speedup calculado:")
print(speedup_df.to_string(index=False))
print()

# Guardar tabla de speedup como CSV
speedup_df.to_csv("speedup_resultados.csv", index=False)
print("Tabla de speedup guardada en: speedup_resultados.csv")

# ── Función de graficado ──────────────────────────────────────────────────────

def plot_speedup(dims_val, ax, df_sp):
    """Gráfica de speedup vs número de hilos, una línea por número de puntos."""
    data = df_sp[df_sp["dims"] == dims_val]
    puntos_list = sorted(data["puntos"].unique())

    for i, puntos in enumerate(puntos_list):
        sub = data[data["puntos"] == puntos].sort_values("hilos")
        color   = COLORES[i % len(COLORES)]
        marker  = MARCADORES[i % len(MARCADORES)]
        label   = f"{puntos:,} pts"
        ax.plot(sub["hilos"], sub["speedup"],
                color=color, marker=marker, label=label)

    # Línea de speedup ideal (lineal)
    max_hilos = data["hilos"].max()
    x_ideal = np.linspace(1, max_hilos, 100)
    ax.plot(x_ideal, x_ideal, "k--", linewidth=1.2, alpha=0.5, label="Speedup ideal")

    ax.set_xlabel("Número de hilos")
    ax.set_ylabel("Speedup  (T_serial / T_paralelo)")
    ax.set_title(f"Speedup — {dims_val}D")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


def plot_tiempo(dims_val, ax, df_prom):
    """Gráfica de tiempo promedio vs número de puntos, una línea por hilos."""
    data = df_prom[df_prom["dims"] == dims_val]
    hilos_list = sorted(data["hilos"].unique())

    for i, hilos in enumerate(hilos_list):
        sub = data[data["hilos"] == hilos].sort_values("puntos")
        color  = COLORES[i % len(COLORES)]
        marker = MARCADORES[i % len(MARCADORES)]
        label  = "Serial" if hilos == 1 else f"{hilos} hilos"
        ax.plot(sub["puntos"], sub["tiempo_promedio"],
                color=color, marker=marker, label=label)

    ax.set_xlabel("Número de puntos")
    ax.set_ylabel("Tiempo promedio (s)")
    ax.set_title(f"Tiempo de ejecución — {dims_val}D")
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


# ── Generar gráficas ──────────────────────────────────────────────────────────

for dims_val in [2, 3]:

    # ── Gráfica 1: Speedup vs Hilos ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_speedup(dims_val, ax, speedup_df)
    plt.tight_layout()
    fname = f"speedup_{dims_val}d.png"
    plt.savefig(fname, bbox_inches="tight")
    print(f"Guardada: {fname}")
    plt.close()

    # ── Gráfica 2: Tiempo vs Puntos ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_tiempo(dims_val, ax, promedios)
    plt.tight_layout()
    fname = f"tiempo_{dims_val}d.png"
    plt.savefig(fname, bbox_inches="tight")
    print(f"Guardada: {fname}")
    plt.close()

# ── Gráfica 3: Speedup 2D vs 3D en la misma figura ───────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_speedup(2, axes[0], speedup_df)
plot_speedup(3, axes[1], speedup_df)
plt.suptitle("Comparación de Speedup: 2D vs 3D", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("speedup_comparacion.png", bbox_inches="tight")
print("Guardada: speedup_comparacion.png")
plt.close()

print()
print("═" * 50)
print("✅ Gráficas generadas:")
print("   speedup_2d.png          — speedup vs hilos (2D)")
print("   speedup_3d.png          — speedup vs hilos (3D)")
print("   tiempo_2d.png           — tiempo vs puntos  (2D)")
print("   tiempo_3d.png           — tiempo vs puntos  (3D)")
print("   speedup_comparacion.png — 2D y 3D lado a lado")
print("   speedup_resultados.csv  — tabla completa de speedup")
print("═" * 50)