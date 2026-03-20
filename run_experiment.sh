#!/bin/bash
# =============================================================================
# run_experiment.sh
# Experimento de rendimiento: K-Means Serial vs Paralelo
#
# Uso:
#   chmod +x run_experiment.sh
#   ./run_experiment.sh
#
# Requisitos antes de ejecutar:
#   1. Compilar ambos programas:
#        g++ -O2 -o kmeans_serial   kmeans_serial.cpp
#        g++ -O2 -fopenmp -o kmeans_parallel kmeans_parallel.cpp
#   2. Tener los CSV de entrada generados con synthetic_clusters.ipynb
#      en la carpeta ./data/  con el formato:
#        data/puntos_2d_100000.csv
#        data/puntos_2d_200000.csv   ... etc.
#        data/puntos_3d_100000.csv   ... etc.
# =============================================================================

# ── Configuración del experimento ────────────────────────────────────────────

K=5                          # número de clusters (ajusta según tu dataset)
REPETICIONES=10              # promedio de cuántas ejecuciones
RESULTADOS_DIR="./resultados"
TIEMPOS_CSV="tiempos.csv"

# Número de puntos a probar
PUNTOS_LIST=(100000 200000 300000 400000 600000 800000 1000000)

# Detectar cores virtuales del sistema
CORES=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
echo "Cores virtuales detectados: $CORES"

# Configuraciones de hilos según el enunciado
HILOS_LIST=(
    1
    $(( CORES / 2 < 1 ? 1 : CORES / 2 ))
    $CORES
    $(( CORES * 2 ))
)

# Eliminar duplicados en la lista de hilos
HILOS_LIST=($(printf '%s\n' "${HILOS_LIST[@]}" | sort -nu))

echo "Configuraciones de hilos: ${HILOS_LIST[*]}"
echo ""

# ── Preparar directorios y archivo de resultados ─────────────────────────────

mkdir -p "$RESULTADOS_DIR"

# Encabezado del CSV de tiempos
echo "dims,puntos,hilos,repeticion,tiempo_seg" > "$TIEMPOS_CSV"

# ── Función auxiliar: extraer tiempo de la salida del programa ───────────────

# Los programas imprimen: "Tiempo K-Means: X.XXXXXX segundos."
# Esta función extrae el número X.XXXXXX
extract_time() {
    # $1 = salida completa del programa
    echo "$1" | grep "Tiempo K-Means:" | grep -oP '[0-9]+\.[0-9]+'
}

# ── Función: ejecutar N repeticiones y guardar tiempos ───────────────────────

run_and_record() {
    local PROGRAMA=$1     # ./kmeans_serial o ./kmeans_parallel
    local INPUT=$2        # archivo CSV de entrada
    local K_VAL=$3        # número de clusters
    local DIMS=$4         # 2 o 3
    local HILOS=$5        # número de hilos (solo para paralelo)
    local PUNTOS=$6       # número de puntos (para el CSV de resultados)
    local SALIDA=$7       # archivo CSV de salida

    for rep in $(seq 1 $REPETICIONES); do
        if [ "$PROGRAMA" == "./kmeans_serial" ]; then
            OUTPUT=$("$PROGRAMA" "$INPUT" "$K_VAL" "$DIMS" "$SALIDA" 2>&1)
            TIPO="serial"
            HILOS_LOG=1
        else
            OUTPUT=$("$PROGRAMA" "$INPUT" "$K_VAL" "$DIMS" "$HILOS" "$SALIDA" 2>&1)
            TIPO="paralelo"
            HILOS_LOG=$HILOS
        fi

        TIEMPO=$(extract_time "$OUTPUT")

        if [ -z "$TIEMPO" ]; then
            echo "  [WARN] No se pudo extraer tiempo (rep $rep). Salida: $OUTPUT"
            TIEMPO="NA"
        fi

        echo "$DIMS,$PUNTOS,$HILOS_LOG,$rep,$TIEMPO" >> "$TIEMPOS_CSV"
        echo "  [${TIPO}] dims=${DIMS} puntos=${PUNTOS} hilos=${HILOS_LOG} rep=${rep} → ${TIEMPO}s"
    done
}

# ── BUCLE PRINCIPAL ───────────────────────────────────────────────────────────

TOTAL_CONFIG=$(( ${#PUNTOS_LIST[@]} * 2 ))   # 2 = número de dims
CONFIG_ACTUAL=0

for DIMS in 2 3; do
    for PUNTOS in "${PUNTOS_LIST[@]}"; do

        CONFIG_ACTUAL=$(( CONFIG_ACTUAL + 1 ))
        INPUT="./data/puntos_${DIMS}d_${PUNTOS}.csv"

        # Verificar que existe el archivo de entrada
        if [ ! -f "$INPUT" ]; then
            echo "⚠️  Archivo no encontrado: $INPUT — saltando"
            continue
        fi

        echo "════════════════════════════════════════"
        echo "Configuración ${CONFIG_ACTUAL}/${TOTAL_CONFIG}: dims=${DIMS} puntos=${PUNTOS}"
        echo "════════════════════════════════════════"

        SALIDA_SERIAL="${RESULTADOS_DIR}/resultado_serial_${DIMS}d_${PUNTOS}.csv"

        # ── Corridas SERIALES ───────────────────────────────────────────
        echo "▶ Versión SERIAL"
        run_and_record "./kmeans_serial" "$INPUT" "$K" "$DIMS" 1 "$PUNTOS" "$SALIDA_SERIAL"

        # ── Corridas PARALELAS (por cada configuración de hilos) ────────
        for HILOS in "${HILOS_LIST[@]}"; do
            echo "▶ Versión PARALELA — $HILOS hilos"
            SALIDA_PAR="${RESULTADOS_DIR}/resultado_par_${DIMS}d_${PUNTOS}_h${HILOS}.csv"
            run_and_record "./kmeans_parallel" "$INPUT" "$K" "$DIMS" "$HILOS" "$PUNTOS" "$SALIDA_PAR"
        done

        echo ""
    done
done

# ── Resumen final ─────────────────────────────────────────────────────────────

echo "════════════════════════════════════════"
echo "✅ Experimento completo"
echo "   Tiempos guardados en: $TIEMPOS_CSV"
echo "   Resultados en:        $RESULTADOS_DIR/"
echo ""
echo "Siguiente paso:"
echo "   python3 plot_speedup.py"
echo "════════════════════════════════════════"