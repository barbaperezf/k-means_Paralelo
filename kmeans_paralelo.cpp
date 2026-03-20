/*
 * K-Means Clustering — Versión Paralela con OpenMP
 * Cómputo Paralelo
 *
 * Uso:
 *   ./kmeans_parallel <entrada.csv> <k> <dims> <hilos> <salida.csv>
 *
 * Ejemplo 2D con 8 hilos:
 *   ./kmeans_parallel puntos_2d.csv 5 2 8 resultado_2d.csv
 *
 * Compilar:
 *   g++ -O2 -fopenmp -o kmeans_parallel kmeans_parallel.cpp
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <limits>
#include <algorithm>
#include <omp.h>           // ← OpenMP

// ─────────────────────────────────────────────
//  ESTRUCTURAS  (idénticas a la versión serial)
// ─────────────────────────────────────────────

struct Point {
    double x, y, z;
    int    cluster;

    Point() : x(0.0), y(0.0), z(0.0), cluster(-1) {}
    Point(double x, double y, double z = 0.0)
        : x(x), y(y), z(z), cluster(-1) {}
};

// ─────────────────────────────────────────────
//  DISTANCIA EUCLIDIANA
// ─────────────────────────────────────────────

inline double euclidean_distance(const Point& a, const Point& b, int dims) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    if (dims == 3) {
        double dz = a.z - b.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    return std::sqrt(dx*dx + dy*dy);
}

// ─────────────────────────────────────────────
//  LECTURA CSV
// ─────────────────────────────────────────────

bool read_csv(const std::string& filename, std::vector<Point>& points, int dims) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: no se pudo abrir " << filename << "\n";
        return false;
    }
    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (first_line) {
            first_line = false;
            char c = line[0];
            if (!std::isdigit(c) && c != '-' && c != '+' && c != '.') continue;
        }
        std::stringstream ss(line);
        std::string token;
        Point p;
        if (!std::getline(ss, token, ',')) continue;
        p.x = std::stod(token);
        if (!std::getline(ss, token, ',')) continue;
        p.y = std::stod(token);
        if (dims == 3) {
            if (!std::getline(ss, token, ',')) continue;
            p.z = std::stod(token);
        }
        points.push_back(p);
    }
    file.close();
    return true;
}

// ─────────────────────────────────────────────
//  PASO 1: INICIALIZACIÓN DE CENTROIDES
// ─────────────────────────────────────────────

std::vector<Point> initialize_centroids(const std::vector<Point>& points, int k) {
    std::vector<int> indices(points.size());
    for (int i = 0; i < (int)points.size(); i++) indices[i] = i;
    for (int i = 0; i < k; i++) {
        int j = i + rand() % ((int)points.size() - i);
        std::swap(indices[i], indices[j]);
    }
    std::vector<Point> centroids(k);
    for (int i = 0; i < k; i++) {
        centroids[i] = points[indices[i]];
        centroids[i].cluster = i;
    }
    return centroids;
}

// ─────────────────────────────────────────────
//  PASO 2 PARALELO: ASIGNACIÓN DE PUNTOS
// ─────────────────────────────────────────────

/*
 * ¿Por qué es seguro paralelizar aquí?
 *
 * Cada iteración del for trabaja sobre points[i] de forma INDEPENDIENTE:
 *   - Lee   : centroids[]  →  solo lectura, sin modificación → seguro compartir
 *   - Escribe: points[i].cluster → cada hilo escribe en su propio índice i
 *
 * No hay dos hilos que escriban en la misma posición de memoria.
 * → NO hay condición de carrera → #pragma omp parallel for es suficiente.
 *
 * changed se acumula con reduction(||:changed) para combinar el booleano
 * de todos los hilos de forma segura sin sección crítica.
 */
bool assign_clusters_parallel(std::vector<Point>& points,
                               const std::vector<Point>& centroids,
                               int dims) {
    bool changed = false;
    int  k       = (int)centroids.size();
    int  n       = (int)points.size();

    #pragma omp parallel for schedule(static) reduction(||:changed)
    for (int i = 0; i < n; i++) {
        double min_dist    = std::numeric_limits<double>::max();
        int    best_cluster = 0;

        for (int c = 0; c < k; c++) {
            double d = euclidean_distance(points[i], centroids[c], dims);
            if (d < min_dist) {
                min_dist    = d;
                best_cluster = c;
            }
        }

        if (points[i].cluster != best_cluster) {
            points[i].cluster = best_cluster;
            changed = true;         // reduction(||) lo combina al final
        }
    }
    return changed;
}

// ─────────────────────────────────────────────
//  PASO 3 PARALELO: ACTUALIZACIÓN DE CENTROIDES
// ─────────────────────────────────────────────

/*
 * ¿Por qué es peligroso paralelizar sin cuidado?
 *
 * Todos los hilos hacen   sum_x[p.cluster] += p.x
 * Si dos hilos tienen puntos del mismo cluster, ambos leen y escriben
 * sum_x[c] al mismo tiempo → condición de carrera → resultado incorrecto.
 *
 * SOLUCIÓN: arrays locales por hilo
 * ──────────────────────────────────
 * Cada hilo tiene su propia copia de sum_x, sum_y, sum_z, count.
 * Cada hilo acumula sus puntos de forma completamente privada.
 * Al final, en una sección #pragma omp critical, cada hilo suma
 * su acumulador local al acumulador global (una operación rápida).
 * El hilo principal calcula los promedios finales.
 *
 * Esto es equivalente al patrón "map + reduce":
 *   map:    cada hilo suma su porción del dataset
 *   reduce: se combinan los resultados parciales
 */
void update_centroids_parallel(std::vector<Point>& centroids,
                                const std::vector<Point>& points,
                                int k, int dims) {
    // Acumuladores GLOBALES (compartidos, solo el hilo principal los lee al final)
    std::vector<double> global_sum_x(k, 0.0);
    std::vector<double> global_sum_y(k, 0.0);
    std::vector<double> global_sum_z(k, 0.0);
    std::vector<int>    global_count(k, 0);

    int n = (int)points.size();

    #pragma omp parallel
    {
        // Acumuladores LOCALES: privados para cada hilo (en su stack)
        std::vector<double> local_sum_x(k, 0.0);
        std::vector<double> local_sum_y(k, 0.0);
        std::vector<double> local_sum_z(k, 0.0);
        std::vector<int>    local_count(k, 0);

        // Cada hilo procesa su porción de puntos (sin sincronización)
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            int c = points[i].cluster;
            local_sum_x[c] += points[i].x;
            local_sum_y[c] += points[i].y;
            if (dims == 3) local_sum_z[c] += points[i].z;
            local_count[c]++;
        }

        // Combinar locales → globales en sección crítica
        // Solo un hilo entra a la vez, pero la operación es O(k), no O(n)
        // → el cuello de botella es mínimo
        #pragma omp critical
        {
            for (int c = 0; c < k; c++) {
                global_sum_x[c] += local_sum_x[c];
                global_sum_y[c] += local_sum_y[c];
                if (dims == 3) global_sum_z[c] += local_sum_z[c];
                global_count[c] += local_count[c];
            }
        }
    } // fin región paralela — barrera implícita aquí

    // Calcular promedios (hilo principal, O(k))
    for (int c = 0; c < k; c++) {
        if (global_count[c] == 0) {
            // Centroide vacío: reubicar aleatoriamente
            int idx = rand() % n;
            centroids[c].x = points[idx].x;
            centroids[c].y = points[idx].y;
            if (dims == 3) centroids[c].z = points[idx].z;
        } else {
            centroids[c].x = global_sum_x[c] / global_count[c];
            centroids[c].y = global_sum_y[c] / global_count[c];
            if (dims == 3) centroids[c].z = global_sum_z[c] / global_count[c];
        }
    }
}

// ─────────────────────────────────────────────
//  ESCRITURA CSV
// ─────────────────────────────────────────────

bool write_csv(const std::string& filename,
               const std::vector<Point>& points,
               int dims) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: no se pudo crear " << filename << "\n";
        return false;
    }
    if (dims == 2) file << "x,y,cluster\n";
    else           file << "x,y,z,cluster\n";
    file << std::fixed;
    for (const auto& p : points) {
        if (dims == 2)
            file << p.x << "," << p.y << "," << p.cluster << "\n";
        else
            file << p.x << "," << p.y << "," << p.z << "," << p.cluster << "\n";
    }
    file.close();
    return true;
}

// ─────────────────────────────────────────────
//  FUNCIÓN PRINCIPAL DE K-MEANS PARALELO
// ─────────────────────────────────────────────

double kmeans_parallel(std::vector<Point>& points, int k, int dims, int num_threads) {
    // Configurar número de hilos
    omp_set_num_threads(num_threads);

    // Paso 1: inicializar centroides (serial, solo al inicio)
    std::vector<Point> centroids = initialize_centroids(points, k);

    auto t_start = std::chrono::high_resolution_clock::now();

    int  iteration = 0;
    bool changed   = true;

    while (changed) {
        // ── Paso 2: asignación paralela ──────────
        changed = assign_clusters_parallel(points, centroids, dims);

        // ── Paso 3: actualización paralela ───────
        if (changed) {
            update_centroids_parallel(centroids, points, k, dims);
        }

        iteration++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "Convergió en " << iteration << " iteraciones.\n";
    std::cout << "Hilos usados : " << num_threads << "\n";
    std::cout << "Tiempo K-Means: " << elapsed << " segundos.\n";

    return elapsed;
}

// ─────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Uso: " << argv[0]
                  << " <entrada.csv> <k> <dims> <hilos> <salida.csv>\n";
        std::cerr << "  dims : 2 o 3\n";
        std::cerr << "  hilos: número de hilos OpenMP\n";
        return 1;
    }

    std::string input_file  = argv[1];
    int         k           = std::atoi(argv[2]);
    int         dims        = std::atoi(argv[3]);
    int         num_threads = std::atoi(argv[4]);
    std::string output_file = argv[5];

    if (k < 1)              { std::cerr << "Error: k debe ser >= 1\n";         return 1; }
    if (dims != 2 && dims != 3) { std::cerr << "Error: dims debe ser 2 o 3\n"; return 1; }
    if (num_threads < 1)    { std::cerr << "Error: hilos debe ser >= 1\n";     return 1; }

    srand((unsigned)time(nullptr));

    // ── Paso 0: Leer datos ─────────────────────
    std::vector<Point> points;
    std::cout << "Leyendo " << input_file << "...\n";
    if (!read_csv(input_file, points, dims)) return 1;

    int cores = omp_get_max_threads();
    std::cout << "Puntos leídos  : " << points.size()  << "\n";
    std::cout << "k = " << k << "  dims = " << dims    << "\n";
    std::cout << "Cores virtuales: " << cores           << "\n";
    std::cout << "Hilos pedidos  : " << num_threads     << "\n\n";

    if ((int)points.size() < k) {
        std::cerr << "Error: k mayor que número de puntos\n";
        return 1;
    }

    // ── Ejecutar K-Means paralelo ──────────────
    double elapsed = kmeans_parallel(points, k, dims, num_threads);

    // ── Guardar resultado ──────────────────────
    std::cout << "\nGuardando resultado en " << output_file << "...\n";
    if (!write_csv(output_file, points, dims)) return 1;

    std::cout << "Listo. Tiempo total: " << elapsed << " s\n";
    return 0;
}