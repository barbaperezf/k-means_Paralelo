/*
 * K-Means Clustering — Versión Serial
 * Cómputo Paralelo
 *
 * Uso:
 *   ./kmeans_serial <archivo_entrada.csv> <k> <dims> <archivo_salida.csv>
 *
 * Ejemplo 2D:
 *   ./kmeans_serial puntos_2d.csv 5 2 resultado_2d.csv
 *
 * Ejemplo 3D:
 *   ./kmeans_serial puntos_3d.csv 5 3 resultado_3d.csv
 *
 * Formato entrada CSV (2D):  x,y
 * Formato entrada CSV (3D):  x,y,z
 * Formato salida CSV  (2D):  x,y,cluster
 * Formato salida CSV  (3D):  x,y,z,cluster
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

// ─────────────────────────────────────────────
//  ESTRUCTURAS
// ─────────────────────────────────────────────

struct Point {
    double x, y, z;   // coordenadas (z = 0.0 si es 2D)
    int    cluster;    // ID del cluster asignado (-1 = sin asignar)

    Point() : x(0.0), y(0.0), z(0.0), cluster(-1) {}
    Point(double x, double y, double z = 0.0)
        : x(x), y(y), z(z), cluster(-1) {}
};

// ─────────────────────────────────────────────
//  FUNCIÓN DE DISTANCIA EUCLIDIANA
// ─────────────────────────────────────────────

/*
 * Distancia euclidiana entre un punto y un centroide.
 * Si dims == 2 ignora la componente z.
 */
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
//  LECTURA DEL CSV DE ENTRADA
// ─────────────────────────────────────────────

/*
 * Lee un archivo CSV con puntos 2D (x,y) o 3D (x,y,z).
 * Ignora la primera línea si es un encabezado (detectado
 * porque la primera columna no es un número).
 * Retorna false si no pudo abrir el archivo.
 */
bool read_csv(const std::string& filename, std::vector<Point>& points, int dims) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: no se pudo abrir el archivo " << filename << "\n";
        return false;
    }

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // Detectar encabezado: si la primera celda no es número, saltarla
        if (first_line) {
            first_line = false;
            char c = line[0];
            if (!std::isdigit(c) && c != '-' && c != '+' && c != '.') {
                continue; // es encabezado, omitir
            }
        }

        std::stringstream ss(line);
        std::string token;
        Point p;

        // Leer x
        if (!std::getline(ss, token, ',')) continue;
        p.x = std::stod(token);

        // Leer y
        if (!std::getline(ss, token, ',')) continue;
        p.y = std::stod(token);

        // Leer z si es 3D
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

/*
 * Selecciona k puntos aleatorios del dataset como centroides iniciales.
 * Usamos muestreo sin reemplazo para evitar centroides duplicados.
 */
std::vector<Point> initialize_centroids(const std::vector<Point>& points, int k) {
    std::vector<int> indices(points.size());
    for (int i = 0; i < (int)points.size(); i++) indices[i] = i;

    // Mezcla aleatoria (Fisher-Yates parcial)
    for (int i = 0; i < k; i++) {
        int j = i + rand() % ((int)points.size() - i);
        std::swap(indices[i], indices[j]);
    }

    std::vector<Point> centroids(k);
    for (int i = 0; i < k; i++) {
        centroids[i] = points[indices[i]];
        centroids[i].cluster = i; // el centroide lleva su propio ID
    }
    return centroids;
}

// ─────────────────────────────────────────────
//  PASO 2: ASIGNACIÓN DE PUNTOS AL CENTROIDE MÁS CERCANO
// ─────────────────────────────────────────────

/*
 * Para cada punto, calcula la distancia euclidiana a todos los centroides
 * y asigna el punto al centroide más cercano.
 *
 * Retorna true si ALGÚN punto cambió de cluster (no convergió aún).
 * Retorna false si NINGÚN punto cambió (convergencia).
 */
bool assign_clusters(std::vector<Point>& points,
                     const std::vector<Point>& centroids,
                     int dims) {
    bool changed = false;
    int k = (int)centroids.size();

    for (auto& p : points) {
        double min_dist = std::numeric_limits<double>::max();
        int    best_cluster = 0;

        for (int c = 0; c < k; c++) {
            double d = euclidean_distance(p, centroids[c], dims);
            if (d < min_dist) {
                min_dist = d;
                best_cluster = c;
            }
        }

        if (p.cluster != best_cluster) {
            p.cluster = best_cluster;
            changed = true;
        }
    }
    return changed;
}

// ─────────────────────────────────────────────
//  PASO 3: ACTUALIZACIÓN DE CENTROIDES
// ─────────────────────────────────────────────

/*
 * Recalcula la posición de cada centroide como el promedio
 * de todos los puntos asignados a él.
 *
 * Si algún centroide quedó vacío (sin puntos asignados),
 * lo reposiciona en un punto aleatorio para evitar NaN.
 */
void update_centroids(std::vector<Point>& centroids,
                      const std::vector<Point>& points,
                      int k,
                      int dims) {
    // Acumuladores: suma de coordenadas y conteo por cluster
    std::vector<double> sum_x(k, 0.0), sum_y(k, 0.0), sum_z(k, 0.0);
    std::vector<int>    count(k, 0);

    for (const auto& p : points) {
        int c = p.cluster;
        sum_x[c] += p.x;
        sum_y[c] += p.y;
        if (dims == 3) sum_z[c] += p.z;
        count[c]++;
    }

    for (int c = 0; c < k; c++) {
        if (count[c] == 0) {
            // Centroide vacío: reasignar a un punto aleatorio
            int idx = rand() % (int)points.size();
            centroids[c].x = points[idx].x;
            centroids[c].y = points[idx].y;
            if (dims == 3) centroids[c].z = points[idx].z;
        } else {
            centroids[c].x = sum_x[c] / count[c];
            centroids[c].y = sum_y[c] / count[c];
            if (dims == 3) centroids[c].z = sum_z[c] / count[c];
        }
    }
}

// ─────────────────────────────────────────────
//  ESCRITURA DEL CSV DE SALIDA
// ─────────────────────────────────────────────

/*
 * Escribe el resultado en un CSV con el ID de cluster añadido.
 * Formato 2D: x,y,cluster
 * Formato 3D: x,y,z,cluster
 */
bool write_csv(const std::string& filename,
               const std::vector<Point>& points,
               int dims) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: no se pudo crear " << filename << "\n";
        return false;
    }

    // Encabezado
    if (dims == 2) file << "x,y,cluster\n";
    else           file << "x,y,z,cluster\n";

    // Datos
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
//  FUNCIÓN PRINCIPAL DE K-MEANS
// ─────────────────────────────────────────────

double kmeans(std::vector<Point>& points, int k, int dims) {
    // Inicializar centroides aleatoriamente (Paso 1)
    std::vector<Point> centroids = initialize_centroids(points, k);

    auto t_start = std::chrono::high_resolution_clock::now();

    int iteration = 0;
    bool changed  = true;

    while (changed) {
        // Paso 2: asignar puntos al centroide más cercano
        changed = assign_clusters(points, centroids, dims);

        // Paso 3: actualizar posición de los centroides
        if (changed) {
            update_centroids(centroids, points, k, dims);
        }

        iteration++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "Convergió en " << iteration << " iteraciones.\n";
    std::cout << "Tiempo de K-Means: " << elapsed << " segundos.\n";

    return elapsed;
}

// ─────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // ── Validar argumentos ──────────────────────
    if (argc != 5) {
        std::cerr << "Uso: " << argv[0]
                  << " <entrada.csv> <k> <dims> <salida.csv>\n";
        std::cerr << "  dims: 2 o 3\n";
        return 1;
    }

    std::string input_file  = argv[1];
    int         k           = std::atoi(argv[2]);
    int         dims        = std::atoi(argv[3]);
    std::string output_file = argv[4];

    if (k < 1) {
        std::cerr << "Error: k debe ser >= 1\n";
        return 1;
    }
    if (dims != 2 && dims != 3) {
        std::cerr << "Error: dims debe ser 2 o 3\n";
        return 1;
    }

    // ── Semilla aleatoria ──────────────────────
    srand((unsigned)time(nullptr));

    // ── Paso 0: Leer datos ─────────────────────
    std::vector<Point> points;
    std::cout << "Leyendo " << input_file << "...\n";
    if (!read_csv(input_file, points, dims)) return 1;

    std::cout << "Puntos leídos: " << points.size() << "\n";
    std::cout << "k = " << k << ", dims = " << dims << "\n\n";

    if ((int)points.size() < k) {
        std::cerr << "Error: k (" << k << ") mayor que número de puntos ("
                  << points.size() << ")\n";
        return 1;
    }

    // ── Ejecutar K-Means ───────────────────────
    double elapsed = kmeans(points, k, dims);

    // ── Guardar resultado ──────────────────────
    std::cout << "\nGuardando resultado en " << output_file << "...\n";
    if (!write_csv(output_file, points, dims)) return 1;

    std::cout << "Listo. Tiempo total: " << elapsed << " s\n";
    return 0;
}