// K-Means Clustering —  Paralelo con OpenMP
// Fernando Barba y Nicolas Robles


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
#include <omp.h>


// Estructura para instanciar un punto
// Con su constructor en 2D (z se puede omitir porque inicia en 0) o 3D
struct Point {
    double x, y, z;
    int cluster;

    Point() : x(0.0), y(0.0), z(0.0), cluster(-1) {}
    Point(double x, double y, double z = 0.0)
        : x(x), y(y), z(z), cluster(-1) {}
};


// Funcion para calcular distancia euclidiana entre dos puntos
inline double euclidean_distance(const Point& a, const Point& b, int dims) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;

    // Si es 3D, incluye la coordenada z, si no, solo x e y
    if (dims == 3) {
        double dz = a.z - b.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    return std::sqrt(dx*dx + dy*dy);
}

// Funcion para leer un archivo CSV y cargar los puntos en un vector de Point
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

        // Si la primera linea no es un numero, se lo salta
        if (first_line) {
            first_line = false;
            char c = line[0];
            if (!std::isdigit(c) && c != '-' && c != '+' && c != '.') {
                continue;
            }
        }

        std::stringstream ss(line);
        std::string token;
        Point p;

        // Lee los puntos x, y, y z si es 3D
        if (!std::getline(ss, token, ',')) continue;
        p.x = std::stod(token);

        if (!std::getline(ss, token, ',')) continue;
        p.y = std::stod(token);

        if (dims == 3) {
            if (!std::getline(ss, token, ',')) continue;
            p.z = std::stod(token);
        }

        // Agrega los puntos leidos al vector de puntos
        points.push_back(p);
    }

    file.close();
    return true;
}

// Paso 1: Crea k centroides iniciales seleccionando puntos aleatorios de los datos
// Regresa el vector de centroides iniciales, con su cluster aleatoriamente asignado
// No se paraleliza porque solo se ejecuta una vez y k es chico, entonces no vale la pena el overhead de paralelizarlo
std::vector<Point> initialize_centroids(const std::vector<Point>& points, int k) {
    std::vector<int> indices(points.size());
    for (int i = 0; i < (int)points.size(); i++) indices[i] = i;

    // Asigna de forma random los indices a los puntos de los centroides
    for (int i = 0; i < k; i++) {
        int j = i + rand() % ((int)points.size() - i);
        std::swap(indices[i], indices[j]);
    }

    // Crea un vector de centroides y de los indices aleatorios, les asigna uno
    std::vector<Point> centroids(k);
    for (int i = 0; i < k; i++) {
        centroids[i] = points[indices[i]];
        centroids[i].cluster = i;
    }
    return centroids;
}

// Paso 2: asigna cada punto al centroide mas cercano
// Si se paraleliza porque este es el loop mas pesado del algoritmo porque recorre todos los puntos y los compara con los k centroides
bool assign_clusters(std::vector<Point>& points, const std::vector<Point>& centroids, int dims) {
    bool changed = false;
    int k = (int)centroids.size();
    int n = (int)points.size();

    // Se usa schedule(static) porque todos los puntos hacen el mismo trabajo - Cada hilo tiene su bloque de puntos del mismo tamaño
    // Se usa reduction(||:changed) para que si algun hilo marca un cambio, el resultado final sea true
    for (int i = 0; i < n; i++) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;

        // Para cada centroide, calcula la distancia al punto y actualiza el cluster mas cercano
        for (int c = 0; c < k; c++) {
            double d = euclidean_distance(points[i], centroids[c], dims);
            if (d < min_dist) {
                min_dist = d;
                best_cluster = c;
            }
        }

        // Marca si hubo un cambio de cluster para este punto
        if (points[i].cluster != best_cluster) {
            points[i].cluster = best_cluster;
            changed = true;
        }
    }
    return changed;
}

// Paso 3: actualiza la posicion de cada centroide
// Si se paraleliza porque el loop de acumulacion de sumas es el segundo mas pesado porque recorre todos los puntos
void update_centroids(std::vector<Point>& centroids, const std::vector<Point>& points, int k, int dims) {
    std::vector<double> sum_x(k, 0.0), sum_y(k, 0.0), sum_z(k, 0.0);
    std::vector<int>    count(k, 0);
    int n = (int)points.size();

    // Abre hilos y los paraleliza
    #pragma omp parallel
    {
        // Cada hilo tiene sus propios acumuladores locales para evitar conflictos de escritura
        std::vector<double> local_sum_x(k, 0.0), local_sum_y(k, 0.0), local_sum_z(k, 0.0);
        std::vector<int>    local_count(k, 0);

        // Se reparten los puntos entre los hilos con schedule(static) para que todos tengan el mismo trabajo 
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            int c = points[i].cluster;
            local_sum_x[c] += points[i].x;
            local_sum_y[c] += points[i].y;
            if (dims == 3) local_sum_z[c] += points[i].z;
            local_count[c]++;
        }

        // Cada hilo vacia sus acumuladores locales al global con critical para evitar condiciones de carrera
        #pragma omp critical
        {
            for (int c = 0; c < k; c++) {
                sum_x[c] += local_sum_x[c];
                sum_y[c] += local_sum_y[c];
                if (dims == 3) sum_z[c] += local_sum_z[c];
                count[c] += local_count[c];
            }
        }
    }

    // Como solo son k iteraciones, no tiene caso paralelizar para evitar el overhead
    for (int c = 0; c < k; c++) {

        // Si el cluster esta vacio, se vuelve a asignar con un punto aleatorio
        if (count[c] == 0) {
            int idx = rand() % (int)points.size();
            centroids[c].x = points[idx].x;
            centroids[c].y = points[idx].y;
            if (dims == 3) centroids[c].z = points[idx].z;
        } else {
            // Si no esta vacio, actualiza el centroide al promedio del cluster
            centroids[c].x = sum_x[c] / count[c];
            centroids[c].y = sum_y[c] / count[c];
            if (dims == 3) centroids[c].z = sum_z[c] / count[c];
        }
    }
}

// Funcion para escribir los puntos con su cluster asignado en un archivo CSV
// Para 2D es x,y,cluster y para 3D es x,y,z,cluster --> Separados por ,
bool write_csv(const std::string& filename, const std::vector<Point>& points, int dims) {
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

// Corre K-Means: repite pasos 2 y 3 hasta que ningun punto cambie de cluster
// Regresa el tiempo que tardo en converger
double kmeans(std::vector<Point>& points, int k, int dims) {

    // Paso 1: inicializa los centroides
    std::vector<Point> centroids = initialize_centroids(points, k);

    auto t_start = std::chrono::high_resolution_clock::now();

    int iteration = 0;
    bool changed  = true;

    // Repite el paso 2 y 3 hasta que no haya cambios en los clusters
    while (changed) {
        changed = assign_clusters(points, centroids, dims);

        if (changed) {
            update_centroids(centroids, points, k, dims);
        }

        iteration++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "Convergio en " << iteration << " iteraciones.\n";
    std::cout << "Tiempo de K-Means: " << elapsed << " segundos.\n";

    return elapsed;
}

int main(int argc, char* argv[]) {

    // Verifica que se tengan los 6 argumentos necesarios (agrega num_hilos vs serial)
    if (argc != 6) {
        std::cerr << "Uso: " << argv[0] << " <entrada.csv> <k> <dims> <num_hilos> <salida.csv>\n";
        return 1;
    }

    // Lee los argumentos de la linea de comandos y los asigna a las variables
    std::string input_file  = argv[1];
    int         k           = std::atoi(argv[2]);
    int         dims        = std::atoi(argv[3]);
    int         num_threads = std::atoi(argv[4]);
    std::string output_file = argv[5];

    if (k < 1) {
        std::cerr << "Error: k debe ser >= 1\n";
        return 1;
    }
    if (dims != 2 && dims != 3) {
        std::cerr << "Error: dims debe ser 2 o 3\n";
        return 1;
    }
    if (num_threads < 1) {
        std::cerr << "Error: num_hilos debe ser >= 1\n";
        return 1;
    }

    // Se fija el numero de hilos para OpenMP desde el argumento
    omp_set_num_threads(num_threads);

    // Semilla para generar los primeros centroides aleatorios (se usa el tiempo actual)
    srand((unsigned)time(nullptr));

    std::vector<Point> points;
    std::cout << "Leyendo " << input_file << "...\n";
    if (!read_csv(input_file, points, dims)) return 1;

    std::cout << "Puntos leidos: " << points.size() << "\n";
    std::cout << "k = " << k << ", dims = " << dims << ", hilos = " << num_threads << "\n\n";

    if ((int)points.size() < k) {
        std::cerr << "Error: k (" << k << ") mayor que numero de puntos (" << points.size() << ")\n";
        return 1;
    }

    // Se ejecuta el algoritmo de K-Means hasta que converja
    double elapsed = kmeans(points, k, dims);

    std::cout << "\nGuardando resultado en " << output_file << "...\n";
    if (!write_csv(output_file, points, dims)) return 1;

    std::cout << "Listo. Tiempo total: " << elapsed << " s\n";
    return 0;
}
