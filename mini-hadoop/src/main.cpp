#include <mpi.h>

#include "count.hpp"
#include "utils.hpp"
#include "viz.hpp"
#include <omp.h>
#include <chrono>
#include <iostream>
#include <string>

// shared arg struct (also used in other .cpp)
struct Args {
    std::string mode, path;
    int topN = 20;
    int chunk_lines = 400;
    int bar_width = 50;
};

// Forward decls
void run_static(const Args& a, int rank, int size);
void run_dynamic(const Args& a, int rank, int size);

static void usage(int rank, const char* argv0) {
    if (rank == 0) {
        std::cerr
          << "Usage:\n"
          << "  " << argv0 << " static  <corpus.txt> [--top N]\n"
          << "  " << argv0 << " dynamic <corpus.txt> [--top N] [--chunk-lines M] [--bar-width W]\n";
    }
}

Args parse_args(int rank, int argc, char** argv) {
    Args a;
    if (argc >= 3) { a.mode = argv[1]; a.path = argv[2]; }
    for (int i=3; i<argc; ++i) {
        std::string s = argv[i];
        if (s=="--top" && i+1<argc) a.topN = std::stoi(argv[++i]);
        else if (s=="--chunk-lines" && i+1<argc) a.chunk_lines = std::stoi(argv[++i]);
        else if (s=="--bar-width" && i+1<argc) a.bar_width = std::stoi(argv[++i]);
    }
    if (a.mode!="static" && a.mode!="dynamic") usage(rank, argv[0]);
    return a;
}

// make parse_args visible to other TUs
Args parse_args(int, int, char**);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank=0, size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) { usage(rank, argv[0]); MPI_Finalize(); return 0; }
    Args args = parse_args(rank, argc, argv);

    if (rank == 0) {
        std::cerr << "Hybrid parallelism: " << size
                  << " MPI ranks × " << omp_get_max_threads()
                  << " OpenMP threads\n";
    }

    if (args.mode == "static")      run_static(args, rank, size);
    else if (args.mode == "dynamic") run_dynamic(args, rank, size);
    else usage(rank, argv[0]);

    MPI_Finalize();
    return 0;
}
