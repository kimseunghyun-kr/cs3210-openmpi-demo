#include "count.hpp"
#include "utils.hpp"
#include "viz.hpp"
#include <chrono>
#include <iostream>
extern "C" {
#include <mpi.h>
}
struct Args {
    std::string mode, path;
    int topN = 20;
    int chunk_lines = 400; // unused in static
    int bar_width = 50;
};

extern Args parse_args(int rank, int argc, char** argv); // from main.cpp

void run_static(const Args& a, int rank, int size) {
    auto t0 = std::chrono::steady_clock::now();

    std::vector<int> sendcounts(size, 0), displs(size, 0);
    std::vector<char> filebuf;

    if (rank == 0) {
        filebuf = slurp_file(a.path);
        whitespace_cuts(filebuf, size, sendcounts, displs);
    }

    int mycount = 0;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &mycount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<char> mychunk(mycount);
    MPI_Scatterv(rank == 0 ? filebuf.data() : nullptr, sendcounts.data(), displs.data(),
                 MPI_CHAR, mychunk.data(), mycount, MPI_CHAR, 0, MPI_COMM_WORLD);

    // OpenMP counting
    Counter local = count_chunk_omp(mychunk.data(), mychunk.size(), omp_get_max_threads());

    // Serialize & variable-size gather to rank 0
    std::vector<char> blob;
    serialize_counter(local, blob);
    int mysz = (int)blob.size();

    std::vector<int> sizes(size, 0), disps(size, 0);
    MPI_Gather(&mysz, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<char> recvbuf;
    if (rank == 0) {
        int total = 0; for (int s : sizes) total += s;
        recvbuf.resize(total);
        for (int r = 1; r < size; ++r) disps[r] = disps[r-1] + sizes[r-1];
    }

    MPI_Gatherv(blob.data(), mysz, MPI_CHAR,
                rank == 0 ? recvbuf.data() : nullptr,
                sizes.data(), disps.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        Counter global;
        // merge self
        merge_into(global, local);
        // merge others
        for (int r = 1; r < size; ++r) {
            if (sizes[r] == 0) continue;
            Counter tmp;
            deserialize_counter(recvbuf.data() + disps[r], sizes[r], tmp);
            merge_into(global, tmp);
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        print_static_bytes(sendcounts, a.bar_width);

        auto top = topN(global, a.topN);
        std::cout << "\nTop " << a.topN << " words (static):\n";
        print_topN(top);
        std::cout << "\nTime: " << ms << " ms\n";
    }
}
