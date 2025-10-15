#include <mpi.h>

#include "count.hpp"
#include "utils.hpp"
#include "viz.hpp"
#include <chrono>
#include <iostream>

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
    // ------------------------------------------------------------------------
    // (1) MPI_Scatter
    // ------------------------------------------------------------------------
    // Distribute the number of bytes each rank should process.
    // Rank 0 sends one integer (sendcounts[r]) to each rank.
    // Every rank receives its own 'mycount' (local chunk size).
    // Effectively, each process learns how many bytes it will receive next.
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &mycount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<char> mychunk(mycount);
    // ------------------------------------------------------------------------
    // (2) MPI_Scatterv
    // ------------------------------------------------------------------------
    // Distribute the actual file data.
    //
    // - On rank 0: 'filebuf' contains the entire file.
    //   The 'sendcounts' array defines how many bytes go to each rank.
    //   The 'displs' array defines the starting offset for each rank’s data.
    //
    // - On all other ranks: the receive buffer 'mychunk' will be filled with
    //   exactly 'mycount' bytes assigned to that process.
    //
    // Result: Each rank now has its local text segment to process independently.
    MPI_Scatterv(rank == 0 ? filebuf.data() : nullptr, sendcounts.data(), displs.data(),
                 MPI_CHAR, mychunk.data(), mycount, MPI_CHAR, 0, MPI_COMM_WORLD);

    // OpenMP counting
    Counter local = count_chunk_omp(mychunk.data(), mychunk.size(), omp_get_max_threads());

    // Serialize & variable-size gather to rank 0
    std::vector<char> blob;
    serialize_counter(local, blob);
    int mysz = (int)blob.size();

    std::vector<int> sizes(size, 0), disps(size, 0);

    // ------------------------------------------------------------------------
    // (3) MPI_Gather
    // ------------------------------------------------------------------------
    // First gather the size of each worker's serialized result.
    //
    // - Each rank sends its local 'mysz' (result byte size) to the master.
    // - Rank 0 collects all these sizes into 'sizes'.
    //
    // This allows rank 0 to compute exact displacements for the next step.
    MPI_Gather(&mysz, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<char> recvbuf;
    if (rank == 0) {
        int total = 0; for (int s : sizes) total += s;
        recvbuf.resize(total);
        for (int r = 1; r < size; ++r) disps[r] = disps[r-1] + sizes[r-1];
    }

    // ------------------------------------------------------------------------
    // (4) MPI_Gatherv
    // ------------------------------------------------------------------------
    // Collect the actual serialized word count results.
    //
    // - Each worker sends its 'blob' (serialized Counter).
    // - Rank 0 receives variable-sized blobs, placed contiguously into 'recvbuf'.
    //   The 'sizes' array gives the byte length per rank,
    //   and 'disps' gives the byte offset for each.
    //
    // After this, rank 0 holds *all* partial word count results in 'recvbuf',
    // ready to be deserialized and merged.
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
