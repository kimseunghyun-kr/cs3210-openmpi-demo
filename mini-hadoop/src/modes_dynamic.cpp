#include <mpi.h>

#include "count.hpp"
#include "utils.hpp"
#include "viz.hpp"
#include <chrono>
#include <iostream>

struct Args {
    std::string mode, path;
    int topN = 20;
    int chunk_lines = 400;
    int bar_width = 50;
};

extern Args parse_args(int rank, int argc, char** argv); // from main.cpp

enum { TAG_WORK=1, TAG_DONE=2, TAG_STOP=3 };

void run_dynamic(const Args& a, int rank, int size) {
    if (size < 2) {
        if (rank == 0) std::cerr << "dynamic mode requires at least 2 ranks.\n";
        return;
    }

    auto t0 = std::chrono::steady_clock::now();

    if (rank == 0) {
        auto buf = slurp_file(a.path);

        // line offsets
        std::vector<size_t> line_starts; line_starts.reserve(1<<20);
        line_starts.push_back(0);
        for (size_t i=0;i<buf.size();++i) if (buf[i]=='\n') line_starts.push_back(i+1);
        if (line_starts.back()!=buf.size()) line_starts.push_back(buf.size());
        size_t num_lines = line_starts.size() - 1;

        struct Chunk { int id; size_t a,b; size_t bytes() const { return b-a; } };
        std::vector<Chunk> chunks; chunks.reserve(num_lines / a.chunk_lines + 2);
        for (size_t i=0; i<num_lines;) {
            size_t j = std::min(num_lines, i + (size_t)a.chunk_lines);
            chunks.push_back({ (int)chunks.size(), line_starts[i], line_starts[j] });
            i = j;
        }
        const int M = (int)chunks.size();

        int next_idx = 0;
        int active = 0;

        std::vector<size_t> bytes_assigned(size, 0), bytes_completed(size, 0);
        size_t total_bytes = buf.size();
        auto last_print = std::chrono::steady_clock::now();

        // prime
        for (int w=1; w<size && next_idx<M; ++w) {
            auto &c = chunks[next_idx++];
            int hdr[2] = { c.id, (int)c.bytes() };
            MPI_Send(hdr, 2, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);
            if (c.bytes()) MPI_Send(buf.data()+c.a, (int)c.bytes(), MPI_CHAR, w, TAG_WORK, MPI_COMM_WORLD);
            bytes_assigned[w] += c.bytes();
            active++;
        }

        Counter global;
        std::vector<char> tmp;

        while (active > 0) {
            MPI_Status st;
            int meta[2]; // [chunk_id, payload_size]
            MPI_Recv(meta, 2, MPI_INT, MPI_ANY_SOURCE, TAG_DONE, MPI_COMM_WORLD, &st);
            const int src = st.MPI_SOURCE;
            const int cid = meta[0], psz = meta[1];

            tmp.resize(psz);
            if (psz) MPI_Recv(tmp.data(), psz, MPI_CHAR, src, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Counter part; if (psz) deserialize_counter(tmp.data(), tmp.size(), part);
            merge_into(global, part);

            // progress
            size_t bytes_this = 0;
            if (cid >= 0 && cid < M) bytes_this = chunks[cid].bytes();
            bytes_completed[src] += bytes_this;

            // assign next or stop
            if (next_idx < M) {
                auto &c = chunks[next_idx++];
                int hdr[2] = { c.id, (int)c.bytes() };
                MPI_Send(hdr, 2, MPI_INT, src, TAG_WORK, MPI_COMM_WORLD);
                if (c.bytes()) MPI_Send(buf.data()+c.a, (int)c.bytes(), MPI_CHAR, src, TAG_WORK, MPI_COMM_WORLD);
                bytes_assigned[src] += c.bytes();
            } else {
                int stop[2] = { -1, 0 };
                MPI_Send(stop, 2, MPI_INT, src, TAG_STOP, MPI_COMM_WORLD);
                active--;
            }

            // occasional dashboard
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print).count() > 250) {
                last_print = now;
                print_dynamic_progress(total_bytes, bytes_assigned, bytes_completed,
                                       a.bar_width, next_idx, M);
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        print_dynamic_assigned(bytes_assigned, a.bar_width);

        auto top = topN(global, a.topN);
        std::cout << "\nTop " << a.topN << " words (dynamic):\n";
        print_topN(top);
        std::cout << "\nTime: " << ms << " ms\n";
    } else {
        // workers
        for (;;) {
            MPI_Status st;
            int hdr[2];
            MPI_Recv(hdr, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            if (st.MPI_TAG == TAG_STOP) break;
            if (st.MPI_TAG == TAG_WORK) {
                int cid = hdr[0], nb = hdr[1];
                std::vector<char> chunk(nb);
                if (nb) MPI_Recv(chunk.data(), nb, MPI_CHAR, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                Counter local = count_chunk_omp(chunk.data(), chunk.size(), omp_get_max_threads());

                std::vector<char> blob;
                serialize_counter(local, blob);
                int meta[2] = { cid, (int)blob.size() };
                MPI_Send(meta, 2, MPI_INT, 0, TAG_DONE, MPI_COMM_WORLD);
                if (!blob.empty()) MPI_Send(blob.data(), (int)blob.size(), MPI_CHAR, 0, TAG_DONE, MPI_COMM_WORLD);
            }
        }
    }
}
