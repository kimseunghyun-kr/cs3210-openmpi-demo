#pragma once
#include "utils.hpp"
#include <cctype>
#include <omp.h>
#include <string>
#include <vector>

inline bool is_word_char(unsigned char c) {
    return std::isalnum(c) || c=='\''; // simple heuristic; keep apostrophes
}
inline char lower_char(unsigned char c) { return (char)std::tolower(c); }

inline void count_words_span(const char* data, size_t len, Counter& out) {
    const char* p = data;
    const char* end = data + len;
    std::string tok; tok.reserve(32);
    while (p < end) {
        unsigned char c = (unsigned char)*p++;
        if (is_word_char(c)) {
            tok.push_back(lower_char(c));
        } else if (!tok.empty()) {
            out[std::move(tok)]++; tok.clear(); tok.reserve(32);
        }
    }
    if (!tok.empty()) out[std::move(tok)]++;
}

// Hybrid: split chunk by threads, count per-thread, merge
inline Counter count_chunk_omp(const char* data, size_t n, int nthreads) {
    if (n == 0) return {};
    if (nthreads <= 0) nthreads = 1;
    std::vector<Counter> locals((size_t)nthreads);

#pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        size_t start = (n * (size_t)tid) / (size_t)nthreads;
        size_t end   = (n * (size_t)(tid + 1)) / (size_t)nthreads;

        // Expand boundaries to avoid cutting a token in half
        // Move start to next non-word boundary (unless tid==0)
        if (tid != 0) {
            while (start < end && is_word_char((unsigned char)data[start])) ++start;
        }
        // Move end backward to previous non-word boundary (unless last thread)
        if (tid != nthreads - 1) {
            while (end > start && is_word_char((unsigned char)data[end - 1])) --end;
        }

        if (start < end) count_words_span(data + start, end - start, locals[(size_t)tid]);
    }

    Counter merged;
    for (auto& m : locals) merge_into(merged, m);
    return merged;
}
