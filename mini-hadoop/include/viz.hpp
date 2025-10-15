#pragma once
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

inline std::string ascii_bar(double frac, int width) {
    frac = std::max(0.0, std::min(1.0, frac));
    int filled = (int)std::round(frac * width);
    std::string s(width, ' ');
    for (int i = 0; i < filled; ++i) s[i] = '#';
    return s;
}

inline void print_static_bytes(const std::vector<int>& sendcounts, int barw) {
    size_t totalB = 0; for (int b : sendcounts) totalB += (size_t)b;
    std::cerr << "\n[static] per-rank bytes processed\n";
    for (size_t r = 0; r < sendcounts.size(); ++r) {
        double f = totalB ? (double)sendcounts[r] / (double)totalB : 0.0;
        std::cerr << "Rank " << r << " [" << ascii_bar(f, barw) << "]  "
                  << sendcounts[r] << "B\n";
    }
}

inline void print_dynamic_progress(size_t total_bytes,
                                   const std::vector<size_t>& bytes_assigned,
                                   const std::vector<size_t>& bytes_completed,
                                   int barw, int next_idx, int total_chunks) {
    size_t done = std::accumulate(bytes_completed.begin(), bytes_completed.end(), (size_t)0);
    double frac_total = total_bytes ? (double)done / (double)total_bytes : 0.0;
    std::cerr << "\n[dynamic] progress: " << (int)(frac_total * 100.0) << "%  "
              << next_idx << "/" << total_chunks << " chunks\n";
    for (size_t r = 1; r < bytes_assigned.size(); ++r) {
        double f = bytes_assigned[r] ? (double)bytes_completed[r] / (double)bytes_assigned[r] : 0.0;
        std::cerr << "Rank " << r << " [" << ascii_bar(f, barw) << "]  "
                  << bytes_completed[r] << "/" << bytes_assigned[r] << "B\n";
    }
}

inline void print_dynamic_assigned(const std::vector<size_t>& bytes_assigned, int barw) {
    size_t tot = 0; for (size_t r = 1; r < bytes_assigned.size(); ++r) tot += bytes_assigned[r];
    std::cerr << "\n[dynamic] per-rank assigned bytes\n";
    for (size_t r = 1; r < bytes_assigned.size(); ++r) {
        double f = tot ? (double)bytes_assigned[r] / (double)tot : 0.0;
        std::cerr << "Rank " << r << " [" << ascii_bar(f, barw) << "]  "
                  << bytes_assigned[r] << "B\n";
    }
}
