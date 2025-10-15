#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>

using Counter = std::unordered_map<std::string, uint64_t>;

// ------------ I/O ------------
inline std::vector<char> slurp_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open file: " + path);
    f.seekg(0, std::ios::end);
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> buf(n);
    if (n) f.read(buf.data(), (std::streamsize)n);
    return buf;
}

// Split file [0..N) into 'size' contiguous ranges, then advance interior cut points
// to the next whitespace to avoid splitting tokens.
inline void whitespace_cuts(const std::vector<char>& buf, int size,
                            std::vector<int>& sendcounts, std::vector<int>& displs) {
    const size_t N = buf.size();
    sendcounts.assign(size, 0);
    displs.assign(size, 0);
    if (size <= 0) return;

    std::vector<size_t> cuts(size + 1);
    for (int r = 0; r <= size; ++r) cuts[r] = (N * (size_t)r) / (size_t)size;

    auto is_ws = [](char c){ return std::isspace((unsigned char)c); };
    for (int r = 1; r < size; ++r) {
        size_t i = cuts[r];
        while (i < N && !is_ws(buf[i])) ++i;
        cuts[r] = i;
    }
    for (int r = 0; r < size; ++r) {
        size_t a = cuts[r], b = cuts[r + 1];
        displs[r]    = (int)a;
        sendcounts[r]= (int)(b - a);
    }
}

// ------------ serialization of Counter ------------
inline void serialize_counter(const Counter& m, std::vector<char>& out) {
    out.clear();
    uint64_t sz = m.size();
    out.insert(out.end(), (char*)&sz, (char*)&sz + sizeof(sz));
    for (const auto& kv : m) {
        uint64_t klen = kv.first.size();
        out.insert(out.end(), (char*)&klen, (char*)&klen + sizeof(klen));
        out.insert(out.end(), kv.first.data(), kv.first.data() + klen);
        uint64_t c = kv.second;
        out.insert(out.end(), (char*)&c, (char*)&c + sizeof(c));
    }
}

inline void deserialize_counter(const char* buf, size_t len, Counter& out) {
    out.clear();
    const char* p = buf;
    const char* e = buf + len;
    auto need = [&](size_t n){ if ((size_t)(e - p) < n) throw std::runtime_error("deserialize truncated"); };
    need(sizeof(uint64_t));
    uint64_t sz = *(const uint64_t*)p; p += sizeof(uint64_t);
    for (uint64_t i = 0; i < sz; ++i) {
        need(sizeof(uint64_t));
        uint64_t klen = *(const uint64_t*)p; p += sizeof(uint64_t);
        need(klen + sizeof(uint64_t));
        std::string key(p, p + klen); p += klen;
        uint64_t c = *(const uint64_t*)p; p += sizeof(uint64_t);
        out.emplace(std::move(key), c);
    }
}

// ------------ merging & top-k ------------
inline void merge_into(Counter& dst, const Counter& src) {
    for (const auto& kv : src) dst[kv.first] += kv.second;
}

inline std::vector<std::pair<std::string, uint64_t>> topN(const Counter& c, int N) {
    std::vector<std::pair<std::string, uint64_t>> v(c.begin(), c.end());
    auto by_count = [](auto& a, auto& b){ return a.second > b.second; };
    if ((int)v.size() > N) {
        std::nth_element(v.begin(), v.begin() + N, v.end(), by_count);
        v.resize(N);
    }
    std::sort(v.begin(), v.end(), by_count);
    return v;
}

inline void print_topN(const std::vector<std::pair<std::string, uint64_t>>& v) {
    size_t w = 0;
    for (auto& kv : v) w = std::max(w, kv.first.size());
    for (auto& kv : v) {
        std::cout.width((std::streamsize)w);
        std::cout << std::left << kv.first << "  " << kv.second << "\n";
    }
}
