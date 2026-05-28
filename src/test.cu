#include <vector>
#include <unordered_set>
#include <random>
#include <thread>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cmath>

struct RD3BinResult {
    // Probabilities over sampled reuse events:
    // bin0: rd <= 448 (tile)
    // bin1: 448 < rd <= 1179648 (mid)
    // bin2: rd > 1179648 (far)
    double p_tile = 0.0;
    double p_mid  = 0.0;
    double p_far  = 0.0;

    long long sampled_cnt = 0;      // how many reuse events sampled & classified
    long long total_reuse_cnt = 0;  // how many reuse events exist (old!=-1)
};

// ---- classify one backward reuse pair (old -> k) into 3 bins using unordered_set ----
// rd = number of DISTINCT sectors in (old+1 .. k-1)
//
// Early stop:
// - if distinct > far_th => bin2 (far), stop scanning
//
// Finally:
// - if distinct <= tile_th => bin0 (tile)
// - else => bin1 (mid)
//
// Notes:
// - unordered_set can grow; reserve helps performance but is not a cap.
// - We reuse the same unordered_set per thread, calling clear() per window.
static inline int classify_backward_3bin_set_earlystop(
    const int* col_ind,
    int old,
    int k,
    int sector_size,
    int tile_th,   // 448
    int far_th,    // 1179648
    std::unordered_set<int>& seen
) {
    if (k <= old + 1) {
        // empty interval => rd = 0
        return 0;
    }

    seen.clear();
    // For tile cases, distinct is small; reserve modestly.
    // Do NOT reserve far_th (too huge). Let it grow only if needed.
    if (seen.bucket_count() < 2048) {
        // One-time warmup for the thread-local set
        seen.reserve(2048);
    }

    // Scan backward (often beneficial for cache/locality; same RD as forward scan)
    for (int i = k - 1; i >= old + 1; --i) {
        int s = col_ind[i] / sector_size;
        seen.insert(s);

        // early stop for "far"
        if ((int)seen.size() > far_th) 
        {
            return 2;
        }
    }

    int rd = (int)seen.size();
    if (rd <= tile_th) return 0;
    return 1;
}

// ---- main API ----
// 1) sequentially build prev_pos[k] (needed for backward RD)
// 2) build list of reuse indices
// 3) sample reuse indices (deterministic given seed)
// 4) parallel classify sampled windows using std::thread
RD3BinResult compute_sector_rd_3bin_sampled_parallel_threads(
    const int* col_ind,
    int ncol_or_nrow,   // SHOULD match col_ind range (often ncol, not nrow)
    int nnz,
    int sector_size,
    double sample_prob, // e.g. 0.01 for 1%
    uint64_t seed,
    int tile_th = 448,
    int far_th  = 1179648,
    int num_threads = 0 // 0 => use hardware_concurrency()
) {
    RD3BinResult out;

    // ---- Stage 1: compute prev_pos (sequential) ----
    int num_sectors = (ncol_or_nrow + sector_size - 1) / sector_size;
    std::vector<int> last_pos(num_sectors, -1);
    std::vector<int> prev_pos(nnz, -1);
    std::vector<int> reuse_indices;
    reuse_indices.reserve(nnz / 4);

    for (int k = 0; k < nnz; ++k) {
        int s = col_ind[k] / sector_size;
        int old = last_pos[s];
        prev_pos[k] = old;
        if (old != -1) reuse_indices.push_back(k);
        last_pos[s] = k;
    }

    out.total_reuse_cnt = (long long)reuse_indices.size();
    if (out.total_reuse_cnt == 0 || sample_prob <= 0.0) {
        return out;
    }

    // ---- Stage 2: sample reuse events (sequential for reproducibility) ----
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution bern(sample_prob);

    std::vector<int> sampled;
    sampled.reserve((size_t)std::ceil(out.total_reuse_cnt * sample_prob));

    for (int k : reuse_indices) {
        if (bern(rng)) sampled.push_back(k);
    }

    out.sampled_cnt = (long long)sampled.size();
    if (out.sampled_cnt == 0) {
        return out;
    }

    // threads
    if (num_threads <= 0) {
        num_threads = (int)std::thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = 4;
    }
    num_threads = std::min(num_threads, (int)sampled.size());
    if (num_threads <= 0) num_threads = 1;

    // ---- Stage 3: parallel classify ----
    std::atomic<long long> cnt_tile{0}, cnt_mid{0}, cnt_far{0};

    auto worker = [&](int tid, int begin, int end) {
        // thread-local unordered_set (reused for each window)
        std::unordered_set<int> seen;
        // warm reserve: for tile-ish cases, 2k buckets is plenty
        seen.reserve(2048);

        long long local_tile = 0, local_mid = 0, local_far = 0;

        for (int idx = begin; idx < end; ++idx) {
            int k = sampled[idx];
            int old = prev_pos[k];
            if (old < 0) continue; // should not happen

            int bin = classify_backward_3bin_set_earlystop(
                col_ind, old, k, sector_size, tile_th, far_th, seen
            );

            if (bin == 0) local_tile++;
            else if (bin == 1) local_mid++;
            else local_far++;
        }

        cnt_tile.fetch_add(local_tile, std::memory_order_relaxed);
        cnt_mid.fetch_add(local_mid, std::memory_order_relaxed);
        cnt_far.fetch_add(local_far, std::memory_order_relaxed);
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int N = (int)sampled.size();
    int chunk = (N + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int begin = t * chunk;
        int end = std::min(N, begin + chunk);
        if (begin >= end) break;
        threads.emplace_back(worker, t, begin, end);
    }

    for (auto& th : threads) th.join();

    long long tile = cnt_tile.load(std::memory_order_relaxed);
    long long mid  = cnt_mid.load(std::memory_order_relaxed);
    long long far  = cnt_far.load(std::memory_order_relaxed);

    // normalize by sampled_cnt (not total reuse)
    out.p_tile = double(tile) / double(out.sampled_cnt);
    out.p_mid  = double(mid)  / double(out.sampled_cnt);
    out.p_far  = double(far)  / double(out.sampled_cnt);

    return out;
}

// ------------------ Example usage ------------------
// Compile: g++ -O2 -std=c++17 rd_sample.cpp -o rd_sample
// int main() {
//     // Example col_ind stream (replace with real pointer)
//     std::vector<int> col = {0,1,2,0,  3,4,5,0,  1,2,3,4,  0,6,7,8,  0};
//     int nnz = (int)col.size();
//
//     RD3BinResult res = compute_sector_rd_3bin_sampled_parallel_threads(
//         col.data(),
//         /*ncol_or_nrow=*/1024,   // should cover col_ind range
//         nnz,
//         /*sector_size=*/8,
//         /*sample_prob=*/1.0,     // sample all reuse events for demo
//         /*seed=*/42,
//         /*tile_th=*/448,
//         /*far_th=*/1179648,
//         /*num_threads=*/4
//     );
//
//     printf("reuse_total=%lld sampled=%lld\n", res.total_reuse_cnt, res.sampled_cnt);
//     printf("p_tile=%.6f p_mid=%.6f p_far=%.6f\n", res.p_tile, res.p_mid, res.p_far);
// }