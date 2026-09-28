// tests/rocm/qsa_score_bench.cpp - portable vs WMMA QSA scorer, kernel timing.
//
// A tool, not a test (the suite's correctness gates live in the parity files).
// Times both arms of native_qsa_score at the shapes the engine actually hits
// (pooled rows = n_kv/4 + 1; 8k/32k/65k contexts), median of 9 device-timed
// repetitions after warmup, hipEvent around the launch + sync. The scorer's
// share of a decode step is what decides whether patch 0007's arm gets wired
// into the engine (PORTING.md 12c promotion rule); this exists to produce that
// number reproducibly.
//
// Run: qsa_score_bench [--selftest]   (both forms run the same timing)

#include "strata/kernels/native_qsa_score.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int D = 128, HEADS = 4, R = 4;

int failures = 0;

struct Rng {
    uint64_t s = 0x9e3779b97f4a7c15ull;
    float next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (float)((int64_t)(s >> 11) % 2000001 - 1000000) / 1000000.0f;
    }
};

double median(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int bench_shape(int n_kv, int reps) {
    const int max_cells = ((n_kv + R - 1) / R) * R;   // capacity: n_kv rounded to blocks
    const int max_blocks = max_cells / R + 1;
    const int full = n_kv / R;

    Rng rng;
    std::vector<float> pooled((size_t)max_blocks * D), query((size_t)HEADS * D);
    for (auto& v : pooled) v = rng.next();
    for (auto& v : query) v = rng.next();
    std::vector<int32_t> step(4, 0);
    step[0] = n_kv - 1; step[1] = n_kv; step[2] = full;
    step[3] = n_kv < 2051 ? n_kv : 2051;

    float *dp = nullptr, *dq = nullptr, *dc = nullptr;
    int32_t* ds = nullptr;
    if (hipMalloc(&dp, pooled.size() * 4) != hipSuccess) return 1;
    if (hipMalloc(&dq, query.size() * 4) != hipSuccess) return 1;
    if (hipMalloc(&dc, (size_t)max_cells * 4) != hipSuccess) return 1;
    if (hipMalloc(&ds, step.size() * 4) != hipSuccess) return 1;
    hipMemcpy(dp, pooled.data(), pooled.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dq, query.data(), query.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(ds, step.data(), step.size() * 4, hipMemcpyHostToDevice);

    strata::kernels::QsaShapes s;
    s.idx_dim = D; s.idx_n_head = HEADS; s.idx_block = R; s.idx_top_k = 2048;
    hipStream_t stream = nullptr;
    hipStreamCreate(&stream);

    hipEvent_t beg, end;
    hipEventCreate(&beg);
    hipEventCreate(&end);

    struct Arm { const char* name; bool wmma; };
    const Arm arms[] = {{"portable", false}, {"wmma", true}};

    std::printf("  n_kv=%-6d pooled_rows=%-6d", n_kv, full + 1);
    for (const Arm& arm : arms) {
        strata::kernels::native_qsa_score_set_wmma(arm.wmma);
        for (int w = 0; w < 3; ++w) {   // warmup (page cache, graphs, clocks)
            strata::kernels::native_qsa_score(dp, dq, nullptr, s, ds, max_blocks, max_cells, dc, stream);
        }
        if (hipStreamSynchronize(stream) != hipSuccess) { std::printf("  %s SYNC FAIL", arm.name); continue; }
        std::vector<float> ms;
        for (int r = 0; r < reps; ++r) {
            hipEventRecord(beg, stream);
            strata::kernels::native_qsa_score(dp, dq, nullptr, s, ds, max_blocks, max_cells, dc, stream);
            hipEventRecord(end, stream);
            if (hipStreamSynchronize(stream) != hipSuccess) break;
            float t = 0.0f;
            hipEventElapsedTime(&t, beg, end);
            ms.push_back(t);
        }
        std::printf("  %s median %.4f ms (n=%zu)", arm.name, median(ms), ms.size());
    }
    std::printf("\n");
    strata::kernels::native_qsa_score_set_wmma(false);

    hipEventDestroy(beg);
    hipEventDestroy(end);
    hipStreamDestroy(stream);
    hipFree(dp); hipFree(dq); hipFree(dc); hipFree(ds);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: qsa_score_bench [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return 1;
    std::printf("qsa_score_bench on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  device-timed, median of 9 after 3 warmups; both arms, same buffers\n");

    const int reps = 9;
    for (int n_kv : {8192, 32768, 65536}) {
        if (bench_shape(n_kv, reps) != 0) { ++failures; }
    }

    if (failures) return 1;
    std::printf("qsa_score_bench OK\n");
    (void)selftest;
    return 0;
}
