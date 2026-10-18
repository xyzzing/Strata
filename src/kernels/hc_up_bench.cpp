// src/kernels/hc_up_bench.cpp - does the fused HC `up` kernel have a reachable memory ceiling?
//
// The engine's up call reads w_up = bf16 [10240][320] = 6.55 MB per (layer, half); the in-situ profile puts it at
// 15.6 us = ~420 GB/s of the ~750 GB/s this card streams. This bench replicates the kernel's exact row-dot pattern
// (one warp per row column-group, 32 columns x 4 streams = 128 rows per block) and tests whether a different grid
// or ILP shape moves the achieved bandwidth. It is the stop/go discriminator: if no variant reaches ~650 GB/s,
// the engine kernel is not going anywhere and nothing is changed.
//
//     build/hc_up_bench [mode: all|0|1|2|3|4] [iters]
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(e, w) do { cudaError_t _e = (e); if (_e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(_e)); std::exit(1); } } while (0)

namespace {
constexpr int N = 2560;      // n_embd
constexpr int LR = 320;      // hc_lr
constexpr int HC = 4;
constexpr int D = N * HC;    // 10240 rows
constexpr int UP_COLS = 32;
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;

struct Bf16x8 { float4 a, b; };
__device__ __forceinline__ Bf16x8 unpack8(const uint4 w) {
    return {make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u),
                        __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u)),
            make_float4(__uint_as_float(w.z << 16), __uint_as_float(w.z & 0xffff0000u),
                        __uint_as_float(w.w << 16), __uint_as_float(w.w & 0xffff0000u))};
}
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    const Bf16x8 u = unpack8(w);
    const float4 x0 = *reinterpret_cast<const float4*>(x);
    const float4 x1 = *reinterpret_cast<const float4*>(x + 4);
    float acc = 0.0f;
    acc = fmaf(u.a.x, x0.x, acc); acc = fmaf(u.a.y, x0.y, acc);
    acc = fmaf(u.a.z, x0.z, acc); acc = fmaf(u.a.w, x0.w, acc);
    acc = fmaf(u.b.x, x1.x, acc); acc = fmaf(u.b.y, x1.y, acc);
    acc = fmaf(u.b.z, x1.z, acc); acc = fmaf(u.b.w, x1.w, acc);
    return acc;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// rows_per_block: how many of the 128 rows a block owns (0 = all). grid.y splits them.
template <int ROWS>
__global__ void __launch_bounds__(THREADS) up_kernel(const uint16_t* __restrict__ w, const float* __restrict__ lo,
                                                     float* __restrict__ y) {
    __shared__ __align__(16) float slo[LR];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UP_COLS;
    for (int k = t; k < LR; k += THREADS) slo[k] = lo[k];
    __syncthreads();
    const int base = (ROWS == 128) ? 0 : blockIdx.y * ROWS;
    const int end = (ROWS == 128) ? 128 : base + ROWS;
    for (int r = base + warp; r < end; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        float acc = dot8(*reinterpret_cast<const uint4*>(w + (size_t) i * LR + (size_t) lane * 8), slo + lane * 8);
        if (lane < LR / 8 - 32)
            acc += dot8(*reinterpret_cast<const uint4*>(w + (size_t) i * LR + (size_t) (32 + lane) * 8),
                        slo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) y[i] = acc;
    }
}

// 8 warps: 16 rows per warp.
__global__ void __launch_bounds__(256) up8_kernel(const uint16_t* __restrict__ w, const float* __restrict__ lo,
                                                  float* __restrict__ y) {
    __shared__ __align__(16) float slo[LR];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UP_COLS;
    for (int k = t; k < LR; k += 256) slo[k] = lo[k];
    __syncthreads();
    for (int r = warp; r < HC * UP_COLS; r += 8) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        float acc = dot8(*reinterpret_cast<const uint4*>(w + (size_t) i * LR + (size_t) lane * 8), slo + lane * 8);
        if (lane < LR / 8 - 32)
            acc += dot8(*reinterpret_cast<const uint4*>(w + (size_t) i * LR + (size_t) (32 + lane) * 8),
                        slo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) y[i] = acc;
    }
}

// 4-row ILP: four independent rows in flight per warp, one dot each, then four warp_sums.
__global__ void __launch_bounds__(THREADS) up4_kernel(const uint16_t* __restrict__ w, const float* __restrict__ lo,
                                                      float* __restrict__ y) {
    __shared__ __align__(16) float slo[LR];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UP_COLS;
    for (int k = t; k < LR; k += THREADS) slo[k] = lo[k];
    __syncthreads();
    for (int r0 = warp; r0 < HC * UP_COLS; r0 += WARPS * 4) {
        float a[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = r0 + j * WARPS;
            a[j] = 0.0f;
            if (r < HC * UP_COLS) {
                const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
                a[j] = dot8(*reinterpret_cast<const uint4*>(w + (size_t) i * LR + (size_t) lane * 8), slo + lane * 8);
                if (lane < LR / 8 - 32)
                    a[j] += dot8(*reinterpret_cast<const uint4*>(w + (size_t) i * LR + (size_t) (32 + lane) * 8),
                                 slo + (32 + lane) * 8);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = r0 + j * WARPS;
            if (r >= HC * UP_COLS) continue;
            const float s = warp_sum(a[j]);
            const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
            if (lane == 0) y[i] = s;
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "all";
    const int iters = argc > 2 ? std::atoi(argv[2]) : 100;
    const size_t bytes = (size_t) D * LR * 2;
    const int copies = 16;
    std::vector<uint16_t> hw((size_t) D * LR);
    std::mt19937 rng(7);
    for (auto& v : hw) v = (uint16_t) (rng() & 0x3f00u);
    std::vector<float> hlo(LR);
    for (int i = 0; i < LR; ++i) hlo[i] = (float) (i % 13) * 0.01f - 0.06f;
    std::vector<void*> w(copies);
    for (auto& p : w) { CK(cudaMalloc(&p, bytes), "malloc w"); CK(cudaMemcpy(p, hw.data(), bytes, cudaMemcpyHostToDevice), "copy w"); }
    float *lo = nullptr, *y = nullptr;
    CK(cudaMalloc(&lo, LR * sizeof(float)), "malloc lo");
    CK(cudaMemcpy(lo, hlo.data(), LR * sizeof(float), cudaMemcpyHostToDevice), "copy lo");
    CK(cudaMalloc(&y, (size_t) D * sizeof(float)), "malloc y");
    cudaStream_t st; CK(cudaStreamCreate(&st), "stream");
    unsigned nsm = 0; cudaDeviceGetAttribute((int*) &nsm, cudaDevAttrMultiProcessorCount, 0);
    std::printf("hc_up_bench: %u SM, w_up %.2f MB, %d copies, iters %d\n", nsm, bytes / 1e6, copies, iters);
    auto run = [&](const char* name, int mode) {
        auto launch = [&](int copy) {
            const uint16_t* wp = (const uint16_t*) w[copy % copies];
            switch (mode) {
                case 0: up_kernel<128><<<80, THREADS, 0, st>>>(wp, lo, y); break;
                case 1: up_kernel<64><<<dim3(80, 2), THREADS, 0, st>>>(wp, lo, y); break;
                case 2: up_kernel<32><<<dim3(80, 4), THREADS, 0, st>>>(wp, lo, y); break;
                case 3: up8_kernel<<<80, 256, 0, st>>>(wp, lo, y); break;
                case 4: up4_kernel<<<80, THREADS, 0, st>>>(wp, lo, y); break;
            }
        };
        for (int i = 0; i < 5; ++i) launch(i);
        CK(cudaDeviceSynchronize(), "warm");
        cudaEvent_t a, b; CK(cudaEventCreate(&a), "ev"); CK(cudaEventCreate(&b), "ev");
        CK(cudaEventRecord(a, st), "rec");
        for (int i = 0; i < iters; ++i) launch(i + 5);
        CK(cudaEventRecord(b, st), "rec");
        CK(cudaEventSynchronize(b), "sync");
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b), "elapsed");
        const double us = 1000.0 * ms / iters, gbs = bytes / (us * 1e3);
        std::printf("  %-22s %8.1f us  %7.1f GB/s  (%.0f%% of 750)\n", name, us, gbs, 100.0 * gbs / 750.0);
        CK(cudaEventDestroy(a), "ev"); CK(cudaEventDestroy(b), "ev");
    };
    const struct { const char* n; int m; } M[] = {{"V0 baseline (80x128)", 0}, {"V1 rows/2 (160 blk)", 1},
                                                 {"V2 rows/4 (320 blk)", 2}, {"V3 8 warps (256 th)", 3}, {"V4 4-row ILP", 4}};
    for (auto& e : M) if (std::strcmp(which, "all") == 0 || std::atoi(which) == e.m) run(e.n, e.m);
    return 0;
}
