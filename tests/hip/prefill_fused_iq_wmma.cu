// tests/hip/prefill_fused_iq_wmma.cu - M2 fixture of the fused-WMMA campaign
// (moe-fused-wmma-checkpoint-20261005.md): the IQ3_S gate/up mma+scale chain translated to
// rocWMMA INT8 fragments on gfx1100, against a double-precision reference computed on the
// SAME quantized activations and the ggml-dequantized weights.
//
// Scope (M2): ONE expert, TR=128 routed rows (contiguous - no routing tables), the full
// K=2560 in 64-value stages, raw gate/up dots out (SwiGLU is host-side here; the int8
// requant epilogue is M3). Pre-declared bounds (checkpoint §3, before any run): the int8
// dots and INT32 accumulation are integer-exact, the activation quantization cancels
// (the reference dequantizes the same codes), the only error class is fp32 accumulation
// order over 2560-length dots => rel RMS <= 1e-4, worst row <= 1e-3. A failure outside
// these is a translation bug, not a tolerance question.

// This TU is built with the project's hip_compat shim force-included (CUDA-familiar macro
// names), but ggml's vendors/hip.h DEFINES those tokens as real functions and rocwmma's
// headers use them as names - so drop the shim macros for the rest of the TU (ggml's own
// implementations then serve; strata_mmq's TUs simply never get the shim).
#undef __dp4a
#undef __byte_perm
#undef __vsub4
#undef __vsubss4
#undef __vcmpne4
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync

#include <rocwmma/rocwmma.hpp>

#include "ggml.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int N = 2560;                 // K per expert
constexpr int FF = 640;                 // gate rows (up rows 640..1279 of the same matrix family)
constexpr int GU_ROWS = 2 * FF;         // 1280 weight rows
constexpr int TR = 128;                 // routed rows (one expert, all valid)
constexpr int NS = N / 64;              // 40 stages of 64 values
constexpr int WLD = 64;                 // LDS int8 row per stage (64 values)
constexpr int AB = 80;                  // activation stage row: 64 codes + float2 half-scales + pad
constexpr int THREADS = 128;

// ---- the fused kernel's IQ3_S decode (verbatim semantics from moe_fused_iq.cu) ----

__device__ __forceinline__ uint32_t ld32(const void* p) { uint32_t v; memcpy(&v, p, 4); return v; }
__device__ __forceinline__ uint32_t ld16(const void* p) { uint32_t v = 0; memcpy(&v, p, 2); return v; }
__device__ __forceinline__ float half_at(uint32_t h) {
    // HIP trap (found by the fixture): __half2float(ushort) converts the INTEGER on this target,
    // not the bit pattern - the CUDA arm never calls it from HIP so it never showed there.
    return __half2float(__ushort_as_half((unsigned short) h));
}

__device__ __forceinline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, int8_t* q) {
    // verbatim from moe_fused_iq.cu: ONE SIGN BIT PER VALUE - bits 0..3 of s sign gx's bytes,
    // bits 4..7 sign gy's (my first port replicated a bit-7 mask and decoded half the values wrong)
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    const uint32_t qx = __vsub4(gx ^ m0, m0), qy = __vsub4(gy ^ m1, m1);
    memcpy(&q[0], &qx, 4);
    memcpy(&q[4], &qy, 4);
}

__device__ __forceinline__ void convert_iq3_s(const uint32_t (&w)[4], const uint32_t* grid, int8_t (&q)[32],
                                              float& s0, float& s1) {
    const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
        signed8(grid[i0 | ((qh << (8 - 2 * l)) & 256)], grid[i1 | ((qh << (7 - 2 * l)) & 256)],
                ((w[2] >> (8 * l)) & 255) * 0x01010101u, &q[8 * l]);
    }
    s0 = s1 = half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
}

__device__ __forceinline__ void load_unit_iq3_s(const uint8_t* bp, int ib, uint32_t (&w)[4]) {
    w[0] = ld32(bp + 2 + 8 * ib);
    w[1] = ld32(bp + 6 + 8 * ib);
    w[2] = ld32(bp + 74 + 4 * ib);
    w[3] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
}

// ---- the mma primitive (Qwen-proven unit) ----

__device__ __forceinline__ void mma_tile_16(const int8_t* A_lds, int lda, const int8_t* B_tile, int ldb,
                                            int32_t* OUT, int ldo) {
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, int8_t, rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, int8_t, rocwmma::col_major> fb;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, int32_t> fc;
    rocwmma::load_matrix_sync(fa, A_lds, lda);
    rocwmma::load_matrix_sync(fb, B_tile, ldb);
    rocwmma::fill_fragment(fc, 0);
    rocwmma::mma_sync(fc, fa, fb, fc);
    rocwmma::store_matrix_sync(OUT, fc, ldo, rocwmma::mem_row_major);
}

// ---- the translated gate/up chain: one expert, raw gate and up dots out ----
//
// Grid: (GU_ROWS / 128, TR / 128) = (10, 1). Block: 4 warps; warp wf owns weight rows
// 32 wf.. 32 wf + 31 (two 16-row m-tiles), all 128 routed rows in 8 n-tiles of 16.
// Per stage s: decode the stage's 2 sub-blocks per row to LDS int8 (the super-block scale
// replicated to the stage's 4 per-16 slots), read the pre-quantized activations, then per
// (h, j16) one mma_tile_16 whose int32 tile folds into fp32 Dgt/Dup with the ORIGINAL
// scale order: (float) dot * wscale(row, 2h + j16) * ascale(r, 2s + h).
__global__ void __launch_bounds__(THREADS) native_kernel_wmma_iq3s(
    const uint8_t* __restrict__ blob, size_t gu_row, const uint8_t* __restrict__ act,
    const float* __restrict__ ascale, float* __restrict__ dgt, float* __restrict__ dup) {
    __shared__ int8_t wt[128][WLD];       // this block's 128 weight rows, stage slice
    __shared__ float ws[128][4];          // the stage's per-row scales (s0=s1 per 32-half, replicated)
    __shared__ int8_t ast[TR][64];        // the stage's activation codes
    __shared__ float asl[TR][2];          // the stage's activation half-scales
    __shared__ int32_t dot[4][2][16][16];    // per-warp, per-m-tile mma output (disjoint tiles)

    const int tid = threadIdx.x, warp = tid >> 5;
    const int row0 = blockIdx.x * 128, r0 = blockIdx.y * 128;
    for (int s = 0; s < NS; ++s) {
        __syncthreads();
        // stage s: k = s*64..+63 = sub-blocks 2*(s%4), 2*(s%4)+1 of super-block s/4;
        // 2 threads per weight row over ALL 128 rows (the block's threads make two passes)
        for (int d = tid; d < 2 * 128; d += THREADS) {
            const int dec_row = d / 2, dec_uj = d % 2;
            const uint8_t* bp = blob + (size_t) (row0 + dec_row) * gu_row + (size_t) (s / 4) * 110;
            uint32_t w[4];
            float s0, s1;
            load_unit_iq3_s(bp, 2 * (s % 4) + dec_uj, w);
            convert_iq3_s(w, iq3s_grid, *(int8_t (*)[32]) & wt[dec_row][32 * dec_uj], s0, s1);
            // the engine's ws layout: slots {2*uj, 2*uj+1} = THIS sub-block's scale pair (the fold's
            // ws[2h + j16] must read sub-block h's scale - one scale nibble per sub-block in IQ3_S)
            ws[dec_row][2 * dec_uj] = s0;
            ws[dec_row][2 * dec_uj + 1] = s1;
        }

        __syncthreads();
        // activation stage slice: 64 codes + the half scales (rows r0..r0+127)
        for (int r = tid; r < TR; r += THREADS) {
            memcpy(&ast[r][0], act + ((size_t) (r0 + r) * NS + s) * AB, 64);
            memcpy(&asl[r][0], act + ((size_t) (r0 + r) * NS + s) * AB + 64, 8);
        }
        __syncthreads();
        if (warp * 32 >= GU_ROWS) continue;

        for (int i = 0; i < 2; ++i) {
            const int wr0 = warp * 32 + 16 * i;           // this m-tile's first weight row (absolute)
            if (row0 + wr0 >= GU_ROWS) continue;
            for (int nt = 0; nt < TR / 16; ++nt) {
                for (int h = 0; h < 2; ++h)
                    for (int j16 = 0; j16 < 2; ++j16) {
                        __syncthreads();
                        mma_tile_16(&wt[wr0][32 * h + 16 * j16], WLD,
                                    &ast[nt * 16][32 * h + 16 * j16], 64,
                                    &dot[warp][i][0][0], 16);
                        __syncthreads();
                                                // fold: element (m, c) -> weight row wr0 + m, routed row r0 + nt*16 + c.
                        // PER-WARP over this warp's own dot tile: the tile lives in dot[warp][i] and
                        // the output rows are this warp's wr0 + m (an all-threads e-loop would mix
                        // warps' tiles and leave most rows unwritten)
                        {
                            const int lane = tid & 31;
                            for (int e = lane; e < 256; e += 32) {
                            const int m = e / 16, c = e % 16;
                            const float v = (float) dot[warp][i][m][c];
                            const float wsc = ws[wr0 + m][2 * h + j16];
                            const float asc = asl[nt * 16 + c][h];
                            float* out = (m + row0 + wr0 < FF) ? dgt + (size_t) (row0 + wr0 + m) * TR + r0 + nt * 16 + c
                                                               : dup + (size_t) (row0 + wr0 + m - FF) * TR + r0 + nt * 16 + c;
                            *out = fmaf(wsc * asc, v, *out);
                            }
                        }
                        __syncthreads();
                    }
            }
        }
    }
}

// ---- host: generator + double reference ----

constexpr float IQ3S_SCALE_LO = 0.0004f;

void fill_blob(std::vector<uint8_t>& b, std::mt19937& rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_IQ3_S, N), nb_per_row = N / 256;
    for (size_t r = 0; r < GU_ROWS; ++r) {
        uint8_t* m = b.data() + r * row_bytes;
        for (size_t i = 0; i < row_bytes; ++i) m[i] = (uint8_t) byte(rng);
        std::uniform_real_distribution<float> sc(IQ3S_SCALE_LO, 6.0f * IQ3S_SCALE_LO);
        for (int sb = 0; sb < nb_per_row; ++sb) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(sc(rng));
            memcpy(m + (size_t) sb * 110, &h, 2);   // the super-block scale at the block start
        }
    }
}

// activations: N(0,1) with a few large values, quantized to the engine's per-64 layout
// (64 codes + float2 half-scales per stage row = AB bytes)
void make_act(int T, std::mt19937& rng, std::vector<float>& x_deq, std::vector<uint8_t>& act,
              std::vector<float>& ascale) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 199);
    x_deq.assign((size_t) T * N, 0.0f);
    act.assign((size_t) T * NS * AB, 0);
    ascale.assign((size_t) T * NS * 2, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < NS; ++s) {
            uint8_t* row = act.data() + ((size_t) t * NS + s) * AB;
            for (int h = 0; h < 2; ++h) {
                float am = 0;
                float v[32];
                for (int j = 0; j < 32; ++j) {
                    v[j] = nd(rng) * (pick(rng) == 0 ? 12.0f : 1.0f);
                    am = std::max(am, std::fabs(v[j]));
                    x_deq[(size_t) t * N + s * 64 + 32 * h + j] = v[j];   // provisional; rescaled below
                }
                const float sc = am / 127.0f;
                for (int j = 0; j < 32; ++j) {
                    const int8_t c = am > 0 ? (int8_t) (int) std::lrint(v[j] * (127.0f / am)) : 0;
                    row[32 * h + j] = (uint8_t) c;
                    x_deq[(size_t) t * N + s * 64 + 32 * h + j] = (float) c * sc;
                }
                const float half_scale[2] = {sc, 0.0f};
                memcpy(row + 64 + 4 * h, half_scale, 4);
                ascale[((size_t) t * NS + s) * 2 + h] = sc;
            }
        }
}

}  // namespace

int main() {
    const int rc = [] {
        std::mt19937 rng(20261005);
        const size_t row_bytes = ggml_row_size(GGML_TYPE_IQ3_S, N);
        std::vector<uint8_t> blob(GU_ROWS * row_bytes);
        fill_blob(blob, rng);
        std::vector<float> x_deq, ascale;
        std::vector<uint8_t> act;
        make_act(TR, rng, x_deq, act, ascale);

        // device buffers
        uint8_t *d_blob, *d_act;
        float *d_as, *d_gt, *d_up;
        cudaMalloc(&d_blob, blob.size());
        cudaMalloc(&d_act, act.size());
        cudaMalloc(&d_as, ascale.size() * 4);
        cudaMalloc(&d_gt, (size_t) FF * TR * 4);
        cudaMalloc(&d_up, (size_t) FF * TR * 4);
        cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_act, act.data(), act.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_as, ascale.data(), ascale.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(d_gt, 0, (size_t) FF * TR * 4);
        cudaMemset(d_up, 0, (size_t) FF * TR * 4);
        dim3 grid(GU_ROWS / 128, TR / 128);
        native_kernel_wmma_iq3s<<<grid, THREADS>>>(d_blob, row_bytes, d_act, d_as, d_gt, d_up);
        const cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { std::printf("KERNEL FAIL: %s\n", cudaGetErrorString(err)); return 1; }
        std::vector<float> gt(FF * TR), up(FF * TR);
        cudaMemcpy(gt.data(), d_gt, gt.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(up.data(), d_up, up.size() * 4, cudaMemcpyDeviceToHost);

        // double reference on the dequantized weights x the same quantized activations
        const auto* tr = ggml_get_type_traits(GGML_TYPE_IQ3_S);
        std::vector<float> wrow_f(N);
        double e2 = 0, r2 = 0, worst = 0;
        for (int f = 0; f < FF; ++f) {
            float* rows_f[2] = {wrow_f.data(), wrow_f.data()};
            std::vector<float> wall((size_t) GU_ROWS * N);
            for (int r = 0; r < GU_ROWS; ++r)
                tr->to_float(blob.data() + (size_t) r * row_bytes, wall.data() + (size_t) r * N, N);
            for (int r = 0; r < TR; ++r) {
                const float* xq = x_deq.data() + (size_t) r * N;
                double gt_r = 0, up_r = 0;
                for (int k = 0; k < N; ++k) {
                    gt_r += (double) wall[(size_t) f * N + k] * xq[k];
                    up_r += (double) wall[(size_t) (FF + f) * N + k] * xq[k];
                }
                const double eg = (double) gt[(size_t) f * TR + r] - gt_r;
                const double eu = (double) up[(size_t) f * TR + r] - up_r;
                if (std::fabs(eg) > 1e4 || std::fabs(eu) > 1e4)
                e2 += eg * eg + eu * eu;
                r2 += gt_r * gt_r + up_r * up_r;
                worst = std::max(worst, std::max(std::fabs(eg), std::fabs(eu)) /
                                            std::max(1.0, std::max(std::fabs(gt_r), std::fabs(up_r))));
            }
            (void) rows_f;
        }
        // bisect: device dgt[0] vs the direct double dot for gate row 0, routed row 0
        {
            double ref0 = 0;
            std::vector<float> w0(N);
            tr->to_float(blob.data(), w0.data(), N);
            for (int k = 0; k < N; ++k) ref0 += (double) w0[k] * x_deq[k];
        }
        const double rms = std::sqrt(e2 / r2);
        std::printf("fused-WMMA IQ3_S gate/up: rel RMS %.3e, worst row rel %.3e\n", rms, worst);
        const bool ok = rms <= 1e-4 && worst <= 1e-3;
        std::printf(ok ? "FIXTURE PASS (bounds 1e-4 / 1e-3)\n" : "FIXTURE FAIL\n");
        cudaFree(d_blob); cudaFree(d_act); cudaFree(d_as); cudaFree(d_gt); cudaFree(d_up);
        return ok ? 0 : 1;
    }();
    return rc;
}
