// tests/hip/qsa_int8_q_fixture.cu - INT8-Q feasibility fixture for the QSA prompt attention
// (the qsa-attn recon's INT8-MMA candidate; checkpoint qsa-attn-recon-20261005.md §7).
//
// Three score paths compared for one (64 query rows x 16 keys x 256 dims) tile on gfx1100:
//   A. device int8-mma  - rocWMMA int8->i32 per 64-group (Q int8 codes x K int8 codes),
//                         fp32 fold with per-group scales (the INT8-Q candidate)
//   B. hilo emulation   - the production fp16 hi/lo Q path's arithmetic, per-element in double
//                         (qup-scaled fp16 hi + fp16 residual, fp16-exact K codes, fp32 group
//                         scales) - the incumbent's precision class
//   C. fp64 truth       - the true fp32 Q x dequantized K in double
// Metrics (pre-declared in checkpoint §7):
//   1. arithmetic isolation: A vs double-on-the-same-dequantized-operands, rel RMS <= 5e-2
//      (integer-exact dots => this isolates the fp32 fold; expect ~1e-7)
//   2. softmax max weight |delta|: softmax(s_i8) vs softmax(s_hilo_emul) <= 2e-2
//      (the candidate's ADDED deviation over the incumbent class)
//   3. output rel RMS: (p.v) int8 vs hilo <= 2e-2
//   4. vs fp64 truth reported for context (total deviation incl. the quant class)
// Q distribution: N(0,1) with 12x outliers at 1/200 (the make_x class - per-group max-abs
// quant absorbs them; the fixture measures the residual).

#include <rocwmma/rocwmma.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int ROWS = 64;         // query rows (one expert-free tile; 4 warps x 16 rows)
constexpr int NKEYS = 16;        // keys per tile
constexpr int HD = 256;          // head dim
constexpr int NG = HD / 64;      // 64-value groups
constexpr int THREADS = 128;

__device__ __forceinline__ void mma_tile_i8(const int8_t* A, int lda, const int8_t* B, int ldb,
                                            int* OUT, int ldo) {
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, int8_t, rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, int8_t, rocwmma::col_major> fb;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, int32_t> fc;
    rocwmma::fill_fragment(fc, 0);
    for (int kk = 0; kk < 4; ++kk) {   // the group is 64 deep: four k=16 steps, exact in i32
        rocwmma::load_matrix_sync(fa, A + kk * 16, lda);
        rocwmma::load_matrix_sync(fb, B + kk * 16, ldb);
        rocwmma::mma_sync(fc, fa, fb, fc);
    }
    rocwmma::store_matrix_sync(OUT, fc, ldo, rocwmma::mem_row_major);
}

// the INT8-Q score kernel: per 64-group, rocWMMA int8 dot -> i32, fold with q_sc * k_sc in fp32
__global__ void __launch_bounds__(THREADS) scores_int8_q(
    const int8_t* __restrict__ qi8, const float* __restrict__ qsc,
    const int8_t* __restrict__ k8, const float* __restrict__ ksc,
    float* __restrict__ s_out) {
    __shared__ int8_t qa[ROWS][64];
    __shared__ int8_t kb[NKEYS][64];
    __shared__ int32_t dot[THREADS / 32][16][16];
    __shared__ float acc[ROWS][16];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < ROWS * 16; i += THREADS) acc[i / 16][i % 16] = 0.0f;
    for (int g = 0; g < NG; ++g) {
        __syncthreads();
        for (int i = tid; i < ROWS * 64; i += THREADS) {
            const int r = i / 64, k = i % 64;
            qa[r][k] = qi8[(size_t) r * NG * 64 + g * 64 + k];
        }
        for (int i = tid; i < NKEYS * 64; i += THREADS) {
            const int j = i / 64, k = i % 64;
            kb[j][k] = k8[(size_t) j * NG * 64 + g * 64 + k];
        }
        __syncthreads();
        // every warp owns one 16-row m-tile of the staged tile (all warps hit every barrier)
        mma_tile_i8(&qa[warp * 16][0], 64, &kb[0][0], 64, &dot[warp][0][0], 16);
        __syncthreads();
        for (int e = lane; e < 256; e += 32) {
            const int m = e / 16, j = e % 16;
            acc[warp * 16 + m][j] = fmaf(qsc[(size_t) (warp * 16 + m) * NG + g] * ksc[(size_t) j * NG + g],
                                         (float) dot[warp][m][j], acc[warp * 16 + m][j]);
        }
        __syncthreads();
    }
    for (int e = tid; e < ROWS * NKEYS; e += THREADS) s_out[e] = acc[e / NKEYS][e % NKEYS];
}

}  // namespace

int main() {
    std::mt19937 rng(20261006);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 199);
    std::uniform_int_distribution<int> code8(-128, 127);

    // Q: ROWS x HD fp32 with outliers; per-64 max-abs int8 quant (the quant_act_nat class)
    std::vector<float> q((size_t) ROWS * HD);
    std::vector<int8_t> qi8((size_t) ROWS * HD);
    std::vector<float> qsc((size_t) ROWS * NG);
    for (int r = 0; r < ROWS; ++r) {
        for (int g = 0; g < NG; ++g) {
            float am = 0;
            for (int j = 0; j < 64; ++j) {
                float& v = q[(size_t) r * HD + g * 64 + j];
                v = nd(rng) * (pick(rng) == 0 ? 12.0f : 1.0f);
                am = std::max(am, std::fabs(v));
            }
            qsc[(size_t) r * NG + g] = am / 127.0f;
            for (int j = 0; j < 64; ++j)
                qi8[(size_t) r * HD + g * 64 + j] = am > 0 ?
                    (int8_t) (int) std::lrint(q[(size_t) r * HD + g * 64 + j] * (127.0f / am)) : 0;
        }
    }
    // K: NKEYS keys x HD int8 codes + per-64 fp32 scales (the production int8 KV class)
    std::vector<int8_t> k8((size_t) NKEYS * HD);
    std::vector<float> ksc((size_t) NKEYS * NG);
    for (int j = 0; j < NKEYS; ++j)
        for (int k = 0; k < HD; ++k) k8[(size_t) j * HD + k] = (int8_t) code8(rng);
    for (int j = 0; j < NKEYS; ++j)
        for (int g = 0; g < NG; ++g) ksc[(size_t) j * NG + g] = 0.02f + 0.01f * ((j + g) % 3);

    // device int8 path
    int8_t *d_q, *d_k;
    float *d_qs, *d_ks, *d_out;
    cudaMalloc(&d_q, qi8.size()); cudaMalloc(&d_k, k8.size());
    cudaMalloc(&d_qs, qsc.size() * 4); cudaMalloc(&d_ks, ksc.size() * 4);
    cudaMalloc(&d_out, (size_t) ROWS * NKEYS * 4);
    cudaMemcpy(d_q, qi8.data(), qi8.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k8.data(), k8.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qs, qsc.data(), qsc.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ks, ksc.data(), ksc.size() * 4, cudaMemcpyHostToDevice);
    float* d_out2;
    cudaMalloc(&d_out2, (size_t) ROWS * NKEYS * 4);
    scores_int8_q<<<1, THREADS>>>(d_q, d_qs, d_k, d_ks, d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("KERNEL FAIL\n"); return 1; }
    std::vector<float> s8((size_t) ROWS * NKEYS);
    cudaMemcpy(s8.data(), d_out, s8.size() * 4, cudaMemcpyDeviceToHost);
    // double-launch diff probe: same inputs, second launch into a fresh buffer
    cudaMemset(d_out2, 0, (size_t) ROWS * NKEYS * 4);
    scores_int8_q<<<1, THREADS>>>(d_q, d_qs, d_k, d_ks, d_out2);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("KERNEL2 FAIL\n"); return 1; }
    std::vector<float> s8b((size_t) ROWS * NKEYS);
    cudaMemcpy(s8b.data(), d_out2, s8b.size() * 4, cudaMemcpyDeviceToHost);
    {
        int diffs = 0, worst = -1;
        float wd = 0;
        for (size_t i = 0; i < s8.size(); ++i)
            if (s8[i] != s8b[i]) { ++diffs; if (std::fabs(s8[i] - s8b[i]) > wd) { wd = std::fabs(s8[i] - s8b[i]); worst = (int) i; } }
        std::printf("RACEPROBE diffs=%d / %zu, worst delta %f at index %d (row %d, col %d)\n",
                    diffs, s8.size(), (double) wd, worst, worst >= 0 ? worst / NKEYS : -1, worst >= 0 ? worst % NKEYS : -1);
    }

    // hilo emulation + fp64 truth, per element in double
    // (the production arm: qup-scaled fp16 hi + fp16 residual; K codes exact in fp16;
    //  fp32 group accumulate; qdown * k_scale per group)
    std::vector<double> s_hilo((size_t) ROWS * NKEYS), s_ref((size_t) ROWS * NKEYS);
    std::vector<double> q_deq((size_t) ROWS * HD), k_deq((size_t) NKEYS * HD);
    for (int r = 0; r < ROWS; ++r) {
        // qup/qdown from the row's max |q| (the production exponent choreography)
        float qm = 0;
        for (int d = 0; d < HD; ++d) qm = std::max(qm, std::fabs(q[(size_t) r * HD + d]));
        const float qup = qm > 0 ? 16384.0f / qm : 1.0f;   // scale the max to 2^14
        const float qdown = 1.0f / qup;
        for (int d = 0; d < HD; ++d) {
            const _Float16 hi = (_Float16) (q[(size_t) r * HD + d] * qup);
            const _Float16 lo = (_Float16) ((double) q[(size_t) r * HD + d] * qup - (double) hi);
            q_deq[(size_t) r * HD + d] = ((double) hi + (double) lo) * qdown;
        }
    }
    for (int j = 0; j < NKEYS; ++j)
        for (int g = 0; g < NG; ++g) {
            const float ks = ksc[(size_t) j * NG + g];
            for (int k = 0; k < 64; ++k)
                k_deq[(size_t) j * HD + g * 64 + k] = (double) k8[(size_t) j * NG * 64 + g * 64 + k] * ks;
        }
    for (int m = 0; m < ROWS; ++m)
        for (int j = 0; j < NKEYS; ++j) {
            double ah = 0, ar = 0;
            for (int d = 0; d < HD; ++d) {
                const double kd = k_deq[(size_t) j * HD + d];
                ah += q_deq[(size_t) m * HD + d] * kd;      // the fp16-hi/lo Q reconstruction
                ar += (double) q[(size_t) m * HD + d] * kd; // the fp64 truth
            }
            s_hilo[(size_t) m * NKEYS + j] = ah;
            s_ref[(size_t) m * NKEYS + j] = ar;
        }
    // NOTE: the hilo emulation's Q operand is the fp32 Q scaled by qup/qdown - the fp16 hi/lo
    // representation of (x * qup) reconstructs x to ~22 bits, which the emulated double loop
    // below models as the fp32 value (the ~22-bit class is invisible at double precision).
    // For the score-error comparison the hilo path is therefore ~the fp32-Q truth, and the
    // INT8-Q deviation is measured against it directly (the checkpoint's metric 2/3).

    // metrics
    // 1. arithmetic isolation: s8 vs the double sum on the DEQUANTIZED int8 operands
    double iso2 = 0, iso_r2 = 0;
    {
        double e2 = 0, r2 = 0;
        for (int m = 0; m < ROWS; ++m)
            for (int j = 0; j < NKEYS; ++j) {
                double a = 0;
                for (int g = 0; g < NG; ++g)
                    for (int k = 0; k < 64; ++k)
                        a += (double) qi8[(size_t) m * HD + g * 64 + k] * (double) k8[(size_t) j * NG * 64 + g * 64 + k] *
                             (double) qsc[(size_t) m * NG + g] * (double) ksc[(size_t) j * NG + g];
                const double e = (double) s8[(size_t) m * NKEYS + j] - a;
                e2 += e * e;
                r2 += a * a;
            }
        iso2 = e2;
        iso_r2 = r2;
    }
    const double iso_rms = iso_r2 > 0 ? std::sqrt(iso2 / iso_r2) : 0.0;

    // softmax metrics: s8 vs hilo (the ADDED deviation), and s8 vs fp64 truth
    double wd_add = 0, wd_tot = 0, out_add = 0, out_tot = 0;
    for (int m = 0; m < ROWS; ++m) {
        double m8 = -1e30, mh = -1e30, mr = -1e30;
        for (int j = 0; j < NKEYS; ++j) {
            m8 = std::max(m8, (double) s8[(size_t) m * NKEYS + j]);
            mh = std::max(mh, s_hilo[(size_t) m * NKEYS + j]);
            mr = std::max(mr, s_ref[(size_t) m * NKEYS + j]);
        }
        double z8 = 0, zh = 0, zr = 0;
        for (int j = 0; j < NKEYS; ++j) {
            z8 += std::exp((double) s8[(size_t) m * NKEYS + j] - m8);
            zh += std::exp(s_hilo[(size_t) m * NKEYS + j] - mh);
            zr += std::exp(s_ref[(size_t) m * NKEYS + j] - mr);
        }
        double o8 = 0, oh = 0, orr = 0, wd = 0;
        for (int j = 0; j < NKEYS; ++j) {
            const float vj = (float) nd(rng);
            const double p8 = std::exp((double) s8[(size_t) m * NKEYS + j] - m8) / z8;
            const double ph = std::exp(s_hilo[(size_t) m * NKEYS + j] - mh) / zh;
            const double pr = std::exp(s_ref[(size_t) m * NKEYS + j] - mr) / zr;
            o8 += p8 * (double) vj; oh += ph * (double) vj; orr += pr * (double) vj;
            wd = std::max(wd, std::fabs(p8 - ph));
        }
        wd_add = std::max(wd_add, wd);
        out_add = std::max(out_add, std::fabs(o8 - oh));
        out_tot = std::max(out_tot, std::fabs(o8 - orr));
        (void) wd_tot;
    }
    std::printf("INT8-Q arithmetic isolation vs double(dequant operands): rel RMS %.3e\n", iso_rms);
    {
        double d_hilo = 0, d_ref = 0;
        for (int m = 0; m < ROWS; ++m)
            for (int j = 0; j < NKEYS; ++j) {
                d_hilo = std::max(d_hilo, std::fabs((double) s8[(size_t) m * NKEYS + j] - s_hilo[(size_t) m * NKEYS + j]));
                d_ref = std::max(d_ref, std::fabs((double) s8[(size_t) m * NKEYS + j] - s_ref[(size_t) m * NKEYS + j]));
            }
        std::printf("score-domain max |s8 - hilo| %.4e, max |s8 - fp64 truth| %.4e\n", d_hilo, d_ref);
    }
    std::printf("INT8-Q vs hilo incumbent: softmax max weight delta %.3e, output max |delta| %.3e\n",
                wd_add, out_add);
    const bool ok = iso_rms <= 5e-2 && wd_add <= 2e-2 && out_add <= 2e-2;
    std::printf(ok ? "FIXTURE PASS (bounds 5e-2 / 2e-2 / 2e-2)\n" : "FIXTURE FAIL\n");

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_qs); cudaFree(d_ks); cudaFree(d_out);
    return ok ? 0 : 1;
}
