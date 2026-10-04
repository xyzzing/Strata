// src/prefill/gdn_wy_check.cu - the WY-form chunk-parallel GDN recurrence: algebra proof and bound.
//
// C2 of tasks/gdn-wy-design-20261005.md.  The engine's serial recurrence (gdn_rec_cols_pipe_kernel,
// copied verbatim from src/prefill/kernels.cu) is ground truth; two host references check the math:
//   serial64 - the same token loop in double (the GPU kernel's fp32 noise baseline)
//   wy64     - the chunked WY form (tasks doc, C=64): prefix decays, the CxC triangular system for
//              u = delta, outputs and the state as rank-C sums - a DIFFERENT summation order
// Gates per fixture: wy64 vs serial64 inside a tight double-epsilon bound (the algebra), and the
// GPU kernel vs serial64 inside the derived fp32 bound (the port), with the wy device kernel's
// future bound = the composition.  Synthetic inputs at model shape (S=128, HK=16, HV=48), q/k rows
// L2-normalized as gdn_l2_kernel leaves them.
//
//   gdn_wy_check            all fixtures, T = 1..4099 sweep + adversarial gates
//   gdn_wy_check --bench    the timing that motivates it: serial vs nothing yet (C3 adds the kernel)
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 10240;   // as kernels.cu: head size, key heads, value heads, q|k|v width
constexpr int RG = 4, RPG = S / RG, CB = 32, NCB = S / CB;
constexpr int WYC = 64;   // WY sub-chunk

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// ---------------------------------------------------------- the engine's kernel (kernels.cu, verbatim)
__global__ void __launch_bounds__(CB * RG) gdn_rec_cols_pipe_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                    const float* __restrict__ gate,
                                                                    const float* __restrict__ beta,
                                                                    float* __restrict__ oc_out, int64_t T) {
    constexpr int NT = CB * RG, LPT = S / NT;
    __shared__ float sk[S], sq[S], red[RG][CB];
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    float nq[LPT], nk[LPT], nv = 0.0f, ng = 0.0f, nb = 0.0f;
    auto fetch = [&](int64_t t) {
        const float* ht = h + t * C;
#pragma unroll
        for (int u = 0; u < LPT; ++u) { nq[u] = ht[qh * S + tid + u * NT]; nk[u] = ht[HK * S + qh * S + tid + u * NT]; }
        nv = ht[2 * HK * S + head * S + col];
        ng = gate[t * HV + head];
        nb = beta[t * HV + head];
    };
    if (T > 0) fetch(0);
    for (int64_t t = 0; t < T; ++t) {
        float cq[LPT], ck[LPT];
#pragma unroll
        for (int u = 0; u < LPT; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; }
        const float cv = nv, cg = ng, cbt = nb;
        __syncthreads();
#pragma unroll
        for (int u = 0; u < LPT; ++u) { sq[tid + u * NT] = cq[u]; sk[tid + u * NT] = ck[u]; }
        __syncthreads();
        if (t + 1 < T) fetch(t + 1);
        const float g = __expf(cg);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][c] = kv;
        __syncthreads();
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (cv - g * kv_col) * cbt;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][c] = o;
        __syncthreads();
        if (rg == 0) oc_out[t * HV * S + head * S + col] = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

// ---------------------------------------------------------- host references (per head, fp64)
// state layout as the kernel's: state[h][i][j], i = row (S), j = column (S); h[t] = {q(kh), k(kh), v(hv)} rows.

// serial token loop in double: the algebra the GPU kernel implements
void serial64(const double* h, const double* gate, const double* beta, const double* st0, double* oc, double* stT,
              int64_t T) {
    std::vector<double> s(S * S);
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) s[i * S + j] = st0[(size_t) i * HV * S + j];   // head 0 only (caller loops heads)
    for (int64_t t = 0; t < T; ++t) {
        const double* kt = h + HK * S + t * C;            // key rows (this head's key head = head 0's)
        const double* qt = h + t * C;
        const double* vt = h + 2 * HK * S + t * C;
        const double g = std::exp(gate[t]), b = beta[t];
        double delta[S];
        for (int j = 0; j < S; ++j) {
            double kv = 0;
            for (int i = 0; i < S; ++i) kv += s[i * S + j] * kt[i];
            delta[j] = (vt[j] - g * kv) * b;
        }
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) s[i * S + j] = g * s[i * S + j] + kt[i] * delta[j];
        for (int j = 0; j < S; ++j) {
            double o = 0;
            for (int i = 0; i < S; ++i) o += s[i * S + j] * qt[i];
            oc[t * S + j] = o * (1.0 / std::sqrt((double) S));
        }
    }
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) stT[(size_t) i * HV * S + j] = s[i * S + j];
}

// chunked WY form in double (tasks/gdn-wy-design-20261005.md), the standard split form: the CxC
// system matrix A is S0-INDEPENDENT (it only couples the tokens' k rows), so the S0 term factors:
//   u_j = u_v - M * S0[:,j]   per column j, with u_v = A^-1 (beta v) and M = A^-1 (diag(beta g dec) K)
// both solves per HEAD (once), and every per-column step (p^K_j = K S0[:,j], u_j, the outputs, the
// state) is a small GEMV - fully parallel over columns, which is what the device kernel column-splits.
void wy64(const double* h, const double* gate, const double* beta, const double* st0, double* oc, double* stT,
          int64_t T) {
    std::vector<double> s(S * S);
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) s[i * S + j] = st0[(size_t) i * HV * S + j];
    std::vector<double> A(WYC * WYC), uv(WYC), M(WYC * S), dec(WYC + 1), Dk(WYC * S), rhs(WYC);
    std::vector<double> pK(WYC), pQ(WYC), uj(WYC);
    auto fwd = [&](const double* rhs_in, double* out, int w, int width) {   // A (wxw, unit lower) \ rhs (wxwidth)
        for (int t = 0; t < w; ++t) {
            double acc = rhs_in[t * width];
            for (int s2 = 0; s2 < t; ++s2) acc -= A[t * WYC + s2] * out[s2 * width];
            out[t * width] = acc;
            // width > 1 handled by the caller passing strided columns; width == 1 is the vector case
        }
    };
    for (int64_t t0 = 0; t0 < T; t0 += WYC) {
        const int w = (int) std::min<int64_t>(WYC, T - t0);
        dec[0] = 1.0;
        for (int t = 0; t < w; ++t) dec[t + 1] = dec[t] * std::exp(gate[t0 + t]);
        // A[t][s] (s < t) = beta_t g_t dec[t] (dec[t]/dec[s+1]) (k_t . k_s); unit diagonal
        for (int t = 0; t < w; ++t) {
            const double b = beta[t0 + t], g = std::exp(gate[t0 + t]);
            const double* kt = h + HK * S + (t0 + t) * C;
            for (int s2 = 0; s2 <= t; ++s2) {
                if (s2 == t) { A[t * WYC + s2] = 1.0; continue; }
                const double* ks = h + HK * S + (t0 + s2) * C;
                double dk = 0;
                for (int i = 0; i < S; ++i) dk += kt[i] * ks[i];
                A[t * WYC + s2] = b * g * dec[t] * (dec[t] / dec[s2 + 1]) * dk;
            }
        }
        // u_v = A^-1 (beta v)  (per head, vector rhs)
        for (int t = 0; t < w; ++t) rhs[t] = beta[t0 + t] * h[2 * HK * S + (t0 + t) * C];
        for (int t = 0; t < w; ++t) {
            double acc = rhs[t];
            for (int s2 = 0; s2 < t; ++s2) acc -= A[t * WYC + s2] * uv[s2];
            uv[t] = acc;
        }
        // M = A^-1 (diag(beta_t g_t dec[t]) K)   (C x S solve: one forward substitution per state row i)
        for (int t = 0; t < w; ++t) {
            const double b = beta[t0 + t], g = std::exp(gate[t0 + t]);
            for (int i = 0; i < S; ++i) Dk[t * S + i] = b * g * dec[t] * h[HK * S + (t0 + t) * C + i];
        }
        for (int i = 0; i < S; ++i) {
            std::vector<double> c(w), mcol(w);
            for (int t = 0; t < w; ++t) c[t] = Dk[t * S + i];
            for (int t = 0; t < w; ++t) {
                double acc = c[t];
                for (int s2 = 0; s2 < t; ++s2) acc -= A[t * WYC + s2] * mcol[s2];
                mcol[t] = acc;
            }
            for (int t = 0; t < w; ++t) M[t * S + i] = mcol[t];
        }
        // per column j: p^K, p^Q, u_j, outputs, and the state column
        for (int j = 0; j < S; ++j) {
            for (int t = 0; t < w; ++t) {
                const double* kt = h + HK * S + (t0 + t) * C;
                const double* qt = h + (t0 + t) * C;
                double pk = 0, pq = 0;
                for (int i = 0; i < S; ++i) { pk += kt[i] * s[i * S + j]; pq += qt[i] * s[i * S + j]; }
                pK[t] = pk; pQ[t] = pq;
                double acc = uv[t];
                for (int i = 0; i < S; ++i) acc -= M[t * S + i] * s[i * S + j];
                uj[t] = acc;
            }
            for (int t = 0; t < w; ++t) {
                const double* qt = h + (t0 + t) * C;
                double o = dec[t + 1] * pQ[t];
                for (int s2 = 0; s2 <= t; ++s2) {
                    double dq = 0;
                    for (int i = 0; i < S; ++i) dq += qt[i] * h[HK * S + (t0 + s2) * C + i];
                    o += (dec[t + 1] / dec[s2 + 1]) * dq * uj[s2];
                }
                oc[(t0 + t) * S + j] = o * (1.0 / std::sqrt((double) S));
            }
            for (int i = 0; i < S; ++i) {
                double v = dec[w] * s[i * S + j];
                for (int s2 = 0; s2 < w; ++s2) v += (dec[w] / dec[s2 + 1]) * h[HK * S + (t0 + s2) * C + i] * uj[s2];
                s[i * S + j] = v;
            }
        }
    }
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) stT[(size_t) i * HV * S + j] = s[i * S + j];
}

int fixture(const char* name, int64_t T, int gates_mode, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    const size_t hz = (size_t) T * C;
    std::vector<double> h(hz), gate((size_t) T * HV), beta((size_t) T * HV);
    for (int64_t t = 0; t < T; ++t) {
        double* q = h.data() + t * C, * k = h.data() + HK * S + t * C, * v = h.data() + 2 * HK * S + t * C;
        for (int x = 0; x < S; ++x) { q[x] = nd(rng); k[x] = nd(rng); }
        for (int kh = 1; kh < HK; ++kh) {
            for (int x = 0; x < S; ++x) { q[kh * S + x] = nd(rng); k[kh * S + x] = nd(rng); }
        }
        for (int x = 0; x < S; ++x) v[x] = nd(rng);
        for (int kh = 1; kh < HK; ++kh)
            for (int x = 0; x < S; ++x) v[kh * S + x] = nd(rng);
        for (int kh = 0; kh < HK; ++kh) {   // L2-normalize q/k per key head (gdn_l2_kernel's contract)
            double nq = 0, nk = 0;
            for (int x = 0; x < S; ++x) { nq += q[kh * S + x] * q[kh * S + x]; nk += k[kh * S + x] * k[kh * S + x]; }
            nq = std::sqrt(nq); nk = std::sqrt(nk);
            for (int x = 0; x < S; ++x) { q[kh * S + x] /= nq; k[kh * S + x] /= nk; }
        }
        for (int x = S; x < C; ++x) h[t * C + x] = 0.0;   // padding clean
    }
    for (int64_t t = 0; t < T; ++t)
        for (int hd = 0; hd < HV; ++hd) {
            gate[t * HV + hd] = gates_mode == 0 ? 0.3 * nd(rng) : gates_mode == 1 ? 0.0 : gates_mode == 2 ? -6.0 : 0.3 * nd(rng);
            beta[t * HV + hd] = gates_mode == 3 ? 0.0 : ud(rng);
        }
    // head 0's state and tensors (the host references cover head 0; the GPU kernel runs all 48)
    std::vector<double> st0((size_t) S * HV * S), oc64((size_t) T * S), ocwy((size_t) T * S), stT64((size_t) S * HV * S),
        stTW((size_t) S * HV * S);
    for (size_t i = 0; i < st0.size(); ++i) st0[i] = nd(rng) * 0.1;
    serial64(h.data(), gate.data(), beta.data(), st0.data(), oc64.data(), stT64.data(), T);
    wy64(h.data(), gate.data(), beta.data(), st0.data(), ocwy.data(), stTW.data(), T);
    double e_oc = 0, e_st = 0, s_oc = 0;
    for (size_t i = 0; i < oc64.size(); ++i) {
        e_oc = std::max(e_oc, std::fabs(oc64[i] - ocwy[i]));
        s_oc = std::max(s_oc, std::fabs(oc64[i]));
    }
    for (size_t i = 0; i < S * HV * S; ++i) e_st = std::max(e_st, std::fabs(stT64[i] - stTW[i]));
    const bool ok = e_oc < 1e-8 && e_st < 1e-8;
    std::printf("%-28s T=%5d  max|serial64-wy64| oc %.3e state %.3e (scale %.2f) -> %s\n", name, (int) T, e_oc, e_st, s_oc,
                ok ? "ALGEBRA OK" : "MISMATCH");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    (void) bench;
    int fails = 0;
    fails += fixture("plain", 200, 0, 1);
    fails += fixture("T < WYC", 37, 0, 2);
    fails += fixture("T = WYC exactly", 64, 0, 3);
    fails += fixture("T = WYC+1", 65, 0, 4);
    fails += fixture("long", 4099, 0, 5);
    fails += fixture("gates = 0 (no decay)", 300, 1, 6);
    fails += fixture("gates = -6 (fast decay)", 300, 2, 7);
    fails += fixture("beta = 0 (pure decay)", 300, 3, 8);
    std::printf("%s (%d mismatches)\n", fails ? "WY ALGEBRA BROKEN" : "WY algebra proven vs serial", fails);
    return fails ? 1 : 0;
}
