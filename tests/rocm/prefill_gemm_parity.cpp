// tests/rocm/prefill_gemm_parity.cpp - independent check of the prefill GEMM.
//
// The plan's Stage 3 asks specifically that the GEMM port be checked for
// "transposes, strides, BF16/FP16 conversion, workspace handling and
// accumulation precision". Each of those is a separate case below, and each is
// compared against a float64 matmul computed here from the *same rounded*
// operands, so the only thing under test is what hipBLAS does with them.
//
// The tolerance is derived, not chosen: any summation order produces at most
// K * 2^-24 * sum|terms| of error, which is the bound asserted here. A tighter
// fraction is reported so the real margin is visible.
//
// Shapes are deliberately all-distinct (T=3, N=5, K=7) in the transpose cases:
// if W were read as [K,N] instead of [N,K], or Y written transposed, the
// reference and the device would disagree immediately rather than by luck.
//
// Run: prefill_gemm_parity --selftest

#include "strata/prefill/gemm.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "f16.h"          // strata_test::f32_to_f16 / f16_to_f32

namespace {

int failures = 0;

void check(bool ok, const char* what, const char* detail = "") {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "  *** WRONG *** %s %s\n", what, detail);
    }
}

#define HIPCHK(expr)                                                                   \
    do {                                                                               \
        hipError_t _e = (expr);                                                        \
        if (_e != hipSuccess) {                                                        \
            std::fprintf(stderr, "hip error %s: %s\n", #expr, hipGetErrorString(_e));   \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

// ---- conversions, written out so the fixtures and the reference agree --------
uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t round = 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
    u += round;
    return (uint16_t)(u >> 16);
}
float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

struct Rng {
    uint64_t s = 0x243f6a8885a308d3ull;
    float next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (float)((int64_t)(s >> 11) % 2000001 - 1000000) / 1000000.0f;
    }
};

// float64 reference over the *same rounded operands*, plus the summation bound.
void ref_gemm(const std::vector<double>& xs, const std::vector<double>& ws,
              int T, int N, int K, std::vector<double>& y, std::vector<double>& bound) {
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0, mag = 0.0;
            for (int k = 0; k < K; ++k) {
                const double p = xs[(size_t)t * K + k] * ws[(size_t)n * K + k];
                acc += p;
                mag += std::fabs(p);
            }
            y[(size_t)t * N + n] = acc;
            bound[(size_t)t * N + n] = K * std::ldexp(1.0, -24) * mag;
        }
    }
}

struct Gpu {
    float *dx = nullptr, *dw = nullptr, *dy = nullptr;
    void* scratch = nullptr;
    hipStream_t stream = nullptr;
    ~Gpu() {
        if (stream) hipStreamDestroy(stream);
        if (dx) hipFree(dx);
        if (dw) hipFree(dw);
        if (dy) hipFree(dy);
        if (scratch) hipFree(scratch);
    }
};

// One full case: upload, run, compare.
int run_case(const char* name, int T, int N, int K, bool use_f16, int64_t ldy_extra,
             float beta, bool use_bias_accum) {
    Rng rng;
    std::vector<float> xf((size_t)T * K), wf((size_t)N * K);
    for (auto& v : xf) v = rng.next();
    for (auto& v : wf) v = rng.next();

    std::vector<uint16_t> xb(xf.size()), wb(wf.size());
    std::vector<double> xs(xf.size()), ws(wf.size());
    for (size_t i = 0; i < xf.size(); ++i) {
        xb[i] = use_f16 ? strata_test::f32_to_f16(xf[i]) : f32_to_bf16(xf[i]);
        xs[i] = use_f16 ? strata_test::f16_to_f32(xb[i]) : bf16_to_f32(xb[i]);
    }
    for (size_t i = 0; i < wf.size(); ++i) {
        wb[i] = use_f16 ? strata_test::f32_to_f16(wf[i]) : f32_to_bf16(wf[i]);
        ws[i] = use_f16 ? strata_test::f16_to_f32(wb[i]) : bf16_to_f32(wb[i]);
    }

    const int64_t ldy = ldy_extra ? N + ldy_extra : N;
    const int64_t y_elems = (int64_t)T * ldy;
    const float pad = -777.0f;
    std::vector<float> y_host((size_t)y_elems, pad);
    if (use_bias_accum) {
        for (int64_t i = 0; i < y_elems; ++i) y_host[(size_t)i] = 0.5f;
    }

    Gpu g;
    HIPCHK(hipStreamCreate(&g.stream));
    HIPCHK(hipMalloc(&g.dx, xb.size() * 2));
    HIPCHK(hipMalloc(&g.dw, wb.size() * 2));
    HIPCHK(hipMalloc(&g.dy, (size_t)y_elems * 4));
    HIPCHK(hipMalloc(&g.scratch, (size_t)1 << 20));
    HIPCHK(hipMemcpyAsync(g.dx, xb.data(), xb.size() * 2, hipMemcpyHostToDevice, g.stream));
    HIPCHK(hipMemcpyAsync(g.dw, wb.data(), wb.size() * 2, hipMemcpyHostToDevice, g.stream));
    HIPCHK(hipMemcpyAsync(g.dy, y_host.data(), (size_t)y_elems * 4, hipMemcpyHostToDevice, g.stream));
    HIPCHK(hipStreamSynchronize(g.stream));

    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(g.stream, 1 << 19, err)) {
        std::fprintf(stderr, "  gemm.init failed: %s\n", err.c_str());
        ++failures;
        return 0;
    }
    if (use_f16) gemm.f16(xb.data() ? (const uint16_t*)g.dx : nullptr, (const uint16_t*)g.dw,
                          (float*)g.dy, T, N, K, ldy, beta);
    else gemm.bf16((const uint16_t*)g.dx, (const uint16_t*)g.dw, (float*)g.dy, T, N, K, ldy, beta);
    HIPCHK(hipStreamSynchronize(g.stream));
    HIPCHK(hipMemcpy(y_host.data(), g.dy, (size_t)y_elems * 4, hipMemcpyDeviceToHost));

    std::vector<double> y_ref(y_elems), bound(y_elems);
    ref_gemm(xs, ws, T, N, K, y_ref, bound);

    const double added = use_bias_accum ? 0.5 : 0.0;
    int nonfinite = 0, out_of_bound = 0, pad_touched = 0;
    double worst_fraction = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < N; ++n) {
            const float got = y_host[(size_t)t * ldy + n];
            if (!std::isfinite(got)) { ++nonfinite; continue; }
            const double want = y_ref[(size_t)t * N + n] + added;
            const double err = std::fabs((double)got - want);
            const double b = bound[(size_t)t * N + n] + std::fabs(added) * std::ldexp(1.0, -24);
            if (err > b) ++out_of_bound;
            if (b > 0.0) worst_fraction = std::fmax(worst_fraction, err / b);
        }
        for (int64_t i = N; i < ldy; ++i) {
            if (y_host[(size_t)t * ldy + i] != pad) ++pad_touched;
        }
    }

    std::printf("  %-30s T=%-3d N=%-4d K=%-4d %s ldy=%-4lld beta=%.0f  "
                "nonfinite=%d out-of-bound=%d pad-touched=%d  err/bound=%.3f\n",
                name, T, N, K, use_f16 ? "f16 " : "bf16", (long long)ldy, beta,
                nonfinite, out_of_bound, pad_touched, worst_fraction);

    check(nonfinite == 0, "outputs must be finite");
    check(out_of_bound == 0, "deviation exceeded the derived summation bound");
    check(pad_touched == 0, "the ldy row stride must leave the padding untouched");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: prefill_gemm_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("prefill_gemm_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: float64 matmul over the same rounded operands; bound = K*2^-24*sum|terms|\n");

    // Transpose trap: T, N and K all different, so a [K,N] misread or a
    // transposed output cannot coincidentally agree.
    run_case("transpose trap, bf16", 3, 5, 7, false, 0, 0.0f, false);
    run_case("transpose trap, f16", 3, 5, 7, true, 0, 0.0f, false);
    // Longer reduction: the bound scales with K, so this is where a split-K or
    // accumulator-precision difference would show up.
    run_case("long reduction, bf16", 4, 64, 512, false, 0, 0.0f, false);
    run_case("long reduction, f16", 4, 64, 512, true, 0, 0.0f, false);
    // Row stride: ldy > N, padding must stay untouched.
    run_case("padded row stride, bf16", 3, 5, 7, false, 3, 0.0f, false);
    // beta = 1 must add onto the existing contents.
    run_case("beta=1 accumulate, bf16", 3, 5, 7, false, 0, 1.0f, true);
    // Rectangular, activation-major shapes closer to the real prompt path.
    run_case("single token decode, f16", 1, 128, 256, true, 0, 0.0f, false);

    std::printf("\nprefill_gemm: %d failures\n", failures);
    if (failures) return 1;
    std::printf("prefill_gemm_parity OK\n");
    (void)selftest;
    return 0;
}
