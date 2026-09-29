// tests/rocm/native_qsa_ops_parity.cpp - independent check of two QSA sub-ops.
//
// No upstream test exists for either (their parity tests were in the omitted
// bench/micro tree, against llama.cpp's CUDA oracle). Both are small, both run
// per layer per token, and both have a detail that a plausible implementation
// could get wrong without any output looking strange:
//
//   * `native_qsa_rms_norm_weighted` picks a DIFFERENT reduction kernel at
//     n_cols < 1024 and at n_cols >= 1024 (256 threads vs 1024), so both branches
//     are exercised here - a bug in one branch would never show in the other.
//   * `native_qsa_gate_apply` reads its gate from the SECOND half of each
//     `q_full` row. A first-half read produces a perfectly well-formed output
//     that is wrong everywhere, so the fixture poisons the first half with values
//     chosen to produce a visibly different sigmoid and checks the output follows
//     the second half.
//
// Both are tolerance-based rather than bit-exact, and for the same reason: the
// reduction order and `rsqrtf`/`expf` are library and schedule dependent. The
// tolerances below are derived from what those cost, not chosen for convenience.
//
// One layout note worth recording: the header writes the shape as `[n_cols,n_rows]`,
// which reads as column-major, but the kernel indexes `blockIdx.x * n_cols + col` -
// row-major, with `n_cols` as the row width and the reduction axis. The test uses
// the kernel's reading, and asserts against a reference built the same way.
//
// Run: native_qsa_ops_parity --selftest

#include "strata/kernels/native_qsa.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

struct Rng {
    uint64_t s = 0x3f84d5b5b5470917ull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
};

// ---- native_qsa_rms_norm_weighted -------------------------------------------
int run_rms(Rng& rng, int n_cols, int n_rows, bool in_place) {
    using namespace strata::kernels;
    const float eps = 1e-5f;
    const size_t n = (size_t)n_cols * n_rows;

    std::vector<float> input(n), gamma(n_cols);
    for (auto& v : input) v = rng.uniform(-2.0f, 2.0f);
    for (auto& v : gamma) v = rng.uniform(-1.0f, 1.0f);

    // The device computes `(scale * x) * gamma` as two rounded fp32 multiplies, so
    // the reference mirrors that order rather than collapsing it into one fma.
    std::vector<double> ref(n);
    for (int r = 0; r < n_rows; ++r) {
        double ss = 0.0;
        for (int c = 0; c < n_cols; ++c) {
            const double x = input[(size_t)r * n_cols + c];
            ss += x * x;
        }
        const float mean = (float)(ss / (double)n_cols);
        const float scale = 1.0f / std::sqrt(mean + eps);
        for (int c = 0; c < n_cols; ++c) {
            const float x = input[(size_t)r * n_cols + c];
            ref[(size_t)r * n_cols + c] = (double)((float)((float)(scale * x) * gamma[c]));
        }
    }

    float *din = nullptr, *dgamma = nullptr, *dout = nullptr;
    HIPCHK(hipMalloc(&din, n * 4));
    HIPCHK(hipMalloc(&dgamma, (size_t)n_cols * 4));
    if (!in_place) HIPCHK(hipMalloc(&dout, n * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(din, input.data(), n * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dgamma, gamma.data(), (size_t)n_cols * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    native_qsa_rms_norm_weighted(din, dgamma, in_place ? din : dout, n_cols, n_rows, eps, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<float> got(n);
    HIPCHK(hipMemcpy(got.data(), in_place ? din : dout, n * 4, hipMemcpyDeviceToHost));

    double worst = 0.0, scale = 0.0;
    int nonfinite = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i])) { ++nonfinite; continue; }
        worst = std::fmax(worst, std::fabs((double)got[i] - ref[i]));
        scale = std::fmax(scale, std::fabs(ref[i]));
    }
    const double rel = worst / (scale + 1e-30);
    // rsqrtf is 1 ulp; the reduction is a tree over n_cols/32 partials plus a warp
    // tree, so ~log2(n_cols) roundings. 1e-6 covers both with room to spare.
    const double tol = 1e-6;
    std::printf("  %-34s n_cols=%-5d rows=%d in_place=%d  nonfinite=%d  rel=%.3e  (tol %.0e)\n",
                "rms_norm_weighted", n_cols, n_rows, (int)in_place, nonfinite, rel, tol);
    check(nonfinite == 0, "rms_norm_weighted must be finite");
    check(rel < tol, "rms_norm_weighted exceeded the tolerance justified by rsqrtf and the reduction");
    hipStreamDestroy(stream);
    hipFree(din); hipFree(dgamma); if (dout) hipFree(dout);
    return 0;
}

// ---- native_qsa_gate_apply ---------------------------------------------------
int run_gate(Rng& rng, int n_head, int head_dim, bool in_place, bool poison_first_half) {
    using namespace strata::kernels;
    const size_t n = (size_t)n_head * head_dim;

    std::vector<float> attn(n), q_full((size_t)n_head * 2 * head_dim);
    for (auto& v : attn) v = rng.uniform(-2.0f, 2.0f);
    for (auto& v : q_full) v = rng.uniform(-3.0f, 3.0f);
    if (poison_first_half) {
        // If the kernel read the first half instead of the second, sigmoid would be
        // ~1 here and ~0.5 for the real gate - a difference far outside tolerance.
        for (int h = 0; h < n_head; ++h)
            for (int c = 0; c < head_dim; ++c)
                q_full[(size_t)h * 2 * head_dim + c] = 12.0f;
        for (int h = 0; h < n_head; ++h)
            for (int c = 0; c < head_dim; ++c)
                q_full[(size_t)h * 2 * head_dim + head_dim + c] = -12.0f;
    }

    std::vector<double> ref(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t head = i / head_dim, channel = i % head_dim;
        const float raw = q_full[head * 2 * head_dim + head_dim + channel];   // SECOND half
        const double sig = 1.0 / (1.0 + std::exp(-(double)raw));
        ref[i] = (double)attn[i] * sig;
    }

    float *dattn = nullptr, *dq = nullptr, *dout = nullptr;
    HIPCHK(hipMalloc(&dattn, n * 4));
    HIPCHK(hipMalloc(&dq, q_full.size() * 4));
    if (!in_place) HIPCHK(hipMalloc(&dout, n * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dattn, attn.data(), n * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dq, q_full.data(), q_full.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    native_qsa_gate_apply(dattn, dq, in_place ? dattn : dout, n_head, head_dim, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<float> got(n);
    HIPCHK(hipMemcpy(got.data(), in_place ? dattn : dout, n * 4, hipMemcpyDeviceToHost));

    double worst = 0.0;
    int nonfinite = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i])) { ++nonfinite; continue; }
        worst = std::fmax(worst, std::fabs((double)got[i] - ref[i]));
    }
    std::printf("  %-34s heads=%d dim=%d in_place=%d poisoned_first_half=%d  nonfinite=%d  worst|err|=%.3e\n",
                "gate_apply", n_head, head_dim, (int)in_place, (int)poison_first_half, nonfinite, worst);
    check(nonfinite == 0, "gate_apply must be finite");
    check(worst < 1e-6, "gate_apply exceeded the tolerance justified by expf");
    hipStreamDestroy(stream);
    hipFree(dattn); hipFree(dq); if (dout) hipFree(dout);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: native_qsa_ops_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("native_qsa_ops_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: float64 with the kernel's own operation order mirrored in fp32\n");

    Rng rng;
    // both reduction branches, and the exact in-place alias the contract allows
    if (run_rms(rng, 256, 4, false) != 0) check(false, "rms small aborted");
    if (run_rms(rng, 1024, 2, false) != 0) check(false, "rms large branch aborted");
    if (run_rms(rng, 512, 3, true) != 0) check(false, "rms in-place aborted");
    // the gate half, and the poisoned first half
    if (run_gate(rng, 4, 128, false, false) != 0) check(false, "gate aborted");
    if (run_gate(rng, 2, 256, true, false) != 0) check(false, "gate in-place aborted");
    if (run_gate(rng, 4, 128, false, true) != 0) check(false, "gate poisoned aborted");

    std::printf("\nnative_qsa_ops: %d failures\n", failures);
    if (failures) return 1;
    std::printf("native_qsa_ops_parity OK\n");
    (void)selftest;
    return 0;
}
