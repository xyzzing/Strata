// tests/rocm/native_moe_combine_parity.cpp - independent check of the MoE combine.
//
// The contract in `include/strata/kernels/native_moe.hpp` is unusually specific
// about arithmetic, and that specificity is the whole point of this file:
//
//   "The first product rounds to F32, following products accumulate with FMA in
//    expert order, and optional shared is added once afterward."
//
// That is a statement about rounding, not about mathematics, so the reference here
// is an exact fp32 simulation of it - not a float64 ideal, which would agree with
// either accumulation order and therefore check nothing. Two exact simulations are
// computed:
//
//   fused   sum = fma(parts[e], w[e], sum)        for e = 1..k-1   <- what the contract says
//   rounded sum = sum + (parts[e] * w[e])         for e = 1..k-1   <- what -ffp-contract=off produces
//
// and the test reports which one the device matches bit-for-bit. This exists
// because this rollout builds every HIP translation unit with `-ffp-contract=off`
// (a deliberate decision, made so the in-tree parity references agree - see
// PORTING.md section 7). That flag is project-wide, and a project-wide flag can
// silently change a kernel whose contract promises fusion. Only a test that pins
// accumulation order can see the difference; a float64 comparison cannot.
//
// A float64 comparison is still reported, because it bounds how far either order
// is from the ideal.
//
// Run: native_moe_combine_parity --selftest

#include "strata/kernels/native_moe.hpp"

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
    uint64_t s = 0xbe5466cf34e90c6cull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
};

// Exact fp32 simulation of the two accumulation orders the contract distinguishes.
void reference(const std::vector<float>& parts, const std::vector<float>& w,
               const float* shared, int n_embd, int k,
               std::vector<float>& fused, std::vector<float>& rounded,
               std::vector<double>& ideal) {
    fused.assign((size_t)n_embd, 0.0f);
    rounded.assign((size_t)n_embd, 0.0f);
    ideal.assign((size_t)n_embd, 0.0);
    for (int c = 0; c < n_embd; ++c) {
        float f = parts[(size_t)c] * w[0];              // first product: rounded in both
        float r = f;
        double d = (double)parts[(size_t)c] * (double)w[0];
        for (int e = 1; e < k; ++e) {
            const float p = parts[(size_t)e * n_embd + c];
            f = std::fma(p, w[(size_t)e], f);           // contracted
            r = r + (p * w[(size_t)e]);                 // separately rounded
            d += (double)p * (double)w[(size_t)e];
        }
        if (shared) {
            f += shared[c];                             // no multiply, so identical either way
            r += shared[c];
            d += (double)shared[c];
        }
        fused[(size_t)c] = f;
        rounded[(size_t)c] = r;
        ideal[(size_t)c] = d;
    }
}

int run_case(const char* name, int n_embd, int k, bool with_shared, Rng& rng) {
    using namespace strata::kernels;

    std::vector<float> parts((size_t)k * n_embd), w((size_t)k), shared((size_t)n_embd);
    for (auto& v : parts) v = rng.uniform(-1.5f, 1.5f);
    for (auto& v : w) v = rng.uniform(-0.5f, 0.5f);
    for (auto& v : shared) v = rng.uniform(-0.3f, 0.3f);

    std::vector<float> fused, rounded;
    std::vector<double> ideal;
    reference(parts, w, with_shared ? shared.data() : nullptr, n_embd, k, fused, rounded, ideal);

    // How many of the two simulations actually differ? If none do, the case cannot
    // distinguish the orders and would report a match for the wrong reason.
    int distinguishable = 0;
    for (int c = 0; c < n_embd; ++c) if (fused[(size_t)c] != rounded[(size_t)c]) ++distinguishable;

    float *dp = nullptr, *dw = nullptr, *ds = nullptr, *dout = nullptr;
    HIPCHK(hipMalloc(&dp, parts.size() * 4));
    HIPCHK(hipMalloc(&dw, w.size() * 4));
    HIPCHK(hipMalloc(&dout, (size_t)n_embd * 4));
    if (with_shared) HIPCHK(hipMalloc(&ds, shared.size() * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dp, parts.data(), parts.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dw, w.data(), w.size() * 4, hipMemcpyHostToDevice, stream));
    if (with_shared) HIPCHK(hipMemcpyAsync(ds, shared.data(), shared.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    native_moe_combine(dp, dw, with_shared ? ds : nullptr, dout, n_embd, k, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<float> got((size_t)n_embd);
    HIPCHK(hipMemcpy(got.data(), dout, (size_t)n_embd * 4, hipMemcpyDeviceToHost));

    int diff_fused = 0, diff_rounded = 0, nonfinite = 0;
    double worst_ideal = 0.0;
    for (int c = 0; c < n_embd; ++c) {
        if (!std::isfinite(got[(size_t)c])) { ++nonfinite; continue; }
        if (std::memcmp(&got[(size_t)c], &fused[(size_t)c], 4) != 0) ++diff_fused;
        if (std::memcmp(&got[(size_t)c], &rounded[(size_t)c], 4) != 0) ++diff_rounded;
        const double den = std::fmax(1.0, std::fabs(ideal[(size_t)c]));
        worst_ideal = std::fmax(worst_ideal, std::fabs((double)got[(size_t)c] - ideal[(size_t)c]) / den);
    }

    const char* verdict = "neither order (!!)";
    if (diff_fused == 0) verdict = "FMA (contracted) - as the contract states";
    else if (diff_rounded == 0) verdict = "separately rounded - NOT what the contract states";

    std::printf("  %-28s k=%-3d shared=%d  means=%d  vs fused=%d  vs rounded=%d  ideal rel=%.2e  -> %s\n",
                name, k, (int)with_shared, distinguishable, diff_fused, diff_rounded,
                worst_ideal, verdict);

    check(nonfinite == 0, "the combine output must be finite");
    if (k > 1) {
        check(diff_fused == 0 || diff_rounded == 0,
              "the output must match one of the two exact fp32 accumulation orders");
        // the contract's own claim
        check(diff_fused == 0, "the contract says experts 1..k-1 accumulate with FMA");
    }
    check(distinguishable > 0 || k == 1,
          "the case must be able to tell the two orders apart, or it proves nothing");

    hipStreamDestroy(stream);
    hipFree(dp); hipFree(dw); hipFree(dout); if (ds) hipFree(ds);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: native_moe_combine_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("native_moe_combine_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: exact fp32 simulations of both accumulation orders the contract\n"
                "  distinguishes, plus a float64 ideal to bound the distance from either\n");

    Rng rng;
    run_case("single expert",        256, 1,  false, rng);   // the "ordinary multiply" path
    run_case("k=2",                  256, 2,  false, rng);
    run_case("k=2 with shared",      256, 2,  true,  rng);
    run_case("k=8 with shared",      512, 8,  true,  rng);
    run_case("k=15 with shared",     512, 15, true,  rng);   // the largest k the contract allows
    run_case("k=15 without shared",  128, 15, false, rng);

    std::printf("\nnative_moe_combine: %d failures\n", failures);
    if (failures) return 1;
    std::printf("native_moe_combine_parity OK\n");
    (void)selftest;
    return 0;
}
