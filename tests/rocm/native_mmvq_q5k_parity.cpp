// tests/rocm/native_mmvq_q5k_parity.cpp - independent check of the Q5_K decode-step GEMV.
//
// `native_mmvq(type, ...)` IS covered - `iq_parity` calls the dispatcher and it
// passes for all ten IQ/QK types. But that test's type list is
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_M, IQ4_NL, IQ4_XS, Q2_0, Q3_K:
// **Q5_K is not in it**, and Q5_K is one of the formats the packs use. This file
// covers the Q5_K path end to end through `native_q5_k_f32`, which composes the
// Q8_1 activation quantization with the quantized GEMV - i.e. exactly what a
// decode step does for a Q5_K projection.
//
// A correction worth recording, because it changed a gate decision. The previous
// checkpoint listed `native_mmvq_*` as untested on the strength of a name-based
// search: no test file is called `native_mmvq_parity`, so the name never appeared.
// The dispatcher is in fact exercised by `iq_parity`, whose MMVQ check compares
// against a float64 matvec for every type it covers. A name-based audit
// over-reports gaps; the audit was redone by grepping for the SYMBOLS.
//
// TWO CONSTRUCTIONS, for the same reason the MMQ test uses two:
//
//   * activations chosen to quantize to Q8_1 EXACTLY (integer multiples of 2^-5
//     with a +/-127 multiple per 32-block, so `d = amax/127` is a power of two and
//     `round(x/d)` is lossless) - this isolates the GEMV from the quantizer and
//     permits a tight bound;
//   * ordinary random activations, where the Q8_1 rounding dominates and the
//     tolerance is derived from what 8 bits per element can cost.
//
// The weights come from ggml's own reference quantizer and decoder, the oracle
// already justified in `prefill_mmq_parity.cpp`.
//
// Run: native_mmvq_q5k_parity --selftest

#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kQ5K = 13;          // GGML_TYPE_Q5_K

int failures = 0;
// Cases that could not run because the reference is absent. Counted, printed
// and turned into the declared skip signature at exit: a skipped case is not a
// passing one, and the suite must say so in its exit code.
int g_skipped = 0;

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
    uint64_t s = 0x1d8e4e27c47d124full;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
    int range(int lo, int hi) { return lo + (int)(next() % (uint32_t)(hi - lo + 1)); }
};

// Exact-quantizing activations: every value an integer multiple of 2^-5, with a
// +/-127 multiple in each 32-value block so the block scale is exactly 2^-5.
void build_exact_activations(int ncols, int n_in, std::vector<float>& x, Rng& rng) {
    x.assign((size_t)ncols * n_in, 0.0f);
    const float d0 = std::ldexp(1.0f, -5);
    for (int j = 0; j < ncols; ++j) {
        for (int b = 0; b < n_in / 32; ++b) {
            const int base = j * n_in + b * 32;
            x[base] = 127.0f * d0;
            for (int i = 1; i < 32; ++i) x[base + i] = (float)rng.range(-127, 127) * d0;
        }
    }
}

void build_random_activations(int ncols, int n_in, std::vector<float>& x, Rng& rng) {
    x.assign((size_t)ncols * n_in, 0.0f);
    for (auto& v : x) v = rng.uniform(-1.5f, 1.5f);
}

int run_case(const char* name, int n_in, int n_out, int ncols, bool exact_activations, Rng& rng) {
    using namespace strata::kernels;

    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type)kQ5K);
    if (tt == nullptr || tt->from_float_ref == nullptr || tt->to_float == nullptr) {
        std::printf("  %-30s ggml has no reference path for Q5_K; SKIPPED (not passed)\n", name);
        ++g_skipped;
        return 0;
    }
    if (n_in % (int)tt->blck_size != 0) {
        // A bad shape is a defect in THIS TEST, not a missing artifact: it must
        // read as a failure, never as a skip.
        check(false, "test bug: n_in must be a multiple of the Q5_K block size", name);
        return 0;
    }
    ggml_quantize_init((ggml_type)kQ5K);

    // Weights: n_out rows of n_in values, quantized by ggml and decoded by ggml.
    const size_t row_bytes = (size_t)(n_in / (int)tt->blck_size) * (size_t)tt->type_size;
    std::vector<uint8_t> w((size_t)n_out * row_bytes, 0);
    std::vector<double> w_ref((size_t)n_out * n_in, 0.0);
    {
        std::vector<float> row(n_in), back(n_in);
        for (int r = 0; r < n_out; ++r) {
            for (auto& v : row) v = rng.uniform(-1.0f, 1.0f);
            uint8_t* dst = w.data() + (size_t)r * row_bytes;
            tt->from_float_ref(row.data(), dst, n_in);
            tt->to_float(dst, back.data(), n_in);
            for (int k = 0; k < n_in; ++k) w_ref[(size_t)r * n_in + k] = (double)back[k];
        }
    }

    std::vector<float> x;
    if (exact_activations) build_exact_activations(ncols, n_in, x, rng);
    else build_random_activations(ncols, n_in, x, rng);

    const size_t y_elems = (size_t)ncols * n_out;
    const size_t scratch = native_q8_1_bytes(n_in, ncols);
    // NOTE the argument order: (type, n_in, n_out). Passing (type, n_out, n_in)
    // makes the helper divide by the wrong block count and throw from its own
    // validate_shape - which is how this test found out.
    check(native_mmvq_weight_bytes(kQ5K, n_in, n_out) == w.size(),
          "native_mmvq_weight_bytes must size the Q5_K weights");

    uint8_t* dw = nullptr;
    uint8_t* dscratch = nullptr;
    float* dx = nullptr;
    float* dy = nullptr;
    HIPCHK(hipMalloc(&dw, w.size()));
    HIPCHK(hipMalloc(&dscratch, scratch));
    HIPCHK(hipMalloc(&dx, x.size() * 4));
    HIPCHK(hipMalloc(&dy, y_elems * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dw, w.data(), w.size(), hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    native_q5_k_f32(dw, dx, dscratch, dy, n_in, n_out, ncols, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<float> y(y_elems);
    HIPCHK(hipMemcpy(y.data(), dy, y_elems * 4, hipMemcpyDeviceToHost));

    // float64 reference: everything except the activation rounding is exact.
    double worst_excess = 0.0, worst_err = 0.0, worst_rel = 0.0;
    int nonfinite = 0;
    const double q8_rel = exact_activations ? 0.0 : (0.5 / 127.0);   // half a step per element
    for (int j = 0; j < ncols; ++j) {
        for (int r = 0; r < n_out; ++r) {
            double acc = 0.0, mag = 0.0;
            for (int k = 0; k < n_in; ++k) {
                const double term = w_ref[(size_t)r * n_in + k] * (double)x[(size_t)j * n_in + k];
                acc += term;
                mag += std::fabs(term);
            }
            const float got = y[(size_t)j * n_out + r];
            if (!std::isfinite(got)) { ++nonfinite; continue; }
            const double err = std::fabs((double)got - acc);
            // The activation rounding contributes at most q8_rel * sum|w*x|; the
            // accumulation contributes the usual fp32 summation term.
            const double bound = q8_rel * mag + n_in * std::ldexp(1.0, -24) * mag + 1e-9;
            worst_excess = std::fmax(worst_excess, err - bound);
            worst_err = std::fmax(worst_err, err);
            worst_rel = std::fmax(worst_rel, err / (std::fabs(acc) + 1e-30));
        }
    }

    // `rel` is reported for information only: it is relative to the REFERENCE value,
    // which cancels to near zero for some rows, so it can be enormous on a case that
    // passes comfortably. The bound excess is the measure that means something.
    std::printf("  %-30s n_in=%-4d n_out=%-4d ncols=%d exact_acts=%d  nonfinite=%d  "
                "worst=%.2e  bound excess=%.2e  (rel %.1e, meaningless near a zero reference)\n",
                name, n_in, n_out, ncols, (int)exact_activations, nonfinite,
                worst_err, worst_excess, worst_rel);

    check(nonfinite == 0, "the GEMV output must be finite");
    check(worst_excess <= 0.0,
          "the GEMV exceeded the bound derived from the activation rounding and the summation");

    hipStreamDestroy(stream);
    hipFree(dw); hipFree(dscratch); hipFree(dx); hipFree(dy);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: native_mmvq_q5k_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("native_mmvq_q5k_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: float64 matvec over Q5_K blocks decoded by ggml's own reference decoder\n"
                "  (Q5_K is not in iq_parity's type list, so this path had no test)\n");

    Rng rng;
    // exact activations isolate the GEMV from the quantizer
    if (run_case("exact activations, 1 column",  512, 64, 1, true,  rng) != 0) {
        check(false, "case aborted early", "exact activations, 1 column");
    }
    if (run_case("exact activations, 2 columns", 512, 64, 2, true,  rng) != 0) {
        check(false, "case aborted early", "exact activations, 2 columns");
    }
    // ordinary activations: the Q8_1 rounding dominates and the bound reflects it
    if (run_case("random activations, 1 column", 256, 32, 1, false, rng) != 0) {
        check(false, "case aborted early", "random activations, 1 column");
    }
    if (run_case("random activations, 4 columns",512, 48, 4, false, rng) != 0) {
        check(false, "case aborted early", "random activations, 4 columns");
    }
    // a decode-shaped projection: 2560 -> 5120 is gate+up, run here at ncols = 1
    if (run_case("wide projection, 1 column",   2560, 256, 1, false, rng) != 0) {
        check(false, "case aborted early", "wide projection, 1 column");
    }

    std::printf("\nnative_mmvq_q5k: %d failures, %d skipped cases\n", failures, g_skipped);
    if (failures) return 1;
    if (g_skipped) {
        // Declared signature (tests/rocm/unavailable.json). Skipped coverage is
        // exit 5, never a pass.
        std::printf("native_mmvq_q5k_parity SKIPPED %d case(s): missing reference path\n", g_skipped);
        return 5;
    }
    std::printf("native_mmvq_q5k_parity OK\n");
    (void)selftest;
    return 0;
}
