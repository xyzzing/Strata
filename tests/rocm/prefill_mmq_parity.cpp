// tests/rocm/prefill_mmq_parity.cpp - independent check of the prompt path's MMQ,
// across every quantization type the path claims to cover.
//
// `src/prefill/moe_mmq.cu` routes prompt-time expert matrices through llama.cpp's
// MMQ kernels: the weights stay quantized and the activations are rounded to
// q8_1. This test covers all eight types MMQ claims (Q2_0 and the seven
// i-quants). The model's own weights are IQ3_S, so Q2_0 alone would not be
// enough - one passing format does not certify the others.
//
// WHAT THE ORACLE IS, AND WHY IT IS LEGITIMATE
//
// The weights are built and decoded through ggml's own reference path:
// `ggml_type_traits::from_float_ref` to produce VALID blocks and `to_float` to
// decode them. That is a clear scalar CPU implementation of the formats - in fact
// it is their definition - while the code under test is a separate GPU
// implementation of the same formats. Comparing them is a real check. The Q2_0
// case additionally cross-checks ggml's decoder against one written out here, so
// the oracle is not taken on trust for the format where that is cheap to avoid.
//
// WHAT IS AVOIDED, DELIBERATELY
//
// ggml's q8_1 MMQ activation layout is intricate (128-value groups, transposed,
// padded, scales and partial sums interleaved). Rather than decode it - and risk
// a test that fails because the TEST is wrong - the activations are chosen so
// that quantizing them is EXACT: integer multiples of 2^-5 with a +/-127 multiple
// in every block, so d = amax/127 = 2^-5 exactly (a power of two) and
// round(x/d) recovers the integer. The reference can then use the original float
// activations. If that assumption breaks the failure is wild rather than
// marginal, which is why the test says so before anything else.
//
// The tolerance is derived, not chosen: integer products accumulate exactly in
// int32 and the per-block scales are applied in fp32, so the bound is the usual
// summation bound over the reduction length. In practice every case is
// bit-exact, and the reported err/bound shows how much room is left.
//
// Run: prefill_mmq_parity --selftest

#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"
// ggml.h exposes types and type traits; the IQ2 quantize_* entry points used for
// the S3.2b' fixtures live in ggml/src/ggml-quants.h (same library).
#include "ggml-quants.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;
// Cases that could not run because the reference is absent. Counted and turned
// into the declared skip signature at exit: skipped is exit 5, never a pass.
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
    uint64_t s = 0x13198a2e03707344ull;
    uint32_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (uint32_t)(s >> 32);
    }
    int range(int lo, int hi) { return lo + (int)(next() % (uint32_t)(hi - lo + 1)); }
};

// Activations that quantize to q8_1 exactly: integer multiples of 2^-5, with a
// +/-127 multiple present in every 32-value block so d comes out as exactly 2^-5
// whatever grouping the quantizer uses.
void build_activations(int R, int K, std::vector<float>& x) {
    Rng rng;
    x.assign((size_t)R * K, 0.0f);
    const float d0 = std::ldexp(1.0f, -5);
    for (int r = 0; r < R; ++r) {
        for (int b = 0; b < K / 32; ++b) {
            const int base = r * K + b * 32;
            x[base] = 127.0f * d0;                        // pins amax = 127*d0
            for (int i = 1; i < 32; ++i) x[base + i] = (float)rng.range(-127, 127) * d0;
        }
    }
}

// Builds N x K of quantized weights of `type` and decodes them with ggml's CPU
// implementation. Returns false when ggml has no reference path for the type.
bool build_weights(int type, int N, int K, int& blck, std::vector<uint8_t>& blocks,
                   std::vector<double>& w) {
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type)type);
    if (tt == nullptr || tt->to_float == nullptr) return false;
    // ggml's i-quant quantizers read codebook maps that ggml_quantize_init()
    // builds; without it they abort with "forgot to call ggml_quantize_init()?".
    ggml_quantize_init((ggml_type)type);
    blck = (int)tt->blck_size;
    if (K % blck != 0) return false;
    // S3.2b' closes here: ggml sets from_float_ref = NULL for IQ2_XXS/IQ2_XS
    // because *real* quantization needs an importance matrix from calibration.
    // A fixture does not: the low-level quantize_iq2_xxs/xs accept any non-null
    // imatrix (ggml-quants.c asserts only non-null), and make_iq_fixtures.cpp
    // already uses exactly this for the dequant fixtures iq_parity verifies.
    // A uniform imatrix is valid, unsteered input - the same licence as the
    // random weights below.
    if (tt->from_float_ref == nullptr && type != 16 && type != 17) return false;

    const size_t row_bytes = (size_t)(K / blck) * (size_t)tt->type_size;
    blocks.assign(row_bytes * (size_t)N, 0);
    w.assign((size_t)N * K, 0.0);

    Rng rng;
    std::vector<float> row(K), back(K);
    for (int r = 0; r < N; ++r) {
        for (int k = 0; k < K; ++k) row[k] = (float)rng.range(-1000, 1000) / 1000.0f;
        uint8_t* dst = blocks.data() + (size_t)r * row_bytes;
        if (type == 16) {
            // A uniform importance matrix: valid input, no steering.
            std::vector<float> imatrix((size_t)K, 1.0f);
            if (quantize_iq2_xxs(row.data(), dst, 1, K, imatrix.data()) != row_bytes)
                return false;
        } else if (type == 17) {
            std::vector<float> imatrix((size_t)K, 1.0f);
            if (quantize_iq2_xs(row.data(), dst, 1, K, imatrix.data()) != row_bytes)
                return false;
        } else {
            tt->from_float_ref(row.data(), dst, K);
        }
        tt->to_float(dst, back.data(), K);
        for (int k = 0; k < K; ++k) w[(size_t)r * K + k] = (double)back[k];
    }
    return true;
}

int run_case(const char* name, int type, int R, int N, int K) {
    using namespace strata::prefill::mmq;

    if (!supported(type)) {
        std::printf("  %-26s mmq::supported() says no; SKIPPED (not passed)\n", name);
        ++g_skipped;
        return 0;
    }
    int blck = 0;
    std::vector<uint8_t> w_blocks;
    std::vector<double> w;
    if (!build_weights(type, N, K, blck, w_blocks, w)) {
        std::printf("  %-26s no ggml reference path for this type; SKIPPED (not passed)\n", name);
        ++g_skipped;
        return 0;
    }

    std::vector<float> x;
    build_activations(R, K, x);

    const size_t w_bytes = matrix_bytes(type, N, K);
    const size_t xq_bytes = q8_bytes(R, K);
    check(w_bytes == w_blocks.size(), "matrix_bytes must size the weight blocks");

    uint8_t *dw = nullptr, *dxq = nullptr;
    float *dx = nullptr, *ddst = nullptr;
    int32_t *dbounds = nullptr, *dids = nullptr;
    HIPCHK(hipMalloc(&dw, w_bytes));
    HIPCHK(hipMalloc(&dxq, xq_bytes));
    HIPCHK(hipMalloc(&dx, (size_t)R * K * 4));
    HIPCHK(hipMalloc(&ddst, (size_t)R * N * 4));
    HIPCHK(hipMalloc(&dbounds, 2 * 4));
    HIPCHK(hipMalloc(&dids, (size_t)R * 4));

    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));

    std::vector<int32_t> bounds = {0, R}, ids(R);
    for (int i = 0; i < R; ++i) ids[i] = i;
    std::vector<float> dst((size_t)R * N, -999.0f);

    // `x` must be a DEVICE pointer: quantize() is a kernel launcher. Passing host
    // memory here faulted the GPU on a host address, which is how that was found.
    HIPCHK(hipMemcpyAsync(dx, x.data(), (size_t)R * K * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dw, w_blocks.data(), w_bytes, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dbounds, bounds.data(), 8, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dids, ids.data(), (size_t)R * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(ddst, dst.data(), (size_t)R * N * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    quantize(dx, nullptr, dxq, type, K, K, R, stream);

    Product p;
    p.w = dw;
    p.type = type;
    p.w_rows = N;
    p.w_cols = K;
    p.expert_bytes = w_bytes;
    p.n = 1;
    p.xq = dxq;
    p.bounds = dbounds;
    p.ids = dids;
    p.total_rows = R;
    p.max_rows = R;
    p.dst = ddst;
    p.ld_dst = N;

    Context ctx;
    ctx.run(p, stream);
    {
        const hipError_t e = hipStreamSynchronize(stream);
        if (e != hipSuccess) {
            check(false, "mmq run failed", hipGetErrorString(e));
            return 0;
        }
    }
    HIPCHK(hipMemcpy(dst.data(), ddst, (size_t)R * N * 4, hipMemcpyDeviceToHost));

    double worst = 0.0, worst_fraction = 0.0;
    int nonfinite = 0, out_of_bound = 0, bit_exact = 1;
    for (int r = 0; r < R; ++r) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0, mag = 0.0;
            for (int k = 0; k < K; ++k) {
                const double term = w[(size_t)n * K + k] * (double)x[(size_t)r * K + k];
                acc += term;
                mag += std::fabs(term);
            }
            const float got = dst[(size_t)r * N + n];
            if (!std::isfinite(got)) { ++nonfinite; continue; }
            const double err = std::fabs((double)got - acc);
            const double bound = K * std::ldexp(1.0, -24) * mag;
            if (err > bound) ++out_of_bound;
            if (bound > 0.0) worst_fraction = std::fmax(worst_fraction, err / bound);
            if (err != 0.0) bit_exact = 0;
            worst = std::fmax(worst, err);
        }
    }

    std::printf("  %-26s blck=%-3d K=%-4d N=%-4d  nonfinite=%d out-of-bound=%d  "
                "worst|err|=%.3e  %s  err/bound=%.3f\n",
                name, blck, K, N, nonfinite, out_of_bound, worst,
                bit_exact ? "BIT-EXACT" : "         ", worst_fraction);

    check(nonfinite == 0, "MMQ output must be finite");
    check(out_of_bound == 0,
          "MMQ deviated from the ggml-decoded reference beyond the summation bound "
          "(if this is wild rather than marginal, question the exact-quantization "
          "assumption in build_activations first)");

    hipStreamDestroy(stream);
    hipFree(dw); hipFree(dx); hipFree(dxq); hipFree(ddst); hipFree(dbounds); hipFree(dids);
    return 0;
}

// The Q2_0 decoder is small enough to write out here, so it is: it cross-checks
// ggml's to_float instead of taking the oracle on trust.
void q2_0_format_crosscheck() {
    Rng rng;
    const int K = 256;
    std::vector<float> row(K), mine(K), theirs(K);
    for (int k = 0; k < K; ++k) row[k] = (float)rng.range(-1000, 1000) / 1000.0f;
    std::vector<uint8_t> blk((size_t)(K / 64) * 18, 0);
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type)42);
    tt->from_float_ref(row.data(), blk.data(), K);

    // documented layout: {fp16 d; uint8_t qs[16]} over 64 values,
    // value = (code - 1) * d, low bits first
    for (size_t b = 0; b < blk.size() / 18; ++b) {
        uint16_t dh;
        std::memcpy(&dh, blk.data() + b * 18, 2);
        const uint32_t sign = (uint32_t)(dh & 0x8000u) << 16;
        const uint32_t exp = (dh >> 10) & 0x1fu, mant = dh & 0x3ffu;
        float d = 0.0f;
        if (exp != 0) {
            const uint32_t u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
            std::memcpy(&d, &u, 4);
        }
        for (int j = 0; j < 64; ++j) {
            const uint8_t byte = blk[b * 18 + 2 + j / 4];
            const int q = (byte >> ((j % 4) * 2)) & 0x03;
            mine[b * 64 + j] = (float)(q - 1) * d;
        }
    }
    tt->to_float(blk.data(), theirs.data(), K);

    int differ = 0;
    for (int k = 0; k < K; ++k) if (mine[k] != theirs[k]) ++differ;
    std::printf("  %-26s %d of %d values differ from ggml's decoder\n",
                "Q2_0 format cross-check", differ, K);
    check(differ == 0, "the hand-written Q2_0 decoder must agree with ggml's");
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: prefill_mmq_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("prefill_mmq_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  weights: built and decoded by ggml's own reference quantizer/dequantizer\n"
                "  activations: chosen to quantize to q8_1 exactly, so the reference may use\n"
                "  the original floats (stated first because it is the load-bearing assumption)\n");

    // The eight types MMQ claims, by their real ggml ids from the pinned ggml.h.
    const struct { int type; const char* name; } types[] = {
        {42, "Q2_0"}, {16, "IQ2_XXS"}, {17, "IQ2_XS"}, {22, "IQ2_S"},
        {18, "IQ3_XXS"}, {21, "IQ3_S"}, {20, "IQ4_NL"}, {23, "IQ4_XS"},
    };
    int supported_count = 0;
    for (const auto& t : types) if (strata::prefill::mmq::supported(t.type)) ++supported_count;
    std::printf("  mmq::supported() reports %d of 8 pack types supported\n", supported_count);
    check(supported_count == 8, "every type the MMQ path claims must report supported");

    q2_0_format_crosscheck();

    // Model-shaped: gate+up is [1280, N], down is [N, 640]; K is a multiple of
    // both QK2_0 (64) and QK_K (256), which the i-quants require.
    for (const auto& t : types) {
        if (run_case(t.name, t.type, 4, 1280, 512) != 0) {
            check(false, "case aborted early", t.name);
        }
    }
    // and a long-reduction, few-row case: the shape a short prompt produces
    if (run_case("IQ3_S single row", 21, 1, 256, 1024) != 0) {
        check(false, "case aborted early", "IQ3_S single row");
    }

    std::printf("\nprefill_mmq: %d failures, %d skipped cases\n", failures, g_skipped);
    if (failures) return 1;
    if (g_skipped) {
        // Declared signature (tests/rocm/unavailable.json). A case with no
        // reference ran zero comparisons; that is skipped coverage, exit 5.
        std::printf("prefill_mmq_parity SKIPPED %d case(s): missing reference path\n", g_skipped);
        return 5;
    }
    std::printf("prefill_mmq_parity OK\n");
    (void)selftest;
    return 0;
}
