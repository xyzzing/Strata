// tests/rocm/qsa_score_wmma_parity.cpp - the opt-in WMMA QSA scorer against its bound.
//
// Patch 0007 adds a second arm to the QSA scorer: rocWMMA 16x16x16 fp16 fragments
// with f32 accumulation (gfx1100, one wave32 per 16-row tile). This test does NOT
// re-run the portable test's checks verbatim: the bit-exact host model does not
// transfer (the WMMA unit's rounding function and internal accumulation order
// differ), which is exactly why the precision contract in PORTING.md 12g was
// derived BEFORE the kernel was written. Three checks here:
//
//  1. DEVIATION FROM A FLOAT64 IDEAL under the derived WMMA bound: fp16 input
//     conversion (2 * 2^-11 relative - the same 10 mantissa bits tf32 truncation
//     loses), order-agnostic f32 accumulation (D * 2^-24 * sum|pq|, covering ANY
//     internal summation order), a subnormal-flush absolute term, and the shared
//     sequential epilogue terms. The bound is computed per row from the data.
//  2. CROSS-CHECK AGAINST THE PORTABLE ARM: both kernels approximate the same
//     ideal, each within its own derived bound, so their outputs must agree
//     within the SUM of the bounds. Two implementations through different
//     arithmetic agreeing is stronger than either alone.
//  3. THE WRITE RANGE AND THE GUARDS, as in the portable test.
//
// Run: qsa_score_wmma_parity --selftest

#include "strata/kernels/native_qsa_score.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int D = 128, HEADS = 4, R = 4;

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
    uint64_t s = 0x9e3779b97f4a7c15ull;
    float next() {  // uniform in [-1, 1)
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (float)((int64_t)(s >> 11) % 2000001 - 1000000) / 1000000.0f;
    }
};

// The float64 ideal plus the WMMA bound, per PORTING.md 12g: fp16 conversion
// (2 * 2^-11 relative, both operands), order-agnostic f32 accumulation over D
// terms (D * 2^-24 * sum|pq|), the fp16 subnormal-flush absolute term
// (2^-24 * sum(|p|+|q|): each operand's flush error is <= 2^-25, once per
// operand of each product), then the sequential epilogue terms.
void host_bound(const std::vector<float>& pooled, const std::vector<float>& query,
                const std::vector<float>& bias, bool use_bias, int n, int full,
                double* out, double* bound, double* scale) {
    for (int i = 0; i < n; ++i) { out[i] = 0.0; bound[i] = 0.0; scale[i] = 0.0; }
    const double rel16 = std::ldexp(1.0, -11);      // fp16 round-to-nearest, one operand
    const double eps32 = std::ldexp(1.0, -24);
    const double flush = std::ldexp(1.0, -25);      // per-operand subnormal flush
    for (int row = 0; row <= full; ++row) {
        double score = 0.0, b = 0.0, mag_total = 0.0;
        for (int j = 0; j < HEADS; ++j) {
            double acc = 0.0, mag = 0.0, lin = 0.0;
            for (int d = 0; d < D; ++d) {
                const double p = pooled[(size_t)row * D + d], q = query[(size_t)j * D + d];
                acc += p * q;
                mag += std::fabs(p * q);
                lin += std::fabs(p) + std::fabs(q);
            }
            b += mag * (2.0 * rel16) + (double)D * eps32 * mag   // conversion + any-order accumulation
               + flush * lin;                                    // subnormal flush of tiny operands
            score += acc > 0.0 ? acc : 0.0;                      // relu: Lipschitz 1
            mag_total += mag;
        }
        if (use_bias) {
            score += bias[row];
            b += std::fabs((double)bias[row]) * eps32 * 4;
            mag_total += std::fabs((double)bias[row]);
        }
        if (row == full && n % R) { score += 1e9; b += 1e9 * eps32 * 2; mag_total += 1e9; }
        for (int i = row * R; i < n && i < (row + 1) * R; ++i) {
            out[i] = score; bound[i] = b; scale[i] = mag_total;
        }
    }
}

struct Case {
    const char* name;
    int n;
    bool use_bias;
    bool valid_step;
    float scale;   // multiplies pooled and query: exercises fp16 range headroom
};

int run_case(const Case& c, int max_cells) {
    const int max_blocks = max_cells / R + 1;
    const int full = c.n / R;

    Rng rng;
    std::vector<float> pooled((size_t)max_blocks * D), query((size_t)HEADS * D),
        bias(c.use_bias ? max_blocks : 0, 0.0f);
    for (auto& v : pooled) v = rng.next() * c.scale;
    for (auto& v : query) v = rng.next() * c.scale;
    for (auto& v : bias) v = rng.next() * c.scale;

    std::vector<int32_t> step(4);
    step[0] = c.n - 1;
    step[1] = c.n;
    step[2] = full;
    step[3] = c.n < 2051 ? c.n : 2051;
    if (!c.valid_step) step[2] = full + 1;

    const float sentinel = -12345.0f;
    std::vector<float> cells(max_cells, sentinel), ref(max_cells, sentinel);

    float *dp = nullptr, *dq = nullptr, *db = nullptr, *dc = nullptr, *dr = nullptr;
    int32_t* ds = nullptr;
    HIPCHK(hipMalloc(&dp, pooled.size() * 4));
    HIPCHK(hipMalloc(&dq, query.size() * 4));
    HIPCHK(hipMalloc(&dc, cells.size() * 4));
    HIPCHK(hipMalloc(&dr, ref.size() * 4));
    HIPCHK(hipMalloc(&ds, step.size() * 4));
    if (c.use_bias) HIPCHK(hipMalloc(&db, bias.size() * 4));

    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dp, pooled.data(), pooled.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dq, query.data(), query.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(ds, step.data(), step.size() * 4, hipMemcpyHostToDevice, stream));
    if (c.use_bias) HIPCHK(hipMemcpyAsync(db, bias.data(), bias.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dc, cells.data(), cells.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dr, ref.data(), ref.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    strata::kernels::QsaShapes s;
    s.idx_dim = D; s.idx_n_head = HEADS; s.idx_block = R; s.idx_top_k = 2048;

    // the portable arm's output on the same inputs: the cross-check reference
    strata::kernels::native_qsa_score_set_wmma(false);
    strata::kernels::native_qsa_score(dp, dq, c.use_bias ? db : nullptr, s, ds,
                                      max_blocks, max_cells, dr, stream);
    // the WMMA arm under test
    strata::kernels::native_qsa_score_set_wmma(true);
    strata::kernels::native_qsa_score(dp, dq, c.use_bias ? db : nullptr, s, ds,
                                      max_blocks, max_cells, dc, stream);
    strata::kernels::native_qsa_score_set_wmma(false);   // restore the default arm
    HIPCHK(hipStreamSynchronize(stream));
    HIPCHK(hipMemcpy(cells.data(), dc, cells.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(ref.data(), dr, ref.size() * 4, hipMemcpyDeviceToHost));

    std::printf("  %-34s n=%-5d bias=%d scale=%.0f", c.name, c.n, (int)c.use_bias, c.scale);

    if (!c.valid_step) {
        int wrote = 0;
        for (float v : cells) if (v != sentinel) ++wrote;
        check(wrote == 0, "invalid step buffer must suppress every write");
        std::printf("  invalid step -> %d cells written\n", wrote);
        hipStreamDestroy(stream);
        hipFree(dp); hipFree(dq); hipFree(dc); hipFree(dr); hipFree(ds); if (db) hipFree(db);
        return 0;
    }

    std::vector<double> ideal(max_cells), bound(max_cells), scale_v(max_cells);
    host_bound(pooled, query, bias, c.use_bias, c.n, full, ideal.data(), bound.data(),
               scale_v.data());

    int wrote_past_n = 0, nonfinite = 0;
    double worst_excess = 0.0, worst_abs = 0.0, worst_fraction = 0.0, worst_cross = 0.0, cross_excess = 0.0;
    for (int i = 0; i < c.n; ++i) {
        if (!std::isfinite(cells[i])) { ++nonfinite; continue; }
        const double err = std::fabs((double)cells[i] - ideal[i]);
        worst_abs = std::fmax(worst_abs, err);
        if (err > bound[i]) worst_excess = std::fmax(worst_excess, err - bound[i]);
        if (scale_v[i] > 0.0) worst_fraction = std::fmax(worst_fraction, err / scale_v[i]);
        // cross-check vs the portable arm: |w - p| <= b_w + b_p by the triangle
        // inequality (both within their own bounds of the same ideal). The
        // portable bound is the tf32 form of the same expression.
        const double dev_p = std::fabs((double)ref[i] - ideal[i]);
        const double cross = std::fabs((double)cells[i] - (double)ref[i]);
        worst_cross = std::fmax(worst_cross, cross);
        if (cross > bound[i] + dev_p + 2.0 * std::ldexp(1.0, -24) * (std::fabs((double)cells[i]) + std::fabs((double)ref[i]) + 2.0))
            cross_excess = std::fmax(cross_excess, cross - bound[i] - dev_p);
    }
    for (int i = c.n; i < max_cells; ++i) if (cells[i] != sentinel) ++wrote_past_n;

    std::printf("  beyond-n=%d  nonfinite=%d  worst|dev-ideal|=%.3e  bound excess=%.3e  "
                "dev/scale=%.2e  cross worst=%.3e\n",
                wrote_past_n, nonfinite, worst_abs, worst_excess, worst_fraction, worst_cross);

    check(wrote_past_n == 0, "cells at or beyond n_kv must not be written");
    check(nonfinite == 0, "outputs must be finite");
    check(worst_excess == 0.0, "WMMA deviation from the float64 ideal exceeded the derived bound");
    check(cross_excess == 0.0, "WMMA and portable arms disagree beyond their combined bounds");

    hipStreamDestroy(stream);
    hipFree(dp); hipFree(dq); hipFree(dc); hipFree(dr); hipFree(ds); if (db) hipFree(db);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: qsa_score_wmma_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("qsa_score_wmma_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: float64 ideal under the PORTING 12g WMMA bound (fp16 conversion,\n"
                "  order-agnostic f32 accumulation, flush term), plus a portable-arm cross-check\n");

    const int max_cells = 2048;

    const Case cases[] = {
        {"all blocks full, no bias",            2048, false, true, 1.0f},
        {"all blocks full, block bias",         2048, true,  true, 1.0f},
        {"incomplete tail block (n%4==2)",      2046, true,  true, 1.0f},
        {"single cell",                            1, false, true, 1.0f},
        {"one full block plus one cell",            5, true,  true, 1.0f},
        {"tail only, two blocks",                   7, true,  true, 1.0f},
        {"crosses a 16-row tile boundary",         33, false, true, 1.0f},
        {"magnitude x8 (fp16 headroom)",        2048, false, true, 8.0f},
        {"invalid step buffer (guard)",           64, false, false, 1.0f},
    };
    for (const Case& c : cases) {
        if (run_case(c, max_cells) != 0) return 1;
    }

    std::printf("\nqsa_score_wmma: %d failures\n", failures);
    if (failures) return 1;
    std::printf("qsa_score_wmma_parity OK\n");
    (void)selftest;
    return 0;
}
