// tests/rocm/qsa_score_parity.cpp - independent check of the ported QSA scorer.
//
// The upstream tree ships no reference for this kernel (its parity test lived in
// the omitted bench/micro/ tree), so this file provides one written from the
// documented contract in include/strata/kernels/native_qsa_score.hpp rather than
// from the kernel body:
//
//   cells[row*4+i] = sum_h relu( dot(pooled[row], query[h]) )
//                  + (bias ? bias[row] : 0)
//                  + (row == full && n % 4 ? 1e9 : 0)      for row*4+i < n
//
// Four things are checked, and they are deliberately different kinds of check:
//
//  1. DEVIATION FROM A FLOAT64 IDEAL. The kernel's operands are truncated to
//     tf32 (the CUDA path feeds raw F32 bits to mma.tf32), so it cannot equal a
//     float64 answer. The tolerance is not a magic number: it is the truncation
//     and accumulation error *derived per row from the data*, and the test
//     asserts the observed error stays inside it.
//  2. THE SAME ARITHMETIC ON THE HOST. A scalar host implementation with the
//     same truncation and the same accumulation order must agree bit-for-bit.
//     This catches an indexing or masking mistake that a loose bound would hide.
//  3. THE WRITE RANGE. Cells at and beyond n_kv must never be written.
//  4. THE INVALID-COUNT GUARDS. A step buffer that fails the contract must
//     suppress all writes rather than write something plausible.
//
// Run: qsa_score_parity --selftest
// Exit: 0 all checks passed, 1 a check failed, 2 usage.

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

float tf32(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u &= 0xFFFFE000u;
    std::memcpy(&x, &u, 4);
    return x;
}

float fadd(float a, float b) { return (float)((double)a + (double)b); }

struct Rng {
    uint64_t s = 0x9e3779b97f4a7c15ull;
    float next() {  // uniform in [-1, 1)
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (float)((int64_t)(s >> 11) % 2000001 - 1000000) / 1000000.0f;
    }
};

// The host model of the kernel's own arithmetic: same truncation, same
// sequential F32 order. Expected to be bit-identical to the device.
void host_exact(const std::vector<float>& pooled, const std::vector<float>& query,
                const std::vector<float>& bias, bool use_bias, int n, int full,
                float* out) {
    for (int i = 0; i < n; ++i) out[i] = 0.0f;
    for (int row = 0; row <= full; ++row) {
        float h[HEADS];
        for (int j = 0; j < HEADS; ++j) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) {
                acc = fadd(acc, tf32(pooled[(size_t)row * D + d]) * tf32(query[(size_t)j * D + d]));
            }
            h[j] = acc > 0.0f ? acc : 0.0f;
        }
        float sum = fadd(fadd(fadd(h[0], h[1]), h[2]), h[3]);
        if (use_bias) sum = fadd(sum, bias[row]);
        sum = fadd(sum, (row == full && n % R) ? 1e9f : 0.0f);
        sum = fadd(sum, 0.0f);
        for (int i = row * R; i < n && i < (row + 1) * R; ++i) out[i] = sum;
    }
}

// The float64 ideal plus the bound the tf32 truncation justifies, derived from
// the actual magnitudes in the row rather than assumed.
void host_bound(const std::vector<float>& pooled, const std::vector<float>& query,
                const std::vector<float>& bias, bool use_bias, int n, int full,
                double* out, double* bound, double* scale) {
    for (int i = 0; i < n; ++i) { out[i] = 0.0; bound[i] = 0.0; scale[i] = 0.0; }
    const double rel_tf32 = std::ldexp(1.0, -11);   // one truncated operand
    const double eps32 = std::ldexp(1.0, -24);
    for (int row = 0; row <= full; ++row) {
        double score = 0.0, b = 0.0, mag_total = 0.0;
        for (int j = 0; j < HEADS; ++j) {
            double acc = 0.0, mag = 0.0;
            for (int d = 0; d < D; ++d) {
                const double p = pooled[(size_t)row * D + d], q = query[(size_t)j * D + d];
                acc += p * q;
                mag += std::fabs(p * q);
            }
            b += mag * (2.0 * rel_tf32) + D * eps32 * mag;   // both operands truncated, then summed
            score += acc > 0.0 ? acc : 0.0;                  // relu: Lipschitz 1
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
};

int run_case(const Case& c, int max_cells) {
    const int max_blocks = max_cells / R + 1;
    const int full = c.n / R;

    Rng rng;
    std::vector<float> pooled((size_t)max_blocks * D), query((size_t)HEADS * D),
        bias(c.use_bias ? max_blocks : 0, 0.0f);
    for (auto& v : pooled) v = rng.next();
    for (auto& v : query) v = rng.next();
    for (auto& v : bias) v = rng.next();

    std::vector<int32_t> step(4);
    step[0] = c.n - 1;                                        // kStepPos
    step[1] = c.n;                                            // kStepNKv
    step[2] = full;                                           // kStepNBid
    step[3] = c.n < 2051 ? c.n : 2051;                        // kStepWidth
    if (!c.valid_step) step[2] = full + 1;                    // break the contract

    const float sentinel = -12345.0f;
    std::vector<float> host_cells(max_cells, sentinel), cells(max_cells, sentinel);

    float *dp = nullptr, *dq = nullptr, *db = nullptr, *dc = nullptr;
    int32_t* ds = nullptr;
    HIPCHK(hipMalloc(&dp, pooled.size() * 4));
    HIPCHK(hipMalloc(&dq, query.size() * 4));
    HIPCHK(hipMalloc(&dc, cells.size() * 4));
    HIPCHK(hipMalloc(&ds, step.size() * 4));
    if (c.use_bias) HIPCHK(hipMalloc(&db, bias.size() * 4));

    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dp, pooled.data(), pooled.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dq, query.data(), query.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(ds, step.data(), step.size() * 4, hipMemcpyHostToDevice, stream));
    if (c.use_bias) HIPCHK(hipMemcpyAsync(db, bias.data(), bias.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dc, cells.data(), cells.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    strata::kernels::QsaShapes s;
    s.idx_dim = D; s.idx_n_head = HEADS; s.idx_block = R; s.idx_top_k = 2048;
    strata::kernels::native_qsa_score(dp, dq, c.use_bias ? db : nullptr, s, ds,
                                      max_blocks, max_cells, dc, stream);
    HIPCHK(hipStreamSynchronize(stream));
    HIPCHK(hipMemcpy(cells.data(), dc, cells.size() * 4, hipMemcpyDeviceToHost));

    std::printf("  %-34s n=%-5d bias=%d", c.name, c.n, (int)c.use_bias);

    if (!c.valid_step) {
        int wrote = 0;
        for (float v : cells) if (v != sentinel) ++wrote;
        check(wrote == 0, "invalid step buffer must suppress every write");
        std::printf("  invalid step -> %d cells written\n", wrote);
        hipStreamDestroy(stream);
        hipFree(dp); hipFree(dq); hipFree(dc); hipFree(ds); if (db) hipFree(db);
        return 0;
    }

    std::vector<float> exact(max_cells), unused;
    host_exact(pooled, query, bias, c.use_bias, c.n, full, exact.data());
    std::vector<double> ideal(max_cells), bound(max_cells), scale(max_cells);
    host_bound(pooled, query, bias, c.use_bias, c.n, full, ideal.data(), bound.data(),
               scale.data());

    int exact_diff = 0, wrote_past_n = 0, nonfinite = 0;
    double worst_excess = 0.0, worst_abs = 0.0, worst_fraction = 0.0;
    for (int i = 0; i < c.n; ++i) {
        if (std::memcmp(&cells[i], &exact[i], 4) != 0) ++exact_diff;
        if (!std::isfinite(cells[i])) { ++nonfinite; continue; }
        const double err = std::fabs((double)cells[i] - ideal[i]);
        worst_abs = std::fmax(worst_abs, err);
        // A mathematically derived bound, not a chosen tolerance.
        if (err > bound[i]) worst_excess = std::fmax(worst_excess, err - bound[i]);
        // Reported, not asserted: the deviation as a fraction of the row's own
        // magnitude. Asserting a *relative* error against the result would be
        // unsatisfiable wherever ReLU plus cancellation lands the score near
        // zero while the summands are O(1) - the derived bound above is the
        // check, and this is the number that explains it.
        if (scale[i] > 0.0) worst_fraction = std::fmax(worst_fraction, err / scale[i]);
    }
    for (int i = c.n; i < max_cells; ++i) if (cells[i] != sentinel) ++wrote_past_n;

    std::printf("  exact=%d  beyond-n=%d  nonfinite=%d  worst|dev-ideal|=%.3e  "
                "bound excess=%.3e  dev/scale=%.2e\n",
                exact_diff, wrote_past_n, nonfinite, worst_abs, worst_excess, worst_fraction);

    check(exact_diff == 0, "host model of the same arithmetic must be bit-identical");
    check(wrote_past_n == 0, "cells at or beyond n_kv must not be written");
    check(nonfinite == 0, "outputs must be finite");
    check(worst_excess == 0.0, "deviation from the float64 ideal exceeded the derived bound");

    hipStreamDestroy(stream);
    hipFree(dp); hipFree(dq); hipFree(dc); hipFree(ds); if (db) hipFree(db);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: qsa_score_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("qsa_score_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: float64 ideal with a per-row derived bound, plus a host model\n"
                "  of the same tf32-sequential arithmetic (expected bit-identical)\n");

    // max_cells must equal idx_top_k's budget: 2048 cells, so 513 blocks.
    const int max_cells = 2048;

    const Case cases[] = {
        {"all blocks full, no bias",            2048, false, true},
        {"all blocks full, block bias",         2048, true,  true},
        {"incomplete tail block (n%4==2)",      2046, true,  true},
        {"single cell",                            1, false, true},
        {"one full block plus one cell",            5, true,  true},
        {"tail only, two blocks",                   7, true,  true},
        {"invalid step buffer (guard)",            64, false, false},
    };
    for (const Case& c : cases) {
        if (run_case(c, max_cells) != 0) return 1;
    }

    std::printf("\nqsa_score: %d failures\n", failures);
    if (failures) return 1;
    std::printf("qsa_score_parity OK\n");
    (void)selftest;
    return 0;
}
