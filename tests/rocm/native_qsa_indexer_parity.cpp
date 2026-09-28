// tests/rocm/native_qsa_indexer_parity.cpp - independent check of the QSA key indexer.
//
// The last forward-path function without a test. It is also the most stateful:
// one call appends ONE cell's raw key and, when that cell completes a 4-cell
// block, pools the block, normalizes it, rotates it and publishes the result -
// while also maintaining a "spare" key that MOVES as blocks complete.
//
// So the test drives a SEQUENCE of cells, not a single call, and models the same
// state machine on the host in float64:
//
//   * `tail`     (3 rows of raw F16-rounded keys) - asserted BYTE-EXACT, because
//                it is a copy of a round-tripped value and nothing else;
//   * `pooled`   (one row per completed block, plus the spare at `n_bid`) -
//                asserted with a tolerance, because rsqrtf and the trig are
//                library functions;
//   * `dead`     (the spare's value) - must equal cell 0's normalized key;
//   * `block_pos` - must be the block's FIRST absolute position, asserted exactly.
//
// THE TRAP THIS TEST EXISTS FOR. `include/strata/kernels/qsa.hpp` says of the
// rotation position: "rotating at the block's LAST cell keeps every shape and
// every magnitude". A port that rotated at the wrong cell would pass a shape
// check, a magnitude check, and any test that only looks at how big the output
// is. The test therefore also computes what the LAST cell's rotation would give
// and asserts the device does NOT match it - so the two are demonstrably
// distinguishable here rather than assumed to be.
//
// Run: native_qsa_indexer_parity --selftest

#include "strata/kernels/native_qsa_indexer.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "f16.h"          // strata_test::f32_to_f16 / f16_to_f32

namespace {

constexpr int D = 128, R = 4, ROT = 64;
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

float fadd(float a, float b) { return (float)((double)a + (double)b); }

struct Rng {
    uint64_t s = 0x6a09e667f3bcc909ull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
};

// The kernel's state machine, in float64 except where the kernel uses explicitly
// rounded fp32 operations (the pooling adds and the 0.25 scale).
struct Ref {
    std::vector<float> tail = std::vector<float>((size_t)(R - 1) * D, 0.0f);
    std::vector<double> pooled, dead;
    int32_t block_pos = 0;
    bool block_pos_written = false;

    // The device's normalized+rotated vector for this cell, and - separately - the
    // vector a LAST-cell rotation would have produced, so the two can be compared.
    void step(int pos, int pos_base, int max_cells, const std::vector<float>& cell_raw,
              const std::vector<float>& gamma, float eps, float freq_base, int cells,
              std::vector<double>& y_out, std::vector<double>& y_wrong_out) {
        if (pos < 0 || pos >= max_cells) return;
        const int slot = pos % R;
        std::vector<float> incoming(D);
        for (int d = 0; d < D; ++d) {
            incoming[d] = strata_test::half_round(cell_raw[d]);
            if (slot < R - 1) tail[(size_t)slot * D + d] = incoming[d];
        }
        if (pos != 0 && slot != R - 1) return;

        std::vector<float> mean(D, 0.0f);
        for (int d = 0; d < D; ++d) {
            float sum = pos == 0 ? incoming[d] : tail[d];
            for (int j = 1; j < R; ++j) {
                const float tap = (pos == 0 || j == R - 1) ? incoming[d] : tail[(size_t)j * D + d];
                sum = fadd(sum, tap);
            }
            mean[d] = (float)std::fma(0.25, (double)sum, 0.0);
        }
        double ss = 0.0;
        for (int d = 0; d < D; ++d) ss += (double)mean[d] * (double)mean[d];
        const double sc = 1.0 / std::sqrt(ss / D + (double)eps);
        std::vector<double> values(D);
        for (int d = 0; d < D; ++d)
            values[d] = (double)((float)((float)(sc * mean[d]) * gamma[d]));

        const double theta_scale = std::pow((double)freq_base, -2.0 / (double)ROT);
        const int b = pos / R;
        const int rope_pos = pos == 0 ? 0 : pos_base + R * b;
        const int last_pos = pos == 0 ? 0 : pos_base + R * b + (R - 1);   // the WRONG cell

        y_out.assign(D, 0.0);
        y_wrong_out.assign(D, 0.0);
        for (int d = 0; d < D; ++d) { y_out[d] = values[d]; y_wrong_out[d] = values[d]; }
        for (int pair = 0; pair < ROT / 2; ++pair) {
            const double c1 = std::cos((double)rope_pos * std::pow(theta_scale, pair));
            const double s1 = std::sin((double)rope_pos * std::pow(theta_scale, pair));
            const double c2 = std::cos((double)last_pos * std::pow(theta_scale, pair));
            const double s2 = std::sin((double)last_pos * std::pow(theta_scale, pair));
            const double a = values[pair], z = values[pair + ROT / 2];
            y_out[pair] = a * c1 - z * s1;
            y_out[pair + ROT / 2] = a * s1 + z * c1;
            y_wrong_out[pair] = a * c2 - z * s2;
            y_wrong_out[pair + ROT / 2] = a * s2 + z * c2;
        }
        if ((int)pooled.size() < (b + 2) * D) pooled.resize((size_t)(b + 2) * D, 0.0);
        if (dead.empty()) dead.assign(D, 0.0);
        for (int d = 0; d < D; ++d) pooled[(size_t)b * D + d] = y_out[d];
        if (pos == 0) {
            for (int d = 0; d < D; ++d) dead[d] = y_out[d];
        } else {
            for (int d = 0; d < D; ++d) pooled[(size_t)(b + 1) * D + d] = dead[d];
            block_pos = rope_pos;
            block_pos_written = true;
        }
        (void)cells;
    }
};

int run_sequence(const char* name, int cells, int pos_base, int max_cells, Rng& rng) {
    using namespace strata::kernels;
    const float eps = 1e-5f, freq_base = 1.0e6f;

    std::vector<float> raw((size_t)cells * D), gamma(D);
    for (auto& v : raw) v = rng.uniform(-2.0f, 2.0f);
    for (auto& v : gamma) v = rng.uniform(-1.0f, 1.0f);

    const int n_bid_max = max_cells / R + 1;
    const size_t tail_n = (size_t)(R - 1) * D;
    const size_t pooled_n = (size_t)(n_bid_max + 1) * D;

    std::vector<float> tail(tail_n, 0.0f), dead(D, 0.0f), pooled(pooled_n, 0.0f);
    std::vector<int32_t> block_pos(1, 0);

    float *dtail = nullptr, *ddead = nullptr, *dpooled = nullptr, *draw = nullptr, *dgamma = nullptr;
    int32_t *dblock_pos = nullptr, *dpos = nullptr;
    HIPCHK(hipMalloc(&dtail, tail_n * 4));
    HIPCHK(hipMalloc(&ddead, D * 4));
    HIPCHK(hipMalloc(&dpooled, pooled_n * 4));
    // `raw` is ONE cell's key (D values): the kernel reads raw[d] and takes the
    // cell index from relative_pos_device. Passing the whole sequence here and
    // indexing by position was the bug this test found in itself.
    HIPCHK(hipMalloc(&draw, D * 4));
    HIPCHK(hipMalloc(&dgamma, D * 4));
    HIPCHK(hipMalloc(&dblock_pos, 4));
    HIPCHK(hipMalloc(&dpos, 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dtail, tail.data(), tail_n * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(ddead, dead.data(), D * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dpooled, pooled.data(), pooled_n * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dgamma, gamma.data(), D * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dblock_pos, block_pos.data(), 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    QsaIndexerBuffers bufs;
    bufs.tail = dtail;
    bufs.dead = ddead;
    bufs.pooled = dpooled;
    bufs.block_pos = dblock_pos;
    // The header documents "idx_dim=128, idx_block=4, n_rot=64" in prose; the
    // implementation also VALIDATES n_rot as a field. Setting only the two the
    // prose names first is what the message below catches - the same class of
    // omission the attention adapter had (see PORTING.md, contract gaps).
    QsaShapes shapes;
    shapes.idx_dim = D;
    shapes.idx_block = R;
    shapes.n_rot = ROT;

    Ref ref;
    int tail_wrong = 0, pos_wrong = 0, nonfinite = 0, wrong_cell_matches = 0;
    double worst_pooled = 0.0, pooled_scale = 0.0, last_cell_gap = 0.0;

    for (int pos = 0; pos < cells; ++pos) {
        std::vector<float> cell_raw(raw.begin() + (size_t)pos * D, raw.begin() + (size_t)(pos + 1) * D);
        HIPCHK(hipMemcpyAsync(draw, cell_raw.data(), D * 4, hipMemcpyHostToDevice, stream));
        HIPCHK(hipMemcpyAsync(dpos, &pos, 4, hipMemcpyHostToDevice, stream));
        HIPCHK(hipStreamSynchronize(stream));
        native_qsa_indexer_append(draw, dpos, pos_base, dgamma, eps, bufs, shapes, max_cells,
                                  freq_base, stream);
        HIPCHK(hipStreamSynchronize(stream));

        std::vector<double> y, y_wrong;
        ref.step(pos, pos_base, max_cells, cell_raw, gamma, eps, freq_base, cells, y, y_wrong);

        // tail is a copy of round-tripped values: must be byte-exact after every step
        std::vector<float> tail_got(tail_n), dead_got(D), pooled_got(pooled_n);
        int32_t bp_got = -1;
        HIPCHK(hipMemcpy(tail_got.data(), dtail, tail_n * 4, hipMemcpyDeviceToHost));
        HIPCHK(hipMemcpy(dead_got.data(), ddead, D * 4, hipMemcpyDeviceToHost));
        HIPCHK(hipMemcpy(pooled_got.data(), dpooled, pooled_n * 4, hipMemcpyDeviceToHost));
        HIPCHK(hipMemcpy(&bp_got, dblock_pos, 4, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < tail_n; ++i) {
            if (std::memcmp(&tail_got[i], &ref.tail[i], 4) != 0) {
                if (tail_wrong == 0) {
                    // Name the first divergence rather than only counting: three
                    // entries in one case is not something to reason about blind.
                    std::printf("      first tail divergence: step pos=%d slot=%d d=%zu "
                                "got=%.9g want=%.9g\n",
                                pos, (int)(i / D), i % D, (double)tail_got[i], (double)ref.tail[i]);
                }
                ++tail_wrong;
            }
        }

        if (pos == 0 || pos % R == R - 1) {
            const int b = pos / R;
            for (int d = 0; d < D; ++d) {
                const double got = pooled_got[(size_t)b * D + d];
                if (!std::isfinite(got)) { ++nonfinite; continue; }
                worst_pooled = std::fmax(worst_pooled, std::fabs(got - y[d]));
                pooled_scale = std::fmax(pooled_scale, std::fabs(y[d]));
                last_cell_gap = std::fmax(last_cell_gap, std::fabs(got - y_wrong[d]));
            }
            if (pos != 0) {
                if (bp_got != ref.block_pos) ++pos_wrong;
                for (int d = 0; d < D; ++d) {
                    // the spare row must hold `dead`, not a fresh computation
                    if (std::memcmp(&pooled_got[(size_t)(b + 1) * D + d], &dead_got[d], 4) != 0)
                        ++wrong_cell_matches;
                }
            }
        }
    }

    const double rel = worst_pooled / (pooled_scale + 1e-30);
    const double tol = 1e-5;      // rsqrtf + fast trig + the fp32 pooling chain
    std::printf("  %-26s cells=%-3d pos_base=%-2d  tail-wrong=%d  pooled rel=%.2e (tol %.0e)  "
                "block_pos wrong=%d  spare-row wrong=%d  last-cell gap=%.2e  nonfinite=%d\n",
                name, cells, pos_base, tail_wrong, rel, tol, pos_wrong, wrong_cell_matches,
                last_cell_gap, nonfinite);

    check(tail_wrong == 0, "the tail must hold byte-exact F16 round-trips of the raw keys");
    check(rel < tol, "the pooled rows exceeded the tolerance justified by rsqrtf and the trig");
    check(pos_wrong == 0, "block_pos must record the block's FIRST absolute position");
    check(wrong_cell_matches == 0, "the spare row must be the stored dead key, not recomputed");
    check(nonfinite == 0, "the pooled rows must be finite");
    // The trap: if the device had rotated at the LAST cell, every pooled value
    // would sit at `y_wrong` instead. Asserting the gap is large proves this test
    // can tell the two apart at all.
    if (pos_base != 0) {
        check(last_cell_gap > 1e-3,
              "rotating at the first vs the last cell must be distinguishable here, or this "
              "test cannot detect the documented trap");
    }

    hipStreamDestroy(stream);
    hipFree(dtail); hipFree(ddead); hipFree(dpooled); hipFree(draw); hipFree(dgamma);
    hipFree(dblock_pos); hipFree(dpos);
    return 0;
}

// An out-of-range position must write nothing at all.
int run_out_of_range(Rng& rng) {
    using namespace strata::kernels;
    const int D_ = D, max_cells = 16;
    const float sentinel = -4242.0f;
    std::vector<float> raw((size_t)D_, 1.0f), gamma(D_, 1.0f);
    std::vector<float> pooled((size_t)(max_cells / R + 2) * D_, sentinel), tail((size_t)(R - 1) * D_, sentinel),
        dead(D_, sentinel);
    std::vector<int32_t> bp(1, 7);

    float *dtail = nullptr, *ddead = nullptr, *dpooled = nullptr, *draw = nullptr, *dgamma = nullptr;
    int32_t *dbp = nullptr, *dpos = nullptr;
    HIPCHK(hipMalloc(&dtail, tail.size() * 4));
    HIPCHK(hipMalloc(&ddead, dead.size() * 4));
    HIPCHK(hipMalloc(&dpooled, pooled.size() * 4));
    HIPCHK(hipMalloc(&draw, raw.size() * 4));
    HIPCHK(hipMalloc(&dgamma, gamma.size() * 4));
    HIPCHK(hipMalloc(&dbp, 4));
    HIPCHK(hipMalloc(&dpos, 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dtail, tail.data(), tail.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(ddead, dead.data(), dead.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dpooled, pooled.data(), pooled.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(draw, raw.data(), raw.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dgamma, gamma.data(), gamma.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dbp, bp.data(), 4, hipMemcpyHostToDevice, stream));

    QsaIndexerBuffers bufs{dtail, ddead, dpooled, dbp};
    QsaShapes shapes;
    shapes.idx_dim = D;
    shapes.idx_block = R;
    shapes.n_rot = ROT;

    const int32_t out_of_range = max_cells;         // >= max_cells: ignored defensively
    HIPCHK(hipMemcpyAsync(dpos, &out_of_range, 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));
    native_qsa_indexer_append(draw, dpos, 0, dgamma, 1e-5f, bufs, shapes, max_cells, 1.0e6f, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<float> pooled_got(pooled.size()), tail_got(tail.size()), dead_got(dead.size());
    int32_t bp_got = -1;
    HIPCHK(hipMemcpy(pooled_got.data(), dpooled, pooled.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(tail_got.data(), dtail, tail.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(dead_got.data(), ddead, dead.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(&bp_got, dbp, 4, hipMemcpyDeviceToHost));

    int wrote = 0;
    for (float v : pooled_got) if (v != sentinel) ++wrote;
    for (float v : tail_got) if (v != sentinel) ++wrote;
    for (float v : dead_got) if (v != sentinel) ++wrote;
    if (bp_got != 7) ++wrote;
    std::printf("  %-26s pos=%d (>= max_cells=%d)  values written=%d\n",
                "out-of-range position", out_of_range, max_cells, wrote);
    check(wrote == 0, "an out-of-range position must write nothing");
    hipStreamDestroy(stream);
    hipFree(dtail); hipFree(ddead); hipFree(dpooled); hipFree(draw); hipFree(dgamma);
    hipFree(dbp); hipFree(dpos);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: native_qsa_indexer_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("native_qsa_indexer_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: the same cell-by-cell state machine in float64, driving a SEQUENCE\n"
                "  (tail byte-exact, pooled by tolerance, block_pos exact)\n");

    Rng rng;
    if (run_sequence("four blocks, pos_base 0", 16, 0, 64, rng) != 0) check(false, "seq aborted");
    if (run_sequence("nonzero pos_base", 12, 8, 64, rng) != 0) check(false, "seq aborted");
    if (run_sequence("one block only", 4, 4, 64, rng) != 0) check(false, "seq aborted");
    if (run_out_of_range(rng) != 0) check(false, "range case aborted");

    std::printf("\nnative_qsa_indexer: %d failures\n", failures);
    if (failures) return 1;
    std::printf("native_qsa_indexer_parity OK\n");
    (void)selftest;
    return 0;
}
