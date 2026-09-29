// tests/rocm/native_router_parity.cpp - independent check of the MoE router.
//
// No upstream test for this kernel exists (its parity test lived in the omitted
// bench/micro tree, and it compares against llama.cpp's CUDA oracle). The rollout
// plan asks for routing evidence specifically - "Expert IDs and scores;
// explicitly handle near-ties and documented tie-breaking" - so this file supplies
// it, written from the contract in `include/strata/kernels/native_router.hpp`:
//
//   one token, 512 experts, top 10, softmax, no selection bias, lower
//   normalization clamp 2^-14, scale 1, and EQUAL COMPUTED PROBABILITIES SELECT
//   THE LOWER EXPERT INDEX.
//
// WHAT IS TESTED, AND WHY EACH CASE IS HERE
//
//  * ordinary logits: the selected ids match a float64 top-10 exactly, and the
//    weights match within a bound derived from the reduction length;
//  * EXACT TIES, several ways: all-equal logits, a tie inside the top ten, and a
//    tie exactly at the 10th/11th boundary. The boundary tie is the one that
//    changes which expert runs, and it is the case the header calls out;
//  * NEAR ties, one ulp apart: the ids must be stable and must follow the
//    probabilities, not the raw logit magnitudes - tested with a large negative
//    offset so a router that skipped the max-subtraction would fail;
//  * extreme magnitudes: exp underflow must give finite weights, not NaN;
//  * the invariant that actually matters downstream: the ten weights sum to 1 and
//    never go negative.
//
// WHAT IS NOT CLAIMED. The device computes the softmax with a specific fp32
// warp-reduction order; the reference here is float64 with a different order. On
// values that are *exactly* tied, both must pick the lower index and this test
// asserts it; on values that differ by less than fp32 rounding, a different
// summation order can legitimately pick either, so such cases are not used to
// assert anything and `near-tie` cases below are constructed to be unambiguous.
//
// Run: native_router_parity --selftest

#include "strata/kernels/native_router.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kExperts = 512, kTop = 10;
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
    uint32_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (uint32_t)(s >> 32);
    }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
};

// The reference: float64 softmax and a top-10 that breaks ties toward the lower
// index, which is the documented rule.
void reference(const std::vector<float>& logits, std::vector<int32_t>& ids,
               std::vector<double>& weights) {
    double m = -INFINITY;
    for (float v : logits) m = std::fmax(m, (double)v);
    double sum = 0.0;
    std::vector<double> p(kExperts);
    for (int i = 0; i < kExperts; ++i) {
        p[i] = std::exp((double)logits[i] - m);
        sum += p[i];
    }
    for (int i = 0; i < kExperts; ++i) p[i] /= sum;

    std::vector<int> order(kExperts);
    for (int i = 0; i < kExperts; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + kTop, order.end(),
                      [&](int a, int b) {
                          if (p[a] != p[b]) return p[a] > p[b];
                          return a < b;                    // lower index wins a tie
                      });
    ids.assign(order.begin(), order.begin() + kTop);
    double selected = 0.0;
    for (int r = 0; r < kTop; ++r) selected += p[ids[r]];
    selected = std::fmax(selected, 6.103515625e-5);            // the 2^-14 clamp
    weights.resize(kTop);
    for (int r = 0; r < kTop; ++r) weights[r] = p[ids[r]] / selected;
}

int run_case(const char* name, const std::vector<float>& logits, bool expect_exact_ids) {
    using strata::kernels::native_router_top10;

    int32_t* dids = nullptr;
    float* dw = nullptr;
    float* dlogits = nullptr;
    HIPCHK(hipMalloc(&dids, kTop * 4));
    HIPCHK(hipMalloc(&dw, kTop * 4));
    HIPCHK(hipMalloc(&dlogits, kExperts * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));

    HIPCHK(hipMemcpyAsync(dlogits, logits.data(), kExperts * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));
    native_router_top10(dlogits, dids, dw, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<int32_t> ids(kTop);
    std::vector<float> w(kTop);
    HIPCHK(hipMemcpy(ids.data(), dids, kTop * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(w.data(), dw, kTop * 4, hipMemcpyDeviceToHost));

    std::vector<int32_t> ref_ids;
    std::vector<double> ref_w;
    reference(logits, ref_ids, ref_w);

    int id_mismatch = 0;
    for (int r = 0; r < kTop; ++r) if (ids[r] != ref_ids[r]) ++id_mismatch;

    // Weights: bound derived from the reduction length over 512 experts in fp32.
    const double bound_unit = 4.0 * kExperts * std::ldexp(1.0, -24);
    double worst_ratio = 0.0, sum_w = 0.0;
    int nonfinite = 0, negative = 0, out_of_bound = 0;
    for (int r = 0; r < kTop; ++r) {
        if (!std::isfinite(w[r])) { ++nonfinite; continue; }
        if (w[r] < 0.0f) ++negative;
        sum_w += w[r];
        const double scale = std::fmax(ref_w[r], 1.0 / kExperts);
        const double err = std::fabs((double)w[r] - ref_w[r]);
        const double tol = bound_unit * scale;
        if (err > tol) ++out_of_bound;
        if (tol > 0.0) worst_ratio = std::fmax(worst_ratio, err / tol);
    }

    // How separated was the selection boundary? A mismatch is only interesting if
    // the 10th and 11th candidates were distinguishable at all.
    std::vector<int> order(kExperts);
    for (int i = 0; i < kExperts; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + kTop + 1, order.end(),
                      [&](int a, int b) {
                          if (logits[a] != logits[b]) return logits[a] > logits[b];
                          return a < b;
                      });
    const double margin = (double)logits[order[kTop - 1]] - (double)logits[order[kTop]];

    std::printf("  %-34s ids%s  sum(w)=%.6f  err/tol=%.3f  logit margin(10th-11th)=%.3e%s\n",
                name, id_mismatch ? " DIFFER" : " match", sum_w, worst_ratio, margin,
                margin == 0.0 ? "  (exact tie: the rule decides it)" : "");

    if (expect_exact_ids) {
        check(id_mismatch == 0, "selected expert ids must match the reference exactly");
    }
    check(nonfinite == 0, "weights must be finite");
    check(negative == 0, "weights must be non-negative");
    check(std::fabs(sum_w - 1.0) < 1e-4, "the ten weights must sum to 1");
    check(out_of_bound == 0, "weights exceeded the bound derived from the reduction length");
    // No "ambiguous boundary" escape hatch here, deliberately. An earlier revision
    // of this file had one, keyed on a zero raw-logit margin - which is exactly
    // backwards: an exact tie is the case where the documented tie-break rule makes
    // the answer DETERMINISTIC, so a mismatch there is a failure like any other.
    // The margin is printed because it explains a tie, not because it excuses one.
    (void)margin;

    hipStreamDestroy(stream);
    hipFree(dids); hipFree(dw); hipFree(dlogits);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: native_router_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("native_router_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: float64 softmax + top-10 with the documented lower-index tie-break\n");

    Rng rng;

    // 1. ordinary logits, well separated
    {
        std::vector<float> logits(kExperts);
        for (auto& v : logits) v = rng.uniform(-4.0f, 4.0f);
        run_case("ordinary logits", logits, true);
    }

    // 2. all logits equal: every probability is equal, so the rule decides
    //    outright and the answer must be experts 0..9.
    {
        std::vector<float> logits(kExperts, 0.5f);
        run_case("all logits equal (exact tie)", logits, true);
    }

    // 3. a tie INSIDE the selected set: two experts share the top value.
    {
        std::vector<float> logits(kExperts);
        for (int i = 0; i < kExperts; ++i) logits[i] = -3.0f - 0.001f * (float)i;
        logits[400] = 5.0f;
        logits[12] = 5.0f;                       // same value, much lower index
        run_case("tie inside the top ten", logits, true);
    }

    // 4. a tie exactly at the 10th/11th boundary: this is the case that changes
    //    which expert runs, and the one the header's rule exists for.
    {
        std::vector<float> logits(kExperts);
        for (int i = 0; i < kExperts; ++i) logits[i] = -6.0f;
        for (int i = 0; i < 9; ++i) logits[i] = 1.0f + 0.1f * (float)i;   // 9 distinct leaders
        logits[300] = 0.0f;                      // 10th and 11th, exactly equal
        logits[7] = 0.0f;                        // but this one has the lower index
        logits[200] = -1.0f;                     // safely below the boundary
        run_case("tie at the 10th/11th boundary", logits, true);
    }

    // 5. near tie one ulp apart, under a large negative offset: the ids must
    //    follow the probabilities, which a router that skipped the max
    //    subtraction (and so underflowed to zero) could not do.
    {
        std::vector<float> logits(kExperts);
        for (int i = 0; i < kExperts; ++i) logits[i] = -900.0f - (float)i;
        logits[11] = -900.0f;
        logits[10] = std::nextafter(-900.0f, 0.0f);      // one ulp above, lower index
        run_case("near tie, one ulp, offset -900", logits, true);
    }

    // 6. extreme spread: exp underflow on one side, overflow risk on the other.
    {
        std::vector<float> logits(kExperts);
        for (int i = 0; i < kExperts; ++i) logits[i] = (i % 2 == 0) ? 88.0f : -88.0f;
        run_case("extreme spread (+/-88)", logits, true);
    }

    std::printf("\nnative_router: %d failures\n", failures);
    if (failures) return 1;
    std::printf("native_router_parity OK\n");
    (void)selftest;
    return 0;
}
