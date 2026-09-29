// tests/rocm/s2_expert_grouped_parity.cpp - the plumbing of the GPU expert tier.
//
// `s2_expert_grouped.cu` was the last kernel file with no test at all, and it is
// the one the feasibility gate was being held on. What it does is COMPOSE
// arithmetic that is already verified elsewhere - S2 dequantization and GEMV
// (`dequant_s2_parity`, `s2_gemv_parity`), q8_0 activation quantization
// (`quantize_act_parity`), SiLU - into one grouped evaluation, so that a layer's
// expert hits become four launches instead of five per expert.
//
// WHAT THIS FILE TESTS, AND WHAT IT LEANS ON. It tests the part that is unique to
// this file and that the arithmetic tests cannot reach: the grouping, the routing
// and the buffer sizing. It does NOT re-derive the expert arithmetic; it asserts
// that `moe_hit_grouped_s2` with n_hits = 1 agrees BIT-EXACTLY with the same hit
// inside a group, which is the strongest available statement without reimplementing
// the pack format here. The decomposition is stated rather than implied.
//
// THE CLAIM WORTH TESTING is the one the header shouts about:
//
//   "`dst_index`  where each hit's answer goes ... **THIS IS NOT A CONVENIENCE -
//    IT IS WHAT LETS THE TWO HALVES MEET.**"
//
// A layer routes ten experts; the CPU produces the misses at their ROUTED indices
// and the GPU produces the hits "in whatever order the cache happens to hold
// them". A kernel that wrote hits sequentially would land them on the wrong rows
// and every downstream number would still look like a plausible activation. So the
// fixtures use a PERMUTED dst_index, and the test checks that each hit lands on
// its own routed row and that untouched rows stay untouched.
//
// The blobs are filled with bytes below 0x10 so that no fp16 scale field has an
// infinity/NaN exponent: the values are meaningless as weights but finite and
// distinct per slot, which is exactly what a routing test needs.
//
// Run: s2_expert_grouped_parity --selftest

#include "strata/kernels/s2_expert_grouped.hpp"

#include "f16.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int H = 2560;          // n_embd, the kernel's fixed geometry
constexpr int FF = 640;          // n_ff
constexpr int Q8B = 34;          // sizeof(block_q8_0)
constexpr int Q8K = 32;

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

// The same, for the fixture's member functions: those return bool or a vector, so
// they cannot `return 1`. A HIP failure there is a hard test failure either way.
#define HIPCK_MEMBER(expr)                                                             \
    do {                                                                               \
        hipError_t _e = (expr);                                                        \
        if (_e != hipSuccess) {                                                        \
            std::fprintf(stderr, "hip error %s: %s\n", #expr, hipGetErrorString(_e));   \
            std::abort();                                                              \
        }                                                                              \
    } while (0)

struct Rng {
    uint64_t s = 0xbb67ae8584caa73bull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    // Below 0x10, so no fp16 exponent field reaches infinity or NaN.
    uint8_t byte() { return (uint8_t)(next() & 0x0fu); }
};

struct Fixture {
    std::vector<uint8_t> blob;
    std::vector<uint8_t> xq;
    uint8_t *dblob = nullptr;
    float *dout = nullptr;
    uint8_t *dxq = nullptr, *dscratch = nullptr;
    int32_t *dslot = nullptr, *ddst = nullptr;
    hipStream_t stream = nullptr;
    int n_slots = 8;
    size_t blob_bytes = 0, scratch_bytes = 0;
    static constexpr size_t kCanary = 64;

    bool init(Rng& rng, int64_t n_hits) {
        blob_bytes = 1u << 20;
        scratch_bytes = (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(n_hits, H, FF);
        blob.assign(blob_bytes * (size_t)n_slots, 0);
        // Distinct per slot: slot s's bytes are s-derived, so a hit's answer
        // identifies which slot it came from.
        for (int s = 0; s < n_slots; ++s) {
            uint8_t* p = blob.data() + (size_t)s * blob_bytes;
            for (size_t i = 0; i < blob_bytes; ++i) p[i] = (uint8_t)((rng.byte() + s) & 0x0fu);
        }
        xq.assign((size_t)(H / Q8K) * Q8B, 0);
        for (auto& b : xq) b = rng.byte();

        HIPCK_MEMBER(hipMalloc(&dblob, blob.size()));
        HIPCK_MEMBER(hipMalloc(&dxq, xq.size()));
        HIPCK_MEMBER(hipMalloc(&dscratch, scratch_bytes + kCanary));
        HIPCK_MEMBER(hipMalloc(&dout, (size_t)16 * H * 4));
        HIPCK_MEMBER(hipMalloc(&dslot, 64 * 4));
        HIPCK_MEMBER(hipMalloc(&ddst, 64 * 4));
        HIPCK_MEMBER(hipStreamCreate(&stream));
        HIPCK_MEMBER(hipMemcpyAsync(dblob, blob.data(), blob.size(), hipMemcpyHostToDevice, stream));
        HIPCK_MEMBER(hipMemcpyAsync(dxq, xq.data(), xq.size(), hipMemcpyHostToDevice, stream));
        // canaries past the end of the scratch region
        std::vector<uint8_t> canary(kCanary, 0xa5);
        HIPCK_MEMBER(hipMemcpyAsync((uint8_t*)dscratch + scratch_bytes, canary.data(), kCanary,
                              hipMemcpyHostToDevice, stream));
        HIPCK_MEMBER(hipStreamSynchronize(stream));
        return true;
    }

    /// Runs the grouped kernel for the given hits and returns every row of `out`.
    std::vector<float> run(const std::vector<int32_t>& slots, const std::vector<int32_t>& dsts,
                           const float* x_scales = nullptr) {
        const int64_t n = (int64_t)slots.size();
        HIPCK_MEMBER(hipMemcpyAsync(dslot, slots.data(), slots.size() * 4, hipMemcpyHostToDevice, stream));
        HIPCK_MEMBER(hipMemcpyAsync(ddst, dsts.data(), dsts.size() * 4, hipMemcpyHostToDevice, stream));
        std::vector<float> sentinel((size_t)16 * H, -777.0f);
        HIPCK_MEMBER(hipMemcpyAsync(dout, sentinel.data(), sentinel.size() * 4, hipMemcpyHostToDevice, stream));
        HIPCK_MEMBER(hipStreamSynchronize(stream));
        strata::kernels::moe_hit_grouped_s2(dblob, dslot, ddst, n, (int64_t)blob_bytes, dxq, dscratch,
                                            dout, stream, x_scales);
        HIPCK_MEMBER(hipStreamSynchronize(stream));
        std::vector<float> got(sentinel.size());
        HIPCK_MEMBER(hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost));
        return got;
    }

    void teardown() {
        if (stream) hipStreamDestroy(stream);
        hipFree(dblob); hipFree(dxq); hipFree(dscratch); hipFree(dout); hipFree(dslot); hipFree(ddst);
    }
};

// 1 + 2: each hit lands on its own routed row, and matches the single-hit call.
int run_routing(Rng& rng) {
    const int64_t n_hits = 4;
    Fixture f;
    if (!f.init(rng, n_hits)) return 1;

    // A deliberately unsorted, non-contiguous mapping: hits are produced in cache
    // order and must land on routed rows.
    const std::vector<int32_t> slots = {5, 1, 6, 2};
    const std::vector<int32_t> dsts = {9, 3, 0, 11};

    const std::vector<float> grouped = f.run(slots, dsts);

    int row_wrong = 0, untouched_wrong = 0, nonfinite = 0;
    double worst_gap = 0.0;
    for (size_t h = 0; h < slots.size(); ++h) {
        // single-hit call for the same expert, into a fresh output
        const std::vector<float> single = f.run({slots[h]}, {dsts[h]});
        const float* a = &grouped[(size_t)dsts[h] * H];
        const float* b = &single[(size_t)dsts[h] * H];
        for (int d = 0; d < H; ++d) {
            if (!std::isfinite(a[d])) ++nonfinite;
            if (std::memcmp(&a[d], &b[d], 4) != 0) {
                ++row_wrong;
                worst_gap = std::fmax(worst_gap, std::fabs((double)a[d] - (double)b[d]));
            }
        }
    }
    // rows nobody was routed to must still hold the sentinel
    for (int row = 0; row < 16; ++row) {
        bool referenced = false;
        for (int32_t d : dsts) if (d == row) referenced = true;
        if (referenced) continue;
        for (int d = 0; d < H; ++d)
            if (grouped[(size_t)row * H + d] != -777.0f) ++untouched_wrong;
    }

    std::printf("  %-34s grouped vs per-hit single: differing=%d (worst %.2e)  "
                "untouched rows altered=%d  nonfinite=%d\n",
                "routing via dst_index", row_wrong, worst_gap, untouched_wrong, nonfinite);
    check(row_wrong == 0,
          "a hit inside a group must equal the same hit evaluated alone, bit for bit");
    check(untouched_wrong == 0, "rows no hit was routed to must be left alone");
    check(nonfinite == 0, "the fixture must produce finite values, or the comparison proves nothing");

    // The permutation must actually be doing something: if the kernel wrote hits
    // sequentially, row dsts[h] would hold hit h's neighbour instead. Compare
    // against the sequential arrangement to show the two are distinguishable.
    int differs_from_sequential = 0;
    for (size_t h = 0; h < slots.size(); ++h) {
        if (dsts[h] == (int32_t)h) continue;
        if (std::memcmp(&grouped[(size_t)dsts[h] * H], &grouped[(size_t)h * H], (size_t)H * 4) != 0)
            ++differs_from_sequential;
    }
    std::printf("  %-34s %zu of %zu routed rows differ from the sequential arrangement\n",
                "routing is observable", (size_t)differs_from_sequential, slots.size());
    check(differs_from_sequential > 0,
          "the permuted routing must be distinguishable from sequential writes, or this test "
          "cannot detect the failure the header warns about");

    f.teardown();
    return 0;
}

// 3: the scratch region is exactly what the sizing helper promises.
int run_scratch(Rng& rng) {
    const int64_t n_hits = 6;
    Fixture f;
    if (!f.init(rng, n_hits)) return 1;

    std::vector<int32_t> slots(n_hits), dsts(n_hits);
    for (int i = 0; i < n_hits; ++i) { slots[i] = i % f.n_slots; dsts[i] = i; }
    f.run(slots, dsts);

    std::vector<uint8_t> canary(Fixture::kCanary, 0);
    HIPCK_MEMBER(hipMemcpy(canary.data(), (uint8_t*)f.dscratch + f.scratch_bytes, Fixture::kCanary,
                           hipMemcpyDeviceToHost));
    int clobbered = 0;
    for (uint8_t b : canary) if (b != 0xa5) ++clobbered;
    std::printf("  %-34s n_hits=%lld  scratch=%zu bytes  canary bytes clobbered=%d\n",
                "scratch sizing", (long long)n_hits, f.scratch_bytes, clobbered);
    check(clobbered == 0,
          "moe_hit_grouped_scratch_bytes must be sufficient: bytes past it were overwritten");
    check(f.scratch_bytes > 0, "the sizing helper must return a positive size");

    f.teardown();
    return 0;
}

// 4: x_scales is optional, and null keeps the previous behaviour exactly.
int run_x_scales(Rng& rng) {
    const int64_t n_hits = 3;
    Fixture f;
    if (!f.init(rng, n_hits)) return 1;

    const std::vector<int32_t> slots = {2, 4, 6};
    const std::vector<int32_t> dsts = {1, 2, 3};

    const std::vector<float> plain = f.run(slots, dsts);
    const std::vector<float> plain_again = f.run(slots, dsts);

    std::vector<float> scales((size_t)(H / Q8K), 0.5f);
    float* dscales = nullptr;
    HIPCK_MEMBER(hipMalloc(&dscales, scales.size() * 4));
    HIPCK_MEMBER(hipMemcpyAsync(dscales, scales.data(), scales.size() * 4, hipMemcpyHostToDevice, f.stream));
    HIPCK_MEMBER(hipStreamSynchronize(f.stream));
    const std::vector<float> scaled = f.run(slots, dsts, dscales);

    int rerun_diff = 0, scaled_diff = 0;
    for (size_t i = 0; i < plain.size(); ++i) {
        if (std::memcmp(&plain[i], &plain_again[i], 4) != 0) ++rerun_diff;
        if (std::memcmp(&plain[i], &scaled[i], 4) != 0) ++scaled_diff;
    }
    std::printf("  %-34s null-vs-null differences=%d  null-vs-scales differences=%d\n",
                "x_scales is optional", rerun_diff, scaled_diff);
    check(rerun_diff == 0, "two identical runs must agree bit for bit");
    check(scaled_diff > 0,
          "supplying fp32 activation scales must change the result, or the parameter is inert");

    hipFree(dscales);
    f.teardown();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: s2_expert_grouped_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("s2_expert_grouped_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  plumbing only: grouping, dst_index routing, slot indexing and scratch sizing.\n"
                "  The expert ARITHMETIC is covered by dequant_s2_parity and s2_gemv_parity.\n");

    Rng rng;
    if (run_routing(rng) != 0) check(false, "routing case aborted");
    if (run_scratch(rng) != 0) check(false, "scratch case aborted");
    if (run_x_scales(rng) != 0) check(false, "x_scales case aborted");

    std::printf("\ns2_expert_grouped: %d failures\n", failures);
    if (failures) return 1;
    std::printf("s2_expert_grouped_parity OK\n");
    (void)selftest;
    return 0;
}
