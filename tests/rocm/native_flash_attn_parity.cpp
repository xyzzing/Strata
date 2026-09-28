// tests/rocm/native_flash_attn_parity.cpp - independent check of the short-context
// attention adapter.
//
// No upstream test exists (its parity test lived in the omitted bench/micro tree,
// against llama.cpp's CUDA oracle). The contract in
// `include/strata/kernels/native_flash_attn.hpp` is exact enough to build one:
//
//   geometry Q24x256, KV2x256, one query/sequence, scale=1/16, no ALiBi,
//   softcap or sink; q/output contiguous [24,256] F32; k/v [capacity,2,256] F16
//   with only step[kStepWidth] rows initialized; an optional 256-entry F16
//   ADDITIVE mask broadcast over heads; padding synthesized as zero K/V and -inf
//   mask WITHOUT reading it; and an invalid step writes status=UnsupportedStep
//   plus NaN output without reading q/k/v/mask.
//
// Two clauses in that contract are the reason this file is worth writing:
//
//   * "padding ... may contain arbitrary bytes" - so the padding-robustness case
//     below fills every unused row of k/v AND the unused mask entries with NaN bit
//     patterns. A kernel that reads them fails loudly; one that synthesizes them
//     is unaffected. Nothing else in the suite would notice the difference.
//   * the query-to-KV head map is `head / 12`, NOT `head % 2`. The heads are made
//     distinguishable here so those two rules cannot give the same answer.
//
// The reference is float64 over the f16-decoded k/v and the f32 q, mirroring the
// kernel's order (scale q first - it is a power of two, so that is exact - then
// accumulate, then add the additive mask).
//
// Run: native_flash_attn_parity --selftest

#include "strata/kernels/native_flash_attn.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "f16.h"          // strata_test::f32_to_f16 / f16_to_f32

namespace {

constexpr int kHeads = 24, kKvHeads = 2, kDim = 256, kCapacity = 256;
constexpr float kScale = 0.0625f;          // 1/16, as the kernel launches with

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
    uint64_t s = 0xa4093822299f31d0ull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
};

struct Case {
    const char* name;
    int width;              // live keys: step[kStepWidth] == step[kStepNKv]
    bool with_mask;
    bool poison_padding;    // NaN in every unused k/v row and mask entry
    bool valid_step;        // false => expect UnsupportedStep + NaN output
};

int run_case(const Case& c, Rng& rng) {
    using namespace strata::kernels;

    std::vector<float> q((size_t)kHeads * kDim);
    for (auto& v : q) v = rng.uniform(-2.0f, 2.0f);

    // k/v: [capacity, 2, 256] F16. Values chosen so the two KV heads are easy to
    // tell apart, and so a head%2 mapping cannot coincide with head/12.
    std::vector<uint16_t> k((size_t)kCapacity * kKvHeads * kDim);
    std::vector<uint16_t> v((size_t)kCapacity * kKvHeads * kDim);
    for (int cell = 0; cell < kCapacity; ++cell) {
        for (int kv = 0; kv < kKvHeads; ++kv) {
            for (int d = 0; d < kDim; ++d) {
                const size_t i = ((size_t)cell * kKvHeads + kv) * kDim + d;
                if (cell < c.width) {
                    k[i] = strata_test::f32_to_f16(rng.uniform(-1.0f, 1.0f) + (float)kv * 0.5f);
                    v[i] = strata_test::f32_to_f16(rng.uniform(-1.0f, 1.0f) + (float)kv * 0.25f);
                } else if (c.poison_padding) {
                    k[i] = 0x7e00u;      // NaN
                    v[i] = 0x7e00u;      // NaN
                } else {
                    k[i] = strata_test::f32_to_f16(0.0f);
                    v[i] = strata_test::f32_to_f16(0.0f);
                }
            }
        }
    }

    std::vector<uint16_t> mask(kDim);
    for (int j = 0; j < kDim; ++j) {
        if (j < c.width) mask[j] = strata_test::f32_to_f16(rng.uniform(-0.5f, 0.5f));
        else mask[j] = c.poison_padding ? 0x7e00u : strata_test::f32_to_f16(-1000.0f);
    }
    // Guarantee the documented "at least one valid key unmasked" precondition by
    // keeping a large negative additive value from ever applying to cell 0.
    if (c.with_mask) mask[0] = strata_test::f32_to_f16(0.0f);

    std::vector<int32_t> step(4);
    step[kStepPos] = c.width - 1;
    step[kStepNKv] = c.width;
    step[kStepNBid] = c.width / 4;
    step[kStepWidth] = c.width;
    if (!c.valid_step) step[kStepNBid] = c.width / 4 + 1;      // break the contract

    const size_t kv_elems = (size_t)kCapacity * kKvHeads * kDim;
    const size_t q_elems = (size_t)kHeads * kDim;

    float* dq = nullptr;
    uint16_t* dk = nullptr;
    uint16_t* dv = nullptr;
    float* dout = nullptr;
    int32_t* dstep = nullptr;
    int32_t* dstatus = nullptr;
    uint16_t* dmask = nullptr;
    HIPCHK(hipMalloc(&dq, q_elems * 4));
    HIPCHK(hipMalloc(&dk, kv_elems * 2));
    HIPCHK(hipMalloc(&dv, kv_elems * 2));
    HIPCHK(hipMalloc(&dout, q_elems * 4));
    HIPCHK(hipMalloc(&dstep, 4 * 4));
    HIPCHK(hipMalloc(&dstatus, 4));
    HIPCHK(hipMalloc(&dmask, kDim * 2));

    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dq, q.data(), q_elems * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dk, k.data(), kv_elems * 2, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dv, v.data(), kv_elems * 2, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dstep, step.data(), 16, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dmask, mask.data(), kDim * 2, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    const int32_t sentinel = -12345;
    HIPCHK(hipMemcpy(dstatus, &sentinel, 4, hipMemcpyHostToDevice));

    // The header documents Q24x256/KV2x256; the implementation additionally
    // validates idx_block == 4 and idx_top_k >= 256, so those are set here too.
    // A caller working from the header alone would get an invalid_argument - noted
    // in PORTING.md, because the contract in the header is incomplete.
    QsaShapes shapes;
    shapes.n_head = kHeads;
    shapes.n_head_kv = kKvHeads;
    shapes.head_dim = kDim;
    shapes.idx_block = 4;
    shapes.idx_top_k = 2048;
    native_flash_attn_short_step(dq, dk, dv, dstep, kCapacity, kCapacity, shapes, dout, dstatus,
                                 c.with_mask ? dmask : nullptr, stream);
    HIPCHK(hipStreamSynchronize(stream));

    int32_t status = -1;
    std::vector<float> out(q_elems);
    HIPCHK(hipMemcpy(&status, dstatus, 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(out.data(), dout, q_elems * 4, hipMemcpyDeviceToHost));

    if (!c.valid_step) {
        int not_nan = 0;
        for (float x : out) if (!std::isnan(x)) ++not_nan;
        std::printf("  %-30s invalid step -> status=%d, %d of %zu outputs not NaN\n",
                    c.name, status, not_nan, out.size());
        check(status == kNativeFlashAttnUnsupportedStep, "an invalid step must report UnsupportedStep");
        check(not_nan == 0, "an invalid step must write NaN output");
        hipStreamDestroy(stream);
        hipFree(dq); hipFree(dk); hipFree(dv); hipFree(dout);
        hipFree(dstep); hipFree(dstatus); hipFree(dmask);
        return 0;
    }

    // Reference: float64, mirroring the kernel's order.
    std::vector<double> ref(q_elems, 0.0);
    double vmax = 0.0;
    for (int head = 0; head < kHeads; ++head) {
        const int kv = head / 12;                       // the documented map
        std::vector<double> score(c.width);
        double m = -INFINITY;
        for (int cell = 0; cell < c.width; ++cell) {
            double dot = 0.0;
            for (int d = 0; d < kDim; ++d) {
                const size_t ki = ((size_t)cell * kKvHeads + kv) * kDim + d;
                const double kd = (double)strata_test::f16_to_f32(k[ki]);
                const double qs = (double)(q[(size_t)head * kDim + d] * kScale);
                dot += qs * kd;
            }
            if (c.with_mask) dot += (double)strata_test::f16_to_f32(mask[cell]);
            score[cell] = dot;
            m = std::fmax(m, dot);
        }
        double sum = 0.0;
        std::vector<double> p(c.width);
        for (int cell = 0; cell < c.width; ++cell) {
            p[cell] = std::exp(score[cell] - m);
            sum += p[cell];
        }
        for (int d = 0; d < kDim; ++d) {
            double acc = 0.0;
            for (int cell = 0; cell < c.width; ++cell) {
                const size_t vi = ((size_t)cell * kKvHeads + kv) * kDim + d;
                acc += p[cell] * (double)strata_test::f16_to_f32(v[vi]);
            }
            ref[(size_t)head * kDim + d] = acc / sum;
        }
    }
    for (size_t i = 0; i < v.size(); ++i) {
        const float x = strata_test::f16_to_f32(v[i]);
        if (std::isfinite(x)) vmax = std::fmax(vmax, std::fabs((double)x));
    }

    // Bound derived from the reduction: a softmax over `width` cells accumulated in
    // fp32, then a convex combination of f16 values bounded by vmax.
    const double bound = 8.0 * (double)c.width * std::ldexp(1.0, -24) * vmax + 1e-7;
    double worst = 0.0;
    int nonfinite = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        if (!std::isfinite(out[i])) { ++nonfinite; continue; }
        worst = std::fmax(worst, std::fabs((double)out[i] - ref[i]));
    }
    std::printf("  %-30s width=%-4d mask=%d poisoned=%d  nonfinite=%d  worst|err|=%.3e  "
                "bound=%.3e  err/bound=%.3f\n",
                c.name, c.width, (int)c.with_mask, (int)c.poison_padding, nonfinite,
                worst, bound, worst / bound);

    check(nonfinite == 0, "the output must be finite for a valid step");
    check(worst <= bound, "deviation from the float64 reference exceeded the derived bound");

    hipStreamDestroy(stream);
    hipFree(dq); hipFree(dk); hipFree(dv); hipFree(dout);
    hipFree(dstep); hipFree(dstatus); hipFree(dmask);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: native_flash_attn_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("native_flash_attn_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: float64 over the f16-decoded k/v; q->kv map head/12; scale 1/16\n");

    Rng rng;
    const Case cases[] = {
        {"ordinary, no mask",              8,   false, false, true},
        {"additive mask",                  8,   true,  false, true},
        {"single key",                     1,   false, false, true},
        {"full context (256)",             256, true,  false, true},
        {"padding poisoned with NaN",      12,  true,  true,  true},
        {"padding poisoned, no mask",      5,   false, true,  true},
        {"invalid step (guard)",           8,   false, false, false},
    };
    for (const Case& c : cases) {
        if (run_case(c, rng) != 0) { check(false, "case aborted early", c.name); }
    }

    std::printf("\nnative_flash_attn: %d failures\n", failures);
    if (failures) return 1;
    std::printf("native_flash_attn_parity OK\n");
    (void)selftest;
    return 0;
}
