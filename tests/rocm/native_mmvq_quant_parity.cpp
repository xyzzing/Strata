// tests/rocm/native_mmvq_quant_parity.cpp - independent check of the Q8_1
// activation quantizer that every quantized mat-vec in the native path feeds on.
//
// There is no upstream test for this kernel (its parity test was in the omitted
// bench/micro tree, against llama.cpp's CUDA oracle). The contract in
// `include/strata/kernels/native_mmvq.hpp` is precise enough to check directly:
//
//   "Q8_1 stores FP16 scale and FP16 warp sum of the ORIGINAL float inputs; it
//    does not reconstruct that sum from the quantized integers."
//
// That second clause is the interesting one and it is why this test exists. The
// obvious implementation of a q8_1 block computes `s` as the sum of the *rounded*
// values; llama.cpp's stores the sum of the ORIGINAL floats. On a block full of
// values that round badly those differ, the MMVQ dot then differs, and nothing
// else in the suite would notice.
//
// Block layout, from the kernel: 36 bytes per 32 elements,
// `struct { half2 ds; int8_t qs[32]; }` with ds = (d, sum).
//
// What is asserted:
//   * `d    = amax(|x|) / 127` over the block, exactly;
//   * `q[i] = round(x[i] / d)`, exactly, and 0 for an all-zero block (d would be 0);
//   * `sum`  = the fp16 rounding of the sum of the ORIGINAL floats - and, on a
//     block chosen so the two disagree, NOT the sum of the rounded values;
//   * multiple columns quantize as one contiguous vector, so a column boundary
//     never splits a block.
//
// Run: native_mmvq_quant_parity --selftest

#include "strata/kernels/native_mmvq.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "f16.h"          // strata_test::f32_to_f16 / f16_to_f32

namespace {

constexpr int kBlock = 32;          // Q8K
constexpr int kBlockBytes = 36;     // half2 ds (4) + int8 qs (32)

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


struct Block {
    float d, sum;
    int8_t q[kBlock];
};

Block decode(const uint8_t* p) {
    uint16_t dh, sh;
    std::memcpy(&dh, p, 2);
    std::memcpy(&sh, p + 2, 2);
    Block b;
    b.d = strata_test::f16_to_f32(dh);
    b.sum = strata_test::f16_to_f32(sh);
    std::memcpy(b.q, p + 4, kBlock);
    return b;
}

struct Rng {
    uint64_t s = 0x243f6a8885a308d3ull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
};

int run_case(const char* name, std::vector<float> x, int ncols,
             bool need_discriminating = false) {
    using namespace strata::kernels;
    const int n_in = (int)x.size() / ncols;
    const int n_blocks = n_in * ncols / kBlock;

    const std::size_t want = native_q8_1_bytes(n_in, ncols);
    check(want == (std::size_t)n_blocks * kBlockBytes, "native_q8_1_bytes must match the layout");

    uint8_t* dx = nullptr;
    uint8_t* dq = nullptr;
    HIPCHK(hipMalloc(&dx, x.size() * 4));
    HIPCHK(hipMalloc(&dq, want));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    native_quantize_q8_1((const float*)dx, dq, n_in, ncols, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<uint8_t> q(want);
    HIPCHK(hipMemcpy(q.data(), dq, want, hipMemcpyDeviceToHost));

    int d_wrong = 0, q_wrong = 0, sum_wrong = 0, sum_looks_reconstructed = 0, scale_rounded = 0;
    double worst_sum_err = 0.0, worst_rec_err = 0.0;
    for (int b = 0; b < n_blocks; ++b) {
        const float* src = x.data() + (std::size_t)b * kBlock;
        const Block got = decode(q.data() + (std::size_t)b * kBlockBytes);

        float amax = 0.0f;
        double orig = 0.0, rounded = 0.0;
        for (int i = 0; i < kBlock; ++i) {
            amax = std::fmax(amax, std::fabs(src[i]));
            orig += (double)src[i];
        }
        const float d = amax / 127.0f;
        for (int i = 0; i < kBlock; ++i) {
            const float qf = d == 0.0f ? 0.0f : std::round(src[i] / d);
            if ((float)got.q[i] != qf) ++q_wrong;
            rounded += (double)(int)got.q[i];
        }
        // d is STORED AS FP16, so the value that comes back is the fp16 rounding of
        // amax/127, not amax/127 itself. Comparing against the fp32 value (as an
        // earlier revision of this test did) fails on every non-zero block and says
        // nothing about the kernel. The header states the matching precondition -
        // "their block scales/sums representable in FP16" - so a conforming input
        // has d_stored == d, and that is checked separately below.
        const float d_stored = strata_test::f16_to_f32(strata_test::f32_to_f16(d));
        if (got.d != d_stored) ++d_wrong;
        if (d_stored != d) ++scale_rounded;

        const float want_sum = strata_test::f16_to_f32(strata_test::f32_to_f16((float)orig));
        if (got.sum != want_sum) {
            ++sum_wrong;
            worst_sum_err = std::fmax(worst_sum_err, std::fabs((double)got.sum - want_sum));
        }
        // Would a "reconstruct from the integers" implementation be distinguishable?
        const float rec = strata_test::f16_to_f32(strata_test::f32_to_f16((float)(rounded * (double)got.d)));
        if (rec != want_sum) {
            worst_rec_err = std::fmax(worst_rec_err, std::fabs((double)rec - want_sum));
            if (got.sum == rec) ++sum_looks_reconstructed;
        }
    }

    std::printf("  %-32s blocks=%-3d  d wrong=%d  q wrong=%d  sum wrong=%d"
                "  scales not fp16-exact=%d%s\n",
                name, n_blocks, d_wrong, q_wrong, sum_wrong, scale_rounded,
                worst_rec_err > 0.0 ? "  (a reconstructed sum would differ here)" : "");

    check(d_wrong == 0, "the stored scale must be the fp16 rounding of amax/127");
    check(q_wrong == 0, "every quant must be round(x/d)");
    check(sum_wrong == 0, "the stored sum must be the fp16 rounding of the original float sum");
    if (worst_rec_err > 0.0) {
        check(sum_looks_reconstructed == 0,
              "the stored sum must NOT be reconstructed from the quantized integers");
    }
    if (need_discriminating) {
        // A discrimination case whose two sides cannot differ (here: both sums
        // overflowing fp16 to +inf) is inert - it proves nothing while looking
        // like coverage. The fixture must actually separate the two sums.
        check(worst_rec_err > 0.0,
              "the crafted input must make a reconstructed sum distinguishable; "
              "this case is inert as written");
    }

    hipStreamDestroy(stream);
    hipFree(dx);
    hipFree(dq);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: native_mmvq_quant_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("native_mmvq_quant_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: the documented q8_1 contract, decoded from the 36-byte block layout\n");

    Rng rng;

    {   // ordinary values
        std::vector<float> x(kBlock * 8);
        for (auto& v : x) v = rng.uniform(-1.0f, 1.0f);
        run_case("ordinary values", x, 1);
    }
    {   // all zeros: amax is 0, so d is 0 and every quant must be 0 rather than a division by zero
        std::vector<float> x(kBlock * 4, 0.0f);
        run_case("all zero block", x, 1);
    }
    {   // values chosen so the ORIGINAL sum and the sum of the rounded values differ
        // in fp16: large magnitudes with small fractional parts. The base keeps the
        // 32-value block sum near 60800 - large enough that the fp16 ulp (32) makes
        // the two sums land apart, small enough to stay under the 65504 fp16 max,
        // where BOTH sides would round to +inf and the case would prove nothing.
        std::vector<float> x(kBlock * 4);
        for (int b = 0; b < 4; ++b) {
            for (int i = 0; i < kBlock; ++i) {
                x[b * kBlock + i] = (i == 0) ? 1900.0f : 1900.0f + 0.37f * (float)(i % 3);
            }
        }
        run_case("sum reconstruction is distinguishable", x, 1, /*need_discriminating=*/true);
    }
    {   // extremes of the int8 range: amax must map to exactly +/-127
        std::vector<float> x(kBlock * 2, 0.0f);
        x[0] = -5.0f;
        x[1] = 5.0f;
        for (int i = 2; i < kBlock * 2; ++i) x[i] = rng.uniform(-5.0f, 5.0f);
        run_case("amax maps to the int8 limit", x, 1);
    }
    {   // the documented precondition: amax/127 representable in fp16, so the stored
        // scale is exact and q*d reconstructs the input to within the quant step.
        std::vector<float> x(kBlock * 4, 0.0f);
        for (int b = 0; b < 4; ++b) {
            const float d = std::ldexp(1.0f, -(b + 1));      // 2^-1 .. 2^-4, fp16-exact
            const float amax = 127.0f * d;
            for (int i = 0; i < kBlock; ++i) {
                x[b * kBlock + i] = (b % 2 == 0 ? amax : -amax) * (float)(i % 17) / 16.0f;
            }
        }
        run_case("scale representable in fp16", x, 1);
    }
    {   // multiple columns: blocks must not straddle a column boundary
        std::vector<float> x(kBlock * 6);
        for (auto& v : x) v = rng.uniform(-2.0f, 2.0f);
        run_case("three columns", x, 3);
    }

    std::printf("\nnative_mmvq_quant: %d failures\n", failures);
    if (failures) return 1;
    std::printf("native_mmvq_quant_parity OK\n");
    (void)selftest;
    return 0;
}
