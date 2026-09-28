// tests/rocm/fused_gdn_conv_l2_parity.cpp - the fused conv+SiLU+L2 against its own parts.
//
// `fused_gdn.hpp` declares three fused GDN entry points; `fused_gdn_step_norm` has its
// test (`fused_gdn_parity`). This is the second: `fused_gdn_conv_l2`, which the header
// states "replaces conv_silu + two l2_norm launches". Both of those already have
// independent passing tests (`native_gdn_preprocess_parity`), so the reference here is
// their composition - the exact decomposition the header claims is equivalent:
//
//   native_gdn_conv_silu(history, qkv, conv_w, raw, silu, channels, 4)
//   native_gdn_l2_norm(h, 2*qk_heads, 128, eps)     // q heads first, then k heads
//
// What can differ is only the REDUCTION ORDER inside the 128-wide norm and the
// fast-math exponential behind SiLU, so the bound is derived from the reduction length
// and applied BOTH as an aggregate and per element (an aggregate over thousands of
// channels can absorb a handful of wrong ones; a max cannot).
//
// The carried history is pure data movement in both paths, so it is compared for
// bit-exact equality - a recomputed or reordered history update is a real defect.
//
// Run: fused_gdn_conv_l2_parity --selftest

#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int HEAD = 128;   // one head is 128 channels
constexpr int D_CONV = 4;
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
            std::fprintf(stderr, "hip error %s: %s\n", #expr, hipGetErrorString(_e));  \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

struct Rng {
    uint64_t s = 0x6364f7e2e193b0e9ull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
};

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += std::fabs((double)a[i] - (double)b[i]);
        den += std::fabs((double)b[i]);
    }
    return num / (den + 1e-30);
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::fmax(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

int run_case(const char* name, int k_heads, int h_v, float input_scale, Rng& rng) {
    using namespace strata::kernels;
    const float eps = 1e-6f;
    const int channels = (2 * k_heads + h_v) * HEAD;
    // The kernel's `qk_heads` parameter counts LEADING HEADS NORMALIZED, and the
    // layer passes 2 * ssm_k_heads: the q heads then the k heads, in channel
    // order (src/core/layer.cpp). The value head tail is left unnormalized.
    const int norm_heads = 2 * k_heads;

    std::vector<float> qkv((size_t)channels), conv_w((size_t)channels * D_CONV),
        hist0((size_t)channels * (D_CONV - 1));
    for (auto& x : qkv) x = rng.uniform(-input_scale, input_scale);
    for (auto& x : conv_w) x = rng.uniform(-0.5f, 0.5f);
    for (auto& x : hist0) x = rng.uniform(-input_scale, input_scale);

    std::vector<float> hist_fused = hist0, hist_ref = hist0;

    float *dhist_fused = nullptr, *dhist_ref = nullptr, *dqkv = nullptr, *dconv_w = nullptr;
    float *dh_fused = nullptr, *dh_ref = nullptr, *draw = nullptr, *dsilu = nullptr;
    HIPCHK(hipMalloc(&dhist_fused, hist_fused.size() * 4));
    HIPCHK(hipMalloc(&dhist_ref, hist_ref.size() * 4));
    HIPCHK(hipMalloc(&dqkv, qkv.size() * 4));
    HIPCHK(hipMalloc(&dconv_w, conv_w.size() * 4));
    HIPCHK(hipMalloc(&dh_fused, qkv.size() * 4));
    HIPCHK(hipMalloc(&dh_ref, qkv.size() * 4));
    HIPCHK(hipMalloc(&draw, qkv.size() * 4));
    HIPCHK(hipMalloc(&dsilu, qkv.size() * 4));

    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dhist_fused, hist_fused.data(), hist_fused.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dhist_ref, hist_ref.data(), hist_ref.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dqkv, qkv.data(), qkv.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dconv_w, conv_w.data(), conv_w.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    // reference: the two already-verified kernels, composed exactly as the fused
    // header claims ("replaces conv_silu + two l2_norm launches"). The SiLU output
    // is copied to dh_ref and the first `norm_heads` rows are normalized in place:
    // the q heads (rows [0, k_heads)), then the k heads ([k_heads, 2*k_heads)).
    native_gdn_conv_silu(dhist_ref, dqkv, dconv_w, draw, dsilu, channels, D_CONV, stream);
    HIPCHK(hipMemcpyAsync(dh_ref, dsilu, qkv.size() * 4, hipMemcpyDeviceToDevice, stream));
    native_gdn_l2_norm(dh_ref, (int64_t)norm_heads, HEAD, eps, stream);

    // fused: one kernel, called exactly as the layer calls it
    fused_gdn_conv_l2(dhist_fused, dqkv, dconv_w, dh_fused, channels, norm_heads, eps, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<float> h_fused(qkv.size()), h_ref(qkv.size());
    std::vector<float> hf_out(hist_fused.size()), hr_out(hist_ref.size());
    HIPCHK(hipMemcpy(h_fused.data(), dh_fused, h_fused.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(h_ref.data(), dh_ref, h_ref.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(hf_out.data(), dhist_fused, hf_out.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(hr_out.data(), dhist_ref, hr_out.size() * 4, hipMemcpyDeviceToHost));

    int nonfinite = 0;
    for (float x : h_fused) if (!std::isfinite(x)) ++nonfinite;

    int hist_wrong = 0;
    for (size_t i = 0; i < hf_out.size(); ++i)
        if (hf_out[i] != hr_out[i]) ++hist_wrong;

    const double err = rel_l1(h_fused, h_ref);
    const double emax = max_abs_diff(h_fused, h_ref);
    double scale = 0.0;
    for (float v : h_ref) scale = std::fmax(scale, std::fabs((double)v));
    // Bound: a 4-tap conv (identical arithmetic in both paths) followed by a
    // 128-wide reduction the fused kernel may order differently, plus the
    // fast-math exponential a few ulp apart between the two implementations.
    const double tol = 8.0 * (double)HEAD * std::ldexp(1.0, -24);

    std::printf("  %-34s k_heads=%-2d h_v=%-2d channels=%-5d  rel=%.3e max=%.1e  "
                "tol=%.1e  hist_wrong=%d  nonfinite=%d\n",
                name, k_heads, h_v, channels, err, emax, tol, hist_wrong, nonfinite);

    check(nonfinite == 0, "the fused kernel must produce finite output");
    check(hist_wrong == 0, "the carried conv history must update bit-identically");
    check(err < tol, "the fused output exceeded the bound derived from the reduction length");
    check(emax < tol * std::fmax(1.0, scale),
          "the fused output has an element past the per-element bound");

    hipStreamDestroy(stream);
    hipFree(dhist_fused); hipFree(dhist_ref); hipFree(dqkv); hipFree(dconv_w);
    hipFree(dh_fused); hipFree(dh_ref); hipFree(draw); hipFree(dsilu);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: fused_gdn_conv_l2_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("fused_gdn_conv_l2_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: native_gdn_conv_silu + native_gdn_l2_norm, both already\n"
                "  independently verified, composed exactly as the fused header claims\n");

    Rng rng;
    run_case("small: 2+2 q/k heads, 1 v head", 2, 1, 1.0f, rng);
    // the real artifact's channel count is (2*16+48)*128 = 10240; run the same
    // q/k geometry with a subsampled value tail to keep the case short
    run_case("artifact q/k geometry", 16, 4, 1.0f, rng);
    // a zero column: the norm's eps floor must return zeros, not NaN
    run_case("quiet input, eps floor", 2, 1, 0.0f, rng);
    // larger inputs stress the SiLU argument range
    run_case("wide input range", 4, 2, 8.0f, rng);

    std::printf("\nfused_gdn_conv_l2: %d failures\n", failures);
    if (failures) return 1;
    std::printf("fused_gdn_conv_l2_parity OK\n");
    (void)selftest;
    return 0;
}
