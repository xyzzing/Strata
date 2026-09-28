// tests/rocm/native_gdn_preprocess_parity.cpp - independent check of the five GDN
// preprocessing helpers that feed the recurrence.
//
// No upstream test exists for these (their parity tests were in the omitted
// bench/micro tree, against llama.cpp's CUDA oracle). All five are small, and all
// five sit upstream of the recurrent step - a wrong gate or a wrong normalization
// scale does not stay small, it becomes the recurrence's input.
//
// The contracts in `include/strata/kernels/native_gdn_preprocess.hpp` are precise
// enough to check directly. Two asymmetries in them are the reason this file is
// worth more than its size suggests:
//
//   * `native_gdn_l2_norm` divides epsilon by 128 before use (`epsilon/128` is
//     passed in) while `native_gdn_out_norm` does NOT. Two normalizations that
//     look identical in the header are not, and a port that assumed otherwise
//     would be wrong by a scale factor nobody would notice from the output.
//   * `native_gdn_gate` uses the pinned `log(1+exp(x))` with a threshold of 20,
//     not `log1p` - so large inputs take the `x` branch and stay finite.
//
// Every reference here is float64 and independent of the kernels. Where the kernel
// deliberately keeps an fp32 store boundary (the `__fmul_rn`/`__fmaf_rn` in
// l2_norm, documenting a boundary between RMSNorm and a following scale), the
// reference mirrors the mathematical result and the test's bound covers the
// rounding.
//
// Run: native_gdn_preprocess_parity --selftest

#include "strata/kernels/native_gdn_preprocess.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int S = 128;
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
    uint64_t s = 0x452821e638d01377ull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
};

double rel_max(const std::vector<float>& got, const std::vector<double>& want) {
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        worst = std::fmax(worst, std::fabs((double)got[i] - want[i]));
        scale = std::fmax(scale, std::fabs(want[i]));
    }
    return worst / (scale + 1e-30);
}

// ---- 1. conv_silu -----------------------------------------------------------
int run_conv(Rng& rng) {
    using namespace strata::kernels;
    const int channels = 96, d_conv = 4;

    std::vector<float> hist((size_t)channels * 3), in((size_t)channels),
        w((size_t)channels * 4);
    for (auto& x : hist) x = rng.uniform(-1.5f, 1.5f);
    for (auto& x : in) x = rng.uniform(-1.5f, 1.5f);
    for (auto& x : w) x = rng.uniform(-1.0f, 1.0f);

    std::vector<double> raw_ref((size_t)channels), silu_ref((size_t)channels),
        hist_ref(hist.begin(), hist.end());
    for (int c = 0; c < channels; ++c) {
        double v[4] = {hist_ref[c * 3], hist_ref[c * 3 + 1], hist_ref[c * 3 + 2], (double)in[c]};
        double sum = 0.0;
        for (int t = 0; t < 4; ++t) sum += v[t] * (double)w[c * 4 + t];
        raw_ref[c] = sum;
        silu_ref[c] = sum / (1.0 + std::exp(-sum));          // x * sigmoid(x)
        for (int t = 0; t < 3; ++t) hist_ref[c * 3 + t] = v[t + 1];
    }

    float *dh = nullptr, *di = nullptr, *dw = nullptr, *draw = nullptr, *dsilu = nullptr;
    HIPCHK(hipMalloc(&dh, hist.size() * 4));
    HIPCHK(hipMalloc(&di, in.size() * 4));
    HIPCHK(hipMalloc(&dw, w.size() * 4));
    HIPCHK(hipMalloc(&draw, (size_t)channels * 4));
    HIPCHK(hipMalloc(&dsilu, (size_t)channels * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dh, hist.data(), hist.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(di, in.data(), in.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dw, w.data(), w.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    native_gdn_conv_silu(dh, di, dw, draw, dsilu, channels, d_conv, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<float> raw((size_t)channels), silu((size_t)channels), hist_out(hist.size());
    HIPCHK(hipMemcpy(raw.data(), draw, raw.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(silu.data(), dsilu, silu.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(hist_out.data(), dh, hist_out.size() * 4, hipMemcpyDeviceToHost));

    const double e_raw = rel_max(raw, raw_ref), e_silu = rel_max(silu, silu_ref);
    double e_hist = 0.0;
    for (size_t i = 0; i < hist_out.size(); ++i)
        e_hist = std::fmax(e_hist, std::fabs((double)hist_out[i] - hist_ref[i]) /
                                       (std::fabs(hist_ref[i]) + 1e-30));
    const double tol = 8.0 * std::ldexp(1.0, -24);
    std::printf("  %-34s raw rel=%.3e  silu rel=%.3e  history rel=%.3e  tol=%.1e\n",
                "conv_silu", e_raw, e_silu, e_hist, tol);
    check(e_raw < tol, "the convolution sum exceeded the fp32 bound");
    check(e_silu < tol, "the SiLU output exceeded the fp32 bound");
    check(e_hist < tol, "the history shift must be exact");
    hipStreamDestroy(stream);
    hipFree(dh); hipFree(di); hipFree(dw); hipFree(draw); hipFree(dsilu);
    return 0;
}

// ---- 2. l2_norm and 5. out_norm --------------------------------------------
int run_norms(Rng& rng) {
    using namespace strata::kernels;
    const int rows = 5, cols = S;
    const float eps = 1e-6f;

    std::vector<float> x((size_t)rows * cols), z((size_t)rows * cols), gamma(cols);
    for (auto& v : x) v = rng.uniform(-2.0f, 2.0f);
    for (auto& v : z) v = rng.uniform(-3.0f, 3.0f);
    for (auto& v : gamma) v = rng.uniform(-1.0f, 1.0f);
    std::vector<float> x0 = x;

    // l2_norm: scale(rms_norm(x, eps/128), 1/sqrt(128))
    std::vector<double> l2_ref(x.size());
    for (int r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (int c = 0; c < cols; ++c) ss += (double)x[(size_t)r * cols + c] * x[(size_t)r * cols + c];
        const double sc = 1.0 / std::sqrt(ss / S + (double)eps / S);
        for (int c = 0; c < cols; ++c)
            l2_ref[(size_t)r * cols + c] = (double)x[(size_t)r * cols + c] * sc / std::sqrt((double)S);
    }
    // out_norm: rms_norm(x, eps) * gamma * sigmoid(z)   -- epsilon NOT divided by S
    //
    // Its INPUT is the buffer l2_norm has already modified in place, so the
    // reference is built from that, rounded back to fp32 as the device stores it.
    // A first revision referenced the ORIGINAL x here and was 6.3e-05 off - not a
    // kernel error but this test's, and the residual was exactly the epsilon term,
    // which only matters when the row magnitude is small. That observation is what
    // the small-magnitude case below now tests deliberately.
    std::vector<double> on_ref(x.size());
    for (int r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double v = (double)(float)l2_ref[(size_t)r * cols + c];
            ss += v * v;
        }
        const double sc = 1.0 / std::sqrt(ss / S + (double)eps);
        for (int c = 0; c < cols; ++c) {
            const double sig = 1.0 / (1.0 + std::exp(-(double)z[(size_t)r * cols + c]));
            const double v = (double)(float)l2_ref[(size_t)r * cols + c];
            on_ref[(size_t)r * cols + c] = v * sc * (double)gamma[c] * sig;
        }
    }

    float *dx = nullptr, *dz = nullptr, *dg = nullptr, *dout = nullptr;
    HIPCHK(hipMalloc(&dx, x.size() * 4));
    HIPCHK(hipMalloc(&dz, z.size() * 4));
    HIPCHK(hipMalloc(&dg, gamma.size() * 4));
    HIPCHK(hipMalloc(&dout, x.size() * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dz, z.data(), z.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dg, gamma.data(), gamma.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    native_gdn_l2_norm(dx, rows, cols, eps, stream);
    native_gdn_out_norm(dx, dz, dg, dout, rows, cols, eps, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<float> l2_got(x.size()), on_got(x.size());
    HIPCHK(hipMemcpy(l2_got.data(), dx, x.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(on_got.data(), dout, x.size() * 4, hipMemcpyDeviceToHost));

    const double e_l2 = rel_max(l2_got, l2_ref), e_on = rel_max(on_got, on_ref);
    const double tol = 8.0 * S * std::ldexp(1.0, -24);
    std::printf("  %-34s l2 rel=%.3e  out_norm rel=%.3e  tol=%.1e\n", "l2_norm + out_norm",
                e_l2, e_on, tol);
    check(e_l2 < tol, "l2_norm exceeded the bound derived from the 128-value reduction");
    check(e_on < tol, "out_norm exceeded the bound derived from the 128-value reduction");

    // The asymmetry the header does not spell out: out_norm uses epsilon as given,
    // l2_norm uses epsilon/128. On ordinary data the two are indistinguishable
    // (4e-07 apart here), so asserting they differ would prove nothing. The regime
    // where it bites is a SMALL-MAGNITUDE row, where the epsilon term dominates the
    // mean square - and that is run as a real case below rather than argued about.
    {
        std::vector<float> tiny(x.size());
        for (auto& v : tiny) v = rng.uniform(-1e-4f, 1e-4f);
        std::vector<double> l2_tiny(tiny.size());
        for (int r = 0; r < rows; ++r) {
            double ss = 0.0;
            for (int c = 0; c < cols; ++c) ss += (double)tiny[(size_t)r * cols + c] * tiny[(size_t)r * cols + c];
            const double sc = 1.0 / std::sqrt(ss / S + (double)eps / S);
            for (int c = 0; c < cols; ++c)
                l2_tiny[(size_t)r * cols + c] = (double)tiny[(size_t)r * cols + c] * sc / std::sqrt((double)S);
        }
        float* dt_ = nullptr;
        HIPCHK(hipMalloc(&dt_, tiny.size() * 4));
        HIPCHK(hipMemcpyAsync(dt_, tiny.data(), tiny.size() * 4, hipMemcpyHostToDevice, stream));
        HIPCHK(hipStreamSynchronize(stream));
        native_gdn_l2_norm(dt_, rows, cols, eps, stream);
        HIPCHK(hipStreamSynchronize(stream));
        std::vector<float> got(tiny.size());
        HIPCHK(hipMemcpy(got.data(), dt_, tiny.size() * 4, hipMemcpyDeviceToHost));
        const double e_tiny = rel_max(got, l2_tiny);
        // the same row normalized with epsilon instead of epsilon/128 - what a port
        // that mixed the two up would produce
        double swap_gap = 0.0;
        for (int r = 0; r < rows; ++r) {
            double ss = 0.0;
            for (int c = 0; c < cols; ++c) ss += (double)tiny[(size_t)r * cols + c] * tiny[(size_t)r * cols + c];
            const double a = 1.0 / std::sqrt(ss / S + (double)eps / S);
            const double b = 1.0 / std::sqrt(ss / S + (double)eps);
            swap_gap = std::fmax(swap_gap, std::fabs(a - b) / a);
        }
        std::printf("  %-34s l2 rel=%.3e  (mixing up epsilon/128 and epsilon would be %.3e off)\n",
                    "small-magnitude row", e_tiny, swap_gap);
        check(e_tiny < tol, "l2_norm on a small-magnitude row exceeded the bound");
        check(swap_gap > 1e-2, "the epsilon placement must be observable here, or the case proves nothing");
        hipFree(dt_);
    }

    hipStreamDestroy(stream);
    hipFree(dx); hipFree(dz); hipFree(dg); hipFree(dout);
    return 0;
}

// ---- 3. beta_gate and 4. gate ----------------------------------------------
int run_gates(Rng& rng) {
    using namespace strata::kernels;
    const int heads = 96;

    std::vector<float> beta(heads), alpha(heads), dt(heads), ssm_a(heads);
    for (auto& v : beta) v = rng.uniform(-6.0f, 6.0f);
    for (auto& v : alpha) v = rng.uniform(-30.0f, 30.0f);      // straddles the threshold of 20
    for (auto& v : dt) v = rng.uniform(-5.0f, 5.0f);
    for (auto& v : ssm_a) v = rng.uniform(-1.0f, 1.0f);

    std::vector<double> beta_ref(heads), gate_ref(heads);
    for (int h = 0; h < heads; ++h) {
        beta_ref[h] = 1.0 / (1.0 + std::exp(-(double)beta[h]));
        const double value = (double)(alpha[h] + dt[h]);       // __fadd_rn: a plain fp32 add
        const double softplus = value > 20.0 ? value : std::log(1.0 + std::exp(value));
        gate_ref[h] = softplus * (double)ssm_a[h];
    }

    float *db = nullptr, *da = nullptr, *ddt = nullptr, *dsa = nullptr, *dgt = nullptr;
    HIPCHK(hipMalloc(&db, heads * 4));
    HIPCHK(hipMalloc(&da, heads * 4));
    HIPCHK(hipMalloc(&ddt, heads * 4));
    HIPCHK(hipMalloc(&dsa, heads * 4));
    HIPCHK(hipMalloc(&dgt, heads * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(db, beta.data(), heads * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(da, alpha.data(), heads * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(ddt, dt.data(), heads * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dsa, ssm_a.data(), heads * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    native_gdn_beta_gate(db, heads, stream);
    native_gdn_gate(da, ddt, dsa, dgt, heads, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<float> beta_got(heads), gate_got(heads);
    HIPCHK(hipMemcpy(beta_got.data(), db, heads * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(gate_got.data(), dgt, heads * 4, hipMemcpyDeviceToHost));

    int nonfinite = 0;
    for (float v : gate_got) if (!std::isfinite(v)) ++nonfinite;
    const double e_beta = rel_max(beta_got, beta_ref), e_gate = rel_max(gate_got, gate_ref);
    const double tol = 4.0 * std::ldexp(1.0, -24);
    int over_threshold = 0;
    for (int h = 0; h < heads; ++h) if ((double)(alpha[h] + dt[h]) > 20.0) ++over_threshold;
    std::printf("  %-34s beta rel=%.3e  gate rel=%.3e  nonfinite=%d  (%d of %d heads above the "
                "softplus threshold)\n", "beta_gate + gate", e_beta, e_gate, nonfinite,
                over_threshold, heads);
    check(e_beta < tol, "sigmoid(beta) exceeded the fp32 bound");
    check(e_gate < tol, "the gate exceeded the fp32 bound");
    check(nonfinite == 0, "the gate must stay finite past the softplus threshold");
    check(over_threshold > 0, "the fixture must exercise the threshold branch, or it proves nothing");

    hipStreamDestroy(stream);
    hipFree(db); hipFree(da); hipFree(ddt); hipFree(dsa); hipFree(dgt);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: native_gdn_preprocess_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("native_gdn_preprocess_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: float64 versions of the contracted formulas\n");

    Rng rng;
    if (run_conv(rng) != 0) check(false, "conv case aborted");
    if (run_norms(rng) != 0) check(false, "norm cases aborted");
    if (run_gates(rng) != 0) check(false, "gate cases aborted");

    std::printf("\nnative_gdn_preprocess: %d failures\n", failures);
    if (failures) return 1;
    std::printf("native_gdn_preprocess_parity OK\n");
    (void)selftest;
    return 0;
}
