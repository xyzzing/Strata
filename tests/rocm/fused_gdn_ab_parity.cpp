// tests/rocm/fused_gdn_ab_parity.cpp - the fused alpha/beta projection against a float64 host reference.
//
// The third `fused_gdn.hpp` entry point. The header states its arithmetic:
//
//   alpha_raw[h] = w_alpha[h, :] . x        (w_* are (h_v, n_embd) BF16, x is FP32)
//   beta_raw[h]  = w_beta[h, :]  . x
//   gate[h]      = softplus(alpha_raw[h] + dt[h]) * ssm_a[h]
//   beta[h]      = sigmoid(beta_raw[h])
//
// Unlike the other two fused tests there is no unfused MMVF kernel in this rollout's
// headers to compose, so the reference is the stronger kind anyway: a float64 host
// implementation over the SAME BF16 weight bits (BF16 -> FP32 widening is exact, so
// decoding the uint16 patterns loses nothing). softplus/sigmoid are the mathematical
// functions in double; the kernel's pinned log(1+exp(x)) threshold at 20 differs from
// the exact function by at most exp(-20) ~ 2e-9, far under the tolerance.
//
// Tolerance. The raw dot product is an n_embd-term fp32 reduction, so
// |err| <= n_embd * 2^-24 * sum|terms| per head. softplus is 1-Lipschitz and scales
// by |ssm_a|; sigmoid's slope is at most 1/4; both add a fixed 1e-6 for the
// transcendentals. The bound is applied per head, as an aggregate and as a max.
//
// Run: fused_gdn_ab_parity --selftest

#include "strata/kernels/fused_gdn.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

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
    uint64_t s = 0xa24baed4963ee40dull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
};

// BF16 bits -> the exactly-representable FP32 value.
float bf16_to_f32(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// FP32 -> BF16 bits by truncation. Round-to-nearest would also do: whatever
// bits this produces, the host reference decodes the SAME bits, so no rounding
// decision is load-bearing. What matters is that the inputs are sane floats -
// arbitrary 16-bit patterns include NaNs and ~1e38 magnitudes, which poison
// both sides identically and make every comparison vacuous.
uint16_t f32_to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

// The pinned CUDA expression: log(1+exp(x)), branch at 20. In double the branch
// and the exact function agree to exp(-20), which the tolerance absorbs.
double softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

int run_case(const char* name, int n_embd, int h_v, Rng& rng,
             const std::vector<float>* x_override = nullptr,
             const std::vector<uint16_t>* alpha_override = nullptr) {
    using namespace strata::kernels;

    std::vector<float> x(n_embd);
    for (auto& v : x) v = rng.uniform(-1.0f, 1.0f);
    if (x_override) x = *x_override;
    std::vector<uint16_t> w_alpha((size_t)h_v * n_embd), w_beta((size_t)h_v * n_embd);
    for (auto& v : w_alpha) v = f32_to_bf16(rng.uniform(-0.05f, 0.05f));
    if (alpha_override) w_alpha = *alpha_override;
    for (auto& v : w_beta) v = f32_to_bf16(rng.uniform(-0.05f, 0.05f));
    std::vector<float> dt(h_v), ssm_a(h_v);
    for (auto& v : dt) v = rng.uniform(-0.1f, 0.1f);
    for (auto& v : ssm_a) v = rng.uniform(-8.0f, -0.5f);

    // host float64 reference over the same BF16 bits
    std::vector<double> gate_ref(h_v), beta_ref(h_v), tol_gate(h_v), tol_beta(h_v);
    for (int h = 0; h < h_v; ++h) {
        double raw_a = 0.0, raw_b = 0.0, sum_abs_a = 0.0, sum_abs_b = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            const double wa = (double)bf16_to_f32(w_alpha[(size_t)h * n_embd + i]);
            const double wb = (double)bf16_to_f32(w_beta[(size_t)h * n_embd + i]);
            raw_a += wa * (double)x[i];
            raw_b += wb * (double)x[i];
            sum_abs_a += std::fabs(wa * (double)x[i]);
            sum_abs_b += std::fabs(wb * (double)x[i]);
        }
        const double raw_tol_a = (double)n_embd * std::ldexp(1.0, -24) * sum_abs_a;
        const double raw_tol_b = (double)n_embd * std::ldexp(1.0, -24) * sum_abs_b;
        gate_ref[h] = softplus(raw_a + (double)dt[h]) * (double)ssm_a[h];
        beta_ref[h] = sigmoid(raw_b);
        // softplus is 1-Lipschitz and the result scales by |ssm_a|; sigmoid's
        // slope is <= 1/4; both add room for the transcendentals themselves.
        tol_gate[h] = (1.0 + std::fabs((double)ssm_a[h])) * raw_tol_a + 1e-6;
        tol_beta[h] = 0.25 * raw_tol_b + 1e-6;
    }

    float *dx = nullptr, *dwa = nullptr, *dwb = nullptr, *ddt = nullptr, *dssm = nullptr;
    float *dgate = nullptr, *dbeta = nullptr;
    HIPCHK(hipMalloc(&dx, x.size() * 4));
    HIPCHK(hipMalloc(&dwa, w_alpha.size() * 2));
    HIPCHK(hipMalloc(&dwb, w_beta.size() * 2));
    HIPCHK(hipMalloc(&ddt, dt.size() * 4));
    HIPCHK(hipMalloc(&dssm, ssm_a.size() * 4));
    HIPCHK(hipMalloc(&dgate, (size_t)h_v * 4));
    HIPCHK(hipMalloc(&dbeta, (size_t)h_v * 4));

    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dwa, w_alpha.data(), w_alpha.size() * 2, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dwb, w_beta.data(), w_beta.size() * 2, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(ddt, dt.data(), dt.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dssm, ssm_a.data(), ssm_a.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    fused_gdn_ab(dx, (const uint16_t*)dwa, (const uint16_t*)dwb, ddt, dssm,
                 dgate, dbeta, n_embd, h_v, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<float> gate(h_v), beta(h_v);
    HIPCHK(hipMemcpy(gate.data(), dgate, gate.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(beta.data(), dbeta, beta.size() * 4, hipMemcpyDeviceToHost));

    int nonfinite = 0;
    for (float v : gate) if (!std::isfinite(v)) ++nonfinite;
    for (float v : beta) if (!std::isfinite(v)) ++nonfinite;

    int gate_wrong = 0, beta_wrong = 0;
    double gate_worst = 0.0, beta_worst = 0.0, gate_tol_worst = 0.0, beta_tol_worst = 0.0;
    for (int h = 0; h < h_v; ++h) {
        const double dg = std::fabs((double)gate[h] - gate_ref[h]);
        const double dbv = std::fabs((double)beta[h] - beta_ref[h]);
        gate_worst = std::fmax(gate_worst, dg);
        beta_worst = std::fmax(beta_worst, dbv);
        gate_tol_worst = std::fmax(gate_tol_worst, tol_gate[h]);
        beta_tol_worst = std::fmax(beta_tol_worst, tol_beta[h]);
        if (!(dg < tol_gate[h])) ++gate_wrong;
        if (!(dbv < tol_beta[h])) ++beta_wrong;
    }

    std::printf("  %-34s n_embd=%-5d h_v=%-3d  gate worst=%.2e (tol %.1e)  "
                "beta worst=%.2e (tol %.1e)  wrong=%d/%d  nonfinite=%d\n",
                name, n_embd, h_v, gate_worst, gate_tol_worst, beta_worst,
                beta_tol_worst, gate_wrong, beta_wrong, nonfinite);

    check(nonfinite == 0, "the fused kernel must produce finite output");
    check(gate_wrong == 0, "a head's gate exceeded its derived bound");
    check(beta_wrong == 0, "a head's beta exceeded its derived bound");

    hipStreamDestroy(stream);
    hipFree(dx); hipFree(dwa); hipFree(dwb); hipFree(ddt); hipFree(dssm);
    hipFree(dgate); hipFree(dbeta);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: fused_gdn_ab_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("fused_gdn_ab_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: float64 host dot/softplus/sigmoid over the SAME BF16 weight\n"
                "  bits (BF16->FP32 widening is exact), bounds derived per head\n");

    Rng rng;
    // the real artifact's projection shape: n_embd activation into h_v = 48 heads
    run_case("artifact-ish: 2048 -> 48 heads", 2048, 48, rng);
    run_case("single head, short", 512, 1, rng);

    {   // softplus threshold: alpha_raw near +25 (pinned branch: softplus(x) = x)
        std::vector<float> x(2048, 1.0f);
        std::vector<uint16_t> wa(48 * 2048);
        // bf16 bits of 25/2048 ~ 0.012207; every head sums to ~25 before dt.
        // Truncating the last 16 bits is fine here: the reference decodes the
        // SAME bits, so no rounding decision is load-bearing.
        const float per = 25.0f / 2048.0f;
        uint32_t bits32 = 0;
        std::memcpy(&bits32, &per, 4);
        for (auto& v : wa) v = (uint16_t)(bits32 >> 16);
        run_case("softplus above threshold", 2048, 48, rng, &x, &wa);
    }
    {   // softplus deep under: alpha_raw near -25, gate ~ ssm_a * 1.4e-11
        std::vector<float> x(2048, 1.0f);
        std::vector<uint16_t> wa(48 * 2048);
        const float per = -25.0f / 2048.0f;
        uint32_t bits32 = 0;
        std::memcpy(&bits32, &per, 4);
        for (auto& v : wa) v = (uint16_t)(bits32 >> 16);
        run_case("softplus below threshold", 2048, 48, rng, &x, &wa);
    }

    std::printf("\nfused_gdn_ab: %d failures\n", failures);
    if (failures) return 1;
    std::printf("fused_gdn_ab_parity OK\n");
    (void)selftest;
    return 0;
}
