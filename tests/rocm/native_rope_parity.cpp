// tests/rocm/native_rope_parity.cpp - independent check of the native RoPE.
//
// No upstream test exists for this kernel (its parity test was in the omitted
// bench/micro tree, against llama.cpp's CUDA oracle). `src/kernels/rope_parity.cpp`
// covers the *other* RoPE path, and its header comment makes the point that
// matters here: choosing the wrong pairing "produces correctly-shaped output with
// scrambled content". The native path adds an IMRoPE table on top of that.
//
// The contract, read off `include/strata/kernels/native_rope.hpp` and the kernel:
//
//   theta = mrope_pos(tab, positions[row], pair) * powf(theta_scale, pair)
//   theta_scale = powf(freq_base, -2/64)        (host float powf, before launch)
//   mrope_pos(tab, pos, pair) = tab ? tab[pos*3 + pair % 3] : pos
//   a = x[start+pair], b = x[start+pair+32]     <- NEOX pairing, not adjacent
//   out[start+pair]      = a*cos - b*sin
//   out[start+pair+32]   = a*sin + b*cos
//   pairs >= 32, and every channel from 64 up, are COPIED UNCHANGED
//
// Three of those get a check that is not a tolerance:
//
//   * the pass-through region is asserted BYTE-EXACT - it is a copy, and a kernel
//     that rotated the whole row would give plausible-looking garbage there;
//   * the rotation is an isometry, so |out| = |in| per pair is checked against a
//     2-ulp budget. A wrong pairing breaks it immediately, and unlike a
//     component-wise comparison it does not depend on the trig library matching;
//   * pair 0 has `powf(theta_scale, 0) == 1` exactly, so its angle is exactly the
//     position - the one place the angle itself is pinned rather than compared.
//
// The component-wise comparison against a float64 reference is tolerance-based,
// because `cosf`/`sinf`/`powf` are library functions whose last ulp is not
// guaranteed to match between the host and the device.
//
// Run: native_rope_parity --selftest

#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"

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
            std::fprintf(stderr, "hip error %s: %s\n", #expr, hipGetErrorString(_e));   \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

struct Rng {
    uint64_t s = 0x9216d5d98979fb1bull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
    int range(int lo, int hi) { return lo + (int)(next() % (uint32_t)(hi - lo + 1)); }
};

constexpr int kRot = 64;          // n_rot, the only value the kernel accepts

// Reference in float64, with the kernel's structure. `tab` is nullptr for the
// text-only path, or the same host copy of the IMRoPE table the device gets.
void reference(const std::vector<float>& x, int rows, int head_dim, float freq_base,
               const std::vector<int>& positions, const std::vector<int32_t>* tab,
               std::vector<double>& out) {
    out.assign(x.size(), 0.0);
    const double theta_scale = std::pow((double)freq_base, -2.0 / (double)kRot);
    for (int row = 0; row < rows; ++row) {
        const size_t start = (size_t)row * head_dim;
        for (int c = 0; c < head_dim; ++c) out[start + c] = (double)x[start + c];   // copy first
        for (int pair = 0; pair < kRot / 2; ++pair) {
            int pos = positions[(size_t)row];
            if (tab) pos = (*tab)[(size_t)pos * 3 + pair % 3];
            const double theta = (double)pos * std::pow(theta_scale, (double)pair);
            const double c = std::cos(theta), s = std::sin(theta);
            const double a = (double)x[start + pair], b = (double)x[start + pair + kRot / 2];
            out[start + pair] = a * c - b * s;
            out[start + pair + kRot / 2] = a * s + b * c;
        }
    }
}

struct Case {
    const char* name;
    int head_dim;
    int rows;
    float freq_base;
    bool use_table;
    bool in_place;
};

int run_case(const Case& tc, Rng& rng) {
    using namespace strata::kernels;
    const float freq_base_f = tc.freq_base;
    const size_t n = (size_t)tc.rows * tc.head_dim;

    std::vector<float> x(n);
    for (auto& v : x) v = rng.uniform(-2.0f, 2.0f);
    std::vector<int> positions((size_t)tc.rows);
    for (int r = 0; r < tc.rows; ++r) positions[(size_t)r] = rng.range(0, 4096);

    // A table, when used, must be a DEVICE int32 [cells][3]. The positions above are
    // cell indices in that case, so they must stay inside the table.
    const int cells = 4097;
    std::vector<int32_t> tab_host;
    std::vector<int32_t> tab_ref;
    if (tc.use_table) {
        tab_host.assign((size_t)cells * 3, 0);
        for (int i = 0; i < cells; ++i) {
            tab_host[(size_t)i * 3 + 0] = i;                 // t
            tab_host[(size_t)i * 3 + 1] = i / 2;             // h differs, so the sector matters
            tab_host[(size_t)i * 3 + 2] = i / 3;             // w differs too
        }
        tab_ref = tab_host;
    }

    std::vector<double> ref;
    reference(x, tc.rows, tc.head_dim, tc.freq_base, positions, tc.use_table ? &tab_ref : nullptr, ref);

    float *dx = nullptr, *dout = nullptr;
    int* dpos = nullptr;
    int32_t* dtab = nullptr;
    HIPCHK(hipMalloc(&dx, n * 4));
    if (!tc.in_place) HIPCHK(hipMalloc(&dout, n * 4));
    HIPCHK(hipMalloc(&dpos, (size_t)tc.rows * 4));
    if (tc.use_table) HIPCHK(hipMalloc(&dtab, tab_host.size() * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dx, x.data(), n * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dpos, positions.data(), (size_t)tc.rows * 4, hipMemcpyHostToDevice, stream));
    if (tc.use_table) HIPCHK(hipMemcpyAsync(dtab, tab_host.data(), tab_host.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    mrope_table_set(tc.use_table ? dtab : nullptr);
    native_rope_apply(dx, tc.in_place ? dx : dout, tc.rows, tc.head_dim, kRot, tc.freq_base, dpos, stream);
    HIPCHK(hipStreamSynchronize(stream));
    mrope_table_set(nullptr);        // leave the process as we found it

    std::vector<float> got(n);
    HIPCHK(hipMemcpy(got.data(), tc.in_place ? dx : dout, n * 4, hipMemcpyDeviceToHost));

    // Component-wise, in the rotated region, against a bound derived from what the
    // kernel's own arithmetic can cost.
    //
    // A flat relative tolerance is WRONG here and the first revision of this test
    // used one. The angle is computed in fp32 as `pos * powf(theta_scale, pair)`,
    // and for small `pair` that angle is LARGE - thousands of radians, because
    // `theta_scale ~ 0.65` decays slowly. One ulp of an angle of 2659 radians is
    // 1.6e-04, and `cos`/`sin` amplify that straight into the output. So the
    // observable error scales with the ANGLE, not with the output: a rotation whose
    // angle is stored in fp32 cannot be compared component-wise to a float64 angle
    // at any tolerance that ignores this.
    //
    // The bound below is therefore per element:
    //     (|a| + |b|) * (a few ulps of theta)  +  a few ulps of the output
    // which is the actual error propagation, not a number picked to pass.
    const float theta_scale_f = std::pow(freq_base_f, -2.0f / (float)kRot);
    double worst = 0.0, scale = 0.0, worst_excess = 0.0;
    int passthrough_wrong = 0;
    double worst_norm = 0.0;

    for (int row = 0; row < tc.rows; ++row) {
        const size_t start = (size_t)row * tc.head_dim;
        for (int c = 0; c < tc.head_dim; ++c) {
            if (c < kRot) {
                const double err = std::fabs((double)got[start + c] - ref[start + c]);
                const int pair = c < kRot / 2 ? c : c - kRot / 2;
                int pos = positions[(size_t)row];
                if (tc.use_table) pos = tab_ref[(size_t)pos * 3 + pair % 3];
                const float theta_f = (float)pos * std::pow(theta_scale_f, (float)pair);
                const size_t other = (c < kRot / 2) ? start + c + kRot / 2 : start + c - kRot / 2;
                const double mag = std::fabs((double)x[start + c]) + std::fabs((double)x[other]);
                const double bound = mag * 8.0 * std::ldexp(1.0, -24) * std::fabs((double)theta_f) +
                                     8.0 * std::ldexp(1.0, -24) * std::fabs(ref[start + c]) + 1e-9;
                worst = std::fmax(worst, err);
                scale = std::fmax(scale, std::fabs(ref[start + c]));
                worst_excess = std::fmax(worst_excess, err - bound);
            } else if (std::memcmp(&got[start + c], &x[start + c], 4) != 0) {
                ++passthrough_wrong;                 // must be an untouched copy
            }
        }
        for (int pair = 0; pair < kRot / 2; ++pair) {
            const double a = (double)x[start + pair], b = (double)x[start + pair + kRot / 2];
            const double in_norm = std::sqrt(a * a + b * b);
            const double out_norm = std::sqrt((double)got[start + pair] * (double)got[start + pair] +
                                              (double)got[start + pair + kRot / 2] *
                                                  (double)got[start + pair + kRot / 2]);
            worst_norm = std::fmax(worst_norm, std::fabs(out_norm - in_norm));
        }
    }
    const double rel = worst / (scale + 1e-30);
    const double norm_scale = std::fmax(1.0, scale);
    const double norm_tol = 4.0 * std::ldexp(1.0, -23) * norm_scale;

    std::printf("  %-30s dim=%-4d rows=%-3d base=%-8.0f table=%d inplace=%d  worst=%.2e rel=%.2e  "
                "bound excess=%.2e  copy-wrong=%d  norm err=%.2e (tol %.0e)\n",
                tc.name, tc.head_dim, tc.rows, (double)tc.freq_base, (int)tc.use_table,
                (int)tc.in_place, worst, rel, worst_excess, passthrough_wrong, worst_norm, norm_tol);

    check(worst_excess <= 0.0,
          "the rotated region exceeded the bound derived from the fp32 angle's own resolution");
    check(passthrough_wrong == 0,
          "channels from n_rot up must be copied byte-exactly, not recomputed");
    check(worst_norm < norm_tol, "the rotation must be an isometry per pair");
    hipStreamDestroy(stream);
    hipFree(dx); if (dout) hipFree(dout); hipFree(dpos); if (dtab) hipFree(dtab);
    return 0;
}

// Pair 0 pins the angle itself: powf(theta_scale, 0) is exactly 1, so
// theta == position, with no dependence on the frequency schedule.
int run_pair_zero(Rng& rng) {
    using namespace strata::kernels;
    const int head_dim = 256, rows = 8;
    std::vector<float> x((size_t)rows * head_dim, 0.0f);
    std::vector<int> positions((size_t)rows);
    for (int r = 0; r < rows; ++r) {
        positions[(size_t)r] = rng.range(1, 5000);
        x[(size_t)r * head_dim + 0] = 1.0f;                 // a = 1
        x[(size_t)r * head_dim + kRot / 2] = 0.0f;          // b = 0  => out = (cos, sin)
    }
    float *dx = nullptr;
    int* dpos = nullptr;
    HIPCHK(hipMalloc(&dx, x.size() * 4));
    HIPCHK(hipMalloc(&dpos, (size_t)rows * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dpos, positions.data(), (size_t)rows * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));
    mrope_table_set(nullptr);
    native_rope_apply(dx, dx, rows, head_dim, kRot, 1.0e6f, dpos, stream);
    HIPCHK(hipStreamSynchronize(stream));
    std::vector<float> got(x.size());
    HIPCHK(hipMemcpy(got.data(), dx, x.size() * 4, hipMemcpyDeviceToHost));

    double worst = 0.0;
    for (int r = 0; r < rows; ++r) {
        const double want_c = std::cos((double)positions[(size_t)r]);
        const double want_s = std::sin((double)positions[(size_t)r]);
        worst = std::fmax(worst, std::fabs((double)got[(size_t)r * head_dim] - want_c));
        worst = std::fmax(worst, std::fabs((double)got[(size_t)r * head_dim + kRot / 2] - want_s));
    }
    std::printf("  %-30s worst|err| against cos/sin(position) = %.3e\n", "pair 0, angle == position", worst);
    check(worst < 1e-5, "pair 0 must rotate by exactly the position");
    hipStreamDestroy(stream);
    hipFree(dx); hipFree(dpos);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: native_rope_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("native_rope_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: float64 NEOX rotation with the kernel's own angle schedule;\n"
                "  the copy region is checked byte-exactly and the rotation as an isometry\n");

    Rng rng;
    const Case cases[] = {
        {"text-only, 256 wide",        256, 6, 1.0e6f, false, false},
        {"text-only, 128 wide",        128, 5, 1.0e6f, false, false},
        {"small freq_base",            256, 4, 1.0e4f, false, false},
        {"in place",                   256, 4, 1.0e6f, false, true},
        {"IMRoPE table",               256, 6, 1.0e6f, true,  false},
        {"IMRoPE table, in place",     128, 4, 1.0e6f, true,  true},
    };
    for (const Case& c : cases) {
        if (run_case(c, rng) != 0) check(false, "case aborted early", c.name);
    }
    if (run_pair_zero(rng) != 0) check(false, "pair 0 aborted");

    std::printf("\nnative_rope: %d failures\n", failures);
    if (failures) return 1;
    std::printf("native_rope_parity OK\n");
    (void)selftest;
    return 0;
}
