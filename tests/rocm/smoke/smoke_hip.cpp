// tests/rocm/smoke/smoke_hip.cpp - Stage 0's "does HIP actually work here" test.
//
// Deliberately tiny and self-checking: it must fail loudly if the toolchain
// builds for the wrong target, if the runtime cannot see the card, or if a
// kernel's arithmetic disagrees with a host computation. It is not a benchmark
// and not a port. Independent of the Strata tree on purpose - a smoke test that
// shares headers with the thing under test cannot tell you the toolchain works.
//
// Build: hipcc --offload-arch=gfx1100 -O2 smoke_hip.cpp -o smoke_hip
// Run:   ./smoke_hip [--gfx gfx1100]
//
// Exit codes: 0 all checks passed, 1 a check failed, 2 usage/bad arguments.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "SMOKE FAIL: %s\n", what);
    return 1;
}

#define HIPCHK(expr)                                                                     \
    do {                                                                                 \
        hipError_t _e = (expr);                                                          \
        if (_e != hipSuccess) {                                                          \
            std::fprintf(stderr, "SMOKE FAIL: %s -> %s\n", #expr, hipGetErrorString(_e)); \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

// Exact by construction: integer arithmetic, no reassociation possible.
__global__ void i32_axpy(const int* a, const int* b, int* c, int n, int s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = s * a[i] + b[i];
}

// Tolerant check: float maths, so the point is a finite, close result, not bits.
__global__ void f32_axpy(const float* a, const float* b, float* c, int n, float s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = fmaf(s, a[i], b[i]);
}

}  // namespace

int main(int argc, char** argv) {
    std::string want_gfx = "gfx1100";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gfx" && i + 1 < argc) want_gfx = argv[++i];
        else if (a == "--help" || a == "-h") { std::printf("usage: smoke_hip [--gfx gfx1100]\n"); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }

    int hip_ver = 0, ndev = 0;
    HIPCHK(hipRuntimeGetVersion(&hip_ver));
    HIPCHK(hipGetDeviceCount(&ndev));
    std::printf("hip_runtime_version=%d\n", hip_ver);
    std::printf("device_count=%d\n", ndev);
    if (ndev < 1) return fail("no HIP device visible (check /dev/kfd access)");

    HIPCHK(hipSetDevice(0));
    hipDeviceProp_t p{};
    HIPCHK(hipGetDeviceProperties(&p, 0));
    std::printf("device_name=%s\n", p.name);
    std::printf("gcn_arch=%s\n", p.gcnArchName);
    std::printf("warp_size=%d\n", p.warpSize);
    std::printf("total_mem_bytes=%zu\n", (size_t)p.totalGlobalMem);
    std::printf("multi_processor_count=%d\n", p.multiProcessorCount);
#ifdef __AMDGCN_WAVEFRONT_SIZE
    std::printf("compile_time_wavefront=%d\n", __AMDGCN_WAVEFRONT_SIZE);
#endif

    if (std::strstr(p.gcnArchName, want_gfx.c_str()) == nullptr)
        return fail(("device gcnArchName does not contain " + want_gfx).c_str());
    // RDNA (gfx10/gfx11) reports 32 here because wave32 is the default
    // wavefront on that family; GCN/CDNA report 64. Either is a real AMD
    // target, so the check is "one of the two", not "64". Kernels that assume a
    // 64-wide wave must not assume it here - see PORTING.md.
    if (p.warpSize != 32 && p.warpSize != 64)
        return fail("warpSize is neither 32 nor 64; this is not an AMD target");

    // ---- exact integer path -------------------------------------------------
    const int n = 1 << 14;              // 16384: several blocks, not one wave
    const int s = 7;
    std::vector<int> ha(n), hb(n), hc(n, 0);
    for (int i = 0; i < n; ++i) { ha[i] = i - n / 2; hb[i] = 3 * i + 1; }

    int *da = nullptr, *db = nullptr, *dc = nullptr;
    HIPCHK(hipMalloc(&da, n * sizeof(int)));
    HIPCHK(hipMalloc(&db, n * sizeof(int)));
    HIPCHK(hipMalloc(&dc, n * sizeof(int)));
    HIPCHK(hipMemcpy(da, ha.data(), n * sizeof(int), hipMemcpyHostToDevice));
    HIPCHK(hipMemcpy(db, hb.data(), n * sizeof(int), hipMemcpyHostToDevice));
    i32_axpy<<<(n + 255) / 256, 256>>>(da, db, dc, n, s);
    HIPCHK(hipGetLastError());
    HIPCHK(hipDeviceSynchronize());
    HIPCHK(hipMemcpy(hc.data(), dc, n * sizeof(int), hipMemcpyDeviceToHost));

    int int_bad = 0;
    for (int i = 0; i < n; ++i) if (hc[i] != s * ha[i] + hb[i]) ++int_bad;
    std::printf("int_exact_mismatches=%d\n", int_bad);
    if (int_bad != 0) return fail("integer kernel disagrees with the host");

    // ---- tolerant float path ------------------------------------------------
    std::vector<float> fa(n), fb(n), fc(n, 0.0f);
    for (int i = 0; i < n; ++i) { fa[i] = 0.5f * (float)i - 100.0f; fb[i] = 0.25f * (float)i + 1.0f; }
    float *ga = nullptr, *gb = nullptr, *gc = nullptr;
    HIPCHK(hipMalloc(&ga, n * sizeof(float)));
    HIPCHK(hipMalloc(&gb, n * sizeof(float)));
    HIPCHK(hipMalloc(&gc, n * sizeof(float)));
    HIPCHK(hipMemcpy(ga, fa.data(), n * sizeof(float), hipMemcpyHostToDevice));
    HIPCHK(hipMemcpy(gb, fb.data(), n * sizeof(float), hipMemcpyHostToDevice));
    f32_axpy<<<(n + 255) / 256, 256>>>(ga, gb, gc, n, 3.0f);
    HIPCHK(hipGetLastError());
    HIPCHK(hipDeviceSynchronize());
    HIPCHK(hipMemcpy(fc.data(), gc, n * sizeof(float), hipMemcpyDeviceToHost));

    double worst = 0.0;
    int nonfinite = 0;
    for (int i = 0; i < n; ++i) {
        const float expect = std::fma(3.0f, fa[i], fb[i]);
        if (!std::isfinite(fc[i])) { ++nonfinite; continue; }
        const double denom = std::fmax(1.0, std::fabs((double)expect));
        worst = std::fmax(worst, std::fabs((double)fc[i] - (double)expect) / denom);
    }
    std::printf("float_worst_rel_error=%.3e\n", worst);
    std::printf("float_nonfinite=%d\n", nonfinite);
    if (nonfinite != 0) return fail("float kernel produced non-finite output");
    if (worst > 1e-6) return fail("float kernel exceeds 1e-6 relative error");

    HIPCHK(hipFree(da)); HIPCHK(hipFree(db)); HIPCHK(hipFree(dc));
    HIPCHK(hipFree(ga)); HIPCHK(hipFree(gb)); HIPCHK(hipFree(gc));
    HIPCHK(hipDeviceReset());

    std::printf("SMOKE_RESULT=pass\n");
    return 0;
}
