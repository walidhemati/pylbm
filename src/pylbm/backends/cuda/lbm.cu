// pylbm hand-written CUDA C fused lattice-Boltzmann kernel (compiled at runtime by NVRTC).
//
// One thread per lattice node, blockDim = (BLOCK_X, 1, 1), grid = (ceil(nx/BLOCK_X), ny, nz).
// Pull scheme, SoA layout f[i * N + n] with n = x + nx * (y + ny * z), half-way bounce-back
// with Ladd's moving-wall correction on the lid (top wall of the last axis, moving in +x),
// BGK / TRT / MRT (D2Q9) collision with Guo forcing.
//
// Compile-time configuration (prepended by cupy_backend.py):
//   Q, D, CX_INIT, CY_INIT, CZ_INIT, W_INIT, OPP_INIT  lattice tables
//   PER_X, PER_Y, PER_Z     1 if the axis is periodic, 0 if bounded by walls
//   LID, FORCE              0/1
//   COLL                    0 BGK, 1 TRT, 2 MRT (D2Q9 only, needs M_INIT, NORM_INIT)
//   STORAGE                 0 fp32, 1 fp16(f - w), 2 bf16(f - w); arithmetic is always fp32
//   PATTERN                 0 two-buffer (A -> B), 1 AA odd step, 2 AA even step (A == B)

#if STORAGE == 0
typedef float store_t;
__device__ __forceinline__ float ld(const store_t v) { return v; }
__device__ __forceinline__ store_t st(const float v) { return v; }
#elif STORAGE == 1
typedef unsigned short store_t;
__device__ __forceinline__ float ld(const store_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ store_t st(const float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
#else
typedef unsigned short store_t;
__device__ __forceinline__ float ld(const store_t h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ store_t st(const float f) {  // round to nearest even
    unsigned int u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
#endif

#if PATTERN == 0
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

extern "C" __global__ void __launch_bounds__(BLOCK_X)
lbm_kernel(const store_t* RESTRICT A, store_t* RESTRICT B, const int nx, const int ny,
           const int nz, const int N, const float ulid, const float fx, const float fy,
           const float fz, const float r0, const float r1, const float r2, const float r3)
{
    const int CX[Q] = CX_INIT;
    const int CY[Q] = CY_INIT;
    const int CZ[Q] = CZ_INIT;
    const float W[Q] = W_INIT;
    const int OPP[Q] = OPP_INIT;
    const bool PER[3] = {PER_X, PER_Y, PER_Z};
    const float shift = (STORAGE == 0) ? 0.0f : 1.0f;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const int y = blockIdx.y, z = blockIdx.z;
    const int crd[3] = {x, y, z};
    const int dims[3] = {nx, ny, nz};
    const int strd[3] = {1, nx, nx * ny};
    const int n = x + nx * (y + ny * z);

    // ------------------------------------------------------------------ stream (pull)
    float f[Q];
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        const int c[3] = {CX[i], CY[i], CZ[i]};
        bool wall = false, top = false;
        int off = 0;
#pragma unroll
        for (int a = 0; a < D; ++a) {
            if (c[a] == 0) continue;
            int s = crd[a] - c[a];
            if (s < 0 || s >= dims[a]) {
                if (PER[a]) {
                    s += (s < 0) ? dims[a] : -dims[a];
                } else {
                    wall = true;
                    if (a == D - 1 && s >= dims[a]) top = true;
                }
            }
            off += (s - crd[a]) * strd[a];
        }
        int idx;
        if (PATTERN == 0)      idx = wall ? OPP[i] * N + n : i * N + n + off;
        else if (PATTERN == 1) idx = wall ? i * N + n : OPP[i] * N + n + off;
        else                   idx = i * N + n;
        f[i] = ld(A[idx]) + shift * W[i];
        if (LID && PATTERN != 2 && top) f[i] += 6.0f * W[i] * CX[i] * ulid;
    }

    // ------------------------------------------------------------------ moments
    float rho = 0.0f, j[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        rho += f[i];
        j[0] += CX[i] * f[i];
        j[1] += CY[i] * f[i];
        j[2] += CZ[i] * f[i];
    }
    const float F[3] = {fx, fy, fz};
    const float inv_rho = 1.0f / rho;
    float u[3];
#pragma unroll
    for (int a = 0; a < 3; ++a) u[a] = (j[a] + (FORCE ? 0.5f * F[a] : 0.0f)) * inv_rho;
    const float usq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    const float uf = u[0] * F[0] + u[1] * F[1] + u[2] * F[2];

    float feq[Q], S[Q];
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        const float cu = CX[i] * u[0] + CY[i] * u[1] + CZ[i] * u[2];
        feq[i] = W[i] * rho * (1.0f - 1.5f * usq + cu * (3.0f + 4.5f * cu));
        const float cf = CX[i] * F[0] + CY[i] * F[1] + CZ[i] * F[2];
        S[i] = FORCE ? W[i] * (3.0f * (cf - uf) + 9.0f * cu * cf) : 0.0f;
    }

    // ------------------------------------------------------------------ collide
    float p[Q];
#if COLL == 0
#pragma unroll
    for (int i = 0; i < Q; ++i) p[i] = f[i] + r0 * (feq[i] - f[i]) + (1.0f - 0.5f * r0) * S[i];
#elif COLL == 1
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        const int o = OPP[i];
        const float np_ = 0.5f * (f[i] + f[o]) - 0.5f * (feq[i] + feq[o]);
        const float nm_ = 0.5f * (f[i] - f[o]) - 0.5f * (feq[i] - feq[o]);
        const float sp_ = 0.5f * (S[i] + S[o]), sm_ = 0.5f * (S[i] - S[o]);
        p[i] = f[i] - r0 * np_ - r1 * nm_ + (1.0f - 0.5f * r0) * sp_ + (1.0f - 0.5f * r1) * sm_;
    }
#else
    const float M[Q][Q] = M_INIT;
    const float NORM[Q] = NORM_INIT;
    // rates: rho, e, eps, jx, qx, jy, qy, pxx, pxy ; conserved moments relax with 0
    const float s[Q] = {0.0f, r1, r2, 0.0f, r3, 0.0f, r3, r0, r0};
    float dm[Q];
#pragma unroll
    for (int k = 0; k < Q; ++k) {
        float mne = 0.0f, ms = 0.0f;
#pragma unroll
        for (int i = 0; i < Q; ++i) {
            mne += M[k][i] * (f[i] - feq[i]);
            ms += M[k][i] * S[i];
        }
        dm[k] = (-s[k] * mne + (1.0f - 0.5f * s[k]) * ms) / NORM[k];
    }
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        float acc = f[i];
#pragma unroll
        for (int k = 0; k < Q; ++k) acc += M[k][i] * dm[k];
        p[i] = acc;
    }
#endif

    // ------------------------------------------------------------------ write
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        const float v = p[i] - shift * W[i];
        if (PATTERN == 0) {
            B[i * N + n] = st(v);
        } else if (PATTERN == 2) {
            B[OPP[i] * N + n] = st(v);
        } else {  // AA odd: push to x + c_i, or bounce back into own slot opp(i)
            const int c[3] = {CX[i], CY[i], CZ[i]};
            bool wall = false, top = false;
            int off = 0;
#pragma unroll
            for (int a = 0; a < D; ++a) {
                if (c[a] == 0) continue;
                int s2 = crd[a] + c[a];
                if (s2 < 0 || s2 >= dims[a]) {
                    if (PER[a]) {
                        s2 += (s2 < 0) ? dims[a] : -dims[a];
                    } else {
                        wall = true;
                        if (a == D - 1 && s2 >= dims[a]) top = true;
                    }
                }
                off += (s2 - crd[a]) * strd[a];
            }
            if (wall) {
                const float corr = (LID && top) ? -6.0f * W[i] * CX[i] * ulid : 0.0f;
                B[OPP[i] * N + n] = st(v + corr);
            } else {
                B[i * N + n + off] = st(v);
            }
        }
    }
}
