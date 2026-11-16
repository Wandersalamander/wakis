// copyright ################################# //
// This file is part of the wakis Package.     //
// Copyright (c) CERN, 2026.                   //
// ########################################### //

// CUDA kernels of the matrix-free GPU stencil backend (wakis/stencil.py).
// Compiled at runtime by cupy.RawModule (NVRTC, no host compiler needed);
// stencil.py prepends `typedef float|double real;` and, for CPML,
// `#define KAPPA`.
//
// One thread per primal node computes all three components of
// r * C0 @ (c * x) (forward) or r * C0.T @ (c * x) (transposed).
// Node index n = i + Nx*j + Nx*Ny*k (x fastest, as in Field); component d is
// stored at d*N + n. `fwd`/`bwd` must stay consistent with `_incidence`, which
// `check_curl_stencil` compares against the assembled solver matrices.

// Forward neighbour of node n along one axis, or -1 when the node is the
// terminal storage node and has no forward primal edge. Periodic axes wrap
// the last physical node (idx == len - 2) to idx = 0, i.e. period len - 1.
__device__ long long fwd(long long n, long long idx, long long len,
                         long long stride, int periodic)
{
    if (idx == len - 1) return -1;
    if (periodic && idx == len - 2) return n - idx * stride;
    return n + stride;
}

// Node whose forward edge ends at m (rows of P.T), or -1 when there is none.
__device__ long long bwd(long long m, long long idx, long long len,
                         long long stride, int periodic)
{
    if (idx > 0) {
        if (periodic && idx == len - 1) return -1;
        return m - stride;
    }
    if (periodic && len > 1) return m + (len - 2) * stride;
    return -1;
}

// (P u)[n] = u[fwd(n)] - u[n] for active nodes, 0 for terminal nodes.
__device__ real dfwd(const real* x, const real* c, long long n, long long nb)
{
    return nb < 0 ? (real)0 : c[nb] * x[nb] - c[n] * x[n];
}

// (P.T u)[m] = -u[m] for active m, plus u[bwd(m)] when it exists.
__device__ real dbwd(const real* x, const real* c, long long m, long long nb,
                     int terminal)
{
    real out = terminal ? (real)0 : -c[m] * x[m];
    if (nb >= 0) out += c[nb] * x[nb];
    return out;
}

// CPML coordinate stretching: the derivative along axis a is scaled by
// kap[a*N + n] = 1/kappa_a at node n. Compiled out without CPML.
#ifdef KAPPA
#define KAP(a, n) kap[(a) * N + (n)]
#else
#define KAP(a, n) ((real)1)
#endif

extern "C" __global__ void curl_fwd(
    const real* x, const real* r, const real* c, const real* kap, real* out,
    const long long Nx, const long long Ny, const long long Nz,
    const int px, const int py, const int pz)
{
    const long long N = Nx * Ny * Nz;
    const long long n = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= N) return;
    const long long i = n % Nx, j = (n / Nx) % Ny, k = n / (Nx * Ny);
    const long long nx = fwd(n, i, Nx, 1, px);
    const long long ny = fwd(n, j, Ny, Nx, py);
    const long long nz = fwd(n, k, Nz, Nx * Ny, pz);
    const real *x0 = x, *x1 = x + N, *x2 = x + 2 * N;
    const real *c0 = c, *c1 = c + N, *c2 = c + 2 * N;

    // C0 rows: [0, -Pz, Py], [Pz, 0, -Px], [-Py, Px, 0]
    out[n] = r[n] * (KAP(1, n) * dfwd(x2, c2, n, ny) - KAP(2, n) * dfwd(x1, c1, n, nz));
    out[N + n] = r[N + n] * (KAP(2, n) * dfwd(x0, c0, n, nz) - KAP(0, n) * dfwd(x2, c2, n, nx));
    out[2 * N + n] = r[2 * N + n] * (KAP(0, n) * dfwd(x1, c1, n, nx) - KAP(1, n) * dfwd(x0, c0, n, ny));
}

extern "C" __global__ void curl_bwd(
    const real* x, const real* r, const real* c, const real* kap, real* out,
    const long long Nx, const long long Ny, const long long Nz,
    const int px, const int py, const int pz)
{
    const long long N = Nx * Ny * Nz;
    const long long m = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (m >= N) return;
    const long long i = m % Nx, j = (m / Nx) % Ny, k = m / (Nx * Ny);
    const long long nx = bwd(m, i, Nx, 1, px);
    const long long ny = bwd(m, j, Ny, Nx, py);
    const long long nz = bwd(m, k, Nz, Nx * Ny, pz);
    const int tx = i == Nx - 1, ty = j == Ny - 1, tz = k == Nz - 1;
    const real *x0 = x, *x1 = x + N, *x2 = x + 2 * N;
    const real *c0 = c, *c1 = c + N, *c2 = c + 2 * N;

    // C0.T rows: [0, Pz.T, -Py.T], [-Pz.T, 0, Px.T], [Py.T, -Px.T, 0]
    out[m] = r[m] * (KAP(2, m) * dbwd(x1, c1, m, nz, tz) - KAP(1, m) * dbwd(x2, c2, m, ny, ty));
    out[N + m] = r[N + m] * (KAP(0, m) * dbwd(x2, c2, m, nx, tx) - KAP(2, m) * dbwd(x0, c0, m, nz, tz));
    out[2 * N + m] = r[2 * N + m] * (KAP(1, m) * dbwd(x0, c0, m, ny, ty) - KAP(0, m) * dbwd(x1, c1, m, nx, tx));
}

// CPML convolution on one face slab, one thread per slab entry p at node
// n = idx[p]. A face normal to `axis` couples two (component, source) pairs
// through the derivative along `axis`:
//   a: c = axis+2, source axis+1 (e.g. x face: Hz from d/dx Ey), sign +1
//   b: c = axis+1, source axis+2 (e.g. x face: Hy from d/dx Ez), sign -1
// (indices mod 3). The directional derivative is g_c / kappa_axis times the
// forward difference (H update) or minus the transposed difference (E
// update) of col * X, i.e. the operators dxy..dzy / dtxy..dtzy. TFSF adds
// t_a/t_b (node-indexed) on z faces. Then
//   psi = b psi + c d,   F_c += scale * mat_c * (+psi_a / -psi_b),
// with scale = -dt for H and +dt for E.
extern "C" __global__ void cpml_psi(
    real* F, const real* X, const real* g, const real* kap, const real* col,
    const real* mat, const long long* idx, real* psi_a, real* psi_b,
    const real* pb, const real* pc, const real* t_a, const real* t_b,
    const long long M, const long long Nx, const long long Ny, const long long Nz,
    const int axis, const int transpose, const int px, const int py, const int pz,
    const int use_t, const real scale)
{
    const long long p = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= M) return;
    const long long N = Nx * Ny * Nz;
    const long long n = idx[p];
    const long long i = n % Nx, j = (n / Nx) % Ny, k = n / (Nx * Ny);
    long long pos, len, stride;
    int periodic;
    if (axis == 0) { pos = i; len = Nx; stride = 1; periodic = px; }
    else if (axis == 1) { pos = j; len = Ny; stride = Nx; periodic = py; }
    else { pos = k; len = Nz; stride = Nx * Ny; periodic = pz; }
    const int ca = (axis + 2) % 3, oa = (axis + 1) % 3;
    const int cb = oa, ob = ca;

    real da, db;
    if (transpose) {
        const long long nb = bwd(n, pos, len, stride, periodic);
        const int terminal = pos == len - 1;
        da = -dbwd(X + oa * N, col + oa * N, n, nb, terminal);
        db = -dbwd(X + ob * N, col + ob * N, n, nb, terminal);
    } else {
        const long long nb = fwd(n, pos, len, stride, periodic);
        da = dfwd(X + oa * N, col + oa * N, n, nb);
        db = dfwd(X + ob * N, col + ob * N, n, nb);
    }
    const real ka = kap[axis * N + n];
    da *= g[ca * N + n] * ka;
    db *= g[cb * N + n] * ka;
    if (use_t) {
        da += t_a[n];
        db += t_b[n];
    }

    psi_a[p] = pb[p] * psi_a[p] + pc[p] * da;
    psi_b[p] = pb[p] * psi_b[p] + pc[p] * db;
    F[ca * N + n] += scale * mat[ca * N + n] * psi_a[p];
    F[cb * N + n] -= scale * mat[cb * N + n] * psi_b[p];
}

#ifndef KAPPA
// Coefficient access of fused_step. Full: one vector per coefficient.
// COMPACT: staircase FIT metrics are tensor products, so L_c and tL_c are
// 1-D tables along axis c, iA_c and itA_c 2-D tables over the two other
// axes b = c+1, d = c+2 (index pos_b + len_b * pos_d), and the material of
// entry (c, n) is a 1-byte id into tables of mu^-1, eps^-1 * BC and eps^-1
// plus the 0/1 mask BC. Both forms give bitwise the same coefficients.
#ifdef COMPACT
struct Coefs {
    const unsigned char* __restrict__ id; const real* __restrict__ T; const unsigned char* __restrict__ bc;
    const real* __restrict__ L; const real* __restrict__ tL; const real* __restrict__ iA; const real* __restrict__ itA;
    long long ntab, N, len[3], off1[3], off2[3];
};
__device__ inline long long face(const Coefs& K, int c, const long long* pos)
{
    const int b = (c + 1) % 3, d = (c + 2) % 3;
    return K.off2[c] + pos[b] + K.len[b] * pos[d];
}
__device__ inline real k_rh(const Coefs& K, int c, long long n, const long long* pos)
{ return K.iA[face(K, c, pos)] * K.T[K.id[c * K.N + n]]; }
__device__ inline real k_ch(const Coefs& K, int c, long long n, const long long* pos)
{ return K.bc[K.id[c * K.N + n]] ? K.L[K.off1[c] + pos[c]] : (real)0; }
__device__ inline real k_re(const Coefs& K, int c, long long n, const long long* pos)
{ return K.itA[face(K, c, pos)] * K.T[K.ntab + K.id[c * K.N + n]]; }
__device__ inline real k_ce(const Coefs& K, int c, long long n, const long long* pos)
{ return K.tL[K.off1[c] + pos[c]]; }
__device__ inline real k_ieps(const Coefs& K, int c, long long n, const long long* pos)
{ return K.T[2 * K.ntab + K.id[c * K.N + n]]; }
#define COEF_PARAMS \
    const unsigned char* __restrict__ id, const real* __restrict__ T, const unsigned char* __restrict__ bc, \
    const long long ntab, const real* __restrict__ L, const real* __restrict__ tL, const real* __restrict__ iA, \
    const real* __restrict__ itA
#define COEF_INIT \
    Coefs K = {id, T, bc, L, tL, iA, itA, ntab, N, {Nx, Ny, Nz}, \
               {0, Nx, Nx + Ny}, {0, Ny * Nz, Ny * Nz + Nz * Nx}};
#else
struct Coefs { const real *__restrict__ rh, *__restrict__ ch, *__restrict__ re, *__restrict__ ce, *__restrict__ ieps; long long N; };
__device__ inline real k_rh(const Coefs& K, int c, long long n, const long long* pos)
{ return K.rh[c * K.N + n]; }
__device__ inline real k_ch(const Coefs& K, int c, long long n, const long long* pos)
{ return K.ch[c * K.N + n]; }
__device__ inline real k_re(const Coefs& K, int c, long long n, const long long* pos)
{ return K.re[c * K.N + n]; }
__device__ inline real k_ce(const Coefs& K, int c, long long n, const long long* pos)
{ return K.ce[c * K.N + n]; }
__device__ inline real k_ieps(const Coefs& K, int c, long long n, const long long* pos)
{ return K.ieps[c * K.N + n]; }
#define COEF_PARAMS \
    const real* __restrict__ rh, const real* __restrict__ ch, const real* __restrict__ re, const real* __restrict__ ce, \
    const real* __restrict__ ieps
#define COEF_INIT Coefs K = {rh, ch, re, ce, ieps, N};
#endif

// (P_a (c_h x_s))[n] for component s: as dfwd, with c_h from the accessor.
// The curl differentiates component s only along axes other than s, so c_h
// of s at the forward neighbour nb uses the edge length at n: `pos` (the
// coordinates of n) serves both nodes.
//
// Tuning, measured on an RTX 5060 Ti (sm_120), 256^3: inlining dfwd_k into
// the full variant doubles its registers (~210) and costs up to 15%, so it
// stays a real call there. The compact variant is fastest inlined, with the
// register cap of four 256-thread blocks per SM (see fused_step).
#ifndef DFWD_ATTR
#ifdef COMPACT
#define DFWD_ATTR inline
#else
#define DFWD_ATTR __noinline__
#endif
#endif
__device__ DFWD_ATTR real dfwd_k(const real* __restrict__ x, const Coefs& K, int s,
                       long long n, long long nb, const long long* pos)
{
    return nb < 0 ? (real)0 : k_ch(K, s, nb, pos) * x[nb] - k_ch(K, s, n, pos) * x[n];
}

// Next H component c at node p: H - dt * r_h C0 (c_h E), row c of C0 being
// +P_{c+1} E_{c+2} - P_{c+2} E_{c+1} (axes and components mod 3).
__device__ real h_next(const real* __restrict__ E, const real* __restrict__ H,
                       const Coefs& K, int c, long long p, const long long* pos,
                       const long long* len, const long long* stride,
                       const int* per, real dt)
{
    const long long N = K.N;
    const int ap = (c + 1) % 3, an = (c + 2) % 3;
    const long long fp = fwd(p, pos[ap], len[ap], stride[ap], per[ap]);
    const long long fn = fwd(p, pos[an], len[an], stride[an], per[an]);
    const real curl = dfwd_k(E + an * N, K, an, p, fp, pos)
                    - dfwd_k(E + ap * N, K, ap, p, fn, pos);
    return H[c * N + p] - dt * (k_rh(K, c, p, pos) * curl);
}

// All pointers are __restrict__: inputs and outputs are distinct buffers,
// and the uint8 material ids (char may alias anything) need not be reloaded
// after every store. FUSED_MIN_BLOCKS caps the registers so that this many
// 256-thread blocks fit per SM; uncapped, the compiler uses up to ~230
// registers and one block per SM keeps too few loads in flight.
#ifndef FUSED_MIN_BLOCKS
#ifdef COMPACT
#define FUSED_MIN_BLOCKS 4
#else
#define FUSED_MIN_BLOCKS 3
#endif
#endif
extern "C" __global__ void __launch_bounds__(256, FUSED_MIN_BLOCKS) fused_step(
    const real* __restrict__ E, const real* __restrict__ H, real* __restrict__ Eo, real* __restrict__ Ho, COEF_PARAMS,
    const real* __restrict__ J, const real dt,
    const long long Nx, const long long Ny, const long long Nz,
    const int px, const int py, const int pz, const long long chunk)
{
    const long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long j = (long long)blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    const long long k0 = (long long)blockIdx.z * chunk;
    const long long k1 = k0 + chunk < Nz ? k0 + chunk : Nz;
    const long long N = Nx * Ny * Nz;
    const long long len[3] = {Nx, Ny, Nz};
    const long long stride[3] = {1, Nx, Nx * Ny};
    const int per[3] = {px, py, pz};
    COEF_INIT

    real carry[3] = {(real)0, (real)0, (real)0};  // c_e * Ho at (i, j, k-1)
    int has_carry = 0;
    for (long long k = k0; k < k1; ++k) {
        const long long m = i + Nx * (j + Ny * k);
        const long long pos[3] = {i, j, k};

        real cm[3];  // c_e * Ho at m
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const real h = h_next(E, H, K, c, m, pos, len, stride, per, dt);
            Ho[c * N + m] = h;
            cm[c] = k_ce(K, c, m, pos) * h;
        }

        // D[a][s]: (P_a.T (c_e Ho_s))[m], i.e. dbwd along axis a of component s.
        real D[3][3];
#pragma unroll
        for (int a = 0; a < 3; ++a) {
            const long long nb = bwd(m, pos[a], len[a], stride[a], per[a]);
            const int terminal = pos[a] == len[a] - 1;
            long long npos[3] = {pos[0], pos[1], pos[2]};
            npos[a] = pos[a] > 0 ? pos[a] - 1 : len[a] - 2;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                if (s == a) continue;
                real d = terminal ? (real)0 : -cm[s];
                if (nb >= 0) {
                    if (a == 2 && has_carry && nb == m - stride[2])
                        d += carry[s];
                    else
                        // tL_s at nb equals tL_s at m (nb differs along a != s).
                        d += k_ce(K, s, nb, pos)
                           * h_next(E, H, K, s, nb, npos, len, stride, per, dt);
                }
                D[a][s] = d;
            }
        }

        // Row c of C0.T: +P_{c+2}.T Ho_{c+1} - P_{c+1}.T Ho_{c+2}.
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const int ap = (c + 1) % 3, an = (c + 2) % 3;
            const real curl = k_re(K, c, m, pos) * (D[an][ap] - D[ap][an]);
            Eo[c * N + m] = E[c * N + m]
                          + dt * (curl - k_ieps(K, c, m, pos) * J[c * N + m]);
        }

#pragma unroll
        for (int c = 0; c < 3; ++c) carry[c] = cm[c];
        has_carry = 1;
    }
}
#endif
