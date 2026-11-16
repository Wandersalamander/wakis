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
// Next H component c at node p: H - dt * r_h C0 (c_h E), row c of C0 being
// +P_{c+1} E_{c+2} - P_{c+2} E_{c+1} (axes and components mod 3).
__device__ real h_next(const real* E, const real* H, const real* rh,
                       const real* ch, int c, long long p, const long long* pos,
                       const long long* len, const long long* stride,
                       const int* per, long long N, real dt)
{
    const int ap = (c + 1) % 3, an = (c + 2) % 3;
    const long long fp = fwd(p, pos[ap], len[ap], stride[ap], per[ap]);
    const long long fn = fwd(p, pos[an], len[an], stride[an], per[an]);
    const real curl = dfwd(E + an * N, ch + an * N, p, fp)
                    - dfwd(E + ap * N, ch + ap * N, p, fn);
    return H[c * N + p] - dt * (rh[c * N + p] * curl);
}

// One complete leapfrog step in a single pass (no CPML):
//   Ho = H - dt * r_h C0 (c_h E)
//   Eo = E + dt * (r_e C0.T (c_e Ho) - ieps J)
// C0.T at node m needs Ho at m and at its backward neighbour along each
// axis. Each thread owns one (i, j) column and marches k over one chunk of
// z-planes: Ho at (i, j, k-1) is carried in registers from the previous
// iteration, the in-plane neighbours (i-1, j, k), (i, j-1, k) and chunk
// starts or periodic wraps are recomputed from E. No thread depends on
// another's output, so inputs and outputs must be distinct buffers. The
// z-march keeps the working set to about one plane per array in cache.
extern "C" __global__ void fused_step(
    const real* E, const real* H, real* Eo, real* Ho,
    const real* rh, const real* ch, const real* re, const real* ce,
    const real* ieps, const real* J, const real dt,
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

    real carry[3] = {(real)0, (real)0, (real)0};  // c_e * Ho at (i, j, k-1)
    int has_carry = 0;
    for (long long k = k0; k < k1; ++k) {
        const long long m = i + Nx * (j + Ny * k);
        const long long pos[3] = {i, j, k};

        real cm[3];  // c_e * Ho at m
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const real h = h_next(E, H, rh, ch, c, m, pos, len, stride, per, N, dt);
            Ho[c * N + m] = h;
            cm[c] = ce[c * N + m] * h;
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
                        d += ce[s * N + nb]
                           * h_next(E, H, rh, ch, s, nb, npos, len, stride, per, N, dt);
                }
                D[a][s] = d;
            }
        }

        // Row c of C0.T: +P_{c+2}.T Ho_{c+1} - P_{c+1}.T Ho_{c+2}.
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const int ap = (c + 1) % 3, an = (c + 2) % 3;
            const real curl = re[c * N + m] * (D[an][ap] - D[ap][an]);
            Eo[c * N + m] = E[c * N + m] + dt * (curl - ieps[c * N + m] * J[c * N + m]);
        }

#pragma unroll
        for (int c = 0; c < 3; ++c) carry[c] = cm[c];
        has_carry = 1;
    }
}
#endif
