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
