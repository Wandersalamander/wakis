# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Matrix-free FIT curl operators for the GPU stencil backend.

The FIT leapfrog operators factor into a fixed +-1 curl stencil between two
diagonal scalings (see ``docs/physicsguide.md``, *Matrix-free GPU stencil*):

    tDsiDmuiDaC   = diag(iA * imu)       @ C0   @ diag(BC * L)
    itDaiDepsDstC = diag(ieps * itA * BC) @ C0.T @ diag(tL)

``C0`` is the unmasked curl built from forward differences ``P_x, P_y, P_z``.
All metrics, materials, PEC masks and periodic seams live in the two
diagonals, so a GPU kernel only reads the field and two coefficient vectors
instead of the sparse matrix values and indices.
"""

import numpy as np
from scipy.sparse import coo_matrix, diags, hstack, vstack

# One thread per primal node computes all three components of
# r * C0 @ (c * x) (forward) or r * C0.T @ (c * x) (transposed).
# Node index n = i + Nx*j + Nx*Ny*k (x fastest, as in Field); component d is
# stored at d*N + n. `fwd`/`bwd` must stay consistent with `_incidence`, which
# `check_curl_stencil` compares against the assembled solver matrices.
_KERNEL_SOURCE = r"""
typedef REAL real;

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

extern "C" __global__ void curl_fwd(
    const real* x, const real* r, const real* c, real* out,
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
    out[n] = r[n] * (dfwd(x2, c2, n, ny) - dfwd(x1, c1, n, nz));
    out[N + n] = r[N + n] * (dfwd(x0, c0, n, nz) - dfwd(x2, c2, n, nx));
    out[2 * N + n] = r[2 * N + n] * (dfwd(x1, c1, n, nx) - dfwd(x0, c0, n, ny));
}

extern "C" __global__ void curl_bwd(
    const real* x, const real* r, const real* c, real* out,
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
    out[m] = r[m] * (dbwd(x1, c1, m, nz, tz) - dbwd(x2, c2, m, ny, ty));
    out[N + m] = r[N + m] * (dbwd(x2, c2, m, nx, tx) - dbwd(x0, c0, m, nz, tz));
    out[2 * N + m] = r[2 * N + m] * (dbwd(x0, c0, m, ny, ty) - dbwd(x1, c1, m, nx, tx));
}
"""

_THREADS_PER_BLOCK = 256


def curl_stencil_coefficients(solver):
    """Return the diagonal scalings of the two FIT leapfrog operators.

    Must be called after boundary assembly and before the first time step,
    while ``solver.BC`` (PEC/periodic mask) still exists.

    Parameters
    ----------
    solver : SolverFIT3D
        Solver with CPU-resident metric and material ``Field`` objects.

    Returns
    -------
    dict of ndarray
        ``r_h, c_h`` and ``r_e, c_e``, each of length ``3*N``, such that
        ``tDsiDmuiDaC = diag(r_h) C0 diag(c_h)`` and
        ``itDaiDepsDstC = diag(r_e) C0.T diag(c_e)``.
    """
    mask = solver.BC.toarray() if hasattr(solver, "BC") else 1.0
    return {
        "r_h": solver.iA.toarray() * solver.imu.toarray(),
        "c_h": mask * solver.L.toarray(),
        "r_e": solver.ieps.toarray() * solver.itA.toarray() * mask,
        "c_e": solver.tL.toarray(),
    }


def _incidence(shape, axis, periodic):
    """Forward-difference matrix with the kernel's ``fwd`` neighbour rule."""
    n = np.arange(np.prod(shape))
    idx = np.unravel_index(n, shape, order="F")[axis]
    length, stride = shape[axis], int(np.prod(shape[:axis]))
    active = idx != length - 1
    neighbour = n + stride
    if periodic:
        wrap = idx == length - 2
        neighbour = np.where(wrap, n - idx * stride, neighbour)
    rows = np.r_[n[active], n[active]]
    cols = np.r_[n[active], neighbour[active]]
    values = np.r_[-np.ones(active.sum()), np.ones(active.sum())]
    # Duplicate (row, col) pairs sum to zero for 2-node periodic axes, as in
    # the assembled solver operator.
    matrix = coo_matrix((values, (rows, cols)), shape=(n.size, n.size)).tocsr()
    matrix.eliminate_zeros()
    return matrix


def _curl(shape, periodic):
    px, py, pz = (_incidence(shape, axis, periodic[axis]) for axis in range(3))
    zero = coo_matrix(px.shape)
    return vstack(
        [
            hstack([zero, -pz, py]),
            hstack([pz, zero, -px]),
            hstack([-py, px, zero]),
        ],
        format="csr",
    )


def periodic_axes(solver):
    """Return the per-axis periodic flags used by the stencil kernels."""
    return tuple(bc.lower() == "periodic" for bc in solver.bc_low)


def check_curl_stencil(solver, coefficients, rtol=1e-12):
    """Verify that the assembled operators are the pure curl stencil.

    This is the guard that lets the sparse assembly remain the source of
    truth: if a future boundary or material model adds a term that is not a
    diagonal scaling of the curl, the stencil backend refuses to run instead
    of silently dropping it.

    Parameters
    ----------
    solver : SolverFIT3D
        Solver whose ``tDsiDmuiDaC`` and ``itDaiDepsDstC`` are still SciPy
        sparse matrices.
    coefficients : dict of ndarray
        Output of :func:`curl_stencil_coefficients`.
    rtol : float, optional
        Tolerance relative to the largest operator entry. The two
        representations differ only by floating-point product order.

    Raises
    ------
    ValueError
        If either operator differs from its stencil factorization.
    """
    curl = _curl((solver.Nx, solver.Ny, solver.Nz), periodic_axes(solver))
    pairs = (
        ("tDsiDmuiDaC", solver.tDsiDmuiDaC, "r_h", curl, "c_h"),
        ("itDaiDepsDstC", solver.itDaiDepsDstC, "r_e", curl.T, "c_e"),
    )
    for name, operator, row, stencil, col in pairs:
        rebuilt = diags(coefficients[row]) @ stencil @ diags(coefficients[col])
        scale = abs(operator).max()
        error = abs(operator - rebuilt).max()
        if error > rtol * scale:
            raise ValueError(
                f"{name} is not a diagonally scaled curl stencil (max deviation "
                f"{error:.3e} vs. scale {scale:.3e}). The boundary or material "
                "setup added a term the GPU stencil backend does not support; "
                "use gpu_backend='sparse' or extend wakis/stencil.py."
            )


class CurlStencil:
    """GPU matrix-free replacement for one FIT leapfrog operator.

    Behaves like the sparse matrix it replaces for ``operator * x`` and
    ``operator @ x`` on flat CuPy field arrays of length ``3*N``.

    Parameters
    ----------
    shape : tuple of int
        Grid shape ``(Nx, Ny, Nz)``.
    row, col : ndarray
        Row and column scalings from :func:`curl_stencil_coefficients`.
    periodic : tuple of bool
        Periodic flag per axis.
    transpose : bool
        False for ``tDsiDmuiDaC`` (``C0``), True for ``itDaiDepsDstC``
        (``C0.T``).
    dtype : numpy dtype
        Field dtype; must be float32 or float64.
    """

    def __init__(self, shape, row, col, periodic, transpose, dtype):
        import cupy as cp

        ctype = {np.dtype(np.float32): "float", np.dtype(np.float64): "double"}
        dtype = np.dtype(dtype)
        if dtype not in ctype:
            raise TypeError(f"Stencil backend supports float32/float64, not {dtype}.")
        source = _KERNEL_SOURCE.replace("REAL", ctype[dtype])
        name = "curl_bwd" if transpose else "curl_fwd"
        self._kernel = cp.RawModule(code=source).get_function(name)
        self._cp = cp
        self.shape = tuple(int(s) for s in shape)
        self.size = int(np.prod(self.shape))
        self.row = cp.asarray(row, dtype=dtype)
        self.col = cp.asarray(col, dtype=dtype)
        self.dtype = dtype
        self._args = (
            *(np.int64(s) for s in self.shape),
            *(np.int32(p) for p in periodic),
        )
        self._grid = ((self.size + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK,)

    def __matmul__(self, x):
        out = self._cp.empty(3 * self.size, dtype=self.dtype)
        x = self._cp.ascontiguousarray(x, dtype=self.dtype)
        self._kernel(
            self._grid,
            (_THREADS_PER_BLOCK,),
            (x, self.row, self.col, out, *self._args),
        )
        return out

    __mul__ = __matmul__
