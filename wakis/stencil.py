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
instead of the sparse matrix values and indices. The CUDA kernels live in
``stencil_kernels.cu``; their ``fwd``/``bwd`` neighbour rules must stay
consistent with :func:`_incidence`, which :func:`check_curl_stencil` compares
against the assembled solver matrices.

With CPML, the same kernels apply the coordinate stretching 1/kappa per
derivative direction, and ``cpml_psi`` updates the convolution fields only on
the boundary slabs, replacing the 12 directional operators ``dxy``..``dtzy``.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags, hstack, vstack

_THREADS_PER_BLOCK = 256
# fused_step: (x, y) thread block; each thread marches _FUSED_CHUNK z-planes.
_FUSED_BLOCK = (32, 8, 1)
_FUSED_CHUNK = 16
_AXES = "xyz"


def curl_stencil_coefficients(solver):
    """Return the diagonal scalings of the FIT leapfrog operators.

    Must be called after boundary assembly and before the first time step,
    while ``solver.BC`` (PEC/periodic mask) and the CPML stretching
    ``ikappa``/``itkappa`` still exist.

    Parameters
    ----------
    solver : SolverFIT3D
        Solver with CPU-resident metric and material ``Field`` objects.

    Returns
    -------
    dict of ndarray
        Vectors of length ``3*N``: ``r_h, c_h`` and ``r_e, c_e`` such that
        ``tDsiDmuiDaC = diag(r_h) C0 diag(c_h)`` and
        ``itDaiDepsDstC = diag(r_e) C0.T diag(c_e)``. With CPML also the
        material-free row scalings ``g_h, g_e`` and the inverse stretchings
        ``k_h = 1/tkappa`` (H update) and ``k_e = 1/kappa`` (E update).
    """
    mask = solver.BC.toarray() if hasattr(solver, "BC") else 1.0
    coefficients = {
        "r_h": solver.iA.toarray() * solver.imu.toarray(),
        "c_h": mask * solver.L.toarray(),
        "r_e": solver.ieps.toarray() * solver.itA.toarray() * mask,
        "c_e": solver.tL.toarray(),
    }
    if solver.activate_cpml:
        # The CPML directional derivatives exclude the material factor, which
        # the update multiplies in separately.
        coefficients.update(
            g_h=solver.iA.toarray(),
            g_e=solver.itA.toarray() * mask,
            k_h=solver.itkappa,
            k_e=solver.ikappa,
        )
    return coefficients


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


def _stencil_operators(solver, coefficients):
    """Yield (name, assembled operator, stencil rebuild) pairs to compare."""
    shape = (solver.Nx, solver.Ny, solver.Nz)
    periodic = periodic_axes(solver)
    if not solver.activate_cpml:
        curl = _curl(shape, periodic)
        yield (
            "tDsiDmuiDaC",
            solver.tDsiDmuiDaC,
            diags(coefficients["r_h"]) @ curl @ diags(coefficients["c_h"]),
        )
        yield (
            "itDaiDepsDstC",
            solver.itDaiDepsDstC,
            diags(coefficients["r_e"]) @ curl.T @ diags(coefficients["c_e"]),
        )
        return

    # CPML: d{c}{a} maps component o = 3-c-a through the derivative along a
    # into component c, e.g. dxy = iA_x / tkappa_y * P_y * BC_z * L_z.
    N = solver.N

    def block(name, component):
        return coefficients[name][component * N : (component + 1) * N]

    for a in range(3):
        diff = _incidence(shape, a, periodic[a])
        for c in range(3):
            if c == a:
                continue
            o = 3 - a - c
            name = _AXES[c] + _AXES[a]
            row_h = block("g_h", c) * block("k_h", a)
            row_e = block("g_e", c) * block("k_e", a)
            yield (
                "d" + name,
                getattr(solver, "d" + name),
                diags(row_h) @ diff @ diags(block("c_h", o)),
            )
            yield (
                "dt" + name,
                getattr(solver, "dt" + name),
                diags(row_e) @ -diff.T @ diags(block("c_e", o)),
            )


def periodic_axes(solver):
    """Return the per-axis periodic flags used by the stencil kernels."""
    return tuple(bc.lower() == "periodic" for bc in solver.bc_low)


def check_curl_stencil(solver, coefficients, rtol=None):
    """Verify that the assembled operators are the pure curl stencil.

    This is the guard that lets the sparse assembly remain the source of
    truth: if a future boundary or material model adds a term that is not a
    diagonal scaling of the curl, the stencil backend refuses to run instead
    of silently dropping it. Without CPML it checks ``tDsiDmuiDaC`` and
    ``itDaiDepsDstC``; with CPML the 12 directional operators ``dxy``..``dtzy``.

    Parameters
    ----------
    solver : SolverFIT3D
        Solver whose operators are still SciPy sparse matrices.
    coefficients : dict of ndarray
        Output of :func:`curl_stencil_coefficients`.
    rtol : float, optional
        Tolerance relative to the largest operator entry. The two
        representations differ only by floating-point product order, so the
        default is ``max(1e-12, 16 * eps)`` of the operator dtype (float32
        operators differ by ~1 ulp). A missing or extra term is an O(1)
        relative deviation and is still caught.

    Raises
    ------
    ValueError
        If an operator differs from its stencil factorization.
    """
    for name, operator, rebuilt in _stencil_operators(solver, coefficients):
        operator = operator.tocsr()  # CPML operators are stored as DIA
        tol = rtol
        if tol is None:
            tol = max(1e-12, 16 * float(np.finfo(operator.dtype).eps))
        scale = abs(operator).max()
        error = abs(operator - rebuilt).max()
        if error > tol * scale:
            raise ValueError(
                f"{name} is not a diagonally scaled curl stencil (max deviation "
                f"{error:.3e} vs. scale {scale:.3e}). The boundary or material "
                "setup added a term the GPU stencil backend does not support; "
                "use gpu_backend='sparse' or extend wakis/stencil.py."
            )


def compact_coefficients(solver, coefficients):
    """Return the fused-step coefficients in compact form, or None.

    On the staircase FIT grid, the edge lengths ``L_c``/``tL_c`` vary only
    along axis ``c`` and the inverse face areas ``iA_c``/``itA_c`` only along
    the two other axes. Materials and the PEC/periodic mask take few distinct
    values. The kernel then rebuilds every coefficient from 1-D and 2-D metric
    tables and a 1-byte material id per entry instead of reading five full
    vectors, with the same products (see ``stencil_kernels.cu``).

    Parameters
    ----------
    solver : SolverFIT3D
        Solver before the first time step, with CPU-resident ``Field`` data.
    coefficients : dict of ndarray
        Output of :func:`curl_stencil_coefficients`, used to verify the
        compact form.

    Returns
    -------
    dict of ndarray or None
        ``L``, ``tL`` (concatenated 1-D tables), ``iA``, ``itA``
        (concatenated 2-D tables), ``ids`` (uint8, length ``3*N``),
        ``table`` (rows mu^-1, eps^-1 * BC, eps^-1) and ``bc`` (uint8 mask
        per table entry). None when the metrics are not tensor products or
        there are more than 256 material combinations.

    Raises
    ------
    ValueError
        If the compact form does not reproduce ``coefficients``.
    """
    shape = (solver.Nx, solver.Ny, solver.Nz)
    N = solver.N
    tables = {}
    for name, planar in (("L", False), ("tL", False), ("iA", True), ("itA", True)):
        values = getattr(solver, name).toarray()
        parts = []
        for c in range(3):
            block = values[c * N : (c + 1) * N].reshape(shape, order="F")
            keep = sorted(((c + 1) % 3, (c + 2) % 3)) if planar else [c]
            for axis in set(range(3)) - set(keep):
                if np.ptp(block, axis=axis).any():
                    return None
            part = block[tuple(slice(None) if a in keep else 0 for a in range(3))]
            if planar and (c + 1) % 3 > (c + 2) % 3:
                part = part.T  # index pos_b + len_b * pos_d with b = c + 1
            parts.append(part.ravel(order="F"))
        tables[name] = np.concatenate(parts)

    mask = solver.BC.toarray() if hasattr(solver, "BC") else np.ones(3 * N)
    if not np.isin(mask, (0, 1)).all():
        return None
    imu, ieps = solver.imu.toarray(), solver.ieps.toarray()
    # Combined integer code of (mu^-1, eps^-1, BC): unique on integers is fast.
    mu_values, mu_code = np.unique(imu, return_inverse=True)
    eps_values, eps_code = np.unique(ieps, return_inverse=True)
    code = (mu_code.astype(np.int64) * eps_values.size + eps_code) * 2
    code += mask.astype(np.int64)
    codes, ids = np.unique(code, return_inverse=True)
    if codes.size > 256:
        return None
    entry_mu = mu_values[codes // 2 // eps_values.size]
    entry_eps = eps_values[codes // 2 % eps_values.size]
    entry_bc = (codes % 2).astype(np.uint8)
    compact = dict(
        tables,
        ids=ids.ravel().astype(np.uint8),
        table=np.stack([entry_mu, entry_eps * entry_bc, entry_eps]),
        bc=entry_bc,
    )
    _verify_compact(solver, compact, coefficients)
    return compact


def _verify_compact(solver, compact, coefficients):
    """Rebuild the full coefficient vectors from ``compact`` and compare."""
    shape = (solver.Nx, solver.Ny, solver.Nz)
    index = np.indices(shape).reshape(3, -1, order="F")
    lengths = np.array(shape)
    off1 = np.r_[0, np.cumsum(lengths)[:2]]
    faces = [lengths[(c + 1) % 3] * lengths[(c + 2) % 3] for c in range(3)]
    off2 = np.r_[0, np.cumsum(faces)[:2]]
    ids, table, bc = compact["ids"], compact["table"], compact["bc"]
    rebuilt = {key: [] for key in ("r_h", "c_h", "r_e", "c_e")}
    for c in range(3):
        b, d = (c + 1) % 3, (c + 2) % 3
        face = off2[c] + index[b] + lengths[b] * index[d]
        line = off1[c] + index[c]
        entry = ids[c * solver.N : (c + 1) * solver.N]
        rebuilt["r_h"].append(compact["iA"][face] * table[0][entry])
        rebuilt["c_h"].append(np.where(bc[entry], compact["L"][line], 0))
        rebuilt["r_e"].append(compact["itA"][face] * table[1][entry])
        rebuilt["c_e"].append(compact["tL"][line])
    for key, parts in rebuilt.items():
        expected = coefficients[key]
        error = np.abs(np.concatenate(parts) - expected).max()
        tol = max(1e-12, 16 * float(np.finfo(np.asarray(expected).dtype).eps))
        if error > tol * np.abs(expected).max():
            raise ValueError(f"Compact stencil coefficients do not reproduce {key}.")


def cpml_faces(solver):
    """Return, per CPML face, its axis and the solver attribute names it uses.

    Low faces share the slab indices of H and E; high H slabs exclude the
    terminal storage plane (``idx_H_*_high``).
    """
    faces = []
    for side, bcs in (("low", solver.bc_low), ("high", solver.bc_high)):
        for axis, name in enumerate(_AXES):
            if bcs[axis].lower() != "cpml":
                continue
            ca, cb = _AXES[(axis + 2) % 3], _AXES[(axis + 1) % 3]
            idx_h = f"idx_H_{name}_high" if side == "high" else f"idx_{name}_low"
            faces.append(
                {
                    "axis": axis,
                    "H": {
                        "idx": idx_h,
                        "psi_a": f"psiHa_{ca}_{side}",
                        "psi_b": f"psiHb_{cb}_{side}",
                        "b": f"pml_b_H_{name}_{side}",
                        "c": f"pml_c_H_{name}_{side}",
                    },
                    "E": {
                        "idx": f"idx_{name}_{side}",
                        "psi_a": f"psiEa_{ca}_{side}",
                        "psi_b": f"psiEb_{cb}_{side}",
                        "b": f"pml_b_E_{name}_{side}",
                        "c": f"pml_c_E_{name}_{side}",
                    },
                }
            )
    return faces


@lru_cache(maxsize=None)
def _module(dtype, kappa, compact=False):
    import cupy as cp

    ctype = {np.dtype(np.float32): "float", np.dtype(np.float64): "double"}
    if dtype not in ctype:
        raise TypeError(f"Stencil backend supports float32/float64, not {dtype}.")
    prelude = f"typedef {ctype[dtype]} real;\n"
    prelude += "#define KAPPA\n" if kappa else ""
    prelude += "#define COMPACT\n" if compact else ""
    source = (Path(__file__).parent / "stencil_kernels.cu").read_text()
    return cp.RawModule(code=prelude + source)


def _blocks(size):
    return ((size + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK,)


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
    kappa : ndarray, optional
        CPML inverse stretching per derivative direction (``k_h`` or
        ``k_e``); None without CPML.
    """

    def __init__(self, shape, row, col, periodic, transpose, dtype, kappa=None):
        import cupy as cp

        self.dtype = np.dtype(dtype)
        module = _module(self.dtype, kappa is not None)
        self._kernel = module.get_function("curl_bwd" if transpose else "curl_fwd")
        self._cp = cp
        self.size = int(np.prod(shape))
        self.row = cp.asarray(row, dtype=self.dtype)
        self.col = cp.asarray(col, dtype=self.dtype)
        # Unused by the kernel without KAPPA; any valid pointer will do.
        self.kappa = self.row if kappa is None else cp.asarray(kappa, self.dtype)
        self._args = (
            *(np.int64(s) for s in shape),
            *(np.int32(p) for p in periodic),
        )

    def __matmul__(self, x):
        out = self._cp.empty(3 * self.size, dtype=self.dtype)
        x = self._cp.ascontiguousarray(x, dtype=self.dtype)
        self._kernel(
            _blocks(self.size),
            (_THREADS_PER_BLOCK,),
            (x, self.row, self.col, self.kappa, out, *self._args),
        )
        return out

    __mul__ = __matmul__


class CPMLStencil:
    """GPU CPML convolution updates on the boundary slabs.

    Replaces the psi recursions and boundary corrections of
    ``SolverFIT3D._one_step_cpml``. The directional derivatives are
    recomputed from the fields on the slab nodes only, so the 12 directional
    operators are not needed. The solver's psi attributes are replaced by the
    GPU arrays that this object updates in place.

    Parameters
    ----------
    solver : SolverFIT3D
        Solver with CPML faces, before the first time step.
    coefficients : dict of ndarray
        Output of :func:`curl_stencil_coefficients`.
    """

    def __init__(self, solver, coefficients):
        import cupy as cp

        self.dtype = np.dtype(solver.dtype)
        self._kernel = _module(self.dtype, True).get_function("cpml_psi")

        def upload(values):
            return cp.asarray(values, dtype=self.dtype)

        self._operands = {
            "H": tuple(upload(coefficients[k]) for k in ("g_h", "k_h", "c_h")),
            "E": tuple(upload(coefficients[k]) for k in ("g_e", "k_e", "c_e")),
        }
        self._shape = tuple(np.int64(s) for s in (solver.Nx, solver.Ny, solver.Nz))
        self._periodic = tuple(np.int32(p) for p in periodic_axes(solver))
        self.faces = []
        for face in cpml_faces(solver):
            entry = {"axis": face["axis"]}
            for field in "HE":
                names = face[field]
                arrays = {
                    key: upload(getattr(solver, name))
                    for key, name in names.items()
                    if key != "idx"
                }
                arrays["idx"] = cp.asarray(getattr(solver, names["idx"]), cp.int64)
                # The solver keeps referring to the live psi state.
                for key in ("psi_a", "psi_b"):
                    setattr(solver, names[key], arrays[key])
                entry[field] = arrays
            self.faces.append(entry)

    def _apply(self, field, target, source, material, scale, t_a, t_b):
        g, kappa, col = self._operands[field]
        for face in self.faces:
            arrays = face[field]
            # TFSF only modifies derivatives along z (dxz, dyz, dtxz, dtyz).
            use_t = t_a is not None and face["axis"] == 2
            size = arrays["idx"].size
            self._kernel(
                _blocks(size),
                (_THREADS_PER_BLOCK,),
                (
                    target,
                    source,
                    g,
                    kappa,
                    col,
                    material,
                    arrays["idx"],
                    arrays["psi_a"],
                    arrays["psi_b"],
                    arrays["b"],
                    arrays["c"],
                    t_a if use_t else g,
                    t_b if use_t else g,
                    np.int64(size),
                    *self._shape,
                    np.int32(face["axis"]),
                    np.int32(field == "E"),
                    *self._periodic,
                    np.int32(use_t),
                    self.dtype.type(scale),
                ),
            )

    def update_H(self, H, E, imu, dt, t_a=None, t_b=None):
        """Advance the H-side psi fields and add their correction to ``H``.

        ``t_a``/``t_b`` are the TFSF terms added to the z-derivatives
        ``dyzEx`` and ``dxzEy``; None when no injection is active.
        """
        self._apply("H", H, E, imu, -dt, t_a, t_b)

    def update_E(self, E, H, ieps, dt, t_a=None, t_b=None):
        """Advance the E-side psi fields and add their correction to ``E``.

        ``t_a``/``t_b`` are the TFSF terms added to ``dtyzHx`` and
        ``dtxzHy``; None when no injection is active.
        """
        self._apply("E", E, H, ieps, dt, t_a, t_b)


class FusedStencilStep:
    """Single-pass GPU leapfrog step (H then E) for the non-CPML path.

    Launches ``fused_step`` once per time step instead of two curl kernels
    plus the elementwise update passes. Each thread recomputes the new H at
    its backward neighbours, so the kernel needs distinct input and output
    buffers: :meth:`__call__` writes into spare arrays and swaps them into
    ``E.array``/``H.array``. References to a field array taken before a step
    therefore point at the spare buffer afterwards.

    Parameters
    ----------
    shape : tuple of int
        Grid shape ``(Nx, Ny, Nz)``.
    periodic : tuple of bool
        Periodic flag per axis.
    dtype : numpy dtype
        Field dtype; must be float32 or float64.
    full : tuple of cupy.ndarray, optional
        ``(r_h, c_h, r_e, c_e)`` coefficient vectors on the GPU; ``ieps`` is
        then passed to :meth:`__call__`.
    compact : dict of ndarray, optional
        Output of :func:`compact_coefficients`. Exactly one of ``full`` and
        ``compact`` must be given.
    chunk : int, optional
        Number of z-planes each thread marches through.
    """

    def __init__(
        self, shape, periodic, dtype, full=None, compact=None, chunk=_FUSED_CHUNK
    ):
        import cupy as cp

        if (full is None) == (compact is None):
            raise ValueError("Pass exactly one of `full` and `compact`.")
        self._cp = cp
        self.dtype = np.dtype(dtype)
        self.compact = compact is not None
        module = _module(self.dtype, False, self.compact)
        self._kernel = module.get_function("fused_step")
        if self.compact:
            self._operands = (
                cp.asarray(compact["ids"]),
                cp.asarray(compact["table"], self.dtype).ravel(),
                cp.asarray(compact["bc"]),
                np.int64(compact["table"].shape[1]),
                *(cp.asarray(compact[k], self.dtype) for k in ("L", "tL", "iA", "itA")),
            )
        else:
            self._operands = tuple(full)
        self.chunk = chunk
        self._grid = (
            -(-shape[0] // _FUSED_BLOCK[0]),
            -(-shape[1] // _FUSED_BLOCK[1]),
            -(-shape[2] // chunk),
        )
        self._args = (
            *(np.int64(s) for s in shape),
            *(np.int32(p) for p in periodic),
            np.int64(chunk),
        )
        self._spare = None

    def __call__(self, E, H, ieps, J, dt):
        """Advance ``E`` and ``H`` (wakis ``Field`` objects) by one step.

        ``ieps`` is the GPU vector of eps^-1; only the ``full`` form reads it.
        """
        cp = self._cp
        if self._spare is None:
            self._spare = (cp.empty_like(E.array), cp.empty_like(H.array))
        E_out, H_out = self._spare
        operands = self._operands if self.compact else (*self._operands, ieps)
        self._kernel(
            self._grid,
            _FUSED_BLOCK,
            (
                E.array,
                H.array,
                E_out,
                H_out,
                *operands,
                J,
                self.dtype.type(dt),
                *self._args,
            ),
        )
        self._spare = (E.array, H.array)
        E.array, H.array = E_out, H_out
