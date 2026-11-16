# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Compare the GPU curl backends of SolverFIT3D: sparse (cuSPARSE) vs. stencil.

The FIT time step is memory-bandwidth bound: each curl product touches every
field value once and does only a few flops with it. The ``'sparse'`` backend
streams the matrix values and indices from GPU memory on every product. The
matrix-free ``'stencil'`` backend reads only the field and two coefficient
vectors. This script times both backends on a PEC box and reports the
effective bandwidth of the curl product against the card's peak.

Usage: ``python 007_GPU_backend_benchmark.py [N_cells_per_axis] [float32|float64]``
"""

import sys
import time

import cupy as cp
import numpy as np

from wakis import GridFIT3D, SolverFIT3D

n_cells = int(sys.argv[1]) if len(sys.argv) > 1 else 128
dtype = np.dtype(sys.argv[2] if len(sys.argv) > 2 else "float64")
n_steps = 200


def gpu_time(function, repeats):
    """Mean GPU time of ``function()`` in seconds, measured with CUDA events."""
    function()  # warm-up: kernel compilation and memory pool growth
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        function()
    stop.record()
    stop.synchronize()
    return cp.cuda.get_elapsed_time(start, stop) / 1e3 / repeats


def curl_bytes(operator, size, itemsize):
    """Minimum bytes moved by one curl product (each array read/written once)."""
    vectors = 2 * 3 * size * itemsize  # read x, write out
    if hasattr(operator, "nnz"):  # CSR: values, column indices, row pointers
        return vectors + operator.nnz * (itemsize + 4) + (3 * size + 1) * 4
    return vectors + 2 * 3 * size * itemsize  # row and column coefficients


properties = cp.cuda.runtime.getDeviceProperties(0)
# memoryClockRate is in kHz and double data rate; 0 on some recent CUDA versions
peak = 2 * properties["memoryClockRate"] * 1e3 * properties["memoryBusWidth"] / 8
print(f"GPU: {properties['name'].decode()}, peak bandwidth ", end="")
print(f"{peak / 1e9:.0f} GB/s" if peak else "unknown")

grid = GridFIT3D(0, 1, 0, 1, 0, 1, n_cells, n_cells, n_cells, verbose=0)
initial = np.random.default_rng(0).normal(size=3 * grid.Nx * grid.Ny * grid.Nz)
results = {}
for backend in ("sparse", "stencil"):
    t0 = time.time()
    solver = SolverFIT3D(
        grid,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        use_gpu=True,
        gpu_backend=backend,
        dtype=dtype,
        verbose=0,
    )
    setup = time.time() - t0
    solver.E.fromarray(cp.asarray(initial, dtype=dtype))
    operator, x = solver.tDsiDmuiDaC, solver.E.array

    t_curl = gpu_time(lambda: operator * x, 50)
    t_step = gpu_time(solver.one_step, n_steps)
    bandwidth = curl_bytes(operator, solver.N, dtype.itemsize) / t_curl
    results[backend] = [field.toarray().get() for field in (solver.E, solver.H)]

    print(
        f"{backend:>8}: setup {setup:6.2f} s | curl {t_curl * 1e3:7.3f} ms "
        f"({bandwidth / 1e9:6.1f} GB/s"
        + (f", {100 * bandwidth / peak:4.1f}% of peak)" if peak else ")")
        + f" | step {t_step * 1e3:7.3f} ms ({solver.N / t_step / 1e6:7.1f} Mcells/s)"
    )
    del solver, operator, x
    cp.get_default_memory_pool().free_all_blocks()

# Same fields after the timed steps: the stencil reproduces the sparse operator.
for name, sparse, stencil in zip("EH", results["sparse"], results["stencil"]):
    error = np.abs(stencil - sparse).max() / np.abs(sparse).max()
    print(f"max relative difference in {name}: {error:.2e}")
