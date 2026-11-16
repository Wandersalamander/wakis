# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Compare the GPU backends of SolverFIT3D: sparse (cuSPARSE) vs. stencil.

The FIT time step is memory-bandwidth bound: it touches every field value a
few times and does only a few flops with it. The ``'sparse'`` backend streams
the curl matrices' values and indices from GPU memory twice per step, plus
the elementwise update passes. Without CPML, TFSF or MPI, the ``'stencil'``
backend runs the whole step as one fused kernel that reads E, H, J and, on
staircase grids with few materials, only 1-D/2-D metric tables and a 1-byte
material id per entry. This script times both backends on a PEC box and
reports the time per step, the GPU memory held by the solver, and for the
sparse backend the effective bandwidth of one curl product.

Usage: ``python 007_GPU_backend_benchmark.py [N_cells_per_axis] [float32|float64]``
"""

import gc
import sys
import time

import cupy as cp
import numpy as np

from wakis import GridFIT3D, SolverFIT3D

n_cells = int(sys.argv[1]) if len(sys.argv) > 1 else 128
dtype = np.dtype(sys.argv[2] if len(sys.argv) > 2 else "float64")
n_steps = 200


def gpu_time(function, repeats):
    """Mean GPU time of ``function()`` in seconds, measured with CUDA events.

    Warms up for 1 s first: after the CPU-bound solver setup the GPU is at
    idle clocks, which would distort short measurements.
    """
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        function()
        cp.cuda.Device().synchronize()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        function()
    stop.record()
    stop.synchronize()
    return cp.cuda.get_elapsed_time(start, stop) / 1e3 / repeats


def csr_curl_bytes(operator, size, itemsize):
    """Minimum bytes moved by one CSR curl product (each array read once)."""
    vectors = 2 * 3 * size * itemsize  # read x, write out
    return vectors + operator.nnz * (itemsize + 4) + (3 * size + 1) * 4


properties = cp.cuda.runtime.getDeviceProperties(0)
# memoryClockRate is in kHz and double data rate; 0 on some recent CUDA versions
peak = 2 * properties["memoryClockRate"] * 1e3 * properties["memoryBusWidth"] / 8
print(f"GPU: {properties['name'].decode()}, peak bandwidth ", end="")
print(f"{peak / 1e9:.0f} GB/s" if peak else "unknown")

grid = GridFIT3D(0, 1, 0, 1, 0, 1, n_cells, n_cells, n_cells, verbose=0)
initial = np.random.default_rng(0).normal(size=3 * grid.Nx * grid.Ny * grid.Nz)
pool = cp.get_default_memory_pool()
results = {}
for backend in ("sparse", "stencil"):
    t0 = time.time()
    solver = SolverFIT3D(
        grid,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        use_gpu=True,
        gpu_backend=backend,
        dtype=dtype.type,  # SolverFIT3D expects a scalar type, not np.dtype
        verbose=0,
    )
    setup = time.time() - t0
    solver.E.fromarray(cp.asarray(initial, dtype=dtype))

    # Fixed step count before the timing, so both backends end at the same
    # time level and their fields can be compared below.
    for _ in range(n_steps):
        solver.one_step()
    memory = pool.used_bytes()  # after the first steps allocated everything
    results[backend] = [field.toarray().get() for field in (solver.E, solver.H)]
    t_step = gpu_time(solver.one_step, n_steps)

    line = (
        f"{backend:>8}: setup {setup:6.2f} s | GPU memory {memory / 1e9:5.2f} GB"
        f" | step {t_step * 1e3:7.3f} ms ({solver.N / t_step / 1e6:7.1f} Mcells/s)"
    )
    operator = solver.tDsiDmuiDaC
    if hasattr(operator, "nnz"):
        x = solver.E.array
        t_curl = gpu_time(lambda: operator * x, 200)
        bandwidth = csr_curl_bytes(operator, solver.N, dtype.itemsize) / t_curl
        line += f" | curl {t_curl * 1e3:7.3f} ms ({bandwidth / 1e9:6.1f} GB/s"
        line += f", {100 * bandwidth / peak:4.1f}% of peak)" if peak else ")"
    print(line)
    del solver, operator
    gc.collect()  # the solver holds reference cycles (bound step methods)
    pool.free_all_blocks()

# Same fields after the same number of steps: the stencil reproduces sparse.
for name, sparse, stencil in zip("EH", results["sparse"], results["stencil"]):
    error = np.abs(stencil - sparse).max() / np.abs(sparse).max()
    print(f"max relative difference in {name}: {error:.2e}")
