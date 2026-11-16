# GPU stencil backend: handoff notes

> Working notes for validating this branch on a GPU machine. **Delete this file before merging.**

Branch: `feature/gpu-stencil-backend`. Written on a machine without an NVIDIA GPU or CuPy, so **none of the GPU code has been executed yet**. Read `AGENTS.md` and `CONTRIBUTING.md` first.

## What the branch changes

1. **`gpu_backend='stencil'`** (new `SolverFIT3D` argument, default `'sparse'`). `wakis/stencil.py` replaces the two cupyx curl matrices (`tDsiDmuiDaC`, `itDaiDepsDstC`) with `CurlStencil` objects. These run matrix-free CUDA kernels (`cupy.RawModule`) and act as drop-in operators for `op * x`. Every existing step path (`_one_step`, `_mpi_one_step`, TFSF, conductivity, ABC and periodic wrappers) is unchanged. The math is in `docs/physicsguide.md`, section *Matrix-free GPU stencil*.
2. **Tripwire**: `check_curl_stencil` verifies at setup that the assembled matrices equal `diag(r) C0 diag(c)`, and raises otherwise. CPML raises (unsupported).
3. **Sparse path, also on CPU**: `_one_step` and `_mpi_one_step` now compute `dt * (K * x)` in place. Before, `self.dt * self.tDsiDmuiDaC * E` evaluated left to right and **copied and scaled the whole sparse matrix every step**.
4. **Sparse GPU path**: cupyx matrices are now CSR instead of CSC (`gpu_sparse_mat` import). This is unbenchmarked; revert that one line if it isn't faster.
5. `examples/007_GPU_backend_benchmark.py`, docs (`usersguide.md`, `physicsguide.md`, `wakis_stencil.rst`), and tests in `tests/test_016_boundary_conditions.py`.

## Already verified on CPU

- The CUDA kernel logic was translated line by line to NumPy and matched the SciPy operators to <1e-12 in 180 configurations. Those covered 5 grid shapes including 1- and 2-cell axes, every PEC/PMC/periodic/ABC combination per axis, graded meshes, and random materials with PEC cells. The script isn't committed; the `--gpu` test below replaces it.
- `test_curl_stencil_factorization` (5 BC sets, STL materials, graded mesh) and `test_curl_stencil_rejects_extra_terms` pass in CPU CI.
- The full CPU suite passes with the in-place step change (see the PR description for the exact result).

## To do on the GPU machine (in order)

```bash
pip install -e ".[test]" cupy-cuda12x   # match the CUDA version
python -m pytest tests/test_016_boundary_conditions.py -k stencil --gpu -v
```

1. **Equivalence test** `test_gpu_stencil_matches_sparse` covers 5 BC sets × float32/float64 and 50 steps, and compares fields against the sparse GPU backend. Likely failure modes and fixes:
   - Kernel compile error (NVRTC): fix the source in `_KERNEL_SOURCE`.
   - `cupyx csr_matrix(<scipy csc/dia>)` conversion error: convert explicitly with `.tocsr()` before `gpu_sparse_mat`.
   - Argument type mismatch: kernel parameters are `long long` (passed as `np.int64`) and `int` (`np.int32`).
   - Numerical mismatch: re-check `fwd`/`bwd` against `_incidence`. The CPU emulation passed, so suspect launch or pointer issues first.
2. **Full GPU suite** `python -m pytest --gpu -v`. This checks that the CSR switch and the in-place step didn't break the sparse GPU path (test_001, 005, 014 use `use_gpu`).
3. **End-to-end wake check with the stencil.** Temporarily pass `gpu_backend="stencil"` in `tests/test_001_pec_cubic_cavity.py::test_simulation` and run it with `--gpu`. It's float32 and compares wake potential and impedance with stored references. If it passes, consider parametrizing it permanently.
4. **Benchmark**: run `python examples/007_GPU_backend_benchmark.py 128 float64`, then again with `float32`, then at the largest size that fits (e.g. 256).
   - Record ms/step, Mcells/s and curl GB/s (% of peak) for both backends.
   - **Baseline from `main`**: the sparse path on this branch already includes fixes 3 and 4. For the honest before/after, run the same case on `main` (`git worktree add ../wakis-main main`). There, drop the `gpu_backend` argument and the `"stencil"` loop entry from a copy of the script.
   - Expectation (unverified): the stencil curl reaches well over half of peak bandwidth and should beat CSR SpMV by roughly 2×. Full-step speedup is smaller, because the elementwise update passes remain.
5. Write the PR with `.github/pull_request_template.md`. Report measured numbers, tested configurations, and the untested paths below honestly.

## If the stencil is slower than expected

- 64-bit index arithmetic: switch to `int` when `3*N < 2**31`.
- z-neighbour reads (stride `Nx*Ny`) miss the cache on large grids. Consider a 2D thread-block tiling or marching along z with a register queue (the standard FD-on-GPU technique).
- Check occupancy and launch size (`_THREADS_PER_BLOCK = 256`).

## Not supported or not tested

- **CPML** with `gpu_backend='stencil'` raises; it uses 12 split DIA operators plus ψ fields. Follow-up options: a dedicated stencil kernel, or capturing the many small launches in a CUDA Graph.
- **MPI + stencil** should work, since halo exchange acts on fields and the tripwire validates the local operators, but it is untested.
- **AMD ROCm**: CuPy compiles `RawModule` via hipRTC; the kernel uses no CUDA-only intrinsics, but it is untested.
- `update_tensors()` on GPU rebuilds SciPy matrices from GPU fields. That was already broken for the sparse GPU path and is not fixed here.

## Optional follow-ups (only if benchmarks justify them)

- **Fused update kernel**: compute `H -= dt * K_h E` and `E += dt * (K_e H - ieps*J)` in the kernel itself, saving the intermediate vector and ~2 elementwise passes per half-step. Expected ~1.3–1.5× on top, at the cost of a stencil-specific step function (a second copy of step logic to maintain).
- float32 by default after validating accumulated wake-potential error.
- CUDA-aware MPI / device-resident halos (`_mpi_communicate` currently moves the field to the host and back every half-step).
