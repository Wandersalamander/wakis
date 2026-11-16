# GPU stencil backend: handoff notes

> Working notes for validating this branch on a GPU machine. **Delete this file before merging.**

Branch: `feature/gpu-stencil-backend`. Written on a machine without an NVIDIA GPU or CuPy, so **none of the GPU code has been executed yet**. Read `AGENTS.md` and `CONTRIBUTING.md` first.

## GPU validation results (RTX 5060 Ti, sm_120, CUDA 13.2 runtime, CuPy 14.2)

- Kernels compile with NVRTC and run. Fixed on the GPU machine: the `check_curl_stencil` tolerance is now dtype-aware (float32 operators failed at 1e-12), and the benchmark script passes `dtype.type` and warms the GPU up before timing.
- **New: single-pass fused step** (`fused_step` in `stencil_kernels.cu`, `FusedStencilStep`, `SolverFIT3D._one_step_fused_stencil`). It is used automatically by the stencil backend without CPML, TFSF or MPI. It marches z-chunks per (i, j) column, carries H of plane k-1 in registers and swaps E/H buffers after each step. Tested against the two-kernel step with chunk lengths 1, 3 and 16 (`test_gpu_fused_step_matches_two_kernel_step`).
- `pytest --gpu`: 250 passed, 6 skipped (no mpi4py). The one failure, `test_mpi_load_state` (`load_state` on GPU), fails identically on upstream main. Physics tests with the stencil forced on (test_001, 005, 012, 014, 015): all pass.
- ms/step, PEC box, median of 3 trials after a 1 s warm-up, same script for both code bases (upstream main = 06e9b05). After 100 steps from identical fields, every branch backend matches upstream to ≤5e-15 (f64) / ≤2e-6 (f32):

  | case | upstream main | sparse | stencil 2-kernel | stencil fused | vs. upstream |
  |---|---|---|---|---|---|
  | 128³ f64 | 7.28 | 4.86 | 3.08 | 1.31 | 5.6× |
  | 128³ f32 | 4.05 | 2.74 | 1.46 | 0.65 | 6.2× |
  | 256³ f64 | 59.3 | 39.3 | 24.9 | 10.9 | 5.5× |
  | 256³ f32 | 35.1 | 22.5 | 12.2 | 5.2 | 6.7× |

  Modelled fused traffic (10 vectors/step) is 83–86% of the 448 GB/s peak. CSR beats the old CSC on the sparse path, so keep it. CPML/TFSF stencil steps are not benchmarked.
- Open: at 256³, `check_curl_stencil` takes ~23 s of the stencil setup; Nsight Compute 2025.1 can't read counters with driver R621 (`LibraryNotLoaded`).

## Stacked branch `feature/gpu-compact-coefficients`

The fused step reads 1-D/2-D metric tables and a uint8 material id per entry instead of the five coefficient vectors when the model allows it (`compact_coefficients`, verified against the full vectors at setup). Kernel tuning per variant: `__launch_bounds__` (3 blocks/SM full, 4 compact), `dfwd_k` a real call in the full variant, inlined in the compact one, and `__restrict__` throughout. Same method as above, ms/step and GPU memory held by the solver:

| case | full fused | compact fused | vs. full | vs. upstream | GPU memory full → compact |
|---|---|---|---|---|---|
| 128³ f64 | 1.29 | 1.07 | 1.21× | 6.8× | 0.60 → 0.41 GB |
| 128³ f32 | 0.65 | 0.54 | 1.20× | 7.5× | 0.30 → 0.21 GB |
| 256³ f64 | 10.5 | 7.96 | 1.32× | 7.4× | 4.83 → 3.27 GB |
| 256³ f32 | 5.18 | 4.07 | 1.27× | 8.6× | 2.42 → 1.66 GB |

Fields match upstream to ≤5e-15 (f64) / ≤2e-6 (f32) after 100 steps. The compact kernel reaches only ~58% of peak on its reduced traffic: it is bound by dependent id → table loads (78 byte loads per node in SASS) and index arithmetic, not by bandwidth. Possible next steps: pack the three component ids of a node into one 32-bit word, 32-bit index arithmetic, and keeping `ieps`, `sigma` and `J_old` off the GPU when there is no conductivity (~3 vectors of memory).

## What the branch changes

1. **`gpu_backend='stencil'`** (new `SolverFIT3D` argument, default `'sparse'`). `wakis/stencil.py` replaces the two cupyx curl matrices (`tDsiDmuiDaC`, `itDaiDepsDstC`) with `CurlStencil` objects. These run matrix-free CUDA kernels and act as drop-in operators for `op * x`. Every existing non-CPML step path (`_one_step`, `_mpi_one_step`, TFSF, conductivity, ABC and periodic wrappers) is unchanged. The math is in `docs/physicsguide.md`, section *Matrix-free GPU stencil*.
   - The kernels are in **`wakis/stencil_kernels.cu`** (plain CUDA C; Python prepends `typedef float|double real;` and `#define KAPPA` for CPML). The file ships via `package_data` in `setup.py`; a wheel build was checked to contain it.
   - **No compiler is needed in the environment.** `cupy.RawModule` compiles at runtime with NVRTC, which CuPy itself depends on, and caches the result in `~/.cupy/kernel_cache`. The first launch compiles in well under a second. ROCm uses hipRTC.
2. **CPML** (`_one_step_cpml_stencil`): the bulk update uses the same kernels with the stretching 1/κ per derivative direction (compiled with `KAPPA`). A `cpml_psi` kernel updates ψ only on the slab nodes of each CPML face (one launch per face per half-step) and replaces the 12 directional DIA operators, which are verified and then deleted. TFSF injection feeds both parts, as in `_one_step_cpml`. To make this possible, `_initialize_CPML_matrices` now keeps `ikappa`/`itkappa` until the first step (`_attrcleanup` frees them).
3. **Tripwire**: `check_curl_stencil` verifies at setup that the assembled operators factor as the stencil: the two curl matrices, or with CPML all 12 directional operators. It raises otherwise.
4. **Sparse path, also on CPU**: `_one_step` and `_mpi_one_step` now compute `dt * (K * x)` in place. Before, `self.dt * self.tDsiDmuiDaC * E` evaluated left to right and **copied and scaled the whole sparse matrix every step**.
5. **Sparse GPU path**: cupyx matrices are now CSR instead of CSC (`gpu_sparse_mat` import). This is unbenchmarked; revert that one line if it isn't faster.
6. `examples/007_GPU_backend_benchmark.py`, docs (`usersguide.md`, `physicsguide.md`, `wakis_stencil.rst`), and tests in `tests/test_016_boundary_conditions.py`.

## Already verified on CPU

- The CUDA kernel logic was translated line by line to NumPy and matched the SciPy operators to <1e-12 in 180 configurations. Those covered 5 grid shapes including 1- and 2-cell axes, every PEC/PMC/periodic/ABC combination per axis, graded meshes, and random materials with PEC cells. The script isn't committed; the `--gpu` test below replaces it.
- **Fake-GPU end-to-end run**: `cupy`/`cupyx` were replaced by a stub whose kernels are NumPy emulations of the `.cu` source, so the real solver glue ran with `use_gpu=True, gpu_backend="stencil"`. That covers field moves, coefficient upload, ψ handover, the step functions, kernel argument lists, and launch grids (each grid was checked to cover exactly its size; index arrays are int64). After 30 steps E, H and every ψ array matched the CPU sparse steps to ≤3e-15 relative error, in 7 configurations:
  - CPML on 1, 2, 3 and all 6 faces, with periodic transverse axes;
  - active TFSF injection;
  - PML, ABC, PEC/PMC/periodic;
  - conductivity, on a graded mesh.

  This run is what tests the Python glue; only the CUDA C itself (syntax, NVRTC, pointer semantics) remains unexecuted.
- `test_curl_stencil_factorization` (8 BC sets incl. PML and CPML, STL materials, graded mesh) and `test_curl_stencil_rejects_extra_terms` pass in CPU CI.
- The full CPU suite passes with the in-place step change (see the PR description for the exact result).

## To do on the GPU machine (in order)

```bash
pip install -e ".[test]" cupy-cuda12x   # match the CUDA version
python -m pytest tests/test_016_boundary_conditions.py -k stencil --gpu -v
```

1. **Equivalence test** `test_gpu_stencil_matches_sparse` covers 8 BC sets (incl. PML and CPML with active TFSF injection) × float32/float64 and 50 steps, and compares fields against the sparse GPU backend. Likely failure modes and fixes:
   - Kernel compile error (NVRTC): fix the source in `_KERNEL_SOURCE`.
   - `cupyx csr_matrix(<scipy csc/dia>)` conversion error: convert explicitly with `.tocsr()` before `gpu_sparse_mat`.
   - Argument type mismatch: kernel parameters are `long long` (passed as `np.int64`) and `int` (`np.int32`).
   - Numerical mismatch: re-check `fwd`/`bwd` against `_incidence`. The CPU emulation passed, so suspect launch or pointer issues first.
2. **Full GPU suite** `python -m pytest --gpu -v`. This checks that the CSR switch and the in-place step didn't break the sparse GPU path (test_001, 005, 014 use `use_gpu`).
3. **End-to-end checks with the stencil.** Temporarily pass `gpu_backend="stencil"` in these tests and run them with `--gpu`. If they pass, consider parametrizing them permanently.
   - `tests/test_001_pec_cubic_cavity.py::test_simulation`: float32 wake potential and impedance vs. stored references.
   - `tests/test_014_cpml.py`: CPML reflection and a CPML + TFSF wake.
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

- **CPML** makes one small `cpml_psi` launch per CPML face and half-step, plus the bulk kernels. If launch overhead shows up on small grids, capture the step in a CUDA Graph.
- **MPI + stencil** should work, since halo exchange acts on fields and the tripwire validates the local operators, but it is untested.
- **AMD ROCm**: CuPy compiles `RawModule` via hipRTC; the kernel uses no CUDA-only intrinsics, but it is untested.
- `update_tensors()` on GPU rebuilds SciPy matrices from GPU fields. That was already broken for the sparse GPU path and is not fixed here.

## Optional follow-ups (only if benchmarks justify them)

- **Fused update kernel**: compute `H -= dt * K_h E` and `E += dt * (K_e H - ieps*J)` in the kernel itself, saving the intermediate vector and ~2 elementwise passes per half-step. Expected ~1.3–1.5× on top, at the cost of a stencil-specific step function (a second copy of step logic to maintain).
- float32 by default after validating accumulated wake-potential error.
- CUDA-aware MPI / device-resident halos (`_mpi_communicate` currently moves the field to the host and back every half-step).
