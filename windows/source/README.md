# Ariadne 0.8.0

Ariadne connects single-crystal diffraction observations to native structure solution, refinement and a persistent scientific agent. This is the complete **local 0.8.0 source distribution**, including the retained DIALS extensions, two Python projects, tests, reproducible examples and illustrated technical reports. Its version identifies the delivered source; it does not assert that a corresponding GitHub tag has been published.

[简体中文](README.zh-CN.md) · [0.8.0 operation guide](docs/v08/Ariadne_0_8_0_Guide.zh-CN.md) · [Native engine](native/README.md) · [Licensing](LICENSING.md)

## Start here

- [Textbook-style technical report, PDF](docs/v08/Ariadne_0_8_0_Textbook.pdf) and [LaTeX source](docs/v08/Ariadne_0_8_0_Textbook.tex): scattering, geometry, pixel likelihoods, scaling, phasing, symmetry, refinement and agent execution.
- [Structure feedback and agent loop, LaTeX](docs/v08/Ariadne_0_8_0_Loop.tex): the separate mathematical account of the closed loop.
- [Runnable 0.8.0 requests and input provenance](examples/v08/PROVENANCE.json).
- `ariadne-xtal capabilities` and `ariadne-agent capabilities`: current machine-readable actions and boundaries.

## What changes in 0.8.0

Strong, geometrically unambiguous reflections can anchor geometry and initialization. Weak, negative and overlapping measurements remain in the appropriate likelihood. The choice of a noise model follows detector calibration; an observed intensity threshold does not decide whether a measurement is retained.

For Gaussian pixels with fixed variances and a linear background, the reduction core preserves the profiled likelihood as `||R I - z||² + residual_constant`. Background is eliminated with QR/SVD; rank deficiency and unmeasurable intensity combinations remain explicit. Structure refinement can predict the components directly through this factor, without inventing a separately measured intensity for every overlapping reflection. True photon counts instead use a nonnegative signal/background Poisson model and conditional profile intervals.

The native phasing engine combines a bounded triplet phase-relation graph, Patterson starts, charge flipping and atomic support. A signed-intensity radial likelihood update replaces rigid amplitude substitution in the new `hybrid` path. Analytic diffraction derivatives, physical ADP parameterizations and blocked calculations support native refinement. Existing `charge_flip` and `atomic_projection` paths remain available.

The native agent can solve, refine and compare candidates with connected work/selection/audit partitions. `solve-observations` initializes from isolated reference-domain evidence, then fits all work groups, including overlapping components. The persistent parent agent calls the same engine through `native.run`, retaining input snapshots, dependencies and restart records.

Independent ready DAG actions can run concurrently in isolated processes. Session and node resource declarations govern worker admission and cooperating BLAS/OMP thread counts. Declared memory is an admission budget, not an operating-system RSS limit; sequential execution remains the default. See the operation guide for the `scheduler` and `resources` fields.

These are implemented native capabilities. **Ariadne 0.8.0 is not a complete SHELXT/SHELXL-equivalent implementation.** Unrestricted chemical composition inference, the full SHELXL instruction language and a full Poisson-pixel structure/geometry/absorption optimizer are outside the native scope. Explicit optional SHELX execution interfaces are retained; official binaries are not included.

## Install the complete source

Run these commands at the top of the unpacked source directory. Python 3.12 or newer is required for the combined installation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './python[dev]' -e './native[dev]'
ariadne-agent capabilities
ariadne-xtal capabilities
```

`python/` supplies reduction likelihoods and persistent agent execution; `native/` supplies independent structure solution and refinement. Install both for pixel integration and joint workflows. Keep the complete source tree: the root scientific scripts, DIALS patches and expert packs are not all contained in either Python wheel. The persistent agent uses POSIX process/locking facilities; Linux, macOS and WSL are the supported execution environments.

The native project includes an optional C++17 intensity/analytic-derivative kernel. `ARIADNE_BUILD_CPP=auto|0|1` controls optional, disabled or required compilation; `ARIADNE_NUMERIC_BACKEND=auto|python|cpp` controls runtime selection. Check the actual backend with `python -m ariadne_xtal.kernels`. The kernel is single-threaded and releases the GIL; the Python reference remains available. Compiler requirements and commands are in the [guide](docs/v08/Ariadne_0_8_0_Guide.zh-CN.md).

## Run the examples

The supplied inputs are ready to use. Commands below run at the repository root; paths inside each native request are relative to `--root`.

```bash
# True-count signal/background integration.
ariadne-xtal run examples/v08/integrate_counts.json --root examples/v08

# Blind synthetic structure solution with declared unit-cell composition.
ariadne-xtal run examples/v08/autosolve.json --root examples/v08

# Blind initialization followed by refinement against overlapping groups.
ariadne-xtal run examples/v08/solve_observations.json --root examples/v08

# Published measured F², starting from an archived FeN4 structure.
ariadne-xtal run examples/v08/refine_fen4.json --root examples/v08

# The same native engine inside a persistent task graph; use a new session.
ariadne-agent init examples/v08/persistent_native.json runs/fen4-session
ariadne-agent run runs/fen4-session
ariadne-agent status runs/fen4-session
```

Each native request writes a new `runs/<action>_.../` directory below its workspace. A successful process is not sufficient evidence of a correct structure: inspect numerical convergence, parameter rank, residuals and held-out diagnostics in the result. These demonstrations contain synthetic observations and a labelled known-model experimental regression, not raw detector images. Phasing receives the **complete unit-cell element list**, not an asymmetric-unit formula.

For JSONL or MCP access, see the [native interface guide](native/README.md). The [0.8.0 guide](docs/v08/Ariadne_0_8_0_Guide.zh-CN.md) explains data formats, scientific units, overlap groups and persistent-agent configuration.

## DIALS reduction and instrument data

The retained prediction, detector geometry, voxel profiles, signed WLS, Poisson–Gaussian treatment, composite scaling and covariance-aware export code resides under `skills/ariadne-reduction/scripts/` and `patches/`. The matching compiled DIALS/dxtbx/cctbx runtime is an external dependency.

`sh scripts/install_local.sh` installs a Python-source overlay for **DIALS 3.30.1** using the matching conda environment. It accepts a base environment and a fresh overlay directory as its two optional arguments. It does not execute scientific validation or install the native structure package automatically; install `native/` in the chosen scientific Python environment as well. A standalone pip cctbx installation does not replace the matched DIALS runtime. Original configuration and detector-workflow instructions remain in the [earlier agent guide](AGENT_GUIDE.zh-CN.md) and [R2 documentation](docs/R2_README.zh-CN.md).

The 0.8.0 examples and tests do not establish end-to-end accuracy on Bruker/Rigaku raw frames or unrestricted complex twins. Indexed component groups require actual geometry, domain assignments and calibration coefficients; the agent does not infer these metadata from a generic intensity table.

## Verification and evidence

```bash
python -m pytest tests tests_rebuild -q
ARIADNE_MCP_STDIO_TEST=1 python -m pytest native/tests -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/v08_reduction.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/validation/run_v08_refinement.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/validation/run_v08_joint.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/validation/run_v08_phasing_autonomy.py
python scripts/verify_release.py
```

Current evidence includes [pixel likelihood measurements](benchmarks/v08_reduction_results.json), [analytic refinement measurements](native/validation/v08_refinement_measurements.json), [joint pixel/structure measurements](native/validation/v08_joint_measurements.json), [blind synthetic phasing and the observation loop](native/validation/v08_phasing/result.json), and [measured FeN4 re-refinement](native/validation/v08_real_fen4/result.json). Test records and example transport results are under [`docs/v08/results/`](docs/v08/results/). Timing is environment- and problem-specific; synthetic recovery and known-model re-refinement answer different validation questions.

[C++ kernel measurements](native/validation/v08_kernel_measurements.json) compare two implementations of the same analytic model. [Scheduler validation](docs/v08/results/scheduler_validation.json) records resource admission, cancellation/recovery and two concurrent known-model refinements; its [request](examples/v08/parallel_native.json) is ready to run.

The release manifest verifies packaged bytes. Running validation changes output files, so a modified working tree need not match the original manifest afterwards. Optional DIALS/cctbx checks need their actual runtime; skipped checks do not count as raw-image validation.

## Historical documentation and licenses

The previous [0.7.0 English README](docs/archive/README_0_7_0.en.md), [0.7.0 Chinese README](docs/archive/README_0_7_0.zh-CN.md) and [native 0.1.0 README](docs/archive/README_Ariadne_Xtal_0_1_0.zh-CN.md) are preserved. Reports under `reports/`, `native/docs/report.*`, old validation directories and `implementation-status.json` describe their original versions; the 0.8.0 report, capabilities and `v08_*` evidence describe this release. The older [Japanese guide](README.ja.md), [native API reference](native/docs/API.md) and [SHELX instruction matrix](native/shelx_instruction_matrix.json) remain available with their original scope.

Existing component licenses apply: [root license](LICENSE), [licensing explanation](LICENSING.md), [native engine license](native/LICENSE), [DIALS patch license](patches/DIALS-LICENSE), dependency notices and report-font notices. No repository-wide relicensing is implied. SHELXT/SHELXL executables and compiled DIALS runtimes are not bundled.
