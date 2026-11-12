# Ariadne 0.8.0

This release joins the retained Ariadne reduction project and the independently
implemented Ariadne-Xtal structure engine in one source distribution. The
reduction baseline is commit `60a3cfb166dab5580c1afdcde248357c4b7e9315`;
the native baseline is the preserved Ariadne-Xtal 0.1.0 source. The release
does not bundle official SHELXT/SHELXL executables or a compiled DIALS runtime.

## Observations and weak or overlapping diffraction

- Adds an importable streaming Gaussian observation operator. Background
  elimination preserves the entire conditional likelihood as
  `||R @ I - z||² + residual_constant`, including its null space.
- Preserves signed estimates. Unidentifiable individual intensities are
  reported as unidentifiable; estimable linear combinations remain available.
- Adds true-count Poisson signal/background fitting, analytic profiled scores,
  and conditional likelihood-ratio intervals with nuisance reoptimization.
- Fixes large-count projected-gradient cancellation with Fisher-coordinate
  scaling and a stable KKT expression.
- Routes using noise provenance and geometry. Calibrated Poisson-Gaussian data
  retain their existing backend and are not silently treated as integer counts.

## Native phase retrieval and symmetry

- Adds hybrid P1 phase retrieval using a signed-intensity radial proximal
  update, sparse damped triplet guidance, Patterson starting points, charge
  flipping and atomic-support projections.
- The intensity update is covariant under a change of intensity units;
  negative observations remain part of the objective.
- Adds explicit candidate-space-group origin-phase fitting, exact periodic
  atom matching and orbit projection to an asymmetric unit.
- Fixes input-order dependence in orbit averaging with mixed isotropic and
  anisotropic displacement tensors.

## Native structural refinement

- Replaces repeated finite-difference diffraction calculations by a blocked
  analytic Jacobian through coordinates, site symmetry, positive ADPs,
  occupancy constraints, scale and domain fractions.
- Uses streamed QR followed by a small SVD for parameter information and
  null-space diagnostics, avoiding a full observation-space left matrix.
- Adds same-phase joint refinement directly against compressed pixel groups.
  The fitted model predicts the observed combinations; it does not require
  a fabricated independent detwinned intensity table.
- Provides an optional compiled C++ numeric kernel and a NumPy reference
  backend. Build, dispatch, numerical-equivalence and performance details are
  documented in the complete technical report and native validation records.

## Native autonomy and execution

- Adds `autosolve` and `solve-observations`, with connected work, selection
  and audit groups; the final audit is evaluated after model selection.
- Connects native actions to the persistent reduction agent through
  `native.run`, typed artifacts, bounded processes and recorded input snapshots.
- Adds resource-admitted DAG execution with configurable concurrency, explicit
  CPU and memory reservations, process isolation and recovery records. The
  default remains sequential. Memory reservations are not OS RSS limits.

## Deliverables and evidence

- Complete source, examples, tests, build configuration and scientific fixtures.
- Illustrated Chinese LaTeX textbook and standalone structure-feedback report,
  with PDF figures and redistributable font subsets.
- Reproducible synthetic blind-phase and overlap-structure experiments, an
  experimental FeN4 known-model regression, performance measurements, and
  actual CLI/MCP/persistent-agent checks.
- Machine-readable capability boundaries and per-file SHA-256 release hashes.

Known production gaps are itemized in the report and capability response.
The recorded synthetic recovery and local timing results are not claims of
full SHELX instruction compatibility, measured MOF success rates or speed
relative to SHELXL.
