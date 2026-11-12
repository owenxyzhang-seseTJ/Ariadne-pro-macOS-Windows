"""Machine-readable capability boundaries; no implicit backend substitution."""
CAPABILITIES = {
    "package": "ariadne-xtal", "version": "0.8.0", "native_full_shelx_parity": False,
    "native": {
        "crystal_geometry": "General positive-definite cells; exact periodic closest image",
        "symmetry": "All Gemmi space-group settings and validated explicit operation groups",
        "scattering": "Independent IT92 kinematic X-ray sum; explicit f-prime/f-double-prime; physical special-site occupancies",
        "phasing": "Multistart P1 charge flipping, atomic projection or hybrid signed-intensity proximal/tangent/Patterson search with supplied unit-cell composition",
        "spacegroup_diagnostics": "Metric/absence diagnostics, explicit candidate Fourier origin search and exact orbit-to-ASU projection; independent selection required",
        "refinement": "All signed work F-squared with blocked analytic diffraction Jacobian, physical constraints and bounded trust-region optimization",
        "derivatives": "Exact symmetry/coordinate/Cholesky/log-tensor/occupancy/scale/twin chains; finite differences only for geometric restraint residuals",
        "constraints": "Site symmetry, continuous origin gauge, native equality/occupancy constraints; see API schema",
        "restraints": "Native distance, angle, similar-distance, planarity, ADP-similarity and rigid-bond residuals",
        "twins": "Merohedral integer reciprocal laws; explicit indexed same-phase overlapping groups with jointly refined domain fractions and nullspace diagnostics",
        "pixel_likelihood": "Gaussian background-profiled observable compression and true Poisson joint signal/background fitting via installed Ariadne package",
        "joint_structure": "Direct R diag(c alpha) |F|^2 structural objective; no required deconvolved intensity table",
        "absolute_structure": "Inversion-twin refinement and fixed-model weighted Friedel-quotient diagnostic",
        "maps": "Unweighted Fo-Fc, 2Fo-Fc, calculated density, Patterson, CCP4/NPZ export",
        "graphics": "Offline 3D ellipsoids and density surfaces, reciprocal space, diagnostics, SVG/PDF output",
        "agent": "Native autosolve and solve-observations with connected work/selection/audit groups, structured CLI/JSONL/MCP and persistent parent-agent native.run",
        "feedback": "Phase from geometrically isolated reference-domain evidence then refine all work groups; select independently and evaluate audit once",
    },
    "legacy_bridge": {
        "engines": ["shelxt", "shelxl"], "official_binaries_bundled": False,
        "requires": "Explicit path to a separately installed executable; bridge enabled at agent startup",
        "semantics": "Original INS/HKL/auxiliary bytes passed through; official engine executes its own features",
        "validation": "Transport tested with mock executables; official numerical parity not measured in this environment",
    },
    "not_native": [
        "Complete SHELXT integrated Laue/space-group/origin/element assignment pipeline",
        "Full SHELXL instruction grammar execution, AFIX/HFIX/riding H, FRAG and rigid-group models",
        "Automatic parsing/execution of every non-merohedral HKLF5 instruction; explicit native component-group interface is implemented",
        "Neutron refinement, extinction, solvent masks/ABIN, SWAT and automatic absorption correction",
        "CGLS/block least-squares, all SHELXL weighting modes and complete CIF archive/checkCIF certification",
        "Raw detector-image indexing/integration/scaling or production instrument control",
        "Automatic unrestricted MOF topology/chemical composition generation; full Poisson-pixel structure/geometry/absorption joint optimizer",
    ],
}


def capabilities():
    import copy
    from .kernels import kernel_status
    result = copy.deepcopy(CAPABILITIES)
    result["numeric_kernel"] = kernel_status()
    return result
