"""Reproducible local structure/domain refinement of raw overlapping pixels.

Synthetic observations use the native, independently unit-tested kinematic
forward model. Initial coordinates are perturbed known coordinates; this
exercise is not an ab initio solution. The original noisy pixel arrays are
retained for an independent dense background-profiled likelihood comparison.

Run with both projects and test dependencies installed::

    OPENBLAS_NUM_THREADS=1 python validation/run_v08_joint.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy
import time

import gemmi
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from ariadne_xtal.joint_refinement import refine_observations
from ariadne_xtal.model import Structure
from ariadne_xtal.scattering import calculated_intensities


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE/"v08_joint_measurements.json")
    args = parser.parse_args()
    helpers = runpy.run_path(str(HERE.parent/"tests/test_joint_v08.py"))
    cases = [
        ("mixed_overlap_exact", {"all_coincident": False}),
        ("complete_overlap_exact", {"all_coincident": True}),
        ("inversion_without_anomalous", {"all_coincident": True, "inversion": True}),
        ("mixed_overlap_noisy", {"all_coincident": False, "noise": 1.}),
    ]
    from ariadne_xtal.kernels import kernel_status
    result = {"schema": "ariadne_joint_structure_validation/1",
              "numeric_kernel": kernel_status(),
              "python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy.__version__, "gemmi": gemmi.__version__,
              "scope": "Synthetic same-phase local refinement from perturbed known coordinates; no ab initio claim",
              "source_sha256": {"joint_refinement.py": hashlib.sha256(
                  (HERE.parent/"src/ariadne_xtal/joint_refinement.py").read_bytes()).hexdigest()},
              "cases": []}
    with threadpool_limits(limits=1):
        for name, settings in cases:
            case = helpers["make_joint_case"](**settings)
            started = time.perf_counter()
            fit = refine_observations(case["start"], case["groups"],
                                      domain_fractions=[.55, .45], refine_domain_fractions=True,
                                      phase_relation="same_phase_supported")
            elapsed = time.perf_counter()-started
            model = Structure.from_dict(fit["structure"])
            fractions = np.asarray(fit["domain_fractions"])
            raw = helpers["raw_profiled_objective"](case, model, fit["scale"], fractions)
            errors = [np.linalg.norm(model.cell.to_cartesian(model.cell.minimum_image(a.frac-b.frac)))
                      for a, b in zip(model.atoms, case["truth"].atoms)]
            row = {"name": name, "success": fit["success"], "wall_seconds": elapsed,
                   "n_atoms": len(model.atoms), "n_regions": len(case["groups"]),
                   "n_pixels": sum(len(r["data"]) for r in case["raw"]),
                   "rank_one_regions": sum(o.rank == 1 for o in case["observations"]),
                   "rank_two_regions": sum(o.rank == 2 for o in case["observations"]),
                   "n_parameters": fit["n_parameters"], "jacobian_rank": fit["jacobian_rank"],
                   "identifiable": fit["identifiable"], "has_covariance": fit["covariance"] is not None,
                   "true_domain_fractions": case["fractions"].tolist(),
                   "fitted_domain_fractions": fractions.tolist(),
                   "domain_fraction_estimates": fit["domain_fraction_estimates"],
                   "domain_fraction_status": fit["domain_fraction_status"],
                   "domain_fraction_standard_uncertainties": fit["domain_fraction_standard_uncertainties"],
                   "fitted_intensity_scale": fit["scale"], "true_intensity_scale": case["scale"],
                   "maximum_coordinate_error_angstrom": float(max(errors)),
                   "maximum_Uiso_error_angstrom_squared": float(max(abs(a.u_iso-b.u_iso)
                       for a, b in zip(model.atoms, case["truth"].atoms))),
                   "initial_objective": fit["initial_objective"],
                   "final_compressed_objective": fit["objective"],
                   "final_raw_profiled_pixel_objective": raw,
                   "objective_difference": float(fit["objective"]-raw),
                   "diffraction_evaluations": fit["diffraction_evaluations"],
                   "component_intensities_were_detwinned": fit["component_intensities_were_detwinned"]}
            if settings.get("inversion"):
                row["fraction_interpretation"] = (
                    "No anomalous scattering and identical profiles: fraction is unobservable. "
                    "The numerical fraction is an arbitrary representative, not a measurement."
                )
            if name == "mixed_overlap_noisy":
                row["example_regions"] = []
                for index in (0, 1):
                    region, observation = case["raw"][index], case["observations"][index]
                    components = fractions*calculated_intensities(model, region["hkl"], scale=fit["scale"])
                    sigma = np.sqrt(region["variance"])
                    residual = (region["data"]-region["profiles"] @ components)/sigma
                    beta = np.linalg.lstsq(region["background"]/sigma[:, None], residual, rcond=None)[0]
                    row["example_regions"].append({
                        "index": index, "rank": observation.rank,
                        "pixel": list(range(len(region["data"]))),
                        "data": region["data"].tolist(), "sigma": sigma.tolist(),
                        "component_predictions": (region["profiles"]*components[None, :]).tolist(),
                        "fitted_background": (region["background"] @ beta).tolist(),
                        "fitted_mean": (region["profiles"] @ components+region["background"] @ beta).tolist(),
                        "compressed_R": observation.R.tolist(),
                    })
            result["cases"].append(row)
            print(name, "rank", f"{fit['jacobian_rank']}/{fit['n_parameters']}",
                  "alpha", fit["domain_fractions"], "objective", fit["objective"],
                  "raw_delta", fit["objective"]-raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
