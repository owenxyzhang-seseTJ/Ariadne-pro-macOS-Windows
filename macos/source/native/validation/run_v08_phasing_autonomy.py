"""Reproduce native blind phasing and undetwinned-evidence agent experiments.

The coordinates below are used only to generate observations and to score the
finished models. Neither solve_p1 nor either autonomous solver receives them.
All experiments are synthetic; no experimental success-rate claim is made.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ariadne_xtal.autonomy import solve_refine  # noqa: E402
from ariadne_xtal.direct_methods import intensity_amplitude_prox  # noqa: E402
from ariadne_xtal.evidence_solver import solve_observations  # noqa: E402
from ariadne_xtal.joint_refinement import GaussianStructureGroup  # noqa: E402
from ariadne_xtal.model import Atom, Cell, Reflections, Structure  # noqa: E402
from ariadne_xtal.phasing import PhasingOptions, solve_p1  # noqa: E402
from ariadne_xtal.refinement import RefinementOptions, refine  # noqa: E402
from ariadne_xtal.scattering import calculated_intensities  # noqa: E402


def base_case():
    spec = importlib.util.spec_from_file_location("original_recovery_case", PROJECT / "validation/run_phasing_recovery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_case()


def matched_rms(reference, model):
    """Assess modulo translation, inversion and same-element assignment."""
    target = np.asarray([atom.frac for atom in reference.atoms])
    recovered = np.asarray([atom.frac for atom in model.atoms])
    expected_elements = np.asarray([atom.element for atom in reference.atoms])
    found_elements = np.asarray([atom.element for atom in model.atoms])
    if sorted(expected_elements) != sorted(found_elements):
        raise ValueError("RMS comparison requires identical unit-cell compositions")
    invalid = expected_elements[:, None] != found_elements[None, :]
    best = np.inf
    for sign in (-1, 1):
        for anchor in np.flatnonzero(found_elements == expected_elements[0]):
            shift = target[0] - sign * recovered[anchor]
            for _ in range(6):
                difference = reference.cell.minimum_image(target[:, None, :] -
                                                           (sign * recovered[None, :, :] + shift))
                distances = np.sum(reference.cell.to_cartesian(difference)**2, axis=-1)
                distances[invalid] = 1e12
                rows, columns = linear_sum_assignment(distances)
                shift += difference[rows, columns].mean(axis=0)
            difference = reference.cell.minimum_image(target[rows] - (sign * recovered[columns] + shift))
            best = min(best, float(np.sqrt(np.mean(np.sum(reference.cell.to_cartesian(difference)**2, axis=-1)))))
    return best


def mixed_case():
    cell, elements = Cell(6.2, 6.8, 7.3, 85, 93, 101), ["O", "O", "N", "N", "C", "C", "C", "C"]
    rng = np.random.default_rng(292)
    coordinates = []
    while len(coordinates) < len(elements):
        trial = rng.uniform(0, 1, 3)
        if all(np.linalg.norm(cell.to_cartesian(cell.minimum_image(trial - x))) > .9 for x in coordinates):
            coordinates.append(trial)
    truth = Structure(cell, [Atom(f"{element}{i}", element, xyz, u_iso=.012)
                             for i, (element, xyz) in enumerate(zip(elements, coordinates))])
    hkl = np.array([h for h in itertools.product(range(-12, 13), repeat=3) if h != (0, 0, 0)])
    hkl = hkl[cell.d_spacing(hkl) > .65]
    intensity = calculated_intensities(truth, hkl)
    sigma = np.sqrt(.8**2 + (.025 * intensity)**2)
    return truth, Reflections(hkl, intensity + rng.normal(size=len(hkl)) * sigma, sigma), elements


def overlapping_case():
    cell = Cell(5.7, 5.7, 6.1)
    coordinates = [[.113, .239, .337], [.371, .628, .793], [.725, .159, .563], [.842, .771, .188]]
    truth = Structure(cell, [Atom(f"C{i+1}", "C", xyz, u_iso=.012) for i, xyz in enumerate(coordinates)])
    hkl = np.array([h for h in itertools.product(range(-11, 12), repeat=3)
                    if h != (0, 0, 0) and h < tuple(-v for v in h)])
    hkl = hkl[cell.d_spacing(hkl) > .55]
    rng, fractions, groups = np.random.default_rng(142), np.array([.63, .37]), []
    for i, h in enumerate(hkl):
        if i % 5 == 0:
            hk, domains, coefficients = np.array([h, h[[1, 0, 2]]]), np.array([0, 1]), np.array([.8, 1.2])
            intensity = calculated_intensities(truth, hk, scale=1.3)
            sigma = np.sqrt(.8**2 + (.04 * np.max(intensity))**2)
            factor = (np.array([[1., 1.]]) if i % 10 == 0 else np.array([[1., .96], [0., .28]])) / sigma
        else:
            hk, domains = h[None, :], np.array([1 if i % 5 == 1 else 0])
            coefficients = np.array([1.1 if i % 5 == 1 else 1.2])
            intensity = calculated_intensities(truth, hk, scale=1.3)
            sigma = np.sqrt(.8**2 + (.04 * np.max(intensity))**2)
            factor = np.array([[1 / sigma]])
        observed = factor @ (coefficients * fractions[domains] * intensity) + rng.normal(size=len(factor))
        groups.append(GaussianStructureGroup(hk, factor, observed, coefficients=coefficients,
                                             domain_ids=domains, label=str(i), valid_pixels=len(observed)))
    return truth, groups, fractions


def phase_case(truth, observations, elements, options):
    before = observations.to_dict()
    started = time.perf_counter()
    result = solve_p1(truth.cell, observations, elements, options)
    phase_seconds = time.perf_counter() - started
    candidates = []
    for candidate in result.candidates:
        fit = refine(candidate["structure"], observations, RefinementOptions(max_nfev=150))
        candidates.append({"trial": candidate["trial"], "initialization": candidate.get("initialization"),
                           "rms_before_angstrom": matched_rms(truth, candidate["structure"]),
                           "rms_after_angstrom": matched_rms(truth, fit.structure),
                           "sigma_rmse": fit.metrics["sigma_rmse"], "converged": fit.success,
                           "identifiable": fit.metrics["identifiable"], "structure": fit.structure.to_dict()})
    return {"observations": len(observations), "negative_intensities": int(np.count_nonzero(observations.intensity < 0)),
            "elements": elements, "cell": truth.cell.to_dict(), "options": vars(options),
            "phase_seconds": phase_seconds, "input_unchanged": observations.to_dict() == before,
            "true_coordinates_supplied_to_solver": False,
            "history": result.history, "diagnostics": result.diagnostics, "candidates": candidates}


def make_figures(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    y = np.linspace(-3, 5, 500)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(y, np.sqrt(np.maximum(y, 0)), color=".55", linestyle="--", label=r"Hard modulus $\sqrt{\max(y,0)}$")
    for sigma, color in ((.15, "#087e8b"), (.8, "#bd5700"), (3., "#6b4c9a")):
        ax.plot(y, intensity_amplitude_prox(.7, y, sigma, .25), color=color, label=rf"Intensity $\sigma={sigma:g}$")
    ax.axhline(.7, color=".75", linewidth=.8)
    ax.axvline(0, color=".8", linewidth=.8)
    ax.set(xlabel=r"Signed measured intensity $y$", ylabel=r"Updated amplitude $r$",
           title="A nonnegative amplitude can retain negative-intensity information")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(folder / "signed_intensity_prox.pdf")
    fig.savefig(folder / "signed_intensity_prox.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.5))
    for case in result["weak_phasing"]:
        axes[0].scatter([case["negative_intensities"]] * len(case["candidates"]),
                        [c["rms_after_angstrom"] for c in case["candidates"]],
                        s=38, label=f"noise floor {case['noise_floor']:g}")
    axes[0].set(xlabel="Negative observations retained", ylabel="Coordinate RMS after refinement (angstrom)")
    axes[0].legend(frameon=False, fontsize=8)
    auto = result["overlap_autonomy"]
    labels = ["Work", "Selection", "Audit"]
    counts = [auto["partition"]["group_counts"][r] for r in ("work", "selection", "audit")]
    bars = axes[1].bar(labels, counts, color=["#087e8b", "#bd5700", "#6b4c9a"])
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, count, str(count), ha="center", va="bottom")
    axes[1].set(ylabel="Undivided observation groups", ylim=(0, max(counts) * 1.17),
                title=f"Independent audit RMS = {auto['audit']['mode_sigma_rmse']:.3f}")
    fig.tight_layout()
    fig.savefig(folder / "blind_recovery_and_evidence.pdf")
    fig.savefig(folder / "blind_recovery_and_evidence.png", dpi=180)
    plt.close(fig)


def run(output=None):
    output = Path(output) if output is not None else PROJECT / "validation/v08_phasing"
    output.mkdir(parents=True, exist_ok=True)
    truth, clean = base_case()
    rng = np.random.default_rng(142)
    weak = []
    for floor in (.8, 2.5, 5.):
        sigma = np.sqrt(floor**2 + (.06 * clean.intensity)**2)
        data = Reflections(clean.hkl, clean.intensity + rng.normal(size=len(clean)) * sigma, sigma)
        report = phase_case(truth, data, ["C"] * 4,
                            PhasingOptions(algorithm="hybrid", trials=4, iterations=400, keep=4))
        report["noise_floor"] = floor
        weak.append(report)
        print(json.dumps({"case": "weak", "floor": floor, "negative": report["negative_intensities"],
                          "rms": [c["rms_after_angstrom"] for c in report["candidates"]]}), flush=True)
    mixed_truth, mixed_data, elements = mixed_case()
    mixed = phase_case(mixed_truth, mixed_data, elements,
                       PhasingOptions(algorithm="hybrid", trials=4, iterations=600, keep=4, seed=120))
    rng = np.random.default_rng(142)
    sigma = np.sqrt(.8**2 + (.06 * clean.intensity)**2)
    automatic_data = Reflections(clean.hkl, clean.intensity + rng.normal(size=len(clean)) * sigma, sigma)
    started = time.perf_counter()
    automatic = solve_refine(truth.cell, automatic_data, ["C"] * 4,
                              phasing_options={"algorithm": "hybrid", "trials": 4, "iterations": 400, "keep": 4},
                              refinement_options={"max_nfev": 120})
    automatic["wall_seconds"] = time.perf_counter() - started
    if automatic["status"] == "completed":
        automatic["coordinate_rms_angstrom"] = matched_rms(truth, Structure.from_dict(automatic["refinement"]["structure"]))
    joint_truth, groups, fractions = overlapping_case()
    started = time.perf_counter()
    joint = solve_observations(joint_truth.cell, ["C"] * 4, groups,
                               phasing_options={"algorithm": "hybrid", "trials": 4, "iterations": 600, "keep": 4},
                               refinement_options={"max_nfev": 150}, domain_fractions=[.5, .5],
                               refine_domain_fractions=True, phase_relation="same_phase_supported")
    joint["wall_seconds"] = time.perf_counter() - started
    joint["truth_domain_fractions"] = fractions.tolist()
    joint["true_coordinates_supplied_to_solver"] = False
    joint["fully_overlapped_groups"] = sum(len(g.hkl) == 2 and len(g.R) == 1 for g in groups)
    joint["partially_overlapped_groups"] = sum(len(g.hkl) == 2 and len(g.R) == 2 for g in groups)
    if joint["status"] == "completed":
        joint["coordinate_rms_angstrom"] = matched_rms(joint_truth, Structure.from_dict(joint["refinement"]["structure"]))
    from ariadne_xtal.kernels import kernel_status
    result = {"schema": "ariadne.phasing-autonomy-validation/1",
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                              "numeric_kernel": kernel_status()},
              "evidence_type": "controlled synthetic experiments, not experimental success-rate estimates",
              "weak_phasing": weak, "mixed_element_phasing": mixed,
              "reflection_autonomy": automatic, "overlap_autonomy": joint}
    # Autonomy's public dictionary contains ndarray partition metadata.
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    (output / "result.json").write_text(json.dumps(result, indent=2, default=encode, allow_nan=False) + "\n")
    (output / "overlap_input.json").write_text(json.dumps({"cell": joint_truth.cell.to_dict(), "elements": ["C"] * 4,
                                                          "groups": [g.to_dict() for g in groups]}, separators=(",", ":")) + "\n")
    make_figures(output, result)
    print(json.dumps({"mixed_rms": [c["rms_after_angstrom"] for c in mixed["candidates"]],
                      "automatic_status": automatic["status"], "automatic_audit": automatic.get("audit", {}).get("sigma_rmse"),
                      "joint_status": joint["status"], "joint_audit": joint.get("audit", {}).get("mode_sigma_rmse"),
                      "joint_rms": joint.get("coordinate_rms_angstrom"), "output": str(output)}), flush=True)
    return result


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
