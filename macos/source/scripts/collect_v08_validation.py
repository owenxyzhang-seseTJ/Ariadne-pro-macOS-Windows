"""Summarize completed, saved 0.8.0 checks without rerunning experiments.

Run after the separate core/native pytest gates and wheel integration check.
This collector rejects missing evidence or failed tests; it never synthesizes
test results. Complete scientific records remain at their linked source paths.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/v08/results"


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def record(relative):
    path = ROOT / relative
    return {"path": relative, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def tests(name, command):
    log = (RESULTS / (name + ".log")).read_text(encoding="utf-8")
    tree = ET.parse(RESULTS / (name + ".xml"))
    suites = list(tree.iter("testsuite"))
    counts = {key: sum(int(s.get(key, "0")) for s in suites)
              for key in ("tests", "errors", "failures", "skipped")}
    if counts["errors"] or counts["failures"]:
        raise RuntimeError(f"A test gate failed: {name}")
    match = re.search(r"\b(\d+) passed\b", log)
    if match is None:
        raise RuntimeError(f"No completed pytest summary in {name}")
    subtests = re.search(r"\b(\d+) subtests passed\b", log)
    skipped = [{"test": case.get("name"), "reason": case.find("skipped").get("message", "")}
               for case in tree.iter("testcase") if case.find("skipped") is not None]
    return {"command": command, "ordinary_tests_passed": int(match.group(1)),
            "subtests_passed": 0 if subtests is None else int(subtests.group(1)),
            "junit_counts_including_subtests": counts, "skipped_cases": skipped,
            "junit": record(f"docs/v08/results/{name}.xml"),
            "stdout": record(f"docs/v08/results/{name}.log")}


def phase_summary(case):
    return {"observations": case["observations"],
            "negative_intensities": case["negative_intensities"],
            "elements": case["elements"],
            "true_coordinates_supplied_to_solver": case["true_coordinates_supplied_to_solver"],
            "candidates": [{key: value for key, value in candidate.items() if key != "structure"}
                           for candidate in case["candidates"]]}


def main():
    gates = [tests("reduction_tests", "python -m pytest tests tests_rebuild -q"),
             tests("native_tests", "ARIADNE_MCP_STDIO_TEST=1 python -m pytest native/tests -q")]
    reduction = read("benchmarks/v08_reduction_results.json")
    refinement = read("native/validation/v08_refinement_measurements.json")
    kernels = read("native/validation/v08_kernel_measurements.json")
    phases = read("native/validation/v08_phasing/result.json")
    joint = read("native/validation/v08_joint_measurements.json")
    real = read("native/validation/v08_real_fen4/result.json")
    scheduler = read("docs/v08/results/scheduler_validation.json")
    wheels = read("docs/v08/results/wheel_integration.json")
    kernel_build = read("native/validation/v08_kernel_packaging.json")
    report_bundle = read("docs/v08/results/report_bundle_check.json")
    pdf_review = read("docs/v08/results/pdf_review.json")
    if report_bundle["status"] != "passed" or pdf_review["status"] != "passed":
        raise RuntimeError("The final report bundle or visual review did not pass")
    native_examples = {}
    for name in ("example_autosolve_cli", "example_overlap_cli"):
        value = read(f"docs/v08/results/{name}.json")
        if not value.get("ok"):
            raise RuntimeError(f"Native CLI example failed: {name}")
        native_examples[name] = value
    if phases["overlap_autonomy"]["status"] != "completed" or not scheduler["measured_workflow"]["all_nodes_succeeded"]:
        raise RuntimeError("A scientific workflow did not complete")
    if not real["native"]["success"] or not all(case["success"] for case in joint["cases"]):
        raise RuntimeError("A structural validation did not converge")
    if not kernels["environment"]["kernel"]["compiled_available"]:
        raise RuntimeError("Compiled-kernel validation did not use a compiled extension")
    native_summary = {}
    for name in ("reflection_autonomy", "overlap_autonomy"):
        value = phases[name]
        native_summary[name] = {
            "status": value["status"], "selected_candidate": value["selected_candidate"],
            "coordinate_rms_angstrom": value["coordinate_rms_angstrom"],
            "audit_evaluations": value["audit_evaluations"],
            "audit": {k: v for k, v in value["audit"].items() if k != "group_diagnostics"},
            "partition_counts": value["partition"].get("group_counts", value["partition"]["counts"]),
        }
        if name == "overlap_autonomy":
            native_summary[name].update(
                domain_fraction_estimates=value["refinement"]["domain_fraction_estimates"],
                true_domain_fractions=value["truth_domain_fractions"],
                component_intensities_were_detwinned=value["component_intensities_were_detwinned"],
                numeric_backend=value["refinement"].get("numeric_backend"),
            )
    pdfs = []
    for name in ("Ariadne_0_8_0_Textbook", "Ariadne_0_8_0_Loop"):
        path = ROOT / "docs/v08" / (name + ".pdf")
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        page_count = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE).group(1))
        pdfs.append({"file": path.relative_to(ROOT).as_posix(), "pages": page_count,
                     "build_bytes": path.stat().st_size})
    evidence_paths = ["benchmarks/v08_reduction_results.json",
                      "native/validation/v08_refinement_measurements.json",
                      "native/validation/v08_kernel_measurements.json",
                      "native/validation/v08_kernel_packaging.json",
                      "native/validation/v08_joint_measurements.json",
                      "native/validation/v08_phasing/result.json",
                      "native/validation/v08_real_fen4/result.json",
                      "docs/v08/results/scheduler_validation.json",
                      "docs/v08/results/wheel_integration.json",
                      "docs/v08/results/report_bundle_check.json",
                      "docs/v08/results/pdf_review.json"]
    answer = {
        "schema": "ariadne.complete-release-validation/1", "version": "0.8.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "baseline_reduction_commit": "60a3cfb166dab5580c1afdcde248357c4b7e9315",
        "baseline_native": "preserved Ariadne-Xtal 0.1.0 source",
        "distribution": "complete local source release; no remote Git tag asserted",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "dependencies": {n: metadata.version(n) for n in
                            ("numpy", "scipy", "gemmi", "matplotlib", "mcp", "pytest", "threadpoolctl")}},
        "tests": gates,
        "totals": {"ordinary_tests_passed": sum(g["ordinary_tests_passed"] for g in gates),
                   "subtests_passed": sum(g["subtests_passed"] for g in gates),
                   "skipped": sum(g["junit_counts_including_subtests"]["skipped"] for g in gates),
                   "failures": 0, "errors": 0},
        "gaussian_likelihood_maximum_relative_error": reduction["gaussian_strong_weak"]["maximum_relative_compressed_likelihood_error"],
        "poisson_joint_background_interval": reduction["joint_count_background"]["joint_background_interval"],
        "gaussian_streaming_workspace": reduction["workspace"],
        "analytic_vs_finite_difference": refinement["refinement"],
        "cpp_kernel": {k: kernels[k] for k in ("environment", "predictor", "refinement", "memory", "memory_interpretation", "cancellation", "interpretation")},
        "blind_phase_recovery": [phase_summary(c) for c in phases["weak_phasing"]] + [phase_summary(phases["mixed_element_phasing"])],
        "autonomous_workflows": native_summary,
        "joint_structure_cases": [{k: v for k, v in case.items() if k != "example_regions"} for case in joint["cases"]],
        "experimental_known_model_regression": {"doi": real["doi"], "observations": real["observations"],
                    "metrics": real["native"]["metrics"], "published_reference": real["published_reference"],
                    "interpretation": real["interpretation"]},
        "scheduler": scheduler,
        "native_examples": native_examples,
        "wheel_integration": wheels, "optional_kernel_build": kernel_build,
        "report_bundle_check": report_bundle, "pdf_visual_review": pdf_review,
        "reports": pdfs, "evidence_files": [record(p) for p in evidence_paths],
        "scope": ["No user raw detector frames were available for an end-to-end instrument test.",
                  "Native DIALS/cctbx/dxtbx-dependent checks are listed as skipped, not passed.",
                  "Synthetic blind recovery is not an experimental MOF success-rate estimate.",
                  "FeN4 is a measured-data, known-model refinement regression.",
                  "Timing ratios compare declared local implementations, not SHELXL.",
                  "Agent memory reservations are not OS hard RSS limits.",
                  "Complete SHELXT/SHELXL instruction/production parity is not claimed."],
    }
    destination = RESULTS / "release_validation.json"
    destination.write_text(json.dumps(answer, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "tests": answer["totals"], "reports": pdfs}))


if __name__ == "__main__":
    main()
