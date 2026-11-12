"""Reproduce the local DAG scheduler tests and two real FeN4 refinements.

From the repository root: python benchmarks/v08_scheduler.py --with-tests
Numerical calls use the Python backend, so this test does not require a C++
compiler. The two independent known-model fits are not a blind phase test.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from ariadne.agent import engine
from ariadne.agent.store import Store
from ariadne.agent.util import atomic_json, file_hash

ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = [
    "tests_rebuild/test_agent.py", "tests_rebuild/test_native_protocol.py",
    "tests_rebuild/test_review_regressions.py", "tests_rebuild/test_native_backend_v08.py",
    "tests_rebuild/test_scheduler_v08.py",
]


def _timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def test_suite(output):
    xml = output/"scheduler_tests.xml"
    command = [sys.executable, "-m", "pytest", *TEST_FILES, "-q", f"--junitxml={xml}"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (output/"scheduler_tests.txt").write_text(completed.stdout+completed.stderr)
    if completed.returncode:
        raise RuntimeError("Scheduler regression failed; inspect scheduler_tests.txt")
    suites = list(ET.parse(xml).getroot().iter("testsuite"))
    return {"command": "python -m pytest "+" ".join(TEST_FILES)+" -q",
            "actual_argv": command, "exit_code": completed.returncode,
            "tests": sum(int(s.attrib["tests"]) for s in suites),
            "failures": sum(int(s.attrib["failures"]) for s in suites),
            "errors": sum(int(s.attrib["errors"]) for s in suites),
            "skipped": sum(int(s.attrib["skipped"]) for s in suites),
            "junit_xml": xml.name, "stdout": "scheduler_tests.txt"}


def measured_workflow(output):
    source = ROOT/"examples/v08/parallel_native.json"
    request = json.loads(source.read_text())
    previous = os.environ.get("ARIADNE_NUMERIC_BACKEND")
    os.environ["ARIADNE_NUMERIC_BACKEND"] = "python"
    try:
        with tempfile.TemporaryDirectory(prefix="ariadne-v08-scheduler-") as temporary:
            session = Path(temporary)/"session"
            engine.initialize(request, session, base=source.parent)
            final = engine.run(session)
            if not all(n["status"] == "succeeded" for n in final["nodes"]):
                raise RuntimeError(f"Measured native workflow failed: {final}")
            resumed = engine.run(session)
            store = Store(session)
            try:
                attempts = [dict(r) for r in store.db.execute("SELECT * FROM attempts ORDER BY started")]
                events = [dict(r) for r in store.db.execute("SELECT * FROM events ORDER BY seq")]
                for event in events:
                    event["payload"] = json.loads(event["payload"])
                nodes = [store.node(n["id"]) for n in final["nodes"]]
                verified = store.verify_events()
                outcomes = {}
                for node in nodes:
                    if node["id"].startswith("refine_"):
                        result = node["result"]["scientific_result"]
                        name = "scheduler_"+node["id"]+".json"
                        atomic_json(output/name, result)
                        outcomes[node["id"]] = {"file": name, "sha256": file_hash(output/name),
                            "success": result["success"], "scale": result["scale"],
                            "metrics": result["metrics"], "warnings": result["warnings"]}
            finally:
                store.close()
    finally:
        if previous is None:
            os.environ.pop("ARIADNE_NUMERIC_BACKEND", None)
        else:
            os.environ["ARIADNE_NUMERIC_BACKEND"] = previous

    start = min(_timestamp(a["started"]) for a in attempts)
    timeline = [{**a, "start_seconds": _timestamp(a["started"])-start,
                 "finish_seconds": _timestamp(a["finished"])-start} for a in attempts]
    allocations, resources, peak = {}, [], {"workers": 0, "cpu_threads": 0, "memory_mb": 0}
    for event in events:
        data = event["payload"]
        if event["kind"] == "resources_admitted":
            allocations[data["node"]] = data
        elif event["kind"] == "attempt_finished":
            allocations.pop(data["node"], None)
        else:
            continue
        row = {"at": event["at"], "event_sequence": event["seq"],
               "seconds": _timestamp(event["at"])-start, "workers": len(allocations),
               "cpu_threads": sum(a["cpu_threads"] for a in allocations.values()),
               "memory_mb": sum(a["memory_mb"] for a in allocations.values())}
        resources.append(row)
        peak = {k: max(peak[k], row[k]) for k in peak}
    a, b = timeline[:2]
    overlap = max(a["start_seconds"], b["start_seconds"]) < min(a["finish_seconds"], b["finish_seconds"])
    graph = [{"id": n["id"], "tool": n["tool"], "depends_on": n["dependencies"]} for n in nodes]
    atomic_json(output/"scheduler_event_timeline.json", {"events": events, "attempts": timeline,
                                                        "resource_reservations": resources, "graph": graph})
    return {"request": str(source.relative_to(ROOT)), "request_sha256": file_hash(source),
            "experiment": "Two initial-model refinements against 71 published measured FeN4 intensities",
            "numeric_backend": "python", "scheduler": request["scheduler"],
            "independent_fits_overlapped": overlap, "peak_reserved_resources": peak,
            "wall_elapsed_seconds": final["wall_elapsed_seconds"],
            "cumulative_action_seconds": final["cumulative_action_seconds"],
            "cumulative_action_seconds_semantics": final["cumulative_action_seconds_semantics"],
            "event_hash_chain_verified": verified,
            "resume_reexecuted_completed_nodes": resumed["actions_used"] != final["actions_used"],
            "actions_used": final["actions_used"], "outcomes": outcomes,
            "timeline": timeline, "resource_reservations": resources, "graph": graph,
            "timeline_clock_basis": "Recorded UTC event stamps; active deadlines use monotonic clocks",
            "all_nodes_succeeded": all(n["status"] == "succeeded" for n in nodes),
            "event_timeline": "scheduler_event_timeline.json"}


def figure(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = result["timeline"]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    colors = ["#26495C", "#B76936", "#48755A", "#929DA6"]
    for i, (item, color) in enumerate(zip(rows, colors)):
        axes[0].barh(i, item["finish_seconds"]-item["start_seconds"],
                     left=item["start_seconds"], height=.55, color=color)
    axes[0].set(yticks=range(len(rows)), yticklabels=[r["node"] for r in rows],
                title="Measured FeN4 workflow: two starts, one durable dependency graph")
    axes[0].invert_yaxis()
    usage = result["resource_reservations"]
    axes[1].step([0]+[r["seconds"] for r in usage], [0]+[r["cpu_threads"] for r in usage],
                 where="post", color=colors[0])
    axes[1].axhline(result["scheduler"]["cpu_threads"], color=colors[1], linestyle="--", linewidth=1)
    axes[1].set(xlabel="Seconds since first action (UTC event timeline)", ylabel="Reserved CPU tokens",
                ylim=(-.1, result["scheduler"]["cpu_threads"]+.45), yticks=[0, 1, 2])
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(directory/("14_scheduler_timeline."+extension), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"docs/v08/results")
    parser.add_argument("--with-tests", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema": "ariadne.scheduler-validation/1", "platform": platform.platform(),
              "python": sys.version, "scope": "Local POSIX subprocess groups; tested on Linux",
              "memory_semantics": "Declared reservations; no measured RSS claim or OS quota",
              "thread_semantics": "Limits cooperating BLAS/OpenMP libraries, not arbitrary subprocess counts",
              "versions": {key: importlib.metadata.version(key) for key in ("numpy", "scipy", "gemmi", "pytest", "threadpoolctl")},
              "tests": test_suite(output) if args.with_tests else None,
              "measured_workflow": measured_workflow(output)}
    figure(result["measured_workflow"], ROOT/"docs/v08/figures")
    atomic_json(output/"scheduler_validation.json", result)
    print(json.dumps({"output": str(output/"scheduler_validation.json"), "tests": result["tests"],
                      "peak_resources": result["measured_workflow"]["peak_reserved_resources"],
                      "independent_fits_overlapped": result["measured_workflow"]["independent_fits_overlapped"]}, indent=2))


if __name__ == "__main__":
    main()
