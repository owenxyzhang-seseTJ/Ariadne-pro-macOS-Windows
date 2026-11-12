"""Persistent DAG tests using the real native subprocess, files and hashes."""

import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from ariadne.agent import engine
from ariadne.agent.store import Store
from ariadne.agent.util import artifact_path, file_hash
from ariadne_xtal.agent import write_json
from ariadne_xtal.io import write_structure
from ariadne_xtal.model import Atom, Cell, Structure
from ariadne_xtal.scattering import calculated_intensities


def _inputs(tmp_path):
    model = Structure(Cell(8, 9, 10), [
        Atom("C1", "C", [.13, .22, .31], u_iso=.024, refine_xyz=False, refine_u=False),
        Atom("O1", "O", [.37, .41, .55], u_iso=.031, refine_xyz=False, refine_u=False),
    ])
    structure_path, pixels_path = tmp_path/"model.json", tmp_path/"pixels.json"
    write_structure(model, structure_path)
    profile = np.array([.02, .06, .12, .2, .2, .2, .12, .06, .02])
    intensity = calculated_intensities(model, [[1, 2, 3]], scale=1.7)[0]
    problem = {"data": (profile*intensity+2.3).tolist(), "profiles": profile[:, None].tolist(),
               "variance": .7, "background": np.ones((len(profile), 1)).tolist(),
               "noise": {"kind": "gaussian", "variance_fixed": True, "source": "synthetic calibration"},
               "component_hkl": [[1, 2, 3]]}
    write_json(pixels_path, problem)
    return structure_path, pixels_path


def _node_result(root, name):
    store = Store(root)
    try:
        return store.node(name)
    finally:
        store.close()


def test_real_native_dag_integrates_refines_packages_and_resumes(tmp_path, monkeypatch):
    structure, pixels = _inputs(tmp_path)
    marker = tmp_path/"input_was_executed.txt"
    shadow = tmp_path/"ariadne_xtal.py"
    shadow.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\nraise RuntimeError('input executed')\n")
    # A basename must resolve before the process runner hashes the executable.
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent)+os.pathsep+os.environ.get("PATH", ""))
    request = {"mode": "custom", "inputs": {"model": str(structure), "pixels": str(pixels), "shadow": str(shadow)},
               "executables": {"scientific_python": "python3"}, "nodes": [
        engine.node("pixels", "native.run", {
            "request": {"action": "integrate", "problem": "pixels.json"},
            "inputs": {"pixels.json": engine.input_ref("pixels"), "ariadne_xtal.py": engine.input_ref("shadow")}}),
        engine.node("structure", "native.run", {
            "request": {"action": "joint-refine", "structure": "model.json", "observations": "groups.json",
                        "options": {"max_nfev": 40}},
            "inputs": {"model.json": engine.input_ref("model"),
                       "groups.json": engine.ref("pixels", "/output_files/structure_groups.json")}}, ["pixels"]),
        *engine.completion_nodes(["structure"]),
    ]}
    root = tmp_path/"session"
    engine.initialize(request, root)
    # The persistent session owns snapshots. Later edits to originals are irrelevant.
    pixels.write_text("original source changed after session initialization")
    structure.write_text("original source changed after session initialization")
    first = engine.run(root, max_steps=1)
    assert first["actions_used"] == 1
    integration = _node_result(root, "pixels")
    assert integration["status"] == "succeeded", integration["error"]
    record = integration["result"]["output_files"]["structure_groups.json"]
    exported = artifact_path(root, record)
    assert json.loads(exported.read_text())["groups"][0]["hkl"] == [[1, 2, 3]]
    command = json.loads((root/"attempts/pixels/001/native.command.json").read_text())
    assert Path(command["argv"][0]).is_absolute()
    assert command["argv"][1:3] == ["-m", "ariadne_xtal"]
    assert command["executable_sha256"] == file_hash(Path(command["argv"][0]).resolve())
    finished = engine.run(root)
    assert finished["actions_used"] == 4
    assert all(node["status"] == "succeeded" for node in finished["nodes"]), finished
    refined = _node_result(root, "structure")["result"]
    assert refined["scientific_result"]["scale"] == pytest.approx(1.7, rel=1e-10)
    assert "refined.json" in refined["output_files"] and "refined.cif" in refined["output_files"]
    imported = refined["output_files"]["inputs/observations.json"]
    assert imported["sha256"] == record["sha256"]
    package = _node_result(root, "package")["result"]["package"]
    with zipfile.ZipFile(artifact_path(root, package)) as archive:
        assert archive.testzip() is None
        assert "manifest.json" in archive.namelist()
    assert engine.run(root)["actions_used"] == 4
    assert not marker.exists(), "An input named after a Python module must remain data"
    store = Store(root)
    try:
        assert store.verify_events()
    finally:
        store.close()


def test_capabilities_also_have_a_persistent_native_result(tmp_path):
    root = tmp_path/"capabilities"
    engine.initialize({"mode": "custom", "inputs": {}, "nodes": [
        engine.node("native", "native.run", {"request": {"action": "capabilities"}})]}, root)
    engine.run(root)
    node = _node_result(root, "native")
    assert node["status"] == "succeeded", node["error"]
    result = json.loads(artifact_path(root, node["result"]["native_result"]).read_text())
    assert result["ok"]
    assert "solve-observations" in {tool["name"] for tool in result["tools"]}


def test_native_validation_error_is_preserved_without_executing_text(tmp_path):
    _, pixels = _inputs(tmp_path)
    marker = tmp_path/"not_executed.txt"
    root = tmp_path/"invalid"
    engine.initialize({"mode": "custom", "inputs": {"pixels": str(pixels)}, "nodes": [
        engine.node("native", "native.run", {
            "request": {"action": "integrate", "problem": "pixels.json",
                        "shell": f"touch {marker}"},
            "inputs": {"pixels.json": engine.input_ref("pixels")}})]}, root)
    engine.run(root)
    node = _node_result(root, "native")
    assert node["status"] == "failed"
    assert "Unknown request fields" in node["error"]
    assert not marker.exists()


@pytest.mark.parametrize("kind", ["filename_escape", "legacy"])
def test_native_adapter_rejects_unregistered_input_paths_and_legacy(tmp_path, kind):
    _, pixels = _inputs(tmp_path)
    request = ({"action": "legacy", "engine": "shelxl", "input": "pixels.json"} if kind == "legacy"
               else {"action": "integrate", "problem": "pixels.json"})
    filename = "pixels.json" if kind == "legacy" else "../escaped.json"
    root = tmp_path/"rejected"
    engine.initialize({"mode": "custom", "inputs": {"pixels": str(pixels)}, "nodes": [
        engine.node("native", "native.run", {"request": request,
                                                "inputs": {filename: engine.input_ref("pixels")}})]}, root)
    engine.run(root)
    node = _node_result(root, "native")
    assert node["status"] == "failed"
    assert not (root/"attempts/native/001/native.command.json").exists()
    assert not (root/"attempts/native/001/escaped.json").exists()


def test_native_subprocess_obeys_the_persistent_action_deadline(tmp_path):
    root = tmp_path/"deadline"
    engine.initialize({"mode": "custom", "inputs": {},
                       "budget": {"action_seconds": .02, "wall_seconds": 30, "max_actions": 1},
                       "nodes": [engine.node("native", "native.run", {"request": {"action": "capabilities"}})]}, root)
    engine.run(root)
    node = _node_result(root, "native")
    assert node["status"] == "failed"
    assert "time budget" in node["error"]
    assert engine.run(root)["actions_used"] == 1
