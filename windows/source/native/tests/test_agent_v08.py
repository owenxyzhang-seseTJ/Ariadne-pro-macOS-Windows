"""Real typed transport from calibrated pixels to archived native structures."""

import json

import numpy as np
import pytest

from ariadne_xtal.agent import AgentWorkspace, sha256, write_json
from ariadne_xtal.io import write_structure
from ariadne_xtal.model import Atom, Cell, Structure
from ariadne_xtal.reduction_bridge import integrate_problem
from ariadne_xtal.scattering import calculated_intensities


def _fixed_model():
    return Structure(Cell(8, 9, 10), [
        Atom("C1", "C", [.13, .22, .31], u_iso=.024, refine_xyz=False, refine_u=False),
        Atom("O1", "O", [.37, .41, .55], u_iso=.031, refine_xyz=False, refine_u=False),
    ])


def _gaussian_problem(model, scale=1.7):
    hkl = np.array([[1, 2, 3]])
    x = np.linspace(-3, 3, 21)
    profile = np.exp(-x*x/2)
    profile /= profile.sum()
    intensity = calculated_intensities(model, hkl, scale=scale)[0]
    return {"data": (profile*intensity+2.3).tolist(), "profiles": profile[:, None].tolist(),
            "variance": .7, "background": np.ones((len(x), 1)).tolist(),
            "noise": {"kind": "gaussian", "variance_fixed": True, "source": "synthetic fixed readout calibration"},
            "component_hkl": hkl.tolist(), "label": "synthetic-region"}


def _result(root, response):
    assert response["ok"], response
    return json.loads((root/response["result"]).read_text())


def test_gaussian_integration_artifact_becomes_a_joint_refinement_input(tmp_path):
    model = _fixed_model()
    write_structure(model, tmp_path/"model.json")
    problem = _gaussian_problem(model)
    # A calibrated mean pattern correlated with the spot must not become signal.
    offset = 3.5*np.asarray(problem["profiles"])[:, 0]
    problem["data"] = (np.asarray(problem["data"])+offset).tolist()
    problem["offset"] = offset.tolist()
    write_json(tmp_path/"pixels.json", problem)
    originals = {p.name: p.read_bytes() for p in (tmp_path/"model.json", tmp_path/"pixels.json")}
    agent = AgentWorkspace(tmp_path)
    integrated = agent.dispatch({"action": "integrate", "problem": "pixels.json"})
    integration = _result(tmp_path, integrated)
    assert integration["status"] == "completed"
    assert integration["known_mean_offset_applied"] is True
    assert integration["route"]["likelihood_action"] == "gaussian_observable_likelihood"
    assert integration["structure_group"]["hkl"] == [[1, 2, 3]]
    assert "R" not in integrated["summary"] and "observation" not in integrated["summary"]
    groups_path = (tmp_path/integrated["result"]).parent/"structure_groups.json"
    refined = agent.dispatch({"action": "joint-refine", "structure": "model.json",
                              "observations": str(groups_path.relative_to(tmp_path)),
                              "options": {"max_nfev": 40}})
    result = _result(tmp_path, refined)
    assert result["success"] and result["identifiable"]
    assert result["scale"] == pytest.approx(1.7, rel=1e-10)
    assert result["component_intensities_were_detwinned"] is False
    assert "covariance" not in refined["summary"] and "structure" not in refined["summary"]
    manifest = json.loads((tmp_path/refined["manifest"]).read_text())
    run = (tmp_path/refined["manifest"]).parent
    assert {item["role"] for item in manifest["inputs"]} == {"structure", "observations"}
    for item in manifest["inputs"]:
        assert sha256(run/item["snapshot"]) == item["sha256"]
    assert (run/"refined.json").is_file() and (run/"refined.cif").is_file()
    assert all((tmp_path/name).read_bytes() == data for name, data in originals.items())


def test_poisson_dispatch_profiles_the_same_count_likelihood(tmp_path):
    problem = {"data": [2, 3, 5], "profiles": [[.2], [.3], [.5]],
               "noise": {"kind": "poisson", "poisson_counts_verified": True,
                         "gain": 1, "readout_variance": 0, "source": "synthetic photon counter"},
               "interval_contrasts": [[1]], "confidence_level": .95}
    write_json(tmp_path/"counts.json", problem)
    response = AgentWorkspace(tmp_path).dispatch({"action": "integrate", "problem": "counts.json"})
    result = _result(tmp_path, response)
    assert result["status"] == "completed"
    assert result["fit"]["intensities"] == pytest.approx([10.], rel=1e-7)
    assert len(result["intervals"]) == 1
    assert result["intervals"][0]["lower"] < 10 < result["intervals"][0]["upper"]
    assert response["summary"]["interval_count"] == 1
    assert "observation" not in result and "structure_group" not in result


def test_readout_noisy_adu_requests_the_retained_calibrated_backend(tmp_path):
    problem = {"data": [-.2, 1.5, 3.2], "profiles": [[.2], [.3], [.5]],
               "noise": {"kind": "poisson_gaussian", "gain": 2., "readout_variance": 4.,
                         "source": "synthetic ADU calibration"}}
    write_json(tmp_path/"adu.json", problem)
    response = AgentWorkspace(tmp_path).dispatch({"action": "integrate", "problem": "adu.json"})
    result = _result(tmp_path, response)
    assert result["status"] == "requires_calibrated_backend"
    assert result["route"]["likelihood_action"] == "existing_calibrated_poisson_gaussian_backend"
    assert "poisson-gaussian" in result["backend"]
    assert "fit" not in result


def test_solve_observations_dispatches_a_known_model_and_archives_partition(tmp_path):
    model = _fixed_model()
    write_structure(model, tmp_path/"model.json")
    hkl = np.array([[h, h % 3, 1] for h in range(1, 21)])
    observed = calculated_intensities(model, hkl, scale=1.7)
    groups = [{"hkl": [h.tolist()], "R": [[1.]], "z": [float(y)], "label": f"r-{i}"}
              for i, (h, y) in enumerate(zip(hkl, observed))]
    write_json(tmp_path/"observations.json", {"groups": groups})
    response = AgentWorkspace(tmp_path).dispatch({
        "action": "solve-observations", "cell": model.cell.to_dict(), "elements": ["C", "O"],
        "structure": "model.json", "observations": "observations.json",
        "options": {"maximum_models": 1}, "refinement_options": {"max_nfev": 40},
    })
    result = _result(tmp_path, response)
    assert result["status"] == "completed"
    assert result["audit_evaluations"] == 1
    assert result["initialization"]["blind_phase_search_performed"] is False
    assert result["refinement"]["scale"] == pytest.approx(1.7, rel=1e-10)
    run = (tmp_path/response["manifest"]).parent
    assert all((run/name).is_file() for name in ("selected.json", "selected.cif", "partition.json"))
    assert "group_diagnostics" not in response["summary"]["audit"]
    assert len(json.dumps(response["summary"])) < 4000


def test_new_dispatch_budgets_fail_before_numerical_work(tmp_path):
    model = _fixed_model()
    write_structure(model, tmp_path/"model.json")
    write_json(tmp_path/"groups.json", {"groups": [{"hkl": [[1, 2, 3]], "R": [[1.]], "z": [1.]}]})
    response = AgentWorkspace(tmp_path).dispatch({"action": "joint-refine", "structure": "model.json",
                                                "observations": "groups.json", "options": {"max_nfev": 5001}})
    assert not response["ok"] and "evaluation budget" in response["error"]["message"]
    noise = {"kind": "gaussian", "variance_fixed": True, "source": "test"}
    with pytest.raises(ValueError, match="background budget"):
        integrate_problem({"data": [1.], "profiles": [[1.]], "variance": 1.,
                           "background": np.ones((1, 65)), "noise": noise})
    with pytest.raises(ValueError, match="design budget"):
        integrate_problem({"data": np.zeros(40_000),
                           "profiles": np.broadcast_to(0., (40_000, 201)), "variance": 1., "noise": noise})
    with pytest.raises(ValueError, match="sixteen"):
        integrate_problem({"data": [1.], "profiles": [[1.]], "variance": 1., "noise": noise,
                           "interval_contrasts": [[1.]]*17})
    no_background = integrate_problem({"data": [2., 3.], "profiles": [[.4], [.6]],
                                        "background": None, "variance": 1., "noise": noise})
    assert no_background["diagnostic_fit"]["intensities"] == pytest.approx([5.])
