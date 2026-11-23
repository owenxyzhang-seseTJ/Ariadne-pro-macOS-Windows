from contextlib import closing

import numpy as np
import pytest

from ariadne.agent import engine
from ariadne.agent.backends import DIALS_OUTPUTS
from ariadne.agent.resolve import _companion_scale, _shelxt_model
from ariadne.agent.store import Store
from ariadne.agent.util import artifact, read_json
from ariadne_xtal.io import write_reflections, write_structure
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.scattering import calculated_intensities


def test_resolution_freezes_observations_and_resumes(tmp_path):
    model = Structure(Cell(10, 11, 12), [Atom('Fe1', 'Fe', [.13, .27, .66]),
                                        Atom('O1', 'O', [.32, .81, .53]),
                                        Atom('N1', 'N', [.21, .44, .15])])
    law = np.diag([1, -1, -1]).tolist()
    h = np.array(list(np.ndindex((7, 7, 7))))-3
    h = h[np.any(h != 0, axis=1)]
    exact = calculated_intensities(model, h, scale=1.4, twin_laws=[law], twin_fractions=[.64, .36])
    rng = np.random.default_rng(126)
    r = Reflections(h, exact + rng.normal(0, 6., len(h)), np.full(len(h), 6.))
    observed, model_path = tmp_path/'obs.json', tmp_path/'model.json'
    write_reflections(r, observed)
    write_structure(model, model_path)
    request = dict(mode='resolve', inputs={'hkl': str(observed), 'structure': str(model_path)},
                   settings=dict(spacegroup_candidates=[], finalists=2, shortlist=2, inspect_rejections=0),
                   cache={'directory': str(tmp_path/'cache')})
    session = tmp_path/'session'
    engine.initialize(request, session)
    first = engine.run(session, max_steps=3)
    assert {n['id']: n['status'] for n in first['nodes']}['screen'] == 'succeeded'
    result = engine.run(session)
    assert all(n['status'] == 'succeeded' for n in result['nodes'])
    with closing(Store(session)) as store:
        candidate = store.node('select')['result']['preferred']
        assert candidate['fraction'] == pytest.approx(.36, abs=.01)
        assert store.node('final-audit')['result']['audit_evaluations'] == 1
        assert store.node('screen')['attempts'] == 1
        preparation = store.node('evidence')['result']
        for role in ('work', 'selection', 'audit'):
            assert read_json(session/preparation[f'{role}_components']['path'])
    engine.run(session)
    with closing(Store(session)) as store:
        assert store.node('final-audit')['attempts'] == 1
    warm = tmp_path/'warm'
    engine.initialize(request, warm)
    engine.run(warm, max_steps=3)
    with closing(Store(warm)) as store:
        evidence = store.node('evidence')['result']
        diagnostic = read_json(warm/evidence['diagnostic']['path'])
        report = store.node('screen')['result']['report']
        rows = read_json(warm/report['path'])['rows']
        assert diagnostic['stage_cache_hit']
        assert any(x.get('batch_statistics', {}).get('stage_cache_hit') for x in rows)



def test_shelxt_peak_height_stays_in_paired_original(tmp_path):
    path = tmp_path/'job_a.res'
    original = ('TITL solved candidate\nCELL 0.71073 10 11 12 90 90 90\nLATT -1\n'
                'SFAC FE O\nUNIT 1 1\nL.S. 8\nBOND\nLIST 4\nFMAP 2\nPLAN 20\n'
                'FE01 1 0.12 0.21 0.31 11.0 0.02 23.65\n'
                'O002 2 0.22 0.81 0.47 11.0 0.03 14.50\nHKLF 4\nEND\n')
    path.write_text(original)
    model = _shelxt_model(path)
    assert [a.element for a in model.atoms] == ['Fe', 'O']
    assert path.read_text() == original


def test_shelxt_pairing_certifies_a_common_scale_and_basis():
    indices = np.column_stack((np.arange(1, 21), np.ones(20), np.zeros(20)))
    y = np.linspace(-18, 900, len(indices))
    sigma = 15+np.arange(len(indices))*.15
    original = Reflections(indices, y, sigma)
    factor = 5.1794
    paired = Reflections(-indices[::-1], np.round(y[::-1]*factor, 2), np.round(sigma[::-1]*factor, 2))
    assert _companion_scale(original, paired, False) == pytest.approx(factor, abs=1e-5)
    with pytest.raises(ValueError, match='candidate basis'):
        _companion_scale(original, paired, True)


def test_resolution_export_requires_corrected_groups():
    request = dict(mode='resolve', inputs={role: role for role in ('hkl', 'experiments', 'predictions', 'profiles')},
                   settings={'pixel_probe': {'target-frames': 12}, 'pixel_export_hklf5': True})
    with pytest.raises(ValueError, match='corrected observation groups'):
        engine.initial_plan(request)
    request['inputs']['corrections'] = 'corrections.json'
    nodes = {node['id']: node for node in engine.initial_plan(request)}
    assert nodes['pixel-export']['args']['inputs'] == [engine.ref('pixel-correct', '/corrected_groups')]
    assert nodes['pixel-correct']['depends_on'] == ['pixel-fit']


def test_resolution_reports_insufficient_independent_observations(tmp_path):
    hkl = tmp_path/'few.json'
    write_reflections(Reflections([[1, 0, 0], [-1, 0, 0]], [10., 11.], [2., 2.]), hkl)
    session = tmp_path/'session'
    engine.initialize(dict(mode='resolve', inputs={'hkl': str(hkl)},
                           settings={'cell': Cell(8, 9, 10).to_dict(), 'spacegroup_candidates': []}), session)
    engine.run(session)
    with closing(Store(session)) as store:
        assert store.node('select')['result']['status'] == 'insufficient_independent_observations'
        assert store.node('final-audit')['result']['audit_evaluations'] == 0


@pytest.mark.parametrize('elements', [None, ['Fe', 'Cl', 'O']])
def test_reduce_handoff_uses_native_resolution_and_explicit_cell_composition(tmp_path, monkeypatch, elements):
    cell = Cell(6.8, 7.9, 9.1)
    atoms = [Atom('Fe1', 'Fe', [.13, .27, .66]), Atom('Cl1', 'Cl', [.32, .81, .53]),
             Atom('O1', 'O', [.21, .44, .15])]
    model = Structure(cell, atoms)
    grid = np.array(list(np.ndindex(13, 13, 13)))-6
    grid = grid[np.any(grid != 0, axis=1)]
    indices = grid[cell.d_spacing(grid) >= .9]
    values = calculated_intensities(model, indices)
    observations = Reflections(indices, values, np.maximum(1., values*.03))
    execute = engine.execute

    def known_reduction_products(ctx, tool, args):
        if tool != 'dials.run':
            return execute(ctx, tool, args)
        outputs = {}
        for role, name in DIALS_OUTPUTS[args['operation']].items():
            output = ctx.directory/name
            if role == 'ins':
                output.write_text('CELL 0.71073 6.8 7.9 9.1 90 90 90\nSFAC Fe Cl O\nUNIT 0 0 0\nHKLF 4\nEND\n')
            elif role == 'hkl':
                write_reflections(observations, output)
            else:
                output.touch()
            outputs[role] = artifact(ctx.root, output, role)
        return outputs

    monkeypatch.setattr(engine, 'execute', known_reduction_products)
    frame = tmp_path/'input.cbf'
    frame.touch()
    settings = {'composition': 'Fe Cl O'}
    if elements is not None:
        settings['elements'] = elements
    root = tmp_path/'session'
    engine.initialize(dict(mode='reduce', inputs={'image': str(frame)}, settings=settings), root)
    result = engine.run(root)
    assert all(item['status'] == 'succeeded' for item in result['nodes'])
    assert 'solve' not in {item['id'] for item in result['nodes']}
    with closing(Store(root)) as store:
        assert store.node('evidence')['dependencies'] == ['export']
        assert 'export' in store.node('review')['dependencies']
        if elements is None:
            assert store.node('select')['result']['status'] == 'supply_unit_cell_composition_or_paired_initial_model'
        else:
            assert any(seed['source'] == 'native P1 work-data initialization'
                       for seed in store.node('seed')['result']['seeds'])
