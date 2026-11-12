"""Generate clearly labelled synthetic examples and published-data requests.

Truth coordinates are used only for simulation here. The solve requests do
not contain or load the truth model. Run from an installed source checkout.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write(name, obj):
    (HERE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parallel_example(model, refinement_options):
    """Two declared initial models; no new or simulated measured intensities."""
    from ariadne.agent import engine
    alternative = model.to_dict()
    for atom, shift in zip(alternative['atoms'][1:], ([.025,-.018,.02],[-.022,.014,-.017])):
        atom['frac'] = [x+d for x,d in zip(atom['frac'],shift)]
        atom['u_iso'] *= 1.2
    write('fen4_alternative_start.json', alternative)
    nodes = []
    for key, model_key in [('refine_a','model_a'), ('refine_b','model_b')]:
        nodes.append(engine.node(key, 'native.run', {
            'request':{'action':'refine','structure':'start.json','hkl':'measured.json',
                       'options':refinement_options,'graphics':False},
            'inputs':{'start.json':engine.input_ref(model_key),
                      'measured.json':engine.input_ref('reflections')}},
            resources={'cpu_threads':1,'memory_mb':256}))
    nodes += engine.completion_nodes(['refine_a','refine_b'])
    write('parallel_native.json', {'schema':'ariadne-agent/1','mode':'custom',
        'inputs':{'model_a':'fen4_start.json','model_b':'fen4_alternative_start.json',
                  'reflections':'fen4_measured.json'},
        'scheduler':{'max_workers':2,'cpu_threads':2,'memory_mb':512,
                     'default_task_memory_mb':256,'isolate_actions':True},
        'budget':{'max_actions':4,'max_decisions':4,'wall_seconds':120,'action_seconds':60},
        'nodes':nodes})


def main():
    from ariadne_xtal.io import write_reflections, write_structure
    from ariadne_xtal.model import Reflections
    suite = load_script('demo_v08_cases', ROOT/'native/validation/run_v08_phasing_autonomy.py')
    truth, noiseless = suite.base_case()
    rng = np.random.default_rng(20261108)
    sigma = np.sqrt(2.**2 + (.025*noiseless.intensity)**2)
    observations = Reflections(noiseless.hkl, noiseless.intensity+rng.normal(size=len(sigma))*sigma, sigma)
    write_reflections(observations, HERE/'synthetic_reflections.json')
    write('autosolve.json', {'action':'autosolve','hkl':'synthetic_reflections.json',
        'cell':truth.cell.to_dict(),'elements':['C']*4,
        'phasing_options':{'algorithm':'hybrid','trials':4,'iterations':400,'keep':4},
        'refinement_options':{'max_nfev':160},'options':{'seed':1729}})
    archived = json.loads((ROOT/'native/validation/v08_phasing/overlap_input.json').read_text())
    write('overlap_groups.json', {'groups':archived['groups']})
    write('solve_observations.json', {'action':'solve-observations',
        'observations':'overlap_groups.json','cell':archived['cell'],'elements':archived['elements'],
        'phasing_options':{'algorithm':'hybrid','trials':4,'iterations':600,'keep':4,'seed':1729},
        'refinement_options':{'max_nfev':180},
        'domain_fractions':[.5,.5],'refine_domain_fractions':True,
        'phase_relation':'same_phase_supported','options':{'seed':1729}})
    write('count_pixels.json', {'data':[8,3],'profiles':[[1],[0]],'background':[[1],[1]],
        'noise':{'kind':'poisson','source':'synthetic integer photon counts',
                 'poisson_counts_verified':True,'gain':1,'readout_variance':0},
        'interval_contrasts':[[1]],'confidence_level':.95})
    write('integrate_counts.json', {'action':'integrate','problem':'count_pixels.json'})
    real = load_script('demo_fen4', ROOT/'native/validation/run_real_fen4.py')
    model, measured = real.load_case()
    write_structure(model,HERE/'fen4_start.json')
    write_reflections(measured,HERE/'fen4_measured.json')
    refine_request = {'action':'refine','structure':'fen4_start.json','hkl':'fen4_measured.json',
        'options':{'weight_a':.0282,'weight_b':.3122,'weight_cycles':20,'covariance_scale':'reduced_chi2'},
        'graphics':False}
    write('refine_fen4.json',refine_request)
    write('persistent_native.json', {'schema':'ariadne-agent/1','mode':'custom',
        'inputs':{'model':'fen4_start.json','reflections':'fen4_measured.json'},
        'nodes':[{'id':'refine','tool':'native.run','args':{'request':refine_request,
            'inputs':{'fen4_start.json':{'$input':'model'},'fen4_measured.json':{'$input':'reflections'}}}},
            {'id':'package','tool':'package','depends_on':['refine'],
             'args':{'evidence':{'$results':['refine']}}}]})
    parallel_example(model, refine_request['options'])
    write('PROVENANCE.json', {'synthetic':['synthetic_reflections.json','overlap_groups.json','count_pixels.json'],
        'truth_coordinates_passed_to_solver':False,
        'experimental':{'files':['fen4_start.json','fen4_measured.json'],
                        'doi':'10.1107/S2056989018012161','meaning':'known-model re-refinement, not ab initio'},
        'derived_initial_model':{'file':'fen4_alternative_start.json','source':'fen4_start.json',
            'nitrogen_fractional_shifts':[[.025,-.018,.02],[-.022,.014,-.017]],
            'nitrogen_u_iso_multiplier':1.2,'iron_inversion_center_unchanged':True},
        'agent_examples':['persistent_native.json','parallel_native.json'],
        'note':'No actual detector images are included in these examples.'})
    print('Generated four native requests, two persistent-agent requests and labelled input data.')


if __name__ == '__main__':
    main()
