"""Measure prediction, joint refinement, or the dense/operator memory trade-off."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time


def peak_memory_mib():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD),
                        ('peak_working_set', ctypes.c_size_t), ('remaining', ctypes.c_size_t * 7)]

        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not get_info(process(), ctypes.byref(counters), counters.cb):
            raise OSError('Could not read the peak Windows working set')
        return counters.peak_working_set / 1024**2
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if platform.system() == 'Darwin' else 1024)


def make_model(atoms, rng):
    from ariadne_xtal.model import Atom, Cell, Structure

    values = rng.uniform(.05, .95, (atoms, 3))
    return Structure(Cell(18., 18., 24., 90., 90., 90.), [
        Atom(f'{element}{i+1}', element, xyz, u_iso=.025+i*.0002)
        for i, (xyz, element) in enumerate(zip(values, ['Zn', 'O', 'C', 'N'] * ((atoms+3)//4)))
    ])


def workload(args):
    import numpy as np
    from ariadne_xtal.derivatives import DiffractionPredictor
    from ariadne_xtal.joint_refinement import GaussianStructureGroup, refine_observations
    from ariadne_xtal.model import Reflections, Structure
    from ariadne_xtal.refinement import RefinementOptions, _Parameterization, refine
    from ariadne_xtal.scattering import calculated_intensities

    rng = np.random.default_rng(15203)
    if args.case == 'predictor':
        if args.structure:
            from ariadne_xtal.io import read_reflections, read_structure

            model = read_structure(args.structure)
            observed = read_reflections(args.hkl)
            hkl = observed.hkl
            if args.d_min:
                hkl = hkl[model.cell.stol_sq(hkl) <= .25 / args.d_min**2]
            if len(hkl) < args.reflections:
                raise ValueError(f'Only {len(hkl)} reflections satisfy selected data and resolution')
            hkl = hkl[np.linspace(0, len(hkl)-1, args.reflections, dtype=int)]
        else:
            model = make_model(args.atoms, rng)
            hkl = rng.integers(-24, 25, size=(args.reflections, 3))
        parameters = _Parameterization(model, RefinementOptions(), 1.2, np.ones(1), [])
        predictor = DiffractionPredictor(parameters, hkl, block_size=args.block_size, backend=args.backend)
        values = np.asarray(parameters.start)
        index = 0

        def run():
            nonlocal index
            index += 1
            x = values.copy()
            x[0] += index * 1e-7  # Prevent the same-parameter result cache from masking the kernel.
            _, intensity, jac = predictor.evaluate(x)
            return {'intensity_sum': float(intensity.sum()), 'jacobian_norm': float(np.linalg.norm(jac)),
                    'reflection_workers': getattr(predictor, 'reflection_workers', 1),
                    'parameters': len(values), 'atoms': len(model.atoms)}

        return run

    if args.case == 'joint':
        truth = make_model(3, rng)
        start = Structure.from_dict(truth.to_dict())
        start.atoms[1].frac += [.004, -.003, .002]
        miller = np.asarray([(h, k, l) for h in range(1, 11) for k in range(-7, 8) for l in range(-7, 8)])
        if args.reflections > len(miller):
            raise ValueError('joint fixture supports at most 2250 groups')
        hkl = miller[:args.reflections]
        transformed = hkl[:, [1, 0, 2]]
        factors = np.column_stack([calculated_intensities(truth, hkl, scale=1.7)*.72,
                                   calculated_intensities(truth, transformed, scale=1.7)*.28])
        groups = []
        for j, (h, t, pair) in enumerate(zip(hkl, transformed, factors)):
            design = (np.array([[.19, .19]]) if j % 3 == 0 else np.array([[.34, .18], [.0, .21]]))
            z = design @ pair + rng.normal(0, 1., len(design))
            groups.append(GaussianStructureGroup([h, t], design, z, coefficients=[1, 1], domain_ids=[0, 1]))

        def run():
            fit = refine_observations(start, groups, RefinementOptions(max_nfev=100),
                                      domain_fractions=[.7, .3], refine_domain_fractions=True,
                                      phase_relation='same_phase_supported')
            return {'success': fit['success'], 'objective': fit['objective'], 'rank': fit['jacobian_rank'],
                    'assembly_evaluations': fit['evaluation_count'],
                    'diffraction_evaluations': fit['diffraction_evaluations'],
                    'cache_hits': fit.get('joint_evaluation_cache_hits', 0)}

        return run

    model = make_model(args.atoms, rng)
    hkl = rng.integers(-24, 25, size=(args.reflections, 3))
    hkl[np.all(hkl == 0, axis=1)] = [1, 0, 0]
    intensity = calculated_intensities(model, hkl, scale=1.)
    data = Reflections(hkl, intensity, np.ones(len(hkl)))

    def run():
        fit = refine(model, data, RefinementOptions(scale=1., max_nfev=10,
                                                   reflection_block_size=args.block_size,
                                                   jacobian_layout=args.layout))
        return {'success': fit.success, 'objective': fit.metrics['chi_squared'],
                'rank': fit.metrics['jacobian_rank'], 'parameters': len(fit.parameters)}

    return run


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('case', choices=['predictor', 'joint', 'layout'])
    cli.add_argument('--atoms', type=int, default=80)
    cli.add_argument('--reflections', type=int, default=8192)
    cli.add_argument('--block-size', type=int, default=1024)
    cli.add_argument('--threads', type=int, default=1)
    cli.add_argument('--backend', choices=['cpp', 'python'], default='cpp')
    cli.add_argument('--layout', choices=['dense', 'operator'], default='dense')
    cli.add_argument('--structure', type=Path, help='Measured benchmark structural CIF')
    cli.add_argument('--hkl', type=Path, help='Matching experimental HKLF 4 reflections')
    cli.add_argument('--d-min', type=float, help='Resolution cutoff for a supplied experimental table')
    cli.add_argument('--repeat', type=int, default=5)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if min(args.atoms, args.reflections, args.block_size, args.threads, args.repeat) < 1:
        cli.error('benchmark sizes must be positive')
    if bool(args.structure) != bool(args.hkl) or (args.structure and args.case != 'predictor'):
        cli.error('supply structure and hkl together for the predictor case')
    os.environ.update(ARIADNE_CPU_THREADS=str(args.threads), ARIADNE_NUMERIC_BACKEND=args.backend,
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      VECLIB_MAXIMUM_THREADS='1')
    import ariadne_xtal
    import numpy
    import scipy

    run = workload(args)
    run()
    elapsed, results = [], []
    for _ in range(args.repeat):
        before = time.perf_counter()
        result = run()
        elapsed.append(time.perf_counter() - before)
        if result.get('success') is False:
            raise RuntimeError(f'Benchmark calculation did not converge: {result}')
        results.append(result)
    payload = dict(case=args.case, atoms=3 if args.case == 'joint' else results[-1].get('atoms', args.atoms),
                   reflections_or_groups=args.reflections, backend=args.backend, threads=args.threads,
                   measured_inputs=bool(args.structure), d_min_angstrom=args.d_min,
                   block_size=args.block_size, layout=args.layout, wall_seconds=elapsed,
                   median_seconds=statistics.median(elapsed), peak_process_memory_mib=peak_memory_mib(),
                   results=results, platform=platform.platform(), python=platform.python_version(),
                   numpy=numpy.__version__, scipy=scipy.__version__, module=ariadne_xtal.__file__)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
