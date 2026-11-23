"""Persistent staged comparison of space groups and exact two-domain twins."""
from __future__ import annotations

from collections import defaultdict
from inspect import getfile
from pathlib import Path
from contextlib import closing

import gemmi
import numpy as np

from ariadne_xtal.io import read_reflections, read_structure, write_reflections, write_structure
from ariadne_xtal.model import Cell
from ariadne_xtal.refinement import RefinementOptions, refine
from ariadne_xtal.scattering import calculated_intensities
from ariadne_xtal.shelx import ShelxDocument
from ariadne_xtal.symmetry import get_operations
from ariadne_xtal.twin_diagnostics import (candidate_rotations, model_group, partition_intensities,
                                          read_intensities, symmetry_evidence, symmetry_models,
                                          twin_laws, twin_screen)

from .loop import ExposureLedger
from .process import Cancelled
from .util import artifact, artifact_identity, artifact_path, atomic_json, digest, read_json


_DEFAULTS = dict(seed=1729, short_evaluations=8, full_evaluations=100, shortlist=8, short_max_reflections=2500,
                 finalists=4, inspect_rejections=1, max_cache_mib=256,
                 selection_z=1.96, min_selection_components=4, max_obliquity=3.,
                 tight_obliquity=.05, anomalous=False)
_OPTIONAL = {'cell', 'spacegroup', 'spacegroup_candidates', 'elements', 'reflection_format',
             'negative_batch_is_free', 'extra_laws', 'coverage_hkl', 'links', 'phasing_options',
             'direct_twin', 'refinement_options', 'solver', 'crosscheck', 'shortlist_band',
             'allow_friedel_averaging', 'anomalous_wavelength', 'pixel_probe', 'pixel_export_hklf5'}
_NEED_SEED_INPUT = 'supply_unit_cell_composition_or_paired_initial_model'


def validate_settings(values):
    extra = set(values) - set(_DEFAULTS) - _OPTIONAL
    if extra:
        raise ValueError(f'Unknown resolution settings: {sorted(extra)}')
    config = {**_DEFAULTS, **values}
    for key, minimum, maximum in (('seed', 0, 2**32-1), ('short_evaluations', 1, 100),
                                  ('full_evaluations', 1, 5000), ('shortlist', 1, 64), ('short_max_reflections', 20, 200_000),
                                  ('finalists', 1, 32), ('inspect_rejections', 0, 64),
                                  ('min_selection_components', 2, 10_000), ('max_cache_mib', 1, 4096)):
        value = config[key]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f'{key} must be an integer in {minimum}..{maximum}')
    if config['short_evaluations'] > config['full_evaluations'] or config['finalists'] > config['shortlist']:
        raise ValueError('Resolution budgets must grow from short to full and narrow from shortlist to finalists')
    if config.get('solver', 'native') not in {'native', 'shelxt'}:
        raise ValueError('solver may be native or shelxt')
    if not isinstance(config.get('spacegroup_candidates', []), list):
        raise ValueError('spacegroup_candidates must be a list of settings in the supplied basis')
    if (type(config['anomalous']) is not bool or
            any(type(config[key]) is not bool for key in ('negative_batch_is_free', 'crosscheck',
                                                        'allow_friedel_averaging', 'pixel_export_hklf5')
                if key in config)):
        raise ValueError('Resolution flags must be booleans')
    if not 0 < config['selection_z'] <= 5 or not 0 <= config.get('shortlist_band', .05) <= 1:
        raise ValueError('Invalid selection confidence or shortlist tolerance')
    probe = config.get('pixel_probe')
    if config.get('pixel_export_hklf5') and probe is None:
        raise ValueError('pixel_export_hklf5 requires a targeted pixel_probe and correction evidence')
    if probe is not None:
        from .backends import ANALYSES, typed_options

        if (not isinstance(probe, dict) or type(probe.get('target-frames')) is not int
                or not 1 <= probe['target-frames'] <= 100_000):
            raise ValueError('pixel_probe requires a target-frames budget in 1..100,000')
        typed_options(probe, ANALYSES['build_overlaps'][3])
    return config


def _path(ctx, record):
    return artifact_path(ctx.root, record)


def _json(ctx, name, value, role):
    path = ctx.directory/name
    atomic_json(path, value)
    return artifact(ctx.root, path, role)


def _cached(ctx, stage, records, options, compute):
    configuration = ctx.request.get('cache')
    if configuration is None:
        return compute()
    from scipy import __version__ as scipy_version
    from ariadne_xtal import scattering, symmetry, twin_diagnostics
    from .stage_cache import StageCache
    from .util import content_id

    dependencies = dict(stage=stage, inputs=[artifact_identity(record) for record in records], options=options,
                        numpy=np.__version__, scipy=scipy_version, gemmi=gemmi.__version__,
                        implementation=[content_id(Path(path)) for path in
                                        (__file__, *(getfile(module) for module in
                                                     (scattering, symmetry, twin_diagnostics)))])
    directory = (ctx.root/Path(configuration['directory']).expanduser()).resolve()

    def completed():
        result = compute()
        _check(ctx)
        return result

    result, hit = StageCache(directory, configuration.get('max_mib', 64)*1024**2).get(dependencies, completed)
    return {**result, 'stage_cache_hit': hit}


def _model(ctx, name, structure):
    path = ctx.directory/name
    write_structure(structure, path)
    return artifact(ctx.root, path, 'structural_hypothesis')


def _check(ctx):
    if ctx.store.get('cancel_requested', False):
        raise Cancelled('Cancelled during symmetry resolution')
    if ctx.remaining() <= 0:
        raise TimeoutError('Symmetry resolution exhausted its action budget')


def _control_cell(control):
    instructions = ShelxDocument.from_path(control).instructions
    cell_card = next((card for card in instructions if card.keyword == 'CELL'), None)
    if cell_card is None:
        raise ValueError('Provide the CELL instruction or the complete unit-cell parameters')
    return Cell(*map(float, cell_card.arguments[1:7]))


def _optional_anomalous(model, settings):
    if not settings['anomalous'] or not settings.get('anomalous_wavelength'):
        return model
    from ariadne_xtal.absolute import with_anomalous_factors
    return with_anomalous_factors(model, settings['anomalous_wavelength'])


def _supergroups(cell, reference, budget=8):
    """Propose settings with an explicit subgroup inclusion at the current origin."""
    base = gemmi.find_spacegroup_by_name(reference, cell.alpha, cell.gamma)
    ref = {op.triplet() for op in base.operations()}
    unit = gemmi.UnitCell(*cell.to_dict().values())
    candidates = []
    for group in gemmi.spacegroup_table():
        if group.centring_type() != base.centring_type():
            continue
        operations = group.operations()
        if len(operations) <= len(ref) or not unit.is_compatible_with_spacegroup(group, 1e-5):
            continue
        if ref < {op.triplet() for op in operations}:
            candidates.append((len(operations), group.xhm()))
    eligible = []
    for _, name in sorted(set(candidates)):
        try:
            get_operations(name, cell=cell)
        except ValueError:
            continue
        eligible.append(name)
    return eligible[:budget], max(0, len(eligible)-budget)


def prepare(ctx, args):
    settings = validate_settings(args['settings'])
    control = _path(ctx, args['ins']) if args.get('ins') else None
    model = read_structure(_path(ctx, args['structure'])) if args.get('structure') else None
    if control is None and args.get('structure') and _path(ctx, args['structure']).suffix.lower() in {'.res', '.ins'}:
        control = _path(ctx, args['structure'])
    cell = Cell.from_dict(settings['cell']) if 'cell' in settings else model.cell if model else (
        _control_cell(control) if control is not None else None)
    if cell is None:
        raise ValueError('Provide a structure, a control file or the six unit-cell parameters')
    if model is not None and not np.allclose(model.cell.matrix, cell.matrix, rtol=1e-5, atol=1e-6):
        raise ValueError('The structural hypothesis must use the declared common cell and basis')
    group = settings.get('spacegroup', model_group(model) if model else 'P 1')
    get_operations(group, cell=cell)
    r = read_intensities(_path(ctx, args['hkl']), settings.get('reflection_format'), control,
                         negative_batch_is_free=settings.get('negative_batch_is_free', False))
    if np.any(np.all(r.hkl == 0, axis=1)):
        raise ValueError('000 is a terminator or direct beam, not a measured reflection')
    names, omitted = (settings['spacegroup_candidates'], 0) if 'spacegroup_candidates' in settings else _supergroups(cell, group)
    if 'spacegroup_candidates' not in settings and group != 'P 1':
        names = ['P 1', *names]
    names = list(dict.fromkeys([group, *names]))
    if len(names) > 32:
        raise ValueError('At most 32 common-basis space-group settings')
    options = {key: settings[key] for key in ('max_obliquity', 'tight_obliquity', 'anomalous')}
    extra_laws = list(settings.get('extra_laws', ()))
    if settings.get('direct_twin'):
        extra_laws.append(settings['direct_twin']['law'])
    options['extra_laws'] = extra_laws
    laws_by_group, compatible = {}, []
    for name in names:
        try:
            get_operations(name, cell=cell)
        except ValueError:
            # Preserve a row for the diagnostic, without treating its
            # incompatible metric as an exact intensity transform.
            continue
        laws_by_group[name] = twin_laws(cell, name, **options)
        compatible.append(name)
    rotations = candidate_rotations(cell, compatible, [law for rows in laws_by_group.values() for law in rows])
    roles, partition = partition_intensities(r, rotations, links=settings.get('links', ()), seed=settings['seed'])
    metadata = {'cell': cell.to_dict(), 'settings': settings, 'groups': names, 'laws_by_group': laws_by_group,
                'candidate_budget_deferred': omitted, 'source': args['hkl']}
    if roles is None:
        return {**metadata, 'status': 'insufficient_independent_observations', 'partition': partition}
    component_ids = partition.pop('component_ids')
    for role in ('work', 'selection', 'audit'):
        mask = roles == role
        data = r.subset(mask)
        data.free_flags = None
        filename = ctx.directory/f'{role}.json'
        write_reflections(data, filename)
        metadata[role] = artifact(ctx.root, filename, f'{role}_intensities')
        metadata[f'{role}_components'] = _json(ctx, f'{role}-components.json', component_ids[mask].tolist(),
                                              f'{role}_components')
    work = read_reflections(_path(ctx, metadata['work']))
    cache_options = dict(cell=cell.to_dict(), spacegroup=group, candidates=names,
                         coverage_hkl=settings.get('coverage_hkl'), **options)
    evidence = _cached(ctx, 'symmetry-evidence', [metadata['work']], cache_options,
                       lambda: symmetry_evidence(cell, work, group, candidates=names,
                                                 coverage_hkl=settings.get('coverage_hkl'), **options))
    if len(work) > settings['short_max_reflections']:
        work_ids = component_ids[roles == 'work']
        rng = np.random.default_rng(settings['seed'])
        order = rng.permutation(np.unique(work_ids))
        counts = dict(zip(*np.unique(work_ids, return_counts=True)))
        cumulative = np.cumsum([counts[group_id] for group_id in order])
        chosen = order[:max(1, int(np.searchsorted(cumulative, settings['short_max_reflections'], side='right')))]
        path = ctx.directory/'short-work.json'
        write_reflections(work.subset(np.isin(work_ids, chosen)), path)
        metadata['short_work'] = artifact(ctx.root, path, 'short_refinement_intensities')
    metadata.update(status='ready', partition=partition, reference=group,
                    diagnostic=_json(ctx, 'symmetry-evidence.json', evidence, 'symmetry_evidence'),
                    n_detector_laws=sum(len([x for x in rows if not x['exact']]) for rows in laws_by_group.values()))
    if model is not None:
        metadata['structure'] = _model(ctx, 'initial.json', _optional_anomalous(model, settings))
    return metadata


def _shelxt_model(path):
    document = ShelxDocument.from_path(path)
    # These cards are generated by SHELXT for later SHELXL cycles or reports.
    # Build a coordinate seed while keeping the untouched RES/HKL pair.
    shell_commands = {'L.S.', 'CGLS', 'BOND', 'LIST', 'FMAP', 'PLAN', 'ACTA'}
    active = []
    for card in document.instructions:
        if card.keyword in shell_commands:
            continue
        if card.kind == 'atom' and len(card.arguments) == 7:
            # SHELXT appends the peak height to its isotropic coordinate card.
            float(card.arguments[-1])
            active.append(' '.join((card.keyword, *card.arguments[:-1])))
        else:
            active.append(card.logical_text)
    return ShelxDocument.from_text('\n'.join(active)+'\n').to_structure()


def _companion_scale(source, partner, anomalous):
    """Certify an unchanged reciprocal basis across a SHELXT intensity rescale."""
    def bucket(reflections):
        result = defaultdict(list)
        for h, value, sigma in zip(reflections.hkl, reflections.intensity, reflections.sigma):
            key = tuple(h) if anomalous else min(tuple(h), tuple(-h))
            result[key].append((value, sigma))
        return result

    reference, candidate = bucket(source), bucket(partner)
    if len(partner) < 10 or any(key not in reference for key in candidate):
        raise ValueError('Work observations do not certify the SHELXT candidate basis')
    x, y = [], []
    for key, entries in candidate.items():
        if len(entries) == len(reference[key]):
            x.extend(sorted(reference[key]))
            y.extend(sorted(entries))
    if len(x) < 10:
        raise ValueError('Too few matched readings to certify the SHELXT candidate basis')
    x, y = np.asarray(x), np.asarray(y)
    # The output scale must explain both intensity and its s.u. in the same
    # units. Use the largest readings to minimize fixed-column rounding error.
    weights = np.abs(x[:, 0]) + x[:, 1]
    strong = weights >= np.quantile(weights, .8)
    scale = float(np.sum(x[strong]*y[strong]) / np.sum(x[strong]**2))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('The SHELXT companion has no consistent positive scale')
    for key, entries in candidate.items():
        available = list(reference[key])
        for row in sorted(entries):
            residuals = np.max(np.abs(np.asarray(available)*scale - row), axis=1) if available else []
            match = int(np.argmin(residuals)) if len(residuals) else -1
            if match < 0 or residuals[match] > .012+2e-7*max(abs(row[0]), row[1]):
                raise ValueError('Paired intensities do not certify the SHELXT candidate basis after rescaling')
            available.pop(match)
    return scale


def _seed_shelxt(ctx, preparation, control):
    from .backends import solve

    control_path = _path(ctx, control)
    r = read_reflections(_path(ctx, preparation['work']))
    if not np.allclose(_control_cell(control_path).matrix, Cell.from_dict(preparation['cell']).matrix,
                       rtol=1e-5, atol=1e-6):
        raise ValueError('Laue-group control and comparison reflections must use the same declared cell')
    hkl_path = ctx.directory/'work-input.hkl'
    write_reflections(r, hkl_path)
    result = solve(ctx, {'ins': control, 'hkl': artifact(ctx.root, hkl_path, 'phase_work_hklf4')})
    # Verify the paired work observations in the same units and basis before
    # using a SHELXT coordinate seed with the original signed intensities.
    source = read_intensities(hkl_path, 'hklf4')
    common_cell = Cell.from_dict(preparation['cell'])
    seeds, rejected = [], []
    for i, candidate in enumerate(result['candidates']):
        if not candidate['eligible_for_refinement']:
            rejected.append({'candidate': candidate, 'reason': candidate['reason']})
            continue
        try:
            model = _shelxt_model(_path(ctx, candidate['res']))
            partner = read_intensities(_path(ctx, candidate['hkl']), 'hklf4', _path(ctx, candidate['res']))
            if not np.allclose(model.cell.matrix, common_cell.matrix, rtol=1e-5, atol=1e-6):
                raise ValueError('candidate needs an explicit change of cell or setting')
            scale = _companion_scale(source, partner, preparation['settings']['anomalous'])
            model = _optional_anomalous(model, preparation['settings'])
        except (ValueError, RuntimeError) as exc:
            rejected.append({'candidate': candidate, 'reason': str(exc)})
            continue
        seeds.append({'model': _model(ctx, f'shelxt-{i}.json', model), 'source': 'SHELXT work data',
                      'paired': candidate, 'companion_intensity_scale': scale})
    return {'seeds': seeds, 'deferred': rejected, 'source': result}


def seed(ctx, args):
    preparation = args['preparation']
    if preparation['status'] != 'ready':
        return {'seeds': [], 'status': preparation['status']}
    settings = preparation['settings']
    if args.get('ins'):
        return _seed_shelxt(ctx, preparation, args['ins'])
    if preparation.get('structure'):
        return {'seeds': [{'model': preparation['structure'], 'source': 'provided initial model'}]}
    if not settings.get('elements'):
        return {'seeds': [], 'status': _NEED_SEED_INPUT}
    if settings['anomalous'] and not settings.get('allow_friedel_averaging', False):
        raise ValueError('Native phase initialization needs declared Friedel averaging or a supplied model')
    from ariadne_xtal.phasing import PhasingOptions, solve_p1

    r = read_reflections(_path(ctx, preparation['work']))
    cell = Cell.from_dict(preparation['cell'])
    options = {'trials': 8, 'iterations': 200, 'algorithm': 'hybrid', **settings.get('phasing_options', {})}
    if options['trials']*options['iterations'] > 100_000:
        raise ValueError('Phase initialization exceeds 100,000 total iterations')
    result = solve_p1(cell, r, settings['elements'], PhasingOptions(**options))
    seeds = [{'model': _model(ctx, f'phase-{i}.json', _optional_anomalous(row['structure'], settings)),
              'source': 'native P1 work-data initialization'} for i, row in enumerate(result.candidates)]
    report = None
    if settings.get('direct_twin'):
        from ariadne_xtal.twin_phasing import direct_twin_phasing

        config = dict(settings['direct_twin'])
        law = config.pop('law')
        phase = direct_twin_phasing(cell, r, settings['elements'], law, options, **config)
        for i, candidate in enumerate(phase.pop('candidates')):
            seeds.append({'model': _model(ctx, f'twin-phase-{i}.json', candidate['structure']),
                          'source': 'joint twin-mixture work-data initialization'})
        phase['audited_rejections'] = [{k: v for k, v in row.items() if k != 'structure'}
                                       for row in phase['audited_rejections']]
        report = _json(ctx, 'twin-phasing.json', phase, 'twin_phasing_search')
    return {'seeds': seeds, 'phasing_report': report}


def _family(group, matrix):
    return (group, 'single' if matrix is None else tuple(np.asarray(matrix).ravel().tolist()))


def screen(ctx, args):
    preparation = args['preparation']
    if preparation['status'] != 'ready':
        return {'status': preparation['status'], 'shortlist': []}
    settings = preparation['settings']
    seeds = [s for stage in args['seeds'] for s in stage.get('seeds', [])]
    if not seeds and any(stage.get('status') == _NEED_SEED_INPUT for stage in args['seeds']):
        return {'status': _NEED_SEED_INPUT, 'shortlist': [], 'diagnostic': preparation['diagnostic']}
    work = read_reflections(_path(ctx, preparation['work']))
    model_cache = {}
    projected = []
    for i, entry in enumerate(seeds):
        _check(ctx)
        model = read_structure(_path(ctx, entry['model']))
        model_cache[i] = model
        projected.append((i, entry))
        try:
            proposals = symmetry_models(model, work, preparation['groups'])
        except (ValueError, RuntimeError) as exc:
            projected[-1] = (i, {**entry, 'projection_diagnostic': str(exc)})
            continue
        for j, proposal in enumerate(proposals):
            if not proposal.get('eligible'):
                continue
            hypothesis = proposal['structure']
            key = len(seeds)+len(model_cache)
            model_cache[key] = hypothesis
            projected.append((key, {'model': _model(ctx, f'projected-{i}-{j}.json', hypothesis),
                                    'source': entry['source'], 'projection': proposal['projection']}))

    rows, best_family = [], {}
    for i, entry in projected:
        _check(ctx)
        model = model_cache[i]
        group = model_group(model)
        # A discovered setting can be screened only if its rotations were
        # closed into the predeclared evidence partition.
        if group not in {gemmi.find_spacegroup_by_name(name, model.cell.alpha, model.cell.gamma).xhm()
                         for name in preparation['laws_by_group']}:
            rows.append({'group': group, 'model': entry['model'], 'eligible': False,
                         'reason': 'setting outside the frozen comparison; independent observations required'})
            continue
        matches = next(rows for name, rows in preparation['laws_by_group'].items()
                       if gemmi.find_spacegroup_by_name(name, model.cell.alpha, model.cell.gamma).xhm() == group)
        laws = matches
        data = _cached(ctx, 'twin-screen', [entry['model'], preparation['work']],
                       {'laws': laws, 'max_cache_mib': settings['max_cache_mib']},
                       lambda: twin_screen(model, work, laws, max_cache_mib=settings['max_cache_mib']))
        for row in data['rows']:
            trial = {**entry, **row, 'group': group, 'id': f'model-{len(rows):04d}',
                     'rmse': float(np.sqrt(row['chi_square']/len(work))), 'eligible': True}
            rows.append(trial)
            key = _family(group, row['matrix'])
            if key not in best_family or trial['rmse'] < best_family[key]['rmse']:
                best_family[key] = trial
        projected_stats = data['cache']
        rows[-1]['batch_statistics'] = {**projected_stats, 'stage_cache_hit': data.get('stage_cache_hit', False)}
    ranked = sorted(best_family.values(), key=lambda row: (row['rmse'], row['id']))
    if not ranked:
        return {'status': 'no_common_basis_initial_model', 'shortlist': [],
                'report': _json(ctx, 'screen.json', {'rows': rows}, 'twin_screen')}
    representatives = []
    for group in dict.fromkeys(row['group'] for row in ranked):
        representatives.append(next(row for row in ranked if row['group'] == group and row['matrix'] is None))
    boundary = ranked[0]['rmse']*(1+settings.get('shortlist_band', .05))
    competitive = [row for row in ranked if row['rmse'] <= boundary and row['fraction_status'] != 'unidentifiable']
    rest = [row for row in ranked if row['fraction_status'] != 'unidentifiable']
    by_id = {row['id']: row for row in [*representatives, *competitive, *rest]}
    chosen = list(by_id.values())[:max(settings['shortlist'], len(representatives))]
    skipped = [row for row in ranked if row['id'] not in {r['id'] for r in chosen}]
    rng = np.random.default_rng(settings['seed'])
    for i in rng.choice(len(skipped), min(len(skipped), settings['inspect_rejections']), replace=False):
        chosen.append({**skipped[i], 'screen_rejection_audit': True})
    return {'status': 'screened', 'shortlist': chosen,
            'report': _json(ctx, 'screen.json', {'rows': rows, 'family_count': len(ranked),
                                               'shortlisted': [r['id'] for r in chosen]}, 'twin_screen')}


def _losses(model, reflections, branch):
    law = branch['matrix']
    p = calculated_intensities(model, reflections.hkl, scale=branch['scale'],
                               twin_laws=None if law is None else [law], twin_fractions=branch.get('fractions'))
    return ((reflections.intensity - p)/reflections.sigma)**2


def fit(ctx, args):
    prep, candidate, stage = args['preparation'], args['candidate'], args['stage']
    settings = prep['settings']
    model = read_structure(_path(ctx, candidate['model']))
    work_artifact = prep.get('short_work', prep['work']) if stage == 'short' else prep['work']
    work = read_reflections(_path(ctx, work_artifact))
    target = settings['short_evaluations' if stage == 'short' else 'full_evaluations']
    options = dict(settings.get('refinement_options', {}))
    if set(options) & {'max_nfev', 'scale', 'twin_laws', 'twin_fractions', 'refine_twin_fractions', 'refine_scale',
                       'weight_a', 'weight_b', 'verbose'}:
        raise ValueError('The paired comparison controls scale, twin variables, weights and evaluation budget')
    fraction = candidate.get('fraction')
    law = candidate['matrix']
    if law is not None:
        fraction = np.clip(.25 if fraction is None else fraction, 1e-5, 1-1e-5)
        options.update(twin_laws=[law], twin_fractions=[1.-fraction, fraction], refine_twin_fractions=True)
    result = refine(model, work, RefinementOptions(max_nfev=target, scale=max(candidate['scale'], 1e-12), **options))
    branch = {key: candidate[key] for key in ('id', 'group', 'matrix', 'source', 'paired', 'projection')
              if key in candidate}
    branch.update(work_observations=len(work), model=_model(ctx, 'refined.json', result.structure), scale=result.scale,
                  fractions=result.twin_fractions.tolist() if law is not None else None,
                  fraction=float(result.twin_fractions[1]) if law is not None else 0.,
                  short_rmse=float(np.sqrt(_losses(result.structure, work, {**branch, 'scale': result.scale,
                      'fractions': result.twin_fractions.tolist() if law is not None else None}).mean())),
                  converged=bool(result.success), identifiable=bool(result.metrics['identifiable']),
                  screen_rejection_audit=candidate.get('screen_rejection_audit', False))
    selection = read_reflections(_path(ctx, prep['selection']))
    component_ids = read_json(_path(ctx, prep['selection_components']))
    losses = _losses(result.structure, selection, branch)
    components = defaultdict(list)
    for group_id, error in zip(component_ids, losses):
        components[group_id].append(float(error))
    component_losses = {str(key): float(np.mean(value)) for key, value in components.items()}
    branch.update(selection_mse=float(np.mean(list(component_losses.values()))),
                  scores=_json(ctx, 'selection-scores.json', component_losses, 'selection_component_scores'))
    if stage == 'full':
        branch['structure_cif'] = _model(ctx, 'refined.cif', result.structure)
        if law is not None:
            screened = twin_screen(result.structure, work, [law], max_cache_mib=settings['max_cache_mib'])
            branch['conditional_fraction_profile'] = screened['rows'][1]
    return branch


def promote(ctx, args):
    settings = args['preparation']['settings']
    valid = [row for row in args['fits'] if not row.get('execution_failed')]
    ranked = sorted(valid, key=lambda row: (row['selection_mse'], row['id']))
    # Check all genuine model classes at least once. A defensible runner-up
    # must be allowed to survive even if the soft budget is tight.
    reference = args['preparation'].get('reference')
    required = [next((r for r in ranked if r['group'] == g and r['matrix'] is None), None)
                for g in dict.fromkeys(r['group'] for r in ranked)]
    if any(r['matrix'] is not None for r in ranked):
        required.append(next(r for r in ranked if r['matrix'] is not None))
    selected = [row for row in required if row is not None]
    chosen = {row['id']: row for row in [*selected, *ranked]}
    limit = max(len(selected), settings['finalists'])
    finalists = list(chosen.values())[:limit]
    inspected = [row for row in ranked if row['screen_rejection_audit'] and row not in finalists]
    finalists.extend(inspected[:settings['inspect_rejections']])
    return {'finalists': finalists, 'reference': reference,
            'short_fits': len(valid), 'short_failures': len(args['fits'])-len(valid)}


def select(ctx, args):
    preparation = args['preparation']
    if preparation['status'] != 'ready':
        return {'status': preparation['status'], 'reason': preparation['partition'].get('reason'),
                'candidates': []}
    screen = args.get('screen')
    if screen is not None and screen.get('status') in {'no_common_basis_initial_model', _NEED_SEED_INPUT}:
        return {'status': screen['status'], 'candidates': []}
    settings = preparation['settings']
    valid = [row for row in args['fits'] if not row.get('execution_failed')]
    eligible = [row for row in valid if row['converged'] and row['identifiable']]
    ranked = sorted(eligible, key=lambda row: (row['selection_mse'], row['id']))
    if not ranked:
        return {'status': 'no_converged_identifiable_model', 'candidates': valid}
    best = ranked[0]
    profile = best.get('conditional_fraction_profile', {})
    best_errors = read_json(_path(ctx, best['scores']))
    rivals, ties = [], []
    for candidate in ranked[1:]:
        errors = read_json(_path(ctx, candidate['scores']))
        if best_errors.keys() != errors.keys():
            raise ValueError('Both models must predict the same sealed selection components')
        gap = np.array([errors[key]-value for key, value in best_errors.items()])
        se = float(np.std(gap, ddof=1)/np.sqrt(len(gap))) if len(gap) > 1 else None
        close = se is None or float(gap.mean()) <= settings['selection_z']*se
        rivals.append({'id': candidate['id'], 'mean_loss_gap': float(gap.mean()),
                       'paired_standard_error': se, 'competitive': close})
        if close:
            ties.append(candidate['id'])
    underpowered = len(best_errors) < settings['min_selection_components']
    boundary = bool(profile and (profile['rank'] < 2 or 0. in profile['fraction_interval']
                                  or 1. in profile['fraction_interval']))
    incomplete = len(eligible) != len(args['fits'])
    status = ('indistinguishable' if ties or underpowered or boundary else
              'comparison_incomplete' if incomplete else 'preferred_for_audit')
    return {'status': status, 'model_eligible': status == 'preferred_for_audit', 'preferred': best, 'contenders': ties, 'paired_comparisons': rivals,
            'selection_components': len(best_errors), 'fraction_boundary': boundary,
            'incomplete_comparison': incomplete, 'candidates': valid,
            'selection_scope': 'component-weighted loss; paired standard errors are descriptive'}


def audit(ctx, args):
    frozen = args['selection']
    if frozen['status'] != 'preferred_for_audit':
        return {'status': frozen['status'], 'audit_evaluations': 0}
    prep = args['preparation']
    candidate = frozen['preferred']
    _check(ctx)
    model = read_structure(_path(ctx, candidate['model']))
    dataset = artifact_identity(prep['source'])
    parameters = digest({key: candidate[key] for key in ('scale', 'matrix', 'fractions')})
    frozen_key = digest({'model': artifact_identity(candidate['model']), 'parameters': parameters,
                         'holdout': artifact_identity(prep['audit'])})
    # Session siblings share the ledger. A rerun can see the same frozen
    # result, while a changed model cannot tune against the opened rows.
    with closing(ExposureLedger(ctx.root.parent/'symmetry-audit.sqlite')) as ledger:
        existing = ledger.db.execute('SELECT frozen, result FROM exposures WHERE dataset=?', (dataset,)).fetchone()
        if existing and existing[0] == frozen_key and existing[1] is not None:
            import json
            return {**json.loads(existing[1]), 'audit_reused': True, 'audit_evaluations': 0}
        ledger.claim(dataset, frozen_key, ctx.root.name)
        reflections = read_reflections(_path(ctx, prep['audit']))
        errors = _losses(model, reflections, candidate)
        result = {'status': 'audited_hypothesis', 'audit_evaluations': 1, 'candidate': candidate['id'],
                  'n': len(errors), 'sigma_rmse': float(np.sqrt(np.mean(errors))),
                  'independent_components': prep['partition']['component_counts']['audit']}
        import json
        ledger.complete(dataset, json.dumps(result))
    return result


def crosscheck(ctx, args):
    import shutil
    import sys

    from .backends import invoke, plugin_root
    from .process import executable

    model = args['selection'].get('preferred', {}).get('structure_cif')
    if model is None and not args.get('fcf'):
        return {'status': 'no_structural_hypothesis', 'logs': {}}
    program = executable('platon', ctx.executables)
    logs = {}
    if model is not None:
        staged = ctx.directory/'candidate.cif'
        shutil.copyfile(_path(ctx, model), staged)
        invoke(ctx, [program, '-m', staged.name], label='addsym')
        logs['addsym'] = artifact(ctx.root, ctx.directory/'addsym.stdout.log', 'platon_addsym')
    if args.get('fcf'):
        original = _path(ctx, args['fcf'])
        table = ctx.directory/'observed.fcf'
        shutil.copyfile(original, table)
        if args.get('fcf_control'):
            script = plugin_root()/'skills/solve-single-crystal/scripts/fix_fcf_for_twinrotmat.py'
            invocation = [sys.executable, str(script), str(table), str(_path(ctx, args['fcf_control'])), '--ready-ok']
            invoke(ctx, invocation, label='prepare-twinrotmat')
            patched = ctx.directory/'observed_tw.fcf'
            if patched.is_file():
                table = patched
        invoke(ctx, [program, '-T1', table.name], label='twinrotmat')
        logs['twinrotmat'] = artifact(ctx.root, ctx.directory/'twinrotmat.stdout.log', 'platon_twinrotmat')
    return {'status': 'diagnostic', 'logs': logs}


def execute(ctx, tool, args):
    stages = {'resolve.prepare': prepare, 'resolve.seed': seed, 'resolve.screen': screen,
              'resolve.fit': fit, 'resolve.promote': promote, 'resolve.select': select,
              'resolve.audit': audit, 'resolve.crosscheck': crosscheck}
    return stages[tool](ctx, args)
