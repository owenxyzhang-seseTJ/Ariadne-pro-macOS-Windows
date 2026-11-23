"""Twinning and symmetry diagnostics on a declared common reciprocal basis."""
from __future__ import annotations

import copy
from fractions import Fraction
from math import lcm

import gemmi
import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2

from .autonomy import evidence_partition
from .io import read_hkl, read_reflections
from .model import Reflections, Structure
from .scattering import structure_factors
from .spacegroup_search import propose_spacegroups
from .symmetry import get_operations, operation_parts, rank_spacegroups, systematic_absences, unique_sites


IDENTITY = np.eye(3, dtype=np.int64)


def read_intensities(path, reflection_format=None, control=None, *, negative_batch_is_free=False):
    """Read a typed observation table, keeping component groups out of HKLF4."""
    from pathlib import Path
    import re

    path = Path(path)
    declared = None
    if control is not None:
        records = re.findall(r'^\s*HKLF\s+(\d+)\b', Path(control).read_text(encoding='utf-8'), re.M | re.I)
        if len(records) > 1:
            raise ValueError("Require a single HKLF instruction")
        declared = records[0] if records else None
        if declared not in {None, '4'}:
            raise ValueError("This intensity route requires HKLF 4. Use indexed observation groups for HKLF 5.")
    if reflection_format == 'cif-f2':
        if negative_batch_is_free:
            raise ValueError("CIF uses measured intensities and explicit HKL, without HKLF4 batch flags")
        document = gemmi.cif.read_file(str(path))
        candidates = []
        for block in document:
            for prefix in ('_refln_', '_refln.'):
                for data, sigma in (('F_squared_meas', 'F_squared_sigma'), ('intensity_meas', 'intensity_sigma')):
                    table = block.find(prefix, ['index_h', 'index_k', 'index_l', data, sigma])
                    if len(table):
                        candidates.append([[gemmi.cif.as_number(value) for value in row] for row in table])
        if len(candidates) != 1:
            raise ValueError("Require one CIF loop of measured signed intensities and uncertainties")
        values = np.asarray(candidates[0])
        return Reflections(values[:, :3], values[:, 3], values[:, 4])
    if path.suffix.lower() == '.json':
        if reflection_format not in {None, 'json'} or negative_batch_is_free:
            raise ValueError("JSON intensities preserve their explicit free flags")
        return read_reflections(path)
    if reflection_format not in {None, 'hklf4'} or (reflection_format is None and declared != '4'):
        raise ValueError("Declare HKLF 4 or provide its matching SHELX control file")
    return read_hkl(path, negative_batch_is_free=negative_batch_is_free)


def _group(name, cell):
    group = gemmi.find_spacegroup_by_name(name, cell.alpha, cell.gamma)
    if group is None:
        raise ValueError(f"Unknown space-group setting: {name}")
    return group


def model_group(model):
    operations = gemmi.GroupOps(model.operations)
    group = gemmi.find_spacegroup_by_ops(operations)
    if group is None:
        raise ValueError("Supply a recognized setting for these explicit structure operations")
    return group.xhm()


def reciprocal_operation(op):
    # apply_to_hkl() is the reference on integral indices.
    return np.array([op.apply_to_hkl(row.tolist()) for row in IDENTITY], dtype=np.int64).T


def _matrix(value):
    m = np.asarray(value, dtype=float)
    if (m.shape != (3, 3) or not np.isfinite(m).all() or np.any(np.abs(m) > 1_000)
            or not np.array_equal(m, np.rint(m)) or abs(round(np.linalg.det(m))) != 1):
        raise ValueError("A lattice twin operator must be an integral unimodular 3 by 3 matrix")
    return m.astype(np.int64)


def _rational_detector_law(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.any(np.abs(matrix) > 1_000)
            or not np.isclose(abs(np.linalg.det(matrix)), 1., rtol=0, atol=1e-8)):
        raise ValueError("Supply a volume-preserving 3 by 3 reciprocal transformation")
    rationals = [Fraction(x).limit_denominator(24) for x in matrix.ravel()]
    denominator = lcm(*(v.denominator for v in rationals))
    if denominator > 24 or not np.allclose([float(v) for v in rationals], matrix.ravel(), rtol=0, atol=1e-8):
        raise ValueError("Use a declared rational parent-lattice law with denominator at most 24")
    numerator = np.array([int(v*denominator) for v in rationals], dtype=np.int64).reshape(3, 3)
    return numerator, denominator


def exact_metric(cell, law):
    metric = cell.reciprocal.T @ cell.reciprocal
    transformed = law.T @ metric @ law
    return bool(np.allclose(transformed, metric, atol=1e-9, rtol=1e-5)
                and np.allclose(transformed, metric, rtol=2e-6,
                                atol=2e-6*max(np.max(np.abs(metric)), 1e-12)))


def twin_laws(cell, spacegroup, *, max_obliquity=3., tight_obliquity=.05,
              anomalous=False, extra_laws=()):
    """Enumerate nonredundant exact/approximate lattice operations.

    A representative is equivalent under left multiplication by an intensity
    symmetry of the structural phase. Friedel inversion is added to that
    equivalence only when it can be assumed for the measured intensities.
    """
    if not 0 <= tight_obliquity <= max_obliquity <= 10:
        raise ValueError("Require 0 <= tight_obliquity <= max_obliquity <= 10 degrees")
    sg = _group(spacegroup, cell)
    get_operations(sg.xhm(), cell=cell)
    uc = gemmi.UnitCell(*cell.to_dict().values())
    rotations = {tuple(m.ravel()): m for m in map(reciprocal_operation, sg.operations())}
    if not anomalous:
        rotations.update({tuple((-m).ravel()): -m for m in list(rotations.values())})

    def key(m):
        return min(tuple((r @ m).ravel()) for r in rotations.values())

    def generated(tolerance):
        # all_ops retains representatives that Gemmi identifies only under
        # Friedel inversion. The data's anomalous setting governs deduplication.
        matrices = [reciprocal_operation(op) for op in
                    gemmi.find_twin_laws(uc, sg, max_obliq=tolerance, all_ops=True)]
        if anomalous and not sg.is_centrosymmetric():
            matrices += [-IDENTITY] + [-m for m in matrices]
        return [m for m in matrices if abs(round(np.linalg.det(m))) == 1]

    tight = {key(m) for m in generated(tight_obliquity)}
    candidates = {}
    identity_key = key(IDENTITY)
    metric = cell.reciprocal.T @ cell.reciprocal
    normalizer = np.linalg.norm(metric)
    declared = [_rational_detector_law(value) for value in extra_laws]
    for m in [*generated(max_obliquity), *(m for m, d in declared if d == 1)]:
        orbit = key(m)
        if orbit == identity_key:
            continue
        error = float(np.linalg.norm(m.T @ metric @ m - metric) / normalizer)
        # Prefer the supplied/conventional matrix with the simplest exact
        # action on indices, without substituting an approximate cell metric.
        exact = exact_metric(cell, m)
        order = (not exact, error, int(np.sum(np.abs(m))), tuple(m.ravel()))
        if orbit not in candidates or order < candidates[orbit][0]:
            candidates[orbit] = (order, dict(matrix=m.tolist(), metric_error=error, exact=exact,
                                              stable_at_tight_tolerance=orbit in tight,
                                              inversion=bool(np.array_equal(m, -IDENTITY)),
                                              route='intensity_mixture' if exact else 'detector_overlap'))
    result = [entry for _, entry in sorted(candidates.values(), key=lambda item: item[0])]
    sublattices = {}
    for numerator, denominator in declared:
        if denominator == 1:
            continue
        matrix = numerator / denominator
        residues = np.indices((denominator,)*3).reshape(3, -1).T
        coincident = np.all((residues @ numerator.T) % denominator == 0, axis=1)
        sublattices.setdefault((denominator, key(numerator)),
            dict(matrix=matrix.tolist(), numerator=numerator.tolist(), denominator=denominator,
                 sublattice_index=int(len(residues)//np.sum(coincident)), exact=False, inversion=False,
                 metric_error=float(np.linalg.norm(matrix.T @ metric @ matrix-metric)/normalizer),
                 route='detector_overlap', stable_at_tight_tolerance=None,
                 scope='Declared sublattice operator. Predicted detector overlaps require indexed domains.'))
    return [*result, *sublattices.values()]


def candidate_rotations(cell, groups, laws):
    """Rotations in the row convention of evidence_partition."""
    rotations = [reciprocal_operation(op).T for name in groups for op in get_operations(name, cell=cell)]
    rotations.extend(_matrix(law['matrix']).T for law in laws if law['exact'])
    return list({tuple(r.ravel()): r for r in rotations}.values())


def partition_intensities(reflections, rotations, *, links=(), seed=1729):
    try:
        partition = evidence_partition(reflections, rotations=rotations, links=links, seed=seed, close_rotations=True)
    except ValueError as exc:
        # Missing coverage is a scientific outcome. Malformed existing flags
        # still fail rather than silently changing the reserved observations.
        if not str(exc).startswith(('Three-way validation requires', 'Existing evidence flags leave too few')):
            raise
        return None, {'reason': str(exc)}
    return partition.pop("roles"), partition


def _absence_classes(group, hkl):
    ops = group.operations()
    absent = systematic_absences(group, hkl)
    centering = np.array(ops.cen_ops, dtype=int)
    centred = np.any((hkl @ centering.T) % gemmi.Op.DEN != 0, axis=1)
    return {'centering': centred, 'screw_or_glide': absent & ~centred}


def _covered_absences(group, reflections, expected):
    actual = _absence_classes(group, reflections.hkl)
    coverage = _absence_classes(group, expected) if expected is not None else {}
    z = reflections.intensity / reflections.sigma
    result = {}
    for name, mask in actual.items():
        measured = np.unique(reflections.hkl[mask], axis=0)
        denominator = None
        covered = None
        if name in coverage:
            predicted = {tuple(h) for h in expected[coverage[name]]}
            denominator = len(predicted)
            covered = sum(tuple(h) in predicted for h in measured)
        result[name] = dict(measurements=int(mask.sum()), unique_observed=len(measured),
                            expected_in_declared_coverage=denominator, observed_in_declared_coverage=covered,
                            positive_violations=int(np.sum(z[mask] > 3)),
                            negative_deviations=int(np.sum(z[mask] < -3)))
    return result


def _half_cell_contrasts(hkl, values):
    total = float(np.sum(values))
    rows = []
    for translation in np.array(list(np.ndindex(2, 2, 2))[1:]):
        odd = (hkl @ translation) % 2 == 1
        n_odd = int(np.count_nonzero(odd))
        n_even = len(odd)-n_odd
        supported = min(n_even, n_odd) >= 8 and total > 0.
        rows.append(dict(translation=(translation*.5).tolist(), even_orbits=n_even, odd_orbits=n_odd,
                         status='sampled' if supported else 'insufficient_parity_coverage',
                         even_mean=float(np.mean(values[~odd])) if n_even else None,
                         odd_mean=float(np.mean(values[odd])) if n_odd else None,
                         patterson_over_origin=float((total-2*np.sum(values[odd]))/total) if supported else None,
                         mask_contrast=(n_even-n_odd)/len(odd) if len(odd) else None))
    return rows


def _intensity_statistics(cell, group, r):
    """Descriptive Wilson moments and local differences among acentric pairs."""
    ops = group.operations()
    buckets = {}
    rotations = [reciprocal_operation(op) for op in ops.sym_ops]
    for h, y, sigma in zip(r.hkl, r.intensity, r.sigma):
        if not np.any(h) or ops.is_systematically_absent(h.tolist()):
            continue
        mates = [tuple(rotation @ h) for rotation in rotations]
        canonical = min(min(mate, tuple(-v for v in mate)) for mate in mates)
        buckets.setdefault(canonical, []).append((y, sigma))
    if len(buckets) < 8:
        return {'unique_allowed': len(buckets), 'status': 'insufficient_coverage'}
    h = np.array(sorted(buckets), dtype=np.int64)
    values = []
    for key in map(tuple, h):
        y, sigma = np.asarray(buckets[key]).T
        w = (sigma.min() / sigma)**2
        values.append(np.dot(y, w) / w.sum() / ops.epsilon_factor(list(key)))
    values = np.asarray(values)
    centric = np.array([ops.is_reflection_centric(row.tolist()) for row in h])
    reciprocal = h @ cell.reciprocal.T
    squared = np.einsum('ij,ij->i', reciprocal, reciprocal)
    # Nearby directions and resolution limit residual anisotropy. Pairing
    # within mod-2 classes avoids comparisons across common pseudo-centering.
    dominant_axis = np.argmax(np.abs(reciprocal), axis=1)
    sectors = (dominant_axis * 8 + (h % 2) @ np.array([4, 2, 1])).astype(int)
    normal = np.full(len(h), np.nan)
    local = []
    for shell in np.array_split(np.argsort(squared, kind='stable'), min(10, len(h)//8)):
        scale = float(np.mean(values[shell]))
        if scale <= 0:
            continue
        normal[shell] = values[shell] / scale
        for sector in np.unique(sectors[shell]):
            ids = shell[(sectors[shell] == sector) & ~centric[shell]]
            pair = ids[:len(ids)//2*2].reshape(-1, 2)
            a, b = values[pair].T
            valid = (a > 0) & (b > 0)
            local.extend((np.abs(a[valid]-b[valid])/(a[valid]+b[valid])).tolist())
    moments = {}
    for name, mask, ideal in (('centric', centric, .968), ('acentric', ~centric, .736)):
        statistic = np.abs(normal[mask & np.isfinite(normal)] - 1)
        moments[name] = {'n': len(statistic), 'mean_abs_e2_minus_1': float(statistic.mean()) if len(statistic) else None,
                         'ideal_untwinned': ideal}
    return {'unique_allowed': len(h), 'moments': moments, 'local_l_pairs': len(local),
            'mean_abs_l': float(np.mean(local)) if local else None,
            'mean_l_squared': float(np.mean(np.square(local))) if local else None,
            'half_cell_pseudotranslation': _half_cell_contrasts(h, values),
            'scope': 'Descriptive shell-normalized moments and local positive acentric pairs with matched parity. '
                     'Half-cell Patterson samples use signed epsilon-adjusted intensities and report parity support '
                     'and the sampling-mask contrast separately.'}


def law_pair_statistics(reflections, laws):
    """Matched, independent index-pair contrasts for each proposed operator."""
    observations = {}
    for h, y, s in zip(reflections.hkl, reflections.intensity, reflections.sigma):
        observations.setdefault(tuple(h), []).append((float(y), float(s)))
    means = {}
    for h, entries in observations.items():
        y, s = np.asarray(entries).T
        w = (s.min() / s)**2
        means[h] = (float(np.dot(w, y) / w.sum()), float(s.min() / np.sqrt(w.sum())))
    result = []
    hkl = np.array(list(means), dtype=np.int64)
    for law in laws:
        pairs = {}
        numerator = np.asarray(law.get('numerator', law['matrix']), dtype=np.int64)
        denominator = law.get('denominator', 1)
        transformed = hkl @ numerator.T
        integral = np.all(transformed % denominator == 0, axis=1)
        for h, mate_array, valid in zip(means, transformed // denominator, integral):
            if not valid:
                continue
            mate = tuple(mate_array)
            if h != mate and mate in means:
                first, second = sorted((h, mate))
                pairs[(first, second)] = (*means[first], *means[second])
        if pairs:
            a, sa, b, sb = np.asarray(list(pairs.values())).T
            valid_corr = len(a) > 2 and a.std() > 0 and b.std() > 0
            disagreement = float(np.mean((a-b)**2/(sa*sa+sb*sb)))
            corr = float(np.corrcoef(a, b)[0, 1]) if valid_corr else None
        else:
            disagreement = corr = None
        result.append({**law, 'integral_partner_indices': int(integral.sum()),
                       'matched_pairs': len(pairs), 'pair_correlation': corr,
                       'pair_mean_standardized_difference_squared': disagreement})
    return result


def symmetry_evidence(cell, reflections, spacegroup, *, candidates=(), coverage_hkl=None,
                      max_obliquity=3., tight_obliquity=.05, anomalous=False, extra_laws=()):
    """Use only the supplied work intensities. Coverage HKLs carry no intensities."""
    expected = None
    if coverage_hkl is not None:
        raw = np.asarray(coverage_hkl, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all() or not np.equal(raw, np.rint(raw)).all():
            raise ValueError("coverage_hkl must contain integral Miller indices")
        expected = np.unique(raw.astype(np.int64), axis=0)
    if reflections.free_flags is not None and np.any(reflections.free_flags):
        reflections = reflections.subset(~reflections.free_flags)
    names = list(dict.fromkeys(_group(name, cell).xhm() for name in [spacegroup, *candidates]))
    if len(names) > 32:
        raise ValueError("At most 32 space-group settings per common basis")
    ranking = rank_spacegroups(cell, reflections, names)
    laws = twin_laws(cell, spacegroup, max_obliquity=max_obliquity, tight_obliquity=tight_obliquity,
                    anomalous=anomalous, extra_laws=extra_laws)
    reference_group = _group(spacegroup, cell)
    base_absent = systematic_absences(reference_group, reflections.hkl)
    base_rotations = {tuple(reciprocal_operation(op).ravel()) for op in reference_group.operations()}
    z = reflections.intensity / reflections.sigma
    for row in ranking:
        group = _group(row['spacegroup'], cell)
        absent = systematic_absences(group, reflections.hkl)
        contrast = absent ^ base_absent
        row['classes'] = _covered_absences(group, reflections, expected)
        row['contrast_observations'] = int(contrast.sum())
        row['contrast_unique_hkl'] = len(np.unique(reflections.hkl[contrast], axis=0))
        row['candidate_only_positive_forbidden'] = int((absent & ~base_absent & (z > 3)).sum())
        rotations = {tuple(reciprocal_operation(op).ravel()) for op in group.operations()}
        expected_contrast = None if expected is None else int(np.count_nonzero(
            systematic_absences(group, expected) ^ systematic_absences(reference_group, expected)))
        row['expected_contrast_in_declared_coverage'] = expected_contrast
        row['coverage_status'] = ('reference' if row['spacegroup'] == reference_group.xhm() else
                                  'observed_absence_contrast' if contrast.any() else
                                  'equivalences_or_model_required' if rotations != base_rotations else
                                  'no_observed_symmetry_contrast')
        filled = np.zeros(len(reflections), dtype=bool)
        for law in laws:
            if law['exact']:
                filled |= absent & ~systematic_absences(group, reflections.hkl @ np.asarray(law['matrix']).T)
        row['potentially_filled_forbidden'] = int(filled.sum())
        row['positive_violations_without_proposed_filling'] = int((absent & ~filled & (z > 3)).sum())
    return {'schema': 'ariadne-symmetry-evidence/1', 'reference_spacegroup': reference_group.xhm(),
            'reflections': len(reflections), 'signed_negative': int(np.sum(reflections.intensity < 0)),
            'friedel_policy': 'retained' if anomalous else 'assumed_equivalent_for_law_enumeration',
            'candidates': ranking, 'twin_laws': law_pair_statistics(reflections, laws),
            'intensity_statistics': _intensity_statistics(cell, reference_group, reflections),
            'status': 'needs_model_or_independent_evidence',
            'coverage_scope': 'declared_indices' if expected is not None else 'observed_indices_only'}


def expand_p1(model):
    atoms = []
    for atom in model.atoms:
        for frac, tensor in unique_sites(model, atom):
            site = copy.deepcopy(atom)
            site.label = f'{atom.element}{len(atoms)+1}'
            site.frac = frac
            if atom.u_aniso is not None:
                site.u_aniso = tensor
            atoms.append(site)
    return Structure(cell=model.cell, atoms=atoms, spacegroup='P 1', wavelength=model.wavelength)


def symmetry_models(model, work, names, **options):
    names = [name for name in names if _group(name, model.cell).xhm() != model_group(model)]
    return propose_spacegroups(expand_p1(model), work.hkl, names, **options) if names else []


def _intensity_columns(model, hkl, matrices, max_cache_mib, metrics):
    """Yield work F² arrays, factoring each unique HKL once per memory batch."""
    if not np.isfinite(max_cache_mib) or not 1 <= max_cache_mib <= 4096:
        raise ValueError("max_cache_mib must lie in 1..4096")
    # Account for the union, inverse indices and NumPy sorting workspace.
    maximum_indices = max(1, int(max_cache_mib * 1024**2) // 120)
    rows = len(hkl)
    if rows > maximum_indices:
        metrics.update(cache_strategy='streamed_columns')
        for matrix in matrices:
            output = np.empty(rows)
            for start in range(0, rows, maximum_indices):
                indices = hkl[start:start+maximum_indices] @ matrix.T
                output[start:start+len(indices)] = np.abs(structure_factors(model, indices))**2
                metrics['evaluated_unique_hkl'] += len(indices)
            yield output
        return
    per_batch = max(1, maximum_indices // rows)
    metrics.update(cache_strategy='deduplicated_union', matrices_per_batch=per_batch)
    for start in range(0, len(matrices), per_batch):
        batch = matrices[start:start+per_batch]
        transformed = np.concatenate([hkl @ matrix.T for matrix in batch])
        unique, inverse = np.unique(transformed, axis=0, return_inverse=True)
        intensity = np.abs(structure_factors(model, unique))**2
        metrics['evaluated_unique_hkl'] += len(unique)
        metrics['maximum_union_rows'] = max(metrics.get('maximum_union_rows', 0), len(unique))
        for indices in inverse.reshape(len(batch), rows):
            yield intensity[indices]


def _two_column_fit(primary, secondary, response, *, rcond=1e-7):
    """Fit a positive scale and bounded fraction using weighted two-column QR."""
    design = np.column_stack((primary, secondary))
    q, r = np.linalg.qr(design, mode='reduced')
    target = q.T @ response
    singular = np.linalg.svd(r, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * rcond))
    choices = [np.zeros(2)]
    for j, column in enumerate(r.T):
        coefficients = np.zeros(2)
        norm = column @ column
        if norm > 0:
            coefficients[j] = max(0., float(column @ target / norm))
        choices.append(coefficients)
    if rank == 2:
        interior = np.linalg.solve(r, target)
        if np.all(interior >= 0):
            choices.append(interior)
    best = min(choices, key=lambda x: np.linalg.norm(r @ x - target))
    scale = float(best.sum())
    estimate = float(best[1]/scale) if scale else None
    residual = design @ best - response
    result = dict(scale=scale, fraction=estimate if rank == 2 else None, rank=rank,
                  singular_values=singular.tolist(), chi_square=float(residual @ residual),
                  fraction_interval=[0., 1.], fraction_status='unidentifiable')
    if estimate is None or rank < 2:
        return result
    threshold = np.linalg.norm(r @ best - target)**2 + chi2.ppf(.95, 1)

    def profile(fraction):
        col = r @ np.array([1.-fraction, fraction])
        norm = col @ col
        scale = max(0., float(col @ target / norm)) if norm else 0.
        return float(np.linalg.norm(col*scale-target)**2 - threshold)

    lower = brentq(profile, 0., estimate) if estimate > 0 and profile(0.) > 0 else 0.
    upper = brentq(profile, estimate, 1.) if estimate < 1 and profile(1.) > 0 else 1.
    result.update(fraction_interval=[lower, upper],
                  fraction_status='interior' if lower > 0 and upper < 1 else 'interval_touches_boundary')
    return result


def twin_screen(model, reflections, laws, *, max_cache_mib=256, covariance=None):
    """Conditional Gaussian work score for the untwinned and exact two-domain models."""
    if reflections.free_flags is not None and np.any(reflections.free_flags):
        mask = ~reflections.free_flags
        reflections = reflections.subset(mask)
        if covariance is not None:
            covariance = np.asarray(covariance)[np.ix_(mask, mask)]
    if len(reflections) < 2:
        raise ValueError("Twin screening needs at least two work observations")
    if covariance is not None:
        from scipy.linalg import cholesky, solve_triangular

        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (len(reflections), len(reflections)) or not np.allclose(covariance, covariance.T):
            raise ValueError("Observation covariance must be a symmetric N by N matrix")
        lower = cholesky(covariance, lower=True)

        def whiten(value):
            return solve_triangular(lower, value, lower=True)
    else:
        def whiten(value):
            return value / reflections.sigma
    exact = []
    for law in laws:
        entry = law if isinstance(law, dict) else {'matrix': law}
        if entry.get('exact') is False:
            continue
        matrix = _matrix(entry['matrix'])
        if exact_metric(model.cell, matrix) and not np.array_equal(matrix, IDENTITY):
            exact.append((entry, matrix))
    metrics = {'requested_hkl': len(reflections)*(len(exact)+1), 'evaluated_unique_hkl': 0}
    y = whiten(reflections.intensity)
    predictions = _intensity_columns(model, reflections.hkl, [IDENTITY]+[m for _, m in exact],
                                     max_cache_mib, metrics)
    primary = whiten(next(predictions))
    denominator = primary @ primary
    scale = max(0., float(primary @ y / denominator)) if denominator else 0.
    residual = scale * primary - y
    rows = [dict(matrix=None, scale=scale, fraction=0., chi_square=float(residual @ residual),
                 fraction_status='single_domain', rank=int(denominator > 0))]
    for (law, _), secondary in zip(exact, predictions):
        fit = _two_column_fit(primary, whiten(secondary), y)
        rows.append({**law, **fit, 'interval_scope': 'fixed-model nominal 95% profile; boundary uncalibrated'})
    return {'schema': 'ariadne-twin-screen/1', 'rows': rows, 'work_observations': len(reflections),
            'cache': metrics, 'deferred_to_detector': len(laws)-len(exact),
            'score_scope': 'conditional on fixed coordinates and declared observation errors'}
