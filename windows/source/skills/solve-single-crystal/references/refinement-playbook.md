# Refinement playbook

## Space-group decision

Test all extinction-compatible groups that are chemically plausible. Record
SHELXT CFOM and initial R values, but decide from the whole evidence set:

- systematic absences and lattice;
- centric intensity statistics;
- connectivity and special-position compatibility;
- metal coordination and ligand completeness;
- ADP behaviour and residual density;
- PLATON missed symmetry;
- Flack/Parsons/Hooft behaviour in noncentrosymmetric candidates;
- credible twin interpretation.

A precisely measured inversion fraction near one half in a noncentrosymmetric
model is compatible with two inversion domains. Compare a centrosymmetric
alternative using the measured intensities and model geometry.

## Model hierarchy

Build in this order:

1. metal and strongest donors;
2. rigid ligand core and carboxylates;
3. topology-confirmed flexible arms;
4. coordinated solvent;
5. discrete non-coordinating solvent;
6. solvent mask only as a last resort.

At each stage compare R factors, shift/esd, ADPs, residual density, bond precision,
and restraint residuals. A lower R value does not validate a chemically wrong
model.

## Disorder

- Use PART and linked/free occupancies only with density or symmetry evidence.
- Keep the occupancy sum chemically and crystallographically consistent.
- Prefer SAME/SADI/DFIX/DANG for poorly resolved geometry and SIMU/RIGU/ISOR for
  supported ADP regularisation.
- Use EADP/EXYZ only when shared positions or indistinguishable components justify
  them.
- Derive bond, angle, ring, and planarity targets from the verified chemical
  graph, a trusted reference structure, or an independently prepared ligand
  conformer. Do not infer aromaticity or planarity from peak arrangement alone.
- Document why each restraint is necessary and remove redundant restraints.

## Ligand topology and optional RDKit preparation

Treat the uploaded ligand file as chemical-topology evidence, not as measured
crystal coordinates.

1. Parse and sanitise the molecule.
2. Report atom count, bond count, formula, formal charge, stereochemistry,
   aromaticity, and disconnected fragments.
3. Confirm that the file matches the intended ligand/protonation state.
4. If coordinates are missing or unusable, generate a separate 3D conformer with
   ETKDG and optimise with MMFF when parameters exist, otherwise UFF.
5. Keep the original file untouched and archive both source and derived file.
6. Use derived distances only as starting restraint targets with uncertainties
   appropriate to diffraction quality.
7. Never transplant the optimised conformer into the final model without fitting
   it to and validating it against observed electron density.

Use `scripts/prepare_ligand.py` for this workflow. If RDKit is unavailable in the
default interpreter, locate a configured Python environment containing RDKit or
continue with topology-only inspection using available chemistry tools.

## Hydrogen atoms

- Use riding H atoms for well-defined organic sites.
- Use aromatic H treatment only on verified aromatic carbon.
- Use rotating methyl treatment on general positions when valid.
- Special-position or symmetry-disordered groups require a compatible H model;
  validate all generated multiplicities and occupancies.
- Do not invent heteroatom-H vectors for disordered species without density and
  hydrogen-bond evidence.
- Keep coordinate-model H treatment distinct from formula H accounting.

## Solvent and additional components

Assign any solvent, counterion, modulator, guest, or reagent-derived component
only when electron density, geometry, composition, and charge balance support it.
Use the user's known candidate list as hypotheses, not permission to place a
species without density.

Retain coordinated and charge-balancing species explicitly. Do not SQUEEZE atoms
required for coordination, connectivity, or electroneutrality.

## SQUEEZE

Use only after discrete pore-solvent models remain unstable or non-unique. Archive
the pre-mask model and mask log. Report:

- number and symmetry of voids;
- total void volume per cell;
- electrons removed per cell;
- plausible composition range based on known candidates and electron count;
- whether masked solvent is excluded from formula, formula weight, density, F000,
  and UNIT;
- the `.fab`/`.sqf` contribution needed for rerunning SHELXL.

## Twinning and outliers

Follow [twinning.md](twinning.md) for the full twin workflow: evidence-gate
limits, detection (SHELXT trials, ADDSYM, PLATON TwinRotMat `-T1` on the
patched FCF), classification, SHELXL TWIN/BASF refinement, HKLF 5 handling,
detwinning, and CIF reporting. In brief: test twin laws suggested by metric
symmetry, reciprocal-space evidence, and PLATON; refine BASF honestly; and
do not retain a twin model solely because BASF refines to a small value.

For each serious outlier:

1. inspect all unmerged observations and symmetry equivalents;
2. check overload, active-mask, frame-boundary, collision, and background flags;
3. compare scan/run consistency;
4. decide whether the problem is experimental or model-based;
5. use OMIT only with documented experimental evidence.

Low-angle outliers in porous materials can reflect diffuse solvent or an
imperfect mask; this alone does not justify omission.

## Alert policy

- Alert A: exhaust every scientifically defensible correction; otherwise stop and
  request the missing experimental fact or data.
- Alert B: fix model errors; explain irreducible resolution, disorder, or data
  limitations.
- Never lower the resolution cutoff, alter formula metadata, relabel a main
  residue as solvent, force U values, or omit reflections only to change alert
  severity.
- For every retained A/B item, quote the alert, give evidence, state the decision,
  and provide a concise English response suitable for a validation form.
