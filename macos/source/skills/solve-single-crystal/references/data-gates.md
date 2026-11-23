# Data gates and input triage

## Evidence levels

### Gate A — complete diffraction frames

Required for true reintegration. Expect continuous experimental frame sequences,
experiment geometry, orientation matrices, and scan metadata. Bruker calibration
files labelled `ACTIVE MASK`, `INITIAL BACKGROUND`, or `BACKGROUND` are not
diffraction frames. SAINT/APEX access may be required.

Possible work: indexing verification, integration-box/background changes,
overload recovery, multi-domain integration, resolution-cutoff selection, then
scaling and absorption correction.

### Gate B — integrated unmerged observations

Files such as Bruker `.raw` may retain per-observation indices, intensities,
sigmas, scan numbers, detector positions, and direction cosines. With a licensed
SADABS installation, rescaling or absorption-correction tests may be possible,
but missing diffraction pixels cannot be reintegrated.

Use `.abs`, `.eps`, rejected-reflection logs, saturated-reflection lists, and
P4P/experiment files to assess the previous correction.

### Gate C — corrected HKL

Sufficient for SHELXT/SHELXL solution and refinement. Not sufficient to change
integration masks, peak profiles, frame queues, overload reconstruction, or
detector-background treatment.

Preserve batch information and the original corrected HKL. Do not replace it with
a repeatedly edited viewer copy unless equivalence is demonstrated.

### Gate D — CIF/RES without matching intensities

Sufficient for syntax, symmetry, connectivity, occupancy, ADP, and geometry
diagnosis. Do not report new R factors or claim a new refinement without the
matching reflection data and any required mask contribution file.

## Minimum information to extract

- cell, wavelength/radiation, temperature, lattice, orientation/setting;
- measured/unique/observed counts, Rint/Rsigma, completeness by resolution;
- theta or sin(theta)/lambda limit and overload handling;
- correction program/version, Tmin/Tmax, decay and scan-wise statistics;
- plausible elements and oxidation states;
- uploaded ligand MOL/SDF/MOL2/SMILES or otherwise unambiguous connectivity;
- known solvents and counterions;
- crystal colour, habit, and dimensions when actually recorded;
- synthesis conditions relevant to composition and charge.

Never infer crystal dimensions, colour, counterions, or solvent identity solely
from checkCIF expectations.

If a ligand file is required but missing, request it before assigning ambiguous
light-atom density or generating geometry restraints. Preserve the uploaded file
unchanged and place all normalised or optimised derivatives under a separate
working filename.

## Program and installation policy

- Search existing PATH and crystallography program folders first.
- Install or link SHELX only from the official distribution and respect licences.
- Treat Bruker APEX/SAINT/SADABS/XPREP as proprietary/licensed software.
- If XPREP is unavailable, use systematic absences, SHELXT space-group trials,
  and PLATON/ADDSYM as the command-line route.
- State clearly when a workflow begins at corrected HKL rather than raw frames.
