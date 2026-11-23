# Validation and delivery

## Final scientific checks

- rerun SHELXL from final INS/HKL and any required FAB with matching basenames;
- verify R1, wR2, GoF, data/restraints/parameters, max shift/esd, and residual
  density against CIF and LST;
- confirm no NPD and inspect all split warnings;
- verify formula, moiety formula, UNIT, Z, F000, density, charge, and SQUEEZE
  treatment;
- run PLATON/ADDSYM and checkCIF/FCF checks;
- for any twin model: law justified, BASF in (0,1) with meaningful esd,
  twin law and BASF recorded in the CIF, paired observed data available to reproduce the fit,
  and model-conditioned FCF labelled when generated (see twinning.md);
- confirm crystal colour, habit, and size are sourced, not guessed;
- preserve the original diffraction or integrated data unchanged.

## Alert explanation file

Create `checkcif/alert_explanation.md` containing:

1. formal model and final statistics;
2. number of Alert A and B items;
3. one section per retained Alert A/B;
4. raw/model evidence;
5. corrective action attempted or reason it was not used;
6. what new experiment would be needed;
7. a short English response suitable for checkCIF or an editor.

Distinguish a diagnostic test from the formal archived model.

## Package layout

```text
project_final_package_VERSION/
├── cif/final.cif
├── hkl/final.hkl
├── fcf/model_detwinned.fcf  # optional model-conditioned map coefficients
├── hkl/final.hkl5       # HKLF 5 file, when a component-based twin was refined
├── hkl/final.fab        # when SQUEEZE/ABIN is used
├── hkl/final.sqf        # when available
├── res/final.res
├── ins/final.ins
├── raw/original_work/
├── raw/ligand.mol
├── checkcif/final.chk
├── checkcif/final.ckf
├── checkcif/final.fcf
├── checkcif/final.lst
├── checkcif/platon_validation.log
├── checkcif/squeeze.log
├── checkcif/alert_explanation.md
├── README.md
├── refinement_notes.md
└── source_manifest.json
```

Use original filenames inside `raw/`. Use a consistent final basename for the
refinement set. Explain that INS, HKL, and FAB must be copied to one directory
with matching basenames for a rerun.

## Archive verification

1. record byte sizes, source roles, and reproducible file pairings;
2. verify the raw-file count against the source archive;
3. parse the final CIF and report space group, formula, colour, and habit;
4. rerun SHELXL from the packaged files;
5. create a new ZIP rather than overwriting an older package;
6. run `unzip -t`;
7. report ZIP size and source manifest.

Do not call the package submission-ready without stating every unresolved
scientific limitation.
