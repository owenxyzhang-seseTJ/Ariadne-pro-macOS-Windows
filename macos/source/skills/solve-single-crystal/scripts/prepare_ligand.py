#!/usr/bin/env python3
"""Validate a ligand file and optionally create an RDKit-optimised reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_rdkit():
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    except ImportError:
        print(
            "error: RDKit is not available in this Python environment. "
            "Use a configured RDKit interpreter or omit --optimize.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    return Chem, rdBase, AllChem, Descriptors, rdMolDescriptors


def read_molecule(Chem, path: Path):
    suffix = path.suffix.lower()
    if suffix in {".mol", ".sdf"}:
        if suffix == ".sdf":
            supplier = Chem.SDMolSupplier(str(path), removeHs=False)
            return next((mol for mol in supplier if mol is not None), None)
        return Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    if suffix == ".mol2":
        return Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
    raise ValueError("supported ligand formats are .mol, .sdf, and .mol2")


def disconnected_fragments(Chem, mol) -> list[str]:
    return [
        Chem.MolToSmiles(fragment, isomericSmiles=True)
        for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--force-field", choices=("auto", "mmff", "uff"), default="auto")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        print("error: output must not overwrite the uploaded ligand", file=sys.stderr)
        return 2
    if not source.is_file():
        print(f"error: ligand not found: {source}", file=sys.stderr)
        return 2

    Chem, rdBase, AllChem, Descriptors, rdMolDescriptors = load_rdkit()
    try:
        mol = read_molecule(Chem, source)
    except Exception as exc:
        print(f"error: failed to parse ligand: {exc}", file=sys.stderr)
        return 2
    if mol is None:
        print("error: RDKit could not sanitise the ligand", file=sys.stderr)
        return 2

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    fragments = disconnected_fragments(Chem, mol)
    force_field = None
    optimisation_status = None

    if args.optimize:
        mol = Chem.AddHs(mol, addCoords=True)
        mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = args.seed
        params.useSmallRingTorsions = True
        embed_status = AllChem.EmbedMolecule(mol, params)
        if embed_status != 0:
            print("error: RDKit ETKDG embedding failed", file=sys.stderr)
            return 4

        has_mmff = AllChem.MMFFHasAllMoleculeParams(mol)
        requested = args.force_field
        if requested == "mmff" and not has_mmff:
            print("error: MMFF parameters are incomplete for this ligand", file=sys.stderr)
            return 4
        if requested == "uff" or (requested == "auto" and not has_mmff):
            force_field = "UFF"
            optimisation_status = AllChem.UFFOptimizeMolecule(
                mol, maxIters=args.max_iters
            )
        else:
            force_field = "MMFF94s"
            properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            optimisation_status = AllChem.MMFFOptimizeMolecule(
                mol,
                mmffVariant="MMFF94s",
                maxIters=args.max_iters,
            )
            if properties is None:
                print("error: failed to initialise MMFF94s", file=sys.stderr)
                return 4

    output.parent.mkdir(parents=True, exist_ok=True)
    Chem.MolToMolFile(mol, str(output), kekulize=False)

    report = {
        "source": str(source),
        "output": str(output),
        "rdkit_version": rdBase.rdkitVersion,
        "atoms": mol.GetNumAtoms(),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "bonds": mol.GetNumBonds(),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        "molecular_weight": round(Descriptors.MolWt(mol), 6),
        "isomeric_smiles": Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True),
        "fragment_count": len(fragments),
        "fragments": fragments,
        "optimised": args.optimize,
        "force_field": force_field,
        "optimisation_status": optimisation_status,
        "warning": (
            "Reference geometry only; do not replace experimental coordinates "
            "without fitting and validating against electron density."
        ),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        target = args.report.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
