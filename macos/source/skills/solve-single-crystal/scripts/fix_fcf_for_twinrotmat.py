#!/usr/bin/env python3
"""Patch a SHELXL-2025 CIF-style FCF so PLATON TwinRotMat can process it.

Verified with SHELXL-2025/1 output and PLATON v230426:

- ``platon -T1 name.fcf < /dev/null`` runs the batch TwinRotMat analysis
  (Section 8 in the log), but aborts with ``Weight Parameter A Negative``
  when the FCF lacks ``_refine_ls_weighting_details`` (SHELXL-2025 does not
  write this item into the .fcf).
- The missing items are taken from the ACTA .cif (weighting details,
  wavelength) with the .ins WGHT card as a fallback.
- The patch is applied to a COPY (default ``<name>_tw.fcf``); the original
  FCF is never modified.

Usage:
    python3 fix_fcf_for_twinrotmat.py final.fcf final.cif [final.ins] [-o out.fcf]

    When no ACTA CIF exists, pass the INS as the second positional
    (``final.fcf final.ins``); it is recognised by its WGHT card.

Exit codes: 0 patched, 1 nothing to do (already complete), 2 usage/IO error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def extract_cif_items(cif_text: str, keys: tuple[str, ...]) -> dict[str, str]:
    """Extract single-line CIF items plus multi-line quoted values."""
    found: dict[str, str] = {}
    lines = cif_text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("#", "loop_", "data_")):
            continue
        for key in keys:
            if key in found:
                continue
            if not stripped.startswith(key):
                continue
            value = stripped[len(key):].strip()
            if not value and i + 1 < len(lines):
                # value on the following line (SHELXL ACTA layout)
                value = lines[i + 1].strip()
                i += 1
            if value.startswith("'"):
                # multi-line quoted string terminated by a lone quote
                buf = [value]
                if not value.endswith("'") or value.count("'") < 2:
                    for extra in lines[i + 1:]:
                        buf.append(extra.strip())
                        if extra.strip().endswith("'"):
                            break
                found[key] = "\n".join(buf)
            elif value.startswith(";"):
                buf = []
                for extra in lines[i + 1:]:
                    if extra.strip() == ";":
                        break
                    buf.append(extra)
                found[key] = ";\n" + "\n".join(buf) + "\n;"
            else:
                found[key] = value
    return found


def extract_wght_from_ins(ins_text: str) -> str | None:
    m = re.search(r"^WGHT\s+([0-9.+-]+)(?:\s+([0-9.+-]+))?", ins_text, re.M)
    if not m:
        return None
    a = float(m.group(1))
    # b is optional (SHELXL default 0.0)
    b = float(m.group(2)) if m.group(2) else 0.0
    scheme = f"'w=1/[\\s^2^(Fo^2^)+({a:.4f}P)^2^]"
    if b:
        scheme += f"{b:+.4f}P"
    scheme += " where P=(Fo^2^+2Fc^2^)/3'"
    return scheme


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fcf", type=Path)
    parser.add_argument("cif", type=Path, nargs="?", default=None)
    parser.add_argument("ins", type=Path, nargs="?", default=None)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument('--ready-ok', action='store_true', help='Return success for a complete input FCF')
    args = parser.parse_args()

    fcf = args.fcf.expanduser().resolve()
    if not fcf.is_file():
        print(f"error: FCF not found: {fcf}", file=sys.stderr)
        return 2

    text = fcf.read_text(errors="replace")
    needs: dict[str, str] = {}

    # 1. Weighting details (required by TwinRotMat).
    if "_refine_ls_weighting_details" not in text:
        wght = None
        cif = args.cif.expanduser().resolve() if args.cif else fcf.with_suffix(".cif")
        if cif.is_file():
            items = extract_cif_items(
                cif.read_text(errors="replace"),
                ("_refine_ls_weighting_details",),
            )
            wght = items.get("_refine_ls_weighting_details")
        if wght is None and args.ins:
            ins = args.ins.expanduser().resolve()
            if ins.is_file():
                wght = extract_wght_from_ins(ins.read_text(errors="replace"))
        if wght is None and args.cif:
            # The second positional is declared as the ACTA CIF, but when no
            # ACTA CIF exists the caller may pass the INS there instead;
            # recognise it by its WGHT card.
            second = args.cif.expanduser().resolve()
            if second.is_file():
                wght = extract_wght_from_ins(second.read_text(errors="replace"))
        if wght is None:
            print(
                "error: no _refine_ls_weighting_details in the ACTA CIF and "
                "no WGHT card in the INS; provide the weighting string manually",
                file=sys.stderr,
            )
            return 2
        needs["_refine_ls_weighting_details"] = wght

    # 2. Wavelength (SHELXL-2025 FCF omits it; TwinRotMat then reports
    #    LAMBDA 0. Only copied when actually recorded in the ACTA CIF —
    #    never assumed, since the analysis runs without it.)
    if "_diffrn_radiation_wavelength" not in text:
        cif = args.cif.expanduser().resolve() if args.cif else fcf.with_suffix(".cif")
        if cif.is_file():
            items = extract_cif_items(
                cif.read_text(errors="replace"),
                ("_diffrn_radiation_wavelength",),
            )
            wl = items.get("_diffrn_radiation_wavelength")
            if wl:
                needs["_diffrn_radiation_wavelength"] = wl

    if not needs:
        print(f"{fcf.name}: already complete (no patch required)")
        return 0 if args.ready_ok else 1

    out = args.output.expanduser().resolve() if args.output else fcf.with_name(
        fcf.stem + "_tw.fcf"
    )
    if out == fcf:
        print("error: output must not overwrite the original FCF", file=sys.stderr)
        return 2

    block = []
    for key, value in needs.items():
        if value.startswith(("'", ";")) or "\n" in value:
            block.append(f"{key}\n{value}")
        else:
            block.append(f"{key} {value}")

    # Insert before the first loop_ after _shelx_F_squared_multiplier, or at
    # the end of the header if that anchor is missing.
    anchor = "_shelx_F_squared_multiplier"
    if anchor in text:
        insertion = "\n\n" + "\n\n".join(block)
        idx = text.index(anchor)
        line_end = text.index("\n", idx)
        text = text[:line_end] + insertion + text[line_end:]
    else:
        m = re.search(r"^loop_\s*$", text, re.M)
        if m:
            text = text[:m.start()] + "\n".join(block) + "\n\n" + text[m.start():]
        else:
            text = text.rstrip() + "\n\n" + "\n".join(block) + "\n"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"patched: {out} ({', '.join(needs)})")
    print(f"next: platon -T1 {out.name} < /dev/null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
