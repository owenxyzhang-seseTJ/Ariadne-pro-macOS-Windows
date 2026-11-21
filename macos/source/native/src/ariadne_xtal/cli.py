"""Command-line entry points; machine output is strict JSON."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .agent import AgentWorkspace, json_safe, serve_jsonl
from .capabilities import capabilities


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ariadne-xtal", description="Auditable crystallographic native and explicit legacy tools")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities", help="Print implemented capabilities and explicit boundaries")
    counts = sub.add_parser("count-refine", help="Experimental refinement against calibrated Poisson pixel groups")
    counts.add_argument("structure", type=Path)
    counts.add_argument("groups", type=Path, help="JSON list of CountStructureGroup specifications")
    counts.add_argument("--options", type=Path, help="JSON file containing native refinement options")
    counts.add_argument("--profile-options", type=Path,
                        help="Opt-in JSON bounds for shared Gaussian peak shifts and widths")
    run = sub.add_parser("run", help="Run one structured JSON job in a workspace")
    run.add_argument("request", help="JSON request file; '-' reads stdin")
    run.add_argument("--root", default=".")
    run.add_argument("--shelxt", help="Explicit executable path to enable official SHELXT")
    run.add_argument("--shelxl", help="Explicit executable path to enable official SHELXL")
    serve = sub.add_parser("serve", help="One JSON request/response per stdin/stdout line")
    serve.add_argument("--root", default=".")
    serve.add_argument("--shelxt")
    serve.add_argument("--shelxl")
    for name in ("inspect", "refine", "maps", "absolute"):
        cmd = sub.add_parser(name)
        cmd.add_argument("structure", help="Workspace-relative JSON/CIF/INS/RES structural model")
        cmd.add_argument("--hkl", required=name != "inspect", help="HKLF4 or reflection JSON")
        cmd.add_argument("--root", default=".")
        cmd.add_argument("--options", help="JSON file containing native options")
        if name == "refine":
            cmd.add_argument("--restraints", help="JSON list of native restraints")
            cmd.add_argument("--constraints", help="JSON list of native exact constraints")
            cmd.add_argument("--no-graphics", action="store_true")
        if name == "maps":
            cmd.add_argument("--scale", type=float, default=1.)
    args = parser.parse_args(argv)
    try:
        if args.command == "capabilities":
            result = capabilities()
        elif args.command == "count-refine":
            from .count_refinement import refine_count_observations
            from .io import read_structure

            groups = json.loads(args.groups.read_text(encoding="utf-8"))
            options = json.loads(args.options.read_text(encoding="utf-8")) if args.options else None
            profiles = json.loads(args.profile_options.read_text(encoding="utf-8")) if args.profile_options else None
            if args.profile_options and not isinstance(profiles, dict):
                raise ValueError("profile-options must be a JSON object")
            result = refine_count_observations(read_structure(args.structure), groups, options,
                                               profile_refinement=profiles)
        else:
            legacy = {name: getattr(args, name) for name in ("shelxt", "shelxl") if getattr(args, name, None)}
            workspace = AgentWorkspace(args.root, legacy)
            if args.command == "serve":
                serve_jsonl(workspace)
                return 0
            if args.command == "run":
                source = sys.stdin.read() if args.request == "-" else Path(args.request).read_text(encoding="utf-8")
                request = json.loads(source)
            else:
                request = {"action": args.command, "structure": args.structure}
                if args.hkl:
                    request["hkl"] = args.hkl
                if args.options:
                    request["options"] = json.loads(Path(args.options).read_text(encoding="utf-8"))
                if args.command == "refine":
                    request["graphics"] = not args.no_graphics
                    for key in ("restraints", "constraints"):
                        if getattr(args, key):
                            request[key] = json.loads(Path(getattr(args, key)).read_text(encoding="utf-8"))
                if args.command == "maps":
                    request["scale"] = args.scale
            result = workspace.dispatch(request)
        print(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if result.get("ok") is False or (args.command == "count-refine" and not result["success"]) else 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": {"type": type(error).__name__, "message": str(error)}}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
