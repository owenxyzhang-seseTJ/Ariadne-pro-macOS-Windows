"""Execute the native crystallographic engine inside the persistent agent."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .backends import invoke
from .process import executable
from .util import artifact, artifact_path, atomic_json, file_hash, read_json


def run_native(ctx, args):
    request, inputs = args["request"], args.get("inputs", {})
    if not isinstance(request, dict) or not isinstance(inputs, dict):
        raise TypeError("Native request and input mapping must be objects")
    if request.get("action") == "legacy":
        raise ValueError("Use the separately registered explicit legacy backend")
    target = ctx.directory/"native"
    target.mkdir()
    for name, record in inputs.items():
        if not isinstance(name, str) or not name or Path(name).name != name or name in {".", "..", "request.json", "runs"}:
            raise ValueError("Native input names must be ordinary flat filenames")
        shutil.copyfile(artifact_path(ctx.root, record), target/name)
        if file_hash(target/name) != record["sha256"]:
            raise RuntimeError("Native input changed while making its isolated snapshot")
    atomic_json(target/"request.json", request)
    python = executable("scientific_python", {
        "scientific_python": ctx.executables.get("scientific_python", sys.executable)})
    try:
        invoke(ctx, [python, "-m", "ariadne_xtal", "run", str(target/"request.json"),
                     "--root", str(target)], label="native")
    except RuntimeError as error:
        # The native CLI emits its structured validation error on stdout.
        # Preserve that explanation instead of reducing it to an exit code.
        output = ctx.directory/"native.stdout.log"
        if output.is_file() and output.stat().st_size < 1_048_576:
            try:
                failed = read_json(output)
            except (ValueError, OSError):
                failed = None
            if isinstance(failed, dict) and failed.get("ok") is False:
                detail = failed.get("error", {}).get("message", "see native.stdout.log")
                raise RuntimeError(f"Native calculation failed: {detail}") from error
        raise
    response = read_json(ctx.directory/"native.stdout.log")
    if not response.get("ok"):
        raise RuntimeError("Native calculation failed; inspect native.stdout.log")
    if request.get("action") == "capabilities":
        result_file = target/"capabilities"/"result.json"
        atomic_json(result_file, response)
    else:
        result_file = (target/response["result"]).resolve()
    if not result_file.is_relative_to(target.resolve()):
        raise ValueError("Native result path leaves the isolated workspace")
    output_files = {p.relative_to(result_file.parent).as_posix(): artifact(ctx.root, p, "native_output")
                    for p in sorted(result_file.parent.rglob("*"))
                    if p.is_file() and not p.is_symlink()}
    return {"engine": "ariadne-native", "summary": response.get("summary"),
            "native_result": artifact(ctx.root, result_file, "native_result"),
            "outputs": list(output_files.values()), "output_files": output_files,
            "scientific_result": read_json(result_file)}
