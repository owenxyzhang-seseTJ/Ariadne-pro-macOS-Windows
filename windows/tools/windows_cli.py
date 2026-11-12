"""Ariadne pro Windows 的计算入口，直接调用原生数值 API。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "python":
        return subprocess.call([sys.executable, *sys.argv[2:]])
    parser = argparse.ArgumentParser(prog="ariadne-pro-win", description="Ariadne pro Windows 0.8.0 原生计算版")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("capabilities", help="显示数值内核与 Windows 支持边界")
    refine_parser = commands.add_parser("refine", help="从结构和反射文件执行原生 F² 精修")
    refine_parser.add_argument("structure", type=Path)
    refine_parser.add_argument("--hkl", type=Path, required=True)
    refine_parser.add_argument("--output", type=Path, required=True)
    refine_parser.add_argument("--options", type=Path)
    refine_parser.add_argument("--restraints", type=Path)
    refine_parser.add_argument("--constraints", type=Path)
    args = parser.parse_args()
    if args.command in (None, "capabilities"):
        from ariadne_xtal.capabilities import capabilities
        from ariadne_xtal.kernels import kernel_status
        result = capabilities()
        result["product"] = "Ariadne pro Windows"
        result["variant"] = "win-x86_64-localopt1"
        result["numeric_kernel"] = kernel_status()
        result["native"]["agent"] = "本 Windows 入口直接提供数值能力；POSIX 常驻调度与父层 MCP 尚未移植。"
        result["windows_support"] = {"native_numerical_core": True,
            "persistent_agent": False, "parent_mcp_server": False,
            "reason": "上层锁与任务进程管理仍依赖 POSIX；本入口提供原生 Windows 计算功能。"}
    else:
        if args.output.exists():
            raise FileExistsError(f"输出目录已存在，停止以保护已有结果：{args.output}")
        from ariadne_xtal.io import read_structure, read_reflections, write_structure
        from ariadne_xtal.refinement import RefinementOptions, refine
        def read_json(path, fallback):
            return json.loads(path.read_text(encoding="utf-8-sig")) if path else fallback
        fitted = refine(read_structure(args.structure), read_reflections(args.hkl),
                        RefinementOptions(**read_json(args.options, {})),
                        restraints=read_json(args.restraints, []), constraints=read_json(args.constraints, []))
        result = fitted.to_dict()
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        write_structure(fitted.structure, args.output / "refined.cif")
        print(json.dumps({"success": fitted.success, "output": str(args.output.resolve()), "metrics": result["metrics"]}, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if fitted.success else 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"ok": False, "error": {"type": type(error).__name__, "message": str(error)}}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
