"""Package the self-contained LaTeX project and its open licensed assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

README = """# Ariadne 0.8.0 LaTeX 工程

本工程包含完整教材和独立的结构反馈报告，以及全部正文、矢量图、
预览图、字体子集及其许可。无需下载图像或字体即可编译。

## 编译

安装带有 XeLaTeX 的 TeX Live 或 MiKTeX，并确保包含 fontspec、TikZ、
amsmath、booktabs、tabularx、titlesec、listings、hyperref 和 Latin Modern。
在此目录运行：

```bash
python build_report.py
```

脚本分别运行三遍 XeLaTeX，生成两个同名 PDF，并检查缺字、交叉引用
和超宽文本。也可分别对两个主 .tex 文件运行 XeLaTeX 三遍。

## 重绘与修改

编译直接使用 figures/ 中现成的矢量 PDF。tools/ 中的重绘脚本还需要
完整源码包内的 Ariadne 数值模块和原始验证 JSON；单独的 LaTeX 工程
不重复包含这些科学计算依赖。请在完整源码包中运行重绘脚本。

字体子集覆盖当前正文。增加汉字后，可准备原始 Noto Serif/Sans SC
Regular/Bold 四份字体，并运行 tools/subset_report_fonts.py 的
--source-dir 参数重建子集。随附字体遵循 SIL OFL 1.1，许可和修改
说明见 fonts/。数学和拉丁补充字体由 TeX 安装提供。

MANIFEST.json 记录包内各文件的 SHA-256。教材和专题报告的 PDF
另行提供，也包含在完整 Ariadne 源码包中。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "docs/v08"
    destination = args.destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("Write the report archive outside docs/v08")
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or "__pycache__" in relative.parts:
            continue
        if relative.parts[0] in {"chapters", "figures", "fonts", "tools"}:
            if path.suffix not in {".tex", ".pdf", ".png", ".json", ".ttf", ".txt", ".py"}:
                continue
        elif relative.as_posix() not in {
            "Ariadne_0_8_0_Textbook.tex", "Ariadne_0_8_0_Loop.tex",
            "preamble.tex", "references.tex", "build_report.py",
        }:
            continue
        files[relative.as_posix()] = path.read_bytes()
    files["README.zh-CN.md"] = README.encode("utf-8")
    manifest = {"version": "0.8.0", "files": [
        {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in sorted(files.items())
    ]}
    files["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo("Ariadne_0_8_0_LaTeX/" + name, (2026, 11, 8, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Report archive CRC failure")
        for item in manifest["files"]:
            content = archive.read("Ariadne_0_8_0_LaTeX/" + item["path"])
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise RuntimeError("Report archive digest failure")
    print(json.dumps({"path": str(destination), "files": len(files),
                      "bytes": destination.stat().st_size,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
