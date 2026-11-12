# Ariadne pro macOS / Windows

Ariadne pro 0.8.0 的 macOS 与 Windows 并行版本，共用同一份源码，各自提供原生 C++ 扩展、入口脚本、固定依赖和验证记录。

| 平台 | 构建标识 | 原生扩展 | 验证 |
|---|---|---|---|
| macOS arm64 | `macos-arm64-localopt1` | `_intensity_cpp.cpython-312-darwin.so`（aarch64，Clang 21） | 49 + 14 + 13 项通过 |
| Windows x64 | `win-x86_64-localopt1` | `_intensity_cpp.cp312-win_amd64.pyd`（x86_64，Clang 20.1.2） | 114 + 13 项通过 |

功能：结构因子与解析导数、结构精修、约束与孪晶、像素似然、P1 解相位、图形数值处理与离线导出。

## 安装

macOS（Apple Silicon，CPython 3.12，已有 DIALS/cctbx 环境）：

```sh
cd macos
chmod +x install_macos.sh ariadne-pro-env
ARIADNE_PRO_PYTHON=/path/to/cctbx/bin/python3.12 ./install_macos.sh
```

Windows（10/11 x64，无需管理员权限）：

```powershell
cd windows
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install_windows.ps1
```

Windows 可选 DIALS 运行时：

```powershell
cd windows
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install_dials_windows.ps1
```

## 使用

macOS：

```sh
./ariadne-pro-env -m ariadne_xtal capabilities
./ariadne-pro-env -m ariadne_xtal refine --help
./ariadne-pro-env my_calculation.py
```

Windows：

```powershell
.\ariadne-pro-win.cmd capabilities
.\ariadne-pro-win.cmd refine "model.cif" --hkl "reflections.hkl" --output "result-new"
.\ariadne-pro-win.cmd python "my_calculation.py"
.\ariadne-pro-dials.cmd find_spots --help
```

## 验证

| 项目 | 结果 |
|---|---|
| macOS 数值回归 | 49 passed |
| macOS 联合像素目标 | 14 passed |
| macOS Python 参考后端 | 13 passed |
| Windows 核心回归 | 114 passed |
| Windows Python 参考后端 | 13 passed |
| FeN4 重精修 | R1(gt)=0.040316，wR2=0.081480，GoF=1.183 |
| P1 合成解相位 | 4/4 候选收敛 |
| DIALS 3.30.1 | 入口注册与 `flex` 阈值调用通过（Windows） |

同机对照：macOS 优化后 C++ 结构因子计算下降 12.6%～18.6%，完整精修 1.20×；Windows C++ 后端比 Python 快 5.5～6.4 倍，完整精修 1.96×。

## 目录

| 路径 | 内容 |
|---|---|
| `macos/` | 源码、arm64 扩展、入口脚本、依赖清单、优化与验证记录 |
| `windows/` | 源码、x64 扩展、启动器、安装脚本、DIALS 插件层与验证记录 |
| `LICENSE`、`LICENSING.md`、`NOTICE.md` | 商业许可与第三方声明 |

## 授权

- 专有软件，商业许可，非开源。
- 许可全文 [LICENSE](LICENSE)；条款 [LICENSING.md](LICENSING.md)；第三方声明 [NOTICE.md](NOTICE.md)。
- 商业使用需书面许可；潜在被许可方可 30 天非生产评估，不得再分发或商用。
- 2026-10-29（commit 407d193）及更早版本为 MIT 许可，不可撤销，见 [LICENSE-MIT](LICENSE-MIT)。
- DIALS 派生文件为 BSD-3-Clause，见 `macos/source/patches/DIALS-LICENSE`、`windows/source/patches/DIALS-LICENSE`，分发须保留声明。
- 不授予商标或品牌使用权。
- 商业许可联系：[@owenxyzhang-seseTJ](https://github.com/owenxyzhang-seseTJ)。
