# Ariadne pro macOS / Windows

Ariadne pro 0.8.0 的 macOS 与 Windows 并行版本。两个平台共用同一份 0.8.0 源码（含本机完成的局部性能优化），各自提供原生 C++ 扩展、入口脚本、固定依赖和独立验证记录。

| 平台 | 构建标识 | 原生扩展 | 验证 |
|---|---|---|---|
| macOS arm64 | `macos-arm64-localopt1` | `_intensity_cpp.cpython-312-darwin.so`（aarch64，Clang 21） | 49 + 14 + 13 项通过 |
| Windows x64 | `win-x86_64-localopt1` | `_intensity_cpp.cp312-win_amd64.pyd`（x86_64，Clang 20.1.2） | 114 + 13 项通过 |

两版均可执行结构因子与解析导数计算、结构精修、约束与孪晶数值计算、像素似然、P1 解相位，以及图形数值处理与离线导出。两个平台都未使用真实原始探测器图像，图像到最终结构的端到端流程未验证。

## 快速开始

macOS（Apple Silicon，CPython 3.12，需已有 DIALS/cctbx 环境）：

```sh
cd macos
chmod +x install_macos.sh ariadne-pro-env
ARIADNE_PRO_PYTHON=/path/to/cctbx/bin/python3.12 ./install_macos.sh
./ariadne-pro-env -m ariadne_xtal capabilities
```

Windows 10/11 x64（无需管理员权限，安装器自行下载 Python 与依赖）：

```powershell
cd windows
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install_windows.ps1
.\ariadne-pro-win.cmd capabilities
```

Windows 可选 DIALS 运行时：`windows\install_dials_windows.ps1`，详见 [windows 报告](windows/Ariadne%20pro%20Windows%20能力与性能报告.md)。

各平台细节见 [macOS 说明](macos/README-macOS.zh-CN.md) 与 Windows 报告。

## 验证摘要

- macOS：数值回归 49 项、联合像素目标 14 项、Python 参考后端 13 项，共 76 次执行，无失败、错误或跳过；其中 13 次为重复后端验证。
- Windows：核心回归 114 项、Python 参考后端 13 项，无失败、错误或跳过。
- 公开 FeN4 实测强度重精修（Windows）：R1(gt)=0.040316，wR2=0.081480，GoF=1.183。
- P1 无噪声合成解相位：4 个候选全部收敛（Windows 实测；macOS 为同一算法后端）。
- DIALS 3.30.1 插件：Windows 上完成运行时导入、入口注册与真实 `flex` 阈值调用核验；macOS 提供对应补丁文件。

性能结论均为同机对照。macOS 报告比较优化前后（C++ 结构因子计算下降 12.6%～18.6%），Windows 报告比较同版本 Python 与 C++ 后端（C++ 快 5.5～6.4 倍，完整精修 1.96 倍）。Apple M5 与 i7-12700H 是不同硬件，跨平台耗时不能归因于操作系统。

## 目录

| 路径 | 内容 |
|---|---|
| `macos/` | macOS 源码、arm64 扩展、入口脚本、优化与验证记录 |
| `windows/` | Windows 源码、x64 扩展、启动器、安装脚本、DIALS 插件层与验证记录 |
| `LICENSE`、`LICENSING.md`、`NOTICE.md` | 商业许可与第三方声明 |

## 支持边界

Windows 原生支持数值计算入口；父层 MCP 与 POSIX 常驻 agent 依赖 `fcntl`，实测在 Windows 不可用，应在 WSL／Linux／macOS 运行。macOS 为 POSIX 平台，不受该限制，但本次未单独重跑这两项的端到端流程。

两个平台都没有真实原始探测器图像，因此 indexing、integration、scaling 到最终结构的科学端到端流程未验证。FeN4 是已知结构对公开合并实测强度的重精修，P1 是单个无噪声合成案例，均不能推算未知结构或实际样品的总体成功率。

## 授权

本仓库为专有软件，采用商业许可，不是开源项目。

- 许可全文见 [LICENSE](LICENSE)，条款说明见 [LICENSING.md](LICENSING.md)，第三方声明见 [NOTICE.md](NOTICE.md)。
- 商业使用需要与版权持有人签订书面许可。潜在被许可方可在 30 天内进行非生产用途评估，不得再分发或商用。
- 2026-10-29（commit 407d193）及更早发布的版本依 MIT 许可授权，该授权不可撤销，全文见 [LICENSE-MIT](LICENSE-MIT)。
- DIALS 派生文件（`macos/source/patches/`、`windows/source/patches/` 与 `windows/tools/dials-pro-overlay.zip` 内的 DIALS 源文件）保留 BSD-3-Clause 许可，声明见各自的 `patches/DIALS-LICENSE`；商业许可不替代上游条件，分发时须保留该声明。
- 未授予商标或品牌使用权。

商业许可咨询请通过 GitHub 账号 [@owenxyzhang-seseTJ](https://github.com/owenxyzhang-seseTJ) 联系版权持有人。
