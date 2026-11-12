# Ariadne pro Windows

Ariadne pro 0.8.0 的 Windows x64 平行版本。独立构建标识 `win-x86_64-localopt1`，自带原生 C++ 数值内核、Windows 启动器与可选的 DIALS 3.30.1 插件层。

本仓库发布的是 Windows 计算版本及其验证记录，源码基于 0.8.0，并包含本机完成的局部性能优化。它与 macOS 版本、Ariadne 0.5.0 相互独立。

## 状态

- 核心回归 114 项通过；Python 参考后端复测 13 项通过；无失败、错误或跳过。
- 原生 C++ 扩展在 Windows 上编译并启用：单内部线程、释放 GIL、严格浮点、无 OpenMP。
- 公开 FeN4 实测强度重精修：R1(gt)=0.040316，wR2=0.081480，GoF=1.183。
- P1 无噪声合成解相位：4 个候选全部收敛。
- DIALS 3.30.1 运行时导入、Ariadne 选点插件注册与真实 `flex` 阈值调用均通过。

Windows 内部后端实测（i7-12700H，单数值线程，5 轮中位数）：

| 任务 | Python | C++ | 加速 |
|---|---:|---:|---:|
| 3 原子／3 畴／7280 反射 | 27.214 ms | 4.971 ms | 5.47× |
| 32 原子／4096 反射 | 152.620 ms | 27.859 ms | 5.48× |
| 80 原子／8192 反射 | 886.431 ms | 139.370 ms | 6.36× |
| 完整局部精修（18 原子／70 参数） | 342.419 ms | 175.128 ms | 1.96× |

详细条件、误差与边界见 [Ariadne pro Windows 能力与性能报告.md](Ariadne%20pro%20Windows%20能力与性能报告.md)。

## 安装

Windows 10/11 x64，无需管理员权限。请解压到短的、可写的独立路径（例如 `C:\AriadneProWin`），不要覆盖其他产品目录。

核心计算版：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install_windows.ps1
```

安装器会下载官方 NuGet CPython 3.12.10，按 `requirements-win-tested.txt` 安装到独立 `overlay`，并使用仓库中已经测试的 `_intensity_cpp.cp312-win_amd64.pyd`。可选 `-BuildKernel` 会用独立 Zig 工具链重新编译，要求先移走已有内核以避免覆盖。

可选 DIALS 运行时：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install_dials_windows.ps1
```

使用 micromamba 2.5.0 平面缓存，默认缓存目录 `C:\apw25`；无写权限时可在本次进程中设置 `ARIADNE_DIALS_CACHE` 指向另一个短的可写目录。DIALS 安装到独立 `dials-env`，Ariadne 插件只写入 `dials-overlay`，原始 DIALS 包保持不变。精确实测环境见 `dials-conda-explicit.txt`。

## 使用

```powershell
.\ariadne-pro-win.cmd capabilities
.\ariadne-pro-win.cmd refine "model.cif" --hkl "reflections.hkl" --output "result-new"
.\ariadne-pro-win.cmd python "my_calculation.py"
```

`refine` 输出目录必须尚不存在；支持 `--options`、`--constraints`、`--restraints` 三个 JSON 参数。结果写入 `result.json` 与 `refined.cif`。

DIALS 使用它自己的解释器：

```powershell
.\ariadne-pro-dials.cmd find_spots --help
```

选择 Ariadne 选点算法时须提供校准增益，例如 `spotfinder.threshold.algorithm=ariadne spotfinder.threshold.ariadne.gain_adu_per_photon=1`，其中数值应替换为探测器实际校准值。

## 目录

| 路径 | 内容 |
|---|---|
| `source/` | 0.8.0 源码与测试，含 Windows 预编译扩展 |
| `tools/` | CLI、DIALS 入口、构建脚本、安装辅助与验收驱动 |
| `validation/` | 部署记录、计时数据、JUnit XML、DIALS 验收结果 |
| `install_*.ps1` | 核心与 DIALS 安装脚本 |
| `ariadne-pro-*.cmd` | 用户命令入口 |

## 支持边界

Windows 原生支持数值计算入口。父层 MCP 与 POSIX 常驻 agent 依赖 `fcntl`，实测在 Windows 不可用，应在 WSL／Linux／macOS 运行；`--help` 能显示不代表执行受支持。

DIALS 插件导入、注册与阈值调用已验证；本仓库没有真实原始探测器图像，因此从图像到最终结构的 indexing、integration、scaling 端到端流程未验证。FeN4 是已知结构对公开合并实测强度的重精修，P1 是单个无噪声合成案例，均不能推算未知结构或实际样品的总体成功率。

## 授权

本仓库为专有软件，采用商业许可，不是开源项目。

- 许可全文见 [LICENSE](LICENSE)，条款说明见 [LICENSING.md](LICENSING.md)，第三方声明见 [NOTICE.md](NOTICE.md)。
- 商业使用需要与版权持有人签订书面许可。潜在被许可方可在 30 天内进行非生产用途评估，不得再分发或商用。
- 2026-10-29（commit 407d193）及更早发布的版本依 MIT 许可授权，该授权不可撤销，全文见 [LICENSE-MIT](LICENSE-MIT)。
- DIALS 派生文件（`source/patches/` 与 `source/dials-*`）保留 BSD-3-Clause 许可，商业许可不替代上游条件，分发时须保留其声明。
- 未授予商标或品牌使用权。

商业许可咨询请通过 GitHub 账号 [@owenxyzhang-seseTJ](https://github.com/owenxyzhang-seseTJ) 联系版权持有人。
