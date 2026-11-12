# Ariadne pro macOS

macOS arm64 平行版本，构建标识 `macos-arm64-localopt1`。源码为 0.8.0，并包含本机完成的局部性能优化；原生 C++ 扩展已随 `source/` 提供。

## 要求

- Apple Silicon（arm64）macOS。
- CPython 3.12（预编译扩展为 `cpython-312-darwin`，不适用于其他 Python 版本或 Intel 架构）。
- 一个可用的 DIALS 3.30.1 + dxtbx 3.30.0 + cctbx 环境。实测使用 cctbx-base 2026.9、NumPy 2.5.4、SciPy 1.18.1、Gemmi 0.7.5。
- 纯 Python 附加依赖见 `requirements-macos-tested.txt`（mcp 1.30.0、plotly 7.1.0、pytest 9.2.0 等）。

## 安装

`install_macos.sh` 只向本目录的 `overlay` 安装纯 Python 依赖，不改动已有 conda 环境。

```sh
chmod +x install_macos.sh ariadne-pro-env
ARIADNE_PRO_PYTHON=/path/to/cctbx/bin/python3.12 ./install_macos.sh
```

脚本会检查 Python 版本、架构和 cctbx 可用性，随后核验扩展：`compiled_available=true`、`resolved=cpp`。

## 使用

```sh
./ariadne-pro-env -m ariadne_xtal capabilities
./ariadne-pro-env -m ariadne_xtal refine --help
./ariadne-pro-env my_calculation.py
```

默认使用 `python3.12`；用 `ARIADNE_PRO_PYTHON` 指向带 cctbx 的解释器。macOS 属于 POSIX 平台，常驻 agent 与 MCP 的代码路径不受 Windows 的 `fcntl` 限制；本次未单独重跑其端到端流程。

DIALS 插件使用 `source/patches/dials-3.30.1-ariadne.patch` 应用到同版本 DIALS 源码；本机实际部署使用独立 patched DIALS 源树，未修改原始安装包。

## 验证

| 项目 | 结果 |
|---|---|
| 数值回归 | 49 passed，2.212 s |
| 联合像素目标 | 14 passed，1.113 s |
| Python 参考后端 | 13 passed，0.392 s |
| 合计 | 76 次执行，0 失败、0 错误、0 跳过；其中 13 次为重复后端验证 |

后端同机对照（Apple M5，单数值线程，交错运行 6 轮中位数）：

| 任务 | 后端 | 优化前（ms） | 优化后（ms） | 降幅 |
|---|---|---:|---:|---:|
| 3 原子／3 畴／7280 反射 | Python | 15.623 | 11.991 | 23.2% |
| 3 原子／3 畴／7280 反射 | C++ | 2.886 | 2.521 | 12.6% |
| 32 原子／4096 反射 | Python | 31.516 | 26.649 | 15.4% |
| 32 原子／4096 反射 | C++ | 5.496 | 4.475 | 18.6% |
| 80 原子／8192 反射 | Python | 145.641 | 134.561 | 7.6% |
| 80 原子／8192 反射 | C++ | 22.352 | 19.365 | 13.4% |
| 完整局部精修（18 原子／70 参数） | Python | 65.067 | 52.976 | 18.6% |
| 完整局部精修（18 原子／70 参数） | C++ | 36.650 | 30.654 | 16.4% |

这是同一版本优化前后的对照，不是与 Ariadne 0.5.0 的版本比较。完整方法、逐元素一致性和边界见 `optimization/Ariadne pro 本机优化报告.md` 与 `optimization/paired-comparison.json`。

## 边界

- 未使用真实原始探测器图像，图像到最终结构的端到端流程未验证；`algorithm_validation` 为 `not_run`。
- 预编译扩展只覆盖 CPython 3.12 arm64；其他组合需要自行用本机编译器重建。
- 本目录的安装脚本经过语法与逻辑检查，并在实测机上以等价步骤部署；未在一台全新的 Mac 上从零执行。

## 目录

| 路径 | 内容 |
|---|---|
| `source/` | 0.8.0 源码、测试与 arm64 扩展 |
| `optimization/` | 优化报告、补丁、前后对照数据、JUnit XML |
| `validation/` | 部署记录 |
| `install_macos.sh` | 纯 Python 依赖安装与核验 |
| `ariadne-pro-env` | 入口脚本 |
