# Ariadne pro Windows 平行版本与能力测试

日期：2026年11月09日。版本：0.8.0，独立构建标识 `win-x86_64-localopt1`。

## 结果

已通过 athena-wsl 调用真正的 Windows x64 Python，建立并验证独立的 Ariadne pro Windows 计算版本。原生 C++ 扩展已启用；核心回归 114 项通过，另有 13 项 Python 参考后端复测通过，均无失败、错误或跳过。两组测试合计 127 次执行，其中 13 次为重复后端验证。

该版本可执行结构因子与解析导数计算、结构精修、约束和孪晶数值计算、像素似然、P1 合成解相位，以及图形数值处理和离线导出。父层 MCP 与 POSIX 常驻 agent 尚未移植。DIALS 运行时、Ariadne 阈值插件及命令入口已核验；没有原始探测器图像，未执行图像到最终结构的完整流程。

## 环境与隔离

- 主机：athena-wsl 所在 Windows 机器，Windows 11 23H2（10.0.22631）。WSL 用于传输和启动原生 Windows 进程。
- CPU：Intel Core i7-12700H，14 核、20 逻辑处理器；内存约 32 GiB。
- 独立目录：`C:\Users\owenz\CrystallographyPrograms\ariadne-pro-win`。
- 核心解释器：NuGet CPython 3.12.10 x64；依赖位于 `overlay`，56 个依赖版本见 `requirements-win-tested.txt`。
- 数值库：NumPy 2.5.4、SciPy 1.18.1、Gemmi 0.7.5。
- 编译器：Zig 0.15.2 提供的 Clang 20.1.2；目标为 Windows x86_64；扩展大小 350720 bytes。
- C++ 构建保留严格浮点，单内部线程、释放 GIL、无 OpenMP、未启用 fast-math。
- DIALS 单独使用 conda-forge Python 3.12.16，DIALS 3.30.1、dxtbx 3.30.0、cctbx-base 2026.10。

Mac 版本与原有 Ariadne 0.5.0 未被覆盖。未升级原 Windows Anaconda；未修改全局 PATH、pip 配置或系统长路径策略。核心 Python 与 DIALS Python 分别保留自身运行目录与 DLL 搜索路径。

## Windows 内部后端性能

下表采用安装完成后的独立计时 `windows-timing.json`。所有数值线程限制为 1；每个模型预热后交替运行 Python/C++ 后端，5 轮，每轮 8 组不同参数；表中为每次计算的轮次中位数。精修为每后端 3 次、完整 `refine` 调用的中位数。

| 任务 | Python（ms） | C++（ms） | C++ 加速比 |
|---|---:|---:|---:|
| 3 原子／3 畴／7280 反射 | 27.214 | 4.971 | 5.47× |
| 32 原子／1 畴／4096 反射 | 152.620 | 27.859 | 5.48× |
| 80 原子／1 畴／8192 反射 | 886.431 | 139.370 | 6.36× |
| 完整局部精修：18 原子／1098 反射／70 参数 | 342.419 | 175.128 | 1.96× |

三组 Jacobian 最大 scaled 差异为 1.996e-14，低于 1e-10 的验收阈值。完整精修两后端均收敛，Jacobian 秩均为 70，均使用 7 次衍射计算；预测差异为 1.012e-13 sigma RMS，χ² 差约 7.958e-13。

这些数字比较同一 Windows 机器上同一 pro 版本的后端。Windows 源码已包含此前 Mac 上完成的局部优化，本次没有测量 Windows 修改前后对照。Mac 的 Apple M5 与本机 i7-12700H 不同，不能将跨机器耗时差异归因于操作系统。这里不提供与 Ariadne 0.5.0、SHELXT 或 SHELXL 的速度结论。

首次全面验收与安装曾并行，原始记录保留在 `windows-measurements.json`；性能结论使用随后单独生成的 `windows-timing.json`。未控制机器的电源策略、温度及用户其他程序；本结果代表该部署机器的实测，非硬件极限。Windows 进程 CPU 计时粒度较粗，表格使用 wall time。

## 内存

80 原子／8192 反射任务，每个后端由独立进程测量：Python 峰值 working set 为 107.32 MiB，C++ 为 99.48 MiB，包含解释器及加载库。返回 Jacobian 为 19.88 MiB。

底层使用 Windows `GetProcessMemoryInfo` 的 `PeakWorkingSetSize`。原始 JSON 的历史字段名为 `peak_rss_bytes`；报告按实际 Windows working-set 定义解释，不能直接与 Mac 的 RSS 值混作同一实验。

## 功能验证证据

| 项目 | 实际结果与范围 |
|---|---|
| 核心与图形检查 | 114 passed，47.920 s；覆盖结构因子、导数、精修、联合像素目标、降维与图形导出；49 条上游 scikit-image／NumPy 弃用警告 |
| Python 后端复测 | 13 passed，0.902 s；强制使用 Python 参考后端 |
| 公开 FeN4 重精修 | 71 个反射、10 参数，R1(gt)=0.04031574937，R1(all)=0.04126821954，wR2=0.08148047291，GoF=1.183324412 |
| P1 解相位 | 四碳无噪声合成模型、4540 反射；4 个候选均精修成功，坐标 RMS 为 3.61e-17～5.13e-16 Å；输入观测未改变 |
| 中文路径与换行 | 中文、空格路径中的 CIF／JSON 读写及 CRLF CIF 回读通过 |
| 实际用户入口 | `.cmd capabilities` 成功；`.cmd refine` 使用中文、空格路径完成精修，实际后端为 C++，R1(gt)=0.04031574937 |
| DIALS 运行时 | 原生导入 dials、dxtbx、cctbx 成功，精确版本由 conda-meta 记录核实 |
| DIALS 插件层 | 两个 indexing 模块来自独立 overlay；SpotFinderThreshold 注册与 PHIL scope 通过；`find_spots --help` 成功 |
| DIALS 阈值调用 | 512×512 合成图像；Poisson 选出 181 像素，Poisson-Gaussian 选出 176 像素；均与直接 API 逐像素一致、遵守 mask |

FeN4 数据来自 Bykov et al., Acta Cryst. E 74 (2018), 1392–1395，DOI：[10.1107/S2056989018012161](https://doi.org/10.1107/S2056989018012161)。该项是已知结构对公开合并实测强度的重精修。P1 测试仅为一个合成模型的恢复案例，不能推算未知结构或实际样品的总体成功率。

## 支持边界

父层 MCP 导入实际报 `ModuleNotFoundError: No module named 'fcntl'`。常驻 agent 执行锁实际报 `Use Linux, macOS or WSL for the agent execution lock`。这些功能应继续在 WSL／Linux／macOS 运行；帮助页能够显示不代表它们已支持 Windows。本 Windows 入口直接调用数值 API。

DIALS 插件导入、注册、阈值调用已验证；没有真实原始图像，因此 indexing、integration、scaling 到最终结构的科学端到端流程仍未验证。初期核心 Python 环境没有 DIALS 的失败记录保留在原始 JSON，后续独立 DIALS 安装结果见 `dials-validation.json`，两者对应不同解释器与时间。

## 交付与复现

- 远端已经安装，可直接使用 `ariadne-pro-win.cmd` 与 `ariadne-pro-dials.cmd`。
- `Ariadne-pro-0.8.0-win-x86_64-localopt1.zip` 包含优化后源码、已测试的 Windows `.pyd`、两个启动器、固定依赖、安装脚本、DIALS 三文件插件补丁及验收记录。
- 包内不附带大体积 Python、DIALS 或第三方依赖运行环境；在新机器运行安装脚本时需要网络。
- 核心安装各阶段与 DIALS 安装步骤已在目标机执行成功。整理后的分发安装器做了语法核查，未在另一台空白 Windows 机器上从零重装。
- DIALS 可选安装脚本固定 DIALS／dxtbx 版本，传递依赖由 channel 解析；精确实测环境见 `dials-conda-explicit.txt`，Windows 上可按该清单重建本次 DIALS 环境。
- 使用方法见 `README-Windows.zh-CN.md`。完整测量数据、JUnit XML 和部署记录随包附带。
