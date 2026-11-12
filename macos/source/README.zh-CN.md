# Ariadne 0.8.0

Ariadne 将单晶衍射观测、原生相位求解、结构精修和可恢复的科学 agent 接在一起。本交付是**完整的 0.8.0 本地源码版本**，包含保留的 DIALS 扩展、两个 Python 项目、测试、示例，以及教材式 LaTeX/PDF 技术报告。版本号说明本包的内容，不表示 GitHub 上已经发布同名标签。

[English](README.md) · [完整使用指南](docs/v08/Ariadne_0_8_0_Guide.zh-CN.md) · [原生引擎](native/README.md) · [许可说明](LICENSING.md)

## 先看这些文件

- [完整技术报告 PDF](docs/v08/Ariadne_0_8_0_Textbook.pdf)及 [LaTeX 主文件](docs/v08/Ariadne_0_8_0_Textbook.tex)：从波的叠加讲到仪器几何、像素似然、缩放、相位、精修与 agent。
- [结构反馈与原生智能体闭环专题 LaTeX](docs/v08/Ariadne_0_8_0_Loop.tex)：单独解释 loop 的数学接口和独立验证。
- [0.8.0 操作指南](docs/v08/Ariadne_0_8_0_Guide.zh-CN.md)：安装、输入约定、示例、调用接口和复现。
- [示例来源说明](examples/v08/PROVENANCE.json)：区分合成观测与实测 FeN₄ 重精修。

## 这次改动的核心

**强反射可以提供几何锚点，弱反射和孪晶观测继续参与测量模型。** 根据探测器噪声、预期计数与轮廓可分辨性选择处理路径，避免用测得的 `I/σ` 截断来决定保留哪些观测。背景扣除后出现负强度，是高斯测量模型允许的结果。

固定方差、固定轮廓的高斯像素经过背景消元后，保存为 `||R I - z||² + residual_constant`。它保留全部可测的强度组合和零空间。结构精修直接预测各分量，并通过这个因子与观测比较；完全重叠时，不必先伪造一张逐分量去孪晶强度表。真正的光子计数使用非负信号—背景联合泊松似然及条件剖面区间。

原生相位模块组合三相位关系、Patterson 起点、电荷翻转与原子支撑。新增 `hybrid` 路径用带符号强度似然控制倒易空间的模更新，保留原来的 `charge_flip` 和 `atomic_projection`。原生精修使用解析衍射导数、物理 ADP 参数化和分块计算。

`autosolve` 完成候选求解、精修与选择；`solve-observations` 从隔离参考畴观测初始化，再对全部工作组联合精修。持久化 agent 通过 `native.run` 调用相同内核，保存输入快照、依赖和恢复记录。work、selection、audit 分工明确，结构预测始终作为解释，不覆盖实验观测。

相互独立且依赖就绪的 DAG 动作可以在隔离进程中并发。会话与节点分别声明资源预算，统一控制 worker 准入和配合设置的 BLAS/OMP 线程数；内存字段是声明预算，不是操作系统 RSS 硬上限。默认保持顺序执行，`scheduler` 与 `resources` 的完整配置见使用指南。

这些是已经写入本包的原生能力。**0.8.0 尚不与 SHELXT/SHELXL 的全部功能等价**：完整指令语言、无限制的化学组成推断、以及泊松像素—结构—几何—吸收整体优化器不在当前原生范围内。保留用户显式启用的 SHELX 接口，包内不含官方二进制程序。

## 安装完整源码

在解压后的仓库根目录执行。组合安装使用 Python 3.12 或更高版本。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './python[dev]' -e './native[dev]'
ariadne-agent capabilities
ariadne-xtal capabilities
```

`python/` 提供观测似然与持久化 agent；`native/` 提供独立结构求解与精修。像素积分及联合工作流需要两者同时安装。请保留完整源码目录，单独一个 wheel 不包含全部 DIALS 补丁、科学脚本和 expert pack。持久化 agent 使用 POSIX 进程及文件锁，支持 Linux、macOS 和 WSL。

原生项目附带可选 C++17 强度/解析导数内核。`ARIADNE_BUILD_CPP=auto|0|1` 控制可选、禁用或必须编译；`ARIADNE_NUMERIC_BACKEND=auto|python|cpp` 控制运行时选择。用 `python -m ariadne_xtal.kernels` 查询实际后端。C++ 核单线程并释放 GIL，Python 参考实现继续保留；编译器要求和完整命令见使用指南。

## 运行现成示例

以下命令均从仓库根目录运行；原生请求内部的文件路径相对于 `--root`。

```bash
# 光子计数：信号、背景与条件区间。
ariadne-xtal run examples/v08/integrate_counts.json --root examples/v08

# 合成反射盲解：已知整个晶胞组成，不提供真值坐标。
ariadne-xtal run examples/v08/autosolve.json --root examples/v08

# 隔离参考畴初始化，再对含重叠的观测组共同精修。
ariadne-xtal run examples/v08/solve_observations.json --root examples/v08

# 公开实测 FeN4：从存档结构出发重精修。
ariadne-xtal run examples/v08/refine_fen4.json --root examples/v08

# 持久化工作流：会话目录必须是新目录。
ariadne-agent init examples/v08/persistent_native.json runs/fen4-session
ariadne-agent run runs/fen4-session
ariadne-agent status runs/fen4-session
```

每次原生计算在其工作区下新建 `runs/<action>_.../`，保存结果和来源。`ok: true` 说明动作执行成功，还应查看数值收敛、参数秩、残差和保留集表现。相位求解要求给出**整个晶胞的元素列表**，不能把不对称单元的组成直接作为该列表。

## 数据还原与 DIALS

保留的反射预测、仪器几何、三维体素轮廓、有符号 WLS、Poisson–Gaussian、复合观测缩放及协方差导出，位于 `skills/ariadne-reduction/scripts/` 和 `patches/`。DIALS/dxtbx/cctbx 的匹配编译环境仍需在外部安装。

`sh scripts/install_local.sh` 为 **DIALS 3.30.1** 创建 Python 源码覆盖层；两个可选参数依次是基础 conda 环境和新的覆盖层目录。脚本不运行科学验证，也不自动安装 `native/`，后者需用选定科学环境的 Python 另行安装。单独安装 pip cctbx 不能代替匹配的 DIALS 环境。旧配置、外部程序路径与仪器工作流说明保留在 [原 agent 指南](AGENT_GUIDE.zh-CN.md)和 [R2 指南](docs/R2_README.zh-CN.md)。

本版演示不是 Bruker/Rigaku 原始帧的完整实测验证。复杂孪晶需要由实际指标化和几何给出分量 HKL、晶畴及校正系数；agent 不从一张普通强度表凭空补齐这些信息。

## 验证与复现

```bash
python -m pytest tests tests_rebuild -q
ARIADNE_MCP_STDIO_TEST=1 python -m pytest native/tests -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/v08_reduction.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/validation/run_v08_refinement.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/validation/run_v08_joint.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/validation/run_v08_phasing_autonomy.py
python scripts/verify_release.py
```

当前数值证据包括 [像素似然](benchmarks/v08_reduction_results.json)、[解析精修](native/validation/v08_refinement_measurements.json)、[像素—结构联合目标](native/validation/v08_joint_measurements.json)、[合成盲解与完整观测闭环](native/validation/v08_phasing/result.json)、[实测 FeN₄ 重精修](native/validation/v08_real_fen4/result.json)。测试 XML 和 CLI 示例记录位于 [`docs/v08/results/`](docs/v08/results/)。这些实验分别回答数学一致性、局部精修、受控盲解及实测回归问题，不能互相替代。性能数字依赖所测问题与环境。

[C++ 内核测量](native/validation/v08_kernel_measurements.json)比较同一解析模型的两种实现；[调度验证](docs/v08/results/scheduler_validation.json)记录资源准入、取消/恢复及两个并发的已知模型精修，配套 [并发请求](examples/v08/parallel_native.json)可直接运行。

发行清单检验的是文件字节。重新运行验证会更新结果文件，此后工作目录与原清单出现差异属于正常情况。需要 DIALS/cctbx 的检查只有在匹配环境中才能完成，跳过项不算实测原始帧验证。

## 历史文件与许可

旧 [0.7.0 英文说明](docs/archive/README_0_7_0.en.md)、[中文说明](docs/archive/README_0_7_0.zh-CN.md)和 [native 0.1.0 说明](docs/archive/README_Ariadne_Xtal_0_1_0.zh-CN.md)均已保留。`reports/`、`native/docs/report.*`、旧验证目录及 `implementation-status.json` 对应原版本；本版以 0.8.0 报告、运行时 capabilities 和 `v08_*` 证据为准。[旧日文说明](README.ja.md)、[基础 API 文档](native/docs/API.md)和 [SHELX 指令矩阵](native/shelx_instruction_matrix.json)仍可查阅，范围以其版本标注为准。

各组成部分继续使用已有许可证：[根目录许可](LICENSE)、[许可说明](LICENSING.md)、[原生引擎许可](native/LICENSE)、[DIALS 补丁许可](patches/DIALS-LICENSE)，以及依赖和报告字体的署名文件。本次整理不改变这些许可范围。源码包不含 SHELX 程序或 DIALS 编译运行库。
