# Ariadne 0.8.0 使用指南

本指南对应完整的 0.8.0 本地源码包。目标是让读者从一份有明确来源的观测出发，运行原生积分、结构求解、精修及 agent，并能判断结果究竟说明了什么。数学推导见 [完整教材](Ariadne_0_8_0_Textbook.pdf)及 [LaTeX 源文件](Ariadne_0_8_0_Textbook.tex)；结构反馈另见 [loop 专题](Ariadne_0_8_0_Loop.tex)。

## 1. 源码包由哪几部分组成

| 目录 | 作用 |
|---|---|
| `python/` | `ariadne` 包：像素观测似然、旧反馈接口、持久化 agent、CLI/MCP |
| `native/` | `ariadne-xtal` 包：原生相位、对称候选、解析精修、密度图、原生 agent |
| `skills/ariadne-reduction/scripts/` | 保留的预测几何、体素轮廓、积分、复合缩放和导出算法 |
| `patches/` | 与 DIALS 3.30.1 匹配的源码补丁及许可证 |
| `examples/v08/` | 可直接执行的 0.8.0 请求、合成输入和有来源的实测回归输入 |
| `tests_rebuild/`、`native/tests/` | 根项目及原生引擎测试 |
| `benchmarks/`、`native/validation/` | 数值实验、可复现脚本与结果 |
| `docs/v08/` | 本版教材、loop 报告、图表、构建脚本和验证记录 |

`ariadne-agent` 与 `ariadne-xtal` 解决不同层次的问题。前者保存任务图、外部环境配置、预算和恢复状态；后者执行一次明确的晶体学计算。默认执行不要求在线语言模型。语言模型可以选择已注册的动作，但结构因子、坐标、似然和协方差由数值程序计算。

本包是两者及保留数据还原代码的完整源码交付，不只是补丁或单个 wheel。它不是 SHELXT/SHELXL 全部指令的等价重写，也不包含官方二进制程序。

## 2. 安装与环境

以下命令在解压后的 Ariadne 根目录执行。组合安装使用 Python 3.12 或更高版本。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './python[dev]' -e './native[dev]'
ariadne-agent capabilities
ariadne-xtal capabilities
```

`requirements-tested.txt` 保存本次验证使用的直接依赖版本，可在独立环境中先安装该文件，再安装两个本地项目。它是复现记录，不是宣称其他满足 `pyproject.toml` 的版本都不能使用。

```bash
python -m pip install -r requirements-tested.txt
python -m pip install -e './python[dev]' -e './native[dev]'
```

持久化 agent 使用 POSIX 文件锁及进程管理，建议在 Linux、macOS 或 WSL 运行。只安装 `native/` 可以执行常规结构求解与精修；像素 `integrate` 和建立高斯观测因子需要根 `ariadne` 包。完整源码目录还保存补丁、科学脚本和 expert pack，不能因安装 wheel 后就删除。

旧科学脚本定位失败时，可显式指定源码根目录：

```bash
export ARIADNE_PLUGIN_ROOT=/absolute/path/to/Ariadne
```

DIALS 编译环境不由上述 pip 安装提供，配置方法见第 9 节。

### 2.1 可选 C++ 数值内核

原生精修中的结构因子强度及解析导数块内累加，可以交给随源码提供的 C++17 扩展。Python 仍负责物理参数化、约束、数据划分与优化控制；NumPy 参考实现保留。

安装时由 `ARIADNE_BUILD_CPP` 控制构建：`auto` 为默认值，尝试可选编译；`0` 只安装 Python 实现；`1` 要求编译成功。编译需要 C++17 编译器和当前 Python 的开发头文件，不需要 pybind11、NumPy 头文件或 OpenMP。需要明确要求编译时，在仓库根目录执行：

```bash
ARIADNE_BUILD_CPP=1 python -m pip install -e ./native
```

已经具备 setuptools 的开发环境，也可直接构建源码树中的扩展；该工具本身不下载依赖：

```bash
python native/tools/build_optional_kernel.py --inplace
python -m ariadne_xtal.kernels
```

构建与运行选择是两件事。运行时 `ARIADNE_NUMERIC_BACKEND=auto` 在扩展可导入时选择 C++，否则使用 Python；`python` 强制参考实现；`cpp` 要求扩展存在，否则明确报错。状态命令报告 requested、resolved、compiled_available 及编译信息。数值计算错误不会触发静默更换后端。

```bash
ARIADNE_NUMERIC_BACKEND=python ariadne-xtal run examples/v08/refine_fen4.json --root examples/v08
ARIADNE_NUMERIC_BACKEND=cpp ariadne-xtal run examples/v08/refine_fen4.json --root examples/v08
```

C++ 核使用单线程、释放 GIL，不创建 OpenMP worker。外层 agent 调度独立动作，数值库线程按会话预算设置，避免多层各自扩大并行度。它改变的是计算实现，观测似然、散射因子、单位和约束保持相同；源码包中的扩展需针对用户的 Python 与平台构建，不能直接移植另一环境生成的二进制。编译成功后，可运行 `OPENBLAS_NUM_THREADS=1 python native/validation/run_v08_kernels.py` 复现同一解析模型的两种实现比较；结果见 [`v08_kernel_measurements.json`](../../native/validation/v08_kernel_measurements.json)。

## 3. 先运行四个示例

随包输入已经生成。若需要重建它们，在仓库根目录执行 `python examples/v08/generate_examples.py`；该脚本会更新同目录的示例输入和请求。

### 3.1 光子计数积分

```bash
ariadne-xtal run examples/v08/integrate_counts.json --root examples/v08
```

此例的两个像素计数为 `[8, 3]`。第一个像素包含信号与背景，第二个只测背景；信号和背景同时估计。返回的条件剖面区间会在每个试探信号下重新优化背景，不把估计背景当成已知真值。区间采用渐近似然比阈值，不承诺稀疏计数或边界情形下的精确覆盖率。

### 3.2 合成反射盲解

```bash
ariadne-xtal run examples/v08/autosolve.json --root examples/v08
```

请求提供晶胞、四个碳原子的完整晶胞组成、带符号强度及不确定度。真值坐标只在生成和验证脚本中使用，没有作为求解输入。`hybrid` 进行多起点相位搜索，然后精修候选、比较独立选择集并评价封存审计集。

### 3.3 含重叠观测的结构闭环

```bash
ariadne-xtal run examples/v08/solve_observations.json --root examples/v08
```

输入包含隔离参考畴、另一晶畴，以及完全或部分重叠的观测组。相位初始化只使用几何上隔离的参考畴 work 观测，其中允许负值；后续精修使用全部 work 组。结构参数和晶畴比例共同解释原观测，没有中间伪造的逐分量去孪晶测量表。

### 3.4 公开实测 FeN₄ 重精修

```bash
ariadne-xtal run examples/v08/refine_fen4.json --root examples/v08
```

该例使用 COD 2242624 的 71 个实测 F² 及其标准不确定度，从已发表结构出发精修 10 个参数。它检验前向模型与局部精修的一致性，不是未知结构的从头求解成功率实验。来源和许可见 [`native/validation/public_data/README.md`](../../native/validation/public_data/README.md)。

以上动作均在 `examples/v08/runs/` 下创建新目录；响应中的 `result` 和 `manifest` 指向该工作区内的结果与来源清单。`ok: true` 仅表示动作执行完成。继续检查结果中的数值收敛、可辨识性、残差、候选选择和审计状态。

## 4. 常规结构输入

结构 JSON 保存晶胞、空间群、独立位点及位移参数。例如：

```json
{
  "cell": {"a": 8, "b": 9, "c": 10, "alpha": 90, "beta": 103, "gamma": 90},
  "spacegroup": "P 1",
  "wavelength": 0.71073,
  "atoms": [
    {"label": "C1", "element": "C", "frac": [0.12, 0.23, 0.34],
     "occupancy": 1, "u_iso": 0.02}
  ]
}
```

反射输入使用 `hkl`、`intensity`、`sigma` 数组，或支持的 HKLF4 文件。`sigma` 严格为正，负强度保留。这里的元素列表与相位请求中的 `elements` 不同：结构 JSON 保存独立位点；P1 相位请求的 `elements` 必须列出**整个晶胞的所有原子元素**，不能只填不对称单元组成。

| 量 | 约定 |
|---|---|
| 晶胞边长、波长 | Å |
| 晶胞角 | 度 |
| 原子位置 | 分数坐标 |
| 位移参数 | `u_iso` 为 Å²；`u_aniso` 为笛卡尔 3×3 张量，单位 Å² |
| 倒易基 | 不含 `2π`；相位写作 `exp(2π i h·x)` |
| 占有率 | 物理位点占有率，完全占据的特殊位置仍为 1 |
| 强度尺度 | 测量单位由输入及比例因子说明，不能在流程中静默改变 |

CIF 可用于结构互操作。INS/RES 的原始文档保存与原生模型转换是两条路径：遇到没有原生含义的自由变量、指令或约束时，转换器返回具体原因。完整复合观测及协方差应保存为原生组数据；HKLF4/HKLF5 不能随意表达任意相关似然。

## 5. 从像素保留可测信息

### 5.1 固定方差的高斯模型

设像素向量为 `y`，反射轮廓为 `P`，背景基为 `B`，方差为已校准且固定的 `variance`。调用：

```python
from ariadne.reduction_native import compress_gaussian

observation = compress_gaussian(
    y, P, variance=variance, background=B, valid=valid_mask, chunk_rows=1024
)
print(observation.rank, observation.nullspace)
print(observation.chi2(trial_intensities))
record = observation.to_dict()
```

得到的因子满足

\[
\min_\beta \|W^{1/2}(y-P I-B\beta)\|^2
=\|R I-z\|^2+c.
\]

`R` 的列对应原反射分量，不能在后续映射时调换单位。其行只保存可测组合。两个完全相同的轮廓只能测到总贡献，单独的两项强度不可辨识；这不是优化失败，而是测量本身缺少区分它们的信息。

流式 QR 处理像素块，减少临时矩阵内存；调用者提供的 `P` 和数据本身仍占内存。本包的测量记录明确区分临时分配峰值与进程总内存，也不把降低内存等同于所有规模下更快。

### 5.2 真正计数与带读出噪声的数据

`PoissonObservation` 对原始整数计数建立 `mu = offset + P I + B beta`，要求均值贡献非负，且 `I`、`beta` 非负。已知 `offset` 是已知期望计数，不是从观测中扣除的随机背景。

```python
from ariadne.reduction_native import PoissonObservation

problem = PoissonObservation(counts, P, background=B)
fit = problem.fit()
print(fit.to_dict())
interval = fit.profile_interval([1.0], confidence_level=0.95)
```

上例的对比向量 `[1.0]` 只适用于单分量问题；多分量要按 `P` 的列数给出要评价的线性组合。`profile_background(fixed_intensities)` 可返回固定信号下重新优化背景的似然及梯度，但当前自动结构闭环仍消费高斯观测组。

结构化 `integrate` 请求必须声明噪声来源。高斯路径要求 `variance_fixed: true`；纯泊松路径要求已确认是真正计数、单位计数增益和零读出方差。带读出噪声的 ADU 应使用保留的校准 Poisson–Gaussian 后端，不能把负值截断或四舍五入后冒充光子数。`route_region` 返回相应路径；原生 `integrate` 对尚需该校准后端的输入给出明确状态。

### 5.3 强与弱的分工

强而清楚的峰适合确定位置、几何和平均轮廓；弱或重叠峰对这些局部自由度的信息不足。0.8.0 根据已声明噪声和轮廓的可辨识性诊断这种区别。它不使用测得的强度符号或 `I/σ` 作为观测保留条件。预期计数若用于计划，须说明独立来源。

自动原生求解的几何隔离判据不等同于强度阈值。孤立但微弱、甚至为负的估计也会保留；重叠组在联合精修中继续提供可测模式。真实图像上的几何锚定仍由 DIALS/保留的还原模块完成。

## 6. 从观测因子到结构

```python
from ariadne_xtal.joint_refinement import GaussianStructureGroup, refine_observations

group = GaussianStructureGroup.from_observation(
    observation,
    hkl=component_hkl,
    coefficients=component_coefficients,
    domain_ids=domain_ids,
    role="work",
)
result = refine_observations(
    initial_structure, groups,
    domain_fractions=[0.6, 0.4], refine_domain_fractions=True,
    phase_relation="same_phase_supported",
)
```

此处 `groups` 是已经构建的多个观测组列表，包含上面构建的 `group`。每一列的模型贡献为

\[
I_j(\theta,\alpha)=c_j\alpha_{d(j)}k|F_{h_j}(\theta)|^2.
\]

`hkl` 是映射到共同结构相晶胞的分量指标；`coefficients` 是已明确的校正系数；`domain_ids` 是从零开始的畴编号；`domain_fractions` 包含全部畴且和为 1。这里的知识应来自实际指标化、几何与校准，不能通过结构先验猜成实验事实。若部分性或校正已经包含在 `P` 的列中，同一因子不得再次乘入 `coefficients`；轮廓列的归一化约定必须与强度单位一同保存。

`joint-refine` 对已知初始结构计算上述目标。`solve-observations` 还执行相位初始化和候选选择。盲初始化至少需要十个独立的隔离 domain-0 work Friedel 对；不满足时可给定明确的初始 `structure`。多个组只有在确实属于同一结构相时才使用 `same_phase_supported`。

背景空间、轮廓、像素掩膜、方差或校正改变后，原压缩因子一般不再适用，需回到像素重新构建。完全重叠且无异常散射的反演畴可能具有严格不可辨识的比例；共同结构也不能创造不存在的信息。

## 7. 结构反馈 loop 怎样使用理想模型

对已知化学组成的 MOF，可把理想拓扑、键长范围、金属配位和有化学依据的候选结构作为假设。它们可减少搜索空间，但仍可能遗漏缺陷、无序、部分占位或另一相。原始像素或实验强度始终固定，结构只负责产生预测。

当前原生闭环采取以下分工：

1. **work**：生成相位候选并精修；隔离参考畴负责初始化，全部 work 组负责最终似然。
2. **selection**：比较已经产生的候选。按选项可将选定模型在 work+selection 上重新精修。
3. **audit**：候选与策略冻结后评价一次，不把审计误差再反馈给同一轮搜索。

Friedel 对、所比较对称下的等价反射和共享像素的重叠分量组成不可分开的证据连通分量，再进行划分。否则，一个观测的等价副本可能同时进入训练与验证。修改候选策略后反复查看同一 audit，不能继续称为独立审计。

原生闭环已实现“隔离观测初始化—全部组结构精修—独立选择—审计”。原始帧的几何、轮廓、吸收与结构在同一个外层交叉验证中全部更新，是更大的问题；本版没有以图像实测证明该完整机制已经解决。

## 8. 原生 agent 与持久化任务

### 8.1 单次任务、JSONL 和 MCP

```bash
ariadne-xtal run request.json --root /absolute/path/to/project
ariadne-xtal serve --root /absolute/path/to/project
ariadne-xtal-mcp --root /absolute/path/to/project
```

JSONL 每行接收一个请求。MCP 提供 `crystal_capabilities` 与 `crystal_execute`。新增求解动作通过 JSON 请求运行；不要把 `autosolve` 或 `solve-observations` 当成未经定义的独立 CLI 子命令。

### 8.2 持久化执行与恢复

```bash
ariadne-agent init examples/v08/persistent_native.json runs/fen4-session
ariadne-agent run runs/fen4-session
ariadne-agent status runs/fen4-session
ariadne-agent result runs/fen4-session refine
ariadne-agent export runs/fen4-session fen4-session.zip --include-inputs
```

相对输入路径按请求文件所在目录解析，新会话目录必须尚不存在。`native.run` 的输入映射把源文件复制到隔离原生工作区，并校验哈希；节点参数不能伪造字面量文件来源。

`cancel` 请求取消，`resume` 只清除会话取消标志。已经成为 failed、cancelled、blocked 或 interrupted 的节点，应在预算允许时显式 `retry SESSION NODE`，再运行会话；成功节点不重做。若进程异常结束留下 running 标记，应先确认旧进程已停止，再使用 `recover ... --confirmed-process-stopped` 将其记录为 interrupted，然后决定是否重试。已经执行过的最终测试不是普通可重试数据查询。具体状态以 `status` 返回内容为准。

持久化入口原有 `mode: solve/refine/reduce` 仍包含配置好的外部 SHELX 流程；要运行新原生路径，使用 `mode: custom` 和 `native.run`，如随包示例。原有执行路径、MCP 会话、planner 与 expert pack 配置保存在 [0.7 操作指南](../../AGENT_GUIDE.zh-CN.md)，其历史能力和验证数字不能代替本版记录。

### 8.3 有资源预算的并发调度

在持久化请求顶层设置 `scheduler`，在需要的节点上声明 `resources`。例如，下列配置最多同时运行两个各占一个线程预算、256 MB 声明内存的动作：

```json
{
  "scheduler": {
    "max_workers": 2,
    "cpu_threads": 2,
    "memory_mb": 512,
    "default_task_memory_mb": 256,
    "isolate_actions": true
  }
}
```

该片段需要合并到完整请求中，不是独立任务。节点可选字段为 `"resources": {"cpu_threads": 1, "memory_mb": 256}`。`cpu_threads` 是会话总线程预算；`memory_mb` 是各活动节点申报量之和的准入上限，**不是实测 RSS 或操作系统内存硬上限**。线程设置约束配合这些环境变量的数值库，也不是 CPU 核心绑核或 cgroup 配额。

默认 `max_workers=1`、`cpu_threads=max_workers`、`memory_mb=null`、`default_task_memory_mb=256`，`isolate_actions` 默认随 `max_workers>1` 启用。多 worker 必须使用隔离动作。每个动作具有独立 Python 进程、POSIX 进程组和 attempt 目录，父调度器负责图状态与资源账本。单节点申报超出会话预算会在执行前拒绝。

```bash
ariadne-agent init examples/v08/parallel_native.json runs/parallel-session
ariadne-agent run runs/parallel-session
ariadne-agent status runs/parallel-session
```

只有依赖已经完成、资源能够容纳的动作才会启动。这里并行的是独立 DAG 节点，单个原生求解动作内部的多起点循环不会因此自动拆分。`run --steps N` 最多准入 N 个动作，并等待已启动动作结束；再次 `run` 从剩余工作继续。数值 BLAS/OMP 线程预算与外层 worker 数共同控制，避免每个进程都默认占用全部线程。

[并发示例](../../examples/v08/parallel_native.json)从两个初值精修同一份 FeN₄ 实测数据，演示调度和恢复；两个初值不是两份独立验证数据。执行 `python benchmarks/v08_scheduler.py --with-tests` 可复现调度检查及该工作流。取消、超时、父进程中断和会话排他锁的实际记录见 [`scheduler_validation.json`](results/scheduler_validation.json)；该实验固定 Python 后端，独立于 C++ 加速测量。

## 9. DIALS、原始图像和旧能力

匹配环境的安装器为：

```bash
sh scripts/install_local.sh \
  /absolute/path/to/dials-conda-prefix \
  /absolute/path/to/NEW-ariadne-runtime
```

它要求或创建匹配 DIALS 3.30.1、dxtbx 3.30.0 的 conda 环境，在新目录应用 Python 源码补丁，保留匹配的编译扩展。不存在基础环境时，`ARIADNE_MAMBA` 可指定 mamba 可执行文件；需要下载匹配源码和依赖。脚本不运行科学验证，末尾说明如何另行安装 `native/`。

默认启动器为 `~/CrystallographyPrograms/ariadne-env`。完整元数据写入覆盖层的 `runtime.json`，发行版本从当前 `python/pyproject.toml` 读取。安装后检查实际 DIALS 命令、输入格式和几何元数据，再配置持久化 agent 的外部程序路径。

保留的 `request-from-images` 可生成已有单晶图像流程请求；复杂孪晶需显式的联合分析工作流。HDF5 master/data 等有侧文件依赖的格式必须保留完整集合，不能只复制主文件。原始图像模板和旧工作流配置见 [原 agent 指南](../../AGENT_GUIDE.zh-CN.md)与 [R2 文档](../R2_README.zh-CN.md)。

0.8.0 新数值核不在导入时依赖 DIALS。它能验证观测模型及结构算法，但这不等于已经在当前环境运行完整原始帧处理。匹配的编译运行库、探测器标定、实际样品数据和仪器格式仍是实测验证的必要输入。

## 10. 验证、性能与报告重建

在仓库根目录执行以下命令，可复现对应的数值实验；多起点求解和精修基准需要一定时间。

```bash
python -m pytest tests tests_rebuild -q
ARIADNE_MCP_STDIO_TEST=1 python -m pytest native/tests -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/v08_reduction.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/validation/run_v08_refinement.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/validation/run_v08_joint.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/validation/run_v08_phasing_autonomy.py
```

| 记录 | 回答的问题 |
|---|---|
| [`v08_reduction_results.json`](../../benchmarks/v08_reduction_results.json) | 压缩目标与像素目标的一致性、重叠可辨识性、计数剖面和临时内存 |
| [`v08_refinement_measurements.json`](../../native/validation/v08_refinement_measurements.json) | 解析导数、同一局部精修问题的耗时、QR/SVD 内存 |
| [`v08_joint_measurements.json`](../../native/validation/v08_joint_measurements.json) | 原像素与共同结构目标一致、可测与不可测孪晶情形 |
| [`v08_phasing/result.json`](../../native/validation/v08_phasing/result.json) | 合成盲解和含重叠观测的完整原生闭环 |
| [`v08_real_fen4/result.json`](../../native/validation/v08_real_fen4/result.json) | 公开实测数据的已知模型重精修 |
| [`v08_kernel_measurements.json`](../../native/validation/v08_kernel_measurements.json) | 同一解析衍射模型的 Python/C++ 数值、计时与内存比较 |
| [`scheduler_validation.json`](results/scheduler_validation.json) | 独立动作并发、资源准入、取消、恢复和实际双初值工作流 |
| [`results/`](results/) | 本版测试 XML 与示例执行记录 |

性能报告包含输入规模、环境和计时范围。解析导数与有限差分的比较不等于同 SHELXL 比较；保存更小的 QR 因子也不等于整个优化器只占常数内存。密集非线性最小二乘仍保存实雅可比矩阵。合成结果与实测局部回归分别提供证据，没有替代真实原始帧的全流程检验。

报告的图表来自随包脚本与结果文件，精确几何图使用 TikZ，数值图使用标准绘图库。安装 XeLaTeX 及报告所列宏包后，在仓库根目录执行 `python docs/v08/build_report.py`；工具同时构建完整教材和 loop 报告，并检查引用和排版警告。图表重建工具位于 `tools/`。`fonts/` 包含本报告使用的重命名字体子集、OFL 许可和来源。若修改中文正文引入子集中没有的字形，应按字体工具说明重建子集。

`python scripts/verify_release.py` 可检查发行清单。请先检查原始交付，再重跑会更新结果文件的实验；后者自然会改变清单中的字节。

## 11. 历史文档与许可

根 `reports/`、`native/docs/report.*`、旧 `validation/real_fen4/`、`implementation-status.json` 和早期验证记录属于以前版本。旧 README 已保存到 [`docs/archive/`](../archive/)，没有删除原配置与能力说明。本版以 0.8.0 教材、当前 `capabilities`、`requirements-tested.txt` 及 `v08_*` 结果为准。

根项目、native、DIALS 补丁、第三方依赖、实测数据和字体分别遵循已有许可。查看 [`LICENSING.md`](../../LICENSING.md)、[`native/LICENSE`](../../native/LICENSE)、[`patches/DIALS-LICENSE`](../../patches/DIALS-LICENSE)及相应来源文件。本次发行整理未改变它们的许可范围。
