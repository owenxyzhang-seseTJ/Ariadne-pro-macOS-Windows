# Ariadne 原生晶体结构引擎 0.8.0

`ariadne-xtal` 是完整 Ariadne 0.8.0 源码包中的原生结构计算项目。它独立计算 X 射线结构因子、相位候选、解析衍射导数、结构精修及不确定度，并通过结构化接口供 agent 调用。与根目录的 `ariadne` 项目共同安装后，可以直接消费像素似然和重叠观测组。

[完整源码入口](../README.zh-CN.md) · [0.8.0 使用指南](../docs/v08/Ariadne_0_8_0_Guide.zh-CN.md) · [教材 PDF](../docs/v08/Ariadne_0_8_0_Textbook.pdf) · [历史 0.1.0 说明](../docs/archive/README_Ariadne_Xtal_0_1_0.zh-CN.md)

## 当前原生能力

| 问题 | 0.8.0 实现 |
|---|---|
| 几何与对称 | 一般晶胞、精确周期最近像、Gemmi 空间群操作、特殊位置、候选群原点搜索与轨道到不对称单元的投影 |
| 散射 | 独立的运动学 IT92 结构因子，物理占有率，各向同性/各向异性 ADP，显式异常散射项 |
| 相位 | 已知完整晶胞组成的 P1 多起点求解；保留电荷翻转、原子投影，新增 triplet/Patterson/带符号强度近端更新的 `hybrid` |
| 精修 | 全部带符号工作集 F²，解析衍射导数，比例、坐标、ADP、占有率和可选孪晶比例；几何限制保留数值导数 |
| 约束与限制 | 位点对称性、连续原点规范、参数相等与占有率和；距离、角度、平面及 ADP 等显式限制 |
| 不确定度 | 雅可比秩、可辨识子空间、条件协方差与物理参数不确定度；不把伪逆零空间当作已知参数 |
| 像素观测 | 调用 `ariadne` 的固定方差高斯背景剖面压缩、真正光子计数的信号—背景联合泊松估计 |
| 重叠与孪晶 | 整数倒易孪晶律；明确索引的同相多畴高斯组直接精修，并保留不可测分量组合 |
| 自动求解 | `autosolve` 和 `solve-observations`，具有 work/selection/audit 分区、候选选择和封存审计 |
| 密度与绘图 | 非加权 Fo−Fc、2Fo−Fc、计算密度、Patterson、周期峰、CCP4/NPZ，以及离线三维与静态图 |
| Agent 接口 | Python、CLI、JSONL、可选 MCP；与持久化父 agent 的 `native.run` 相连 |

本包实现了上述范围的原生求解和精修，**不声称与 SHELXT/SHELXL 全部等价**。完整 SHELXL 指令语言、AFIX/HFIX、任意 FVAR 关系、中子模型、消光、溶剂掩膜及自动无约束组成推断等不在原生范围。明确的非面网孪晶观测组接口已实现，但不等于自动解释任意 HKLF5 指令。完整泊松像素—结构—几何—吸收优化器尚未接入结构闭环。

当前能力以 `ariadne-xtal capabilities` 为准。[旧指令矩阵](shelx_instruction_matrix.json)和 [兼容性说明](docs/COMPATIBILITY.md)保留了原始指令语义的逐项说明，版本较早；新增原生组接口见 0.8.0 指南。

## 安装

推荐在完整 Ariadne **仓库根目录**安装两个项目：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './python[dev]' -e './native[dev]'
ariadne-xtal --version
ariadne-xtal capabilities
```

只安装结构内核时，在本 `native/` 目录执行 `python -m pip install -e '.[dev]'`。其基础包元数据允许 Python 3.10 及以上；组合安装、持久化 agent 和本版完整验证使用 Python 3.12。像素积分与联合观测构建需要同时安装 `ariadne`。

此安装不包含 DIALS 运行环境，也不需要 SHELX 才能执行原生结构任务。DIALS 安装说明见 [完整源码入口](../README.zh-CN.md)。MCP 是本子项目的可选依赖：

```bash
# 在仓库根目录执行。
python -m pip install -e './native[mcp]'
ariadne-xtal-mcp --root /absolute/path/to/project
```

组合安装会按两个项目的依赖约束选择兼容 SDK；旧 0.1.0 文档记录的 SDK 测量不作为 0.8.0 运行环境声明。

### 可选 C++17 内核

`ARIADNE_BUILD_CPP=auto` 默认尝试可选编译，`0` 禁用编译，`1` 要求成功。需要 C++17 编译器和当前 Python 开发头文件；无 pybind11、NumPy 头文件或 OpenMP 依赖。在完整仓库根目录可执行：

```bash
ARIADNE_BUILD_CPP=1 python -m pip install -e ./native
python -m ariadne_xtal.kernels
```

具备 setuptools 的开发环境可用 `python native/tools/build_optional_kernel.py --inplace` 在源码树内构建。运行时通过 `ARIADNE_NUMERIC_BACKEND=auto|python|cpp` 选择；`auto` 在扩展可用时选择 C++，否则回到 NumPy；显式 `cpp` 缺失会报错，数值错误不会静默回退。状态命令给出实际后端和编译信息，`DiffractionPredictor(..., backend=...)` 也可显式选择。

该核实现结构因子强度及解析导数的块内累加，单线程并释放 GIL；参数化、优化器和科学模型保持原接口。它不把整个程序改成 C++，也不自行启动 OpenMP 线程；并发由父 agent 的独立动作调度管理。安装说明见 [完整指南](../docs/v08/Ariadne_0_8_0_Guide.zh-CN.md)。

## 现成示例

以下命令从完整仓库根目录运行：

```bash
ariadne-xtal run examples/v08/autosolve.json --root examples/v08
ariadne-xtal run examples/v08/solve_observations.json --root examples/v08
ariadne-xtal run examples/v08/refine_fen4.json --root examples/v08
ariadne-xtal run examples/v08/integrate_counts.json --root examples/v08
```

这四项分别演示合成反射盲解、含重叠观测的闭环、公开实测数据的已知模型重精修、光子计数积分。相位任务中 `elements` 是**完整晶胞**的元素列表；调用方必须提供实际晶胞和组成。示例来源见 [PROVENANCE.json](../examples/v08/PROVENANCE.json)。

旧 `native/examples/` 继续保留。若要重新生成其中的演示，请先进入 `native/`，再运行 `python examples/generate_examples.py`；它与根目录 `examples/v08/` 的新示例不同。

## 结构化请求

请求文件本身按命令行文件路径读取；`structure`、`hkl`、`observations` 等输入字段必须是 `--root` 内的相对路径。

```json
{
  "action": "refine",
  "structure": "start.json",
  "hkl": "measured.json",
  "options": {
    "max_nfev": 300,
    "jacobian": "analytic",
    "reflection_block_size": 512,
    "covariance_scale": "absolute"
  },
  "graphics": false
}
```

保存为项目目录中的 `request.json` 后执行：

```bash
ariadne-xtal run /absolute/path/to/project/request.json --root /absolute/path/to/project
```

每次任务新建 `runs/<action>_.../`，其中包含输入快照、环境/哈希清单、`result.json` 及相应结构文件。`refine` 可选生成图形；不可表示为 RES 的模型会保留 JSON/CIF 并记录原因。

`ok: true` 表示动作执行完毕。`refine` 还需要看 `success`、`metrics.identifiable` 和残差；自动求解应查看 `status`、候选选择、分区与审计。不可辨识的孪晶比例不会作为有可靠标准差的独立测量报告。

### 重叠观测怎样进入精修

固定方差高斯像素经背景剖面化后，每个局部组保存 `R`、`z` 和 `residual_constant`。`hkl` 给出每一列对应的分量；`coefficients` 是已明确的部分性/校正系数，`domain_ids` 给出晶畴。

```json
{
  "action": "joint-refine",
  "structure": "start.json",
  "observations": "groups.json",
  "domain_fractions": [0.6, 0.4],
  "refine_domain_fractions": true,
  "phase_relation": "same_phase_supported"
}
```

`groups.json` 的顶层为 `{"groups": [...]}`，每个元素是 `GaussianStructureGroup` 的序列化结果。多畴接口要求各分量已经映射到同一结构相的晶胞，且明确声明 `same_phase_supported`。本接口固定原来的像素方差、轮廓、背景空间和校正系数；改变这些量应重新建立观测算子。

`solve-observations` 可先从几何隔离参考畴的 work 观测初始化，再对所有 work 组精修。完全没有足够初始化证据时，可给定 `structure` 作为明确的初始假设；程序不会把混合总强度假装成某一个晶畴的振幅。

## Python 调用

```python
from ariadne_xtal.io import read_structure, read_reflections, write_structure
from ariadne_xtal.refinement import RefinementOptions, refine

model = read_structure("start.json")
data = read_reflections("measured.json")
fit = refine(model, data, RefinementOptions(jacobian="analytic", max_nfev=300))
print(fit.success, fit.metrics["identifiable"], fit.metrics["wr2"])
write_structure(fit.structure, "refined.cif")
```

直接连接像素与结构：

```python
from ariadne.reduction_native import compress_gaussian
from ariadne_xtal.joint_refinement import GaussianStructureGroup, refine_observations

# y、P、variance、B 及 hkl 由调用方的实际积分模型提供。
observation = compress_gaussian(y, P, variance=variance, background=B)
group = GaussianStructureGroup.from_observation(observation, hkl)
result = refine_observations(model, [group])
```

这段代码不先求每个重叠分量的强度，而是直接评价结构在可测强度组合上的预测。实际数据仍需提供足以确定所精修参数的多个组。参数和基础数据类型见 [API 契约](API_CONTRACT.md)；[基础 API 文档](docs/API.md)保留原接口的详细约定。

## JSONL、MCP 与持久化 agent

```bash
ariadne-xtal serve --root /absolute/path/to/project
```

标准输入每行一个请求，标准输出每行一个响应。已注册动作包括 `capabilities`、`inspect`、`solve`、`autosolve`、`solve-observations`、`integrate`、`joint-refine`、`symmetry`、`refine`、`maps`、`absolute` 和 `legacy`。MCP 对外提供 `crystal_capabilities()` 与 `crystal_execute(request)`；没有任意 shell 或 Python 执行工具。

持久化工作流由根项目负责：

```bash
ariadne-agent init examples/v08/persistent_native.json runs/new-session
ariadne-agent run runs/new-session
ariadne-agent status runs/new-session
```

该示例通过 `native.run` 制作隔离输入副本并执行原生精修，然后打包证据。父 agent 还支持 `scheduler` 会话配置和节点 `resources`：独立就绪动作在进程中并发，统一控制线程与申报内存的准入；默认顺序执行。申报内存不是 RSS 硬上限，单个原生求解动作的内部起点也不会自动拆开。详细的恢复、预算和并发示例见 [完整指南](../docs/v08/Ariadne_0_8_0_Guide.zh-CN.md)。

## 显式外部 SHELX 接口

需要保留完整原始指令语义时，可由用户单独安装官方程序并明确指定路径：

```bash
ariadne-xtal serve --root /absolute/path/to/project \
  --shelxt /absolute/path/to/shelxt --shelxl /absolute/path/to/shelxl
```

随后提交例如 `{"action":"legacy","engine":"shelxl","input":"sample.ins","timeout":300}`。程序在隔离目录中传递原始 INS/HKL 及明确的附属文件；原生失败不会自动切换到外部引擎。传输测试使用明确标识的测试可执行程序，未据此宣称官方数值等价。

## 单位、文件与测量含义

内部长度为 Å，角度为度，位置为分数坐标，ADP 为笛卡尔 Å² 张量，倒易基矢不含 `2π`。占有率是物理位点占有率；完全占据的特殊位置仍为 1。SHELX 特殊位置 SOF 在支持的导入/导出子集中按重数换算。

JSON 保存全部原生字段。CIF 用于结构互操作，HKLF4 保留负强度和 batch。原生 `ShelxDocument` 原样保存指令文档，而 `to_structure()` 只转换具有明确原生含义的子集；未翻译关系会返回具体错误。完整复合组及协方差应保留在组文件中，不能用独立标准差强行表示。

`Atom.f_prime` 与 `Atom.f_double_prime` 明确控制异常散射；`with_anomalous_factors()` 可主动调用 Gemmi 的孤立原子色散计算。近吸收边的实验校准仍由调用方负责。负测量强度参与 F² 精修及新 hybrid 强度似然，不因开方需要而从观测中删除。

## 验证、报告与源代码位置

从完整仓库根目录执行：

```bash
python -m pytest tests tests_rebuild -q
ARIADNE_MCP_STDIO_TEST=1 python -m pytest native/tests -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/validation/run_v08_refinement.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/validation/run_v08_joint.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/validation/run_v08_phasing_autonomy.py
```

[实测 FeN₄ 回归](validation/v08_real_fen4/result.json)使用 COD 2242624 的 71 个测量 F²，从已有结构精修 10 个参数；数据来源与许可保存在 [public_data](validation/public_data/README.md)。这是前向模型和局部精修的实测回归。合成盲解、重叠组恢复、导数精度和性能分别记录在 `validation/v08_*`，不据此推断任意 MOF、未知结构或真实帧的成功率。

[C++ 数值与性能记录](validation/v08_kernel_measurements.json)比较相同解析模型的两个后端；编译后可从仓库根目录运行 `OPENBLAS_NUM_THREADS=1 python native/validation/run_v08_kernels.py`。父 agent 的 [调度记录](../docs/v08/results/scheduler_validation.json)独立检验并发、资源准入、取消与恢复。

| 源文件 | 主要职责 |
|---|---|
| `model.py`、`symmetry.py` | 数据类型、晶胞和对称操作 |
| `scattering.py`、`derivatives.py` | 物理结构因子与解析衍射导数 |
| `kernels.py`、`cpp/intensity_kernel.cpp` | 可选 C++ 数值实现、输入契约与实际后端查询 |
| `direct_methods.py`、`phasing.py` | 稀疏相位关系、径向似然更新与双空间求解 |
| `spacegroup_search.py` | 原点和候选空间群假设 |
| `refinement.py`、`restraints.py` | 精修参数化、物理约束、几何限制与协方差 |
| `reduction_bridge.py`、`joint_refinement.py` | 像素观测桥接及重叠组结构目标 |
| `autonomy.py`、`evidence_solver.py` | 候选求解、独立选择和原始观测闭环 |
| `agent.py`、`cli.py`、`mcp_server.py` | 结构化执行及来源记录 |
| `maps.py`、`visualization.py`、`absolute.py` | 密度图、绘图和绝对结构诊断 |

当前完整教材位于根目录 [`docs/v08/`](../docs/v08/)。本子项目的 `docs/report.*`、`validation/real_fen4/` 和旧交付记录属于 0.1.0 历史材料；其原始说明已归档，仍可追溯。

本子项目保留其 [MIT 许可证](LICENSE)。根项目、DIALS 补丁、第三方依赖和字体分别遵循已有许可及 [第三方说明](THIRD_PARTY_NOTICES.md)，本次组合交付不改变各自范围。
