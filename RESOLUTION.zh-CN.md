# 孪晶与空间群解析

Ariadne Pro 包含 `ariadne-reduction` 和 `solve-single-crystal`。还原 skill 交接原始观测、基底和匹配的 INS/HKL。解析 skill 依据实测反射、局部像素及化学证据比较空间群和孪晶模型。CLI、Pro MCP 与工作台共用持久候选图。

一颗单晶衍射样品可以含多个晶畴。面网孪晶的倒易点阵可以完全重叠，需结合反射强度比较单畴和孪晶模型。对分裂或部分重叠的峰，则用独立斑点和探测器像素核查各畴。DIALS 指标化诊断中的 `n_crystal_models` 统计不同的晶格模型对象。`n_crystals` 保留同一计数以兼容旧接口。该阶段的物理畴数为待定，记为 `domain_count: null`。

## 校正强度与初始结构

以下请求从同一晶胞和指数基底的模型、校正 HKLF 4 开始。空间群列表可包含候选子群、超群与设置。直接使用带 `free_flags` 的反射 JSON 时将 `reflection_format` 设为 `json`。结构因子 CIF 用 `cif-f2`，保留带符号的实测强度与标准不确定度。

```json
{
  "mode": "resolve",
  "inputs": {"structure": "initial.cif", "hkl": "measured.hkl"},
  "settings": {
    "reflection_format": "hklf4",
    "spacegroup_candidates": ["P 1", "P -1"],
    "short_evaluations": 8,
    "full_evaluations": 100,
    "shortlist": 8,
    "finalists": 4,
    "inspect_rejections": 1
  },
  "scheduler": {"max_workers": 2, "cpu_threads": 2, "memory_mb": 2048},
  "cache": {"directory": "/absolute/path/project-cache", "max_mib": 256}
}
```

```bash
./macos/ariadne-pro-env -m ariadne.agent.cli init request.json sessions/comparison
./macos/ariadne-pro-env -m ariadne.agent.cli run sessions/comparison
./macos/ariadne-pro-env -m ariadne.agent.cli result sessions/comparison select
```

Windows 对应 `windows\ariadne-pro-win.cmd agent init request.json sessions\comparison`、`agent run` 和 `agent result`。工作台在“孪晶与高级分析”中选择“孪晶与空间群比较”。项目内文件沿用搭配关系。选中科研候选后查看对应的 CIF 和观测证据。

Pro MCP 的标准入口是 `agent_capabilities`、`agent_init`、`agent_submit`、`agent_run_status`、`agent_events` 和 `agent_result`。持久运行支持断连、复用提交键、取消与续跑。`agent_result` 可用 JSON Pointer 读取训练或选择阶段的局部结果，例如 `/preferred/fraction`。连接的服务器应声明 `resolve.prepare` 和 `resolve.screen`。源码 checkout 可用 macOS 启动器执行 `-m ariadne.server`，Windows 执行 `ariadne-pro-win.cmd mcp`。安装 Python 包的环境可使用 `ariadne-mcp`。如果科学包安装在源码树之外，用 `ARIADNE_PLUGIN_ROOT` 指向对应的平台 `source` 目录，以便定位 skill 脚本。

新解初始模型也可提供 `settings.cell` 与全晶胞的 `settings.elements`，或声明 `solver: "shelxt"` 并传适用的 `ins`。其他经确认的 Laue 群控制文件以 `laue_*` 命名，逐个运行。SHELXT 生成的 RES/HKL 保留各自的配对。基底需用配对工作观测核实；需另行转换的候选在报告中列出。复杂 SHELXL 指令、约束与掩膜附档可继续用配对原件进入既有 SHELXL 路线。

## 判别与观测使用

- `resolve.prepare` 按共同倒易操作、Friedel 与显式共享证据连接划分观测。冻结前保存工作、选择与审计三个角色。逐类统计消光、已测覆盖和可由另一个晶畴填充的禁反射。可用 `coverage_hkl` 声明实验覆盖清单。半晶胞 Patterson 取样同时给出奇偶类的观测数、强度和采样掩膜对比度，便于区分赝平移与缺测。
- `resolve.screen` 按物理可行性枚举候选。一份模型的多个孪晶律共用去重的 HKL 与结构因子，两列加权拟合给出条件尺度、畴分数、秩和剖面区间。工作集负责筛选和短精修，选择集用于晋级及同一观测组的配对比较。参考单畴模型和仍有竞争力的高对称、孪晶分支可以并存。
- `select` 可返回并列或证据不足。最终审计只评估冻结且可判别的候选。跨批次误差、反常散射和软模型不确定度需以真实覆盖核查。原生 `absolute` 可作固定模型 Friedel 商诊断，若有实际波长，可显式设置 `anomalous_wavelength`。
- `symmetry-evidence`、`twin-screen` 和 `phase-twin` 也可作为 `native.run` 内的单独动作。需要模型和外部记录交叉验证时可启用 `crosscheck`，指定本机 PLATON。附原始 Fo/Fc 的 `fcf` 与同次生成的 `fcf_control` 可执行有界 TwinRotMat。

## 赝度量与部分重叠

晶胞近似对称用于提出孪晶律。真实探测器峰宽与转角确定是否进入重叠处理。已有共同基底的有理子晶格变换可放在 `extra_laws`，诊断输出给出其交叠子晶格的指数和实测整指数配对数。

既有 `analysis.run` 的 `build_overlaps` 接受 EXPT、完整范围的预测 REFL 和已校准轮廓库。加入 `target-frames` 可按物理扫描分配帧预算。定向候选使用峰分离、局部峰形和解码成本，并保留扫描内随机窗口。连接的重叠峰一起进入拟合。只在定位受影响扫描后扩大范围。

也可以在 `resolve` 请求中提供 `inputs.experiments`、`inputs.predictions` 和 `inputs.profiles`，再加 `settings.pixel_probe: {"target-frames": 150}`。同一会话调度择窗、逐组像素联合拟合与候选比较。工作台用 `settings.resolution_inputs` 按相同角色指定项目内的相对路径，也可在界面选择项目中的 HKL 或 FCF。经验模板可随 `template_*` 输入提交。同一批观测的校正值和全局校正几何分别对应 `corrections` 和 `scale_geometry`。提供其中一种后可用 `settings.pixel_export_hklf5: true` 加挂导出节点。导出器核对全部贡献者、同相证据和校正谱系，生成 HKLF 5 与逐组映射。模型条件 FCF 和原始观测随各自来源归档。

## 完全重叠双畴的盲解

`phase-twin` 适用于完全重叠、精确倒易度量、(T^2) 在 Friedel 等价下为恒等的两畴。给定完整晶胞组成和候选律，直接拟合带符号的双畴混合观测，缺失伴随反射以零观测权重处理。相位迭代交替使用双畴混合的强度近端步与电子密度更新。较接近等比例时使用工作反射估计初始化，Patterson 与随机起点可以混合。GPU 初筛后晋级及审查起点使用 CPU 双精度复算。

原生请求示意中的 `screening` 可写 `{"assume_friedel": true, "backend": "metal", "coarse_iterations": 55, "promote": 3, "inspect_rejections": 1}`。CUDA 对应 `backend: "cuda"`。新解 DAG 里的 `direct_twin` 包含 `law` 和这些参数。观察需支持 Friedel 等价。含可测异常差的结构使用保留成对反射的精修分支。

[打包的解析 skill](macos/source/skills/solve-single-crystal/SKILL.md) · [本轮配对验证](reports/twin-resolution/validation-summary.json)
