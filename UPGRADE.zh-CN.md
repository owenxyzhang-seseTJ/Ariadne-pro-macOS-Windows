# Ariadne-pro 计算与调度升级

基线为 `eba7bbac0000af3a2b671cd69b0f29067d8a38bf`。`macos/source` 是维护入口，Windows 的 Python 与 C++ 源码用 `tools/sync_platform_sources.py` 同步。根级 CI 分别在 macOS arm64 和 Windows x64 从源码构建数值扩展，并运行测量模型与任务生命周期回归。

## 精修速度与内存

- 普通精修默认使用稠密 Jacobian。C++ 在多个独立反射块间并行，同一反射内保留补偿累加和求和顺序。独立大任务最多自动使用 4 个 worker。小任务走单 worker。用 `ARIADNE_CPU_THREADS=1` 固定单 worker，或在会话节点的 `resources.cpu_threads` 中显式申请线程。程序把实际 worker 数写进结果。
- 联合精修在同一参数点复用组装后的残差和 Jacobian。末端按批次计算反射强度。数据行与增广行相同时复用秩分解。软约束、特殊位置、异常散射、孪晶分数和带符号强度的原有物理模型仍用于对照。
- 内存紧张时可在普通精修的选项 JSON 中指定 `"jacobian_layout": "operator"` 和 `"reflection_block_size": 1024`。SciPy 使用分块导数乘积，末端通过 QR/SVD 得到秩与零空间。反复计算导数块会增加时间。伴随恒等式、收敛目标及矩阵秩有独立回归。
- 重叠区准备器按字节和条目共同限制帧及模板缓存，通过 `--frame-cache-mib` 和 `--template-cache-mib` 调整。输出清单记录命中、解码、逐出和峰值缓存字节。
- 结构辅助反馈按观测块做 Cholesky 白化，再用 Householder QR 消去独立先验。连通组的噪声无需展开成观测数平方大小的矩阵。后验协方差因子、观测秩、不可辨识方向和预测似然沿用原接口。
- 新会话在复制输入时同步生成内容标识，并核对复制前后的源文件状态。执行与恢复阶段按原有内容标识检查快照，旋转序列按照原相对路径组织。

`tools/benchmark_upgrade.py` 支持结构因子、联合精修、Jacobian 布局和结构辅助反馈对照。用 `--structure`、`--hkl` 和 `--reflections` 可对实验反射表做同机交错基准。反馈对照用 `feedback --reflections 900 --repeat 7 --output RESULT.json`。输入结构与观测表的路径只在本地传入。

## MCP 长任务

在已有会话上用 `agent_submit(session, request_key, max_steps)` 提交，随后用 `agent_run_status` 和 `agent_events(after, limit)` 查询，可用 `agent_cancel_run` 请求取消。重复提交同一键和参数会得到原运行 ID。取消与完成均记入会话，达到运行终态与科学接受分开表达。

Windows 的 stdio 客户端可拥有服务器整棵进程树。需要客户端断连后继续推进时，从单独终端启动回环 HTTP 服务，客户端连接 `http://127.0.0.1:8766/mcp`。Windows 用 Job Object 管理实际动作及后代，macOS 用进程组。新旧入口均支持工作目录与日志留档。

```powershell
.\ariadne-pro-win.cmd mcp --transport streamable-http --port 8766
```

```bash
python -m ariadne.server --transport streamable-http --port 8766
```

会话请求可加 `"cache": {"directory": "绝对路径", "max_mib": 64}`。相同 CIF/HKL 内容、设置和实现版本共用项目缓存。当前复用范围是 `crystal.audit` 和 `crystal.predict`。阶段结果和缓存来源同时附在本次运行中。旧会话记录可继续读入。新会话使用带算法标识的内容标记。

### 同机项目共享配额

在任一持久目录保存 JSON 配额文件。下面的配置为共享执行器预留八个 CPU slot 和 8 GiB 的任务内存。

```json
{"cpu_threads": 8, "memory_mb": 8192}
```

在启动 MCP 服务、Agent CLI 和本地工作台之前，将 `ARIADNE_HOST_BUDGET` 指向同一份文件的绝对路径。macOS 使用 `export ARIADNE_HOST_BUDGET=/absolute/path/host.json`，PowerShell 使用 `$env:ARIADNE_HOST_BUDGET = 'C:\path\host.json'`。设置 `"memory_mb": null` 可仅共享 CPU 配额。排队时从会话墙钟扣除时间，单动作计时从获准启动开始。

每个 DAG 节点的 `resources` 仍受会话自己的 `scheduler` 限制。调度器按提交顺序分配主机资源。崩溃后由操作系统锁确认所有者及 worker 的存活状态，再回收申请。事件 `host_queued` 和 `resources_admitted` 记录排队和启动，后者含 `host_wait_seconds`。内存字段表示预约额度，操作系统 RSS 独立计量。

DIALS 寻峰、指标化、积分和缩放将节点的 CPU 声明传给各自的进程参数，每个 DIALS 进程配置一个 BLAS 线程。普通还原会话的寻峰可从会话配额借用最多四个 CPU slot。用户可在自定义积分节点设 `resources.cpu_threads`，并在 `options` 中给出 `integration.block.size` 以固定区块划分。多扫描的 `indexing.joint_indexing` 可按实际晶体与扫描关系明确指定。

本地工作台按现有步骤申请同一份主机配额。客户端可在作业设置中提供 `"resources":{"cpu_threads":4,"memory_mb":3072}` 覆盖桌面预约值。

## 实验方法

- `ariadne_xtal.count_refinement.refine_count_observations` 在固定几何与峰形下直接用校准 Poisson 像素拟合结构，工作组背景用剖面似然消元，审计组使用独立给定的背景。CLI 为 `python -m ariadne_xtal count-refine MODEL.cif GROUPS.json --options OPTIONS.json`。每组提供按列对应的 `hkl`、`observation` 和可选的曝光 `coefficients`。观察对象包含 `data`、`profiles` 及可选的 `background`、`offset`、`valid`。
- 在命令中添加 `--profile-options PROFILE.json` 可试验联合修正峰位和峰宽。配置允许 `max_shift` 与 `max_width_ratio`，默认分别为 1 个输入坐标单位和 1.5。组内 `profile_model` 提供 `lower`、`upper` 像素边界及每项反射的 `centres`、`sigma`，支持一至三轴。相同 `profile_group` 共用各轴峰位偏移和总宽度倍率。`audit` 组沿用工作组的偏移、宽度和结构参数。此路径按有限差分和合成留出数据验收。
- `ariadne.lattice_selection.select_lattices` 用工作扫描拟合既有晶格候选的质心混合模型，在保留扫描块比较新增晶格的预测增益。 `tools/validate_observed_lattices.py` 对 DIALS 数据按物理扫描分别运行，从工作块重估各候选晶格。不同扫描中的取向必须按仪器坐标或明确的基变换对应。输出权重属于质心模型，畴体积分数需由强度模型估计。
- `ariadne_xtal.screened_phasing.screened_phase_search` 在低分辨率上筛选可复现的起点，在完整工作反射上进行 CPU 双精度重评。可以保留随机晋级名额，并抽查被淘汰的起点。真实盲解成功率与 GPU 数据仍需专门对照。

对照原有 Python/C++ 内核、独立像素似然和解析特例，运行 `python tools/validate_upgrade.py`。使用当前安装环境时可通过 `--dependency-path` 提供依赖目录。原始工作数据保持在各自的数据目录，基准和回归产物另行写入。
