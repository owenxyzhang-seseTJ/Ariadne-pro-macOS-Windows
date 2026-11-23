---
name: solve-single-crystal
description: Solve and refine small-molecule, coordination-complex and MOF single-crystal structures in Ariadne Pro using corrected HKL, indexed overlap groups or a raw-data handoff. Use for alternate space groups, missed symmetry, pseudo-symmetry, merohedry, inversion and non-merohedral twinning, ligand and disorder modelling, and crystallographic validation.
---

# Ariadne Pro 单晶解析

使用简体中文沟通。先读取 [输入与实验条件](references/data-gates.md)。对孪晶和空间群比较读取 [孪晶与对称判据](references/twinning.md)。对配体、无序、溶剂和限制条件读取 [精修规则](references/refinement-playbook.md)。交付采用 [验证和归档](references/validation-delivery.md)。

默认用 AriadneXL（`ariadne-xtal`）求解和精修。DIALS 还原后交接反射和晶胞。`mode: reduce` 在 `settings.elements` 提供整个晶胞的元素清单后，可接续原生求解。依赖完整 SHELX 指令、外部档案或未受原生引擎支持的观测格式时，按配对输入选择对应外部程序。

## 选择运行入口与观测

1. 调用 Ariadne Pro MCP 的 `agent_capabilities`，确认 `resolve.prepare`、`resolve.screen`、`resolve.select` 可用。通过 `agent_init(request, session)`、`agent_submit(session, request_key)`、`agent_run_status`、`agent_events` 运行。一次提交使用稳定键，断连后读取同一运行。CLI 可用 `python -m ariadne.agent.cli init REQUEST SESSION` 和 `python -m ariadne.agent.cli run SESSION`。
2. 保存原始输入、仪器记录、基底、原点及 Friedel 合并策略。`scripts/inventory_crystal_input.py` 用于输入清点。旋转帧、校准和取向由同包的 [ariadne-reduction](../ariadne-reduction/SKILL.md) 处理。只有坐标时做几何和 ADDSYM 诊断。
3. 校正 HKLF 4 进入 `mode: resolve`，提供同源初模，或已知晶胞及整晶胞元素清单，或 SHELXT 的 INS。JSON 强度保留显式自由集。CIF 格式结构因子文件按内容和字段映射转换。检查 INS/RES 的 `HKLF` 声明，加权负强度及其不确定度保留。
4. 真正的 HKLF 5 与配对 RES/INS 可直接用安装的 SHELXL 精修。由像素生成的重叠观测优先用 `solve-observations` 或 `joint-refine`。只有记录真实观测组、贡献者、同相证据与校正谱系的数据才能生成 HKLF 5。SHELXT 的初解输入来自支持独立解相位的强度。
5. 元素组成、化学计量、溶剂和电荷以实验与化学上下文为准。只有在配体拓扑不明确时索取用户文件。RDKit 构象供连接和限制条件参考。

## 执行候选比较

- 同一颗样品可含多个晶畴。即使只指标化出一套点阵，仍可用带符号强度检验完全重叠的孪晶。对有初模的 HKLF 4，用 `resolve` 在同一基底上比较单畴、孪晶和较高对称的候选。必要时在 `spacegroup_candidates` 给出共同设置中的子群及超群。候选保留配对文件及变换来源。
- 对 `solver: shelxt` 每份经确认的 Laue 群控制文件只跑一次。主控制文件使用 `ins`，其他独立 Laue 群用 `laue_*`。原始观测和候选变换需要可追溯对应。SHELXT 的各组模型先过共同基底检查。
- 检查工作集消光覆盖、奇偶类强度、候选短精修、选择集成组配对分数，以及冻结候选的独立审计状态。无判别力时保留并列解释。完全重叠且畴分数不可辨时保留可测的强度组合。定向像素回读从 `build_overlaps` 的 `target-frames` 预算开始。经校正的同相观测可在 `resolve` 中追加 `pixel_export_hklf5`，并检查导出映射。
- 畴重叠接近完全且强度差很弱时，先从独立反射初始化。实验性的 `phase-twin` 支持同一晶胞、无需拆分强度的精确双畴模型。记录工作集、矩阵、Friedel 假设和预算，CPU 复算 GPU 晋级和抽查结果。
- 如新分支涉及积分、峰形、波长或吸收，回到原始证据来源再选择影响范围。确有坐标或 Fo/Fc 疑问时调用有界的 ADDSYM 或 TwinRotMat。为 SHELX 附档保留原始文件与独立运行副本。

## 精修与交付

先核对可靠骨架、有序轻原子和配位。逐步精修尺度、坐标、适用 ADP、占有率和 H 原子，使用有化学依据的 `PART`、占有率约束与几何或 ADP 限制。溶剂处理和异常反射有原始证据与独立记录。保留合理的替代模型、孪晶律、BASF 及不确定度。

检查收敛、ADP、占有率、差值密度、键合与 checkCIF。原始 HKL、配对 RES/INS、CIF、FCF、LST、校正与掩膜记录和科学判定随本地项目保存。使用 `scripts/validate_crystal_package.py` 检查正式交付。需脱敏发布时采用匿名案例和汇总量。
