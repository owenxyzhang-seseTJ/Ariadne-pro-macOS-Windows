"""由已完成的实测记录生成本机优化报告；不重新运行基准。"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent
paired = json.loads((root / "paired-comparison.json").read_text())
before = json.loads((root / "before/measurements.json").read_text())
after = json.loads((root / "after/measurements.json").read_text())
assert all(before["acceptance"].values()) and all(after["acceptance"].values())
rows = []
for case in paired["predictor"]:
    for backend, result in case["backends"].items():
        timings = result["timings"]
        rows.append(f'| {case["n_atoms"]} 原子 / {case["n_domains"]} 畴 / {case["n_reflections"]} 反射 | {backend} | '
                    f'{timings["baseline"]["median_seconds"]*1000:.3f} | '
                    f'{timings["optimized"]["median_seconds"]*1000:.3f} | '
                    f'{result["time_reduction_percent"]:.1f}% | {result["speedup"]:.3f}× |')
refinement_rows = []
for backend, result in paired["refinement"].items():
    timings = result["timings"]
    refinement_rows.append(f'| {backend} | {timings["baseline"]["median_seconds"]*1000:.3f} | '
                           f'{timings["optimized"]["median_seconds"]*1000:.3f} | '
                           f'{result["time_reduction_percent"]:.1f}% | {result["speedup"]:.3f}× |')
test_rows = []
for filename, scope in (("numeric-tests.xml", "C++ 内核、解析导数、普通精修及边界"),
                        ("joint-tests.xml", "重叠观测联合精修"),
                        ("python-reference-tests.xml", "强制 Python 后端的解析导数与精修")):
    suites = list(ET.parse(root / filename).getroot().iter("testsuite"))
    count = sum(int(s.attrib["tests"]) for s in suites)
    assert all(int(s.attrib[k]) == 0 for s in suites for k in ("errors", "failures", "skipped"))
    test_rows.append(f'| {scope} | {count} | 0 | 0 | [{filename}]({filename}) |')
memory_rows = []
for original, optimized in zip(before["memory"], after["memory"]):
    assert (original["backend"], original["block_size"]) == (optimized["backend"], optimized["block_size"])
    memory_rows.append(f'| {original["backend"]} | {original["block_size"]} | '
                       f'{original["peak_process_rss_bytes"]/2**20:.2f} | '
                       f'{optimized["peak_process_rss_bytes"]/2**20:.2f} |')
cpp = paired["refinement"]["cpp"]
assert all(row["maximum_absolute_difference"] == 0 for row in paired["direct_output_comparison"])
text = f'''# Ariadne pro 本机优化报告

日期：2026年11月09日。软件：Ariadne pro 0.8.0，本地构建 `0.8.0-source-overlay-macos-arm64-localopt1`。

本轮优化已安装并生效。交错对照中，默认 C++ 后端的结构因子及解析导数计算耗时下降 **12.6%～18.6%**；18 原子局部精修耗时下降 **{cpp["time_reduction_percent"]:.1f}%**，速度比 **{cpp["speedup"]:.3f}×**。三组基准模型的原版与优化版预测强度、Jacobian 逐元素一致。

## 实测速度

以下对比对象均为本轮修改前后的 **pro 0.8.0**，原版来自修改前保存的源码和 C++ 二进制。这里的数字不代表 pro 与 Ariadne 0.5.0 的版本比较。

### 结构因子及解析导数

单位：ms/次；每次包含全部反射和畴的强度与解析 Jacobian。表中使用最终交错对照记录。

| 规模 | 后端 | 优化前 | 优化后 | 耗时下降 | 速度比 |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

### 完整局部精修

相同的 18 原子、1098 反射、70 参数合成问题，相同初始模型、目标函数和收敛条件。计时包括一次 `refine()` 调用及其最终统计处理。

| 后端 | 优化前 / ms | 优化后 / ms | 耗时下降 | 速度比 |
|---|---:|---:|---:|---:|
{chr(10).join(refinement_rows)}

C++ 两版的所有精修均成功，最终 χ² 均为 1009.3952271514734，Jacobian 秩均为 70。Python 两版的 χ² 差约 5.9×10⁻¹²，秩均为 70。

速度比 = 优化前耗时 / 优化后耗时；耗时下降 = 1 − 优化后耗时 / 优化前耗时。

## 测量方法与适用范围

本机为 Apple M5、16 GiB 内存、macOS 26.5、arm64，接入电源并关闭低功耗模式。所有数值库线程固定为 1；C++ 内核仍为单线程。没有依靠增加线程获得本轮提升。

最终复测按“原版 → 优化版 → 优化版 → 原版”交错启动四个独立进程，核对每个进程实际加载的包路径。每个进程先预热；每个模型每后端进行 3 轮计时，每轮 8 组不同参数以避免命中结果缓存，后端顺序交替。每个版本、每个后端最终汇总 6 轮的每次均摊耗时中位数。精修每个进程每后端执行 3 次，每版每后端合计 6 次，报告中位数。

前面已经完成的顺序复测也保留：C++ 完整精修 33.969 → 28.767 ms，耗时下降约 15.3%。两次对照的绝对耗时有系统状态波动；本报告使用交错复测作为主要结论，不把耗时变化当作固定保证，也没有做统计显著性检验。

这些是合成数据上的局部精修和内核测量。真实原始衍射图像的 DIALS 全流程、大型 MOF 完整精修及解结构成功率尚未验证。

## 实现改动

1. `native/src/ariadne_xtal/derivatives.py`：用直接批量小矩阵乘法替代每个位点反复规划的 `einsum`，单个旋转时省去无意义的平均；特殊位置仍保留多旋转投影。
2. `native/src/ariadne_xtal/cpp/intensity_kernel.cpp`：每个块预先标记坐标与 ADP 导数是否精确为零，逐反射时省去不存在方向的投影。保留占有率为零时可能非零的占有率导数，也保留溢出报错路径。
3. `native/src/ariadne_xtal/refinement.py`：没有 restraint 时，复用同一 Jacobian 已计算的秩，省去第二次相同的 QR/SVD。
4. `native/tests/test_kernels_v08.py`：补充稀疏方向、混合方向、零占有率、全零方向及数值溢出的回归测试。

物理模型、统计目标、收敛阈值、公共接口和数据格式未改。补偿累加、缓冲区检查及严格浮点编译选项保留；未启用 fast-math。已重新编译并安装 C++ 扩展。

## 数值验证

| 验证范围 | 通过次数 | 失败 | 跳过 | 原始记录 |
|---|---:|---:|---:|---|
{chr(10).join(test_rows)}

合计 76 次测试执行全部通过，其中 13 次是强制 Python 后端重跑同一组测试，并非 76 个互不重复的用例。验证包括独立有限差分、独立 Gemmi 结构因子、特殊位置、孪晶、约束、补偿累加、缓冲区契约和联合观测链式导数。

优化前后完整基准的 6 项验收均通过：所有精修收敛、后端选择正确、Python/C++ 导数一致、预测强度一致、解析/有限差分结果一致，以及压缩似然误差符合原阈值。

交错复测另将三个模型的原版与优化版 C++ 预测强度和 Jacobian 做逐元素比较，六个数组的最大绝对差均为 **0**。这项一致性只针对列出的三个基准模型；更广的边界由上述回归测试覆盖。

## 内存记录

80 原子、8192 反射、318 参数；单位 MiB。各配置分别启动独立进程测一次峰值 RSS，包含解释器和依赖库。macOS 原始 RSS 单位按 bytes 处理。

| 后端 | block size | 优化前峰值 RSS | 优化后峰值 RSS |
|---|---:|---:|---:|
{chr(10).join(memory_rows)}

本轮改善主要在计算速度；这些单次峰值记录不支持额外的内存降低结论。返回的 dense Jacobian 规模保持不变。

## 使用、复现与回退

继续使用原有专用启动器 `/Users/xiaoyuzhang/CrystallographyPrograms/ariadne-pro-env` 或桌面的 `启动 Ariadne pro.command`；新启动的 pro 进程自动加载优化代码。已有长期运行的进程需重新启动后才会加载新代码。包版本仍为 0.8.0，运行清单已记录本地优化构建。

- [交错实测原始汇总](paired-comparison.json)，以及 `paired-1-baseline.json` 至 `paired-4-baseline.json` 四个单进程记录。
- [顺序复测原始数据：优化前](before/measurements.json)、[优化后](after/measurements.json)。
- [可审阅源码补丁](ariadne-pro-local-speedup.patch)：包含三个实现文件和一个测试文件。
- [交错复测驱动](paired_benchmark.py)、[报告生成脚本](write_optimization_report.py)。
- `baseline-source/ariadne_xtal/`：完整原版原生 Python 包及原版 `.so`，保留用于回退和对照；原有 Ariadne 0.5.0 未修改。

如需回退，停止正在运行的 pro 进程，把 `baseline-source/ariadne_xtal/` 中的 `derivatives.py`、`refinement.py`、`cpp/intensity_kernel.cpp` 和 `_intensity_cpp.cpython-312-darwin.so` 复制回 `../Ariadne-pro-0.8.0/native/src/ariadne_xtal/` 对应位置，并在运行清单中去除本地优化标记。此操作只涉及本轮实现，不需重新安装 DIALS 或基础环境。

交错基准可从当前工作目录复现：

```bash
/Users/xiaoyuzhang/CrystallographyPrograms/ariadne-pro-env python 'optimization-20261109/paired_benchmark.py' --source "$PWD/Ariadne-pro-0.8.0" --directory "$PWD/optimization-20261109"
```

该命令会更新本目录中同名的交错计时记录。数值对照所用临时数组已在比较完成后删除；编译中间产物将清理，已安装的优化 `.so`、原版备份及测量记录保留。
'''
target = root / "Ariadne pro 本机优化报告.md"
target.write_text(text)
print(target)
