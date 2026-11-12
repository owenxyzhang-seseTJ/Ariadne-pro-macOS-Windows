# Ariadne 0.8.0 教材与验证材料

本目录的主教材从波的叠加、傅里叶变换和观测噪声出发，推导数据还原、混合相位求解、结构精修与原生 agent 的实现。独立的 loop 报告集中讨论结构信息怎样反馈到不确定观测，以及如何保留独立验证。

## 阅读与编译

- `Ariadne_0_8_0_Textbook.pdf`：完整教材。
- `Ariadne_0_8_0_Loop.pdf`：结构反馈专题。
- 两份同名 `.tex` 是主入口；`chapters/`、`preamble.tex`、`references.tex` 是正文与参考文献。
- `figures/` 同时提供矢量 PDF、预览 PNG 和来源记录。
- `fonts/` 提供经过子集化并重命名的 Noto SC 开放字体、SIL OFL 许可与来源记录。

安装 XeLaTeX 后，在源码根目录执行：

```bash
python docs/v08/build_report.py
```

编译使用已包含的图和字体，不下载外部资源。脚本对每份报告运行三遍以更新目录和交叉引用，并检查缺字、未定义引用及超宽文本。

## 重绘科学图

先安装项目两个 Python 包和绘图库，再运行：

```bash
python docs/v08/tools/make_figures.py
```

图来自解析公式或随包保存的验证 JSON。合成数据、已知模型的实验强度再精修和性能测量有各自的标签。重新运行性能测量会因硬件、库版本、线程设置和系统负载而改变计时。

字体子集已满足本报告。扩写中文内容后，可用四份原始 Noto Serif/Sans SC Regular/Bold TTF 重新生成：

```bash
python docs/v08/tools/subset_report_fonts.py --source-dir /absolute/path/to/noto_fonts
```

## 验证与使用

`results/` 保存本次真实执行的测试日志、JUnit XML 和命令行结果。主要科学实验脚本及完整数值记录位于 `../../benchmarks/` 和 `../../native/validation/`；使用指南见本目录的 `Ariadne_0_8_0_Guide.zh-CN.md`。

发行根目录的 `DELIVERY.json` 与 `RELEASE_MANIFEST.json` 共同确定本次交付内容。旧版报告和旧版验证记录保留在各自历史目录中；不能把它们的测试计数或运行时间当成本次新增内核的结果。
