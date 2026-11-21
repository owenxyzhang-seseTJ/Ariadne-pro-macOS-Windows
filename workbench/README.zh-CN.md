# Ariadne Pro 工作台

在本机浏览器导入衍射帧、反射表或结构模型，运行计算并检查倒易空间、峰形和电子密度。自动还原、原生解析、SHELX 和高级分析使用与 MCP 相同的 DAG 调度与事件存储。

## 启动

在已安装 Ariadne Pro 科学依赖的 Python 3.12 环境下执行。需要 DIALS 时，从安装了 DIALS 的环境启动。

```sh
cd workbench
python -m pip install -r requirements.txt
npm ci
npm run build
python launch.py
```

默认打开回环地址。启动器会选择仓库内与操作系统对应的源码，也可用 `ARIADNE_WORKBENCH_SOURCE` 指向另一份包含 `python/src` 和 `native/src` 的源码。用 `ARIADNE_PROJECTS_DIR` 指向已有项目目录即可读取已有项目。

## 运行记录

新任务在项目的 `jobs/<job-id>/session/` 保存 SQLite 状态、输入项目快照、事件和独立的节点执行目录。自动流程的后续节点引用已提交的前序结果。开启逐步确认时，每次确认单独提交当前步骤，项目继续接收已完成结果和人工选点。

关闭浏览器或工作台进程后，已提交的任务由独立运行进程推进。取消会请求调度器回收任务下的计算进程。点击“继续未完成步骤”会生成新的运行记录，已提交的节点接着供下游使用。同一节点的多次执行存放在不同目录。旧式任务保留历史和文件，重新运行时使用当前 DAG。

以相同的 `ARIADNE_HOST_BUDGET` 配置路径启动工作台和 MCP，两个入口即可共享 CPU 和内存预约。HTTP 接口 `GET /api/projects/{project_id}/jobs/{job_id}/events?after=0&limit=100` 返回事件与下一页游标。

## 批量相位筛选

在“多起点相位求解”的专业参数输入下列设置。探测器归约和默认相位求解沿用原有入口。

```json
{
  "options": {"trials": 12, "iterations": 180},
  "screening": {"coarse_backend": "metal", "coarse_iterations": 55, "promoted": 3}
}
```

`metal` 使用环境中安装的 MLX，`cuda` 使用 CuPy，`batched_cpu` 使用 NumPy。GPU 后端直接报告设备或依赖错误。粗筛输出交给 CPU 双精度复算，任务结果记录候选和筛选诊断。

首页参考模型来自 RASPA 的公共结构示例，来源和文献见模型目录。用户项目和构建缓存由本地环境管理。
