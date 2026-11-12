# Ariadne 0.8：从像素似然直接连接结构模型

## 本次实现的位置与边界

新模块是 `python/src/ariadne/reduction_native.py`。它只依赖 NumPy 和 SciPy，不需要在导入时加载 DIALS。
旧版 DIALS 补丁、反射预测、仪器几何、轮廓学习、Poisson–Gaussian 卷积、缩放和 HKLF 导出代码均予以保留。
本模块接收**已校准像素、固定轮廓、背景基函数和掩膜**；它不自行读取所有型号探测器，也不把不存在的标定信息补成默认值。

旧版已有流式 QR、可观测模态与非负计数拟合。本次并不把 QR 或 Poisson 似然称为新的数学发明。
新增的实质是：

1. 把图像还原的输出定义为可被结构因子直接预测的**似然对象**，而不是强求每个重叠反射都先得到独立强度。
2. 将 Poisson 信号与背景联合估计，并在剖面区间和结构模型评分时重新优化背景。
3. 提供统一的标定／可辨识性路由。正、负或低 I/sigma 不参与删点；计数模型和可观测维数决定计算方式。
4. 将这些对象纳入安装后的 Python 包，不依赖从技能目录临时修改导入路径。

## 1. 从图像线性模型开始

一个重叠区域的固定轮廓模型是

\[
 y=P I+B\beta+\epsilon.
\]

第 j 列 P 是第 j 个反射在各像素上的曝光分数；I_j 是完整轮廓对应的强度。
被探测器边界截断的轮廓可以满足 sum(P[:,j])<1。程序不重新把这列归一化到 1，因此不会把可见部分的计数误称为完整反射的强度。
B 的列描述背景，beta 是其系数。几何、轮廓和探测器响应在本次区域求解中固定。

对于重叠的两个孪晶贡献，若 P_1=P_2=p，则

\[
 P I=p(I_1+I_2).
\]

图像只能确定和，不能确定差。这是映射的零空间问题，不是优化器收敛速度的问题。
任何通过正则化或结构先验得到的拆分都必须带有模型依赖的含义，不能重新标记成两个独立的实验观测。

## 2. Gaussian 分支：消去背景，保留完整二次似然

若误差协方差为固定对角阵 V，令 W=V^{-1}，并白化数据：

\[
 A=W^{1/2}P,\quad C=W^{1/2}B,\quad d=W^{1/2}y.
\]

对背景最小化后，

\[
 \chi^2(I)=\min_\beta\|d-AI-C\beta\|^2
          =\|(1-Q_BQ_B^T)(d-AI)\|^2,
\]

其中 Q_B 是 C 的列空间正交基。先对增广矩阵 [C,A,d] 做流式 QR，可以在每个像素块完成后只保存一个小的上三角因子。
正交变换保持所有残差范数，因此随后的背景投影只需作用在小矩阵上，整个过程中没有 N×N 的 Q。

再将背景投影后的 A 写成薄 SVD：A_perp=U S V^T。保留浮点数意义下的非零奇异值，定义

\[
 R=S V^T,\qquad z=U^T d_\perp,\qquad
 c=\|d_\perp-Uz\|^2.
\]

则对**任意**候选强度 I，

\[
 \boxed{\chi^2(I)=\|RI-z\|^2+c.}
\]

实现中对设计列作数值平衡，最后将 R 变回原始物理强度单位。输出的 nullspace 也是物理 I 坐标中的正交零空间基。
这个变换没有添加先验，没有要求 I 非负，也没有先做独立反射拆分。

### 为什么保留负强度

Gaussian 分支中，I 是有噪声的积分估计。其估计值为负，不代表真实光子数为负。
若 X~N(0,sigma²)，将负估计截到零会使均值变成 sigma/sqrt(2pi)>0。
因此负值必须留在相同的观测模型中。真实的结构强度 |F|² 非负，但拟合它时应面对原来的有符号测量及其不确定度。

### 可观测量的定义

线性组合 a^T I 只有在 a 与 nullspace 正交时才可由该区域独立测量。
`GaussianObservation.contrast(a)` 据此返回估计和方差；不可辨识的独立分量返回 null。
`intensities_internal` 只是最小欧氏范数代表元，服务于诊断和重现，不是宣布拆分成功。

### 背景不确定度

若联合设计满秩，令 C^+ 为背景的广义逆，D=C^+A，则

\[
 \widehat\beta=C^+d-D\widehat I,\quad
 \operatorname{Cov}(\widehat I,\widehat\beta)=-\operatorname{Cov}(\widehat I)D^T,
\]

\[
 \operatorname{Cov}(\widehat\beta)=C^+(C^+)^T+D\operatorname{Cov}(\widehat I)D^T.
\]

程序返回完整的 I/beta 联合协方差；秩缺失时不伪造有限的独立参数误差条。

## 3. Poisson 分支：背景也是待估参数

真正的独立光子计数满足

\[
 n_i\sim\mathrm{Poisson}(\mu_i),\qquad
 \mu_i=o_i+(PI)_i+(B\beta)_i.
\]

已知偏置 o 是已知的**期望计数**，并非减过背景的观测残差。
P、B、o、I 和 beta 均非负。常背景或非负 Bernstein 背景基函数可以保证预测均值非负。
程序直接拒绝负数或分数“计数”；带读出噪声的 ADU 继续走已有 Poisson–Gaussian 卷积后端。

优化目标采用一半 Poisson 偏差：

\[
 L(I,\beta)=\sum_i\left[\mu_i-n_i+n_i\log(n_i/\mu_i)\right].
\]

它与完整负对数概率仅差一个数据常数。在 n_i=0 时，该项严格等于 mu_i；在 n_i>0 且 mu_i=0 时为无穷大。
程序不会用任意小正数替代该支撑边界。对于 mu 接近 n 的情形，用 log1p 对应的级数形式避免相消。

由于

\[
 \nabla_I L=P^T(1-n/\mu),\qquad
 \nabla_\beta L=B^T(1-n/\mu),
\]

并且 L 对非负仿射均值是凸函数，联合估计不存在因该像素模型本身引入的局部极小值。
优化器仍可能因数值误差或预算未达收敛，因此结果同时记录非负约束 KKT 残差，而不是只相信求解器的 success 标记。

列积分用于改变求解变量的数值尺度，但不会改变前向模型：令 s_j=sum_i A_ij，x_j=s_j theta_j，使用 A_ij/s_j 求解。
这把不同支撑面积的信号和背景系数变成相近的“贡献总计数”尺度。像素目标及梯度按行分块计算，不形成完整的附加增广设计矩阵。
每次优化进一步用初始期望 Fisher 对角线调整参数单位，使一个优化步对应可辨识的局部误差尺度。
非负 KKT 残差采用 min(t,grad_t) 的稳定形式，不通过 t-max(0,t-grad_t) 相减计算；两式代数等价，后者却可能在大计数下把非零分数舍入为零。

### 结构直接评分的接口

当结构模型预测 I(theta) 时，调用

\[
 L_{\mathrm{prof}}(I)=\min_{\beta\ge0}L(I,\beta).
\]

在可微处，根据包络定理，

\[
 \nabla_\theta L_{\mathrm{prof}}
 =\left(\frac{\partial I}{\partial\theta}\right)^T
   P^T(1-n/\widehat\mu).
\]

`profile_background` 返回目标值、物理强度梯度、背景解及其收敛性。
背景活跃约束变化时，不把这个目标误称为处处具有相同 Hessian 的光滑问题。
一般 Poisson 图像不存在与 Gaussian R,z 相同的固定小维数充分统计量；因此该分支保留原像素，而不是偷偷改用 Gaussian 近似。

## 4. 区间必须重新优化背景

对可辨识的非负线性组合 t=a^T I，程序计算

\[
 L_p(t)=\min_{I,\beta\ge0:\,a^T I=t}L(I,\beta).
\]

区间边界由 2[L_p(t)-L_min]=chi²_{1,0.95} 解得。
这里被重优化的不仅有其他重叠信号，也有背景；这修正了“先估背景，再将它冻结”为精确值造成的条件误差条缩窄。
chi²_1 阈值是渐近标定，不能声称稀疏计数或边界情形具有精确 95% 覆盖率。需要精确覆盖率的应用可以在同一个原始计数模型上另做 Neyman 构造或参数模拟。

可核算的例子：n1=8、n2=3，mu=[I+b,b]。联合极大似然为 I=5,b=3。
给定 I 时，背景的正根可直接写成

\[
 \widehat b(I)=\frac{11-2I+\sqrt{(2I-11)^2+24I}}{4}.
\]

本次实际计算得到 95% 渐近 LR 区间 [0,12.41928]；将 b 错当成已知 3 时则为 [0.65587,11.89155]。
这不是性能宣传用的统计对照，而是一个有独立解析解的实现检验。

## 5. 强、弱、孪晶的分流原则

`route_region` 不接受观测 I/sigma 参数，因此不会因“测得弱”而删点或挑选有利噪声。
路由有两个相互独立的轴：

- **噪声模型**：固定方差 Gaussian、真实 Poisson 计数、或已有的校准 Poisson–Gaussian 卷积。
- **几何可辨识性**：可分离轮廓、病态重叠、或只能确定若干组合量。

来自独立预测或独立曝光的期望计数可用于规划计算预算，不能改变原始似然，也不能作为新的观测重复加权。
高信噪且分离良好的参考反射适合约束几何；弱区和重叠区仍通过同一个校准前向模型参与结构求解。
若多个看似分离的信号共享一个待估背景参数，它们在消去背景后会产生相关性，不能仅凭信号轮廓互不重叠就宣称两个区域统计独立。

## 6. 实际验证与资源测量

验证环境：Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0，x86_64。
运行命令：

```bash
python -m pytest tests_rebuild/test_reduction_v08.py -q
python benchmarks/v08_reduction.py
```

20 个测试全部通过，包含以下独立核对：

- 与直接白化像素最小二乘比较任意候选 I 的似然；与完整设计的逆信息矩阵比较协方差。
- 保留负强度、完全孪晶仅发布强度和、近重叠的条件信息损失、背景列秩缺失、欠定图像及掩膜内非有限值。
- 非负 Poisson 联合解与单独写的 Nelder–Mead 目标比较；解析分数和背景剖面梯度与有限差分比较。
- 两像素背景剖面与上述解析正根比较；零光子上限与闭式解比较；完整重叠的总强度区间与一维 Poisson 区间比较。
- 检查截断轮廓的曝光分数、完整 Poisson logpmf、Gaussian/Poisson/Poisson–Gaussian 路由。
- 对 10^12 计数、10^7 已知偏置的闭式解作回归检查，防止大参数下非零分数被舍入后错误宣布收敛。

`benchmarks/v08_reduction_results.json` 保存可供绘图的精确数值。本次单次运行的记录为：

| 项目 | 结果 |
|---|---:|
| 100 个任意结构强度预测：压缩与原像素似然最大相对误差 | 1.7370e-15 |
| 完全重叠的两分量总强度 | 10.0000 |
| 完全重叠总强度的 95% 渐近 LR 区间 | [5.01077,17.53934] |
| 完全重叠的独立分量输出 | null；不发布不可辨识拆分 |
| 25 万像素、4 个信号、2 个背景列：流式额外峰值分配 | 515850 bytes |
| 相同输入一次性增广 QR：额外峰值分配 | 43818012 bytes |
| 上述额外峰值分配比 | 84.94 |
| 流式完整压缩耗时 | 0.05949 s |
| 一次性增广 QR 耗时 | 0.02149 s |

内存来自 tracemalloc 的实际跟踪峰值，不包含事先分配的输入数组，也不等同于总进程 RSS。
该对照的稠密路径仅做 QR，而流式路径还完成背景投影与 SVD 诊断。这个小例子中稠密路径更快，流式路径显著减少工作内存；不能据此宣称对全部数据集提速。
结果全部来自受控合成例子，不是实测晶体数据集的结构恢复率或 DIALS 全流程验证。

## 7. 核心公共接口

```python
from ariadne.reduction_native import compress_gaussian, PoissonObservation

obs = compress_gaussian(y, profiles, variance=variance, background=B)
# 原像素模型中消去背景后的目标：
score = obs.chi2(structural_intensities)
residual = obs.R @ structural_intensities - obs.z
# 需报告完整 pixel chi2 时再加 obs.residual_constant。

counts = PoissonObservation(n, profiles, background=B_nonnegative)
profile = counts.profile_background(structural_intensities, include_mean=True)
objective = profile['half_deviance']
gradient_wrt_intensity = profile['gradient']
assert profile['converged']

fitted = counts.fit()
interval_for_sum = fitted.profile_interval([1, 1])
```

Gaussian 的 R 可以含负数，表示正交信息模态；原来的物理轮廓 P 则仍表示像素中的贡献。
若用于孪晶结构精修，应将结构因子、域分数、曝光系数和尺度组合成原始有序的 I(theta)，再通过 R。
不要把 R 当作独立反射表，不要把内部的最小范数代表元当作事先完成了实验去孪晶。
