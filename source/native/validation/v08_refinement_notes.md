# Ariadne 0.8 原生精修：数学、实现与验证

本说明对应 `derivatives.py`、`refinement.py` 和 `joint_refinement.py`。数学身份式由程序直接实现；性能数据取自随包的可复现测量文件。直接法和双空间求解另见相位求解模块。这里的结构恢复实验均为**已知模型附近的局部精修**，不能解释为从未知坐标完成了从头解析。

## 1. 先写清前向物理模型，再求导

晶胞矩阵记为 \(A\)，列向量为实空间基矢；不含 \(2\pi\) 的倒易向量为 \(q=A^{-T}h\)。对一个非相干晶域，结构因子为

\[
F_h=\sum_{j,g} o_j f_j(s)\exp[-2\pi^2 q^T U_{jg}q]
                         \exp[2\pi i h^T x_{jg}],\qquad I_h=k|F_h|^2.
\]

求和中的 \(g\) 遍历原子的不同对称等价位置，特殊位置只计算一次。\(U\) 使用笛卡尔坐标、单位 Å²；\(f=f_0+f'+if''\)，异常散射项保持复数形式。对任一实参数 \(p\)，

\[
\frac{\partial I_h}{\partial p}
 = k\,2\operatorname{Re}\!\left(F_h^*\frac{\partial F_h}{\partial p}\right)
   + |F_h|^2\frac{\partial k}{\partial p}.
\]

每个原子贡献的坐标导数来自 \(2\pi i h^T\partial x/\partial p\)，位移参数导数来自 \(-2\pi^2q^T(\partial U/\partial p)q\)。不再对每个参数分别重算整张反射表。散射因子、倒易几何和特殊位置的张量投影算子可在精修前缓存。

### 物理约束的精确链式法则

- 坐标写成 \(x=x_0+Tv\)。\(T\) 是特殊位置允许的切空间，并扣除连续原点自由度；坐标导数就是 \(T\)。
- 各向同性参数写成 \(U_{\rm iso}=e^v\)，因此始终为正。
- 一般位置的各向异性参数写成 \(U=LL^T\)，则 \(dU=(dL)L^T+L(dL)^T\)。
- 特殊位置写成 \(U=\exp S\)，其中 \(S\) 限于对称性允许的对称矩阵子空间。其导数使用矩阵指数的 Fréchet 导数，不能通过不稳定的特征向量差分获得。
- 占位率和晶域比例的和约束使用 softmax：\(\partial\alpha_i/\partial v_j=\alpha_i(\delta_{ij}-\alpha_j)\)，参考分量的 logit 固定为零。

若 \(S=Q\operatorname{diag}(s_i)Q^T\)，矩阵指数的导数在其特征基中满足

\[
D\exp(S)[E]=Q\{K\odot(Q^TEQ)\}Q^T,\quad
K_{ij}=\frac{e^{s_i}-e^{s_j}}{s_i-s_j},\quad K_{ii}=e^{s_i}.
\]

代码通过 \(e^{(s_i+s_j)/2}\operatorname{sinhc}((s_i-s_j)/2)\) 连续计算该商。因此，即使起始原子是各向同性、三个特征值相同，导数仍然定义良好。对称位置的多重度是离散模型选择；精修终点若改变多重度，程序要求重新声明特殊位置模型，避免散射贡献突然减半。

## 2. 复杂重叠不必先拆成单个强度

背景参数记为 \(\beta\)，像素轮廓矩阵为 \(P\)。已知高斯方差的像素模型为

\[
y=PI+B\beta+\epsilon,\qquad\epsilon\sim N(0,W^{-1}).
\]

先消去线性背景，再进行正交分解，可保留一个小型似然：

\[
\min_\beta\|W^{1/2}(y-PI-B\beta)\|^2
   =\|RI-z\|^2+c.
\]

当两个轮廓完全相同，\(R\) 只有一行有效方向，独立的 \(I_1,I_2\) 没有唯一估计。Ariadne 保留这个可观测方向，将结构直接带入：

\[
r_g(\theta,\alpha)=
 R_g\operatorname{diag}(c_g\alpha_{d_g})\,
 k|F_{h_g}(\theta)|^2-z_g.
\]

\(c_g\) 是已知校正系数，\(d_g\) 给出每个分量所属晶域。程序对所有独立像素区域联合优化 \(\sum_g\|r_g\|^2\)，同时保留常数项 \(c\) 用于比较完整像素似然。它不生成一组虚构的“已去孪晶测量强度”。多个区域共享同一个原子结构，因而局部不能分离的分量，有时可以由整套数据和同相结构模型共同约束。

这不是无条件恢复信息。如果两个晶域仅互为反演、没有异常散射且轮廓完全相同，则 \(|F_h|^2=|F_{-h}|^2\)，晶域比例的导数严格为零。实现会给出零空间，并将 `domain_fraction_estimates` 标为 `null`。`domain_fractions` 仅保留优化器的一个数值代表，不能当成测量。这个反例同时检验了机制的边界。

该接口要求多晶域属于同一个结构相；不同化学相不能仅凭输入一个 twin law 共用原子模型。轮廓、仪器几何、校正系数和高斯方差均作为条件量给定。真正的低计数泊松似然保留像素，不能普遍压缩为一个固定的 \(R,z\)。

## 3. 不构造观测数平方的矩阵

原精修器的完整 SVD 会构造 \(N\times N\) 左奇异向量，即使最终只使用右奇异向量。新实现逐块累积 QR：

\[
R_{t+1}=\operatorname{qr}_R\begin{bmatrix}R_t\\J_{t+1}\end{bmatrix},
\qquad R=U_R\Sigma V^T.
\]

左乘正交矩阵不会改变奇异值，因此这个 \(\Sigma,V\) 与完整雅可比矩阵的可辨识性分析相同；也没有形成 \(J^TJ\) 而平方条件数。无信息方向由 \(V\) 的零空间报告，协方差只在可辨识方向上有意义。

NumPy 参考路径把结构因子的复数临时数组限制为 \(B\times P\)，其中 \(B\) 为反射块大小、\(P\) 为参数数。后加入的可选 C++ 融合路径直接累积实数强度导数，详见第 6 节。SciPy 的稠密信赖域求解器仍然保存实数 \(N\times P\) 雅可比，不能宣称整个精修过程具有常数内存。几何限制条件只对小型限制条件系统作中心差分，不会因此再次计算衍射结构因子。

## 4. 实际验证结果

运行环境：Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0、Gemmi 0.7.5；基准限制 BLAS 为单线程。解析对差分的比较明确使用 `--backend python`，不混入 C++ 加速。结果含源码 SHA-256，详见 `v08_refinement_measurements.json` 与 `v08_joint_measurements.json`。

| 检查 | 实际结果 |
|---|---|
| 5 组独立前向差分检查 | 最大按列尺度归一的导数差为 \(9.02\times10^{-9}\) |
| 覆盖场景 | 倾斜晶胞与异常散射、非对称型空间群、特殊位置及重复 ADP 特征值、占位率和约束、反演及三晶域孪晶 |
| 18 原子、1,098 反射、70 参数的局部精修 | 解析导数中位时间 0.11729 s；同一缓存前向模型的数值差分中位时间 2.68172 s；3 次测量约 22.86 倍 |
| 衍射前向计算次数 | 7 对 988；两条路径保留相同的 18 个负强度 |
| 两条路径的最终预测差 | \(5.32\times10^{-10}\sigma\) RMS；目标函数相同至约 \(10^{-12}\) |
| 3,500×64 雅可比的独立进程峰值 RSS | 原完整 SVD：260,500 KiB；块 QR/SVD：81,444 KiB |
| 新旧聚焦回归检查 | 75 项通过；见下面的命令 |

上表的 Gemmi 独立散射计算器生成了局部精修基准的观测值。时间和进程内存是本机测量，不是所有结构的性能保证，也不是与 SHELXL 程序的速度比较。

重叠结构实验使用 3 个原子、171 个像素区域、7,011 个像素及 11 个自由参数。其数据由已独立验证的本征前向模型合成，初始坐标在真值附近扰动。

| 情形 | 区域可观测秩 | 全局结构秩 | 结果 |
|---|---|---|---|
| 57 个完全重叠、114 个部分重叠 | 1 或 2 | 11/11 | 无噪声恢复 \(\alpha=(0.73,0.27)\) 与坐标 |
| 171 个完全重叠区域 | 均为 1 | 11/11 | 在此局部合成例中仍恢复结构与晶域比例；没有逐点去孪晶 |
| 无异常散射的反演孪晶 | 均为 1 | 10/11 | 比例不可辨识；估计、标准不确定度均返回 `null` |
| 加入已知方差的高斯像素噪声 | 1 或 2 | 11/11 | \(\alpha_0=0.7301858\)，条件标准不确定度 0.0006302；最大坐标误差 0.003365 Å |

含噪声例中，压缩似然与直接对原像素消去背景后的目标函数之差为 \(-2.73\times10^{-12}\)。这说明压缩没有把重叠协方差或背景不确定度丢掉；它不证明一般孪晶都能被唯一求解。

## 5. 可复现命令与接口

```bash
OPENBLAS_NUM_THREADS=1 python -m pytest \
  tests/test_refinement.py tests/test_core.py \
  tests/test_refinement_v08.py tests/test_joint_v08.py -q
OPENBLAS_NUM_THREADS=1 python validation/run_v08_refinement.py --backend python
OPENBLAS_NUM_THREADS=1 python validation/run_v08_joint.py
```

命令从 `native/` 目录运行，并需要安装 Ariadne 与原生 Xtal 两个项目以及测试依赖。

```python
from ariadne_xtal.derivatives import DiffractionPredictor

predictor = DiffractionPredictor(parameterization, hkl,
                               twin_laws=nonidentity_laws, block_size=1024)
model, predicted_intensity, jacobian = predictor.evaluate(values, jacobian=True)
```

`parameterization` 是原生受约束参数化。返回的数组由缓存拥有，调用者不得原地修改。`physical_parameter_jacobian` 用于将协方差变换回坐标、ADP、占位率和比例。`right_singular_system` 可用于任何同类残差的秩与零空间分析。

完整像素到结构的接口是 `GaussianStructureGroup.from_observation` 与 `refine_observations`。导出文件区分 `domain_fraction_estimates`、`domain_fraction_status`、数值代表和结构预测。用于工作集的反射不能同时出现在选择集或审计集；等价反射和 Friedel 对被一起检查，避免验证数据泄漏。

## 6. 可选 C++ 融合内核

令不含占有率的原子位置贡献为 \(C_s=a_s+ib_s\)，结构因子为 \(F=u+iv\)。由 \(2\Re(F^*\partial_pF)\) 直接展开，可得

\[
\partial_p |F|^2 = \sum_s
2(ua_s+vb_s)(\partial_p o_s+o_s\gamma_{sp})
+2(va_s-ub_s)o_s\phi_{sp},
\]

其中 \(\gamma_{sp}=-2\pi^2q^T(\partial_p U_s)q\)、\(\phi_{sp}=2\pi h^T\partial_p x_s\)。内核只存当前一个反射的原子贡献，直接写实数导数，并对结构因子及导数使用 Neumaier 补偿求和。域比例和尺度的稀疏链直接更新输出块；两种后端都避免额外的整块广播临时数组。

`ARIADNE_NUMERIC_BACKEND=auto|python|cpp` 控制实现；`python -m ariadne_xtal.kernels` 查询真实后端。C++17 扩展使用受检 CPython buffer，单线程、释放 GIL、无 OpenMP，不依赖 NumPy C API 或复数 ABI。源码和编译脚本随包交付；纯 Python wheel 经过隔离导入验证。

冻结后的测量见 `v08_kernel_measurements.json`：两种缓存预测器场景的速度比分别为 3.660 和 3.075，整个 18 原子局部精修为 1.848；这些比较两边都是解析导数。8192 反射、318 参数时，默认块长 1024 的独立进程峰值 RSS 为 105736 对 95260 KiB；块长 8192 为 172024 对 113168 KiB。RSS 包含解释器、库、完整输出与临时量。核验记录为 `v08_kernel_tests.xml` 与 `v08_kernel_packaging.json`；完整数学推导、测量口径和图位于教材报告的 C++ 章节。
