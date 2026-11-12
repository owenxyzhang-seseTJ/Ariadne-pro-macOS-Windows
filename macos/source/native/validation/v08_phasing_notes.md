# Ariadne 0.8 原生相位求解与观测证据驱动 agent：数学和实现说明

## 1. 相位问题及本次实现的位置

约定倒易基矢不含 $2\pi$，结构因子为

$$F_h=\sum_j o_j f_j(s_h)\exp(-2\pi^2 q_h^T U_jq_h)\exp(2\pi i h^Tx_j).$$

实空间电子密度对应 `fftn(F_grid)/V`。衍射实验主要约束 $|F_h|^2$，没有直接测到 $\arg F_h$。原子性、非负电子密度、空间群及化学组成可以减少相位自由度，但不能凭空增加独立观测数。

本次保留原有 `charge_flip` 和 `atomic_projection`，新增 `PhasingOptions(algorithm="hybrid")`。它在 P1 全晶胞密度上求相位，随后交给原点搜索、空间群轨道投影及原生精修。直接法和双空间法的核心关系是已有科学方法；本实现的具体改进在于：**将有符号强度似然、稀疏相位关系和原始重叠观测的结构精修串成一条可检查的数据路径**。

## 2. 从有符号强度到径向近端更新

一个已完成背景处理的强度估计可以为负。对具有固定标准不确定度的近似高斯观测，写作 $y\sim N(r^2,\sigma^2)$，其中物理幅值 $r\ge0$。设当前双空间迭代给出的复数系数为 $z=a\exp(i\phi)$。本算法在保持当前相位的条件下，解

$$r_* = \mathop{\arg\min}_{r\ge0}\left[\frac{(r-a)^2}{2\tau}+\frac{(r^2-y)^2}{2\sigma^2}\right].$$

令 $b=\sigma^2/(2\tau)$，非零驻点满足

$$r^3+(b-y)r-ba=0.$$

当 $a>0$ 时，这个三次方程只有一个正根：若 $b-y\ge0$，左边在非负半轴严格递增；若 $b-y<0$，它先下降后上升，而在零点左边已经为负，因此正半轴只有一次过零。该根是全局最小值。当 $a=0$ 时，直接比较边界和非零驻点，得到 $r_*=\sqrt{\max(y-b,0)}$。

`direct_methods.intensity_amplitude_prox` 使用缩放后的 Cardano / 双曲函数表达式，避免为每个反射启动一个优化器，也避免大 $b$ 下两个立方根相减产生消减误差。它用同一公式处理正、零、负强度；并没有先将负强度改写为零。

这里 $\tau$ 是迭代的近端尺度，不是额外的实验方差。对于很可靠的正强度，更新接近 $\sqrt y$；对于误差很大的弱强度，更新允许实空间约束继续决定幅值。这样的“软约束”比把每个噪声强度都压到一个精确球面上更符合原始误差模型。它仍然是非凸相位求解中的一个子步骤，不提供全局收敛保证。

## 3. 强度单位不应改变答案

HKL 文件的强度通常没有绝对电子单位。若同时作 $y\mapsto c y$、$\sigma\mapsto c\sigma$，正确的相位算法应满足 $F\mapsto\sqrt c F$，相位不变。

本实现先用组成散射包络 $e_h=\sqrt{\sum_j f_j(s_h)^2}$ 归一化，得到 $J_h=y_h/e_h^2$ 和 $s_h=\sigma_h/e_h^2$。包络有明确的 8% 下限，以限制高角区噪声放大。再令

$$\tau=\eta\left\langle\max(J_h,0)\right\rangle_{\rm work},$$

其中 $\eta$ 即 `data_relaxation`。正值在这里仅用于选择单位尺度；整个有符号 $J_h$ 仍进入似然。全负输入退回平均 $s_h$ 作为尺度。随机或 Patterson 种子只提供相位，初始未测幅值置零，随后作为潜变量更新。Patterson 向量和其反向向量使用同一个规范代表，避免改变强度单位时因并列峰的浮点排序不同而换了种子。

独立测试将强度和标准不确定度同时乘以 12345，检查候选坐标、标准化残差及强度标度的相应变换。

## 4. 直接法：用有限的三相位关系指导搜索

对近似独立的等价散射原子，Cochran 型关系使强反射的三相位不变量

$$\Phi_{h,k}=\phi_h-\phi_k-\phi_{h-k}$$

倾向于零，其集中程度随 $|E_hE_kE_{h-k}|$ 增大。切线关系因此写成

$$\phi_h\leftarrow\arg\sum_k w_{h,k}\exp\{i(\phi_k+\phi_{h-k})\}.$$

本实现选择至多 256 个强且可靠的 Friedel 配对反射，每个规范目标至多保留 32 个关系；不构造全体反射的平方规模配对表。权重为

$$w_{h,k}=|E_hE_kE_{h-k}|\,r_hr_kr_{h-k},\qquad
r_h=\frac{\max(J_h,0)}{\sqrt{J_h^2+s_h^2}}.$$

这个可靠性因子是搜索权重，不是严格的相位后验。它不改变数据似然，弱负反射也不会从后续精修中删除。圆周均值采用阻尼，并在前四分之一迭代内逐步退火到零。这样，近似三相位关系负责帮助搜索，而不会被当作最终必须精确满足的实验方程。

每条关系满足精确的整数向量恒等式 $h=k+(h-k)$，所以平移原点引入的线性相位完全相消。代码同时更新 Friedel 共轭对。相应测试直接检查原点协变性和共轭对称性。

## 5. Patterson 与实空间步骤

Patterson 图由强度作 Fourier 变换获得，它记录的是原子间矢量。对一个候选非零矢量 $v$，初始化使用正部分的最小超位置图

$$S_v(x)=\min[P_+(x),P_+(x-v)].$$

这个图本身具有一个初始化造成的反演中心。加入明确记录的小随机扰动，使搜索能够离开该偶然对称性。随机密度种子与 Patterson 种子交替使用；没有提供真实原子坐标。

实空间迭代在电荷翻转与原子支持投影之间交替。后者选择符合晶胞度量的分离峰，只保留峰附近 $3\times3\times3$ 网格内的正密度。元素首先按已给定的组成和峰高度赋值，之后由强度精修检验。这不等于完整的自动化学识别、无序建模或氢原子构建。

未测系数只在迭代内部作为潜变量。最终提峰用的诊断图再次置零未测系数；潜变量不会导出为“补测”的观测强度。

## 6. 内存和斜晶胞的 Fourier 网格

分辨率带限由 $q^2=h^TG^*h$ 计算。代码用三个一维轴广播各项，不先创建 $N_{\rm voxel}\times3$ 的 Miller 指数大数组。组成包络按元素种类累加，而不是存储“原子数乘反射数”的散射因子矩阵。

斜晶胞还有一个容易忽略的细节：偶数 FFT 网格的 Nyquist 面在模索引下是自己的反向面，球形度量截断可能在这些面破坏共轭配对。网格已严格大于观测指标的两倍，因此这些面不含观测。本实现显式排除它们，保证密度仍为实数；测试对照直接晶胞度量计算。

## 7. 从原子模型返回像素证据，而不是伪造拆分强度

对于高斯像素噪声、固定轮廓和线性背景，积分阶段的正交消元可以把一组像素表示为

$$\ell_g(\theta)=\tfrac12\|R_g\,\operatorname{diag}(c_g\alpha_{d_g})\,I_g(\theta)-z_g\|^2+C_g/2.$$

这里 $I_g(\theta)=k|F_{h_g}(\theta)|^2$，$c_g$ 是已知标定/曝光/部分度系数，$\alpha_d$ 是非相干同结构畴的体积分数。若两个反射完全重叠，$R_g$ 可以只有一个可观测行；此时不必也不应先给两个分量各造一个观测强度。

新增 `evidence_solver.solve_observations` 先从 **work 集中几何上孤立、属于 domain 0 的单分量组** 提取相位初始化数据。对于单列 $R$、正标定系数 $c$，

$$\widehat I=\frac{R^Tz}{R^TR}\frac1c,\qquad
\sigma_I=\frac1{\sqrt{R^TR}\,c}.$$

该估计保留符号；不使用 $I/\sigma$ 截断。共同的 $\alpha_0$ 只改变任意强度标度。随后，每个相位候选都用 **所有 work 观测组** 联合精修，包括弱组、部分重叠组和完全重叠组。结构约束把不同反射联系起来，但每个重叠分量的输出仍明确称为结构预测，不能充作独立测量值。

如果没有至少十个独立的孤立 work Friedel 对，自动流程要求显式 `initial_model`，再从该模型做联合精修；它不会假装已经解出了缺失的单畴相位。多畴必须显式声明 `phase_relation="same_phase_supported"`。不同化学相的晶体不能直接套用这个共同结构模型。

## 8. agent 的三个证据角色

work 生成和精修假设；selection 选择假设；audit 在选择及可选 work+selection 重拟合后只评价一次。分组依据 Miller 指数、候选空间群等价关系、Friedel 对和共享像素的连通关系，不依据反射强弱。同一像素组、同一等价反射不能跨角色。

本次还将连通分量按规范 Miller 键排序，使单纯重排行序不会重抽 audit。显式给定完整三角色时会保留；相互冲突的角色会报错。审计测试把 audit 的强度、误差或压缩因子大幅修改，验证候选、选择结果及拟合参数逐项不变。

像素之间的固定高斯协方差、背景基、几何和标定必须已正确包含在观测模型中。低计数原始 Poisson 数据不能只凭形式相似就套用这个高斯压缩接口，应使用保留原始计数的独立似然路径。

## 9. 可重复证据

运行：

```bash
python -m pytest native/tests/test_phasing_v08.py native/tests/test_autonomy_v08.py native/tests/test_evidence_solver_v08.py -q
python native/validation/run_v08_phasing_autonomy.py
```

在 Python 3.12 / NumPy 2.3.5 / SciPy 1.17.0 环境，三组新增测试共 **25 项通过**。`native/validation/v08_phasing/result.json` 是本次完整数值记录；其中所有“盲解”只给求解器晶胞、观测及元素组成，真值仅用于模拟和事后误差评价。

实验包括：四碳 P1 模型的三个噪声档；斜晶胞 O2N2C4 的八原子混合元素模型；默认 15% selection / 10% audit 的原生自动求解；2497 个观测组、包括部分及完全重叠的同结构双畴自动求解。最后一例相位初始化只用 1121 个 work 孤立反射，其中 95 个强度为负；随后 1861 个 work 观测组共同约束原子模型，381 个 selection 组用于选择，255 个 audit 组最终评价。精确误差、域分数和墙钟时间以 JSON 中本次运行值为准。

这些是受控的算法恢复实验，能够检查公式、符号、数据流和已实现的能力；它们不是实际 MOF、大无序、多相或任意孪晶数据的成功率估计。本实现提供原生替代工作的实质内核，并不宣称已覆盖 SHELXT/SHELXL 全部指令和经验功能。

## 10. 主要文献

1. Weeks, C. M. et al. Structure solution by minimal-function phase refinement and Fourier filtering. II. Implementation and applications. *Acta Cryst. A* **50**, 210–220 (1994). DOI: **10.1107/S0108767393008992**。原作者公开版本：https://cse.buffalo.edu/faculty/miller/Papers/ActaCryst94A50II.pdf
2. Chang, C.-S. et al. Incorporating Tangent Refinement in the Shake-and-Bake Formalism. *Acta Cryst. A* **53**, 436–444 (1997). DOI: **10.1107/S0108767397003279**。原作者公开版本：https://cse.buffalo.edu/faculty/miller/Papers/ActaCryst97A53.pdf
3. Oszlányi, G. & Sütő, A. The charge flipping algorithm. *Acta Cryst. A* **64**, 123–134 (2008). DOI: **10.1107/S0108767307046028**。
4. Sheldrick, G. M. SHELXT – Integrated space-group and crystal-structure determination. *Acta Cryst. A* **71**, 3–8 (2015). DOI: **10.1107/S2053273314026370**。

代码中一维近端、单位协变性、原点协变性及压缩观测的联合结构目标，均在上文直接推导。文献用于说明继承的直接法/双空间/Patterson 思想，不用于把这些成熟方法宣称成本项目首次提出。
