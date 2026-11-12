"""Reproduce the report figures from equations and archived numerical runs.

Run from any directory using an environment with both Ariadne packages and
Matplotlib installed. No external data downloads or image synthesis occur.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import norm, chi2

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
FONT = HERE / "fonts" / "ReportSans-Regular.ttf"
if not FONT.is_file():
    raise FileNotFoundError("The report bundle must include fonts/ReportSans-Regular.ttf")
font_manager.fontManager.addfont(str(FONT))
FAMILY = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.family": FAMILY, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False, "pdf.fonttype": 42, "savefig.dpi": 180,
    "legend.frameon": False, "lines.linewidth": 1.7})
BLUE, ORANGE, GREEN, GRAY = "#26495C", "#B76936", "#48755A", "#929DA6"
provenance = {}


def save(fig, name, source):
    fig.savefig(OUT/f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    provenance[name] = source


def phase_problem():
    n = 256
    x = np.arange(n)/n
    rho = sum(w*np.exp(-0.5*((x-u+.5)%1-.5)**2/.018**2)
              for w, u in [(1., .13), (.8, .38), (1.2, .72)])
    f = np.fft.rfft(rho)
    rng = np.random.default_rng(8108)
    phase = rng.uniform(-np.pi, np.pi, len(f))
    alternative = abs(f)*np.exp(1j*phase)
    alternative[0] = f[0]
    alternative[-1] = f[-1]
    other = np.fft.irfft(alternative, n=n)
    fig, ax = plt.subplots(2, 2, figsize=(9, 5), layout="constrained")
    for col, (density, title, color) in enumerate([(rho,"原子形状的密度",BLUE),(other,"改变相位后的密度",ORANGE)]):
        ax[0,col].plot(x, density, color=color)
        ax[0,col].set(title=title, xlabel="晶胞分数坐标", ylabel="相对电子密度")
        ax[1,col].vlines(np.arange(31), 0, abs(f[:31]), color=color)
        ax[1,col].set(title="相同的傅里叶模", xlabel="倒易指数 h", ylabel="|F(h)|")
    save(fig,"01_phase_problem",{"kind":"analytic teaching example", "seed":8108})


def weak_bias():
    fig, ax = plt.subplots(1,2,figsize=(9,3.2),layout="constrained")
    x=np.linspace(-3.8,3.8,500)
    ax[0].plot(x,norm.pdf(x),color=BLUE,label="原始带符号观测")
    ax[0].fill_between(x,0,norm.pdf(x),where=x<0,color=BLUE,alpha=.12)
    ax[0].plot(x[x>=0],norm.pdf(x[x>=0]),color=ORANGE,ls="--",label="裁零后的连续部分")
    ax[0].annotate("0 处另有 50% 点质量",xy=(0,.02),xytext=(.9,.27),arrowprops={"arrowstyle":"->","color":ORANGE},fontsize=9)
    ax[0].set(xlabel="观测 / 标准差",ylabel="概率密度",title="真实强度为零")
    ax[0].legend(loc="upper left",fontsize=8)
    snr=np.linspace(0,4,200)
    bias=norm.pdf(snr)-snr*norm.cdf(-snr)
    ax[1].plot(snr,bias,color=ORANGE)
    ax[1].axhline(0,color=GRAY,lw=.8)
    ax[1].set(xlabel="真实强度 / 标准差",ylabel="裁零偏差 / 标准差",title="裁零会抬高弱信号均值")
    save(fig,"03_weak_bias",{"kind":"analytic Gaussian expectation"})


def overlap():
    from ariadne.reduction_native import compress_gaussian
    x=np.arange(121.)-60
    fig,ax=plt.subplots(1,2,figsize=(9,3.4),layout="constrained")
    for mean,color,label in [(0,BLUE,"分量 1"),(2,ORANGE,"近重叠分量 2"),(12,GREEN,"较远分量 2")]:
        p=np.exp(-.5*((x-mean)/4)**2)/(4*np.sqrt(2*np.pi))
        ax[0].plot(x,p,color=color,label=label)
    ax[0].set(xlim=(-16,26),xlabel="像素相对位置",ylabel="每像素轮廓质量",title="同一宽度，不同分离度")
    ax[0].legend(fontsize=9)
    sep=np.r_[0,np.geomspace(.01,20,75)]
    total,difference=[],[]
    for shift in sep:
        p=np.column_stack([np.exp(-.5*((x+s)/4)**2)/(4*np.sqrt(2*np.pi)) for s in [shift/2,-shift/2]])
        o=compress_gaussian(np.zeros(len(x)),p,variance=4.,background=np.ones((len(x),1)))
        total.append(np.linalg.norm(o.R @ np.array([1.,1.])/np.sqrt(2))**2)
        difference.append(np.linalg.norm(o.R @ np.array([1.,-1.])/np.sqrt(2))**2)
    total,difference=np.array(total),np.array(difference)
    ax[1].plot(sep/4,total/total[0],color=BLUE,label="总量方向")
    ax[1].plot(sep/4,difference/total[0],color=ORANGE,label="差量方向")
    ax[1].set(xlabel="两峰分离度 / 轮廓标准差",ylabel="相对信息",title="背景消元后的信息方向")
    ax[1].legend(fontsize=9)
    save(fig,"04_overlap_information",{"kind":"Gaussian profile information computed by compress_gaussian", "pixels":121})


def poisson_profile():
    from ariadne.reduction_native import PoissonObservation
    o=PoissonObservation([8.,3.],[[1.],[0.]],background=[[1.],[1.]])
    fit=o.fit()
    x=np.linspace(0,18,181)
    prof=np.array([2*o.profile_background([t])["half_deviance"] for t in x])
    fixed=np.array([2*o.value_gradient(np.array([t,3.]))[0] for t in x])
    fig,ax=plt.subplots(figsize=(7.6,3.5),layout="constrained")
    ax.plot(x,prof,color=BLUE,label="每个 I 下重新拟合背景")
    ax.plot(x,fixed,color=ORANGE,ls="--",label="固定背景 b=3")
    ax.axhline(chi2.ppf(.95,1),color=GRAY,ls=":",label="95% 渐近 LR 阈值")
    ax.set(xlabel="信号强度 I",ylabel="相对最优解的偏差",ylim=(0,9),title="观测 (8, 3)，模型 (I+b, b)")
    ax.legend(fontsize=9,loc="upper left")
    save(fig,"05_poisson_profile",{"kind":"exact two-pixel Poisson likelihood; asymptotic interval threshold", "data":[8,3]})


def proximal():
    from ariadne_xtal.direct_methods import intensity_amplitude_prox
    a=np.linspace(0,10,301)
    fig,ax=plt.subplots(1,2,figsize=(9,3.4),layout="constrained")
    for y,s,label,color in [(25.,1.,"可靠正强度 y=25, σ=1",BLUE),(25.,25.,"弱约束 y=25, σ=25",ORANGE),(-4.,3.,"负观测 y=-4, σ=3",GREEN)]:
        ax[0].plot(a,intensity_amplitude_prox(a,y,s,variance=8.),color=color,label=label)
        ax[0].axhline(np.sqrt(max(y,0)),color=color,ls=":",alpha=.45)
    ax[0].plot(a,a,color=GRAY,lw=.7)
    ax[0].set(xlabel="当前振幅 a",ylabel="更新振幅 r",title="保留不确定度的振幅更新")
    ax[0].legend(fontsize=7.7,loc="upper left")
    r=np.linspace(0,6,400)
    for y,color in [(4.,BLUE),(-4.,GREEN)]:
        value=(r-2)**2/16+(r*r-y)**2/18
        minimum=intensity_amplitude_prox(2.,y,3.,variance=8.)
        cost=(minimum-2)**2/16+(minimum*minimum-y)**2/18
        ax[1].plot(r,value-cost,color=color,label=f"y={int(y)}, σ=3")
        ax[1].plot(minimum,0,'o',color=color,ms=5)
    ax[1].set(ylim=(-.2,7),xlabel="试探振幅 r",ylabel="目标值减去最小值",title="全局径向最优解（a=2）")
    ax[1].legend(fontsize=9)
    save(fig,"06_radial_prox",{"kind":"equation-derived radial proximal operator", "tau":8.})


def performance():
    data=json.loads((ROOT/'native/validation/v08_refinement_measurements.json').read_text())
    fig,ax=plt.subplots(1,2,figsize=(9,3.4),layout="constrained")
    times=[data['refinement']['summary'][k]['median_wall_seconds'] for k in ['finite_difference','analytic']]
    memory=[v['peak_process_rss_kib']/1024 for v in data['memory']]
    for a,values,labels,ylabel,title in [(ax[0],times,["三点差分","解析导数"],"中位耗时 / 秒","同一前向模型的局部精修"),(ax[1],memory,["完整 SVD","分块 QR + SVD"],"进程峰值 RSS / MiB","3500 × 64 数值矩阵")]:
        a.bar(labels,values,color=[GRAY,BLUE],width=.55)
        for i,v in enumerate(values):a.text(i,v*1.04,f"{v:.3g}",ha="center",fontsize=10)
        a.set(ylabel=ylabel,title=title,ylim=(0,max(values)*1.22))
    save(fig,"07_refinement_performance",{"kind":"measured local benchmark", "source":"native/validation/v08_refinement_measurements.json"})


def joint_pixels():
    data=json.loads((ROOT/'native/validation/v08_joint_measurements.json').read_text())
    regions=data['cases'][-1]['example_regions']
    fig,ax=plt.subplots(1,2,figsize=(9,3.5),layout="constrained")
    for a,region in zip(ax,regions):
        x=np.array(region['pixel']);components=np.array(region['component_predictions'])
        a.errorbar(x,region['data'],yerr=region['sigma'],fmt='.',ms=4,color=GRAY,alpha=.65,lw=.6,label="观测 ± σ")
        a.plot(x,region['fitted_mean'],color=BLUE,label="总预测")
        a.plot(x,components[:,0],color=ORANGE,ls="--",label="结构分量 1")
        a.plot(x,components[:,1],color=GREEN,ls="--",label="结构分量 2")
        a.plot(x,region['fitted_background'],color=GRAY,ls=":",label="拟合背景")
        a.set(title="完全重叠：局部秩 1" if region['rank']==1 else "部分重叠：局部秩 2",xlabel="像素位置",ylabel="像素信号")
    ax[1].legend(fontsize=8)
    save(fig,"08_joint_pixels",{"kind":"synthetic local structure refinement", "source":"native/validation/v08_joint_measurements.json"})


def recovery():
    data=json.loads((ROOT/'native/validation/v08_phasing/result.json').read_text())
    fig,ax=plt.subplots(1,2,figsize=(9,3.5),layout="constrained")
    cases=data['weak_phasing']
    for i,c in enumerate(cases):
        errors=[v['rms_after_angstrom'] for v in c['candidates']]
        ax[0].scatter(np.full(len(errors),c['negative_intensities'])+np.linspace(-10,10,len(errors)),errors,color=BLUE,s=28)
    ax[0].set(xlabel="4540 条观测中的负强度数量",ylabel="精修后坐标 RMS / Å",title="四个未知原子：每档四个起点")
    ax[0].set_yscale('log')
    twin=data['overlap_autonomy']
    values=[twin['truth_domain_fractions'][0],twin['refinement']['domain_fraction_estimates'][0]]
    ax[1].bar(["合成真值","盲解与联合精修"],values,color=[GRAY,BLUE],width=.55)
    ax[1].set(ylim=(0,.8),ylabel="参考晶畴比例",title="混合观测组的完整原生流程")
    for i,v in enumerate(values):ax[1].text(i,v+.018,f"{v:.5f}",ha="center")
    ax[1].text(.5,.12,f"审计 mode RMS = {twin['audit']['mode_sigma_rmse']:.3f}\n坐标 RMS = {twin['coordinate_rms_angstrom']:.4f} Å",transform=ax[1].transAxes,ha="center",fontsize=9)
    save(fig,"09_blind_recovery",{"kind":"synthetic blind recovery; supplied composition, no truth coordinates", "source":"native/validation/v08_phasing/result.json"})


def cpp_kernel():
    data = json.loads((ROOT/'native/validation/v08_kernel_measurements.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.4), constrained_layout=True)
    colors = {'python': GRAY, 'cpp': BLUE}
    labels = {'python': 'NumPy', 'cpp': 'C++'}
    x = np.arange(len(data['predictor']))
    width = .32
    for offset, backend in [(-.5, 'python'), (.5, 'cpp')]:
        values = [case['summary'][backend]['median_wall_seconds']*1000/case['evaluations_per_run']
                  for case in data['predictor']]
        axes[0].bar(x+offset*width, values, width, color=colors[backend], label=labels[backend])
    axes[0].set(xticks=x, xticklabels=['3 原子 / 3 畴\n7,280 反射','32 原子 / 1 畴\n4,096 反射'],
                ylabel='每次预测与解析导数 / ms', title='相同物理模型的数值内核')
    axes[0].legend(fontsize=8)
    refinement = data['refinement']['summary']
    values = [refinement[b]['median_wall_seconds']*1000 for b in ['python','cpp']]
    axes[1].bar(['NumPy','C++'], values, color=[GRAY,BLUE], width=.55)
    axes[1].set(ylabel='完整局部精修 / ms', title='18 原子、1,098 反射')
    for i, value in enumerate(values):
        axes[1].text(i,value*1.03,f'{value:.1f}',ha='center',fontsize=9)
    axes[1].set_ylim(0,max(values)*1.2)
    blocks = sorted({row['block_size'] for row in data['memory']})
    for offset, backend in [(-.5,'python'),(.5,'cpp')]:
        rows = {row['block_size']:row for row in data['memory'] if row['backend']==backend}
        axes[2].bar(np.arange(len(blocks))+offset*width,
                    [rows[b]['peak_process_rss_kib']/1024 for b in blocks], width,
                    color=colors[backend],label=labels[backend])
    axes[2].set(xticks=np.arange(len(blocks)),xticklabels=[str(b) for b in blocks],
                xlabel='反射块大小',ylabel='进程峰值 RSS / MiB', title='80 原子、8,192 反射')
    save(fig,'10_cpp_kernel',{'kind':'measured local CPU benchmark; same analytic model',
                            'source':'native/validation/v08_kernel_measurements.json',
                            'memory_scope':'whole process RSS, including inputs, libraries and global real Jacobian'})


if __name__ == '__main__':
    makers = (phase_problem, weak_bias, overlap, poisson_profile, proximal, performance, joint_pixels, recovery, cpp_kernel)
    for make in makers:
        make()
    provenance['14_scheduler_timeline'] = {
        'kind': 'measured event timeline of real concurrent native refinement',
        'source': 'docs/v08/results/scheduler_event_timeline.json',
        'reproduce': 'python benchmarks/v08_scheduler.py',
    }
    (OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"Wrote {len(makers)} reproducible figure pairs to {OUT}; scheduler figure is built separately")
