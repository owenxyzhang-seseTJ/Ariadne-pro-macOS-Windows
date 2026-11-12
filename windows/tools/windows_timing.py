"""安装完成后单独测量 Windows 性能，避免安装进程争用 CPU。"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from threadpoolctl import threadpool_limits, threadpool_info
from windows_benchmark import benchmark_functions
from ariadne_xtal.kernels import kernel_status

root=Path(__file__).resolve().parent.parent
bench=benchmark_functions(root/'source')
result={'product':'Ariadne pro Windows','generated_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'executable':sys.executable,'kernel':kernel_status(),
        'scope':'独立计时；不与安装、编译或回归测试并行；单数值线程', 'predictor':[]}
with threadpool_limits(limits=1):
    result['threadpools']=threadpool_info()
    for name in ('three_atoms_three_domains','thirty_two_atoms','eighty_atoms_memory'):
        result['predictor'].append(bench['predictor_benchmark'](name,repeats=5))
    result['refinement']=bench['refinement_benchmark']()
assert all(row['success'] for row in result['refinement']['runs'])
assert all(row['maximum_scaled_jacobian_difference']<1e-10 for row in result['predictor'])
(root/'results/windows-timing.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
for row in result['predictor']:
    print(row['case'],{name:summary['median_wall_seconds']/row['evaluations_per_run']*1000 for name,summary in row['summary'].items()},flush=True)
print('refinement',{name:summary['median_wall_seconds']*1000 for name,summary in result['refinement']['summary'].items()},flush=True)
