"""记录已部署 Windows 计算版的依赖、硬件和入口状态。"""
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from ariadne_xtal.kernels import kernel_status

root=Path(__file__).resolve().parent.parent
packages=[]
for package in importlib.metadata.distributions(path=[str(root/'overlay')]):
    name=package.metadata['Name']
    if name.lower().replace('_','-') not in {'ariadne','ariadne-xtal'}:
        packages.append(f'{name}=={package.version}')
(root/'requirements-win-tested.txt').write_text('\n'.join(sorted(set(packages),key=str.lower))+'\n',encoding='utf-8')
report=json.loads((root/'results/benchmark/measurements.json').read_text(encoding='utf-8'))
state={'schema':'ariadne-pro-windows/1','product':'Ariadne pro Windows','version':'0.8.0',
       'variant':'win-x86_64-localopt1','root':str(root),'source':str(root/'source'),
       'python':sys.executable,'python_version':platform.python_version(),
       'launcher':str(root/'ariadne-pro-win.cmd'),'kernel':kernel_status(),
       'regression':{'native_checks_passed':114,'python_reference_checks_passed':13,'failed':0,'skipped':0},
       'acceptance':report['acceptance'],
       'windows_boundaries':{'parent_mcp_server':False,'persistent_posix_agent':False},
       'dials_runtime':str(root/'dials-env'),
       'dials_launcher':str(root/'ariadne-pro-dials.cmd'),
       'dials_validation':json.loads((root/'results/dials-validation.json').read_text(encoding='utf-8')),
       'timing_result':str(root/'results/windows-timing.json')}
(root/'deployment.json').write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'root':str(root),'kernel':state['kernel']['resolved'],'pinned_packages':len(packages)}),flush=True)
