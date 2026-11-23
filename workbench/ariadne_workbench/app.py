from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
from PIL import Image
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse,FileResponse
from starlette.routing import Route,Mount
from starlette.staticfiles import StaticFiles

from .science import SOURCE,clean,dump,classify,structure_view,reflection_view,frame_array,pixel_profile,detector_view
from .density import density_view
from . import workflow

ROOT=Path(__file__).resolve().parents[1]
PROJECTS=Path(os.environ.get('ARIADNE_PROJECTS_DIR',ROOT/'projects')).resolve()
PROJECTS.mkdir(parents=True,exist_ok=True)
LOCK=threading.Lock()


def folder_for(project_id):
    if not re.fullmatch(r'[a-z0-9-]{1,60}',project_id):raise ValueError('项目编号无效。')
    folder=(PROJECTS/project_id).resolve()
    if not folder.is_relative_to(PROJECTS) or not (folder/'project.json').is_file():raise ValueError('项目不存在。')
    return folder


def project_for(project_id):
    folder=folder_for(project_id)
    with workflow.project_lock(folder):
        return folder,workflow.load_project(folder)


def safe_file(folder,relative):
    path=(folder/str(relative)).resolve()
    if not path.is_relative_to(folder) or not path.is_file():raise ValueError('文件不在当前项目中或不存在。')
    return path


def refresh(project,folder,files):
    for path in files:
        role=classify(path);relative=path.relative_to(folder).as_posix()
        project.setdefault('files',[]).append({'path':relative,'name':path.name,'role':role,'bytes':path.stat().st_size})
        if role=='frame':project.setdefault('frames',[]).append(relative)
        elif role!='artifact':project[role]=relative
        if role in {'dials_reflections','reflections'}:
            project['selection']=[];project['excluded']=[]
    project['frames']=sorted(set(project.get('frames',[])))
    if project.get('structure'):
        try:
            from .structure_input import display_model
            from ariadne_xtal.symmetry import unique_sites
            model,compatibility=display_model(folder/project['structure']);project['cell']=model.cell.to_dict()
            project['structureCompatibility']=compatibility
            if compatibility['nativeCompatible']:
                project['elements']=[atom.element for atom in model.atoms for _ in unique_sites(model,atom)]
            project['spacegroup']=model.spacegroup
        except Exception as error:project.setdefault('importWarnings',[]).append(str(error))
    project['updated']=time.time();dump(folder/'project.json',project)
    return project


def new_project(name):
    project_id=uuid.uuid4().hex[:12];folder=PROJECTS/project_id;(folder/'input').mkdir(parents=True)
    project={'id':project_id,'name':str(name or '未命名项目')[:100],'created':time.time(),'updated':time.time(),'files':[],
             'selection':[],'frames':[],'checkpoints':False}
    dump(folder/'project.json',project);return folder,project


def read_jobs(folder):
    jobs=[]
    for path in (folder/'jobs').glob('*/job.json'):
        try:
            job=json.loads(path.read_text(encoding='utf-8'))
            if job.get('workflow') == workflow.SCHEMA:
                jobs.append(workflow.status(path.parent,job))
                continue
            if job['status'] in {'queued','running'} and job.get('pid'):
                if os.name == 'nt':
                    from ariadne.agent.windows_job import pid_alive
                    if not pid_alive(job['pid']):job['status']='interrupted'
                else:
                    try:os.kill(job['pid'],0)
                    except ProcessLookupError:job['status']='interrupted'
                    except PermissionError:pass
            log=path.parent/'run.log'
            job['logTail']=''
            if log.is_file():
                with log.open('rb') as stream:
                    size=stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, size-32_000))
                    job['logTail']=stream.read(32_000).decode('utf-8', errors='replace')[-8000:]
            jobs.append(job)
        except (ValueError,OSError):pass
    return sorted(jobs,key=lambda x:x['created'],reverse=True)


NATIVE_TOOLS=[
 ('inspect','结构检查','结构与晶胞检查',['structure'],'检查晶胞、对称性、原子与配位显示。'),
 ('refine','结构精修','结构求解与精修',['structure','reflections'],'使用带符号 F² 观测进行原生精修。'),
 ('autosolve','自动求解与精修','结构求解与精修',['reflections','cell','elements'],'生成候选结构、精修并独立比较；需要整个晶胞组成。'),
 ('solve','多起点相位求解','结构求解与精修',['reflections','cell','elements'],'电荷翻转、原子投影与 hybrid 相位路径。'),
 ('resolve','孪晶与空间群比较','孪晶与高级分析',['reflections'],'共用观测划分、候选筛选、精修与留出比较。'),
 ('symmetry','空间群候选检查','结构求解与精修',['structure','reflections'],'比较声明的空间群候选和原点。'),
 ('maps','电子密度与 Patterson 图','结构与密度',['structure','reflections'],'导出 Fo−Fc、计算密度、Patterson 与 CCP4 文件。'),
 ('absolute','绝对结构诊断','结构求解与精修',['structure','reflections'],'基于固定模型的 Friedel 商诊断，附实际适用条件。'),
 ('integrate-pixels','校准像素积分','积分与复杂样品',['pixels'],'Gaussian / Poisson 信号与背景拟合、区间与可辨识性。'),
 ('joint-refine','重叠观测联合精修','积分与复杂样品',['structure','observations'],'结构与晶畴比例对原始压缩似然联合精修。'),
 ('solve-observations','从重叠观测求解结构','积分与复杂样品',['observations','cell','elements'],'隔离参考畴初始化，再对工作观测共同精修。'),
]
REDUCTION_LABELS={'import':'导入衍射帧','find_spots':'自动找点','index':'指标化与晶胞','refine':'精修实验几何','integrate':'反射积分','scale':'缩放与校正','export':'导出反射','predict':'预测反射'}
ANALYSIS_LABELS={'soft_assign':'多晶畴软分配','joint_refine':'联合几何优化','learn_profiles':'轮廓库学习','build_overlaps':'构建重叠组','joint_fit':'重叠像素拟合','export_hklf5':'HKLF5 导出','compare_hypotheses':'独立候选比较','scaling_geometry':'缩放几何诊断','global_scale':'复合观测整体缩放'}


async def capabilities(request):
    from ariadne_xtal.capabilities import capabilities as native_capabilities
    from ariadne.contracts import analysis_contracts
    dials_ready=importlib.util.find_spec('dials') is not None
    tools=[{'id':key,'name':name,'group':group,'requires':requires,'description':description,'available':True} for key,name,group,requires,description in NATIVE_TOOLS]
    for action,label in REDUCTION_LABELS.items():
        required=['frames'] if action=='import' else ['experiments'] if action=='find_spots' else ['experiments','dials_reflections']
        tools.append({'id':f'dials:{action}','name':label,'group':'衍射数据还原','requires':required,
                      'description':'调用当前安装的 DIALS / Ariadne 扩展。','available':dials_ready,
                      'reason':None if dials_ready else '当前 Python 环境未载入 DIALS。'})
    contracts=analysis_contracts()
    for key,label in ANALYSIS_LABELS.items():
        spec=contracts[key] if key in contracts else next((v for k,v in contracts.items() if k.startswith(key)),None)
        if hasattr(spec,'model_dump'):spec=spec.model_dump()
        tools.append({'id':f'analysis:{key}','name':label,'group':'孪晶与高级分析','requires':[],
                      'description':'按输入角色运行 Ariadne 的科学分析适配器。','available':True,
                      'inputRoles':spec.get('inputs',[]) if spec else [],'contract':spec})
    for engine in ['shelxt','shelxl']:
        from .shelx_engine import engine_info
        info=engine_info(engine)
        tools.append({'id':f'legacy:{engine}','name':engine.upper()+(' 精修' if engine=='shelxl' else ' 结构求解'),'group':'外部引擎','requires':['structure','reflections'],
                      'description':'使用本机官方程序，按原始 INS / RES 指令执行；可单独选择 HKL，每次运行保存在独立目录。',**info})
        if engine=='shelxl':tools.append({'id':'legacy:shelxl-maps','name':'SHELXL 电子密度','group':'外部引擎','requires':['structure','reflections'],
                                         'description':'独立副本设置 L.S. 0 / LIST 6，并跳过 ACTA 报告生成，仅计算结构因子；生成未加权 Fo−Fc、Fc 与 Patterson 图。',**info})
    return JSONResponse(clean({'version':'0.8.0','tools':tools,'native':native_capabilities(),
                               'runtime':{'dials':dials_ready,'python':sys.version.split()[0],
                                          'source':str(SOURCE),'workflow':workflow.SCHEMA},
                               'visualization':{'structure':True,'reciprocal':True,'frames':dials_ready,'density':True,'showcase':'Blender GLB'}}))


async def list_projects(request):
    rows=[]
    for path in PROJECTS.glob('*/project.json'):
        try:rows.append(json.loads(path.read_text(encoding='utf-8')))
        except ValueError:pass
    return JSONResponse(clean(sorted(rows,key=lambda x:x['updated'],reverse=True)))


async def import_project(request):
    form=await request.form(max_files=20000,max_fields=20000,max_part_size=512*1024*1024)
    project_id=request.path_params.get('project_id')
    folder,project=project_for(project_id) if project_id else new_project(form.get('name','未命名项目'))
    copied=[]
    for item in form.getlist('files'):
        if not getattr(item,'filename',None):continue
        relative=PurePosixPath(item.filename.replace('\\','/'))
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('导入文件名不能包含上级目录。')
        target=folder/'input'/relative
        if target.exists():target=target.with_name(f'{target.stem}-{uuid.uuid4().hex[:6]}{target.suffix}')
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('wb') as stream:
            while chunk:=await item.read(1024*1024):stream.write(chunk)
        copied.append(target)
    if not copied:raise ValueError('请选择至少一个数据文件。')
    def commit_import():
        with workflow.project_lock(folder):
            return refresh(workflow.load_project(folder),folder,copied)
    project=await asyncio.to_thread(commit_import)
    return JSONResponse(clean(project),status_code=201)


async def demo(request):
    body=await request.json();kind=body.get('kind','fen4')
    folder,project=new_project({'fen4':'FeN₄ · 精修示例','pixels':'校准像素 · 积分示例','mof5':'MOF-5 · 结构示例','hkust1':'HKUST-1 · 结构示例'}.get(kind,'示例项目'))
    sources={'fen4':[SOURCE/'examples/v08/fen4_start.json',SOURCE/'examples/v08/fen4_measured.json'],
             'pixels':[SOURCE/'examples/v08/count_pixels.json'],
             'mof5':[ROOT/'public/models/sources/mof5.cif'],
             'hkust1':[ROOT/'public/models/sources/hkust1.cif']}
    if kind not in sources:raise ValueError('未知示例。')
    paths=[]
    for src in sources[kind]:
        target=folder/'input'/src.name;shutil.copyfile(src,target);paths.append(target)
    project['demo']=kind
    return JSONResponse(clean(await asyncio.to_thread(refresh,project,folder,paths)))


async def get_project(request):
    folder=folder_for(request.path_params['project_id'])
    with workflow.project_lock(folder):
        jobs=read_jobs(folder)
        project=workflow.load_project(folder)
    if project.get('structure'):
        try:
            from .structure_input import display_model
            model,compatibility=await asyncio.to_thread(display_model,folder/project['structure'])
            project['structureCompatibility']=compatibility
            if not project.get('cell'):project['cell']=model.cell.to_dict()
        except (ValueError,RuntimeError) as error:
            project['structureCompatibility']={'displayOnly':True,'nativeCompatible':False,'message':str(error),'issues':[str(error)]}
    project['jobs']=jobs
    project['artifacts']=[{'path':path.relative_to(folder).as_posix(),'name':path.name,'bytes':path.stat().st_size}
                          for path in (folder/'jobs').rglob('*')
                          if path.is_file() and path.suffix in {'.cif','.hkl','.ins','.res','.fcf','.lst','.fab','.npz','.ccp4','.json','.log','.expt','.refl','.mtz'}
                          and path.name not in workflow.CONTROL_FILES
                          and ('session' not in path.relative_to(folder).parts[2:3]
                               or 'attempts' in path.relative_to(folder).parts[3:4])]
    return JSONResponse(clean(project))


async def settings(request):
    folder=folder_for(request.path_params['project_id']);body=await request.json()
    with workflow.project_lock(folder):
        project=workflow.load_project(folder)
        update_settings(project,body)
        dump(folder/'project.json',project)
    return JSONResponse(clean(project))


def update_settings(project,body):
    if 'cell' in body:
        from ariadne_xtal.model import Cell
        project['cell']=Cell(**body['cell']).to_dict()
    if 'elements' in body:
        if not isinstance(body['elements'],list) or not 1<=len(body['elements'])<=2500:raise ValueError('晶胞组成应为 1–2500 个元素符号。')
        import gemmi
        if any(not isinstance(x,str) or gemmi.Element(x).atomic_number==0 for x in body['elements']):raise ValueError('元素符号无效。')
        project['elements']=body['elements']
    if 'checkpoints' in body:project['checkpoints']=bool(body['checkpoints'])
    if 'selection' in body:
        values=body['selection']
        if not isinstance(values,list) or len(values)>100000 or any(type(x)is not int or x<0 for x in values):raise ValueError('选点数据无效。')
        project['selection']=sorted(set(values))
    if 'excluded' in body:
        values=body['excluded']
        if not isinstance(values,list) or len(values)>100000 or any(type(x)is not int or x<0 for x in values):raise ValueError('屏蔽索引无效。')
        project['excluded']=sorted(set(values))
    if 'name' in body:project['name']=str(body['name'])[:100]


async def views(request):
    folder,project=project_for(request.path_params['project_id']);kind=request.path_params['kind']
    if kind=='structure':
        if not project.get('structure'):return JSONResponse({'atoms':[],'bonds':[],'cellEdges':[]})
        return JSONResponse(clean(await asyncio.to_thread(structure_view,folder/project['structure'])))
    if kind=='reflections':return JSONResponse(await asyncio.to_thread(reflection_view,project,folder))
    if kind=='detector':
        probe=(float(request.query_params['x']),float(request.query_params['y'])) if 'x' in request.query_params and 'y' in request.query_params else None
        return JSONResponse(await asyncio.to_thread(detector_view,project,folder,int(request.query_params.get('frame',0)),probe))
    if kind=='density':return JSONResponse(clean(await asyncio.to_thread(density_view,project,folder,request.query_params.get('map'),float(request.query_params.get('sigma',3.)))))
    if kind=='profile':
        ref_id=request.query_params.get('reflection');row=None
        if ref_id is not None:
            data=await asyncio.to_thread(reflection_view,project,folder)
            row=next((r for r in data['reflections'] if r['id']==int(ref_id)),None)
        value=await asyncio.to_thread(pixel_profile,project,folder,row,int(request.query_params.get('frame',0)))
        return JSONResponse(clean(value))
    raise ValueError('未知视图。')


async def frame(request):
    folder,project=project_for(request.path_params['project_id']);index=int(request.path_params['index'])
    data=await asyncio.to_thread(frame_array,project,folder,index)
    stride=max(1,int(np.ceil(max(data.shape)/1200)));sample=data[::stride,::stride]
    positive=sample[np.isfinite(sample)&(sample>0)];scale=float(np.percentile(positive,80)) if len(positive) else 1.
    sample=np.arcsinh(np.maximum(np.nan_to_num(sample),0)/max(scale,.01))
    top=max(float(np.percentile(sample,99.8)),.001);pixels=np.uint8(np.clip(sample/top,0,1)*255)
    image=Image.fromarray(255-pixels if request.query_params.get('theme')=='light' else pixels)
    buffer=io.BytesIO();image.save(buffer,format='PNG')
    from starlette.responses import Response
    return Response(buffer.getvalue(),media_type='image/png',headers={'X-Frame-Width':str(data.shape[1]),'X-Frame-Height':str(data.shape[0])})


async def start_job(request):
    folder=folder_for(request.path_params['project_id']);body=await request.json()
    catalog=json.loads((await capabilities(request)).body)['tools']
    job=await asyncio.to_thread(submit_job,folder,body,catalog)
    return JSONResponse(job,status_code=202)


def submit_job(folder,body,catalog):
    with workflow.project_lock(folder):
        project=workflow.load_project(folder)
        if any(j['status'] in {'running','queued'} for j in read_jobs(folder)):
            raise ValueError('当前项目已有任务在运行。')
        return _submit_job(folder,project,body,catalog)


def _submit_job(folder,project,body,catalog):
    action=body.get('action','automatic')
    settings=body.get('settings',{})
    if not isinstance(settings,dict):raise ValueError('设置必须是 JSON 对象。')
    if 'screening' in settings and action != 'solve':
        raise ValueError('screening 参数用于多起点相位求解。')
    requested_action=action
    shelx=bool(project.get('structure') and Path(project['structure']).suffix.lower() in {'.ins','.res'})
    # The generic refinement entry follows the source format. Explicit native
    # requests remain strict; an official run never rewrites SHELX directives.
    if action=='refine' and shelx and settings.get('engine')!='native':action='legacy:shelxl'
    if action=='maps' and shelx and settings.get('engine')!='native':action='legacy:shelxl-maps'
    selected=next((t for t in catalog if t['id']==action),None)
    if action!='automatic':
        if selected is None:raise ValueError('动作未注册。')
        if not selected['available']:raise ValueError(selected.get('reason','所选动作尚未配置。'))
        if action != 'resolve' and not action.startswith('legacy:') and any(not project.get(key) for key in selected['requires']):raise ValueError('请先补充该动作所需的输入。')
    needs_model=action in {'refine','maps','symmetry','absolute','joint-refine','solve-observations'}
    if action=='resolve':
        inputs = settings.get('resolution_inputs', {})
        if not isinstance(inputs, dict) or not (project.get('reflections') or inputs.get('hkl')):
            raise ValueError('请提供反射表或项目内的反射文件路径。')
        if not (project.get('structure') or project.get('cell') or settings.get('cell')
                or any(role in {'structure', 'ins'} or role.startswith('laue_') for role in inputs)):
            raise ValueError('请提供初始结构、SHELXT 控制文件或晶胞与整个晶胞的元素组成。')
    if needs_model and shelx:
        from .structure_input import native_model
        native_model(folder/project['structure'])
    official=action if action.startswith('legacy:') else 'legacy:shelxl' if action=='automatic' and shelx and not project.get('frames') and not project.get('pixels') else None
    if official:
        from .shelx_engine import engine_info,input_plan
        engine=official.split(':')[1].removesuffix('-maps');info=engine_info(engine)
        if not info['available']:raise ValueError(info['reason'])
        plan=input_plan(project,folder,settings,engine)
        # Freeze the selected pair for this job, including retries, even if the
        # user subsequently changes the project's active files.
        settings={**settings,'shelx_structure':plan['original'].relative_to(folder).as_posix()}
        if plan['reflections'] is not None:settings['shelx_reflections']=plan['reflections'].relative_to(folder).as_posix()
    if action=='automatic':
        if project.get('frames'):
            actions=[f'dials:{x}' for x in ('import','find_spots','index','refine','integrate','scale','export')]
            if project.get('elements'):actions+=['autosolve']
        elif project.get('pixels'):actions=['integrate-pixels']
        elif official:actions=[official,'legacy:shelxl-maps']
        elif project.get('structure') and project.get('reflections'):actions=['refine','maps']
        elif project.get('reflections') and project.get('cell') and project.get('elements'):actions=['autosolve','maps']
        else:raise ValueError('请补充反射数据，或晶胞参数与整个晶胞组成。')
        if project.get('checkpoints'):
            actions=[a for a in actions if a not in project.get('completedActions',[])][:1]
            if not actions:raise ValueError('当前流程已完成。可在专业工作区选择要重新运行的步骤。')
    else:actions=[action]
    job_id=uuid.uuid4().hex[:12];jobfolder=folder/'jobs'/job_id;jobfolder.mkdir(parents=True)
    job={'id':job_id,'projectId':project['id'],'action':action,'requestedAction':requested_action,'status':'queued','created':time.time(),
         'steps':[{'action':a,'status':'queued'} for a in actions],'settings':settings}
    if official:job['engine']=official.split(':')[1].removesuffix('-maps')
    workflow.initialize(jobfolder,project,job)
    project['lastJob']=job_id;project['updated']=time.time();dump(folder/'project.json',project)
    run=workflow.submit(jobfolder)
    return {**job,'pid':run['pid'],'runId':run['id']}


async def control_job(request):
    folder,_=project_for(request.path_params['project_id']);jobfile=safe_file(folder,f"jobs/{request.path_params['job_id']}/job.json")
    job=json.loads(jobfile.read_text(encoding='utf-8'));body=await request.json();action=body.get('action')
    if job.get('workflow') == workflow.SCHEMA:
        result=await asyncio.to_thread(control_workflow,folder,jobfile.parent,job,action)
        return JSONResponse(result)
    if action=='cancel':
        with LOCK:
            job=json.loads(jobfile.read_text(encoding='utf-8'))
            if job['status'] not in {'running','queued'}:raise ValueError('该任务已经结束。')
            (jobfile.parent/'cancel').touch()
            if os.name!='nt' and job.get('pid'):
                running_command=subprocess.run(['ps','-p',str(job['pid']),'-o','command='],capture_output=True,text=True,check=False).stdout
                if str(jobfile) in running_command:
                    try:os.killpg(job['pid'],signal.SIGTERM)
                    except ProcessLookupError:pass
            job=json.loads(jobfile.read_text(encoding='utf-8'))
            if job['status'] in {'running','queued'}:
                job.update(status='cancelled',message='任务已取消，已完成步骤和日志已保留。',finished=time.time())
                for step in job['steps']:
                    if step['status']=='running':step['status']='cancelled'
                dump(jobfile,job)
    elif action=='resume':
        if job['status'] not in {'failed','cancelled','interrupted'}:raise ValueError('当前任务不需要恢复。')
        if any(j['status']=='running' for j in read_jobs(folder)):raise ValueError('当前项目已有运行中的任务。')
        (jobfile.parent/'cancel').unlink(missing_ok=True)
        # A retry is a new run so earlier partial outputs remain reviewable.
        request_body={'action':job['action'],'settings':job.get('settings',{})}
        job['retryRequested']=time.time();dump(jobfile,job)
        scope=dict(request.scope)
        payload=json.dumps(request_body).encode()
        async def receive():return {'type':'http.request','body':payload,'more_body':False}
        return await start_job(Request(scope,receive))
    else:raise ValueError('未知任务操作。')
    return JSONResponse(job)


def control_workflow(folder,jobfolder,job,action):
    from ariadne.agent import async_runs

    with workflow.project_lock(folder):
        current=workflow.status(jobfolder,job)
        if action=='cancel':
            if current['status'] not in {'queued','running'}:
                raise ValueError('该任务已经结束。')
            async_runs.cancel(jobfolder/'session',current['runId'])
        elif action=='resume':
            if current['status'] not in {'failed','cancelled','interrupted'}:
                raise ValueError('当前任务不需要恢复。')
            if any(j['status'] in {'queued','running'} for j in read_jobs(folder)):
                raise ValueError('当前项目已有运行中的任务。')
            project=workflow.load_project(folder)
            workflow.resume(jobfolder)
            project['lastJob']=job['id'];project['updated']=time.time();dump(folder/'project.json',project)
        else:
            raise ValueError('未知任务操作。')
        return workflow.status(jobfolder,job)


async def job_events(request):
    from ariadne.agent.async_runs import events

    folder=folder_for(request.path_params['project_id'])
    jobfile=safe_file(folder,f"jobs/{request.path_params['job_id']}/job.json")
    if json.loads(jobfile.read_text(encoding='utf-8')).get('workflow') != workflow.SCHEMA:
        return JSONResponse({'events': [], 'next_cursor': 0, 'more': False})
    after=int(request.query_params.get('after', 0))
    limit=int(request.query_params.get('limit', 100))
    return JSONResponse(await asyncio.to_thread(events,jobfile.parent/'session',after,limit))


async def download(request):
    folder,_=project_for(request.path_params['project_id']);path=safe_file(folder,request.path_params['path'])
    return FileResponse(path,filename=path.name)


class LocalOnly(BaseHTTPMiddleware):
    async def dispatch(self,request,call_next):
        host=request.url.hostname
        if host not in {'127.0.0.1','localhost','testserver','[::1]','::1'}:return JSONResponse({'error':'仅接受本机连接。'},403)
        origin=request.headers.get('origin')
        if origin and origin not in {f'http://127.0.0.1:{request.url.port}',f'http://localhost:{request.url.port}'}:
            return JSONResponse({'error':'请求来源与本地工作台不一致。'},403)
        try:return await call_next(request)
        except (ValueError,KeyError,FileNotFoundError,TypeError) as error:return JSONResponse({'error':str(error)},400)
        except Exception as error:return JSONResponse({'error':str(error),'type':type(error).__name__},500)


routes=[Route('/api/capabilities',capabilities),Route('/api/projects',list_projects,methods=['GET']),
        Route('/api/projects',import_project,methods=['POST']),Route('/api/demo',demo,methods=['POST']),
        Route('/api/projects/{project_id}',get_project),Route('/api/projects/{project_id}/import',import_project,methods=['POST']),
        Route('/api/projects/{project_id}/settings',settings,methods=['PATCH']),
        Route('/api/projects/{project_id}/view/{kind}',views),Route('/api/projects/{project_id}/frame/{index:int}.png',frame),
        Route('/api/projects/{project_id}/jobs',start_job,methods=['POST']),
        Route('/api/projects/{project_id}/jobs/{job_id}',control_job,methods=['POST']),
        Route('/api/projects/{project_id}/jobs/{job_id}/events',job_events),
        Route('/api/projects/{project_id}/files/{path:path}',download),
        Mount('/',StaticFiles(directory=ROOT/'dist',html=True,check_dir=False),name='frontend')]
app=Starlette(routes=routes);app.add_middleware(LocalOnly)
