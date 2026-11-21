from __future__ import annotations
import shutil
import re
from pathlib import Path
import subprocess
import sys

from .science import SOURCE,dump,run_native,clean


def command(argv,cwd,log,timeout=None,*,environment=None):
    print(' '.join(map(str,argv)),flush=True)
    result=subprocess.run(list(map(str,argv)),cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                          check=False,timeout=timeout,env=environment)
    if result.returncode:raise RuntimeError(f'计算程序返回 {result.returncode}；详细输出保存在日志中。')


def dials(action,project,folder,out,settings,log):
    files=lambda key:folder/project[key]
    args=[];updates={}
    if action=='import':
        if not project.get('frames'):raise ValueError('没有可导入的原始衍射帧。')
        args=[str(folder/f) for f in project['frames']]
    elif action=='find_spots':args=[files('experiments')]
    else:
        reflection=files('dials_reflections')
        if action=='index' and (project.get('selection') or project.get('excluded')):
            from dials.array_family import flex
            table=flex.reflection_table.from_file(str(reflection));excluded=set(project.get('excluded',[]))
            selected=[i for i in (project.get('selection') or range(len(table))) if i not in excluded]
            if not selected:raise ValueError('没有可用于指标化的斑点；请恢复部分斑点。')
            if any(type(i) is not int or i<0 or i>=len(table) for i in selected):raise ValueError('选点索引无效。')
            table.select(flex.size_t(selected)).as_file(str(out/'selected.refl'));reflection=out/'selected.refl'
        args=[files('experiments'),reflection]
    if action in {'import','index','refine','integrate','scale','predict'}:args+=['output.experiments=output.expt'] if action!='predict' else []
    if action in {'find_spots','index','refine','integrate','scale','predict'}:args+=['output.reflections=output.refl']
    if action=='export':args+=['format=shelx','shelx.hklout=output.hkl']
    options = dict(settings.get('dials_options', {}))
    threads = settings.get('_admitted_cpu_threads')
    environment = None
    if threads is not None:
        from ariadne.agent.backends import DIALS_PROCESS_OPTIONS
        from ariadne.agent.process import thread_environment

        key = DIALS_PROCESS_OPTIONS.get(action)
        if key:
            requested = options.setdefault(key, threads)
            if type(requested) is not int or not 1 <= requested <= threads:
                raise ValueError(f'{key} 必须为 1–{threads} 之间的整数。')
        environment = thread_environment(1 if key else threads)
    for key,value in options.items():
        if not isinstance(key,str) or key.startswith(('output.','input.')) or '\n' in key:raise ValueError('无效的专业参数名称。')
        if isinstance(value,(dict,list)):raise ValueError('DIALS 专业参数必须是单个值。')
        args.append(f'{key}={str(value).lower() if isinstance(value,bool) else value}')
    module='dials_import' if action=='import' else action
    command([sys.executable,'-m',f'dials.command_line.{module}',*args],out,log,environment=environment)
    expected = {
        'import': ('output.expt',),
        'find_spots': ('output.refl',),
        'predict': ('output.refl',),
        'export': ('output.hkl',),
    }.get(action, ('output.expt', 'output.refl'))
    for name in expected:
        path = out/name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'DIALS {action} 未生成 {name}。')
    for name,key in [('output.expt','experiments'),('output.refl','dials_reflections'),('output.hkl','reflections')]:
        if name in expected:updates[key]=(out/name).relative_to(folder).as_posix()
    if 'experiments' in updates:
        from dxtbx.model.experiment_list import ExperimentListFactory
        expts=ExperimentListFactory.from_json_file(str(folder/updates['experiments']),check_format=False)
        crystal=next((x.crystal for x in expts if x.crystal is not None),None)
        if crystal:
            updates['cell']=dict(zip(('a','b','c','alpha','beta','gamma'),crystal.get_unit_cell().parameters()))
    if action in {'find_spots','index'}:updates.update(selection=[],excluded=[])
    return updates


def analysis(action,project,folder,out,settings,log):
    from ariadne.contracts import spec_for
    method=settings.get('refinement_method','anchors');spec=spec_for(action,method)
    if not spec:raise ValueError('分析动作不可用。')
    inputs=settings.get('inputs',[])
    if not spec.minimum_inputs<=len(inputs)<=spec.maximum_inputs:raise ValueError('请按声明的输入角色选择文件。')
    paths=[]
    for name in inputs:
        path=(folder/name).resolve()
        if not path.is_relative_to(folder) or not path.is_file():raise ValueError('输入文件必须在当前项目中。')
        paths.append(str(path))
    argv=[sys.executable,str(SOURCE/'skills/ariadne-reduction/scripts'/spec.executable)]
    if action=='compare_hypotheses':argv += [paths[0],'comparison.json',*paths[1:]]
    elif action=='joint_refine' and method=='em':
        argv += [*paths[:3],'analysis','--centroid-selection',paths[3]]
        if len(paths)>4:argv += ['--compare-models',*paths[4:]]
    else:argv += [*paths,'twin.hkl' if action=='export_hklf5' else 'analysis']
    if action=='joint_fit':
        argv += ['--estimator',settings.get('estimator','signed-wls'),'--confidence-level',str(settings.get('confidence_level',.95))]
        if settings.get('fit_positions'):argv+=['--fit-positions']
    if action=='joint_refine':argv+=['--method',method]
    if action=='build_overlaps' and settings.get('position_derivatives'):argv+=['--position-derivatives']
    command(argv,out,log);return {}


def legacy(action,project,folder,out,settings,log):
    from .shelx_engine import engine_info,input_plan
    map_only=action=='legacy:shelxl-maps';engine='shelxl' if map_only else action.split(':',1)[1]
    info=engine_info(engine)
    if not info['available']:raise ValueError(info['reason'])
    plan=input_plan(project,folder,settings,engine);stem=plan['stem'];inputs=[]
    for relative,source in plan['files']:
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        inputs.append({'source':source.relative_to(folder).as_posix(),'staged':relative.as_posix(),'bytes':source.stat().st_size})
    argv=[info['executable'],stem,*plan['extra_args']]
    result={'engine':engine,'executable':info['executable'],'argv':argv,'inputs':inputs,
            'instruction_handling':'original bytes; instructions executed by selected official program',
            'scientific_acceptance':'not_assessed','status':'running'}
    if map_only:
        from .shelx_maps import prepare_map_input
        result['map_recipe']=prepare_map_input(plan,out)
        result['instruction_handling']='derived map-only copy; exact source instructions archived; changes listed in map_recipe'
    dump(out/'result.json',result)
    try:
        command(argv,out,log,timeout=600)
        log.flush()
        if getattr(log,'name',None):
            version=re.search(r'Version\s+([0-9]+/[0-9]+)',Path(log.name).read_text(errors='replace'))
            if version:result['version']=engine.upper()+'-'+version.group(1)
        if map_only:
            from .shelx_maps import maps_from_fcf
            fcf=out/(stem+'.fcf')
            if not fcf.is_file():raise RuntimeError('SHELXL 未生成 LIST 6 FCF；请检查程序日志。')
            result.update(status='completed',**maps_from_fcf(fcf,out))
            dump(out/'surface.json',{'type':'unweighted Fo-Fc','source':'SHELXL LIST 6'})
            dump(out/'result.json',result)
            return {'surface':(out/'surface.json').relative_to(folder).as_posix()}
        candidates=[out/(stem+'.res')] if engine=='shelxl' else sorted(out.glob(stem+'*.res'))
        model=next((p for p in candidates if p.is_file() and p.stat().st_size),None)
        if model is None:raise RuntimeError(f'{engine.upper()} 未生成 RES；请检查日志中的程序诊断。')
    except Exception as error:
        result.update(status='failed',error=str(error));dump(out/'result.json',result);raise
    result.update(status='completed',structure=model.relative_to(folder).as_posix())
    listing=out/(stem+'.lst')
    if listing.is_file():
        version=re.search(r'SHELX[LT][ -]+[0-9][0-9./-]*',listing.read_text(errors='replace'))
        if version:result['version']=version.group(0)
    updates={}
    updates['structure']=model.relative_to(folder).as_posix();updates['refinementEngine']=engine
    if plan['reflections'] is not None:updates['reflections']=(out/(stem+'.hkl')).relative_to(folder).as_posix()
    cif=out/(model.stem+'.cif')
    if cif.is_file():
        import gemmi
        try:
            block=gemmi.cif.read_file(str(cif)).sole_block();metrics={}
            version=block.find_value('_shelx_SHELXL_version_number')
            if version:result['version']='SHELXL-'+gemmi.cif.as_string(version)
            for key,tag in {'r1_all':'_refine_ls_R_factor_all','r1_observed':'_refine_ls_R_factor_gt',
                            'wr2':'_refine_ls_wR_factor_ref','goodness_of_fit':'_refine_ls_goodness_of_fit_ref',
                            'max_shift_esd':'_refine_ls_shift/su_max','residual_peak':'_refine_diff_density_max',
                            'residual_hole':'_refine_diff_density_min'}.items():
                value=block.find_value(tag)
                if value and value not in {'.','?'}:metrics[key]=gemmi.cif.as_number(value)
            if metrics:updates['metrics']=clean(metrics);result['metrics']=clean(metrics)
        except Exception as error:result['metadataWarning']=str(error)
    # Metadata extraction is read-only and must never invalidate a completed
    # official run merely because its instructions have no native translation.
    try:
        from .structure_input import display_model
        structure,compatibility=display_model(model)
        updates.update(cell=structure.cell.to_dict(),spacegroup=structure.spacegroup,structureCompatibility=compatibility)
        result['warnings']=list(compatibility.get('scientificWarnings',[]))
    except Exception as error:result['previewWarning']=str(error)
    if result.get('metrics',{}).get('max_shift_esd',0)>.1:
        result.setdefault('warnings',[]).append(f'最大 shift/esd = {result["metrics"]["max_shift_esd"]:g}；需检查精修收敛。')
    updates['refinementWarnings']=result.get('warnings',[])
    updates['lastRefinement']={'engine':engine,'result':(out/'result.json').relative_to(folder).as_posix()}
    dump(out/'result.json',result)
    return updates


def execute_step(action, project, folder, out, settings, log, *, follow_refine=False):
    if action.startswith('dials:'):
        return dials(action.split(':', 1)[1], project, folder, out, settings, log)
    if action.startswith('analysis:'):
        return analysis(action.split(':', 1)[1], project, folder, out, settings, log)
    if action.startswith('legacy:'):
        if follow_refine:
            settings = {**settings, 'shelx_structure': project['structure'],
                        'shelx_reflections': project.get('reflections')}
        return legacy(action, project, folder, out, settings, log)
    result = run_native(action, project, folder, out, settings)
    updates = {}
    for name in ('structure', 'surface'):
        path = out/(name+'.json')
        if path.exists():
            updates[name] = path.relative_to(folder).as_posix()
    metrics = result.get('metrics', result.get('refinement', {}).get('metrics'))
    if metrics:
        updates['metrics'] = metrics
    return updates
