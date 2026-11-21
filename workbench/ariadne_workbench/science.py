from __future__ import annotations

import collections
import itertools
import json
import math
import os
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from ariadne_xtal.io import read_structure, read_reflections, write_structure
from ariadne_xtal.model import Cell, Structure

PLATFORM = 'windows' if os.name == 'nt' else 'macos'
SOURCE = Path(os.environ.get('ARIADNE_WORKBENCH_SOURCE',
                             Path(__file__).resolve().parents[2] / PLATFORM / 'source')).expanduser().resolve()
RAW_SUFFIXES = {'.sfrm', '.cbf', '.img', '.osc', '.mccd', '.h5', '.hdf5'}
COLORS = {'C':'#65716f','H':'#c9ccc9','O':'#b65d45','N':'#7a90a0','Zn':'#bd925f',
          'Cu':'#b67d4a','Fe':'#b3542c','S':'#bba463','Cl':'#81947f','Zr':'#98a099'}


def clean(value):
    if isinstance(value, np.ndarray):return clean(value.tolist())
    if isinstance(value, np.generic):return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):return None
    if isinstance(value, dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [clean(v) for v in value]
    return value


def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),ensure_ascii=False,allow_nan=False,indent=2), encoding='utf-8')
    temporary.replace(path)


def classify(path):
    ext=path.suffix.lower()
    if ext in RAW_SUFFIXES:return 'frame'
    if ext in {'.cif','.res','.ins'}:return 'structure'
    if ext in {'.ccp4','.map'}:return 'density'
    if ext in {'.hkl','.mtz'}:return 'reflections' if ext=='.hkl' else 'artifact'
    if ext=='.expt':return 'experiments'
    if ext=='.refl':return 'dials_reflections'
    if ext=='.json':
        try:
            obj=json.loads(path.read_text(encoding='utf-8'))
            if isinstance(obj,dict):
                if 'cell' in obj and 'atoms' in obj:return 'structure'
                if 'hkl' in obj and 'intensity' in obj:return 'reflections'
                if 'data' in obj and 'profiles' in obj:return 'pixels'
                if 'groups' in obj:return 'observations'
        except (ValueError,UnicodeError):pass
    return 'artifact'


def structure_view(path):
    from ariadne_xtal.visualization import expanded_sites, unit_cell_edges
    from .structure_input import display_model
    s,compatibility=display_model(path)
    sites=expanded_sites(s,maximum_sites=2500)
    points=np.array([a['cart'] for a in sites])
    cell=s.cell.matrix
    shifts=list(itertools.product((-1,0,1),repeat=3))
    translated=np.concatenate([points+np.array(t)@cell.T for t in shifts])
    tree=cKDTree(translated)
    radii=np.array([gemmi.Element(a['element']).covalent_r for a in sites])
    bonds=[];seen=set();ghosts={}
    atoms=[{'id':i,'label':a['label'],'element':a['element'],'position':clean(a['cart']),
            'fractional':clean(a['frac']),'occupancy':a['occupancy'],
            'uCart':clean(a['u_cart']),'color':COLORS.get(a['element'],'#ac9272')}
           for i,a in enumerate(sites)]
    from scipy.stats import chi2
    for atom in atoms:
        values,axes=np.linalg.eigh(np.asarray(atom['uCart']))
        atom['adpValid']=bool(np.min(values)>=-1e-12)
        if np.min(values)>=-1e-12:
            if np.linalg.det(axes)<0:axes[:,2]*=-1
            atom['ellipsoid']={'radii':clean(np.sqrt(np.maximum(values,0)*chi2.ppf(.5,3))),
                               'axes':clean(axes),'probability':.5}
        else:atom['adpWarning']='NPD：位移参数非正定，不能绘制概率热椭球；当前以球棍显示。'
    for i,p in enumerate(points):
        for index in tree.query_ball_point(p,4.5):
            image,j=divmod(index,len(sites));q=translated[index]
            distance=float(np.linalg.norm(q-p))
            if distance<.4 or distance>1.2*(radii[i]+radii[j]):continue
            key=tuple(sorted((tuple(np.round(p,6)),tuple(np.round(q,6)))))
            if key in seen:continue
            seen.add(key)
            if shifts[image]!=(0,0,0):
                ghost_key=(j,shifts[image])
                if ghost_key not in ghosts:
                    ghosts[ghost_key]=len(atoms)
                    atoms.append({**atoms[j],'id':len(atoms),'position':clean(q),'periodicImage':list(shifts[image])})
                end=ghosts[ghost_key]
            else:end=j
            bonds.append([i,end])
            if len(bonds)>=10000:break
        if len(bonds)>=10000:break
    composition=dict(collections.Counter(a['element'] for a in sites))
    return {'cell':s.cell.to_dict(),'matrix':clean(cell),'spacegroup':s.spacegroup,'compatibility':compatibility,
            'volume':s.cell.volume,'atoms':atoms,'bonds':bonds,
            'cellEdges':clean(unit_cell_edges(s.cell)),'composition':composition,
            'asymmetricAtoms':len(s.atoms),'unitCellAtoms':len(sites),
            'bondConvention':'共价半径距离接触，仅用于显示；不推断键级。'}


def reflection_view(project,folder):
    selection=set(project.get('selection',[]));q=None;experiments=None;domains=None;panels=None
    experiment_ids=None;imageset_ids=None;frame_paths=[];indexed=None
    if project.get('reflections') and not project.get('dials_reflections'):
        r=read_reflections(folder/project['reflections']);hkl=r.hkl;intensity=r.intensity;sigma=r.sigma
        frame=None;xy=None
    elif project.get('dials_reflections'):
        from dials.array_family import flex
        table=flex.reflection_table.from_file(str(folder/project['dials_reflections']))
        n=len(table)
        hkl=np.asarray(table['miller_index']) if 'miller_index' in table else np.zeros((n,3),int)
        intensity=np.full(n,np.nan);sigma=np.full(n,np.nan)
        for field in ('intensity.sum.value','intensity.prf.value','intensity.scale.value'):
            if field not in table:continue
            values=np.asarray(table[field]);variance=field.replace('.value','.variance')
            variances=np.asarray(table[variance]) if variance in table else np.full(n,np.nan)
            valid=np.isfinite(values)&np.isfinite(variances)&(variances>0)
            intensity[valid]=values[valid];sigma[valid]=np.sqrt(variances[valid])
        xyz=np.asarray(table['xyzobs.px.value']) if 'xyzobs.px.value' in table else None
        xy=xyz[:,:2] if xyz is not None else None;frame=np.floor(xyz[:,2]).astype(int) if xyz is not None else None
        panels=np.asarray(table['panel']) if 'panel' in table else None
        experiment_ids=np.asarray(table['id']) if 'id' in table else None
        imageset_ids=np.asarray(table['imageset_id']) if 'imageset_id' in table else experiment_ids
        indexed=np.any(hkl!=0,axis=1)
        if project.get('experiments'):
            from dxtbx.model.experiment_list import ExperimentListFactory
            experiments=ExperimentListFactory.from_json_file(str(folder/project['experiments']),check_format=False)
            if xyz is not None and panels is not None:
                table.centroid_px_to_mm(experiments)
                table.map_centroids_to_reciprocal_space(experiments)
            crystals=list(experiments.crystals())
            domains=[crystals.index(experiments[int(i)].crystal) if 0<=i<len(experiments) and experiments[int(i)].crystal is not None else None for i in experiment_ids] if experiment_ids is not None else None
            # Image centroids use scan array indices; the project may contain multiple sweeps.
            for imageset in experiments.imagesets():
                scan=imageset.get_scan();start=scan.get_array_range()[0] if scan else 0
                frame_paths.append((start,[str(Path(imageset.get_path(i)).resolve()) for i in range(len(imageset))]))
        if 'rlp' in table:q=np.asarray(table['rlp'])
    else:return {'reflections':[],'total':0,'shown':0}
    cell=project.get('cell')
    metric=q is not None or bool(cell)
    if q is None:q=hkl@Cell(**cell).reciprocal.T if cell else np.asarray(hkl,dtype=float)
    path_to_frame={str((folder/path).resolve()):i for i,path in enumerate(project.get('frames',[]))}
    limit=min(len(hkl),50000)
    data=[]
    for i in range(limit):
        length=float(np.linalg.norm(q[i]));d=1/length if metric and length>0 else None
        row={'id':i,'hkl':hkl[i].tolist(),'q':q[i].tolist(),'d':d,'intensity':float(intensity[i]),
             'sigma':float(sigma[i]),'selected':i in selection}
        if indexed is not None:row['indexed']=bool(indexed[i])
        if xy is not None:
            source_frame=int(frame[i])
            if frame_paths and imageset_ids is not None:
                image_id=int(imageset_ids[i])
                if 0<=image_id<len(frame_paths):
                    start,paths=frame_paths[image_id];local=source_frame-start
                    source_frame=path_to_frame.get(paths[local],-1) if 0<=local<len(paths) else -1
            row.update(x=float(xy[i][0]),y=float(xy[i][1]))
            if source_frame>=0:row['frame']=source_frame
        if panels is not None:row['panel']=int(panels[i])
        if domains is not None and domains[i] is not None:row['domain']=int(domains[i])
        data.append(row)
    values=[r['d'] for r in data if r['d'] is not None]
    return clean({'reflections':data,'total':len(hkl),'shown':limit,'reciprocalUnit':'Å⁻¹' if metric else 'hkl',
                  'resolution':{'dMin':min(values),'dMax':max(values),'count':len(values),'shownOnly':limit<len(hkl)} if values else None,
                  'negativeCount':int(np.sum(intensity<0))})


def detector_geometry(panel,beam,probe=None):
    s0=beam.get_s0();width,height=panel.get_image_size()
    if probe is not None:
        x,y=probe
        if not 0<=x<=width or not 0<=y<=height:raise ValueError('像素位置在当前探测器面板之外。')
        return clean({'x':x,'y':y,'d':panel.get_resolution_at_pixel(s0,(x,y))})
    wavelength=beam.get_wavelength();axis=np.asarray(s0,float);axis/=np.linalg.norm(axis)
    reference=np.asarray(panel.get_fast_axis(),float);u=reference-axis*np.dot(reference,axis);u/=np.linalg.norm(u);v=np.cross(axis,u)
    minimum=panel.get_max_resolution_at_corners(s0);rings=[]
    for d in [x for x in (.4,.5,.6,.8,1.,1.5,2.,3.,5.,8.,12.,20.,40.) if x>=minimum and x>wavelength/2][:6]:
        angle=2*math.asin(wavelength/(2*d));segments=[];points=[];visible=[]
        for phi in np.linspace(0,2*math.pi,721):
            ray=axis*math.cos(angle)+(u*math.cos(phi)+v*math.sin(phi))*math.sin(angle)
            try:
                p=panel.millimeter_to_pixel(panel.get_ray_intersection(tuple(ray)))
            except RuntimeError:
                if len(points)>1:segments.append(points)
                points=[];continue
            if not all(math.isfinite(x) for x in p):continue
            points.append(list(p))
            if 24<p[0]<width-80 and 24<p[1]<height-24:visible.append(list(p))
        if len(points)>1:segments.append(points)
        if visible:rings.append({'d':d,'segments':segments,'label':min(visible,key=lambda p:(p[0]-.75*width)**2+(p[1]-.32*height)**2)})
    return clean({'wavelength':wavelength,'size':[width,height],'beamCenter':panel.get_beam_centre_px(s0),
                  'geometricDMin':minimum,'rings':rings,'panel':0})


def detector_view(project,folder,index=0,probe=None):
    paths=project.get('frames',[])
    if not 0<=index<len(paths):raise ValueError('请先选择一张原始衍射帧。')
    path=(folder/paths[index]).resolve();experiment=None
    if project.get('experiments'):
        from dxtbx.model.experiment_list import ExperimentListFactory
        experiments=ExperimentListFactory.from_json_file(str(folder/project['experiments']),check_format=False)
        experiment=next((e for e in experiments if e.imageset is not None and any(Path(e.imageset.get_path(i)).resolve()==path for i in range(len(e.imageset)))),None)
    if experiment is not None:panel,beam=experiment.detector[0],experiment.beam
    else:
        import dxtbx
        image=dxtbx.load(str(path));panel,beam=image.get_detector()[0],image.get_beam()
    return detector_geometry(panel,beam,probe)


def frame_array(project,folder,index):
    paths=project.get('frames',[])
    if not paths:raise ValueError('请先导入原始衍射帧。')
    if not 0<=index<len(paths):raise ValueError('帧序号超出数据范围。')
    import dxtbx
    image=dxtbx.load(str(folder/paths[index]));raw=image.get_raw_data()
    if isinstance(raw,tuple):raw=raw[0]
    return np.asarray(raw.as_numpy_array() if hasattr(raw,'as_numpy_array') else raw,dtype=float)


def pixel_profile(project,folder,reflection=None,frame=0):
    if project.get('pixels'):
        p=json.loads((folder/project['pixels']).read_text(encoding='utf-8'));values=np.asarray(p['data'],float)
        result={'x':list(range(len(values))),'observed':values.tolist(),'label':'声明的校准像素积分问题','source':'pixels'}
        profiles=np.asarray(p['profiles'],float)
        result['components']=profiles.T.tolist()
        return clean(result)
    if not project.get('frames'):return None
    if not reflection or 'x' not in reflection:return None
    frame=reflection.get('frame',frame)
    data=frame_array(project,folder,frame)
    x,y=int(reflection['x']),int(reflection['y']);radius=10
    low,high=max(0,x-radius),min(data.shape[1],x+radius+1)
    lo_y,hi_y=max(0,y-2),min(data.shape[0],y+3)
    return clean({'x':list(range(low,high)),'observed':data[lo_y:hi_y,low:high].sum(axis=0).tolist(),
                  'label':f'第 {frame+1} 帧 · 反射附近 5 像素宽条带的计数和','source':'frame','frame':frame,
                  'unit':'原始计数 / ADU（由探测器决定）'})


def run_native(action,project,folder,output,settings):
    from ariadne_xtal.refinement import RefinementOptions,refine
    from .structure_input import native_model
    needs_structure=action in {'refine','maps','symmetry','absolute','joint-refine','solve-observations'}
    needs_reflections=action in {'refine','maps','symmetry','absolute','solve','autosolve'}
    s=native_model(folder/project['structure']) if needs_structure and project.get('structure') else None
    r=read_reflections(folder/project['reflections']) if needs_reflections and project.get('reflections') else None
    options=settings.get('options',{})
    if not isinstance(options,dict):raise ValueError('options 必须是参数对象。')
    if options.get('max_nfev',200)>5000:raise ValueError('本地单次精修最多允许 5000 次目标求值。')
    if action in {'refine','maps','symmetry','absolute'} and (s is None or r is None):raise ValueError('该操作需要结构文件和反射数据。')
    result={}
    if action=='inspect':
        result={'structure':structure_view(folder/project['structure'])}
    elif action=='refine':
        if s is None or r is None:raise ValueError('结构精修需要结构文件和反射数据。')
        options.setdefault('max_nfev',200)
        fit=refine(s,r,RefinementOptions(**options),settings.get('restraints'),settings.get('constraints'))
        result=fit.to_dict();s=fit.structure
    elif action in {'solve','autosolve'}:
        if r is None:raise ValueError('结构求解需要反射数据。')
        cell=Cell(**project['cell']);elements=project.get('elements',[])
        if not elements:raise ValueError('请提供整个晶胞的元素组成。')
        if action=='autosolve':
            from ariadne_xtal.autonomy import solve_refine
            result=solve_refine(cell,r,elements,phasing_options=settings.get('phasing_options',{'trials':4,'iterations':300}),
                                refinement_options=settings.get('refinement_options',{'max_nfev':200}),
                                spacegroup_candidates=settings.get('spacegroup_candidates'),options=options)
            if result.get('refinement',{}).get('structure'):s=Structure.from_dict(result['refinement']['structure'])
        else:
            from ariadne_xtal.phasing import PhasingOptions,solve_p1
            screening = settings.get('screening')
            if screening is None:
                fit = solve_p1(cell, r, elements, PhasingOptions(**{'trials':4,'iterations':300,**options}))
                candidates, result = fit.candidates, fit.to_dict()
            else:
                from ariadne_xtal.screened_phasing import screened_phase_search

                if not isinstance(screening, dict):
                    raise ValueError('screening 必须是参数对象。')
                phase_options = PhasingOptions(**{'algorithm':'hybrid','trials':12,'iterations':300,**options})
                if phase_options.trials * phase_options.iterations > 20_000:
                    raise ValueError('工作台相位求解预算为 20,000 次合计迭代。')
                result = screened_phase_search(cell, r, elements, phase_options, **screening)
                candidates = result['candidates']
                for field in ('candidates', 'audited_rejections'):
                    result[field] = [{**item, 'structure': item['structure'].to_dict()}
                                     for item in result[field]]
            for i,candidate in enumerate(candidates):
                write_structure(candidate['structure'],output/f'candidate-{i+1}.cif')
            if candidates:
                s = candidates[0]['structure']
    elif action=='integrate-pixels':
        from ariadne_xtal.reduction_bridge import integrate_problem
        result=integrate_problem(json.loads((folder/project['pixels']).read_text(encoding='utf-8')))
    elif action in {'joint-refine','solve-observations'}:
        groups=json.loads((folder/project['observations']).read_text(encoding='utf-8'));groups=groups.get('groups',groups) if isinstance(groups,dict) else groups
        if action=='joint-refine':
            from ariadne_xtal.joint_refinement import refine_observations
            result=refine_observations(s,groups,options,**{k:settings[k] for k in ('domain_fractions','refine_domain_fractions','phase_relation','constraints','restraints') if k in settings})
            if result.get('structure'):s=Structure.from_dict(result['structure'])
        else:
            from ariadne_xtal.evidence_solver import solve_observations
            result=solve_observations(Cell(**project['cell']),project['elements'],groups,initial_model=s,
                                      phasing_options=settings.get('phasing_options'),refinement_options=settings.get('refinement_options'),options=options)
            if result.get('refinement',{}).get('structure'):s=Structure.from_dict(result['refinement']['structure'])
    elif action=='symmetry':
        from ariadne_xtal.spacegroup_search import propose_spacegroups
        proposals=propose_spacegroups(s,r.hkl,settings.get('candidates',[s.spacegroup]),
                                      tolerance=settings.get('tolerance',.18),phase_limit=settings.get('phase_limit',.08))
        result={'proposals':[]}
        for i,p in enumerate(proposals):
            if isinstance(p.get('structure'),Structure):
                write_structure(p['structure'],output/f'symmetry-{i+1}.cif');p={**p,'structure':p['structure'].to_dict()}
            result['proposals'].append(p)
    elif action=='absolute':
        from ariadne_xtal.absolute import friedel_quotient
        result=friedel_quotient(s,r,**options)
    elif action=='maps':
        from ariadne_xtal.maps import difference_map,patterson_map,model_map
        from ariadne_xtal.visualization import density_surface_mesh
        for name,density in [('difference',difference_map(s,r)),('patterson',patterson_map(s.cell,r,structure=s)),('calculated',model_map(s,r.hkl))]:
            density.save_npz(output/f'{name}.npz');density.save_ccp4(output/f'{name}.ccp4')
            if name=='difference':
                sd=float(np.std(density.values));surfaces=[]
                for sign in (1,-1):
                    level=sign*3*sd
                    if float(np.min(density.values))<level<float(np.max(density.values)):
                        vertices,faces=density_surface_mesh(density,level)
                        surfaces.append({'vertices':clean(vertices),'faces':clean(faces),'sign':sign})
                dump(output/'surface.json',{'surfaces':surfaces,'levelSigma':3,'type':'unweighted Fo-Fc'})
        result={'maps':['difference','patterson','calculated'],'convention':'未加权 Fo-Fc；等值面为 ±3σ。'}
    else:raise ValueError('未注册的原生动作。')
    if s is not None and action in {'refine','solve','autosolve','joint-refine','solve-observations'}:
        write_structure(s,output/'structure.cif');write_structure(s,output/'structure.json')
    dump(output/'result.json',result)
    return clean(result)
