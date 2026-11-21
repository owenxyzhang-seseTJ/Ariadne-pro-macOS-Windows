"""Local, geometry-preserving map display. The input grid is never modified."""
from functools import lru_cache
from pathlib import Path
import math

import gemmi
import numpy as np

from ariadne_xtal.maps import DensityMap,load_map
from ariadne_xtal.model import Cell
from ariadne_xtal.visualization import density_surface_mesh,unit_cell_edges


MAP_NAMES={'difference':'差值密度 · Fo−Fc','calculated':'计算密度 · Fc','patterson':'Patterson 图'}
MAP_EXTENSIONS={'.ccp4','.map'}


def map_catalog(project,folder):
    maps=[]
    if project.get('surface'):
        parent=(folder/project['surface']).resolve().parent
        if parent.is_relative_to(folder.resolve()):
            for key,name in MAP_NAMES.items():
                source=parent/f'{key}.npz'
                if source.is_file():maps.append({'id':key,'name':name,'source':source.relative_to(folder.resolve()).as_posix(),'download':source.with_suffix('.ccp4').relative_to(folder.resolve()).as_posix() if source.with_suffix('.ccp4').is_file() else None,'signed':key=='difference'})
    for entry in project.get('files',[]):
        path=(folder/entry['path']).resolve()
        if path.is_relative_to(folder.resolve()) and path.is_file() and path.suffix.lower() in MAP_EXTENSIONS:
            maps.append({'id':'file:'+entry['path'],'name':entry['name'],'source':entry['path'],'download':entry['path'],'signed':True})
    return maps


@lru_cache(maxsize=8)
def contour_map(path,mtime,size,sigma,signed):
    path=Path(path)
    origin=np.zeros(3)
    if path.suffix=='.npz':density=load_map(path)
    else:
        ccp4=gemmi.read_ccp4_map(str(path),setup=False)
        origin=np.asarray([ccp4.header_float(i) for i in (50,51,52)])
        if not np.isfinite(origin).all():raise ValueError('密度图的原点坐标无效。')
        if ccp4.grid.point_count>16_777_216:raise ValueError('密度网格过大；请导入不超过 16777216 个体素的显示图。')
        # Gemmi restores axis order, grid start and symmetry to the full cell.
        # Unknown samples remain NaN rather than becoming invented zero density.
        ccp4.setup(float('nan'),gemmi.MapSetup.Full)
        values=np.asarray(ccp4.grid)
        if not np.isfinite(values).all():raise ValueError('密度图含未采样区域或非有限值；请导入完整晶胞的 CCP4 / MAP 网格。')
        cell=ccp4.grid.unit_cell
        density=DensityMap(np.array(values,dtype=float),Cell(*cell.parameters),'imported','map units')
    values=np.asarray(density.values,float)
    if values.size>16_777_216:raise ValueError('密度网格超过当前显示上限。')
    if min(values.shape)<2:raise ValueError('密度图每个维度至少需要 2 个采样点。')
    mean=float(values.mean());sd=float(values.std());original_shape=list(values.shape)
    if values.size>524_288:
        from scipy.ndimage import map_coordinates
        scale=(values.size/524_288)**(1/3)
        shape=np.maximum(2,np.floor(np.asarray(values.shape)/scale).astype(int))
        coords=np.meshgrid(*(np.arange(n)*old/n for n,old in zip(shape,values.shape)),indexing='ij')
        values=map_coordinates(values,np.asarray(coords),order=1,mode='grid-wrap',prefilter=False)
    display=DensityMap(values,density.cell,density.kind,density.units)
    surfaces=[]
    if sd>0:
        for sign in ((1,-1) if signed else (1,)):
            level=mean+sign*sigma*sd
            if values.min()<level<values.max():
                vertices,faces=density_surface_mesh(display,level)
                surfaces.append({'vertices':(vertices+origin).tolist(),'faces':faces.tolist(),'sign':sign,'level':level})
    return {'surfaces':surfaces,'levelSigma':sigma,'cell':density.cell.to_dict(),'cellEdges':(np.asarray(unit_cell_edges(density.cell))+origin).tolist(),
            'statistics':{'min':float(density.values.min()),'max':float(density.values.max()),'mean':mean,'sigma':sd,
                          'grid':original_shape,'displayGrid':list(values.shape),'units':density.units},
            'normalization':'(density − map mean) / map standard deviation'}


def density_view(project,folder,map_id=None,sigma=3.):
    if not math.isfinite(sigma) or not .5<=sigma<=8:raise ValueError('等值面范围为 0.5–8σ。')
    maps=map_catalog(project,folder)
    if not maps:return {'surfaces':[],'availableMaps':[]}
    selected=next((item for item in maps if item['id']==map_id),None) if map_id else maps[0]
    if selected is None:raise ValueError('所选密度图不在当前项目中。')
    path=(folder/selected['source']).resolve();stat=path.stat()
    result=contour_map(str(path),stat.st_mtime_ns,stat.st_size,round(sigma,2),selected['signed'])
    return {**result,'mapId':selected['id'],'type':selected['name'],'source':selected['source'],'availableMaps':maps}
