"""Map-only SHELXL recipes and Fourier maps from official LIST 6 output."""
from pathlib import Path
import shutil

import gemmi
import numpy as np

from ariadne_xtal.maps import DensityMap
from ariadne_xtal.model import Cell
from ariadne_xtal.shelx import ShelxDocument


def prepare_map_input(plan,out):
    """Change only staged cycle/list controls, retaining exact source copies."""
    source=plan['original'];main=ShelxDocument.from_path(source)
    hkl=next((x for x in main.instructions if x.active and x.keyword=='HKLF'),None)
    if hkl is None:raise ValueError('SHELXL 密度计算需要主 INS / RES 中的 HKLF 指令。')
    changes=[]
    for relative,original in plan['files']:
        if original not in plan['instruction_files']:continue
        staged=out/relative;archive=out/'source_instructions'/relative
        archive.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(staged,archive)
        document=ShelxDocument.from_path(staged);lines=document.raw_text.splitlines(keepends=True)
        for item in document.instructions:
            # ACTA requires least-squares cycles in SHELXL. Map-only runs use
            # LIST 6 rather than ACTA CIF/report generation; record this change.
            if not item.active or item.keyword not in {'L.S.','CGLS','LIST','ACTA'}:continue
            start=item.line_number-1;end=start
            while '=' in lines[end].split('!',1)[0]:end+=1
            for i in range(start,end+1):lines[i]='REM map-only replaced: '+lines[i]
            changes.append({'file':str(relative),'line':item.line_number,'original':item.logical_text,'change':'commented in map-only copy'})
        if original==source:
            newline='\r\n' if '\r\n' in document.raw_text else '\n'
            lines.insert(hkl.line_number-1,'L.S. 0'+newline+'LIST 6'+newline)
        staged.write_bytes(''.join(lines).encode('utf-8',errors='surrogateescape'))
    return {'purpose':'structure-factor calculation only; no least-squares cycles',
            'added':['L.S. 0','LIST 6'],'changes':changes,'original_copies':'source_instructions/'}


def maps_from_fcf(path,out):
    document=gemmi.cif.read_file(str(path))
    if len(document)!=1:raise ValueError('SHELXL FCF 必须包含一个反射数据块。')
    block=document.sole_block()
    # SHELX uses core-CIF tags, not the mmCIF categories of ReflnBlock.
    cell=Cell(*(gemmi.cif.as_number(block.find_value('_cell_'+tag) or '?') for tag in
                ('length_a','length_b','length_c','angle_alpha','angle_beta','angle_gamma')))
    operations=list(block.find_values('_space_group_symop_operation_xyz')) or list(block.find_values('_symmetry_equiv_pos_as_xyz'))
    if operations:group=gemmi.find_spacegroup_by_ops(gemmi.GroupOps([gemmi.Op(gemmi.cif.as_string(op)) for op in operations]))
    else:
        name=block.find_value('_space_group_name_H-M_alt') or block.find_value('_symmetry_space_group_name_H-M')
        group=gemmi.find_spacegroup_by_name(gemmi.cif.as_string(name)) if name else None
    if group is None:raise ValueError('SHELXL FCF 缺少可识别的空间群。')
    multiplier=gemmi.cif.as_number(block.find_value('_shelx_F_squared_multiplier') or '1')
    if multiplier!=1:raise ValueError('该 FCF 使用非单位 F² 缩放因子；请先提供已换算的 LIST 6 或 CCP4，避免误报电子密度单位。')
    table=block.find('_refln_',['index_h','index_k','index_l','F_squared_meas','F_calc','phase_calc'])
    if not len(table):raise ValueError('SHELXL FCF 缺少 LIST 6 的振幅与相位列。')
    array=np.asarray([[gemmi.cif.as_number(value) for value in row] for row in table],dtype=float)
    if not np.isfinite(array).all() or not np.equal(array[:,:3],np.rint(array[:,:3])).all():raise ValueError('SHELXL FCF 的反射索引或数值无效。')
    hkl=np.asarray(array[:,:3],dtype=np.int32);observed,fc,phase=array[:,3:].T
    if not len(hkl) or not all(np.isfinite(a).all() for a in (observed,fc,phase)):raise ValueError('SHELXL LIST 6 的结构因子/相位数据缺失或非有限。')
    if np.any(fc<0):raise ValueError('SHELXL FCF 的计算振幅不能为负。')
    phase_factor=np.exp(1j*np.deg2rad(phase));fo=np.sqrt(np.maximum(observed,0))
    coefficients={'difference':(fo-fc)*phase_factor,'calculated':fc*phase_factor,'patterson':observed.astype(complex)}
    unit_cell=gemmi.UnitCell(cell.a,cell.b,cell.c,cell.alpha,cell.beta,cell.gamma);counts={}
    for name,values in coefficients.items():
        data=gemmi.ComplexAsuData(unit_cell,group,hkl,np.asarray(values,dtype=np.complex64))
        data.ensure_asu();size=data.get_size_for_hkl(min_size=[16,16,16],sample_rate=3.)
        if np.prod(size)>16_777_216:raise ValueError('SHELXL 密度网格超过当前显示上限。')
        grid=data.transform_f_phi_to_map(exact_size=size)
        density=DensityMap(np.array(grid,dtype=float),cell,name,'e^2/angstrom^3' if name=='patterson' else 'e/angstrom^3',
                           {'source':'SHELXL LIST 6','fcf':Path(path).name,'weighting':'unweighted',
                            'negative_F_squared':'clipped only for Fo amplitude; retained in Patterson'})
        density.save_npz(out/f'{name}.npz');density.save_ccp4(out/f'{name}.ccp4');counts[name]=list(density.values.shape)
    return {'source':'SHELXL LIST 6','fcf':Path(path).name,'reflections':len(hkl),'maps':counts,
            'convention':'unweighted (sqrt(max(Fo²,0))−Fc) exp(iφc); Fc exp(iφc); Patterson Fo². No sigmaA weighting.'}
