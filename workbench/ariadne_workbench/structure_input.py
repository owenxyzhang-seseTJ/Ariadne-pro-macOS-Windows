"""Read-only SHELX display, kept separate from strict native computation."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ariadne_xtal.io import read_structure
from ariadne_xtal.model import Cell
from ariadne_xtal.shelx import NativeConversionError,ShelxDocument


@dataclass(frozen=True)
class _PreviewAtom:
    label: str
    element: str
    frac: np.ndarray
    occupancy: float
    u_iso: float
    u_aniso: np.ndarray | None


@dataclass(frozen=True)
class _PreviewModel:
    """Display data only: invalid ADPs remain intact, never a native model."""
    cell: Cell
    atoms: list[_PreviewAtom]
    spacegroup: str
    wavelength: float
    operations: list


@lru_cache(maxsize=12)
def _display_model(path,mtime,size):
    path=Path(path)
    try:
        return read_structure(path),{'nativeCompatible':True,'displayOnly':False,'issues':[]}
    except NativeConversionError as error:
        # CCTBX reads atom coordinates, occupancies, ADPs and symmetry without
        # executing L.S./ANIS/ACTA/FMAP/PLAN or changing any instruction bytes.
        issues=error.issues
    if path.suffix.lower() not in {'.ins','.res'}:raise ValueError('不能读取该结构文件。')
    document=ShelxDocument.from_path(path)
    if any(x.kind=='include' and x.active for x in document.instructions):
        raise ValueError('SHELX 结构包含外部指令文件；请使用完整 CIF 预览，原始 INS/RES 交由 SHELXL。')
    from cctbx import xray,adptbx
    try:parsed=xray.structure.from_shelx(filename=str(path),strictly_shelxl=False)
    except Exception as error:raise ValueError(f'SHELX 结构预览未成功：{error}') from error
    cell=Cell(*parsed.unit_cell().parameters());atoms=[];npd=[]
    for site in parsed.scatterers():
        tensor=None
        if site.flags.use_u_aniso():
            u11,u22,u33,u12,u13,u23=adptbx.u_star_as_u_cart(parsed.unit_cell(),site.u_star)
            tensor=np.array([[u11,u12,u13],[u12,u22,u23],[u13,u23,u33]])
        u_iso=float(np.trace(tensor)/3) if tensor is not None else site.u_iso
        if (tensor is not None and np.linalg.eigvalsh(tensor).min()<-1e-12) or (tensor is None and u_iso<0):npd.append(site.label)
        atoms.append(_PreviewAtom(site.label,site.element_symbol(),np.asarray(site.site),site.occupancy,u_iso,tensor))
    if not atoms:raise ValueError('该 SHELX 文件没有可显示的原子。')
    cell_record=next((x for x in document.instructions if x.active and x.keyword=='CELL'),None)
    if cell_record is None:raise ValueError('SHELX 文件缺少 CELL 波长与晶胞参数。')
    wavelength=float(cell_record.arguments[0])
    from ariadne_xtal.symmetry import get_operations
    spacegroup=str(parsed.space_group_info())
    operations=get_operations(spacegroup,[operation.as_xyz() for operation in parsed.space_group().all_ops()],cell)
    model=_PreviewModel(cell,atoms,spacegroup,wavelength,operations)
    return model,{'nativeCompatible':False,'displayOnly':True,'issues':issues,
                  'message':'SHELX 结构可只读预览；原始指令保留。原生精修不能执行这些指令，请使用 SHELXL。',
                  'npdAtoms':npd,'scientificWarnings':[f'{len(npd)} 个 NPD 原子（{", ".join(npd)}）；保留原始 U，以球棍显示。'] if npd else [],
                  'sourceFormat':path.suffix.lower().lstrip('.')}


def display_model(path):
    path=Path(path).resolve();stat=path.stat()
    return _display_model(str(path),stat.st_mtime_ns,stat.st_size)


def native_model(path):
    """Computation stays strict; display compatibility never enables refinement."""
    try:return read_structure(path)
    except NativeConversionError as error:
        raise ValueError('该 SHELX 文件可预览，但含原生引擎不支持的指令。请使用 SHELXL 接口处理精修；'
                         '原始文件未被删改。详情：\n'+'\n'.join(error.issues)) from error
