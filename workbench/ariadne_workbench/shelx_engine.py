"""Installed SHELX programs and byte-preserving, project-local input staging."""
from pathlib import Path
import os
import re
import shutil

from ariadne_xtal.shelx import ShelxDocument


def engine_info(engine):
    if engine not in {'shelxt','shelxl'}:raise ValueError('未知 SHELX 引擎。')
    variable=f'ARIADNE_{engine.upper()}_PATH'
    configured=os.environ.get(variable)
    found=configured or shutil.which(engine)
    path=Path(found).expanduser() if found else None
    available=bool(path and path.is_absolute() and path.is_file() and os.access(path,os.X_OK))
    return {'available':available,'executable':str(path.resolve()) if available else None,
            'source':'environment' if configured else 'PATH',
            'reason':None if available else f'{variable} 配置无效。' if configured else f'未找到 {engine.upper()}；请安装官方程序并配置 {variable}。'}


def input_plan(project,folder,settings,engine):
    """Resolve an explicit project pair; never search other folders by basename."""
    folder=Path(folder).resolve()
    def local(name):
        if not isinstance(name,str) or not name:raise ValueError('请选择 SHELX 结构输入。')
        path=(folder/name).resolve()
        if not path.is_relative_to(folder) or not path.is_file():raise ValueError('SHELX 输入必须是当前项目内的文件。')
        return path
    original=local(settings.get('shelx_structure') or project.get('structure'))
    if original.suffix.lower() not in {'.ins','.res'}:raise ValueError('SHELX 接口需要原始 INS / RES 文件。')
    root=original.parent
    stem=original.stem if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}',original.stem) else 'model'
    files={Path(stem+'.ins'):original};active=set();visited=set();instructions=[]
    def add(relative,source):
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('SHELX 附件路径不能离开结构输入目录。')
        source=source.resolve()
        if not source.is_relative_to(root) or not source.is_file():raise ValueError(f'SHELX 附件缺失或越界：{relative}')
        if relative in files and files[relative]!=source:raise ValueError(f'SHELX 附件目标重名：{relative}')
        files[relative]=source
        return source
    def visit(source):
        if source in active:raise ValueError('SHELX include 存在循环引用。')
        if source in visited:return
        if len(visited)+len(active)>=1024:raise ValueError('SHELX include 文件过多。')
        active.add(source)
        document=ShelxDocument.from_path(source)
        instructions.extend(x for x in document.instructions if x.active)
        for item in document.instructions:
            if item.kind!='include' or not item.active:continue
            name=item.logical_text.lstrip('+').strip()
            if not name or '\x00' in name:raise ValueError('SHELX include 文件名无效。')
            relative=Path(name);visit(add(relative,root/relative))
        active.remove(source);visited.add(source)
    visit(original)
    needs_hkl=engine=='shelxt' or not any(x.keyword=='HKLF' and x.arguments and x.arguments[0].startswith('-') for x in instructions)
    reflections=None
    if needs_hkl:
        selected=settings.get('shelx_reflections') or project.get('reflections')
        if selected:reflections=local(selected)
        else:
            companion=original.with_suffix('.hkl')
            if companion.is_file():reflections=local(str(companion.relative_to(folder)))
        if reflections is None or reflections.suffix.lower()!='.hkl':raise ValueError('SHELXL / SHELXT 需要 HKL 文件；请在输入配对中选择当前项目的 HKL。')
        target=Path(stem+'.hkl')
        if target in files and files[target]!=reflections:raise ValueError('HKL 与 SHELX include 目标重名。')
        files[target]=reflections
    fab=original.with_suffix('.fab')
    if fab.exists():add(Path(stem+'.fab'),fab)
    elif any(x.keyword=='ABIN' for x in instructions):raise ValueError(f'ABIN 需要结构旁的 {original.stem}.fab；请补充该文件。')
    extra=settings.get('extra_args',[])
    if not isinstance(extra,list) or any(not isinstance(x,str) or '\x00' in x for x in extra):raise ValueError('额外选项必须是字符串列表。')
    return {'original':original,'reflections':reflections,'stem':stem,'files':list(files.items()),'instruction_files':visited,'extra_args':extra}
