"""为平行版本安装可选 DIALS 环境并建立独立插件层。"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

root=Path(__file__).resolve().parent.parent
if sys.platform!='win32':
    raise SystemExit('请使用原生 Windows Python。')
python=root/'dials-env/python.exe'
if not python.exists():
    tool=root/'micromamba25/Library/bin/micromamba.exe'
    if not tool.exists():
        archive=root/'downloads/micromamba-2.5.0-1.tar.bz2'
        archive.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve('https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/win-64/micromamba-2.5.0-1.tar.bz2',archive)
        with tarfile.open(archive) as package:
            for item in package.getmembers():
                if item.isfile() and (item.name=='Library/bin/micromamba.exe' or item.name.startswith('info/licenses/')):
                    destination=(root/'micromamba25'/item.name).resolve()
                    if not destination.is_relative_to((root/'micromamba25').resolve()):
                        raise ValueError('归档路径越界')
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    destination.write_bytes(package.extractfile(item).read())
    cache=Path(os.environ.get('ARIADNE_DIALS_CACHE',os.environ.get('SystemDrive','C:')+'\\apw25'))
    cache.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env['CONDA_PKGS_DIRS']=str(cache/'pkgs')
    subprocess.run([str(tool),'--no-rc','--root-prefix',str(cache),'create','--prefix',str(root/'dials-env'),'--override-channels','--channel','https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge','--yes','python=3.12','dials=3.30.1','dxtbx=3.30.0'],env=env,check=True)
overlay=root/'dials-overlay/src/dials'
if not overlay.exists():
    shutil.copytree(root/'dials-env/Lib/site-packages/dials',overlay)
with zipfile.ZipFile(root/'tools/dials-pro-overlay.zip') as package:
    for item in package.infolist():
        destination=(root/'dials-overlay'/item.filename).resolve()
        if not destination.is_relative_to((root/'dials-overlay').resolve()):
            raise ValueError('插件路径越界')
        if not item.is_dir():
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(package.read(item))
print('DIALS 运行时与独立 Ariadne 插件层已就绪。')
