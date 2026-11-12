"""用独立 DIALS 解释器加载 Ariadne 插件并运行 DIALS 命令。"""
import os
from pathlib import Path
import runpy
import sys
root=Path(__file__).resolve().parent.parent
dll_handle=os.add_dll_directory(str(Path(sys.prefix)/'Library/bin'))
for path in reversed([root/'source/python/src',root/'source/native/src',root/'dials-overlay/src',root/'overlay']):
    sys.path.insert(0,str(path))
if len(sys.argv)<2:
    print('Usage: ariadne-pro-dials.cmd <find_spots|index|integrate|scale|...> [arguments]')
    raise SystemExit(0)
command=sys.argv.pop(1)
if command=='python':
    if len(sys.argv)>2 and sys.argv[1]=='-c':
        code=sys.argv[2]
        sys.argv=sys.argv[2:]
        exec(compile(code,'<command>','exec'),{'__name__':'__main__'})
    elif len(sys.argv)>1:
        script=sys.argv[1]
        sys.argv=sys.argv[1:]
        runpy.run_path(script,run_name='__main__')
    else:
        raise SystemExit('Supply a script path or -c expression.')
else:
    if not command.replace('_','').isalnum():
        raise SystemExit('Invalid DIALS command name.')
    runpy.run_module('dials.command_line.'+command,run_name='__main__')
