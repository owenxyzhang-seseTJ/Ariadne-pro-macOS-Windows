"""Desktop double-click entry for the local workbench."""
from pathlib import Path
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import json
import webbrowser
import uvicorn


def ours(port, source=None):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/capabilities',timeout=.5) as response:
            payload = json.load(response)
            runtime = payload.get('runtime', {})
            return (runtime.get('workflow') == 'ariadne-workbench-dag/1'
                    and (source is None or runtime.get('source') == source))
    except (OSError,ValueError,urllib.error.URLError):return False


def main():
    root=Path(__file__).resolve().parent
    platform = 'windows' if os.name == 'nt' else 'macos'
    default_source = root.parent / platform / 'source'
    source = os.environ.get('ARIADNE_WORKBENCH_SOURCE', str(default_source))
    selected = Path(source).expanduser().resolve()
    source_folders = ('native/src', 'python/src')
    paths = [str(selected / folder) for folder in source_folders]
    if not all(Path(path).is_dir() for path in paths):
        raise SystemExit('请为 ARIADNE_WORKBENCH_SOURCE 指定包含 native/src 和 python/src 的目录。')
    source = str(selected)
    previous = os.environ.get('ARIADNE_PLUGIN_ROOT')
    obsolete = {Path(previous).resolve() / folder for folder in source_folders} if previous else set()
    inherited = [path for path in os.environ.get('PYTHONPATH', '').split(os.pathsep) if path]
    sys.path[:] = list(dict.fromkeys([*paths, *(path for path in sys.path if Path(path).resolve() not in obsolete)]))
    os.environ['PYTHONPATH'] = os.pathsep.join(
        dict.fromkeys([*paths, *(path for path in inherited if Path(path).resolve() not in obsolete)]))
    os.environ['ARIADNE_WORKBENCH_SOURCE'] = source
    os.environ['ARIADNE_PLUGIN_ROOT'] = source
    if not (root/'dist/assets/app.js').is_file():raise SystemExit('请先在 frontend 目录运行 npm install 与 npm run build。')
    for port in range(8790,8810):
        if ours(port, source):
            webbrowser.open(f'http://127.0.0.1:{port}');print(f'已打开正在运行的工作台：http://127.0.0.1:{port}');return
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))!=0:break
    else:raise SystemExit('8790–8809 端口均不可用，请关闭不需要的本地服务后重试。')
    url=f'http://127.0.0.1:{port}'
    def open_ready():
        for _ in range(50):
            if ours(port, source):webbrowser.open(url);return
            time.sleep(.1)
    threading.Thread(target=open_ready,daemon=True).start()
    print(f'Ariadne Pro 工作台：{url}\n关闭此终端或按 Ctrl+C 停止服务。',flush=True)
    uvicorn.run('ariadne_workbench.app:app',host='127.0.0.1',port=port)


if __name__=='__main__':main()
