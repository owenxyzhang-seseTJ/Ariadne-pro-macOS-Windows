"""Bounded process invocation: argv only, isolated attempts, file-backed logs."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Callable

from .util import atomic_json, now, file_hash


THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def thread_environment(cpu_threads=1, *, base=None):
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 4096:
        raise ValueError('cpu_threads must be an integer in 1..4096')
    environment = dict(os.environ if base is None else base)
    environment.update({key: str(cpu_threads) for key in THREAD_VARIABLES})
    environment['OMP_DYNAMIC'] = 'FALSE'
    environment['MKL_DYNAMIC'] = 'FALSE'
    environment['ARIADNE_CPU_THREADS'] = str(cpu_threads)
    return environment


def stop_process_group(child, *, grace=0.3):
    """Terminate a session-owned group, wait for its leader, kill stragglers."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


class Cancelled(RuntimeError):
    pass


class NeedsReview(RuntimeError):
    pass


def executable(name: str, mapping: dict[str, str]) -> str:
    configured = mapping.get(name, name)
    resolved = shutil.which(configured)
    if not resolved:
        raise RuntimeError(f'Required executable not found: {name}. Configure executables.{name}.')
    return resolved


def run_process(argv: list[str], directory: Path, timeout: float, cancelled: Callable[[], bool],
                on_pid: Callable[[int], None] | None = None, label: str = 'process',
                log_limit_bytes: int = 256*1024*1024, cpu_threads: int = 1) -> dict:
    if not argv or not all(isinstance(x, str) and '\x00' not in x for x in argv):
        raise ValueError('argv must be a list of strings')
    if timeout <= 0:
        raise TimeoutError('time budget exhausted')
    if cancelled():
        raise Cancelled('cancellation requested before process launch')
    environment = thread_environment(cpu_threads)
    managed = os.environ.get('ARIADNE_MANAGED_ACTION') == '1'
    atomic_json(directory/(label+'.command.json'), {'argv': argv, 'started': now(), 'timeout': timeout,
                'cpu_threads': cpu_threads, 'thread_environment': {key: environment[key] for key in THREAD_VARIABLES},
                'executable_sha256': file_hash(Path(argv[0]).resolve())})
    stdout = directory/(label+'.stdout.log')
    stderr = directory/(label+'.stderr.log')
    started = time.monotonic()
    with stdout.open('wb') as out, stderr.open('wb') as err:
        child = subprocess.Popen(argv, cwd=directory, stdout=out, stderr=err,
                                 stdin=subprocess.DEVNULL, shell=False, start_new_session=not managed, env=environment)
        try:
            if on_pid:
                on_pid(child.pid)
            while child.poll() is None:
                if cancelled():
                    raise Cancelled('cancellation requested')
                if time.monotonic()-started > timeout:
                    raise TimeoutError('external calculation exceeded its time budget')
                if stdout.stat().st_size+stderr.stat().st_size > log_limit_bytes:
                    raise RuntimeError('external logs exceeded the storage budget')
                time.sleep(.05)
            if stdout.stat().st_size+stderr.stat().st_size > log_limit_bytes:
                raise RuntimeError('external logs exceeded the storage budget')
        except BaseException:
            if managed:
                # The parent scheduler owns the containing action process group
                # and will reap any grandchildren when this worker exits.
                child.terminate()
                try:
                    child.wait(timeout=.3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            else:
                stop_process_group(child)
            raise
        finally:
            if not managed and child.poll() is not None:
                stop_process_group(child, grace=0.)
    result = dict(returncode=child.returncode, elapsed_seconds=time.monotonic()-started,
                  stdout=stdout.name, stderr=stderr.name)
    atomic_json(directory/(label+'.process.json'), result)
    if child.returncode:
        with stderr.open('rb') as stream:
            stream.seek(max(0, stderr.stat().st_size-3000))
            tail = stream.read().decode(errors='replace')
        raise RuntimeError(f'{Path(argv[0]).name} exited {child.returncode}: {tail}')
    return result
