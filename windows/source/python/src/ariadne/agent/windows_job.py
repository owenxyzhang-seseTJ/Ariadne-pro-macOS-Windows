"""Own a Windows process tree from its first executable instruction."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import sys


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("per_process_time", ctypes.c_int64), ("per_job_time", ctypes.c_int64),
        ("flags", wintypes.DWORD),
        ("minimum_working_set", ctypes.c_size_t), ("maximum_working_set", ctypes.c_size_t),
        ("active_process_limit", wintypes.DWORD), ("affinity", ctypes.c_size_t),
        ("priority_class", wintypes.DWORD), ("scheduling_class", wintypes.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _BasicLimits), ("io_counters", ctypes.c_uint64 * 6),
        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
    ]


def _api():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    for name, result, arguments in [
        ("CreateJobObjectW", wintypes.HANDLE, [ctypes.c_void_p, wintypes.LPCWSTR]),
        ("SetInformationJobObject", wintypes.BOOL,
         [wintypes.HANDLE, wintypes.INT, ctypes.c_void_p, wintypes.DWORD]),
        ("OpenProcess", wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]),
        ("AssignProcessToJobObject", wintypes.BOOL, [wintypes.HANDLE, wintypes.HANDLE]),
        ("CloseHandle", wintypes.BOOL, [wintypes.HANDLE]),
        ("WaitForSingleObject", wintypes.DWORD, [wintypes.HANDLE, wintypes.DWORD]),
    ]:
        fn = getattr(kernel, name)
        fn.restype, fn.argtypes = result, arguments
    return kernel


def pid_alive(pid):
    kernel = _api()
    handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return ctypes.get_last_error() == 5  # Access denied: conservatively report live.
    try:
        return kernel.WaitForSingleObject(handle, 0) != 0
    finally:
        kernel.CloseHandle(handle)


class OwnedProcess:
    def __init__(self, argv, **kwargs):
        self._kernel = kernel = _api()
        self._job = kernel.CreateJobObjectW(None, None)
        if not self._job:
            raise ctypes.WinError(ctypes.get_last_error())
        child = None
        try:
            limits = _ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel.SetInformationJobObject(self._job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            gate = str(Path(__file__).with_name("windows_spawn.py"))
            child = self._child = subprocess.Popen(
                [sys.executable, "-I", "-S", gate, *argv], stdin=subprocess.PIPE, **kwargs)
            # The gate is a pipe. The launcher cannot start argv before assignment.
            handle = kernel.OpenProcess(0x101, False, child.pid)  # SET_QUOTA | TERMINATE
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not kernel.AssignProcessToJobObject(self._job, handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                kernel.CloseHandle(handle)
            with child.stdin:
                child.stdin.write(b"1")
        except BaseException:
            if child is not None:
                child.kill()
                child.wait()
            self.close()
            raise

    def __getattr__(self, name):
        return getattr(self._child, name)

    def close(self):
        if self._job:
            self._kernel.CloseHandle(self._job)
            self._job = None

    def stop(self):
        self.close()
        return self._child.wait()
