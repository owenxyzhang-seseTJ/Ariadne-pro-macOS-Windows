from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import errno
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


CONTENT_ALGORITHM = 'blake2b-256'


def _hasher(algorithm):
    if algorithm == CONTENT_ALGORITHM:
        return hashlib.blake2b(digest_size=32)
    if algorithm == 'sha256':
        return hashlib.sha256()  # Read historical sessions using their recorded format.
    raise ValueError(f'unknown content algorithm: {algorithm}')


def digest(value: Any, algorithm=CONTENT_ALGORITHM) -> str:
    h = _hasher(algorithm)
    h.update(dumps(value).encode())
    return h.hexdigest()


def file_hash(path: Path, algorithm='sha256') -> str:
    h = _hasher(algorithm)
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def content_id(path: Path, algorithm=CONTENT_ALGORITHM) -> str:
    return f'{algorithm}:{file_hash(path, algorithm)}'


def is_artifact(record) -> bool:
    return (isinstance(record, dict) and {'path', 'size_bytes', 'role'} <= record.keys()
            and ('content_id' in record or 'sha256' in record))


def artifact_identity(record) -> str:
    return record['content_id'] if 'content_id' in record else 'sha256:' + record['sha256']


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(dumps(value)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> Any:
    def bad_constant(value):
        raise ValueError(f'nonfinite JSON number: {value}')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=bad_constant)


def contained(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError('require a relative artifact path')
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('artifact path escapes session')
    return path


def artifact(root: Path, path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('artifact must be a real file inside the session')
    return dict(path=path.relative_to(root).as_posix(), content_id=content_id(path),
                size_bytes=path.stat().st_size, role=role)


def artifact_path(root: Path, value: dict[str, Any]) -> Path:
    if not is_artifact(value):
        raise ValueError('expected a verified artifact reference')
    path = contained(root, value['path'])
    identity = artifact_identity(value)
    if (not path.is_file() or path.stat().st_size != value['size_bytes']
            or content_id(path, identity.partition(':')[0]) != identity):
        raise ValueError('artifact content changed or is missing')
    return path


def pointer(value: Any, path: str) -> Any:
    if not path:
        return value
    if not path.startswith('/'):
        raise ValueError('JSON pointer must start with /')
    for token in path[1:].split('/'):
        key = token.replace('~1', '/').replace('~0', '~')
        if isinstance(value, list):
            if not key.isdigit() or (len(key) > 1 and key.startswith('0')):
                raise ValueError('invalid list index in JSON pointer')
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise ValueError('JSON pointer traverses a scalar')
    return value


@contextmanager
def exclusive_lock(path: Path, *, blocking=False):
    """Exclusive advisory lock released by the operating system on process exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != 'nt':
        import fcntl

        with path.open('a') as stream:
            try:
                flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
                fcntl.flock(stream.fileno(), flags)
            except BlockingIOError as exc:
                raise LockBusy('another runner holds this session') from exc
            yield
        return

    import msvcrt

    with path.open('a+b') as stream:
        stream.seek(0)
        while True:
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EDEADLK}:
                    raise
                if not blocking:
                    raise LockBusy('another runner holds this session') from exc
                time.sleep(.05)
        yield


class LockBusy(RuntimeError):
    """Another live owner holds the run lock."""
