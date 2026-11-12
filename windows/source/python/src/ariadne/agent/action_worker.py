"""Internal POSIX action worker. Only the parent commits DAG state/results."""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path


def _watch_parent(descriptor):
    # The write end exists only in the scheduler. EOF includes SIGKILL/crash.
    # Registered child adapters stay in this worker's process group.
    try:
        while os.read(descriptor, 1):
            pass
    except OSError:
        pass
    os.killpg(os.getpgrp(), signal.SIGKILL)


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2 or os.name != 'posix' or os.getpgrp() != os.getpid():
        raise RuntimeError('internal worker requires its own POSIX process group and parent pipe')
    threading.Thread(target=_watch_parent, args=(int(arguments[1]),), daemon=True).start()
    from .actions import Context, execute
    from .process import Cancelled, NeedsReview
    from .store import Store
    from .util import atomic_json, read_json
    payload = read_json(Path(arguments[0]))
    directory = Path(payload['directory'])
    store = Store(Path(payload['root']))
    started = time.monotonic()
    try:
        context = Context(store, payload['node'], directory, started,
                          deadline=payload['deadline'], resources=payload['resources'])
        if context.remaining() <= 0:
            raise TimeoutError('time budget exhausted before worker import completed')
        result = execute(context, payload['tool'], payload['args'])
        if context.remaining() <= 0:
            raise TimeoutError('scientific action exceeded its time budget')
        if store.get('cancel_requested'):
            raise Cancelled('cancelled before committing calculation result')
        output = {'status': 'succeeded', 'result': result}
    except Cancelled as error:
        output = {'status': 'cancelled', 'error': str(error)}
    except NeedsReview as error:
        output = {'status': 'blocked', 'error': str(error)}
    except KeyboardInterrupt:
        output = {'status': 'interrupted', 'error': 'worker interrupt'}
    except BaseException as error:  # noqa: BLE001 - Persist any action failure at the worker boundary.
        output = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
    finally:
        store.close()
    atomic_json(directory/'worker.output.json', output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
