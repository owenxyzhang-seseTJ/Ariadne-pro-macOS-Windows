"""A cooperative FIFO budget shared by local Ariadne processes."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import time
import uuid

from .util import LockBusy, atomic_json, exclusive_lock, read_json


RESOURCE_LIMITS = {"cpu_threads": 4096, "memory_mb": 16_777_216}


def validate_resources(resources, *, capacity=False):
    if not isinstance(resources, dict) or set(resources) != set(RESOURCE_LIMITS):
        raise ValueError("Specify cpu_threads and memory_mb for the shared host budget")
    for key, upper in RESOURCE_LIMITS.items():
        value = resources[key]
        if capacity and key == "memory_mb" and value is None:
            continue
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{key} must be an integer in 1..{upper}")
    return dict(resources)


class HostBudget:
    def __init__(self, configuration):
        self.configuration = Path(configuration).expanduser().resolve()
        self.directory = self.configuration.with_name(self.configuration.name + ".queue")
        self._lock = self.directory / "queue.lock"
        self._state = self.directory / "queue.json"
        self.capacity()  # Validate before creating a queue.

    @classmethod
    def configured(cls):
        value = os.environ.get("ARIADNE_HOST_BUDGET")
        if value:
            return cls(value)
        return None

    def capacity(self):
        return validate_resources(read_json(self.configuration), capacity=True)

    def _load(self):
        return read_json(self._state) if self._state.exists() else []

    def _lease(self, token, role):
        return exclusive_lock(self.directory / f"{token}.{role}.lock")

    def _unlink_lease(self, token):
        for role in ("owner", "worker"):
            (self.directory / f"{token}.{role}.lock").unlink(missing_ok=True)

    def _reap(self, entries):
        live, expired = [], []
        for entry in entries:
            token = entry["token"]
            try:
                with self._lease(token, "owner"), self._lease(token, "worker"):
                    expired.append(token)
            except LockBusy:
                live.append(entry)
        if expired:
            atomic_json(self._state, live)
            for token in expired:
                self._unlink_lease(token)
        return live

    @staticmethod
    def _check_fits(resources, limits):
        for key, limit in limits.items():
            if limit is not None and resources[key] > limit:
                raise ValueError(f"Requested {key} exceeds the configured host budget")

    @contextmanager
    def request(self, resources):
        resources = validate_resources(resources)
        self._check_fits(resources, self.capacity())
        token = uuid.uuid4().hex
        try:
            with self._lease(token, "owner"):
                with exclusive_lock(self._lock, blocking=True):
                    entries = self._reap(self._load())
                    entries.append({"token": token, "admitted": False, **resources})
                    atomic_json(self._state, entries)
                try:
                    yield HostRequest(self, token)
                finally:
                    with exclusive_lock(self._lock, blocking=True):
                        try:
                            with self._lease(token, "worker"):
                                entries = [entry for entry in self._load() if entry["token"] != token]
                                atomic_json(self._state, entries)
                        except LockBusy:
                            # The worker will retain the admission until it exits.
                            pass
        finally:
            with exclusive_lock(self._lock, blocking=True):
                if not any(row["token"] == token for row in self._load()):
                    self._unlink_lease(token)

    @contextmanager
    def worker(self, token):
        """Keep a running worker's reservation live during coordinator teardown."""
        if not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("invalid host reservation token")
        # Registration and verification are serialized with queue recovery.
        with ExitStack() as lease:
            with exclusive_lock(self._lock, blocking=True):
                if not any(row["token"] == token and row["admitted"] for row in self._load()):
                    raise RuntimeError("the owner no longer holds a host reservation")
                lease.enter_context(self._lease(token, "worker"))
            yield


class HostRequest:
    def __init__(self, host, token):
        self.host = host
        self.token = token
        self.queued_at = time.monotonic()

    def try_admit(self):
        host = self.host
        with exclusive_lock(host._lock, blocking=True):
            entries = host._reap(host._load())
            limits = host.capacity()
            request = next(row for row in entries if row["token"] == self.token)
            host._check_fits(request, limits)
            if request["admitted"]:
                return True
            pending = next((row for row in entries if not row["admitted"]), None)
            if pending is None or pending["token"] != self.token:
                return False
            active = [row for row in entries if row["admitted"]]
            if any(limit is not None and pending[key] + sum(row[key] for row in active) > limit
                   for key, limit in limits.items()):
                return False
            pending["admitted"] = True
            atomic_json(host._state, entries)
            return True

    def worker_payload(self):
        return {"configuration": str(self.host.configuration), "token": self.token}
