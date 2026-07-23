"""Process-safe guard against accidentally starting two training runs.

The lock is advisory and is held for the lifetime of the trainer.  POSIX
record locks are released automatically if the process exits or crashes, so a
stale metadata file does not block a later run.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import shlex
import sys
import time


LOCK_PATH_ENV = "NASIMEMU_TRAIN_LOCK_PATH"


class TrainingLockError(RuntimeError):
    """Raised when another process already owns the training lock."""


class TrainingRunLock:
    """A held training lock. Keep this object alive for the whole run."""

    def __init__(self, path, handle):
        self.path = path
        self._handle = handle
        self._released = False

    def release(self):
        if self._released:
            return
        self._released = True
        try:
            fcntl.lockf(
                self._handle.fileno(), fcntl.LOCK_UN, 0, 0, os.SEEK_SET,
            )
        finally:
            self._handle.close()


def default_training_lock_path():
    """Return the repository-local lock path, unless explicitly overridden."""
    return os.environ.get(
        LOCK_PATH_ENV,
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "training_data", ".training.lock"),
    )


def _holder_metadata(extra=None):
    metadata = {
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cwd": os.getcwd(),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
    }
    if extra:
        metadata.update(extra)
    return metadata


def acquire_training_lock(path=None, metadata=None):
    """Acquire and return the sole training lock without waiting.

    Merely finding an old lock file is harmless: only an active kernel lock
    rejects the new run. This makes the guard safe across crashes and reboots.
    """
    path = os.path.abspath(path or default_training_lock_path())
    os.makedirs(os.path.dirname(path), exist_ok=True)

    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(path, flags, 0o664)
    handle = os.fdopen(fd, "r+", encoding="utf-8")

    try:
        fcntl.lockf(
            handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB,
            0, 0, os.SEEK_SET,
        )
    except BlockingIOError:
        handle.seek(0)
        holder = handle.read().strip() or "holder metadata unavailable"
        handle.close()
        raise TrainingLockError(
            "Another NASimEmu training run is already active.\n"
            f"Lock: {path}\n"
            f"Holder: {holder}\n"
            "Stop the existing trainer before starting another. Evaluation "
            "commands using --eval are not blocked."
        )

    lock = TrainingRunLock(path, handle)
    atexit.register(lock.release)

    handle.seek(0)
    handle.truncate()
    json.dump(_holder_metadata(metadata), handle, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    return lock


def _pid_is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def hold_lock_for_pid(pid, path=None, poll_seconds=2.0):
    """Adopt the lock for a trainer that started before this guard existed."""
    if not _pid_is_alive(pid):
        raise TrainingLockError(f"Cannot protect PID {pid}: process is not running")

    lock = acquire_training_lock(
        path,
        metadata={"role": "adopted-lock-holder", "training_pid": int(pid)},
    )
    try:
        while _pid_is_alive(pid):
            time.sleep(poll_seconds)
    finally:
        lock.release()


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description="Hold the NASimEmu training lock for an existing trainer",
    )
    parser.add_argument("--follow-pid", type=int, required=True)
    parser.add_argument("--lock-path", default=None)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)

    try:
        hold_lock_for_pid(
            args.follow_pid, path=args.lock_path,
            poll_seconds=args.poll_seconds,
        )
    except TrainingLockError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
