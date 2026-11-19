from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any


_PROCESS_LOCK_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[Path, tuple[int, int, int, int]] = {}


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid > 0xFFFFFFFF:
        return False
    if os.name == "nt":
        # On Windows, ``os.kill(pid, 0)`` is not a POSIX-style liveness
        # probe: signal value 0 is CTRL_C_EVENT.  Using it here can interrupt
        # every process attached to the worker console while one job merely
        # checks the owner of a queue lock.
        try:
            import ctypes
            from ctypes import wintypes

            process_query_limited_information = 0x1000
            still_active = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.DWORD),
            ]
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL

            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                error_code = ctypes.get_last_error()
                if error_code == 87:  # ERROR_INVALID_PARAMETER: no such PID.
                    return False
                # Access denied proves that the process exists. Any other
                # failure is inconclusive and must preserve the lock.
                return True
            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    # An inconclusive probe must keep the lock rather than
                    # risk deleting a live worker's ownership marker.
                    return True
                return exit_code.value == still_active
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            # Preserve the lock when the host cannot safely inspect a PID.
            return True
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


class GpuLockCancelled(RuntimeError):
    """Raised when a queued job receives the shared cancel token."""


class SharedGpuLock:
    """Small cross-platform exclusive lock for a named local resource lane."""

    def __init__(
        self,
        jobs_root: str | os.PathLike[str],
        writer: Any,
        name: str = "gpu_render",
        label: str = "GPU render",
        timeout_s: int | None = None,
        stale_s: int = 12 * 60 * 60,
    ) -> None:
        self.jobs_root = Path(jobs_root).expanduser().resolve()
        self.writer = writer
        self.name = name
        self.label = label
        self.timeout_s = timeout_s
        self.stale_s = stale_s
        self.lock_path = self.jobs_root / f"{name}.lock"
        self.fd: int | None = None
        self._entered = False
        self._last_wait_update = 0.0
        self._last_stale_check = 0.0

    def _log(self, message: str) -> None:
        """Lock ownership must not depend on a writable diagnostic log."""
        try:
            self.writer.log(message)
        except OSError:
            pass

    def _cancel(self, message: str) -> None:
        try:
            self.writer.cancelled(message)
        except OSError:
            pass
        raise GpuLockCancelled(message)

    def _discard_unwritten_owner_marker(self, fd: int) -> None:
        """Undo only the inode this attempt created, never another owner's lock."""
        try:
            identity = os.fstat(fd)
        except OSError:
            identity = None
        try:
            os.close(fd)
        except OSError:
            pass
        self.fd = None
        if identity is None:
            return
        try:
            current = self.lock_path.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                self.lock_path.unlink()
        except OSError:
            pass

    def __enter__(self) -> "SharedGpuLock":
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        if getattr(self.writer, "cancel_requested", False):
            message = f"Cancelled while waiting for {self.label} lock"
            self._cancel(message)
        process_id = os.getpid()
        thread_id = threading.get_ident()
        reused = False
        with _PROCESS_LOCK_GUARD:
            held = _PROCESS_LOCKS.get(self.lock_path)
            if held and held[0] == process_id and held[1] == thread_id:
                _PROCESS_LOCKS[self.lock_path] = (held[0], held[1], held[2] + 1, held[3])
                self._entered = True
                reused = True
        if reused:
            self._log(f"Reused {self.label} lock in current worker")
            return self
        start = time.monotonic()
        while True:
            if getattr(self.writer, "cancel_requested", False):
                message = f"Cancelled while waiting for {self.label} lock"
                self._cancel(message)
            self._clear_stale_lock()
            try:
                self.fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = (
                    f"pid={os.getpid()}\n"
                    f"job_id={getattr(self.writer, 'job_id', '')}\n"
                    f"host={socket.gethostname()}\n"
                    f"platform={sys.platform}/{os.name}\n"
                    f"acquired_at={int(time.time())}\n"
                )
                payload_bytes = payload.encode("utf-8")
                try:
                    if os.write(self.fd, payload_bytes) != len(payload_bytes):
                        raise OSError("Incomplete GPU lock ownership marker.")
                except BaseException:
                    self._discard_unwritten_owner_marker(self.fd)
                    raise
                with _PROCESS_LOCK_GUARD:
                    _PROCESS_LOCKS[self.lock_path] = (process_id, thread_id, 1, self.fd)
                self._entered = True
                self._log(f"Acquired {self.label} lock")
                return self
            except FileExistsError:
                waited = int(time.monotonic() - start)
                now = time.monotonic()
                if now - self._last_wait_update >= 10 or waited < 2:
                    try:
                        self.writer.update(
                            state="running",
                            phase="queued",
                            message=f"Waiting for {self.label} lock ({waited}s)",
                        )
                    except OSError:
                        pass
                    self._last_wait_update = now
                if self.timeout_s is not None and waited > self.timeout_s:
                    raise RuntimeError(f"Timed out waiting for {self.label} lock")
                time.sleep(0.25)

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._entered:
            return
        release_fd: int | None = None
        nested_release = False
        process_id = os.getpid()
        thread_id = threading.get_ident()
        with _PROCESS_LOCK_GUARD:
            held = _PROCESS_LOCKS.get(self.lock_path)
            if held and held[0] == process_id and held[1] == thread_id:
                if held[2] > 1:
                    _PROCESS_LOCKS[self.lock_path] = (held[0], held[1], held[2] - 1, held[3])
                    self._entered = False
                    nested_release = True
                else:
                    release_fd = held[3]
                    _PROCESS_LOCKS.pop(self.lock_path, None)
            elif self.fd is not None:
                release_fd = self.fd
        if nested_release:
            self._log(f"Released nested {self.label} lock")
            return
        self._entered = False
        try:
            if release_fd is not None:
                os.close(release_fd)
            if self.lock_path.exists():
                self.lock_path.unlink()
            self._log(f"Released {self.label} lock")
        except Exception as release_exc:
            self._log(f"Failed to release {self.label} lock: {release_exc}")

    def _clear_stale_lock(self) -> None:
        if not self.lock_path.exists():
            return
        now = time.monotonic()
        if now - self._last_stale_check < 30:
            return
        self._last_stale_check = now
        try:
            age_s = time.time() - self.lock_path.stat().st_mtime
            text = self.lock_path.read_text(encoding="utf-8", errors="ignore")
            pid = 0
            for line in text.splitlines():
                if line.startswith("pid="):
                    try:
                        pid = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        pid = 0
                    break
            owner_is_dead = bool(pid) and not _pid_is_alive(pid)
            owner_is_unknown_and_stale = not pid and age_s > self.stale_s
            if owner_is_dead or owner_is_unknown_and_stale:
                self.lock_path.unlink()
                self._log(f"Removed stale {self.label} lock")
        except Exception as exc:
            message = str(exc)
            if "being used by another process" not in message and "Permission denied" not in message:
                self._log(f"Could not inspect {self.label} lock: {exc}")
