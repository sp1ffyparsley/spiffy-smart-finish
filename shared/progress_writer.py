from __future__ import annotations

import errno
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .job_paths import ensure_job_dirs


SCHEMA_VERSION = 1
REPLACE_RETRY_SECONDS = 2.0
REPLACE_RETRY_MAX_DELAY_SECONDS = 0.20


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_transient_replace_error(exc: OSError) -> bool:
    return getattr(exc, "winerror", None) in {5, 32, 33} or exc.errno in {
        errno.EACCES,
        errno.EBUSY,
        errno.EPERM,
    }


def _replace_with_retry(tmp: Path, path: Path) -> None:
    deadline = time.monotonic() + REPLACE_RETRY_SECONDS
    delay = 0.01
    while True:
        try:
            os.replace(tmp, path)
            return
        except OSError as exc:
            if not _is_transient_replace_error(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, REPLACE_RETRY_MAX_DELAY_SECONDS)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        _replace_with_retry(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        _replace_with_retry(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


class ProgressWriter:
    def __init__(
        self,
        jobs_root: str | os.PathLike[str] | None,
        job_id: str,
        title: str,
        worker_name: str,
    ) -> None:
        self.job_id = job_id
        self.job_dir = ensure_job_dirs(jobs_root, job_id)
        self.progress_path = self.job_dir / "progress.json"
        self.manifest_path = self.job_dir / "manifest.json"
        self.log_path = self.job_dir / "worker.log"
        self.stdout_path = self.job_dir / "stdout.txt"
        self.stderr_path = self.job_dir / "stderr.txt"
        self.commands_path = self.job_dir / "commands.txt"
        self.progress_write_error_path = self.job_dir / "progress_write_errors.log"
        self.temp_dir = self.job_dir / "temp"
        self.started_at = utc_now()
        self.write_seq = 0
        self.data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "title": title,
            "worker_name": worker_name,
            "state": "queued",
            "phase": "queued",
            "message": "",
            "percent": 0,
            "file_index": 0,
            "file_total": 0,
            "current_file": {"display_name": "", "source_path": ""},
            "metrics": {},
            "errors": [],
            "started_at_utc": self.started_at,
            "updated_at_utc": self.started_at,
            "write_seq": 0,
            "can_cancel": True,
            "output_folder": "",
            "log_folder": str(self.job_dir),
        }
        self.write()

    @property
    def cancel_requested(self) -> bool:
        return (self.job_dir / "cancel.request").exists()

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}\n"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def command(self, argv: list[str]) -> None:
        with self.commands_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(argv, ensure_ascii=False) + "\n")

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest = dict(manifest)
        manifest.setdefault("job_id", self.job_id)
        manifest.setdefault("created_at", self.started_at)
        manifest.setdefault("worker_pid", os.getpid())
        atomic_write_json(self.manifest_path, manifest)

    def update(self, **fields: Any) -> None:
        for key, value in fields.items():
            if key == "current_file" and isinstance(value, dict):
                self.data["current_file"].update(value)
            elif key == "metrics" and isinstance(value, dict):
                self.data["metrics"].update(value)
            elif key == "errors":
                self.data["errors"] = value
            else:
                self.data[key] = value
        self.write()

    def error(self, message: str, phase: str = "error", exc: Exception | None = None) -> None:
        errors = list(self.data.get("errors") or [])
        errors.append(message if exc is None else f"{message}: {exc}")
        self.update(state="error", phase=phase, message=message, errors=errors, can_cancel=False)
        self.log(f"ERROR {message}" + (f": {exc}" if exc else ""))

    def done(self, message: str = "Done") -> None:
        self.update(state="done", phase="done", message=message, percent=100, can_cancel=False)
        self.log(message)

    def cancelled(self, message: str = "Cancelled") -> None:
        self.update(state="cancelled", phase="cancelled", message=message, can_cancel=False)
        self.log(message)

    def write(self) -> None:
        self.write_seq += 1
        self.data["updated_at_utc"] = utc_now()
        self.data["write_seq"] = self.write_seq
        try:
            atomic_write_json(self.progress_path, self.data)
        except OSError as exc:
            # Progress is presentation state. A transient reader/AV lock must
            # never terminate the media process that the progress UI observes.
            try:
                with self.progress_write_error_path.open("a", encoding="utf-8") as f:
                    f.write(f"[{utc_now()}] write_seq={self.write_seq} {type(exc).__name__}: {exc}\n")
            except OSError:
                pass


def load_input_list(path: str | os.PathLike[str]) -> list[str]:
    p = Path(path)
    out: list[str] = []
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        value = raw.strip().strip('"')
        if value:
            out.append(str(Path(value).resolve()))
    return out
