from __future__ import annotations

import os
import re
import stat
import time
import uuid
from pathlib import Path


JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def scripts_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_jobs_root() -> Path:
    return scripts_root() / "_jobs"


def make_job_id(prefix: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def job_dir(jobs_root: str | os.PathLike[str] | None, job_id: str) -> Path:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError(
            "job_id must be 1-128 characters and contain only letters, digits, dots, underscores, or hyphens"
        )
    root = (Path(jobs_root).expanduser() if jobs_root else default_jobs_root()).resolve()
    candidate = root / job_id
    validate_job_directory(candidate, root)
    return candidate


def validate_job_directory(path: Path, parent: Path) -> None:
    """Job-owned directories must be direct, unlinked children of their owner."""
    if path.parent != parent:
        raise ValueError(f"Job directory escaped its owner: {path}; check the job path and links")
    resolved_parent = path.resolve().parent
    resolved_owner = parent.resolve()
    same_owner = resolved_parent == resolved_owner
    if not same_owner:
        # Concurrent directory creation can make Windows resolve one spelling
        # with the extended \\?\ namespace and the other without it. Do not
        # strip prefixes or relax containment: require actual directory identity.
        try:
            same_owner = resolved_parent.samefile(resolved_owner)
        except OSError:
            same_owner = False
    if not same_owner:
        raise ValueError(f"Job directory escaped its owner: {path}; check the job path and links")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise ValueError(f"Job directory must be an ordinary directory, not a link: {path}")


def ensure_job_dirs(jobs_root: str | os.PathLike[str] | None, job_id: str) -> Path:
    root = job_dir(jobs_root, job_id)
    root.mkdir(parents=True, exist_ok=True)
    validate_job_directory(root, root.parent)
    validate_job_directory(root / "temp", root)
    (root / "temp").mkdir(exist_ok=True)
    validate_job_directory(root / "temp", root)
    return root


def collision_safe_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 2
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1
