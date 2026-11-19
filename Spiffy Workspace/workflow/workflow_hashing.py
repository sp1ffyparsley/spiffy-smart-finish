"""Fast media fingerprints for move-safe local caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SampleFingerprint:
    path: str
    exists: bool
    file_size: int = 0
    sample_bytes: int = 0
    algorithm: str = ""
    first_hash: str = ""
    middle_hash: str = ""
    last_hash: str = ""
    media_uid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _new_hasher():
    try:
        import blake3  # type: ignore

        return "blake3", blake3.blake3()
    except Exception:
        return "sha256", hashlib.sha256()


def _digest(data: bytes, *, algorithm: str = "") -> tuple[str, str]:
    if algorithm == "sha256":
        return "sha256", hashlib.sha256(data).hexdigest()
    try:
        import blake3  # type: ignore

        return "blake3", blake3.blake3(data).hexdigest()
    except Exception:
        return "sha256", hashlib.sha256(data).hexdigest()


def stable_json_hash(value: Any, *, prefix: str = "") -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    algo, hexdigest = _digest(text.encode("utf-8", "replace"))
    return f"{prefix}{algo}_{hexdigest}" if prefix else f"{algo}_{hexdigest}"


def file_sample_fingerprint(
    path: str | Path,
    *,
    sample_bytes: int = 1024 * 1024,
    algorithm: str = "",
) -> SampleFingerprint:
    """Return a path-independent sparse content fingerprint.

    The absolute path is reported as a locator only; it is not part of the
    ``media_uid``.  First/middle/last samples catch common replacements while
    staying cheap enough for desktop-open and extension-resolve calls.
    """

    p = Path(str(path or "")).expanduser()
    sample_n = max(4096, int(sample_bytes or 1024 * 1024))
    try:
        stat = p.stat()
        size = int(stat.st_size)
    except Exception:
        return SampleFingerprint(path=str(p), exists=False, sample_bytes=sample_n)

    def _read_at(handle, offset: int) -> bytes:
        handle.seek(max(0, min(int(offset), max(0, size - 1))))
        return handle.read(min(sample_n, size))

    try:
        with p.open("rb") as handle:
            first = handle.read(min(sample_n, size))
            middle_offset = max(0, (size // 2) - (sample_n // 2))
            middle = _read_at(handle, middle_offset) if size > sample_n else first
            last_offset = max(0, size - sample_n)
            last = _read_at(handle, last_offset) if size > sample_n else first
    except Exception:
        return SampleFingerprint(path=str(p), exists=False, file_size=size, sample_bytes=sample_n)

    algo, first_hash = _digest(first, algorithm=str(algorithm or "").strip().lower())
    _, middle_hash = _digest(middle, algorithm=algo)
    _, last_hash = _digest(last, algorithm=algo)
    uid_payload = {
        "schema": "media_uid_sparse_samples_v1",
        "file_size": size,
        "sample_bytes": sample_n,
        "algorithm": algo,
        "first": first_hash,
        "middle": middle_hash,
        "last": last_hash,
    }
    _, uid_hash = _digest(
        json.dumps(uid_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        algorithm=algo,
    )
    return SampleFingerprint(
        path=str(p),
        exists=True,
        file_size=size,
        sample_bytes=sample_n,
        algorithm=algo,
        first_hash=first_hash,
        middle_hash=middle_hash,
        last_hash=last_hash,
        media_uid=f"mid_{algo}_{uid_hash}",
    )


__all__ = ["SampleFingerprint", "file_sample_fingerprint", "stable_json_hash"]
