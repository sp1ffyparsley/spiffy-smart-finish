from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def which(name: str) -> str:
    return shutil.which(name) or ""


def standard_ffmpeg() -> str:
    return which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"


def standard_ffprobe() -> str:
    return which("ffprobe") or r"C:\ffmpeg\bin\ffprobe.exe"


def has_encoder(ffmpeg: str, encoder: str) -> bool:
    try:
        result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20)
        return encoder.lower() in (result.stdout + result.stderr).lower()
    except Exception:
        return False


def even(value: int) -> int:
    value = int(round(value))
    if value < 2:
        return 2
    return value if value % 2 == 0 else value - 1


def fit_within(width: int, height: int, max_w: int, max_h: int) -> tuple[int, int]:
    scale = min(max_w / width, max_h / height)
    return even(width * scale), even(height * scale)


def fill_frame(width: int, height: int, frame_w: int, frame_h: int) -> tuple[int, int]:
    scale = max(frame_w / width, frame_h / height)
    return even(width * scale), even(height * scale)


def keep_aspect_target(width: int, height: int, long_edge: int) -> tuple[int, int]:
    if width >= height:
        return even(long_edge), even(height * (long_edge / width))
    return even(width * (long_edge / height)), even(long_edge)


def layout_filter(layout_id: str, src_w: int, src_h: int, target_w: int, target_h: int, flags: str = "lanczos") -> str:
    target_w = even(target_w)
    target_h = even(target_h)
    if layout_id == "fit_into_frame":
        return (
            f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=decrease:flags={flags},"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )
    if layout_id == "fill_crop_center":
        return (
            f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=increase:flags={flags},"
            f"crop={target_w}:{target_h},setsar=1"
        )
    if target_w and target_h and (target_w != src_w or target_h != src_h):
        return f"scale={target_w}:{target_h}:flags={flags},setsar=1"
    return "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,setsar=1"


def run_logged(argv: list[str], log, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    log("CMD " + repr(argv))
    env = os.environ.copy()
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)


def part_path(final_path: Path) -> Path:
    return final_path.with_name(final_path.stem + ".tmp" + final_path.suffix)
