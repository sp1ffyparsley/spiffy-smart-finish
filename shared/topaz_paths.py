from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def existing(paths: list[str]) -> str:
    for p in paths:
        if p and Path(p).exists():
            return p
    return ""


def find_gigapixel() -> dict:
    root = os.environ.get("TOPAZ_GIGAPIXEL_DIR", "")
    candidates = []
    if root:
        candidates += [str(Path(root) / "gigapixel.exe"), str(Path(root) / "Topaz Gigapixel.exe")]
    candidates += [
        r"C:\Program Files\Topaz Labs LLC\Topaz Gigapixel\gigapixel.exe",
        r"C:\Program Files\Topaz Labs LLC\Topaz Gigapixel\bin\gigapixel.bat",
        r"C:\Program Files\Topaz Labs LLC\Topaz Gigapixel\Topaz Gigapixel.exe",
    ]
    exe = existing(candidates)
    info = {"path": exe, "cli_usable": False, "cli_reason": "not found", "help_output": ""}
    if not exe:
        return info
    try:
        result = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=10)
        text = (result.stdout or "") + (result.stderr or "")
        info["help_output"] = text.strip()
        if "CLI access requires enterprise license" in text:
            info["cli_reason"] = "CLI access requires enterprise license"
        elif result.returncode == 0 and text.strip():
            info["cli_usable"] = True
            info["cli_reason"] = "help command succeeded"
        else:
            info["cli_reason"] = f"help exited {result.returncode}"
    except subprocess.TimeoutExpired:
        info["cli_reason"] = "help command timed out"
    except Exception as exc:
        info["cli_reason"] = str(exc)
    return info


def find_realesrgan(root: Path | None = None) -> dict:
    import shutil

    workspace_root = root or Path(__file__).resolve().parents[1]
    env_exe = os.environ.get("REALESRGAN_NCNN_EXE", "")
    candidates = []
    if env_exe:
        candidates.append(env_exe)
    candidates += [
        str(workspace_root / "tools" / "realesrgan-ncnn-vulkan" / "realesrgan-ncnn-vulkan.exe"),
        str(workspace_root / "tools" / "realesrgan-ncnn-vulkan.exe"),
    ]
    path_hit = shutil.which("realesrgan-ncnn-vulkan.exe") or shutil.which("realesrgan-ncnn-vulkan") or ""
    if path_hit:
        candidates.append(path_hit)

    exe = existing(candidates)
    info = {
        "path": exe,
        "usable": False,
        "reason": "not found",
        "models_path": "",
        "available_models": [],
        "help_output": "",
    }
    if not exe:
        return info

    exe_path = Path(exe)
    models_path = exe_path.parent / "models"
    if models_path.exists():
        info["models_path"] = str(models_path)
        info["available_models"] = sorted({p.stem.rsplit("-", 1)[0] if p.stem.endswith(("-x2", "-x3", "-x4")) else p.stem for p in models_path.glob("*.param")})

    try:
        result = subprocess.run([exe, "-h"], capture_output=True, text=True, timeout=10, cwd=str(exe_path.parent))
        text = (result.stdout or "") + (result.stderr or "")
        info["help_output"] = text.strip()
        if "Usage: realesrgan-ncnn-vulkan" in text:
            info["usable"] = True
            info["reason"] = "help command succeeded"
        else:
            info["reason"] = f"help exited {result.returncode}"
    except subprocess.TimeoutExpired:
        info["reason"] = "help command timed out"
    except Exception as exc:
        info["reason"] = str(exc)
    return info


def find_topaz_video() -> dict:
    import shutil

    root_env = os.environ.get("TOPAZ_VIDEO_DIR") or ""
    roots = [root_env] if root_env else []
    if sys.platform == "darwin":
        # Accept either the application bundle or its binary folder. An app
        # bundle is a directory, but its CLI engines are not at the bundle root.
        if root_env and root_env.rstrip("/\\").lower().endswith(".app"):
            bundle = Path(root_env).expanduser()
            roots = [str(bundle / "Contents" / "MacOS"),
                     str(bundle / "Contents" / "Resources"), *roots]
        roots += [
            "/Applications/Topaz Video AI.app/Contents/MacOS",
            "/Applications/Topaz Video.app/Contents/MacOS",
            "/Applications/Topaz Video AI.app/Contents/Resources",
            "/Applications/Topaz Video.app/Contents/Resources",
        ]
    else:
        roots += [r"C:\Program Files\Topaz Labs LLC\Topaz Video"]

    ffmpeg_candidates: list[str] = []
    ffprobe_candidates: list[str] = []
    app_candidates: list[str] = []
    for root in roots:
        if not root:
            continue
        root_path = Path(root)
        ffmpeg_candidates += [str(root_path / "ffmpeg.exe"), str(root_path / "ffmpeg")]
        ffprobe_candidates += [str(root_path / "ffprobe.exe"), str(root_path / "ffprobe")]
        app_candidates += [str(root_path / "Topaz Video.exe"), str(root_path / "Topaz Video AI"), str(root_path / "Topaz Video")]
    ffmpeg_candidates += [shutil.which("ffmpeg") or ""]
    ffprobe_candidates += [shutil.which("ffprobe") or ""]

    ffmpeg = existing(ffmpeg_candidates)
    ffprobe = existing(ffprobe_candidates)
    app = existing(app_candidates)
    root = str(Path(ffmpeg).parent) if ffmpeg else (roots[0] if roots else "")
    model_candidates = []
    if os.environ.get("TVAI_MODEL_DIR"):
        model_candidates.append(os.environ["TVAI_MODEL_DIR"])
    if sys.platform == "darwin":
        model_candidates += [
            str(Path.home() / "Library" / "Application Support" / "Topaz Labs LLC" / "Topaz Video" / "models"),
            str(Path.home() / "Library" / "Application Support" / "Topaz Video AI" / "models"),
        ]
    else:
        model_candidates += [r"C:\ProgramData\Topaz Labs LLC\Topaz Video\models"]
    model_dir = existing(model_candidates)
    info = {
        "root": root if Path(root).exists() else "",
        "app": app,
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "model_dir": model_dir,
        "tvai_up": False,
        "tvai_fi": False,
        "filter_help": "",
        "reason": "",
    }
    if not info["ffmpeg"]:
        info["reason"] = "Topaz bundled FFmpeg not found; set TOPAZ_VIDEO_DIR to the installed app or binary directory"
        return info
    try:
        up = subprocess.run([info["ffmpeg"], "-hide_banner", "-h", "filter=tvai_up"], capture_output=True, text=True, timeout=20)
        fi = subprocess.run([info["ffmpeg"], "-hide_banner", "-h", "filter=tvai_fi"], capture_output=True, text=True, timeout=20)
        info["filter_help"] = (up.stdout + up.stderr)[:4000]
        info["tvai_up"] = "Filter tvai_up" in (up.stdout + up.stderr)
        info["tvai_fi"] = "Filter tvai_fi" in (fi.stdout + fi.stderr)
        if not info["tvai_up"]:
            info["reason"] = "tvai_up filter missing"
    except Exception as exc:
        info["reason"] = str(exc)
    return info


def find_optional_tools() -> dict:
    import shutil

    candidates = {
        "imagemagick": ["magick.exe", "magick"],
        "jpegmini": [
            r"C:\Program Files\JPEGmini Pro\JPEGmini Pro.exe",
            r"C:\Program Files\Beamr\JPEGmini Pro\JPEGmini Pro.exe",
        ],
        "cjpeg": ["cjpeg.exe", "cjpeg"],
        "oxipng": ["oxipng.exe", "oxipng"],
        "gfpgan": ["gfpgan.exe", "gfpgan"],
        "codeformer": ["codeformer.exe", "codeformer"],
    }
    found: dict[str, str] = {}
    for name, values in candidates.items():
        hit = ""
        for value in values:
            if "\\" in value or ":" in value:
                if Path(value).exists():
                    hit = value
                    break
            else:
                hit = shutil.which(value) or ""
                if hit:
                    break
        found[name] = hit
    return found
