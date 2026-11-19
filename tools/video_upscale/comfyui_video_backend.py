"""Optional local ComfyUI video enhancement for the incumbent Smart Finish CLI.

Runs one bounded GAN video job on the same GPU/job ownership as Final Delivery.
Uses real ComfyUI UpscaleModelLoader/ImageUpscaleWithModel nodes, not a mock
"model selected" path.  Video is decoded exactly once to job-local frames,
and the original audio is retained through a lossless ProRes/PCM intermediate.
This frame model does NOT have the temporal guarantees of SeedVR2/RelayVSR.
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Iterator
from urllib import error, parse, request
from comfyui_model_catalog import (
    DEFAULT_MODEL_POLICY, ModelSpec, get_model, get_model_by_filename,
    select_model, verify_model_file,
)


ROOT = Path(__file__).resolve().parents[2]
SPIFFY = ROOT / "Spiffy Workspace"
DEFAULT_COMFY_ROOT = (Path.home() / "Documents" / "AutoHotkey" / "SS Scripts External Assets"
                      / "AI Media" / "tools" / "ComfyUI")
MODEL_NAME = get_model("realesrgan_x4plus").filename
MODEL_SHA256 = get_model("realesrgan_x4plus").sha256
MAX_FRAMES = 1800
MAX_DURATION_SECONDS = 60
MAX_OUTPUT_PIXELS = 33_554_432
MAX_NATIVE_GAN_PIXELS = 33_554_432


def _run(argv: list[str], *, timeout: int = 300, cwd: Path | None = None,
         cancel: Callable[[], bool] | None = None) -> None:
    """Bounded, cancellable FFmpeg process. Kill only the subprocess we own."""
    if type(timeout) is not int or timeout < 1 or timeout > 3600:
        raise ValueError("Media stage timeout must be a bounded positive integer.")
    if cancel is not None and cancel():
        raise RuntimeError("ComfyUI media stage cancelled before launch.")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(argv, cwd=str(cwd) if cwd else None,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, creationflags=flags)
    deadline = time.monotonic() + timeout
    try:
        while True:
            if cancel is not None and cancel():
                raise RuntimeError("ComfyUI media stage cancelled by user.")
            if time.monotonic() >= deadline:
                raise TimeoutError("ComfyUI media stage exceeded its hard deadline.")
            try:
                _stdout, stderr = proc.communicate(timeout=min(0.4, max(0.01, deadline - time.monotonic())))
                if proc.returncode:
                    message = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
                    raise RuntimeError(f"Media stage failed with exit {proc.returncode}: {message[-1600:]}")
                return
            except subprocess.TimeoutExpired:
                continue
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()


def _json_http(base: str, suffix: str, *, data: dict[str, Any] | None = None,
               timeout: float = 10.0) -> dict[str, Any]:
    # Only endpoints allocated by _owned_comfyui, never externally supplied URLs.
    payload = json.dumps(data, separators=(",", ":")).encode() if data is not None else None
    req = request.Request(base + suffix, data=payload,
                          headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            decoded = json.load(response)
    except error.HTTPError as exc:
        # ComfyUI 400s include the exact unregistered model/node/input
        # reason. Do not discard it as a generic "HTTP Error 400" or
        # swallow it during the private server's contextmanager teardown.
        try:
            detail = exc.read(8192).decode("utf-8", "replace")
        except OSError:
            detail = ""
        raise RuntimeError(
            f"Private ComfyUI API {suffix} rejected the job (HTTP {exc.code}): {detail[:3000]}"
        ) from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("ComfyUI did not return a JSON object.")
    return decoded


def _upscale_graph(filename: str, *, output_width: int, output_height: int,
                   output_prefix: str, model_name: str = MODEL_NAME,
                   source_preservation: float = 0.0) -> dict[str, Any]:
    if (Path(filename).name != filename or filename.startswith(".") or not filename.lower().endswith(".png")
            or "/" in filename or "\\" in filename):
        raise ValueError("ComfyUI frame input must be a basename PNG from the job input folder.")
    if (not output_prefix.startswith("Spiffy_20") or len(output_prefix) > 100
            or not output_prefix.replace("_", "").isalnum()):
        raise ValueError("Unexpected ComfyUI output prefix.")
    if (type(output_width) is not int or type(output_height) is not int
            or min(output_width, output_height) < 64 or output_width % 2 or output_height % 2
            or output_width * output_height > MAX_OUTPUT_PIXELS):
        raise ValueError("ComfyUI output geometry exceeds the existing safe final canvas.")
    get_model_by_filename(model_name)
    if (type(source_preservation) not in (int, float)
            or not math.isfinite(source_preservation) or not 0 <= source_preservation <= 0.25):
        raise ValueError("Source-preservation blend must be a finite fraction from 0 through 0.25.")
    if source_preservation and output_width * output_height > 3840 * 2160:
        raise ValueError("Source-preservation compositing is not yet qualified beyond 4K on this GPU.")
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": filename}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {
            "image": ["1", 0], "upscale_model": ["2", 0]}},
        "4": {"class_type": "ImageScale", "inputs": {
            "image": ["3", 0], "upscale_method": "lanczos", "width": output_width,
            "height": output_height, "crop": "disabled"}},
        "5": {"class_type": "SaveImage", "inputs": {
            "images": ["4", 0], "filename_prefix": output_prefix}},
    }
    if source_preservation:
        graph["6"] = {"class_type": "ImageScale", "inputs": {
            "image": ["1", 0], "upscale_method": "lanczos", "width": output_width,
            "height": output_height, "crop": "disabled"}}
        graph["7"] = {"class_type": "ImageBlend", "inputs": {
            "image1": ["4", 0], "image2": ["6", 0],
            "blend_mode": "normal", "blend_factor": float(source_preservation)}}
        graph["5"]["inputs"]["images"] = ["7", 0]
    return graph


def _upscale_batch_graph(filenames: list[str], *, output_width: int, output_height: int,
                         output_prefix: str, model_name: str = MODEL_NAME,
                         source_preservation: float = 0.0) -> dict[str, Any]:
    """One Comfy prompt / one pinned model, with ordered frames through ImageBatch.

    Small-frame batching is optional, never used to force large 6K tensors
    into the 10 GB GPU. A batch never crosses the source/job identity.
    """
    if not isinstance(filenames, list) or not 1 <= len(filenames) <= 4:
        raise ValueError("Local ComfyUI batch must contain one through four frames.")
    if len(set(filenames)) != len(filenames):
        raise ValueError("ComfyUI batch cannot reuse an input frame name.")
    graph = _upscale_graph(filenames[0], output_width=output_width,
                           output_height=output_height, output_prefix=output_prefix,
                           model_name=model_name, source_preservation=source_preservation)
    last = ["1", 0]
    for i, name in enumerate(filenames[1:], start=1):
        # Reuse the first node's filename-validation contract.
        _upscale_graph(name, output_width=output_width, output_height=output_height,
                       output_prefix=output_prefix, model_name=model_name,
                       source_preservation=source_preservation)
        input_id, batch_id = str(10 + i), str(20 + i)
        graph[input_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph[batch_id] = {"class_type": "ImageBatch", "inputs": {
            "image1": last, "image2": [input_id, 0]}}
        last = [batch_id, 0]
    graph["3"]["inputs"]["image"] = last
    if source_preservation:
        graph["6"]["inputs"]["image"] = last
    return graph


def _batch_size(source_width: int, source_height: int, output_width: int, output_height: int,
                *, available_gpu_mib: float, native_scale: int) -> int:
    """Conservative per-GPU batching, rather than a global batch=4 switch."""
    source_pixels = source_width * source_height
    output_pixels = output_width * output_height
    native_pixels = source_pixels * native_scale**2
    if available_gpu_mib >= 3200 and source_pixels <= 640 * 360 and max(output_pixels, native_pixels) <= 1920 * 1080:
        return 3
    if available_gpu_mib >= 6500 and source_pixels <= 1280 * 720 and max(output_pixels, native_pixels) <= 3840 * 2160:
        return 2
    return 1


def _exact_frame_key(frame: Path) -> tuple[int, str]:
    """Full byte identity only: no perceptual-similarity shortcuts."""
    if not frame.is_file() or frame.is_symlink():
        raise RuntimeError("ComfyUI frame for exact reuse must be an ordinary PNG file.")
    before = frame.stat()
    with frame.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = frame.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Decoded source frame changed while computing exact identity.")
    return after.st_size, digest


def _qualified_checkpoint(comfy_root: Path, model_id: str = "realesrgan_x4plus") -> Path:
    return verify_model_file(comfy_root, get_model(model_id))


@contextmanager
def _owned_comfyui(comfy_root: Path, tmp: Path, *, timeout: int = 120,
                   model_name: str = MODEL_NAME) -> Iterator[str]:
    """Start only our isolated local server; never stop a user's ComfyUI session."""
    exe = comfy_root / ".venv" / "Scripts" / "python.exe"
    entry = comfy_root / "main.py"
    if not exe.is_file() or not entry.is_file():
        raise FileNotFoundError("Installed ComfyUI Python runtime is missing.")
    for folder in ("decoded", "output", "comfy_temp"):
        (tmp / folder).mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    args = [str(exe), "-u", str(entry), "--listen", "127.0.0.1", "--port", str(port),
            "--disable-auto-launch", "--lowvram", "--disable-metadata",
            "--disable-all-custom-nodes", "--disable-partner-nodes", "--offline",
            "--input-directory", str(tmp / "decoded"), "--output-directory", str(tmp / "output"),
            "--temp-directory", str(tmp / "comfy_temp")]
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    out = (tmp / "server.stdout.log").open("w", encoding="utf-8")
    err = (tmp / "server.stderr.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(args, cwd=comfy_root, stdin=subprocess.DEVNULL, stdout=out,
                            stderr=err, creationflags=flags)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Isolated ComfyUI exited {proc.returncode} before readiness.")
            ready = False
            try:
                obj = _json_http(base, "/system_stats", timeout=1.0)
                if isinstance(obj.get("system"), dict):
                    nodes = _json_http(base, "/object_info/UpscaleModelLoader", timeout=4.0)
                    loader = nodes.get("UpscaleModelLoader", {})
                    field = loader.get("input", {}).get("required", {}).get("model_name", [[]])
                    candidates = (field[0] if isinstance(field, list) and field and isinstance(field[0], list)
                                  else field[1].get("options", [])
                                  if isinstance(field, list) and len(field) > 1 and field[0] == "COMBO"
                                  and isinstance(field[1], dict) else [])
                    if model_name not in candidates:
                        raise RuntimeError("ComfyUI cannot register selected local model file. "
                                           f"API loader present={bool(loader)}, advertised={str(candidates)[:280]}.")
                    ready = True
            except (ConnectionError, TimeoutError, error.URLError, OSError):
                time.sleep(0.5)
            # Do not yield inside the readiness exception handler. Exceptions
            # raised by downstream processing (especially OSError) must escape
            # unchanged rather than becoming 'generator didn't stop after throw'.
            if ready:
                yield base
                return
        raise TimeoutError("Isolated ComfyUI startup exceeded bounded readiness timeout.")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=8)
        out.close()
        err.close()


def _process_frame(base: str, frame: Path, *, output_root: Path, output_width: int,
                   output_height: int, prefix: str, timeout: int,
                   cancel: Callable[[], bool], model_name: str = MODEL_NAME,
                   source_preservation: float = 0.0) -> Path:
    """Single-frame compatibility wrapper for the bounded batch renderer."""
    return _process_batch(base, [frame], output_root=output_root,
                          output_width=output_width, output_height=output_height,
                          prefix=prefix, timeout=timeout, cancel=cancel,
                          model_name=model_name, source_preservation=source_preservation)[0]


def _process_batch(base: str, frames: list[Path], *, output_root: Path,
                   output_width: int, output_height: int, prefix: str,
                   timeout: int, cancel: Callable[[], bool],
                   model_name: str = MODEL_NAME, source_preservation: float = 0.0) -> list[Path]:
    """Run the model exactly once for one small ordered frame batch."""
    graph = _upscale_batch_graph([frame.name for frame in frames],
                                 output_width=output_width, output_height=output_height,
                                 output_prefix=prefix, model_name=model_name,
                                 source_preservation=source_preservation)
    answer = _json_http(base, "/prompt", data={"prompt": graph, "client_id": "spiffy-local"}, timeout=20)
    prompt_id = answer.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError("ComfyUI rejected the real upscaling graph: " + str(answer)[:800])
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if cancel():
            raise RuntimeError("User cancelled ComfyUI frame processing.")
        history = _json_http(base, "/history/" + parse.quote(prompt_id, safe=""), timeout=8)
        entry = history.get(prompt_id)
        if isinstance(entry, dict):
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError("ComfyUI model failed: " + str(status.get("messages") or status)[:900])
            outputs = entry.get("outputs") or {}
            generated = (outputs.get("5") or {}).get("images") or []
            if generated:
                if len(generated) != len(frames):
                    raise RuntimeError(f"ComfyUI output frame coverage disagreed with submitted batch: {len(generated)} vs {len(frames)}.")
                output_root = output_root.resolve(strict=True)
                resolved: list[Path] = []
                for record in generated:
                    if record.get("type") != "output":
                        raise RuntimeError("ComfyUI result was not a job output frame.")
                    name, subfolder = record.get("filename"), record.get("subfolder", "")
                    if (not isinstance(name, str) or not name.endswith(".png") or not isinstance(subfolder, str)
                            or Path(name).name != name or any(part in {"..", "."} for part in Path(subfolder).parts)):
                        raise RuntimeError("ComfyUI returned an unsafe output filename.")
                    output = (output_root / subfolder / name).resolve(strict=True)
                    if not output.is_relative_to(output_root) or not output.is_file() or output.is_symlink():
                        raise RuntimeError("ComfyUI result is outside the isolated output directory.")
                    with output.open("rb") as f:
                        magic = f.read(8)
                    if magic != b"\x89PNG\r\n\x1a\n":
                        raise RuntimeError("ComfyUI produced a non-PNG image frame.")
                    resolved.append(output)
                if len(set(resolved)) != len(resolved):
                    raise RuntimeError("ComfyUI returned duplicate output frames.")
                return resolved
            if status.get("completed"):
                raise RuntimeError("ComfyUI completed without an image output.")
        time.sleep(0.35)
    raise TimeoutError("ComfyUI frame inference exceeded the per-frame deadline.")


def _preview_guard(input_frame: Path, output_frame: Path,
                   width: int, height: int) -> tuple[bytes, bytes]:
    """Reject corrupted frames and large global color/exposure changes."""
    from PIL import Image, ImageStat

    with Image.open(output_frame) as image:
        if image.size != (width, height):
            raise RuntimeError("ComfyUI returned incorrect output dimensions.")
        image.load()
        small_out = image.convert("RGB").resize((80, 80), Image.Resampling.BILINEAR)
    with Image.open(input_frame) as image:
        small_src = image.convert("RGB").resize((80, 80), Image.Resampling.BILINEAR)
    dst = ImageStat.Stat(small_out).mean
    src = ImageStat.Stat(small_src).mean
    if any(abs(a - b) > 17 for a, b in zip(src, dst)):
        raise RuntimeError("ComfyUI altered the source's global color or brightness beyond the accepted guard.")
    return small_src.tobytes(), small_out.tobytes()


def _temporal_guard(previous: tuple[bytes, bytes],
                    current: tuple[bytes, bytes]) -> None:
    """Catch large AI-only frame flashes without blocking normal scene cuts."""
    from PIL import Image, ImageChops, ImageStat

    if len(previous) != 2 or len(current) != 2 or any(len(buf) != 80 * 80 * 3 for buf in (*previous, *current)):
        raise ValueError("Temporal check needs complete 80x80 RGB previews.")

    def change(first: bytes, second: bytes) -> float:
        a, b = Image.frombytes("RGB", (80, 80), first), Image.frombytes("RGB", (80, 80), second)
        return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3

    source_delta = change(previous[0], current[0])
    enhanced_delta = change(previous[1], current[1])
    if source_delta <= 0.6 and enhanced_delta > 2.8:
        raise RuntimeError("ComfyUI changed nearly static source texture between frames.")
    if enhanced_delta > max(20.0, source_delta * 2.5 + 6.0):
        raise RuntimeError("ComfyUI introduced a large frame-to-frame change absent from the original source.")


def _validate_source(info: dict[str, Any], target_long_edge: int,
                     model: ModelSpec | None = None,
                     region_mode: str = "full") -> tuple[int, int, str]:
    from quality_advisor import _even_aspect_size, _unqualified_source_reasons

    width, height = int(info.get("width") or 0), int(info.get("height") or 0)
    model = model if model is not None else get_model("realesrgan_x4plus")
    if get_model(model.id) != model:
        raise ValueError("Untrusted model specification.")
    if region_mode not in {"full", "motioncoach_face"}:
        raise ValueError("Unqualified selective enhancement region.")
    if width < 64 or height < 64:
        raise ValueError("Invalid source dimensions.")
    if region_mode == "full" and width * height * model.native_scale**2 > MAX_NATIVE_GAN_PIXELS:
        raise ValueError(f"{model.id} native tensor exceeds the qualified 10 GB GPU pixel budget.")
    issues = _unqualified_source_reasons(info, width, height)
    if issues:
        raise ValueError("ComfyUI source needs separate qualification: " + ", ".join(issues))
    rate = str(info.get("fps_ratio") or "")
    try:
        fps = Fraction(rate)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("ComfyUI input requires an exact rational source frame rate.") from exc
    if not 1 <= fps <= 60:
        raise ValueError("ComfyUI currently supports source-frame-rate 1–60 FPS, without invented frames.")
    if info.get("frame_rate_evidence") != "presentation-timestamps" or info.get("video_pts_uniform") is not True:
        raise ValueError("ComfyUI requires verified uniform decoded presentation timestamps, not just an FPS header.")
    try:
        count = int(info["video_pts_count"])
        first = Fraction(str(info["video_pts_start"]))
        interval = Fraction(str(info["video_pts_average_interval"]))
        last = Fraction(str(info["video_pts_end_exclusive"]))
        tick = Fraction(str(info["video_pts_time_base"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("ComfyUI requires an exact decoded source frame clock and final endpoint.") from exc
    if (count < 2 or tick <= 0 or abs(first) > tick
            or abs(interval - 1 / fps) > tick
            or abs(last - Fraction(count, 1) / fps) > 2 * tick):
        raise ValueError("ComfyUI source clock cannot be represented without retiming or shifting frames.")
    for track in info.get("audio_streams") or []:
        try:
            audio_rate = int(track["sample_rate"])
            start = Fraction(str(track.get("start_time") or "0"))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("ComfyUI cannot establish source audio synchronization.") from exc
        if audio_rate <= 0 or abs(start) > Fraction(1, audio_rate):
            raise ValueError("ComfyUI does not yet qualify audio streams with a nonzero start offset.")
    duration = float(info.get("duration") or 0)
    if not 0 < duration <= MAX_DURATION_SECONDS:
        raise ValueError(f"ComfyUI route currently accepts finished clips of at most {MAX_DURATION_SECONDS} seconds.")
    if type(target_long_edge) is not int or not max(width, height) <= target_long_edge <= 7680:
        raise ValueError("ComfyUI requested long edge must not downscale the original or exceed 7680.")
    if target_long_edge > model.native_scale * max(width, height):
        raise ValueError(
            f"The selected {model.native_scale}x model cannot produce native AI detail at that output resolution.")
    out_w, out_h = _even_aspect_size(width, height, target_long_edge)
    if out_w * out_h > MAX_OUTPUT_PIXELS:
        raise ValueError("ComfyUI result exceeds the bounded final pixel budget.")
    return out_w, out_h, f"{fps.numerator}/{fps.denominator}"


def _verify_reconstructed_clock(source_info: dict[str, Any],
                                candidate_info: dict[str, Any], expected_fps: str,
                                *, phase: str) -> None:
    """Check frame-start origin, cadence and final picture endpoint after encoding."""
    try:
        rate = Fraction(expected_fps)
        source_count = int(source_info["video_pts_count"])
        source_start = Fraction(str(source_info["video_pts_start"]))
        source_end = Fraction(str(source_info["video_pts_end_exclusive"]))
        source_tick = Fraction(str(source_info["video_pts_time_base"]))
        candidate_count = int(candidate_info["video_pts_count"])
        candidate_start = Fraction(str(candidate_info["video_pts_start"]))
        candidate_end = Fraction(str(candidate_info["video_pts_end_exclusive"]))
        candidate_step = Fraction(str(candidate_info["video_pts_average_interval"]))
        candidate_tick = Fraction(str(candidate_info["video_pts_time_base"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"{phase} is missing exact decoded presentation-clock evidence.") from exc
    tolerance = 2 * max(source_tick, candidate_tick)
    if (candidate_info.get("frame_rate_evidence") != "presentation-timestamps"
            or candidate_info.get("video_pts_uniform") is not True
            or candidate_count != source_count
            or abs(candidate_start - source_start) > tolerance
            or abs(candidate_step - 1 / rate) > candidate_tick
            or abs(candidate_end - source_end) > tolerance):
        raise RuntimeError(f"{phase} changed the source frame clock or final presentation endpoint.")


def _preflight_frame_budget(info: dict[str, Any], fps: str, max_frames: int) -> None:
    """Reject oversized work before writing thousands of PNG files to disk."""
    if type(max_frames) is not int or not 1 <= max_frames <= MAX_FRAMES:
        raise ValueError("Invalid maximum frame budget.")
    header = info.get("video_frame_count")
    if header not in (None, "", "0", 0, "N/A"):
        if type(header) not in (int, str) or not str(header).isdigit():
            raise ValueError("Video frame count metadata must be a bounded integer.")
        if int(header) > max_frames:
            raise ValueError("ComfyUI source frame count exceeds the configured budget.")
    duration = float(info.get("duration") or 0)
    expected = duration * float(Fraction(fps))
    if not math.isfinite(expected) or expected > max_frames + 3:
        raise ValueError("ComfyUI source duration and frame rate exceed the frame budget.")


def _resource_admission(info: dict[str, Any], width: int, height: int, *,
                        available_disk_bytes: int, snapshot: dict[str, Any],
                        native_scale: int = 4,
                        native_source_pixels_override: int | None = None) -> dict[str, Any]:
    """Fail early instead of filling the drive or running the 10 GB GPU out of memory."""
    source_pixels = int(info["width"]) * int(info["height"])
    frames = int(info["video_pts_count"])
    output_pixels = width * height
    if type(native_scale) is not int or native_scale not in (2, 4):
        raise ValueError("Unsupported model-native tensor scale.")
    if native_source_pixels_override is not None and (
            type(native_source_pixels_override) is not int
            or not 32 * 32 <= native_source_pixels_override <= source_pixels):
        raise ValueError("Selective native-tensor pixel budget has invalid source bounds.")
    native_pixels = (native_source_pixels_override or source_pixels) * native_scale**2
    if min(source_pixels, output_pixels, frames) <= 0:
        raise ValueError("ComfyUI resource admission needs positive source/output geometry and frame count.")
    # Conservative, bounded raw RGB input + output PNG and ProRes scratch.
    # PNGs can compress well; the guard intentionally does not assume they do.
    estimated_scratch = frames * (source_pixels * 3 + output_pixels * 5)
    required_disk = estimated_scratch + 5 * 1024**3
    if type(available_disk_bytes) is not int or available_disk_bytes < required_disk:
        raise RuntimeError("ComfyUI scratch disk headroom is too low for a safe full-frame run.")
    free_mib = snapshot.get("gpu_memory_free_mib")
    # The model allocates its own native x2/x4 tensors BEFORE the ImageScale
    # stage, which may be much larger than a requested 4K final raster.
    required_gpu_mib = 6144 if max(output_pixels, native_pixels) > 3840 * 2160 else 3584
    if type(free_mib) not in (int, float) or not math.isfinite(free_mib) or free_mib < required_gpu_mib:
        raise RuntimeError(f"ComfyUI GPU memory admission needs at least {required_gpu_mib} MiB free.")
    ram = snapshot.get("ram_available_bytes")
    if type(ram) is int and ram < 4 * 1024**3:
        raise RuntimeError("ComfyUI system memory headroom is below the safe 4 GiB threshold.")
    return {"estimated_scratch_bytes": estimated_scratch,
            "native_tensor_pixels": native_pixels,
            "native_scale": native_scale,
            "required_free_disk_bytes": required_disk,
            "observed_free_gpu_mib": float(free_mib),
            "required_free_gpu_mib": required_gpu_mib,
            "scope": "conservative_admission_not_measured_peak"}


def _verify_full_decode(candidate: Path, ffmpeg: str, *,
                        duration_seconds: float,
                        cancel: Callable[[], bool],
                        run_command: Callable[..., Any] = _run) -> float:
    """Decode every delivered video frame and every audio track to null.

    FFprobe metadata and sampled timestamps are not a full corruption check.
    This exact-run verification creates no new media and is a mandatory gate
    before a new ComfyUI delivery becomes visible in the user's output folder.
    """
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError("Full-decode verification requires the private final candidate.")
    if (type(duration_seconds) not in (float, int)
            or not math.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError("Full-decode verification needs positive source duration.")
    if cancel():
        raise RuntimeError("ComfyUI final decode cancelled before verification.")
    timeout = min(3600, max(120, int(math.ceil(duration_seconds * 8))))
    started = time.monotonic()
    run_command([
        ffmpeg, "-hide_banner", "-nostdin", "-v", "error",
        "-xerror", "-err_detect", "explode",
        "-threads", "4", "-i", str(candidate),
        "-map", "0:v:0", "-map", "0:a?", "-f", "null", "-",
    ], timeout=timeout, cancel=cancel)
    return round(time.monotonic() - started, 3)


def _exact_audio_stream_args(info: dict[str, Any]) -> list[str]:
    """Keep each audio stream's declared valid sample extent, not AAC padding.

    Decoder output can include padded trailing samples even when the source's
    MP4 edit list says they are not valid audio. Trim *each* track in decoded
    sample units and reset the already-validated zero origin before PCM mux.
    This is not an audio resampler or a request to pad missing source samples.
    """
    count = int(info.get("audio_stream_count") or 0)
    if count == 0:
        return ["-map", "0:v:0", "-an"]
    tracks = info.get("audio_streams")
    if not isinstance(tracks, list) or len(tracks) != count:
        raise ValueError("Cannot preserve all source audio tracks without exact per-track metadata.")
    graphs: list[str] = []
    maps = ["-map", "0:v:0"]
    for index, track in enumerate(tracks):
        try:
            samples = int(track["sample_count"])
            rate = int(track["sample_rate"])
            channels = int(track["channels"])
            origin = Fraction(str(track.get("start_time") or "0"))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Audio track {index+1} has no exact, usable sample evidence.") from exc
        if (samples < 1 or rate < 8000 or channels < 1
                or Fraction(str(track["sample_count"])) != samples
                or abs(origin) > Fraction(1, rate)):
            raise ValueError(f"Audio track {index+1} has unsafe sample count or start offset.")
        graphs.append(f"[1:a:{index}]atrim=start_sample=0:end_sample={samples},asetpts=N/SR/TB[a{index}]")
        maps.extend(["-map", f"[a{index}]"])
    return ["-filter_complex", ";".join(graphs), *maps]


def _publish_verified_candidate(candidate: Path, destination_base: Path, source: Path,
                                job_id: str, cancel: Callable[[], bool]) -> Path:
    """Atomic collision-safe publication of media already fully verified.

    The lock belongs to the existing Final Delivery publication owner. No
    unverified candidate or partially copied cross-volume file is made public.
    """
    from shared.job_paths import collision_safe_path
    from workflow.atomic_io import artifact_publication_lease

    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError("Verified delivery candidate must be an ordinary existing file.")
    if candidate.resolve() == source.resolve():
        raise ValueError("Cannot publish a source video as its own candidate.")
    destination_base.parent.mkdir(parents=True, exist_ok=True)
    with artifact_publication_lease(destination_base, timeout_seconds=30):
        if cancel():
            raise RuntimeError("ComfyUI job cancelled before publication.")
        output = collision_safe_path(destination_base)
        if output.resolve(strict=False) == source.resolve():
            raise ValueError("Output would alias the original source.")
        if output.exists() or output.is_symlink():
            raise FileExistsError("ComfyUI selected an occupied final-output path.")
        if candidate.stat().st_dev == destination_base.parent.stat().st_dev:
            os.replace(candidate, output)
        else:
            staged = destination_base.parent / f".{destination_base.stem}.{job_id}.part.mp4"
            if staged.exists() or staged.is_symlink():
                raise FileExistsError("ComfyUI cross-volume staging path is occupied.")
            try:
                with candidate.open("rb") as read, staged.open("xb") as write:
                    shutil.copyfileobj(read, write, length=8 * 1024 * 1024)
                    write.flush()
                    os.fsync(write.fileno())
                with candidate.open("rb") as first, staged.open("rb") as second:
                    if hashlib.file_digest(first, "sha256").digest() != hashlib.file_digest(second, "sha256").digest():
                        raise RuntimeError("Cross-volume ComfyUI delivery copy failed checksum verification.")
                if cancel():
                    raise RuntimeError("ComfyUI job cancelled before final file publication.")
                os.replace(staged, output)
            finally:
                if staged.exists():
                    staged.unlink()
        return output


def run_comfyui_delivery(source: Path, *, jobs_root: Path, output_root: Path | None = None,
                         job_id: str = "", target_long_edge: int = 3840,
                         crf: int = 16, preset: str = "slow",
                         comfy_root: Path = DEFAULT_COMFY_ROOT,
                         max_frames: int = MAX_FRAMES,
                         model_id: str = DEFAULT_MODEL_POLICY,
                         source_preservation: float = 0.0,
                         compression_strategy: str = "fixed",
                         region_mode: str = "full",
                         motioncoach_pose_root: Path | None = None,
                         motioncoach_source_binding_id: str = "") -> tuple[int, dict[str, Any]]:
    """One actual ComfyUI GAN + source audio + single final lossy encode.

    Uses the existing ProgressWriter, SharedGpuLock, final_mp4/verify_video,
    and collision-safe job/output ownership. No new service or output watcher.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(SPIFFY) not in sys.path:
        sys.path.insert(0, str(SPIFFY))
    from shared.ffmpeg_utils import standard_ffmpeg, standard_ffprobe
    from shared.media_probe import probe_video, verify_video
    from shared.progress_writer import ProgressWriter, atomic_write_json
    from shared.gpu_lock import SharedGpuLock
    from shared.job_paths import make_job_id
    from video_upscale import final_mp4, validate_intermediate_audio, validate_final_delivery
    from workflow.workflow_hashing import file_sample_fingerprint
    from delivery_insights import resource_snapshot

    if type(crf) is not int or not 0 <= crf <= 25 or preset not in {"medium", "slow", "slower"}:
        raise ValueError("ComfyUI output must use an admitted high-quality H.264 preset.")
    if type(max_frames) is not int or not 1 <= max_frames <= MAX_FRAMES:
        raise ValueError("Invalid ComfyUI frame-count budget.")
    if type(compression_strategy) is not str or compression_strategy not in {"fixed", "measured"}:
        raise ValueError("Compression strategy must be fixed or measured.")
    if region_mode not in {"full", "motioncoach_face"}:
        raise ValueError("Only full-frame or published MotionCoach face ROI execution is supported.")
    if region_mode == "motioncoach_face" and (
            motioncoach_pose_root is None or not motioncoach_source_binding_id):
        raise ValueError("Selective face mode requires explicit published MotionCoach pose evidence and source binding.")
    if region_mode == "full" and (motioncoach_pose_root is not None or motioncoach_source_binding_id):
        raise ValueError("MotionCoach evidence may only be passed with explicitly selected face ROI mode.")
    if (type(source_preservation) not in (int, float)
            or not math.isfinite(source_preservation) or not 0 <= source_preservation <= 0.25):
        raise ValueError("Source-preservation blend must be between 0 and 0.25.")
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in {".mp4", ".mov", ".mkv", ".m4v", ".webm"}:
        raise ValueError("A supported finished source video is required.")
    source_identity = file_sample_fingerprint(source).to_dict()
    if not source_identity.get("exists") or not source_identity.get("media_uid"):
        raise ValueError("Cannot establish a stable source-content identity.")
    root = Path(jobs_root).expanduser().resolve()
    selected_job = job_id or make_job_id("smart_finish_comfyui")
    # Command Palette creates its manifest before calling us. Never overwrite
    # another attempt's ComfyUI result, even if that attempt was interrupted.
    if (root / selected_job / "comfyui_delivery_result.json").exists():
        raise RuntimeError("ComfyUI job already claimed; choose a new job identity.")
    writer = ProgressWriter(root, selected_job,
                            "Smart Finish · ComfyUI", "video_upscale")
    temporary = writer.temp_dir / "comfyui_gan"
    incoming = temporary / "decoded"
    outgoing = temporary / "enhanced"
    incoming.mkdir(parents=True, exist_ok=True)
    outgoing.mkdir(parents=True, exist_ok=True)
    ffmpeg, ffprobe = standard_ffmpeg(), standard_ffprobe()
    result: dict[str, Any] = {"schema": "spiffy_comfyui_delivery_result_v1", "job_id": writer.job_id,
                              "state": "running", "backend": "comfyui_frame_gan_v1",
                              "source_path": str(source), "output": "", "errors": [],
                              "frame_count": None, "working_resolution": None,
                              "stage_seconds": {}}
    output_path: Path | None = None
    started = time.monotonic()
    try:
        ok, info, reason = verify_video(source, ffprobe, inspect_timestamps=True)
        if not ok:
            raise RuntimeError("Source failed exact decode/timestamp preflight: " + str(reason))
        chosen = select_model(model_id, source_long_edge=max(int(info["width"]), int(info["height"])),
                              target_long_edge=target_long_edge)
        _qualified_checkpoint(comfy_root, chosen.id)
        result["model"] = chosen.public_info()
        result["model_policy"] = model_id
        result["source_preservation"] = float(source_preservation)
        out_w, out_h, fps = _validate_source(info, target_long_edge, chosen, region_mode=region_mode)
        face_plan = None
        native_roi_pixels = None
        if region_mode == "motioncoach_face":
            from motioncoach_face_bridge import load_published_face_plan

            face_plan = load_published_face_plan(
                source, Path(motioncoach_pose_root),
                source_binding_id=motioncoach_source_binding_id, info=info,
            )
            native_roi_pixels = max(
                (x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in face_plan["boxes_xyxy"])
            if native_roi_pixels * chosen.native_scale**2 > MAX_NATIVE_GAN_PIXELS:
                raise ValueError("MotionCoach facial ROI still exceeds safe model-native tensor limits.")
            result["region"] = {
                "mode": region_mode, "schema": face_plan["schema"],
                "frame_count": face_plan["frame_count"],
                "pose_source": face_plan["pose_source"],
                "identity_claim": "none", "model_scope": "neural_face_region_only",
                "background": "source_lanczos_no_neural_reconstruction",
            }
        else:
            result["region"] = {"mode": "full", "model_scope": "full_frame"}
        if source_preservation and out_w * out_h > 3840 * 2160:
            raise ValueError("Source-preservation blend is not qualified above 4K on this GPU.")
        _preflight_frame_budget(info, fps, max_frames)
        result["resource_admission"] = _resource_admission(
            info, out_w, out_h,
            available_disk_bytes=int(shutil.disk_usage(writer.temp_dir).free),
            snapshot=resource_snapshot(),
            native_scale=chosen.native_scale,
            native_source_pixels_override=native_roi_pixels,
        )
        writer.update(state="running", phase="preflight", message="Qualifying CUDA model and source frames")
        # A real decoded image sequence, never duplicated frames or guessed FPS.
        decode_started = time.monotonic()
        _run([ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
              "-i", str(source), "-map", "0:v:0", "-vsync", "0",
              "-pix_fmt", "rgb24", "-start_number", "0", str(incoming / "frame_%06d.png")],
             timeout=600, cancel=lambda: writer.cancel_requested)
        result["stage_seconds"]["decode"] = round(time.monotonic() - decode_started, 3)
        input_frames = sorted(incoming.glob("frame_*.png"))
        if not input_frames or len(input_frames) > max_frames:
            raise RuntimeError("ComfyUI decoded frame count is zero or exceeds the admitted job budget.")
        recorded = info.get("video_frame_count")
        if recorded not in (None, "", "N/A", "0") and int(recorded) != len(input_frames):
            raise RuntimeError("Decoded input frame count disagrees with the source stream header.")
        result["frame_count"] = len(input_frames)
        result["working_resolution"] = [out_w, out_h]
        batch_size = (1 if face_plan is not None else
                      _batch_size(int(info["width"]), int(info["height"]), out_w, out_h,
                                  available_gpu_mib=float(result["resource_admission"]["observed_free_gpu_mib"]),
                                  native_scale=chosen.native_scale))
        result["processing_strategy"] = {"schema": "comfyui_bounded_batch_v1",
                                         "batch_size": batch_size,
                                         "native_scale": chosen.native_scale,
                                         "execution": "real_ComfyUI_ImageBatch_or_single_frame",
                                         "model_cached_per_private_server": True,
                                         "unique_frames_inferred": 0,
                                         "exact_source_frames_reused": 0}
        atomic_write_json(writer.job_dir / "comfyui_delivery_result.json", result)
        with SharedGpuLock(root, writer, "gpu_render", "Topaz/AI GPU"):
            # Resource observations taken before media decode can be stale,
            # especially when other user agents use the GPU concurrently.
            # Re-admit with the *same* model-native tensor while holding the
            # incumbent shared lock, immediately before launching ComfyUI.
            result["gpu_admission_after_lock"] = _resource_admission(
                info, out_w, out_h,
                available_disk_bytes=int(shutil.disk_usage(writer.temp_dir).free),
                snapshot=resource_snapshot(),
                native_scale=chosen.native_scale,
                native_source_pixels_override=native_roi_pixels,
            )
            startup = time.monotonic()
            with _owned_comfyui(comfy_root, temporary, model_name=chosen.filename) as base:
                result["stage_seconds"]["isolated_server_startup"] = round(time.monotonic() - startup, 3)
                if batch_size > 1:
                    try:
                        batch_nodes = _json_http(base, "/object_info/ImageBatch", timeout=5.0)
                        if "ImageBatch" not in batch_nodes:
                            batch_size = 1
                    except (OSError, ValueError, error.URLError, TimeoutError):
                        batch_size = 1
                    if batch_size == 1:
                        result["processing_strategy"]["batch_size"] = 1
                        result["processing_strategy"]["fallback"] = "ImageBatch_node_unavailable"
                inference_start = time.monotonic()
                previous_preview: tuple[bytes, bytes] | None = None
                frame_cache: dict[tuple[int, str], Path] = {}
                if face_plan is not None:
                    from selective_face_enhancement import prepare_face_crop, compose_face_roi

                    # ComfyUI LoadImage only sees the exact isolated
                    # --input-directory (temporary/decoded). Keep per-frame
                    # face_<ordinal>.png crops beside the decoded originals,
                    # never in an unregistered sibling folder.
                    roi_dir = incoming
                    roi_cache_dir = temporary / "face_verified_cache"
                    roi_cache_dir.mkdir(parents=True, exist_ok=True)
                    face_ai_cache: dict[tuple[int, str, int, int], Path] = {}
                    for index, frame in enumerate(input_frames):
                        if writer.cancel_requested:
                            raise RuntimeError("MotionCoach selective face processing was cancelled.")
                        writer.update(
                            phase="comfyui_face_roi",
                            message=f"Pose-guided neural face crop {index+1}/{len(input_frames)}",
                            percent=round((index / len(input_frames)) * 80, 1),
                        )
                        crop = roi_dir / f"face_{index:06d}.png"
                        rect = prepare_face_crop(
                            frame, face_plan["boxes_xyxy"][index],
                            source_size=(int(info["width"]), int(info["height"])),
                            output_size=(out_w, out_h), crop_path=crop,
                        )
                        region_w, region_h = rect[2] - rect[0], rect[3] - rect[1]
                        size, digest = _exact_frame_key(crop)
                        cache_key = (size, digest, region_w, region_h)
                        rendered = face_ai_cache.get(cache_key)
                        if rendered is None:
                            generated = _process_frame(
                                base, crop, output_root=temporary / "output",
                                output_width=region_w, output_height=region_h,
                                prefix=f"Spiffy_2026_{writer.job_id.replace('-', '_')}_{index:06d}"[:96],
                                timeout=240, cancel=lambda: writer.cancel_requested,
                                model_name=chosen.filename,
                            )
                            rendered = roi_cache_dir / f"face_ai_{index:06d}.png"
                            os.replace(generated, rendered)
                            face_ai_cache[cache_key] = rendered
                            result["processing_strategy"]["unique_frames_inferred"] += 1
                        else:
                            if not rendered.is_file():
                                raise RuntimeError("An exact previously validated face-region result vanished.")
                            result["processing_strategy"]["exact_source_frames_reused"] += 1
                        out_file = outgoing / f"frame_{index:06d}.png"
                        compose_face_roi(
                            frame, rendered, out_file, rect,
                            output_size=(out_w, out_h),
                            source_preservation=source_preservation,
                        )
                        preview = _preview_guard(frame, out_file, out_w, out_h)
                        if previous_preview is not None:
                            _temporal_guard(previous_preview, preview)
                        previous_preview = preview
                        crop.unlink()
                        frame.unlink()
                    result["stage_seconds"]["neural_inference_and_validation"] = round(
                        time.monotonic() - inference_start, 3,
                    )
                else:
                    # Preserve existing full-frame batching/exact-frame reuse.
                    for start in range(0, len(input_frames), batch_size):
                        if writer.cancel_requested:
                            raise RuntimeError("ComfyUI job cancelled by user.")
                        group = input_frames[start:start + batch_size]
                        writer.update(phase="comfyui_upscale",
                                      message=f"ComfyUI {chosen.id}: frames {start+1}-{start+len(group)}/{len(input_frames)}",
                                      percent=round((start / len(input_frames)) * 80, 1))
                        source_keys = [_exact_frame_key(frame) for frame in group]
                        unseen: list[tuple[Path, tuple[int, str]]] = []
                        pending: set[tuple[int, str]] = set()
                        for frame, key in zip(group, source_keys):
                            if key not in frame_cache and key not in pending:
                                unseen.append((frame, key))
                                pending.add(key)
                        new_outputs: dict[tuple[int, str], Path] = {}
                        if unseen:
                            generated = _process_batch(
                                base, [frame for frame, _ in unseen],
                                output_root=temporary / "output",
                                output_width=out_w, output_height=out_h,
                                prefix=f"Spiffy_2026_{writer.job_id.replace('-', '_')}_{start:06d}"[:96],
                                timeout=240 * len(unseen),
                                cancel=lambda: writer.cancel_requested,
                                model_name=chosen.filename,
                                source_preservation=source_preservation,
                            )
                            new_outputs = {key: rendered for (_, key), rendered in zip(unseen, generated)}
                            if len(new_outputs) != len(unseen):
                                raise RuntimeError("ComfyUI exact-frame batch did not return each unique model result.")
                        for i, (frame, key) in enumerate(zip(group, source_keys), start=start):
                            out_file = outgoing / f"frame_{i:06d}.png"
                            if out_file.exists():
                                raise RuntimeError("ComfyUI output frame already exists in this job.")
                            if key in new_outputs:
                                os.replace(new_outputs.pop(key), out_file)
                                frame_cache[key] = out_file
                                result["processing_strategy"]["unique_frames_inferred"] += 1
                            else:
                                source_output = frame_cache.get(key)
                                if source_output is None or not source_output.is_file():
                                    raise RuntimeError("A previously verified exact AI frame is missing.")
                                try:
                                    os.link(source_output, out_file)
                                except OSError:
                                    shutil.copyfile(source_output, out_file)
                                result["processing_strategy"]["exact_source_frames_reused"] += 1
                            preview = _preview_guard(frame, out_file, out_w, out_h)
                            if previous_preview is not None:
                                _temporal_guard(previous_preview, preview)
                            previous_preview = preview
                            # Release one processed input promptly. With 1,800
                            # frames this avoids retaining every source PNG alongside
                            # the growing 4K/6K output sequence.
                            if frame.parent == incoming and frame.name == f"frame_{i:06d}.png":
                                frame.unlink()
                result["stage_seconds"]["neural_inference_and_validation"] = round(time.monotonic() - inference_start, 3)
        if len(list(outgoing.glob("frame_*.png"))) != len(input_frames):
            raise RuntimeError("ComfyUI output failed exact frame coverage verification.")
        # Lossless working stage retains all source audio tracks (24-bit PCM),
        # allowing the incumbent final_mp4 encoder to do one and only one lossy
        # video pass. This is not a new rendition of the original source.
        intermediate = temporary / "comfyui_enhanced.mov"
        writer.update(phase="lossless_intermediate", message="Retaining frame clock and original audio")
        intermediate_start = time.monotonic()
        _run([ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
              "-framerate", fps, "-start_number", "0", "-i", str(outgoing / "frame_%06d.png"),
              "-i", str(source), *_exact_audio_stream_args(info), "-c:v", "prores_ks",
              "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-c:a", "pcm_s24le",
              "-video_track_timescale", "60000", str(intermediate)],
             timeout=1800, cancel=lambda: writer.cancel_requested)
        result["stage_seconds"]["prores_audio_assembly"] = round(time.monotonic() - intermediate_start, 3)
        working_info = probe_video(intermediate, ffprobe, inspect_timestamps=True)
        if (int(working_info["width"]), int(working_info["height"])) != (out_w, out_h):
            raise RuntimeError("Lossless ComfyUI working intermediate has unexpected dimensions.")
        if int(working_info.get("video_frame_count") or 0) != len(input_frames):
            raise RuntimeError("ComfyUI working intermediate changed frame count.")
        _verify_reconstructed_clock(info, working_info, fps, phase="ComfyUI lossless intermediate")
        validate_intermediate_audio(info, working_info, phase="ComfyUI lossless intermediate",
                                    require_sample_extent=bool(info.get("audio_stream_count")))
        selected_crf = crf
        if compression_strategy == "measured":
            from compression_optimizer import calibrate_h264

            writer.update(phase="compression_calibration",
                          message="Measuring a bounded H.264 compression sample against the lossless master",
                          percent=82)
            calibration = calibrate_h264(
                intermediate, ffmpeg=ffmpeg, temp_root=temporary,
                duration=float(info["duration"]), fps=fps, width=out_w,
                height=out_h, baseline_crf=crf,
                cancel=lambda: writer.cancel_requested, run_command=_run,
            )
            result["compression_calibration"] = calibration
            selected_crf = int(calibration["selected_crf"])
            result["stage_seconds"]["compression_calibration"] = float(calibration.get("pilot_elapsed_seconds", 0))
        destination = Path(output_root) if output_root is not None else source.parent / "_SpiffyDelivery"
        destination = destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        variant_suffix = f"_sourceblend{round(100 * source_preservation):02d}" if source_preservation else ""
        if face_plan is not None:
            variant_suffix += "_motioncoach_faceonly"
        if compression_strategy == "measured":
            variant_suffix += f"_measured_crf{selected_crf}"
        destination_base = destination / f"{source.stem}__smart_finish_{chosen.id}_{out_w}x{out_h}{variant_suffix}.mp4"
        candidate = temporary / "comfyui_verified_candidate.mp4"
        writer.update(phase="final_h264", message="One quality-preserving H.264 delivery encode", percent=85)
        encode_start = time.monotonic()
        final_mp4(intermediate, candidate, ffmpeg, working_info, "keep_source_aspect",
                  out_w, out_h, selected_crf, preset, writer, duration_s=float(info.get("duration") or 0),
                  frame_rate={"mode": "cfr", "fps": fps}, original_info=info)
        result["stage_seconds"]["h264_final_encode"] = round(time.monotonic() - encode_start, 3)
        passed, final_info, reason = verify_video(candidate, ffprobe, inspect_timestamps=True)
        if not passed:
            raise RuntimeError("Final ComfyUI MP4 is not fully decodable: " + str(reason))
        validate_final_delivery(final_info, expected_width=out_w, expected_height=out_h, preset_id="")
        validate_intermediate_audio(info, final_info, phase="ComfyUI final delivery",
                                    require_sample_extent=bool(info.get("audio_stream_count")))
        _verify_reconstructed_clock(info, final_info, fps, phase="ComfyUI final delivery")
        if int(final_info.get("video_frame_count") or 0) != len(input_frames):
            raise RuntimeError("Final ComfyUI MP4 lost or invented frames.")
        writer.update(phase="final_decode_qa", percent=94,
                      message="Decoding all final video frames and audio tracks for integrity")
        result["stage_seconds"]["full_final_decode"] = _verify_full_decode(
            candidate, ffmpeg, duration_seconds=float(info["duration"]),
            cancel=lambda: writer.cancel_requested,
        )
        after_source = file_sample_fingerprint(source).to_dict()
        if (after_source.get("media_uid") != source_identity["media_uid"]
                or after_source.get("file_size") != source_identity.get("file_size")):
            raise RuntimeError("Source content changed during AI processing; refusing delivery.")
        if writer.cancel_requested:
            raise RuntimeError("ComfyUI job cancelled before verified delivery.")
        output_path = _publish_verified_candidate(
            candidate, destination_base, source, writer.job_id,
            cancel=lambda: writer.cancel_requested,
        )
        result.update({"state": "passed", "output": str(output_path), "final_outputs": [str(output_path)],
                       "elapsed_seconds": round(time.monotonic() - started, 2),
                       "compression": {"codec": "libx264", "crf": selected_crf, "preset": preset,
                                       "strategy": compression_strategy,
                                       "requested_baseline_crf": crf,
                                       "lossy_video_encodes": 1,
                                       "calibration_pilot_encodes": (
                                           len(result.get("compression_calibration", {}).get("candidate_results", []))
                                           if compression_strategy == "measured" else 0)},
                       "video": {"width": out_w, "height": out_h, "fps": fps,
                                 "frames": len(input_frames)},
                       "verification": {"source_identity": "sparse_content_rechecked",
                                        "presentation_clock": "decoded_verified",
                                        "audio_sample_extent": "verified",
                                        "final_full_decode": "pass"},
                       "quality_scope": "source_consistency_and_frame_integrity_not_temporal_vsr_perceptual_equivalence"})
        atomic_write_json(writer.job_dir / "comfyui_delivery_result.json", result)
        writer.done("ComfyUI enhanced and compressed video verified")
        return 0, result
    except Exception as exc:
        if output_path is not None and output_path.is_file():
            # Publication is deliberately AFTER all media validation. A later
            # receipt/log failure must not erase a verified user-facing file;
            # record it for reconciliation instead of silently deleting it.
            result["published_output_requires_reconciliation"] = str(output_path)
        result.update({"state": "cancelled" if writer.cancel_requested else "failed",
                       "errors": [str(exc)], "elapsed_seconds": round(time.monotonic() - started, 2)})
        atomic_write_json(writer.job_dir / "comfyui_delivery_result.json", result)
        writer.error(str(exc), "comfyui_failure")
        return 2 if writer.cancel_requested else 1, result
    finally:
        # Failed jobs can generate tens of gigabytes of PNG/ProRes data.
        # Preserve only a bounded log tail and the existing progress/result
        # JSON. Never follow a junction to delete other project files.
        if result["state"] != "passed":
            log = temporary / "server.stderr.log"
            try:
                if log.is_file() and not log.is_symlink():
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 65536))
                        tail = stream.read(65536)
                    (writer.job_dir / "comfyui_server_stderr_tail.log").write_bytes(tail)
            except OSError:
                pass
        if temporary.parent.resolve() == writer.temp_dir.resolve() and not temporary.is_symlink():
            shutil.rmtree(temporary, ignore_errors=True)
