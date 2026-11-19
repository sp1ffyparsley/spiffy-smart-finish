"""Short, source-bound H.264 compression calibration against a lossless master.

No final output is created here. The existing Final Delivery encoder remains
the only producer of the finished MP4. This optional calibration may choose a
slightly smaller CRF only when short full-resolution lossless-reference pilots
pass strict VMAF/SSIM/PSNR gates. It does not purport to prove subjective
grain, fine-texture or facial quality from one objective metric.
"""
from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Callable


SCHEMA = "spiffy_measured_compression_v1"
ADMITTED_CRFS = (18, 20)
MAX_NATIVE_PIXELS = 3840 * 2160
MIN_DURATION_SECONDS = 5.0
MIN_SAMPLE_FRAMES = 8
MAX_PILOT_WALL_SECONDS = 45.0
# Deliberately conservative; a barely-passing sample is not a good reason to
# discard source grain or facial microtexture. Missing metrics never pass.
GATES = {
    "vmaf_mean": 98.2,
    "vmaf_min": 94.0,
    "float_ssim_mean": 0.996,
    "psnr_y_mean_db": 47.5,
}


def sample_windows(duration: float, fps: str, *, output_pixels: int) -> list[dict[str, float]]:
    """Select tiny bounded, non-overlapping source-relative probe windows."""
    try:
        rate = Fraction(fps)
        length = float(duration)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("Compression calibration requires a valid source timebase.") from exc
    if not math.isfinite(length) or length <= 0 or rate <= 0:
        raise ValueError("Compression calibration requires positive finite duration and rate.")
    if length < MIN_DURATION_SECONDS or rate < 12 or rate > 60 or output_pixels > MAX_NATIVE_PIXELS:
        return []
    span_frames = max(MIN_SAMPLE_FRAMES, min(24, math.ceil(float(rate) * 0.65)))
    segment_seconds = float(Fraction(span_frames, 1) / rate)
    if length < segment_seconds + 1:
        return []
    portions = (0.30, 0.70) if length >= 15 else (0.5,)
    windows = []
    for at in portions:
        center = length * at
        start_frame = max(0, math.floor((center - segment_seconds / 2) * float(rate)))
        start = float(Fraction(start_frame, 1) / rate)
        if start + segment_seconds >= length:
            start = max(0.0, length - segment_seconds - 1 / float(rate))
        if windows and start < windows[-1]["start_seconds"] + windows[-1]["duration_seconds"]:
            return windows[:1]
        windows.append({"start_seconds": round(start, 6),
                        "duration_seconds": round(segment_seconds, 6),
                        "expected_frames": span_frames})
    return windows


def parse_measurements(value: dict[str, Any], expected_frames: int) -> dict[str, float]:
    """Parse only libvmaf's actual FFmpeg results. Never extrapolate scores."""
    if not isinstance(value, dict) or type(expected_frames) is not int or expected_frames < MIN_SAMPLE_FRAMES:
        raise ValueError("VMAF sample is invalid.")
    frames = value.get("frames")
    if not isinstance(frames, list) or len(frames) != expected_frames:
        raise ValueError("VMAF frame count does not match the tested sample.")
    pooled = value.get("pooled_metrics")
    if not isinstance(pooled, dict):
        raise ValueError("VMAF pooled metrics are missing.")
    requirements = {
        "vmaf_mean": ("vmaf", "mean"),
        "vmaf_min": ("vmaf", "min"),
        "float_ssim_mean": ("float_ssim", "mean"),
        "psnr_y_mean_db": ("psnr_y", "mean"),
    }
    result = {}
    for name, (metric, field) in requirements.items():
        group = pooled.get(metric)
        if not isinstance(group, dict):
            raise ValueError(f"VMAF score is missing {metric}.")
        score = group.get(field)
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError(f"VMAF score {name} is not a finite measured value.")
        result[name] = round(float(score), 6)
    if not 0 <= result["vmaf_min"] <= result["vmaf_mean"] <= 100.1:
        raise ValueError("VMAF score is out of range.")
    if not 0 <= result["float_ssim_mean"] <= 1:
        raise ValueError("SSIM score is out of range.")
    return result


def meets_high_quality_gate(measurement: dict[str, float]) -> bool:
    return all(type(measurement.get(key)) in (float, int)
               and math.isfinite(measurement[key]) and measurement[key] >= minimum
               for key, minimum in GATES.items())


def calibrate_h264(
    intermediate: Path,
    *,
    ffmpeg: str,
    temp_root: Path,
    duration: float,
    fps: str,
    width: int,
    height: int,
    baseline_crf: int,
    cancel: Callable[[], bool],
    run_command: Callable[..., Any],
) -> dict[str, Any]:
    """Return one conservative CRF choice and bounded evidence, never a video.

    Failed or unavailable pilots safely fall back to the requested baseline.
    All actual lossy final encoding, audio/timing QA and publication are still
    performed by the incumbent Final Delivery owner after this returns.
    """
    if type(baseline_crf) is not int or not 0 <= baseline_crf <= 25:
        raise ValueError("A valid finite baseline CRF is required.")
    if type(width) is not int or type(height) is not int or min(width, height) < 64:
        raise ValueError("Invalid compression master geometry.")
    if cancel():
        raise RuntimeError("Compression calibration cancelled before beginning.")
    windows = sample_windows(duration, fps, output_pixels=width * height)
    result: dict[str, Any] = {
        "schema": SCHEMA, "selected_crf": baseline_crf,
        "baseline_crf": baseline_crf, "status": "not_run",
        "codec": "libx264", "full_resolution_measurements": False,
        "candidate_results": [], "target_gates": GATES,
        "sample_windows": windows,
        "note": "VMAF, SSIM and PSNR are objective pilot metrics, not a guarantee of visual grain or detail preservation.",
    }
    if not windows:
        result["status"] = "skipped_short_or_high_resolution_or_unqualified_rate"
        return result
    if baseline_crf <= 14:
        result["status"] = "skipped_maximum_quality_setting_preserved"
        return result
    if baseline_crf >= ADMITTED_CRFS[0]:
        result["status"] = "skipped_no_more_compressed_admitted_candidate"
        return result
    if not intermediate.is_file() or intermediate.is_symlink():
        raise ValueError("Compression reference must be the verified ordinary lossless intermediate.")
    calibration_root = Path(temp_root) / "compression_calibration"
    calibration_root.mkdir(parents=True, exist_ok=False)
    result["status"] = "baseline_retained"
    started = time.monotonic()
    deadline = started + MAX_PILOT_WALL_SECONDS
    result["full_resolution_measurements"] = True
    try:
        # All pilots are at the same native output resolution as the final
        # render. No downscaled HD score is passed off as evidence for 4K/6K.
        for crf in ADMITTED_CRFS:
            if crf <= baseline_crf:
                continue
            all_passed = True
            candidate = {"crf": crf, "samples": [], "passed": False}
            result["candidate_results"].append(candidate)
            for index, window in enumerate(windows):
                if cancel():
                    raise RuntimeError("Compression calibration cancelled by user.")
                if deadline - time.monotonic() < 3:
                    result["status"] = "time_budget_reached_prior_verified_choice_preserved"
                    return result
                clip = calibration_root / f"candidate_{crf}_{index}.mp4"
                stats = calibration_root / f"vmaf_{crf}_{index}.json"
                start = f"{window['start_seconds']:.6f}"
                length = f"{window['duration_seconds']:.6f}"
                try:
                    run_command([
                        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                        "-ss", start, "-i", str(intermediate), "-t", length,
                        "-map", "0:v:0", "-an", "-c:v", "libx264",
                        "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
                        str(clip),
                    ], timeout=min(60, max(1, int(deadline - time.monotonic()))),
                       cancel=cancel)
                    if not clip.is_file() or clip.stat().st_size <= 0:
                        raise RuntimeError("Pilot encoding yielded no local candidate.")
                    # The first libvmaf input must be the compressed
                    # (distorted) picture; the second is the original master.
                    filtergraph = (
                        "[0:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[dist];"
                        "[1:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[reference];"
                        f"[dist][reference]libvmaf=log_fmt=json:log_path={stats.name}"
                        ":feature=name=psnr|name=float_ssim"
                    )
                    run_command([
                        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
                        "-i", str(clip), "-ss", start, "-t", length,
                        "-i", str(intermediate), "-filter_complex", filtergraph,
                        "-f", "null", "-",
                    ], timeout=min(90, max(1, int(deadline - time.monotonic()))),
                       cwd=calibration_root, cancel=cancel)
                    measurements = parse_measurements(
                        json.loads(stats.read_text(encoding="utf-8")),
                        window["expected_frames"],
                    )
                    passed = meets_high_quality_gate(measurements)
                    candidate["samples"].append({
                        "window": index,
                        "size_bytes": clip.stat().st_size,
                        "scores": measurements,
                        "passed": passed,
                    })
                    if not passed:
                        all_passed = False
                        break
                except Exception as exc:
                    if cancel():
                        raise RuntimeError("Compression calibration cancelled by user.") from exc
                    result["status"] = ("pilot_unavailable_baseline_preserved"
                                        if result["selected_crf"] == baseline_crf
                                        else "higher_candidate_unavailable_prior_verified_choice_preserved")
                    result["fallback_reason"] = type(exc).__name__
                    return result
                finally:
                    for path in (clip, stats):
                        if path.is_file() and not path.is_symlink():
                            path.unlink()
            candidate["passed"] = all_passed
            if not all_passed:
                break
            result["selected_crf"] = crf
            result["status"] = "measured_high_quality_candidate"
        return result
    finally:
        result["pilot_elapsed_seconds"] = round(time.monotonic() - started, 3)
        # This directory belongs to the current job. Never delete user media,
        # model checkpoints, or another job's scratch folder.
        if calibration_root.parent.resolve() == Path(temp_root).resolve() and not calibration_root.is_symlink():
            shutil.rmtree(calibration_root, ignore_errors=True)
