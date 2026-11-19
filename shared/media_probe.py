from __future__ import annotations

import json
import hashlib
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

MAX_VIDEO_TRACK_TIMESCALE = 2_147_483_647


def pixel_format_has_alpha(value: Any) -> bool:
    """Conservatively identify source pixel formats with a transparency channel.

    The existing final H.264/ProRes 422 path does not carry alpha. Treating
    such sources as ordinary opaque video silently destroys a motion
    graphic's transparency. Paletted 8-bit formats are alpha-capable;
    without palette inspection they are treated conservatively as at-risk.
    """
    fmt = str(value or "").strip().lower()
    return fmt.startswith((
        "rgba", "bgra", "argb", "abgr", "yuva", "gbrap",
        "ya8", "ya16", "ayuv", "vuya",
    )) or fmt == "pal8"


def parse_rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    try:
        return float(Fraction(value))
    except Exception:
        return 0.0


def _set_rotation_metadata(info: dict[str, Any], raw_value: Any) -> None:
    """Retain non-integral angles and make unreadable metadata explicit."""
    try:
        if isinstance(raw_value, bool):
            raise ValueError("boolean rotation")
        rotation_value = float(raw_value)
        if not math.isfinite(rotation_value):
            raise ValueError("non-finite rotation")
    except (TypeError, ValueError, OverflowError):
        info["rotation_metadata_error"] = "unreadable or non-finite rotation metadata"
        return
    info.pop("rotation_metadata_error", None)
    info["rotation"] = int(rotation_value) if rotation_value.is_integer() else rotation_value


def video_track_timescale_for_time_base(value: str | Fraction) -> int:
    """Return an MP4/MOV integer timescale that exactly represents source PTS ticks.

    For a reduced source time base a/b seconds per tick, an output time base of
    1/b maps each source tick to exactly a output ticks. FFmpeg's MOV muxer
    accepts a signed-int timescale; values outside that bound fail closed.
    ``value`` is seconds per source PTS tick, not a frame-rate value.
    """
    if isinstance(value, bool):
        raise ValueError("Video time base must be a positive rational value.")
    try:
        time_base = value if isinstance(value, Fraction) else Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("Video time base must be a positive rational value.") from exc
    if time_base <= 0:
        raise ValueError("Video time base must be a positive rational value.")
    timescale = time_base.denominator
    if not 1 <= timescale <= MAX_VIDEO_TRACK_TIMESCALE:
        raise ValueError(
            f"Source video time base requires track timescale {timescale}, "
            f"outside FFmpeg's supported range 1..{MAX_VIDEO_TRACK_TIMESCALE}."
        )
    return timescale


def run_json(argv: list[str], timeout: int = 60) -> dict[str, Any]:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip())
    return json.loads(result.stdout or "{}")


def _audio_sample_count(stream: dict[str, Any]) -> int | str | None:
    """Resolve the container's valid audio extent into exact sample frames."""
    try:
        sample_rate = int(stream.get("sample_rate") or 0)
        if sample_rate <= 0:
            return None
        duration_ts = stream.get("duration_ts")
        time_base = Fraction(str(stream.get("time_base") or "0/1"))
        if duration_ts not in (None, "") and time_base > 0:
            samples = Fraction(int(duration_ts)) * time_base * sample_rate
        else:
            duration = stream.get("duration")
            if duration in (None, "", "N/A"):
                duration = (stream.get("tags") or {}).get("DURATION")
                if not isinstance(duration, str):
                    return None
                parts = duration.split(":")
                if len(parts) != 3:
                    return None
                seconds = Fraction(parts[0]) * 3600 + Fraction(parts[1]) * 60 + Fraction(parts[2])
            else:
                seconds = Fraction(str(duration))
            samples = seconds * sample_rate
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    if samples.denominator == 1:
        return int(samples)
    return f"{samples.numerator}/{samples.denominator}"


def _presentation_timestamp_summary(
    frames: Any,
    time_base_value: str,
    declared_frame_count: Any,
) -> dict[str, Any]:
    """Summarize decoded presentation PTS without retaining a frame-sized list."""
    try:
        time_base = Fraction(str(time_base_value))
        if time_base <= 0 or not isinstance(frames, list) or len(frames) < 2:
            raise ValueError("insufficient presentation timestamp evidence")
        pts: list[int] = []
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("malformed frame timestamp row")
            raw = frame.get("best_effort_timestamp")
            if raw in (None, "", "N/A") or isinstance(raw, bool):
                raise ValueError("missing best-effort presentation timestamp")
            if type(raw) is int:
                pts.append(raw)
            elif isinstance(raw, str):
                raw_text = raw.strip()
                if not (raw_text.isdecimal() or (raw_text.startswith("-") and raw_text[1:].isdecimal())):
                    raise ValueError("best-effort presentation timestamp must be an exact integer")
                pts.append(int(raw_text))
            else:
                raise ValueError("best-effort presentation timestamp must be an exact integer")
        if any(right <= left for left, right in zip(pts, pts[1:])):
            raise ValueError("presentation timestamps are not strictly increasing")
        if declared_frame_count is None:
            declared_count_text = ""
        elif type(declared_frame_count) is int:
            declared_count_text = str(declared_frame_count)
        elif isinstance(declared_frame_count, str):
            declared_count_text = declared_frame_count.strip()
        else:
            raise ValueError("declared frame count must be an integer, string, or unavailable")
        declared_count_verified = False
        if declared_count_text and declared_count_text.upper() != "N/A":
            if not declared_count_text.isdecimal():
                raise ValueError("declared frame count metadata is not an exact non-negative integer")
            if int(declared_count_text) != len(pts):
                raise ValueError("decoded presentation frame count disagrees with stream metadata")
            declared_count_verified = True
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise RuntimeError(f"Cannot qualify presentation timestamps: {exc}") from exc

    first = Fraction(pts[0]) * time_base
    last = Fraction(pts[-1]) * time_base
    span = last - first
    interval = span / (len(pts) - 1)
    steps = [right - left for left, right in zip(pts, pts[1:])]
    min_step_ticks = min(steps)
    max_step_ticks = max(steps)

    max_grid_error = max(
        abs(Fraction(value) * time_base - (first + index * interval))
        for index, value in enumerate(pts)
    )
    tolerance = time_base
    exact_tick_step = min_step_ticks == max_step_ticks
    precision_sufficient = time_base <= interval / 4
    # Equal integer PTS steps are exact at any time-base resolution. When
    # steps vary, only call the sequence quantized-uniform if the time base is
    # fine enough to distinguish cadence jitter from a real missing/long frame.
    uniform = exact_tick_step or (precision_sufficient and max_grid_error <= tolerance)
    if exact_tick_step:
        uniformity_resolution = "exact-tick-step"
    elif not precision_sufficient:
        uniformity_resolution = "underresolved-nonuniform"
    elif uniform:
        uniformity_resolution = "quantized-uniform-grid"
    else:
        uniformity_resolution = "nonuniform"
    digest = hashlib.sha256()
    for value in pts:
        timestamp = Fraction(value) * time_base
        digest.update(f"{timestamp.numerator}/{timestamp.denominator}\n".encode("ascii"))

    # PTS records when the final picture starts, not how long it remains visible.
    # Keep its separately decoded duration; never infer it from the previous
    # interval, nominal FPS, rounded container seconds, or the audio track.
    endpoint: dict[str, Any] = {"video_pts_endpoint_evidence": "unavailable"}
    raw_duration = frames[-1].get("duration")
    if raw_duration in (None, "", "N/A"):
        raw_duration = frames[-1].get("pkt_duration")
    if raw_duration not in (None, "", "N/A"):
        if type(raw_duration) is int:
            duration_ticks = raw_duration
        elif isinstance(raw_duration, str) and raw_duration.strip().isdecimal():
            duration_ticks = int(raw_duration.strip())
        else:
            raise RuntimeError("Cannot qualify last-frame duration: expected exact non-negative integer ticks.")
        if duration_ticks < 0:
            raise RuntimeError("Cannot qualify last-frame duration: negative duration ticks.")
        if duration_ticks > 0:
            last_duration = Fraction(duration_ticks) * time_base
            endpoint = {
                "video_pts_endpoint_evidence": "decoded-last-frame-duration",
                "video_pts_last_duration": str(last_duration),
                "video_pts_end_exclusive": str(last + last_duration),
            }

    return {
        "video_pts_count": len(pts),
        "video_frame_count": str(len(pts)),
        "video_pts_time_base": str(time_base),
        "video_pts_start": str(first),
        "video_pts_end": str(last),
        "video_pts_span": str(span),
        "video_pts_average_interval": str(interval),
        "video_pts_average_rate": str(Fraction(len(pts) - 1, 1) / span),
        "video_pts_min_step_ticks": min_step_ticks,
        "video_pts_max_step_ticks": max_step_ticks,
        "video_pts_min_interval": str(Fraction(min_step_ticks) * time_base),
        "video_pts_max_interval": str(Fraction(max_step_ticks) * time_base),
        "video_pts_max_grid_error": str(max_grid_error),
        "video_pts_tolerance": str(tolerance),
        "video_pts_uniform": uniform,
        "video_pts_uniformity_resolution": uniformity_resolution,
        "video_pts_sha256": digest.hexdigest(),
        **endpoint,
        "declared_frame_count_verified": declared_count_verified,
        "is_vfr": not uniform,
        "frame_rate_evidence": "presentation-timestamps",
        "verification_scope": (
            "container-metadata+presentation-timestamps"
            if declared_count_verified else "presentation-timestamps+time-base"
        ),
    }


def probe_video(
    path: str | Path,
    ffprobe: str,
    *,
    inspect_timestamps: bool = False,
) -> dict[str, Any]:
    p = str(path)
    data = run_json([
        ffprobe,
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        p,
    ])
    info: dict[str, Any] = {
        "path": p,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "duration": 0.0,
        "rotation": 0,
        "sar": "1:1",
        "dar": "",
        "codec": "",
        "pix_fmt": "",
        "has_alpha": False,
        "field_order": "",
        "color_primaries": "",
        "color_transfer": "",
        "color_space": "",
        "color_range": "",
        "has_audio": False,
        "audio_stream_count": 0,
        "audio_codec": "",
        "audio_codecs": [],
        "audio_streams": [],
        "is_vfr": False,
        # Header rates are a heuristic, not a packet-timing or full decode result.
        "frame_rate_evidence": "header-heuristic",
        "verification_scope": "container-metadata",
        "decode_verified": False,
    }
    fmt = data.get("format") or {}
    try:
        info["duration"] = float(fmt.get("duration") or 0)
    except Exception:
        pass
    for stream in data.get("streams") or []:
        if stream.get("codec_type") == "video" and not info["width"]:
            info["width"] = int(stream.get("width") or 0)
            info["height"] = int(stream.get("height") or 0)
            info["fps"] = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
            # Retain exact rational evidence alongside the established float.
            # Explicit cadence validation must distinguish 60000/1001 from 60.
            info["fps_ratio"] = str(stream.get("avg_frame_rate") if parse_rate(stream.get("avg_frame_rate"))
                                    else stream.get("r_frame_rate") or "0/1")
            info["video_r_frame_rate"] = str(stream.get("r_frame_rate") or "")
            info["video_avg_frame_rate"] = str(stream.get("avg_frame_rate") or "")
            info["video_time_base"] = str(stream.get("time_base") or "")
            info["video_duration"] = str(stream.get("duration") or "")
            info["video_start_time"] = str(stream.get("start_time") or "0")
            info["video_frame_count"] = str(stream.get("nb_frames") or "")
            info["codec"] = stream.get("codec_name") or ""
            info["pix_fmt"] = stream.get("pix_fmt") or ""
            info["has_alpha"] = pixel_format_has_alpha(info["pix_fmt"])
            info["field_order"] = stream.get("field_order") or ""
            info["sar"] = stream.get("sample_aspect_ratio") or "1:1"
            info["dar"] = stream.get("display_aspect_ratio") or ""
            info["color_primaries"] = stream.get("color_primaries") or ""
            info["color_transfer"] = stream.get("color_transfer") or ""
            info["color_space"] = stream.get("color_space") or ""
            info["color_range"] = stream.get("color_range") or ""
            tags = stream.get("tags") or {}
            side_data = stream.get("side_data_list") or []
            if "rotate" in tags:
                _set_rotation_metadata(info, tags["rotate"])
            for side in side_data:
                if isinstance(side, dict) and "rotation" in side:
                    _set_rotation_metadata(info, side["rotation"])
            r = parse_rate(stream.get("r_frame_rate"))
            avg = parse_rate(stream.get("avg_frame_rate"))
            info["is_vfr"] = bool(r and avg and abs(r - avg) > 0.02)
        elif stream.get("codec_type") == "audio":
            info["has_audio"] = True
            info["audio_stream_count"] += 1
            codec_name = stream.get("codec_name") or ""
            info["audio_codecs"].append(codec_name)
            if not info["audio_codec"]:
                info["audio_codec"] = codec_name
            audio = {
                "index": int(stream.get("index") or 0),
                "codec": codec_name,
                "sample_rate": int(stream.get("sample_rate") or 0),
                "channels": int(stream.get("channels") or 0),
                "channel_layout": str(stream.get("channel_layout") or ""),
                "start_time": str(stream.get("start_time") or "0"),
                "duration": str(stream.get("duration") or ""),
                "duration_tag": str((stream.get("tags") or {}).get("DURATION") or ""),
                "duration_ts": str(stream.get("duration_ts") or ""),
                "time_base": str(stream.get("time_base") or ""),
            }
            sample_count = _audio_sample_count(stream)
            if sample_count is not None:
                audio["sample_count"] = sample_count
            info["audio_streams"].append(audio)
    if inspect_timestamps:
        frame_data = run_json([
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_frames",
            "-show_entries", "frame=best_effort_timestamp,duration,pkt_duration",
            "-of", "json",
            p,
        ])
        timing = _presentation_timestamp_summary(
            frame_data.get("frames"),
            str(info.get("video_time_base") or ""),
            str(info.get("video_frame_count") or ""),
        )
        info.update(timing)
        if timing["video_pts_uniform"] and info.get("video_duration") in (None, "", "N/A"):
            info["video_duration"] = str(
                Fraction(timing["video_pts_span"]) + Fraction(timing["video_pts_average_interval"])
            )
    return info


def probe_image(path: str | Path, ffprobe: str) -> dict[str, Any]:
    info = probe_video(path, ffprobe)
    return {
        "path": str(path),
        "width": info["width"],
        "height": info["height"],
        "pix_fmt": info["pix_fmt"],
        "color_primaries": info["color_primaries"],
        "color_transfer": info["color_transfer"],
        "color_space": info["color_space"],
        "has_alpha": info["has_alpha"],
    }


def verify_video(
    path: str | Path,
    ffprobe: str,
    *,
    inspect_timestamps: bool = False,
) -> tuple[bool, dict[str, Any], str]:
    """Check local metadata, not playback, whole-file integrity or creative acceptance."""
    try:
        target = Path(path)
        before = target.stat()
        info = probe_video(path, ffprobe, inspect_timestamps=inspect_timestamps)
        after = target.stat()
        # Catch normal concurrent export/replacement. This stat check is not a content
        # snapshot: same-size in-place writes with restored timestamps remain unverified.
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            return False, info, "file changed while metadata was being probed"
        if not info["width"] or not info["height"]:
            return False, info, "video stream missing"
        if not math.isfinite(info["duration"]) or info["duration"] <= 0:
            return False, info, "duration missing or not finite and positive"
        return True, info, ""
    except Exception as exc:
        return False, {}, str(exc)
