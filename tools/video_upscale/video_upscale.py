from __future__ import annotations

# This standalone worker bootstraps the repository root before shared imports.
# ruff: noqa: E402

import argparse
import hashlib
import json
import math
import os
import queue
import re
import shutil
import stat as stat_module
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from fractions import Fraction
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SPIFFY_WORKSPACE = ROOT / "Spiffy Workspace"
if str(SPIFFY_WORKSPACE) not in sys.path:
    sys.path.insert(1, str(SPIFFY_WORKSPACE))

from shared.ffmpeg_utils import (
    even,
    fit_within,
    has_encoder,
    keep_aspect_target,
    layout_filter,
    part_path,
    standard_ffmpeg,
    standard_ffprobe,
)
from shared.job_paths import collision_safe_path, job_dir as resolve_job_dir, make_job_id
from shared.media_probe import (
    pixel_format_has_alpha,
    probe_video,
    verify_video,
    video_track_timescale_for_time_base,
)
from shared.progress_writer import ProgressWriter, atomic_write_json, load_input_list
from shared.topaz_paths import find_topaz_video
from shared.gpu_lock import GpuLockCancelled, SharedGpuLock
from workflow.workflow_hashing import file_sample_fingerprint
from workflow.social_export_profiles import get_social_export_profile


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".wmv", ".mxf"}
RENDER_AUDIO_SIDECAR_EXTS = {".aac", ".m4a", ".wav"}
RENDER_TEMP_STEM_RE = re.compile(r"^(?P<base>.+)\.\d+\.\d+$")
MAX_OUTPUT_FPS = 60
RENDER_STABLE_SECONDS = 2.0
POST_RENDER_DELAY_SECONDS = 2.0
RENDER_POLL_SECONDS = 0.5
RENDER_WAIT_TIMEOUT_SECONDS = 6 * 60 * 60
RENDER_FINAL_MUX_GRACE_SECONDS = 12.0
# Native model initialization twice exceeded the old 8s deadline. This is a
# bounded startup allowance, not an export-quality or successful-exit bypass.
MODEL_UPSCALE_SMOKE_TIMEOUT_SECONDS = 30
# Apollo startup exceeded 45s in two real sequential worker attempts although
# the same bounded two-output-frame diagnostic succeeded. Allow initialization
# time while retaining cancellation, exact exit checks and a hard 90s limit.
MODEL_INTERPOLATION_SMOKE_TIMEOUT_SECONDS = 90
TOPAZ_NATIVE_FIRST_USE_RETRY_EXIT_CODES = (3221226505, -1073740791)
TOPAZ_CPU_RECOVERY_MAX_PIXELS = 256 * 256
VIDEOSEAL_PREFLIGHT_TIMEOUT_SECONDS = 180
VIDEOSEAL_RECEIPT_RETENTION_DAYS = 3
TOPAZ_CHECKPOINT_SCHEMA = "spiffy_video_upscale_checkpoint_v1"
DELIVERY_COMPLETION_SCHEMA = "spiffy_video_upscale_delivery_completion_v2"
SOURCE_COMPLETION_SCHEMA = "spiffy_video_upscale_source_completion_v2"
CHECKPOINT_SAMPLE_BYTES = 1024 * 1024
DEFAULT_WATERMARK_STRENGTH = 0.2
DELIVERY_PLAN_SCHEMA = "spiffy_final_delivery_plan_v1"
DELIVERY_GEOMETRY_SCHEMA = "spiffy_final_delivery_geometry_v1"
DELIVERY_GEOMETRY_PRESETS = {"1k": 1920, "2k": 2560, "3k": 3072, "4k": 3840}
MAX_WORKING_GEOMETRY_EDGE = 4096
MAX_WORKING_GEOMETRY_PIXELS = 16_777_216
MAX_FINAL_GEOMETRY_EDGE = 8192
MAX_FINAL_GEOMETRY_PIXELS = 33_554_432
SUPPORTED_LAYOUTS = {"keep_source_aspect", "fit_into_frame", "fill_crop_center"}


PRESET_MODEL = {
    "auto": {"up_model": "prob-4", "params": ":details=0.20:compression=0.15:blend=0.15", "long_edge": 3840, "fps": 60},
    "clean_default": {"up_model": "ahq-12", "params": "", "long_edge": 1920, "fps": None},
    "compression_fix": {"up_model": "prob-4", "params": ":compression=0.35:details=0.10:blend=0.20", "long_edge": 1920, "fps": None},
    "detail_boost": {"up_model": "prob-4", "params": ":details=0.35:blur=0.08:blend=0.15", "long_edge": 1920, "fps": None},
    "face_people": {"up_model": "ahq-12", "params": ":details=0.15:blend=0.20", "long_edge": 1920, "fps": None},
    "anime_art": {"up_model": "ahq-12", "params": ":details=0.20:halo=0.10:blend=0.15", "long_edge": 1920, "fps": None},
    "smooth_fps": {"up_model": "ahq-12", "params": "", "long_edge": 0, "fps": 60},
    "4k_60fps": {"up_model": "prob-4", "params": ":details=0.20:compression=0.15:blend=0.15", "long_edge": 3840, "fps": 60},
    "ss_4k": {"up_model": "prob-4", "params": ":details=0.50:compression=0.0:blend=0.15", "long_edge": 3840, "fps": None},
}


class JobCancelled(Exception):
    pass


@dataclass(frozen=True)
class EnhancementRecipe:
    """One attempt's immutable executable settings; never edits PRESET_MODEL."""

    preset_id: str
    up_model: str
    params: str
    explicit: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {"schema": "spiffy_enhancement_recipe_v1", "preset_id": self.preset_id,
                "up_model": self.up_model, "params": self.params}


def resolve_enhancement_recipe(enhancement: Any) -> EnhancementRecipe:
    """Validate the public B1 object and canonicalize only supported controls.

    Bounds observed in installed Topaz tvai_up help (2026-11-13). Existing
    preset-only parameter bytes remain untouched for legacy compatibility.
    """
    if not isinstance(enhancement, dict) or set(enhancement) - {"preset_id", "overrides"}:
        raise ValueError("enhancement must be an object with preset_id and optional overrides.")
    preset_id = enhancement.get("preset_id")
    if not isinstance(preset_id, str) or preset_id not in PRESET_MODEL:
        raise ValueError("enhancement.preset_id must name an existing enhancement preset.")
    overrides = enhancement.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("enhancement.overrides must be an object.")
    spec = PRESET_MODEL[preset_id]
    bounds = {key: (-1.0, 1.0) for key in ("details", "compression", "blur", "noise", "halo", "preblur")}
    bounds["blend"] = (0.0, 1.0)
    if any(key not in bounds for key in overrides):
        raise ValueError("enhancement.overrides contains unknown controls.")
    if overrides and spec["up_model"] != "prob-4":
        raise ValueError("enhancement.overrides are supported only for incumbent Proteus prob-4 presets.")
    values = {}
    for key, value in overrides.items():
        if type(value) not in (int, float):
            raise ValueError(f"enhancement.overrides.{key} must be a finite number, not a boolean or string.")
        low, high = bounds[key]
        if not low <= value <= high or not math.isfinite(value):
            raise ValueError(f"enhancement.overrides.{key} must be finite and within [{low}, {high}].")
        values[key] = float(value) if value != 0 else 0.0
    params = spec.get("params") or ""
    if values:
        # Merge preset defaults once, then serialize in one stable control order.
        merged = {key: float(value) for key, value in
                  (entry.split("=", 1) for entry in params.split(":") if entry)}
        merged.update(values)
        params = "".join(f":{key}={merged[key]}" for key in bounds if key in merged)
    return EnhancementRecipe(preset_id, spec["up_model"], params)


def delivery_enhancement_recipe(plan: dict[str, Any], preset_id: str) -> EnhancementRecipe:
    if "enhancement" in plan:
        recipe = resolve_enhancement_recipe(plan["enhancement"])
        if plan.get("enhancement_recipe") != recipe.as_dict() or plan.get("enhance_preset") != recipe.preset_id:
            raise ValueError("Delivery enhancement recipe does not match its validated settings/preset.")
        return recipe
    if "enhancement_recipe" in plan:
        raise ValueError("Delivery enhancement recipe requires its validated enhancement object.")
    # Historical direct worker calls retain their fallback behavior.
    spec = PRESET_MODEL.get(preset_id, PRESET_MODEL["auto"])
    return EnhancementRecipe(preset_id, spec["up_model"], spec.get("params") or "", explicit=False)


def enhancement_resume_matches(manifest: dict[str, Any], recipe: EnhancementRecipe) -> bool:
    previous = manifest.get("enhancement_recipe")
    return previous == recipe.as_dict() if recipe.explicit else previous is None


def normalize_frame_rate(value: Any) -> dict[str, str]:
    """Validate an opt-in CFR or exact source-PTS-preserve policy.

    Omitted policies retain legacy behavior. Preserve/VFR is never re-labelled
    CFR and carries no synthetic FPS value or filter syntax.
    """
    if not isinstance(value, dict):
        raise ValueError("frame_rate must be an object with an explicit supported mode.")
    if set(value) == {"mode"} and value.get("mode") == "preserve_source":
        return {"mode": "preserve_source"}
    if set(value) != {"mode", "fps"} or value.get("mode") != "cfr":
        raise ValueError("frame_rate must contain either mode='cfr' and fps, or exactly mode='preserve_source'.")
    raw = value["fps"]
    if type(raw) not in (int, float, str) or (isinstance(raw, float) and not math.isfinite(raw)):
        raise ValueError("frame_rate.fps must be a finite number or exact rational string, never a boolean.")
    text = str(raw)
    if len(text) > 64 or not re.fullmatch(r"[0-9]+(?:/[0-9]+|\.[0-9]+)?", text):
        raise ValueError("frame_rate.fps must be positive decimal or numerator/denominator text.")
    try:
        rate = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("frame_rate.fps is not a valid rational rate.") from exc
    if not Fraction(1) <= rate <= Fraction(120) or rate.denominator > 1_000_000:
        raise ValueError("frame_rate.fps must be between 1 and 120 with a denominator no larger than 1000000.")
    return {"mode": "cfr", "fps": f"{rate.numerator}/{rate.denominator}"}


def normalize_delivery_geometry(value: Any) -> dict[str, Any] | None:
    """Canonical working/final geometry contract for the existing delivery plan.

    Named presets are long-edge values, not implicit width/height or cinema
    aspect ratios: 1K=1920, 2K=2560, 3K=3072, 4K=3840 pixels. Custom exact
    canvases currently support non-destructive fit/pad only. Center crop stays
    fail-closed until a source-bound reframe and caption-collision proof is
    carried through this delivery request.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("geometry must be an object with optional working and final objects.")
    raw = dict(value)
    schema = raw.pop("schema", DELIVERY_GEOMETRY_SCHEMA)
    if schema != DELIVERY_GEOMETRY_SCHEMA:
        raise ValueError("Unsupported delivery geometry schema.")
    if set(raw) - {"working", "final"}:
        raise ValueError("geometry may contain only working and final objects.")

    normalized: dict[str, Any] = {}
    for section in ("working", "final"):
        item = raw.get(section)
        if item is None:
            continue
        if not isinstance(item, dict):
            raise ValueError(f"geometry.{section} must be an object.")
        mode = item.get("mode")
        if section == "final" and mode == "source_display":
            if set(item) != {"mode"}:
                raise ValueError("geometry.final source_display accepts only its mode.")
            normalized[section] = {"mode": "source_display"}
            continue

        if mode == "preset":
            if set(item) not in ({"mode", "preset"}, {"mode", "preset", "long_edge"}):
                raise ValueError(f"geometry.{section} preset requires only mode and preset.")
            preset = item.get("preset")
            if not isinstance(preset, str) or preset.casefold() not in DELIVERY_GEOMETRY_PRESETS:
                raise ValueError(f"geometry.{section}.preset must be one of 1k, 2k, 3k, or 4k.")
            preset = preset.casefold()
            edge = DELIVERY_GEOMETRY_PRESETS[preset]
            if "long_edge" in item and (type(item["long_edge"]) is not int or item["long_edge"] != edge):
                raise ValueError(f"geometry.{section} preset long_edge must match its recorded pixel value.")
            normalized[section] = {"mode": "preset", "preset": preset, "long_edge": edge}
            continue

        max_edge = MAX_WORKING_GEOMETRY_EDGE if section == "working" else MAX_FINAL_GEOMETRY_EDGE
        max_pixels = MAX_WORKING_GEOMETRY_PIXELS if section == "working" else MAX_FINAL_GEOMETRY_PIXELS
        if mode == "long_edge":
            if set(item) != {"mode", "long_edge"}:
                raise ValueError(f"geometry.{section} long_edge requires only mode and long_edge.")
            edge = item.get("long_edge")
            if type(edge) is not int or edge < 64 or edge > max_edge:
                raise ValueError(f"geometry.{section}.long_edge must be an integer from 64 through {max_edge}.")
            normalized[section] = {"mode": "long_edge", "long_edge": edge}
            continue

        if mode == "exact":
            allowed = {"mode", "width", "height", "layout"}
            if section == "working":
                allowed.add("allow_downscale")
            if set(item) - allowed or not {"mode", "width", "height", "layout"}.issubset(item):
                raise ValueError(f"geometry.{section} exact requires width, height, and explicit layout.")
            width = item.get("width")
            height = item.get("height")
            if type(width) is not int or type(height) is not int or width < 64 or height < 64:
                raise ValueError(f"geometry.{section} exact width and height must be integers of at least 64 pixels.")
            output_width = width + (width % 2)
            output_height = height + (height % 2)
            if output_width > max_edge or output_height > max_edge or output_width * output_height > max_pixels:
                raise ValueError(f"geometry.{section} exact canvas exceeds the bounded pixel budget.")
            layout = item.get("layout")
            if layout == "fill_crop_center":
                raise ValueError(
                    "Custom center crop requires source-bound reframe and caption-collision proof; "
                    "the current Final Delivery request does not carry that proof."
                )
            if layout != "fit_into_frame":
                raise ValueError(f"geometry.{section}.exact layout must explicitly be fit_into_frame.")
            result = {"mode": "exact", "width": width, "height": height, "layout": "fit_into_frame"}
            if section == "working":
                allow_downscale = item.get("allow_downscale", False)
                if type(allow_downscale) is not bool:
                    raise ValueError("geometry.working.allow_downscale must be a boolean.")
                result["allow_downscale"] = allow_downscale
            normalized[section] = result
            continue

        raise ValueError(f"geometry.{section}.mode must be preset, long_edge, or exact" +
                         (", or source_display" if section == "final" else "."))

    if not normalized:
        return None
    return {
        "schema": DELIVERY_GEOMETRY_SCHEMA,
        "working": normalized.get("working"),
        "final": normalized.get("final"),
    }


def resolve_custom_final_geometry(info: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    """Resolve a custom Smart Finish master canvas against source display geometry."""
    normalized = normalize_delivery_geometry({"final": value})
    spec = normalized["final"] if normalized else None
    if not isinstance(spec, dict) or spec.get("mode") not in {"exact", "long_edge", "preset"}:
        raise ValueError("Custom final geometry requires an exact, long_edge, or named preset mode.")
    source = source_display_geometry(info)
    if spec["mode"] == "exact":
        requested_width = int(spec["width"])
        requested_height = int(spec["height"])
        output_width = requested_width + (requested_width % 2)
        output_height = requested_height + (requested_height % 2)
    else:
        edge = int(spec["long_edge"])
        output_width, output_height = keep_aspect_target(
            int(source["display_width"]), int(source["display_height"]), edge
        )
        requested_width, requested_height = output_width, output_height
    if (
        output_width > MAX_FINAL_GEOMETRY_EDGE
        or output_height > MAX_FINAL_GEOMETRY_EDGE
        or output_width * output_height > MAX_FINAL_GEOMETRY_PIXELS
    ):
        raise ValueError("Resolved final geometry exceeds the bounded output pixel budget.")
    return {
        **source,
        "policy": "custom_final",
        "geometry": dict(spec),
        "requested_canvas": {"width": requested_width, "height": requested_height},
        "output_width": output_width,
        "output_height": output_height,
        "layout": "fit_into_frame",
        "alignment_padding": {
            "horizontal": output_width - requested_width,
            "vertical": output_height - requested_height,
            "placement": "centered",
        },
    }


def resolve_delivery_working_geometry(
    info: dict[str, Any], preset_id: str, layout_id: str, plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve bounded working pixels and record the source-display transform."""
    plan = dict(plan or {})
    geometry_contract = normalize_delivery_geometry(plan.get("geometry")) if plan.get("geometry") is not None else None
    working_spec = (geometry_contract or {}).get("working")
    variants = [row for row in plan.get("variants") or [] if isinstance(row, dict)]
    source_display_requested = bool(working_spec) or plan.get("adaptive_enhancement") is not None or any(
        row.get("geometry_policy") in {"source_display", "custom_final"} for row in variants
    )
    if not source_display_requested:
        target_width, target_height, frame_width, frame_height = target_dimensions(info, preset_id, layout_id)
        return {
            "schema": "spiffy_working_geometry_proof_v1",
            "mode": "legacy",
            "target_width": target_width,
            "target_height": target_height,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "layout": layout_id,
            "source_display": None,
            "downscale_authorized": False,
            "apply_input_layout": False,
        }

    source = source_display_geometry(info)
    has_legacy_outputs = any(row.get("geometry_policy") not in {"source_display", "custom_final"} for row in variants)
    geometry_changes_working_dimensions = (
        source["source_sample_aspect_ratio"] != "1:1"
        or (
            source["source_rotation_degrees"] in {90, 270}
            and source["source_width"] != source["source_height"]
        )
    )
    if has_legacy_outputs and geometry_changes_working_dimensions:
        raise ValueError(
            "Non-square SAR or 90/270-degree rotation of a non-square frame requires a dedicated "
            "smart_finish_master-only plan so legacy output geometry remains unchanged."
        )
    if plan.get("geometry") is not None and plan.get("adaptive_enhancement") is not None and geometry_changes_working_dimensions:
        raise ValueError(
            "Custom final geometry with adaptive enhancement is not qualified for non-square SAR or autorotation; "
            "use the non-adaptive Smart Finish route for this source."
        )
    display_width = int(source["display_width"])
    display_height = int(source["display_height"])
    display_info = dict(info)
    display_info.update({"width": display_width, "height": display_height, "sar": "1:1", "rotation": 0})

    if working_spec is None:
        adaptive = plan.get("adaptive_enhancement")
        if adaptive is not None:
            from adaptive_enhancement import normalize_policy
            adaptive = normalize_policy(adaptive)
        if adaptive is not None and "working_long_edge" in adaptive:
            edge = max(display_width, display_height, int(adaptive["working_long_edge"]))
            target_width, target_height = keep_aspect_target(display_width, display_height, edge)
            frame_width, frame_height = target_width, target_height
        else:
            target_width, target_height, frame_width, frame_height = target_dimensions(
                display_info, preset_id, layout_id
            )
        return {
            "schema": "spiffy_working_geometry_proof_v1",
            "mode": "source_display_default",
            "target_width": target_width,
            "target_height": target_height,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "requested_canvas": {"width": target_width, "height": target_height},
            "alignment_padding": {"horizontal": 0, "vertical": 0, "placement": "centered"},
            "layout": "fit_into_frame",
            "source_display": source,
            "downscale_authorized": False,
            "apply_input_layout": False,
        }

    if "adaptive_enhancement" in plan:
        raise ValueError("Custom working geometry cannot be combined with adaptive enhancement until its crop owner supports it.")
    if working_spec["mode"] in {"preset", "long_edge"}:
        edge = max(display_width, display_height, int(working_spec["long_edge"]))
        target_width, target_height = keep_aspect_target(display_width, display_height, edge)
        if (
            max(target_width, target_height) > MAX_WORKING_GEOMETRY_EDGE
            or target_width * target_height > MAX_WORKING_GEOMETRY_PIXELS
        ):
            raise ValueError("Source-preserving working geometry exceeds the bounded Topaz pixel budget.")
        return {
            "schema": "spiffy_working_geometry_proof_v1",
            "mode": working_spec["mode"],
            "requested": dict(working_spec),
            "target_width": target_width,
            "target_height": target_height,
            "frame_width": target_width,
            "frame_height": target_height,
            "requested_canvas": {"width": target_width, "height": target_height},
            "alignment_padding": {"horizontal": 0, "vertical": 0, "placement": "centered"},
            "layout": "fit_into_frame",
            "source_display": source,
            "downscale_authorized": False,
            "apply_input_layout": True,
        }

    requested_width = int(working_spec["width"])
    requested_height = int(working_spec["height"])
    target_width = requested_width + (requested_width % 2)
    target_height = requested_height + (requested_height % 2)
    if (
        target_width > MAX_WORKING_GEOMETRY_EDGE
        or target_height > MAX_WORKING_GEOMETRY_EDGE
        or target_width * target_height > MAX_WORKING_GEOMETRY_PIXELS
    ):
        raise ValueError("Resolved working geometry exceeds the bounded Topaz pixel budget.")
    scale = min(target_width / display_width, target_height / display_height)
    allow_downscale = bool(working_spec.get("allow_downscale"))
    if scale < 1.0 and not allow_downscale:
        raise ValueError("Exact working geometry would downscale the source; set allow_downscale=true to authorize it.")
    return {
        "schema": "spiffy_working_geometry_proof_v1",
        "mode": "exact",
        "requested": dict(working_spec),
        "target_width": target_width,
        "target_height": target_height,
        "frame_width": target_width,
        "frame_height": target_height,
        "requested_canvas": {"width": requested_width, "height": requested_height},
        "alignment_padding": {
            "horizontal": target_width - requested_width,
            "vertical": target_height - requested_height,
            "placement": "centered",
        },
        "layout": "fit_into_frame",
        "source_display": source,
        "downscale_authorized": allow_downscale,
        "apply_input_layout": True,
    }


def frame_rate_token(value: int | float | str | Fraction) -> str:
    rate = Fraction(normalize_frame_rate({"mode": "cfr", "fps": str(value) if isinstance(value, Fraction) else value})["fps"])
    return str(rate.numerator) if rate.denominator == 1 else f"{rate.numerator}/{rate.denominator}"


def source_frame_rate(info: dict[str, Any]) -> Fraction:
    raw = info.get("fps_ratio") or str(info.get("fps") or "0")
    try:
        rate = Fraction(str(raw))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise RuntimeError("Cannot determine a finite positive source frame rate.") from exc
    if rate <= 0:
        raise RuntimeError("Cannot determine a finite positive source frame rate.")
    return rate


def source_max_presentation_rate(info: dict[str, Any]) -> Fraction:
    """Return the highest observed instantaneous source cadence from exact PTS."""
    if info.get("frame_rate_evidence") != "presentation-timestamps":
        raise RuntimeError("Source-timing capability checks require decoded presentation timestamp evidence.")
    try:
        pts_count = int(info.get("video_pts_count") or 0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Source-timing capability checks require a valid decoded PTS count.") from exc
    if pts_count < 2 or not str(info.get("video_pts_sha256") or ""):
        raise RuntimeError("Source-timing capability checks require a complete decoded PTS sequence.")
    try:
        minimum_interval = Fraction(str(info.get("video_pts_min_interval") or "0"))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise RuntimeError("Source-timing capability checks require an exact minimum PTS interval.") from exc
    if minimum_interval <= 0:
        raise RuntimeError("Source-timing capability checks require a positive minimum PTS interval.")
    return Fraction(1, 1) / minimum_interval


def validate_source_timing_profile_cap(info: dict[str, Any], max_fps: int) -> Fraction:
    if type(max_fps) is not int or max_fps < 1:
        raise RuntimeError("Source-timing output requires a positive integer profile FPS capability.")
    maximum = source_max_presentation_rate(info)
    if maximum > max_fps:
        raise RuntimeError(
            f"Source PTS reaches {maximum} FPS instantaneously, above this output profile's "
            f"declared {max_fps} FPS source-timing capability. Refusing to resample timestamps."
        )
    return maximum


def frame_rate_progress_metrics(
    variants: list[dict[str, Any]], default_max_fps: int | None,
    *, source_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report each final timing policy and highest observed output cadence."""
    per_output: list[dict[str, Any]] = []
    numeric_rates: list[Fraction] = []
    for variant in variants:
        policy = variant.get("frame_rate")
        if policy is None:
            continue
        normalized = normalize_frame_rate(policy)
        if normalized["mode"] == "preserve_source":
            row = {
                "profile_ids": [str(value) for value in variant.get("profile_ids") or [variant.get("profile_id") or ""]],
                "mode": normalized["mode"],
                "reason": str(variant.get("frame_rate_reason") or "source_preserved"),
            }
            if source_info is not None:
                maximum = source_max_presentation_rate(source_info)
                numeric_rates.append(maximum)
                row.update({
                    "max_instantaneous_fps": str(maximum),
                    "source_pts_count": int(source_info.get("video_pts_count") or 0),
                    "source_time_base": str(source_info.get("video_time_base") or ""),
                    "source_pts_sha256": str(source_info.get("video_pts_sha256") or ""),
                })
            else:
                row["source_pts_status"] = "pending"
            per_output.append(row)
            continue
        rate = Fraction(normalized["fps"])
        numeric_rates.append(rate)
        per_output.append({
            "profile_ids": [str(value) for value in variant.get("profile_ids") or [variant.get("profile_id") or ""]],
            "mode": normalized["mode"],
            "fps": normalized["fps"],
            "reason": str(variant.get("frame_rate_reason") or "requested"),
        })
    if not numeric_rates:
        return {"max_output_fps": default_max_fps, **({"output_frame_rates": per_output} if per_output else {})}
    return {
        "max_output_fps": float(max(numeric_rates)),
        "output_frame_rates": per_output,
    }


def validate_frame_rate_plan(plan: dict[str, Any]) -> dict[str, str] | None:
    if "frame_rate" not in plan:
        if any(isinstance(row, dict) and "frame_rate" in row for row in plan.get("variants") or []):
            raise ValueError("Variant frame_rate requires a job frame_rate contract.")
        return None
    policy = normalize_frame_rate(plan["frame_rate"])
    if policy != plan["frame_rate"]:
        raise ValueError("Job frame_rate must be canonical before execution.")
    if policy["mode"] == "preserve_source":
        if plan.get("frame_rate_boundary") != "source_pts_preserve_v1":
            raise ValueError("Source-preserve frame_rate requires the exact PTS-preservation boundary policy.")
        preserve_rows = [row for row in plan.get("variants") or [] if isinstance(row, dict)]
        if not preserve_rows:
            raise ValueError("Source-preserve frame_rate requires at least one output variant.")
        for row in preserve_rows:
            if not isinstance(row, dict) or row.get("frame_rate") != policy:
                raise ValueError("Every source-preserve variant requires the canonical source PTS policy.")
            ceiling = row.get("max_fps")
            if type(ceiling) is not int or ceiling < 1:
                raise ValueError("Source-preserve variant requires a positive integer profile FPS capability.")
            profile_ids = row.get("profile_ids") or [row.get("profile_id")]
            if not isinstance(profile_ids, list) or not profile_ids or any(
                not isinstance(profile_id, str) or not profile_id.strip() for profile_id in profile_ids
            ):
                raise ValueError("Source-preserve variant requires its canonical profile_ids.")
            if str(row.get("profile_id") or "") not in profile_ids:
                raise ValueError("Source-preserve primary profile is not in profile_ids.")
            expected_caps: set[int] = set()
            for profile_id in profile_ids:
                try:
                    profile = get_social_export_profile(profile_id)
                except KeyError as exc:
                    raise ValueError(f"Unknown profile in source-preserve contract: {profile_id}.") from exc
                if profile.get("supports_source_timing") is not True:
                    raise ValueError(f"Profile {profile_id} does not declare exact source-PTS preservation support.")
                try:
                    profile_cap = Fraction(str(profile.get("max_fps") or 0))
                except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
                    raise ValueError(f"Invalid source-timing FPS capability for profile {profile_id}.") from exc
                if profile_cap <= 0 or profile_cap.denominator != 1:
                    raise ValueError(f"Invalid source-timing FPS capability for profile {profile_id}.")
                expected_caps.add(int(profile_cap))
            if len(expected_caps) != 1 or ceiling != next(iter(expected_caps)):
                raise ValueError("Grouped source-preserve profiles have different timing capabilities.")
            if row.get("frame_rate_reason") != "source_preserved":
                raise ValueError("Source-preserve variant reason must be source_preserved.")
            if row.get("watermark"):
                raise ValueError("Source-PTS preservation with VideoSeal is not yet qualified; use watermark off.")
        return policy
    if plan.get("frame_rate_boundary") != "source_end_context_v1":
        raise ValueError("CFR frame_rate requires the current source-bound endpoint policy.")
    requested = Fraction(policy["fps"])
    retains_requested = False
    for row in plan.get("variants") or []:
        if not isinstance(row, dict) or "frame_rate" not in row:
            raise ValueError("Every variant requires a bound frame_rate.")
        actual = normalize_frame_rate(row["frame_rate"])
        ceiling = row.get("max_fps")
        if type(ceiling) is not int or ceiling < 1 or actual != row["frame_rate"]:
            raise ValueError("Variant frame_rate or ceiling is not canonical.")
        rate = Fraction(actual["fps"])
        if rate > requested or rate > ceiling:
            raise ValueError("Variant frame_rate exceeds the requested rate or profile ceiling.")
        profile_ids = row.get("profile_ids") or [row.get("profile_id")]
        if not isinstance(profile_ids, list) or not profile_ids or any(
            not isinstance(profile_id, str) or not profile_id.strip() for profile_id in profile_ids
        ):
            raise ValueError("Variant frame-rate contract requires its canonical profile_ids.")
        if str(row.get("profile_id") or "") not in profile_ids:
            raise ValueError("Variant frame-rate contract primary profile is not in profile_ids.")
        expected_rates: set[Fraction] = set()
        for profile_id in profile_ids:
            try:
                profile = get_social_export_profile(profile_id)
            except KeyError as exc:
                raise ValueError(f"Unknown profile in frame-rate contract: {profile_id}.") from exc
            try:
                profile_ceiling = Fraction(str(profile.get("max_fps") or 60))
            except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
                raise ValueError(f"Invalid frame-rate ceiling for profile {profile_id}.") from exc
            if profile_ceiling <= 0:
                raise ValueError(f"Invalid frame-rate ceiling for profile {profile_id}.")
            expected_rates.add(min(requested, profile_ceiling))
        if len(expected_rates) != 1:
            raise ValueError("Grouped profiles have different per-output frame-rate contracts.")
        expected_rate = next(iter(expected_rates))
        expected_max_fps = math.ceil(expected_rate)
        expected_reason = "requested" if expected_rate == requested else "profile_ceiling"
        if rate != expected_rate or ceiling != expected_max_fps:
            raise ValueError("Variant rate or max_fps does not match the profile frame-rate contract.")
        if row.get("frame_rate_reason") != expected_reason:
            raise ValueError("Variant frame_rate_reason does not match the profile frame-rate contract.")
        if row.get("watermark"):
            raise ValueError("Explicit frame_rate with VideoSeal is not yet qualified; use watermark off or the unchanged legacy route.")
        retains_requested |= rate == requested
    if not retains_requested:
        raise ValueError("frame_rate requires a master retaining the requested cadence.")
    return policy


def validate_requested_frame_rate(
    info: dict[str, Any],
    policy: dict[str, Any],
    *,
    source_info: dict[str, Any] | None = None,
    max_fps: int | None = None,
) -> None:
    normalized = normalize_frame_rate(policy)
    if normalized["mode"] == "preserve_source":
        if source_info is None:
            raise RuntimeError("Source-preserve output verification requires the original source PTS reference.")
        if max_fps is None:
            raise RuntimeError("Source-preserve output verification requires the declared profile FPS capability.")
        validate_source_timing_preserved(source_info, info, phase="Final delivery source-PTS verification")
        validate_source_timing_profile_cap(source_info, max_fps)
        return
    expected = Fraction(normalized["fps"])
    actual = source_frame_rate(info)
    # A float-only legacy test/probe can tolerate representation error, not a
    # 59.94 -> 60 or 29.97 -> 30 substitution. Native probes retain the ratio.
    matches = actual == expected if info.get("fps_ratio") else abs(float(actual - expected)) < 0.000001
    if not matches:
        raise RuntimeError(f"Output frame rate is {actual} FPS; expected exact {expected} FPS.")


def validate_cadence_extent(source_info: dict[str, Any], output_info: dict[str, Any], policy: dict[str, Any]) -> None:
    """Validate the exact source-derived CFR frame extent and its stream duration.

    Upsampling uses ``source_end_context_v1``: the output covers the source's
    half-open interval with ceil(source_frames * target_rate / source_rate)
    frames. Its final frame may therefore extend to the next CFR boundary while
    audio retains its exact source sample extent. Down-conversion may quantize
    to either adjacent frame count. These are metadata checks; decoded-frame
    and visual-quality qualification remain separate acceptance measurements.
    """
    rate = Fraction(normalize_frame_rate(policy)["fps"])
    try:
        source_rate = source_frame_rate(source_info)
        source_count_text = str(source_info.get("video_frame_count") or "")
        output_count_text = str(output_info.get("video_frame_count") or "")
        source_duration = Fraction(str(source_info.get("video_duration") or 0))
        output_duration = Fraction(str(output_info.get("video_duration") or 0))
        source_time_base = Fraction(str(source_info.get("video_time_base") or "0"))
        output_time_base = Fraction(str(output_info.get("video_time_base") or "0"))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError, RuntimeError) as exc:
        raise RuntimeError("Cannot verify explicit-cadence duration from current video evidence.") from exc
    if source_info.get("frame_rate_evidence") != "presentation-timestamps" or source_info.get("video_pts_uniform") is not True:
        raise RuntimeError("Explicit-cadence source requires uniform decoded presentation timestamp evidence.")
    if not source_count_text.isdecimal() or int(source_count_text) < 1:
        raise RuntimeError("Explicit-cadence source frame count is missing or invalid.")
    if not output_count_text.isdecimal() or int(output_count_text) < 1:
        raise RuntimeError("Final explicit-cadence video frame count is missing or invalid.")
    if source_time_base <= 0 or output_time_base <= 0:
        raise RuntimeError("Explicit-cadence source and output require positive exact video time bases.")
    if source_duration <= 0 or output_duration <= 0:
        raise RuntimeError("Explicit-cadence duration evidence must be positive.")

    source_count = int(source_count_text)
    output_count = int(output_count_text)
    source_pts_count = source_info.get("video_pts_count")
    if source_pts_count is not None and int(source_pts_count) != source_count:
        raise RuntimeError("Explicit-cadence source frame count disagrees with decoded presentation timestamps.")
    exact_source_duration = Fraction(source_count, 1) / source_rate
    source_tolerance = max(source_time_base, Fraction(1, 1000000))
    if abs(source_duration - exact_source_duration) > source_tolerance:
        raise RuntimeError(
            f"Explicit-cadence source duration {source_duration}s disagrees with "
            f"{source_count} frames at {source_rate} FPS ({exact_source_duration}s)."
        )

    expected = Fraction(source_count, 1) * rate / source_rate
    if rate > source_rate:
        allowed_counts = {math.ceil(expected)}
    else:
        allowed_counts = {math.floor(expected), math.ceil(expected)}
    if output_count not in allowed_counts:
        expected_text = ", ".join(str(value) for value in sorted(allowed_counts))
        raise RuntimeError(
            f"Final video frame count {output_count} does not match the source-end "
            f"frame-grid contract at {rate} FPS; expected {expected_text}."
        )

    exact_output_duration = Fraction(output_count, 1) / rate
    output_tolerance = max(output_time_base, Fraction(1, 1000000))
    if abs(output_duration - exact_output_duration) > output_tolerance:
        raise RuntimeError(
            f"Final video duration {output_duration}s does not match {output_count} "
            f"frames at {rate} FPS ({exact_output_duration}s)."
        )


def validate_source_timing_preserved(
    source_info: dict[str, Any],
    candidate_info: dict[str, Any],
    *,
    phase: str,
) -> None:
    """Require exact picture-start timestamps and the final presentation endpoint."""
    for label, info in (("source", source_info), ("candidate", candidate_info)):
        if info.get("frame_rate_evidence") != "presentation-timestamps":
            raise RuntimeError(f"{phase}: {label} lacks decoded presentation timestamp evidence.")
        digest = str(info.get("video_pts_sha256") or "")
        try:
            count = int(info.get("video_pts_count") or 0)
            time_base = Fraction(str(info.get("video_time_base") or "0"))
            pts_time_base = Fraction(str(info.get("video_pts_time_base") or "0"))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise RuntimeError(f"{phase}: {label} has invalid presentation timestamp metadata.") from exc
        if count < 2 or not digest or time_base <= 0 or pts_time_base != time_base:
            raise RuntimeError(f"{phase}: {label} has incomplete presentation timestamp evidence.")
    if int(source_info["video_pts_count"]) != int(candidate_info["video_pts_count"]):
        raise RuntimeError(
            f"{phase}: frame count changed from {source_info['video_pts_count']} "
            f"to {candidate_info['video_pts_count']} during source-timing preservation."
        )
    if source_info["video_pts_sha256"] != candidate_info["video_pts_sha256"]:
        raise RuntimeError(f"{phase}: decoded presentation timestamps changed during source-timing preservation.")
    for key in ("video_pts_start", "video_pts_end", "video_pts_span"):
        try:
            source_value = Fraction(str(source_info[key]))
            candidate_value = Fraction(str(candidate_info[key]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise RuntimeError(f"{phase}: {key} evidence is missing or invalid.") from exc
        if source_value != candidate_value:
            raise RuntimeError(f"{phase}: {key} changed from {source_value} to {candidate_value}.")

    endpoints: list[Fraction] = []
    for label, info in (("source", source_info), ("candidate", candidate_info)):
        if info.get("video_pts_endpoint_evidence") != "decoded-last-frame-duration":
            raise RuntimeError(f"{phase}: {label} lacks exact decoded last-frame endpoint evidence.")
        try:
            last_duration = Fraction(str(info["video_pts_last_duration"]))
            end_exclusive = Fraction(str(info["video_pts_end_exclusive"]))
            last_start = Fraction(str(info["video_pts_end"]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise RuntimeError(f"{phase}: {label} last-frame endpoint evidence is invalid.") from exc
        if last_duration <= 0 or end_exclusive != last_start + last_duration:
            raise RuntimeError(f"{phase}: {label} last-frame endpoint is inconsistent with its duration.")
        endpoints.append(end_exclusive)
    if endpoints[0] != endpoints[1]:
        raise RuntimeError(
            f"{phase}: last-frame endpoint changed from {endpoints[0]} to {endpoints[1]} "
            "during source-timing preservation."
        )


def interpolation_boundary_contract(info: dict[str, Any], target_fps: int | str) -> dict[str, Any]:
    """Provide one held source sample as model context beyond the authored end.

    Apollo otherwise stops at the last source sample timestamp, omitting that
    sample's presentation interval. Context is supplied BEFORE interpolation;
    output is trimmed to the exact source-derived half-open frame extent. No
    post-model padding is permitted to conceal arbitrary missing output.
    """
    try:
        count_text = str(info.get("video_frame_count") or "")
        if not count_text.isdecimal() or info.get("is_vfr"):
            raise ValueError("unknown frame count or VFR")
        count = int(count_text)
        rate = source_frame_rate(info)
        target = Fraction(normalize_frame_rate({"mode": "cfr", "fps": target_fps})["fps"])
        duration = Fraction(str(info.get("video_duration") or "0"))
        tick = Fraction(str(info.get("video_time_base") or "0"))
        start = Fraction(str(info.get("video_pts_start") or info["video_start_time"]))
        if count < 2 or count > 1_000_000_000 or target <= rate or tick <= 0 or start != 0:
            raise ValueError("unqualified source interval")
        if abs(duration - Fraction(count, 1) / rate) > max(tick, Fraction(1, 1000000)):
            raise ValueError("duration and CFR frame count disagree")
    except (KeyError, ValueError, ZeroDivisionError, OverflowError, RuntimeError) as exc:
        raise ValueError("Explicit interpolation endpoint requires consistent CFR frame count, duration and zero-origin timing.") from exc
    return {"schema": "source_end_context_v1", "source_fps": str(rate), "source_frames": count,
            "source_duration": str(Fraction(count, 1) / rate), "target_fps": str(target),
            "context_source_frames": 1, "trim_end_frame": math.ceil(Fraction(count, 1) * target / rate)}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def topaz_model_data_directory_value(
    model_dir: str | Path,
    *,
    separator: str | None = None,
) -> str:
    """Return Topaz's trailing-separator model-data value for the active host."""

    chosen_separator = os.sep if separator is None else separator
    if chosen_separator not in {"/", "\\"}:
        raise ValueError(f"Unsupported path separator: {chosen_separator!r}")
    return str(model_dir).rstrip("\\/") + chosen_separator


def configure_topaz_model_environment(model_dir: str | Path) -> str:
    """Configure Topaz from one discovered, existing model directory."""

    text = str(model_dir or "").strip()
    if not text:
        raise FileNotFoundError(
            "Topaz model directory was not discovered. Set TVAI_MODEL_DIR to the "
            "installed Topaz Video models folder and retry."
        )
    resolved = Path(text).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Discovered Topaz model directory does not exist: {resolved}"
        )
    value = str(resolved)
    os.environ["TVAI_MODEL_DIR"] = value
    os.environ["TVAI_MODEL_DATA_DIR"] = topaz_model_data_directory_value(value)
    return value


def sampled_file_identity(path: str | Path) -> dict[str, Any]:
    """Return the shared sparse identity plus checkpoint-local file state."""

    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    fingerprint = file_sample_fingerprint(resolved, sample_bytes=CHECKPOINT_SAMPLE_BYTES)
    if not fingerprint.exists:
        raise OSError(f"Could not read sparse checkpoint identity: {resolved}")
    return {
        "path": str(resolved),
        "bytes": int(fingerprint.file_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_algorithm": fingerprint.algorithm,
        "sample_bytes": int(fingerprint.sample_bytes),
        "first_hash": fingerprint.first_hash,
        "middle_hash": fingerprint.middle_hash,
        "last_hash": fingerprint.last_hash,
        "media_uid": fingerprint.media_uid,
    }


_CONTENT_IDENTITY_FIELDS = (
    "bytes",
    "sample_algorithm",
    "sample_bytes",
    "first_hash",
    "middle_hash",
    "last_hash",
    "media_uid",
)


def _content_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {key: identity.get(key) for key in _CONTENT_IDENTITY_FIELDS}


def sampled_file_content_matches(path: str | Path, stored_identity: dict[str, Any]) -> bool:
    """Compare a current file with a stored sparse content identity."""

    if not isinstance(stored_identity, dict) or not stored_identity:
        return False
    try:
        observed = sampled_file_identity(path)
    except OSError:
        return False
    return _content_identity(observed) == _content_identity(stored_identity)


def _path_key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


@lru_cache(maxsize=512)
def _cached_static_asset_identity(
    path: str,
    file_size: int,
    mtime_ns: int,
    ctime_ns: int,
    file_id: int,
) -> dict[str, Any]:
    del file_size, mtime_ns, ctime_ns, file_id
    return sampled_file_identity(path)


def topaz_model_asset_identity(model_dir: str | Path, model: str) -> dict[str, Any]:
    """Bind reuse to the selected Topaz descriptor and downloaded model assets."""

    root = Path(model_dir).expanduser().resolve() if str(model_dir or "").strip() else None
    model_id = str(model or "").strip().casefold()
    match = re.fullmatch(r"([a-z]+)-(\d+)", model_id)
    asset_prefix = f"{match.group(1)}-v{match.group(2)}" if match else model_id
    descriptor = root / f"{model_id}.json" if root is not None else None
    candidates: list[Path] = []
    if root is not None and root.is_dir():
        if descriptor is not None and descriptor.is_file():
            candidates.append(descriptor)
        candidates.extend(
            path
            for path in root.glob(f"{asset_prefix}*")
            if path.is_file()
            and path != descriptor
            and path.suffix.casefold() in {".tz3", ".onnx", ".bin", ".pt", ".pth", ".safetensors"}
        )
    identities = []
    for path in sorted(set(candidates), key=lambda item: str(item).casefold()):
        stat = path.stat()
        identities.append(
            _cached_static_asset_identity(
                str(path.resolve()),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
                int(getattr(stat, "st_ino", 0)),
            )
        )
    payload = {
        "model_dir": str(root) if root is not None else "",
        "model": model_id,
        "asset_prefix": asset_prefix,
        "descriptor_present": bool(descriptor is not None and descriptor.is_file()),
        "asset_files": identities,
        "weight_or_engine_count": sum(1 for row in identities if not str(row["path"]).casefold().endswith(".json")),
    }
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["resume_safe"] = bool(payload["descriptor_present"] and payload["weight_or_engine_count"])
    return payload


def build_topaz_checkpoint_fingerprint(
    *,
    stage: str,
    input_path: str | Path,
    topaz_ffmpeg: str | Path,
    preset_id: str,
    model: str,
    model_params: str = "",
    target_width: int,
    target_height: int,
    target_fps: int | str | None = None,
    model_asset_identity: dict[str, Any] | None = None,
    enhancement_recipe: dict[str, Any] | None = None,
    frame_rate: dict[str, Any] | None = None,
    adaptive_fingerprint: str | None = None,
    working_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_identity = sampled_file_identity(input_path)
    runtime_identity = sampled_file_identity(topaz_ffmpeg)
    input_identity.pop("path", None)
    runtime_identity.pop("path", None)
    contract = {
        "schema": TOPAZ_CHECKPOINT_SCHEMA,
        "stage": str(stage),
        "input": input_identity,
        "runtime": runtime_identity,
        "preset_id": str(preset_id),
        "model": str(model),
        "model_params": str(model_params),
        "model_assets": dict(model_asset_identity or {}),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "target_fps": (normalize_frame_rate({"mode": "cfr", "fps": target_fps})["fps"]
                       if isinstance(target_fps, str) else int(target_fps) if target_fps is not None else None),
    }
    if enhancement_recipe is not None:
        contract["enhancement_recipe"] = dict(enhancement_recipe)
    if adaptive_fingerprint is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", adaptive_fingerprint):
            raise ValueError("Invalid adaptive execution fingerprint.")
        contract["adaptive_fingerprint"] = adaptive_fingerprint
    if frame_rate is not None:
        contract["frame_rate"] = normalize_frame_rate(frame_rate)
    if working_geometry is not None:
        contract["working_geometry"] = dict(working_geometry)
    fingerprint = _topaz_contract_fingerprint(contract)
    return {"fingerprint": fingerprint, "contract": contract}


def _topaz_contract_fingerprint(contract: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validated_checkpoint_expectation(expected: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    contract = expected.get("contract") if isinstance(expected.get("contract"), dict) else {}
    fingerprint = str(expected.get("fingerprint") or "")
    if contract.get("schema") != TOPAZ_CHECKPOINT_SCHEMA:
        raise ValueError("Topaz checkpoint expectation has an unsupported schema.")
    if not fingerprint or fingerprint != _topaz_contract_fingerprint(contract):
        raise ValueError("Topaz checkpoint expectation fingerprint is invalid.")
    return fingerprint, contract


def validate_topaz_stage_identity_stable(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    """Fail closed when a render's input, runtime, parameters, or assets drift."""

    before_fingerprint, before_contract = _validated_checkpoint_expectation(before)
    after_fingerprint, after_contract = _validated_checkpoint_expectation(after)
    if before_fingerprint == after_fingerprint and before_contract == after_contract:
        return before

    changed_fields = sorted(
        key
        for key in set(before_contract).union(after_contract)
        if before_contract.get(key) != after_contract.get(key)
    )
    changed_text = ", ".join(changed_fields) or "unknown contract fields"
    raise RuntimeError(
        f"{stage} identity changed while the stage was running ({changed_text}). "
        "The output may not match one stable input/runtime/model state, so it will not be "
        "checkpointed or delivered. Rerun after the source, Topaz runtime, and model assets "
        "have stopped changing."
    )


def checkpoint_expectation_resume_safe(expected: dict[str, Any]) -> bool:
    """Return whether an otherwise valid checkpoint has discoverable model assets."""

    _fingerprint, contract = _validated_checkpoint_expectation(expected)
    model_assets = contract.get("model_assets")
    return bool(isinstance(model_assets, dict) and model_assets.get("resume_safe"))


def topaz_checkpoint_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(f"{output.name}.checkpoint.json")


def write_topaz_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_path: str | Path,
    expected: dict[str, Any],
) -> Path:
    checkpoint = Path(checkpoint_path)
    output = Path(output_path).expanduser().resolve()
    fingerprint, contract = _validated_checkpoint_expectation(expected)
    if output.parent != checkpoint.expanduser().resolve().parent:
        raise ValueError("Topaz checkpoint output must remain inside the job temp directory.")
    payload = {
        "schema": TOPAZ_CHECKPOINT_SCHEMA,
        "fingerprint": fingerprint,
        "contract": contract,
        "output": sampled_file_identity(output),
    }
    atomic_write_json(checkpoint, payload)
    return checkpoint


def resolve_topaz_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected: dict[str, Any],
) -> Path | None:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        return None
    try:
        fingerprint, contract = _validated_checkpoint_expectation(expected)
        payload = load_json(checkpoint)
        if payload.get("schema") != TOPAZ_CHECKPOINT_SCHEMA:
            return None
        if str(payload.get("fingerprint") or "") != fingerprint:
            return None
        if payload.get("contract") != contract:
            return None
        stored_output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
        output = Path(str(stored_output.get("path") or "")).expanduser().resolve()
        if output.parent != checkpoint.parent or not output.is_file():
            return None
        if sampled_file_identity(output) != stored_output:
            return None
        return output
    except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        return None


def retain_verified_topaz_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_path: str | Path,
    expected: dict[str, Any],
) -> dict[str, Any] | None:
    """Persist a checkpoint only when its model assets and receipt revalidate."""

    checkpoint = Path(checkpoint_path)
    if not checkpoint_expectation_resume_safe(expected):
        checkpoint.unlink(missing_ok=True)
        return None
    write_topaz_checkpoint(checkpoint, output_path=output_path, expected=expected)
    resolved = resolve_topaz_checkpoint(checkpoint, expected=expected)
    if resolved is None or _path_key(resolved) != _path_key(output_path):
        checkpoint.unlink(missing_ok=True)
        raise RuntimeError(
            "Topaz checkpoint failed its immediate identity revalidation; "
            "the stage cannot be marked reusable."
        )
    return expected


def validated_resume_temp_dir(
    jobs_root: str | Path,
    resume_job_id: str,
    *,
    current_job_id: str,
) -> Path | None:
    if not str(resume_job_id or "").strip():
        return None
    if resume_job_id == current_job_id:
        raise ValueError("Resume job must be an earlier job, not the current job.")
    previous = resolve_job_dir(jobs_root, resume_job_id)
    try:
        progress = load_json(previous / "progress.json")
        manifest = load_json(previous / "manifest.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read resume job {resume_job_id}: {exc}") from exc
    state = str(progress.get("state") or "").strip().casefold()
    worker = str(progress.get("worker_name") or manifest.get("worker") or "").strip()
    if state not in {"error", "crashed", "cancelled", "canceled"}:
        raise ValueError(f"Resume job must be failed or cancelled; got {state or 'unknown'}.")
    if worker != "video_upscale":
        raise ValueError(f"Resume job belongs to {worker or 'an unknown worker'}, not video_upscale.")
    temp = previous / "temp"
    if not temp.is_dir():
        raise ValueError(f"Resume job has no temp directory: {temp}")
    return temp.resolve()


def materialize_resume_output(source_path: str | Path, destination_path: str | Path) -> tuple[Path, str]:
    """Create a current-job link/copy without mutating prior failure evidence."""

    source = Path(source_path).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()
    if source == destination:
        return destination, "current_job"
    if not source.is_file():
        raise FileNotFoundError(f"Resume output is missing: {source}")
    if destination.exists():
        raise FileExistsError(f"Current job resume destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    method = "hardlink"
    try:
        os.link(source, destination)
    except OSError:
        method = "copy"
        shutil.copy2(source, destination)
    source_identity = sampled_file_identity(source)
    destination_identity = sampled_file_identity(destination)
    if _content_identity(source_identity) != _content_identity(destination_identity):
        destination.unlink(missing_ok=True)
        raise RuntimeError("Materialized resume output failed sparse identity verification.")
    return destination, method


def resolve_current_or_retry_checkpoint(
    *,
    current_checkpoint: Path,
    current_output: Path,
    retry_temp_dir: Path | None,
    expected: dict[str, Any],
) -> tuple[Path | None, str]:
    current = resolve_topaz_checkpoint(current_checkpoint, expected=expected)
    if current is not None:
        return current, "current_job"
    if retry_temp_dir is None:
        return None, ""
    previous_checkpoint = retry_temp_dir / current_checkpoint.name
    previous = resolve_topaz_checkpoint(previous_checkpoint, expected=expected)
    if previous is None:
        return None, ""
    materialized, method = materialize_resume_output(previous, current_output)
    return materialized, f"retry_{method}"


def validate_intermediate_audio(
    source_info: dict[str, Any],
    intermediate_info: dict[str, Any],
    *,
    phase: str,
    require_sample_extent: bool = False,
) -> None:
    source_count = int(
        source_info.get("audio_stream_count")
        if source_info.get("audio_stream_count") is not None
        else bool(source_info.get("has_audio"))
    )
    intermediate_count = int(
        intermediate_info.get("audio_stream_count")
        if intermediate_info.get("audio_stream_count") is not None
        else bool(intermediate_info.get("has_audio"))
    )
    if source_count != intermediate_count:
        raise RuntimeError(
            f"{phase} changed the source audio stream count "
            f"({source_count} -> {intermediate_count}); refusing checkpoint and final export."
        )
    if not require_sample_extent or source_count == 0:
        return
    source_streams = source_info.get("audio_streams")
    intermediate_streams = intermediate_info.get("audio_streams")
    if (
        not isinstance(source_streams, list)
        or not isinstance(intermediate_streams, list)
        or len(source_streams) != source_count
        or len(intermediate_streams) != intermediate_count
    ):
        raise RuntimeError(f"{phase} cannot prove the sample extent of every source audio track.")
    for index, (source_track, output_track) in enumerate(zip(source_streams, intermediate_streams), start=1):
        if not isinstance(source_track, dict) or not isinstance(output_track, dict):
            raise RuntimeError(f"{phase} has incomplete sample evidence for audio track {index}.")
        try:
            source_rate = int(source_track.get("sample_rate") or 0)
            output_rate = int(output_track.get("sample_rate") or 0)
            source_channels = int(source_track.get("channels") or 0)
            output_channels = int(output_track.get("channels") or 0)
            source_samples = Fraction(str(source_track["sample_count"]))
            output_samples = Fraction(str(output_track["sample_count"]))
            source_start = Fraction(str(source_track.get("start_time") or "0"))
            output_start = Fraction(str(output_track.get("start_time") or "0"))
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise RuntimeError(f"{phase} cannot prove sample timing for audio track {index}.") from exc
        if source_rate <= 0 or output_rate != source_rate:
            raise RuntimeError(
                f"{phase} changed audio track {index} sample rate ({source_rate} -> {output_rate})."
            )
        if source_channels <= 0 or output_channels != source_channels:
            raise RuntimeError(
                f"{phase} changed audio track {index} channel count ({source_channels} -> {output_channels})."
            )
        if source_samples.denominator != 1 or output_samples != source_samples:
            raise RuntimeError(
                f"{phase} changed audio track {index} valid sample extent "
                f"({source_samples} -> {output_samples})."
            )
        if abs(output_start - source_start) > Fraction(1, source_rate):
            raise RuntimeError(
                f"{phase} shifted audio track {index} start time "
                f"({source_start} -> {output_start})."
            )


def temporary_videoseal_receipt_path(job_temp_dir: Path, output_path: Path) -> Path:
    identity = hashlib.sha256(str(output_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return job_temp_dir / "videoseal_receipts" / f"{output_path.name}.{identity}.videoseal.json"


def _require_regular_durable_receipt(path: Path) -> None:
    metadata = os.lstat(path)
    reparse = int(getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if not stat_module.S_ISREG(metadata.st_mode) or int(getattr(metadata, "st_file_attributes", 0)) & reparse:
        raise FileExistsError(f"VideoSeal durable receipt conflict: unsafe existing path preserved: {path}")


def validate_delivery_watermark_receipt(
    output: Path, variant: dict[str, Any], watermark_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind the canonical signed receipt to this completed delivery variant."""
    from workflow.video_provenance_watermark import inspect_signed_video_receipt

    result = watermark_result if isinstance(watermark_result, dict) else {}
    if result.get("status") != "pass" or not str(result.get("receipt") or ""):
        raise ValueError("Completed watermark delivery is missing a passed signed receipt.")
    if not str(result.get("output") or "") or _path_key(result["output"]) != _path_key(output):
        raise ValueError("Completed watermark delivery output binding mismatch.")
    if result.get("receipt_storage") == "adjacent_output_v1":
        _require_regular_durable_receipt(Path(str(result["receipt"])))
    evidence = inspect_signed_video_receipt(output, str(result["receipt"]))
    for field, expected in (
        ("asset_id", str(result.get("asset_id") or "")),
        ("namespace", str(variant.get("namespace") or "")),
        ("identifier", str(variant.get("watermark_identifier") or "")),
        ("delivery_profile_id", str(variant.get("profile_id") or "")),
    ):
        if evidence[field] != expected or str(result.get(field) or "") != expected:
            raise ValueError(f"Completed watermark signed receipt binding mismatch: {field}")
    for field in ("receipt_sha256", "receipt_bytes", "output_sha256", "output_bytes"):
        if field in result and result[field] != evidence[field]:
            raise ValueError(f"Completed watermark signed receipt identity mismatch: {field}")
    if result.get("receipt_storage") == "adjacent_output_v1" and _path_key(result["receipt"]) != _path_key(
        output.with_suffix(output.suffix + ".videoseal.json")
    ):
        raise ValueError("Completed watermark durable receipt location mismatch.")
    return evidence


def publish_delivery_watermark_receipt(
    output: Path, variant: dict[str, Any], watermark_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Retain exact signed bytes beside verified media; never replace a sidecar.

    Called only after the worker's final media gates. Staging remains available
    for failure diagnosis/legacy jobs and may be pruned independently afterward.
    """
    from workflow.atomic_io import publish_file_exclusive
    from workflow.video_provenance_watermark import MAX_SIGNED_RECEIPT_BYTES

    evidence = validate_delivery_watermark_receipt(output, variant, watermark_result)
    with Path(evidence["receipt"]).open("rb") as handle:
        raw = handle.read(MAX_SIGNED_RECEIPT_BYTES + 1)
    if len(raw) != evidence["receipt_bytes"] or hashlib.sha256(raw).hexdigest() != evidence["receipt_sha256"]:
        raise ValueError("VideoSeal receipt staging changed after validation.")
    durable = output.resolve().with_suffix(output.suffix + ".videoseal.json")
    # Inspect the lexical leaf before any resolve()/same-path shortcut: a link
    # to staging is not durable even when its current bytes are identical.
    if os.path.lexists(durable):
        _require_regular_durable_receipt(durable)
    if _path_key(evidence["receipt"]) != _path_key(durable):
        # Stage the bounded metadata on the destination volume. The canonical
        # create-if-absent primitive consumes only this private disposable copy.
        with tempfile.TemporaryDirectory(prefix=".videoseal-publication-", dir=durable.parent) as private:
            stage = Path(private) / "receipt.json"
            with stage.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                publish_file_exclusive(stage, durable)
            except FileExistsError:
                _require_regular_durable_receipt(durable)
                with durable.open("rb") as handle:
                    existing = handle.read(MAX_SIGNED_RECEIPT_BYTES + 1)
                if existing != raw:
                    raise FileExistsError(f"VideoSeal durable receipt conflict; existing file preserved: {durable}") from None
    result = {
        **dict(watermark_result or {}),
        "receipt": str(durable),
        "receipt_sha256": evidence["receipt_sha256"],
        "receipt_bytes": evidence["receipt_bytes"],
        "receipt_storage": "adjacent_output_v1",
    }
    validate_delivery_watermark_receipt(output, variant, result)
    return result


def prune_temporary_videoseal_receipts(
    jobs_root: str | Path,
    *,
    older_than_days: float = VIDEOSEAL_RECEIPT_RETENTION_DAYS,
) -> dict[str, int]:
    root = Path(jobs_root)
    cutoff = time.time() - max(0.0, float(older_than_days)) * 86400
    scanned = 0
    removed = 0
    if not root.is_dir():
        return {"scanned": 0, "removed": 0}
    for job_dir in root.iterdir():
        receipt_dir = job_dir / "temp" / "videoseal_receipts"
        if not receipt_dir.is_dir():
            continue
        for receipt in receipt_dir.glob("*.videoseal.json"):
            scanned += 1
            try:
                if receipt.stat().st_mtime <= cutoff:
                    receipt.unlink()
                    removed += 1
            except OSError:
                continue
        try:
            receipt_dir.rmdir()
        except OSError:
            pass
    return {"scanned": scanned, "removed": removed}


def load_delivery_plan(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    plan_path = Path(path).expanduser().resolve()
    payload = load_json(plan_path)
    if not isinstance(payload, dict):
        raise ValueError("Delivery plan must be an object.")
    schema = payload.get("schema")
    if schema not in {
        DELIVERY_PLAN_SCHEMA,
        "spiffy_final_delivery_plan_v2",
        "spiffy_final_delivery_plan_v3",
        "spiffy_final_delivery_plan_v4",
    }:
        raise ValueError(f"Unsupported delivery plan schema: {payload.get('schema')!r}")
    has_adaptive = "adaptive_enhancement" in payload
    has_frame_rate = "frame_rate" in payload
    has_geometry = "geometry" in payload
    if has_adaptive and schema not in {"spiffy_final_delivery_plan_v3", "spiffy_final_delivery_plan_v4"}:
        raise ValueError("Adaptive enhancement requires plan v3 so older workers refuse rather than ignore it.")
    if schema == "spiffy_final_delivery_plan_v3" and not has_adaptive:
        raise ValueError("Plan v3 requires its adaptive enhancement contract.")
    if has_frame_rate and schema not in {
        "spiffy_final_delivery_plan_v2", "spiffy_final_delivery_plan_v3", "spiffy_final_delivery_plan_v4",
    }:
        raise ValueError("Explicit frame_rate requires plan v2 so older workers refuse rather than ignore it.")
    if schema == "spiffy_final_delivery_plan_v2" and not has_frame_rate:
        raise ValueError("Plan v2 requires its explicit frame_rate contract.")
    if has_geometry != (schema == "spiffy_final_delivery_plan_v4"):
        raise ValueError("Custom delivery geometry requires plan v4 so older workers refuse rather than ignore it.")
    if has_geometry:
        geometry = normalize_delivery_geometry(payload.get("geometry"))
        if geometry is None or geometry != payload.get("geometry"):
            raise ValueError("Delivery geometry must be canonical before execution.")
        if geometry.get("working") is not None and has_adaptive:
            raise ValueError("Custom working geometry cannot be combined with adaptive enhancement.")
        requested_profiles = [str(value) for value in payload.get("requested_profile_ids") or []]
        if "smart_finish_master" not in requested_profiles:
            raise ValueError("Custom delivery geometry requires the smart_finish_master output profile.")
        if geometry.get("working") is not None and "native_master" in requested_profiles:
            raise ValueError("Custom working geometry cannot change the isolated legacy native_master path.")
    if not isinstance(payload.get("variants"), list) or not payload["variants"]:
        raise ValueError("Delivery plan must contain at least one variant.")
    for index, variant in enumerate(payload["variants"], start=1):
        if not isinstance(variant, dict):
            raise ValueError(f"Delivery variant {index} must be an object.")
        geometry_policy = variant.get("geometry_policy")
        if geometry_policy is None:
            continue
        if geometry_policy not in {"source_display", "custom_final"}:
            raise ValueError(f"Unsupported delivery geometry policy for variant {index}: {geometry_policy!r}")
        if str(variant.get("profile_id") or "") != "smart_finish_master":
            raise ValueError(f"{geometry_policy} geometry requires smart_finish_master.")
        if int(variant.get("width") or 0) != 0 or int(variant.get("height") or 0) != 0:
            raise ValueError(f"{geometry_policy} geometry requires unresolved dimensions in the plan.")
        if str(variant.get("layout") or "") != "fit_into_frame":
            raise ValueError(f"{geometry_policy} geometry requires the non-destructive fit layout.")
        if geometry_policy == "custom_final":
            final_geometry = (payload.get("geometry") or {}).get("final") if isinstance(payload.get("geometry"), dict) else None
            if not isinstance(final_geometry, dict) or final_geometry.get("mode") not in {"exact", "long_edge", "preset"}:
                raise ValueError("custom_final geometry requires a custom final geometry contract.")
            if variant.get("geometry") != final_geometry:
                raise ValueError("custom_final variant geometry does not match the job geometry contract.")
        elif variant.get("geometry") is not None:
            raise ValueError("source_display variant cannot carry a custom final geometry override.")
    if has_geometry:
        final_geometry = payload["geometry"].get("final")
        custom_final_rows = [
            row for row in payload["variants"] if isinstance(row, dict) and row.get("geometry_policy") == "custom_final"
        ]
        if isinstance(final_geometry, dict) and final_geometry.get("mode") in {"exact", "long_edge", "preset"}:
            if len(custom_final_rows) != 1:
                raise ValueError("Custom final geometry requires exactly one smart_finish_master custom_final variant.")
        elif custom_final_rows:
            raise ValueError("custom_final variant requires a custom final geometry contract.")
    recipe = delivery_enhancement_recipe(payload, str(payload.get("enhance_preset") or "auto"))
    if "adaptive_enhancement" in payload:
        from adaptive_enhancement import normalize_policy, engine_identity
        normalize_policy(payload["adaptive_enhancement"])
        if payload.get("adaptive_engine_identity") != engine_identity():
            raise ValueError("Adaptive engine changed since planning; rebuild this job plan before execution.")
        if recipe.up_model != "prob-4" or (payload.get("enhancement") or {}).get("overrides") or any(row.get("watermark") for row in payload["variants"]):
            raise ValueError("Adaptive natural requires local Proteus, watermark off, and no conflicting manual overrides.")
    validate_frame_rate_plan(payload)
    payload["plan_path"] = str(plan_path)
    return payload


def source_display_geometry(info: dict[str, Any]) -> dict[str, Any]:
    """Resolve source display geometry after SAR correction and orthogonal rotation."""
    source_width = int(info.get("width") or 0)
    source_height = int(info.get("height") or 0)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Smart Finish source-size delivery requires positive probed source dimensions.")
    sar_text = str(info.get("sar") or "1:1").strip()
    try:
        sar = Fraction(sar_text.replace(":", "/"))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Smart Finish source-size delivery cannot interpret source SAR {sar_text!r}.") from exc
    if sar <= 0:
        raise ValueError(f"Smart Finish source-size delivery SAR must be positive; found {sar_text!r}.")
    if info.get("rotation_metadata_error"):
        raise ValueError("Smart Finish source-size delivery cannot interpret source rotation metadata.")
    try:
        raw_rotation = info.get("rotation") or 0
        if isinstance(raw_rotation, bool):
            raise ValueError("boolean rotation")
        rotation_value = float(raw_rotation)
        if not math.isfinite(rotation_value) or not rotation_value.is_integer():
            raise ValueError("rotation must be a finite whole-degree value")
        rotation = int(rotation_value) % 360
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Smart Finish source-size delivery cannot interpret source rotation metadata.") from exc
    if rotation not in {0, 90, 180, 270}:
        raise ValueError(
            "Smart Finish source-size delivery supports only orthogonal rotation metadata "
            f"(0, 90, 180, 270 degrees); found {rotation} degrees."
        )
    if rotation in {90, 270}:
        display_width_exact = Fraction(source_height, 1) / sar
        display_height = source_width
    else:
        display_width_exact = Fraction(source_width, 1) * sar
        display_height = source_height
    # FFmpeg's autorotate path swaps the SAR axis as it rotates the coded frame.
    # Round the resulting square-pixel display width to the nearest whole pixel.
    display_width = int(display_width_exact + Fraction(1, 2))
    if display_width <= 0 or display_height <= 0:
        raise ValueError("Smart Finish source-size delivery resolved non-positive display dimensions.")
    display_aspect = display_width_exact / display_height
    output_width = display_width + (display_width % 2)
    output_height = display_height + (display_height % 2)
    return {
        "policy": "source_display",
        "source_width": source_width,
        "source_height": source_height,
        "source_sample_aspect_ratio": f"{sar.numerator}:{sar.denominator}",
        "source_rotation_degrees": rotation,
        "display_aspect_ratio": f"{display_aspect.numerator}:{display_aspect.denominator}",
        "sample_aspect_ratio": "1:1",
        "rotation_degrees": 0,
        "display_width": display_width,
        "display_height": display_height,
        "output_width": output_width,
        "output_height": output_height,
        "alignment_padding": {
            "horizontal": output_width - display_width,
            "vertical": output_height - display_height,
            "placement": "centered",
        },
    }


def validate_source_display_square_pixels(info: dict[str, Any], *, phase: str) -> None:
    """Require the autorotated working/output stream to be square-pixel and metadata-upright."""
    sar_text = str(info.get("sar") or "0").strip()
    try:
        sar = Fraction(sar_text.replace(":", "/"))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise RuntimeError(f"{phase} has unreadable sample-aspect-ratio evidence: {sar_text!r}.") from exc
    if sar != 1:
        raise RuntimeError(f"{phase} did not normalize source display geometry to square pixels (SAR 1:1); found {sar_text}.")
    if info.get("rotation_metadata_error"):
        raise RuntimeError(f"{phase} has unreadable rotation evidence.")
    try:
        raw_rotation = info.get("rotation")
        if isinstance(raw_rotation, bool):
            raise ValueError("boolean rotation")
        rotation_value = float(raw_rotation or 0)
        if not math.isfinite(rotation_value) or not rotation_value.is_integer():
            raise ValueError("rotation must be a finite whole-degree value")
        rotation = int(rotation_value) % 360
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{phase} has unreadable rotation evidence.") from exc
    if rotation != 0:
        raise RuntimeError(f"{phase} retained unapplied rotation metadata ({rotation} degrees).")


def validate_source_color_pipeline(info: dict[str, Any]) -> None:
    """Do not silently flatten transparency or misinterpret HDR as SDR.

    A color-managed alpha-preserving or HDR conversion path is not part of
    this worker's Topaz/MP4 final delivery. Fail before GPU admission rather
    than present damaged motion graphics or incorrect highlights as valid.
    """
    if info.get("has_alpha") is True or pixel_format_has_alpha(info.get("pix_fmt")):
        raise ValueError(
            "Source contains an alpha-capable pixel format; the current Topaz/MP4 "
            "delivery is opaque and cannot preserve transparency. Export an "
            "explicitly composited opaque source or use an alpha-preserving workflow."
        )
    if str(info.get("color_transfer") or "").strip().lower() in {
        "smpte2084", "arib-std-b67",
    }:
        raise ValueError(
            "HDR PQ/HLG source requires qualified color-managed HDR delivery or "
            "tone mapping; the current Topaz/MP4 output must not silently alter "
            "HDR highlights or color. Provide an explicitly graded SDR source."
        )
    if str(info.get("field_order") or "").strip().lower() not in {
        "", "unknown", "progressive",
    }:
        raise ValueError(
            "An interlaced source needs an explicitly qualified deinterlace "
            "step before enhancement. The current Topaz/MP4 path must not "
            "turn interlace fields into permanent combing artifacts."
        )


def delivery_variants_for_source(
    source: Path,
    info: dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
    plan: dict[str, Any] | None = None,
    selected_source: Path | None = None,
    source_aliases: list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    plan = dict(plan or {})
    protected_source_keys = {_path_key(source)}
    if selected_source is not None:
        protected_source_keys.add(_path_key(selected_source))
    protected_source_keys.update(_path_key(value) for value in source_aliases or [])
    if not plan:
        _target_w, _target_h, frame_w, frame_h = target_dimensions(info, args.preset_id, args.layout)
        return [
            {
                "profile_id": "native_master",
                "output_path": str(final_output_path(source, out_dir)),
                "width": frame_w,
                "height": frame_h,
                "layout": args.layout,
                "crf": int(args.crf),
                "x264_preset": str(args.x264_preset),
                "watermark": args.watermark == "on",
                "watermark_strength": float(args.watermark_strength),
                "namespace": str(args.watermark_namespace or ""),
                "watermark_identifier": str(args.watermark_identifier or ""),
                "max_fps": final_fps_cap(info, args.preset_id),
            }
        ]

    planned_source = str(plan.get("source_path") or "").strip()
    if planned_source:
        planned_key = _path_key(planned_source)
        if planned_key not in protected_source_keys:
            raise ValueError(f"Delivery plan source does not match worker input: {planned_source} != {source}")

    variants: list[dict[str, Any]] = []
    outputs: set[str] = set()
    default_namespace = str(plan.get("brand_namespace") or args.watermark_namespace or "")
    working_geometry = (
        resolve_delivery_working_geometry(info, args.preset_id, args.layout, plan)
        if plan.get("geometry") is not None else None
    )
    _target_w, _target_h, enhanced_width, enhanced_height = target_dimensions_for_delivery(
        info,
        args.preset_id,
        args.layout,
        plan,
    )
    for index, raw in enumerate(plan.get("variants") or [], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Delivery variant {index} must be an object.")
        profile_id = str(raw.get("profile_id") or f"variant_{index}").strip()
        output_text = str(raw.get("output_path") or "").strip()
        if not output_text:
            raise ValueError(f"Delivery variant {profile_id} has no output_path.")
        output = Path(output_text).expanduser().resolve()
        output_key = _path_key(output)
        if output_key in protected_source_keys:
            raise ValueError(f"Delivery variant {profile_id} would overwrite a source or render-source alias.")
        if output_key in outputs:
            raise ValueError(f"Duplicate delivery output path: {output}")
        outputs.add(output_key)
        raw_width = int(raw.get("width") or 0)
        raw_height = int(raw.get("height") or 0)
        geometry_policy = str(raw.get("geometry_policy") or "")
        layout = str(raw.get("layout") or "fit_into_frame")
        source_geometry: dict[str, Any] | None = None
        custom_geometry: dict[str, Any] | None = None
        if raw_width == 0 and raw_height == 0:
            if geometry_policy == "source_display":
                if layout != "fit_into_frame":
                    raise ValueError(f"source_display geometry for {profile_id} requires fit_into_frame layout.")
                source_geometry = source_display_geometry(info)
                width = int(source_geometry["output_width"])
                height = int(source_geometry["output_height"])
            elif geometry_policy == "custom_final":
                custom_geometry = dict(raw.get("geometry") or {})
                source_geometry = resolve_custom_final_geometry(info, custom_geometry)
                width = int(source_geometry["output_width"])
                height = int(source_geometry["output_height"])
                layout = str(source_geometry["layout"])
            else:
                # Legacy zero-sized profiles (including native_master) retain
                # their verified enhanced working geometry.
                width = even(enhanced_width)
                height = even(enhanced_height)
        else:
            if geometry_policy:
                raise ValueError(f"Geometry policy {geometry_policy!r} requires unresolved source dimensions.")
            width = even(raw_width or int(info.get("width") or 0))
            height = even(raw_height or int(info.get("height") or 0))
        if layout not in SUPPORTED_LAYOUTS:
            raise ValueError(f"Unsupported delivery layout for {profile_id}: {layout}")
        variant = {
            "profile_id": profile_id,
            "profile_ids": [str(value) for value in raw.get("profile_ids") or [profile_id]],
            "output_path": str(output),
            "width": width,
            "height": height,
            "layout": layout,
            "crf": int(raw.get("crf") if raw.get("crf") is not None else args.crf),
            "x264_preset": str(raw.get("x264_preset") or args.x264_preset),
            "watermark": bool(raw.get("watermark", args.watermark == "on")),
            "watermark_strength": float(raw.get("watermark_strength") or args.watermark_strength),
            "namespace": str(raw.get("namespace") or default_namespace),
            "watermark_identifier": str(raw.get("watermark_identifier") or args.watermark_identifier or ""),
            "max_fps": int(raw.get("max_fps") or final_fps_cap(info, args.preset_id) or 0) or None,
        }
        if "frame_rate" in raw:
            variant["frame_rate"] = normalize_frame_rate(raw["frame_rate"])
            variant["frame_rate_reason"] = str(raw.get("frame_rate_reason") or "requested")
        if geometry_policy:
            variant["geometry_policy"] = geometry_policy
        if custom_geometry is not None:
            variant["geometry"] = custom_geometry
        if source_geometry is not None:
            variant["source_geometry"] = source_geometry
        if working_geometry is not None:
            variant["working_geometry"] = working_geometry
        variants.append(variant)
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    source_width = even(int(info.get("width") or 0))
    source_height = even(int(info.get("height") or 0))
    for variant in variants:
        width = int(variant["width"])
        height = int(variant["height"])
        same_aspect = bool(
            source_width
            and source_height
            and width
            and height
            and source_width * height == source_height * width
        )
        effective_layout = "source_aspect_exact" if same_aspect else str(variant["layout"])
        signature = (
            width,
            height,
            effective_layout,
            int(variant["crf"]),
            str(variant["x264_preset"]),
            bool(variant["watermark"]),
            float(variant["watermark_strength"]),
            str(variant["namespace"]),
            str(variant["watermark_identifier"]),
            variant["max_fps"],
            str(variant.get("geometry_policy") or ""),
            json.dumps(variant.get("geometry") or {}, sort_keys=True, separators=(",", ":")),
            json.dumps(variant.get("source_geometry") or {}, sort_keys=True, separators=(",", ":")),
            json.dumps(variant.get("working_geometry") or {}, sort_keys=True, separators=(",", ":")),
            json.dumps(variant.get("frame_rate") or {}, sort_keys=True, separators=(",", ":")),
        )
        if signature not in groups:
            variant["profile_ids"] = list(dict.fromkeys(variant["profile_ids"]))
            variant["planned_output_paths"] = [str(variant["output_path"])]
            groups[signature] = variant
            continue
        keeper = groups[signature]
        keeper["profile_ids"] = list(dict.fromkeys([*keeper["profile_ids"], *variant["profile_ids"]]))
        planned_paths = [*keeper["planned_output_paths"], str(variant["output_path"])]
        keeper["planned_output_paths"] = list(dict.fromkeys(planned_paths))
        keeper["runtime_deduplicated"] = True
    return list(groups.values())


def delivery_profile_output_map(variants: list[dict[str, Any]]) -> dict[str, str]:
    """Map every requested delivery profile to its surviving physical file."""

    result: dict[str, str] = {}
    for variant in variants:
        output = str(variant["output_path"])
        for profile_id in variant.get("profile_ids") or [variant.get("profile_id")]:
            if profile_id:
                result[str(profile_id)] = output
    return result


def _delivery_variant_resume_signature(variant: dict[str, Any]) -> dict[str, Any]:
    profile_id = str(variant.get("profile_id") or "")
    signature = {
        "profile_id": profile_id,
        "profile_ids": [str(value) for value in variant.get("profile_ids") or [profile_id]],
        "width": int(variant.get("width") or 0),
        "height": int(variant.get("height") or 0),
        "layout": str(variant.get("layout") or ""),
        "crf": int(variant.get("crf") or 0),
        "x264_preset": str(variant.get("x264_preset") or ""),
        "watermark": bool(variant.get("watermark")),
        "watermark_strength": float(variant.get("watermark_strength") or 0.0),
        "namespace": str(variant.get("namespace") or ""),
        "watermark_identifier": str(variant.get("watermark_identifier") or ""),
        "max_fps": int(variant.get("max_fps")) if variant.get("max_fps") is not None else None,
    }
    if variant.get("geometry_policy"):
        signature["geometry_policy"] = str(variant.get("geometry_policy"))
    if variant.get("source_geometry"):
        signature["source_geometry"] = dict(variant.get("source_geometry") or {})
    if variant.get("geometry"):
        signature["geometry"] = dict(variant.get("geometry") or {})
    if variant.get("working_geometry"):
        signature["working_geometry"] = dict(variant.get("working_geometry") or {})
    if "frame_rate" in variant:
        signature["frame_rate"] = normalize_frame_rate(variant["frame_rate"])
    return signature


def resume_source_aliases(
    resume_manifest: dict[str, Any],
    *,
    source: Path,
    selected_source: Path,
) -> list[str]:
    """Recover a failed job's raw/render-final aliases only for identical source bytes."""

    if not resume_manifest:
        return []
    current_identity = _content_identity(sampled_file_identity(source))
    current_keys = {_path_key(source), _path_key(selected_source)}
    for row in resume_manifest.get("resolved_delivery_sources") or []:
        if not isinstance(row, dict):
            continue
        values = [
            str(value)
            for value in (row.get("source_path"), row.get("selected_source_path"))
            if str(value or "").strip()
        ]
        if not current_keys.intersection({_path_key(value) for value in values}):
            continue
        stored_identity = row.get("source_identity") if isinstance(row.get("source_identity"), dict) else {}
        if _content_identity(stored_identity) == current_identity:
            return values
    return []


def restore_resume_delivery_variants(
    variants: list[dict[str, Any]],
    *,
    source: Path,
    selected_source: Path,
    resume_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Restore prior output paths only when source bytes and delivery settings still match."""

    if not resume_manifest:
        return variants, ""
    source_identity = _content_identity(sampled_file_identity(source))
    source_keys = {_path_key(source), _path_key(selected_source)}
    for row in resume_manifest.get("resolved_delivery_sources") or []:
        if not isinstance(row, dict):
            continue
        prior_keys = {
            _path_key(value)
            for value in (row.get("source_path"), row.get("selected_source_path"))
            if str(value or "").strip()
        }
        if not source_keys.intersection(prior_keys):
            continue
        stored_identity = row.get("source_identity") if isinstance(row.get("source_identity"), dict) else {}
        if _content_identity(stored_identity) != source_identity:
            continue
        prior_variants = row.get("variants") if isinstance(row.get("variants"), list) else []
        if len(prior_variants) != len(variants):
            continue
        if [
            _delivery_variant_resume_signature(item)
            for item in prior_variants
            if isinstance(item, dict)
        ] != [_delivery_variant_resume_signature(item) for item in variants]:
            continue
        restored: list[dict[str, Any]] = []
        output_keys: set[str] = set()
        protected_source_keys = source_keys.union(prior_keys)
        for current, prior in zip(variants, prior_variants, strict=True):
            output = Path(str(prior.get("output_path") or "")).expanduser().resolve()
            if not str(prior.get("output_path") or "").strip():
                raise ValueError("Resume manifest contains a delivery variant without an output path.")
            output_key = _path_key(output)
            if output_key in protected_source_keys:
                raise ValueError("Resume manifest delivery path would overwrite a source or render-source alias.")
            if output_key in output_keys:
                raise ValueError(f"Resume manifest contains duplicate delivery output path: {output}")
            output_keys.add(output_key)
            restored_variant = dict(current)
            restored_variant["output_path"] = str(output)
            restored_variant["planned_output_paths"] = [
                str(Path(value).expanduser().resolve())
                for value in prior.get("planned_output_paths") or [str(output)]
            ]
            restored.append(restored_variant)
        return restored, "validated_retry_manifest"
    return variants, ""


def route_unmatched_delivery_output(
    variant: dict[str, Any],
    *,
    validated_retry_manifest_restored: bool,
) -> tuple[dict[str, Any], str]:
    """Protect an existing invalid retry output by selecting a new physical path."""

    output = Path(str(variant.get("output_path") or "")).expanduser().resolve()
    if not output.exists():
        return dict(variant), ""
    if not validated_retry_manifest_restored:
        raise FileExistsError(
            "Delivery output already exists without a matching failed-job completion "
            f"receipt; source was not changed: {output}"
        )
    rerouted = collision_safe_path(output)
    if rerouted == output or rerouted.exists():
        raise RuntimeError(
            f"Could not allocate a collision-safe retry delivery path for {output}"
        )
    routed = dict(variant)
    routed["output_path"] = str(rerouted)
    routed["retry_collision_source_path"] = str(output)
    return routed, "collision_safe_invalid_retry_delivery"


def build_delivery_completion_expectation(
    *,
    variant: dict[str, Any],
    working_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    working_fingerprint, working_contract = _validated_checkpoint_expectation(working_checkpoint)
    output = Path(str(variant.get("output_path") or "")).expanduser().resolve()
    contract = {
        "schema": DELIVERY_COMPLETION_SCHEMA,
        "working_checkpoint_fingerprint": working_fingerprint,
        **({"enhancement_recipe": dict(working_contract["enhancement_recipe"])}
           if "enhancement_recipe" in working_contract else {}),
        **({"adaptive_fingerprint": working_contract["adaptive_fingerprint"]}
           if "adaptive_fingerprint" in working_contract else {}),
        **({"frame_rate": dict(working_contract["frame_rate"])} if "frame_rate" in working_contract else {}),
        "variant": {
            **_delivery_variant_resume_signature(variant),
            "output_path": str(output),
            "planned_output_paths": [
                str(Path(value).expanduser().resolve())
                for value in variant.get("planned_output_paths") or [str(output)]
            ],
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"fingerprint": fingerprint, "contract": contract}


def build_reusable_delivery_completion_expectation(
    *,
    variant: dict[str, Any],
    retained_working_checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Bind delivery reuse only to an intermediate checkpoint retained on disk."""

    if retained_working_checkpoint is None:
        return None
    return build_delivery_completion_expectation(
        variant=variant,
        working_checkpoint=retained_working_checkpoint,
    )


def select_reusable_working_checkpoint(
    *,
    interpolate: bool,
    upscale_cpu_recovery_used: bool,
    interpolation_cpu_recovery_used: bool,
    retained_upscale_checkpoint: dict[str, Any] | None,
    retained_interpolation_checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep CPU-recovered chains from being advertised as GPU-reusable authority."""

    if upscale_cpu_recovery_used or (interpolate and interpolation_cpu_recovery_used):
        return None
    return retained_interpolation_checkpoint if interpolate else retained_upscale_checkpoint


def _validated_delivery_completion_expectation(
    expected: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    contract = expected.get("contract") if isinstance(expected.get("contract"), dict) else {}
    fingerprint = str(expected.get("fingerprint") or "")
    if contract.get("schema") != DELIVERY_COMPLETION_SCHEMA:
        raise ValueError("Delivery completion expectation has an unsupported schema.")
    calculated = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not fingerprint or fingerprint != calculated:
        raise ValueError("Delivery completion expectation fingerprint is invalid.")
    return fingerprint, contract


def build_delivery_completion_row(
    *,
    output_path: Path,
    expected: dict[str, Any],
    probe: dict[str, Any],
    watermark_result: dict[str, Any] | None,
) -> dict[str, Any]:
    fingerprint, contract = _validated_delivery_completion_expectation(expected)
    output = output_path.expanduser().resolve()
    if _path_key(output) != _path_key(contract["variant"]["output_path"]):
        raise ValueError("Delivery completion output does not match its expected path.")
    return {
        "schema": DELIVERY_COMPLETION_SCHEMA,
        "fingerprint": fingerprint,
        "contract": contract,
        "output": sampled_file_identity(output),
        "probe": dict(probe),
        "watermark_result": dict(watermark_result or {}),
    }


def resolve_completed_delivery_output(
    resume_manifest: dict[str, Any],
    *,
    expected: dict[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    fingerprint, contract = _validated_delivery_completion_expectation(expected)
    for row in resume_manifest.get("completed_delivery_outputs") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("fingerprint") or "") != fingerprint or row.get("contract") != contract:
            continue
        validated = validate_delivery_completion_row(row)
        if validated is not None:
            return Path(validated["output_path"]), row
    return None


def validate_delivery_completion_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Revalidate one stored final-delivery receipt against the current file."""

    if not isinstance(row, dict) or row.get("schema") != DELIVERY_COMPLETION_SCHEMA:
        return None
    expected = {
        "fingerprint": str(row.get("fingerprint") or ""),
        "contract": row.get("contract") if isinstance(row.get("contract"), dict) else {},
    }
    try:
        _fingerprint, contract = _validated_delivery_completion_expectation(expected)
        variant = contract.get("variant") if isinstance(contract.get("variant"), dict) else {}
        stored_output = row.get("output") if isinstance(row.get("output"), dict) else {}
        output = Path(str(stored_output.get("path") or "")).expanduser().resolve()
        if (
            not str(stored_output.get("path") or "").strip()
            or _path_key(output) != _path_key(variant.get("output_path") or "")
            or not output.is_file()
        ):
            return None
        observed = sampled_file_identity(output)
    except (OSError, TypeError, ValueError):
        return None
    if _path_key(observed["path"]) != _path_key(stored_output.get("path") or ""):
        return None
    if _content_identity(observed) != _content_identity(stored_output):
        return None
    if bool(variant.get("watermark")):
        try:
            validate_delivery_watermark_receipt(output, variant, row.get("watermark_result"))
        except (OSError, TypeError, ValueError, RuntimeError):
            return None
    return {
        "row": dict(row),
        "output_path": str(output),
        "output_identity": observed,
    }


def append_delivery_completion(manifest: dict[str, Any], row: dict[str, Any]) -> None:
    rows = [
        item
        for item in manifest.get("completed_delivery_outputs") or []
        if isinstance(item, dict)
        and str(item.get("fingerprint") or "") != str(row.get("fingerprint") or "")
    ]
    rows.append(row)
    manifest["completed_delivery_outputs"] = rows


def validated_completed_source_states(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return completed sources whose source, mapping, and every final output still match."""

    completion_candidates: dict[str, list[dict[str, Any]]] = {}
    for row in manifest.get("completed_delivery_outputs") or []:
        if not isinstance(row, dict):
            continue
        if manifest.get("frame_rate") is not None and (
            not isinstance(row.get("contract"), dict)
            or row["contract"].get("frame_rate") != manifest["frame_rate"]
        ):
            continue
        if manifest.get("enhancement_recipe") is not None and (
            not isinstance(row.get("contract"), dict)
            or row["contract"].get("enhancement_recipe") != manifest["enhancement_recipe"]
        ):
            continue
        validated = validate_delivery_completion_row(row)
        if validated is None:
            continue
        fingerprint = str(row.get("fingerprint") or "")
        completion_candidates.setdefault(fingerprint, []).append(validated)

    states: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    completed_rows = [
        row for row in manifest.get("completed_sources") or [] if isinstance(row, dict)
    ]
    for row in completed_rows:
        aliases = [
            str(value)
            for value in (row.get("selected_source_path"), row.get("source_path"))
            if str(value or "").strip()
        ]
        stored_source = (
            row.get("source_identity") if isinstance(row.get("source_identity"), dict) else {}
        )
        source_candidates = [
            str(value)
            for value in (
                row.get("source_path"),
                row.get("selected_source_path"),
                stored_source.get("path"),
            )
            if str(value or "").strip()
        ]
        matching_sources = list(
            dict.fromkeys(
                value
                for value in source_candidates
                if Path(value).expanduser().is_file()
                and sampled_file_content_matches(value, stored_source)
            )
        )
        matched_source = matching_sources[0] if matching_sources else ""
        expected_count = int(row.get("delivery_output_count") or 0)
        fingerprints = [
            str(value)
            for value in row.get("delivery_completion_fingerprints") or []
            if str(value or "").strip()
        ]
        stored_outputs = [
            value for value in row.get("delivery_outputs") or [] if isinstance(value, dict)
        ]
        reason = ""
        if row.get("schema") != SOURCE_COMPLETION_SCHEMA:
            reason = "source_completion_schema_missing_or_unsupported"
        elif not matched_source:
            reason = (
                "stored_source_identity_missing"
                if not stored_source
                else "completed_source_content_changed_or_missing"
            )
        elif (
            expected_count <= 0
            or len(fingerprints) != expected_count
            or len(set(fingerprints)) != expected_count
            or len(stored_outputs) != expected_count
        ):
            reason = "completed_source_output_evidence_incomplete"

        completion_states: list[dict[str, Any]] = []
        output_paths: list[str] = []
        if not reason:
            for fingerprint in fingerprints:
                matches = completion_candidates.get(fingerprint) or []
                if len(matches) != 1:
                    reason = "completed_delivery_receipt_missing_invalid_or_ambiguous"
                    break
                completion_states.append(matches[0])
                output_paths.append(str(matches[0]["output_path"]))

        if not reason:
            stored_by_path = {
                _path_key(value.get("path") or ""): value
                for value in stored_outputs
                if str(value.get("path") or "").strip()
            }
            if len(stored_by_path) != expected_count:
                reason = "completed_source_output_identity_paths_invalid"
            else:
                completion_path_keys = [
                    _path_key(completion["output_path"])
                    for completion in completion_states
                ]
                if (
                    len(set(completion_path_keys)) != expected_count
                    or set(completion_path_keys) != set(stored_by_path)
                ):
                    reason = "completed_source_output_identity_paths_not_exact"
                else:
                    for completion in completion_states:
                        key = _path_key(completion["output_path"])
                        stored_output = stored_by_path.get(key)
                        if stored_output is None or _content_identity(
                            completion["output_identity"]
                        ) != _content_identity(stored_output):
                            reason = "completed_source_output_identity_mismatch"
                            break

        alias_keys = {_path_key(value) for value in aliases}
        if matched_source:
            alias_keys.add(_path_key(matched_source))
        resolved_variants: list[dict[str, Any]] = []
        profile_paths: dict[str, str] = {}
        if not reason:
            for completion in completion_states:
                contract = (
                    completion["row"].get("contract")
                    if isinstance(completion["row"].get("contract"), dict)
                    else {}
                )
                variant = (
                    contract.get("variant")
                    if isinstance(contract.get("variant"), dict)
                    else {}
                )
                try:
                    signature = _delivery_variant_resume_signature(variant)
                except (TypeError, ValueError):
                    reason = "completed_delivery_variant_contract_invalid"
                    break
                profile_ids = list(
                    dict.fromkeys(
                        str(value)
                        for value in signature.get("profile_ids") or []
                        if str(value).strip()
                    )
                )
                if (
                    not signature.get("profile_id")
                    or not profile_ids
                    or _path_key(variant.get("output_path") or "")
                    != _path_key(completion["output_path"])
                ):
                    reason = "completed_delivery_variant_contract_invalid"
                    break
                for profile_id in profile_ids:
                    prior = profile_paths.get(profile_id)
                    if prior is not None and _path_key(prior) != _path_key(
                        completion["output_path"]
                    ):
                        reason = "completed_delivery_profile_mapping_conflict"
                        break
                    profile_paths[profile_id] = str(completion["output_path"])
                if reason:
                    break
                resolved_variants.append(dict(variant))

        resolved_source: dict[str, Any] | None = None
        if not reason:
            if len(resolved_variants) != expected_count:
                reason = "completed_delivery_variant_count_mismatch"
            else:
                resolved_source = {
                    "selected_source_path": str(
                        row.get("selected_source_path") or matched_source
                    ),
                    "source_path": str(row.get("source_path") or matched_source),
                    "source_identity": dict(stored_source),
                    "physical_output_count": len(resolved_variants),
                    "profile_output_map": dict(profile_paths),
                    "variants": resolved_variants,
                }

        if reason:
            stale.append({"aliases": aliases, "reason": reason})
            continue
        current_source_identity = sampled_file_identity(matched_source)
        canonical_completed_source = {
            **dict(row),
            "selected_source_path": str(
                row.get("selected_source_path") or matched_source
            ),
            "source_path": str(Path(matched_source).expanduser().resolve()),
            "source_identity": current_source_identity,
            "delivery_outputs": [
                dict(value["output_identity"]) for value in completion_states
            ],
        }
        resolved_source["source_path"] = canonical_completed_source["source_path"]
        resolved_source["source_identity"] = current_source_identity
        states.append(
            {
                "state_key": _path_key(matched_source)
                + "|"
                + ",".join(sorted(fingerprints)),
                "aliases": aliases,
                "matched_source_key": _path_key(matched_source),
                "matching_source_keys": sorted(
                    {_path_key(value) for value in matching_sources}
                ),
                "source_keys": sorted(alias_keys),
                "completed_source": canonical_completed_source,
                "resolved_source": resolved_source,
                "completion_rows": [dict(value["row"]) for value in completion_states],
                "output_paths": output_paths,
            }
        )

    source_claims: dict[str, list[int]] = {}
    output_claims: dict[str, list[int]] = {}
    for index, state in enumerate(states):
        for key in state.get("source_keys") or []:
            source_claims.setdefault(str(key), []).append(index)
        for output in state.get("output_paths") or []:
            output_claims.setdefault(_path_key(output), []).append(index)
    conflicted = {
        index
        for owners in [*source_claims.values(), *output_claims.values()]
        if len(owners) > 1
        for index in owners
    }
    if conflicted:
        for index in sorted(conflicted):
            stale.append(
                {
                    "aliases": list(states[index].get("aliases") or []),
                    "reason": "duplicate_or_overlapping_completed_source_state",
                }
            )
        states = [
            state for index, state in enumerate(states) if index not in conflicted
        ]

    return {
        "states": states,
        "stale_or_unproven_completed_sources": stale,
        "completed_source_row_count": len(completed_rows),
        "validated_current_completed_source_count": len(states),
    }


def require_carried_completed_sources_current(
    manifest: dict[str, Any],
    *,
    expected_state_keys: set[str],
) -> None:
    """Fail if any source/output carried into a retry changed before completion."""

    if not expected_state_keys:
        return
    validation = validated_completed_source_states(manifest)
    current = {
        str(state.get("state_key") or "")
        for state in validation["states"]
        if str(state.get("state_key") or "").strip()
    }
    missing = sorted(expected_state_keys.difference(current))
    if missing:
        raise RuntimeError(
            "A source or final output carried from the failed job changed during retry "
            f"reconciliation ({len(missing)} completion state(s)); rerun the affected source."
        )


def build_completed_source_carry_forward(
    manifest: dict[str, Any],
    *,
    scheduled_files: list[Path],
) -> dict[str, Any]:
    """Build a complete retry receipt projection while leaving invalid rows pending."""

    scheduled_keys = {_path_key(path) for path in scheduled_files}
    validation = validated_completed_source_states(manifest)
    states = [
        state
        for state in validation["states"]
        if str(state.get("matched_source_key") or "") in scheduled_keys
    ]
    state_keys = {str(state["state_key"]) for state in states}
    matched_source_keys = {
        str(state["matched_source_key"])
        for state in states
        if str(state.get("matched_source_key") or "").strip()
    }
    completion_by_fingerprint = {
        str(row.get("fingerprint") or ""): dict(row)
        for state in states
        for row in state.get("completion_rows") or []
        if str(row.get("fingerprint") or "").strip()
    }
    completion_rows = list(completion_by_fingerprint.values())
    return {
        "states": states,
        "state_keys": state_keys,
        "pending_files": [
            path
            for path in scheduled_files
            if _path_key(path) not in matched_source_keys
        ],
        "completed_sources": [dict(state["completed_source"]) for state in states],
        "resolved_sources": [dict(state["resolved_source"]) for state in states],
        "completion_rows": completion_rows,
        "final_outputs": [
            str(path)
            for state in states
            for path in state.get("output_paths") or []
        ],
        "watermark_results": [
            dict(row["watermark_result"])
            for row in completion_rows
            if isinstance(row.get("watermark_result"), dict)
            and row.get("watermark_result")
        ],
        "validation": validation,
    }


def update_single_source_delivery_summary(
    manifest: dict[str, Any],
    *,
    total_source_count: int,
    profile_output_map: dict[str, str],
    physical_output_count: int,
) -> None:
    """Expose the convenience profile map only when the whole receipt has one source."""

    if int(total_source_count) == 1:
        manifest["resolved_delivery_profile_output_map"] = dict(profile_output_map)
        manifest["resolved_delivery_physical_output_count"] = int(
            physical_output_count
        )
        return
    manifest["resolved_delivery_profile_output_map"] = {}
    manifest["resolved_delivery_physical_output_count"] = 0


def target_dimensions(info: dict[str, Any], preset_id: str, layout_id: str) -> tuple[int, int, int, int]:
    src_w = int(info["width"])
    src_h = int(info["height"])
    model = PRESET_MODEL.get(preset_id, PRESET_MODEL["auto"])

    if preset_id in {"auto", "4k_60fps", "ss_4k"}:
        effective_long_edge = max(max(src_w, src_h), min(3840, max(src_w, src_h) * 4))
        if layout_id == "keep_source_aspect":
            target_w, target_h = keep_aspect_target(src_w, src_h, effective_long_edge)
        else:
            upscale_w, upscale_h = keep_aspect_target(src_w, src_h, effective_long_edge)
            target_w, target_h = fit_within(src_w, src_h, upscale_w, upscale_h)
        return target_w, target_h, target_w, target_h

    long_edge = int(model.get("long_edge") or 0)
    if long_edge <= 0:
        target_w, target_h = even(src_w), even(src_h)
    else:
        target_long = max(max(src_w, src_h), long_edge)
        target_w, target_h = keep_aspect_target(src_w, src_h, target_long)
    return target_w, target_h, target_w, target_h


def target_dimensions_for_delivery(
    info: dict[str, Any],
    preset_id: str,
    layout_id: str,
    plan: dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    """Resolve the shared working frame from the canonical geometry policy."""
    geometry = resolve_delivery_working_geometry(info, preset_id, layout_id, plan)
    return (
        int(geometry["target_width"]),
        int(geometry["target_height"]),
        int(geometry["frame_width"]),
        int(geometry["frame_height"]),
    )


def should_interpolate(info: dict[str, Any], preset_id: str, writer: ProgressWriter, topaz: dict[str, Any], *, frame_rate: dict[str, Any] | None = None) -> bool:
    if frame_rate is not None:
        normalized = normalize_frame_rate(frame_rate)
        if normalized["mode"] == "preserve_source":
            source_max_presentation_rate(info)
            field_order = str(info.get("field_order") or "").lower()
            if field_order not in {"", "progressive", "unknown"}:
                raise RuntimeError(f"Source-PTS preservation is not qualified for interlaced input ({field_order}).")
            writer.log("Source-PTS preservation selected; Topaz frame interpolation is disabled for this output.")
            return False
        target = Fraction(normalized["fps"])
        if info.get("is_vfr"):
            raise RuntimeError("Explicit frame_rate currently requires constant-rate source evidence; VFR normalization is not qualified.")
        if source_frame_rate(info) >= target:
            return False
        field_order = str(info.get("field_order") or "").lower()
        return bool(topaz.get("tvai_fi")) and field_order in {"", "progressive", "unknown"}
    target_fps = PRESET_MODEL.get(preset_id, {}).get("fps")
    if not target_fps:
        return False
    fps = float(info.get("fps") or 0)
    field_order = str(info.get("field_order") or "").lower()
    if fps >= target_fps - 0.1:
        writer.log(f"Interpolation skipped: source FPS {fps} is already >= {target_fps}")
        return False
    if field_order and field_order not in {"progressive", "unknown"}:
        writer.log(f"Interpolation skipped: source appears interlaced ({field_order}) and no deinterlace step is configured")
        return False
    if not topaz.get("tvai_fi"):
        writer.log("Interpolation skipped: tvai_fi filter missing")
        return False
    return True


def final_fps_cap(info: dict[str, Any], preset_id: str) -> int | None:
    target_fps = PRESET_MODEL.get(preset_id, {}).get("fps")
    if target_fps:
        return min(int(target_fps), MAX_OUTPUT_FPS)
    fps = float(info.get("fps") or 0)
    return MAX_OUTPUT_FPS if fps > MAX_OUTPUT_FPS + 0.1 else None


def require_topaz_interpolation_route(info: dict[str, Any], preset_id: str, interpolate: bool, *, frame_rate: dict[str, Any] | None = None) -> None:
    if frame_rate is not None:
        normalized = normalize_frame_rate(frame_rate)
        if normalized["mode"] == "preserve_source":
            return
        target = Fraction(normalized["fps"])
        if source_frame_rate(info) < target and not interpolate:
            raise RuntimeError(f"Requested {target} FPS requires real Topaz interpolation; refusing plain-FFmpeg duplicated frames.")
        return
    target_fps = PRESET_MODEL.get(preset_id, {}).get("fps")
    if not target_fps:
        return
    source_fps = float(info.get("fps") or 0)
    if source_fps <= 0:
        raise RuntimeError("Cannot determine source FPS for the 60 FPS Topaz workflow.")
    if source_fps < float(target_fps) - 0.1 and not interpolate:
        raise RuntimeError(
            f"{preset_id} requires Topaz frame interpolation from {source_fps:.3f} to {target_fps} FPS, "
            "but tvai_fi is unavailable or unsafe. Refusing to create duplicated frames with plain FFmpeg."
        )


def validate_interpolated_fps(info: dict[str, Any], target_fps: int | str = 60) -> None:
    if isinstance(target_fps, str):
        if info.get("frame_rate_evidence") == "presentation-timestamps" and info.get("video_pts_uniform") is not True:
            raise RuntimeError("Topaz interpolation returned non-uniform presentation timestamps.")
        validate_requested_frame_rate(info, {"mode": "cfr", "fps": target_fps})
        return
    observed = float(info.get("fps") or 0)
    if abs(observed - float(target_fps)) > 0.25:
        raise RuntimeError(f"Topaz interpolation returned {observed:.3f} FPS; expected approximately {target_fps} FPS.")


def validate_working_dimensions(
    info: dict[str, Any],
    *,
    expected_width: int,
    expected_height: int,
    phase: str,
) -> None:
    observed = (int(info.get("width") or 0), int(info.get("height") or 0))
    expected = (even(expected_width), even(expected_height))
    if observed != expected:
        raise RuntimeError(f"{phase} dimensions are {observed}; expected the exact working frame {expected}.")


def validate_final_delivery(
    info: dict[str, Any],
    *,
    expected_width: int,
    expected_height: int,
    preset_id: str,
    frame_rate: dict[str, Any] | None = None,
    source_info: dict[str, Any] | None = None,
    source_timing_max_fps: int | None = None,
) -> None:
    observed_size = (int(info.get("width") or 0), int(info.get("height") or 0))
    expected_size = (even(expected_width), even(expected_height))
    if observed_size != expected_size:
        raise RuntimeError(f"Final delivery dimensions are {observed_size}; expected {expected_size}.")
    if frame_rate is not None:
        normalized = normalize_frame_rate(frame_rate)
        if normalized["mode"] == "preserve_source":
            validate_requested_frame_rate(
                info, normalized, source_info=source_info, max_fps=source_timing_max_fps,
            )
            return
        if info.get("frame_rate_evidence") == "presentation-timestamps" and info.get("video_pts_uniform") is not True:
            raise RuntimeError("Final delivery has non-uniform presentation timestamps for explicit CFR.")
        validate_requested_frame_rate(info, normalized)
        if source_info is not None:
            validate_cadence_extent(source_info, info, frame_rate)
        return
    target_fps = PRESET_MODEL.get(preset_id, {}).get("fps")
    if target_fps:
        observed_fps = float(info.get("fps") or 0)
        if abs(observed_fps - float(target_fps)) > 0.25:
            raise RuntimeError(f"Final delivery is {observed_fps:.3f} FPS; expected approximately {target_fps} FPS.")


def final_output_path(source: Path, out_dir: Path) -> Path:
    return collision_safe_path(out_dir / source.with_suffix(".mp4").name)


def render_temp_base(path: Path) -> Path | None:
    if path.suffix.lower() not in {".m4v", *RENDER_AUDIO_SIDECAR_EXTS}:
        return None
    match = RENDER_TEMP_STEM_RE.match(path.stem)
    if not match:
        return None
    return path.with_name(match.group("base"))


def render_final_candidates(path: Path) -> list[Path]:
    base = render_temp_base(path)
    if not base:
        return []
    candidates = [base.with_suffix(".mp4"), base.with_suffix(".mov")]
    if path.suffix.lower() in RENDER_AUDIO_SIDECAR_EXTS:
        candidates.append(path.with_suffix(".m4v"))
    return candidates


def normalize_input_paths(paths: list[Path], writer: ProgressWriter) -> list[Path]:
    normalized: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        selected = path
        if path.suffix.lower() in RENDER_AUDIO_SIDECAR_EXTS:
            for candidate in render_final_candidates(path):
                if candidate.exists():
                    selected = candidate
                    writer.log(f"Audio render sidecar selected; using companion file: {candidate}")
                    break
        elif not path.exists():
            for candidate in render_final_candidates(path):
                if candidate.exists():
                    selected = candidate
                    writer.log(f"Selected render temp no longer exists; using finished output: {candidate}")
                    break

        key = str(selected.resolve()) if selected.exists() else str(selected)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(selected)

    return normalized


def file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def safe_probe_ready(path: Path, ffprobe: str) -> dict[str, Any] | None:
    try:
        info = probe_video(path, ffprobe)
    except Exception:
        return None
    if not info.get("width") or not info.get("height") or float(info.get("duration") or 0) <= 0:
        return None
    return info


def stable_probe(path: Path, ffprobe: str) -> dict[str, Any] | None:
    first_sig = file_signature(path)
    first_probe = safe_probe_ready(path, ffprobe)
    time.sleep(RENDER_POLL_SECONDS)
    second_sig = file_signature(path)
    second_probe = safe_probe_ready(path, ffprobe)
    if first_sig == second_sig and first_probe and second_probe:
        return second_probe
    return None


def find_stable_render_final(source: Path, ffprobe: str, writer: ProgressWriter) -> tuple[Path, dict[str, Any]] | None:
    for candidate in render_final_candidates(source):
        if candidate == source or not candidate.exists():
            continue
        info = stable_probe(candidate, ffprobe)
        if info:
            writer.log(f"Render temp resolved to finished output: {source} -> {candidate}")
            return candidate, info
    return None


def wait_for_render_final_candidate(source: Path, ffprobe: str, writer: ProgressWriter) -> tuple[Path, dict[str, Any]] | None:
    if not render_temp_base(source):
        return None
    deadline = time.monotonic() + RENDER_FINAL_MUX_GRACE_SECONDS
    while time.monotonic() < deadline:
        if writer.cancel_requested:
            writer.cancelled("Cancelled while waiting for final mux")
            raise JobCancelled("Cancelled")
        finished = find_stable_render_final(source, ffprobe, writer)
        if finished:
            return finished
        time.sleep(RENDER_POLL_SECONDS)
    writer.log(f"No finished mux appeared within {RENDER_FINAL_MUX_GRACE_SECONDS:.0f}s; using selected render stream: {source}")
    return None


def wait_for_render_ready(source: Path, ffprobe: str, writer: ProgressWriter) -> tuple[Path, dict[str, Any]]:
    finished = find_stable_render_final(source, ffprobe, writer)
    if finished:
        return finished

    first_sig = file_signature(source)
    first_probe = safe_probe_ready(source, ffprobe)
    time.sleep(RENDER_POLL_SECONDS)
    second_sig = file_signature(source)
    second_probe = safe_probe_ready(source, ffprobe)

    if first_sig == second_sig and first_probe and second_probe:
        writer.log(f"Input already stable: {source}")
        if render_temp_base(source):
            writer.update(phase="wait_for_render", message=f"Waiting for final mux: {source.name}")
            finished = wait_for_render_final_candidate(source, ffprobe, writer)
            if finished:
                return finished
        return source, second_probe

    writer.log(f"Input appears to still be rendering; waiting for stable file: {source}")
    writer.update(
        phase="wait_for_render",
        message=f"Waiting for render to finish: {source.name}",
        metrics={"stable_seconds_required": RENDER_STABLE_SECONDS, "post_render_delay_seconds": POST_RENDER_DELAY_SECONDS},
    )

    deadline = time.monotonic() + RENDER_WAIT_TIMEOUT_SECONDS
    last_sig: tuple[int, int] | None = None
    stable_since: float | None = None
    last_update = 0.0

    while time.monotonic() < deadline:
        if writer.cancel_requested:
            writer.cancelled("Cancelled while waiting for render")
            raise JobCancelled("Cancelled")

        finished = find_stable_render_final(source, ffprobe, writer)
        if finished:
            writer.update(phase="wait_for_render", message=f"Render done; launching in {POST_RENDER_DELAY_SECONDS:.0f}s")
            time.sleep(POST_RENDER_DELAY_SECONDS)
            return finished

        sig = file_signature(source)
        info = safe_probe_ready(source, ffprobe)
        now = time.monotonic()

        if sig is not None and sig == last_sig and info:
            if stable_since is None:
                stable_since = now
            stable_for = now - stable_since
            if stable_for >= RENDER_STABLE_SECONDS:
                writer.log(f"Render stable for {stable_for:.1f}s: {source}")
                writer.update(phase="wait_for_render", message=f"Render done; launching in {POST_RENDER_DELAY_SECONDS:.0f}s")
                time.sleep(POST_RENDER_DELAY_SECONDS)
                finished = wait_for_render_final_candidate(source, ffprobe, writer)
                if finished:
                    return finished
                final_info = safe_probe_ready(source, ffprobe)
                if final_info:
                    return source, final_info
                stable_since = None
        else:
            stable_since = None

        last_sig = sig
        if now - last_update >= 2.0:
            size = sig[0] if sig else 0
            writer.update(
                phase="wait_for_render",
                message=f"Waiting for render to finish: {source.name}",
                metrics={"current_size_bytes": size, "probe_ready": bool(info)},
            )
            last_update = now

        time.sleep(RENDER_POLL_SECONDS)

    raise TimeoutError(f"Timed out waiting for render to finish: {source}")


def parse_ffmpeg_time_seconds(text: str) -> float | None:
    match = re.search(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def _best_effort_topaz_log(writer: Any, message: str) -> None:
    """Never allow optional diagnostics to replace the native process outcome."""
    try:
        writer.log(message)
    except OSError:
        pass


def process_creation_flags(priority: str) -> int:
    if os.name != "nt":
        return 0
    priority = priority.lower().replace("-", "_")
    if priority == "below_normal":
        return getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    if priority == "idle":
        return getattr(subprocess, "IDLE_PRIORITY_CLASS", 0)
    if priority == "normal":
        return getattr(subprocess, "NORMAL_PRIORITY_CLASS", 0)
    return 0


def _run_cancellable_subprocess(
    argv: list[str],
    *,
    writer: Any,
    timeout: float | None = None,
    cwd: str | None = None,
    priority: str = "normal",
    on_line: Callable[[str], None] | None = None,
    startup_timeout: float | None = None,
    stall_timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Observe the owned child even while it emits no output; reap on every exit.

    One bounded pipe reader belongs to this invocation only. It is not a job
    executor. Capture is limited to a diagnostic tail; existing logging owns
    persistent output. A silent model must never suppress Cancel/deadlines.
    """
    if getattr(writer, "cancel_requested", False):
        raise JobCancelled("Cancelled before starting process")
    started = time.monotonic()
    for budget in (timeout, startup_timeout, stall_timeout):
        if (budget is not None
                and (type(budget) not in (int, float)
                     or not math.isfinite(budget) or budget <= 0)):
            raise ValueError("Process and output-frame watchdog budgets must be positive finite seconds.")
    emitted_frames = 0
    pending_frames = 0
    first_frame_seconds = None
    last_frame_time = started
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1, cwd=cwd,
        creationflags=process_creation_flags(priority),
    )
    assert proc.stdout is not None
    lines: queue.Queue[str] = queue.Queue(maxsize=128)
    stop = threading.Event()
    reader_done = threading.Event()
    reader_errors: list[Exception] = []

    def read_pipe() -> None:
        try:
            while not stop.is_set():
                line = proc.stdout.readline(8192)
                if not line:
                    break
                while not stop.is_set():
                    try:
                        lines.put(line, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            reader_errors.append(exc)
        finally:
            reader_done.set()

    reader = threading.Thread(target=read_pipe, name=f"spiffy-owned-pipe-{proc.pid}", daemon=True)
    reader.start()
    tail: deque[str] = deque()
    tail_length = 0
    try:
        while True:
            if getattr(writer, "cancel_requested", False):
                raise JobCancelled("Cancelled")
            if timeout is not None and time.monotonic() - started >= timeout:
                raise subprocess.TimeoutExpired(argv, timeout, stderr="".join(tail))
            now = time.monotonic()
            if startup_timeout is not None and emitted_frames == 0 and now - started >= startup_timeout:
                raise subprocess.TimeoutExpired(argv, startup_timeout, stderr="No output frame before startup deadline.\n" + "".join(tail))
            if stall_timeout is not None and emitted_frames > 0 and now - last_frame_time >= stall_timeout:
                raise subprocess.TimeoutExpired(argv, stall_timeout, stderr="Output frames stopped advancing.\n" + "".join(tail))
            try:
                line = lines.get(timeout=0.05)
            except queue.Empty:
                if reader_done.is_set() and proc.poll() is not None:
                    break
                continue
            tail.append(line)
            tail_length += len(line)
            while tail_length > 65536 and len(tail) > 1:
                tail_length -= len(tail.popleft())
            if startup_timeout is not None or stall_timeout is not None:
                stripped = line.strip()
                match = re.fullmatch(r"frame=(\d+)", stripped)
                if match:
                    pending_frames = int(match.group(1))
                elif stripped in {"progress=continue", "progress=end"} and pending_frames > emitted_frames:
                    emitted_frames = pending_frames
                    last_frame_time = time.monotonic()
                    if first_frame_seconds is None:
                        first_frame_seconds = last_frame_time - started
            if on_line is not None:
                on_line(line)
        if reader_errors:
            raise RuntimeError("Owned process output reader failed") from reader_errors[0]
        return_code = proc.wait(timeout=1)
        if return_code == 0 and startup_timeout is not None and emitted_frames == 0:
            raise RuntimeError("Guarded render exited without producing an output frame.")
        result = subprocess.CompletedProcess(argv, return_code, "", "".join(tail))
        result.first_frame_seconds = first_frame_seconds
        result.emitted_frames = emitted_frames
        return result
    finally:
        stop.set()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        reader.join(timeout=2)
        if not reader.is_alive():
            proc.stdout.close()
        else:
            raise RuntimeError(f"Owned output reader did not retire for process {proc.pid}; no success accepted.")


def run_process(
    argv: list[str],
    writer: ProgressWriter,
    phase: str,
    timeout: float | None = None,
    duration_s: float | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
    priority: str = "normal",
    startup_timeout: float | None = None,
    stall_timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    writer.command(argv)
    writer.update(phase=phase, message=Path(argv[0]).name)
    last_progress_update = 0.0
    process_started = time.monotonic()
    diagnostic_log_error: str | None = None

    def record_log_error(exc: OSError) -> None:
        nonlocal diagnostic_log_error
        if diagnostic_log_error is None:
            diagnostic_log_error = type(exc).__name__

    def log_diagnostic(message: str) -> None:
        try:
            writer.log(message)
        except OSError as exc:
            # A diagnostic write cannot turn a native failure, timeout or
            # cancellation into a misleading "disk full" process result.
            record_log_error(exc)

    result = None
    process_status = "failed"
    try:
        # FFmpeg's -progress pipe emits many short key/value lines per frame
        # batch. Opening/closing stderr.txt for EACH line causes avoidable
        # filesystem churn on long exports. Keep one owned buffered append
        # handle per subprocess, flushing periodically for live diagnostics,
        # and always on exit/cancel/failure via the context manager.
        log_stream = None
        last_log_flush = None
        try:
            try:
                log_stream = writer.stderr_path.open("a", encoding="utf-8", buffering=65536)
            except OSError as exc:
                record_log_error(exc)

            def on_line(line: str) -> None:
                nonlocal last_progress_update, last_log_flush, log_stream
                now = time.monotonic()
                if log_stream is not None:
                    try:
                        log_stream.write(line)
                        # Flush the first native diagnostic immediately; later
                        # data flushes periodically and on progress boundaries.
                        if (last_log_flush is None or now - last_log_flush >= 0.5
                                or line.strip() in {"progress=continue", "progress=end"}):
                            log_stream.flush()
                            last_log_flush = now
                    except OSError as exc:
                        record_log_error(exc)
                        try:
                            log_stream.close()
                        except OSError as close_exc:
                            record_log_error(close_exc)
                        log_stream = None
                if duration_s and percent_start is not None and percent_end is not None:
                    seen_s = parse_ffmpeg_time_seconds(line)
                    if seen_s is None or now - last_progress_update < 0.8:
                        return
                    last_progress_update = now
                    ratio = max(0.0, min(seen_s / duration_s, 1.0))
                    writer.update(
                        phase=phase,
                        message=f"{Path(argv[0]).name} {seen_s:.1f}s / {duration_s:.1f}s",
                        percent=round(percent_start + (percent_end - percent_start) * ratio, 1),
                    )

            result = _run_cancellable_subprocess(
                argv, writer=writer, timeout=timeout, priority=priority, on_line=on_line,
                **({"startup_timeout": startup_timeout, "stall_timeout": stall_timeout}
                   if startup_timeout is not None or stall_timeout is not None else {}),
            )
        finally:
            if log_stream is not None:
                try:
                    log_stream.close()
                except OSError as exc:
                    record_log_error(exc)
        process_status = "completed" if result.returncode == 0 else "failed"
    except JobCancelled:
        process_status = "cancelled"
        try:
            writer.cancelled("Cancelled")
        except OSError as exc:
            record_log_error(exc)
        raise
    finally:
        native_failure_class = _topaz_native_failure_class(phase, getattr(result, "returncode", None))
        measurement = {"phase": phase, "seconds": round(time.monotonic() - process_started, 4),
                       "first_frame_seconds": getattr(result, "first_frame_seconds", None),
                       "emitted_frames": getattr(result, "emitted_frames", None),
                       "status": process_status, "exit_code": getattr(result, "returncode", None),
                       "native_failure_class": native_failure_class,
                       "native_same_command_retry_allowed": False if native_failure_class == "windows_access_violation" else None}
        if not hasattr(writer, "process_timings"):
            writer.process_timings = []
        if diagnostic_log_error is not None:
            measurement["diagnostic_log_error"] = diagnostic_log_error
        writer.process_timings.append(measurement)
        log_diagnostic("Process timing: " + json.dumps(measurement))
        if diagnostic_log_error is not None:
            measurement["diagnostic_log_error"] = diagnostic_log_error
    log_diagnostic(f"{phase} exited {result.returncode}")
    if result.returncode != 0:
        failure_class = _topaz_native_failure_class(phase, result.returncode)
        if failure_class == "windows_access_violation":
            log_diagnostic(
                f"{phase} encountered native Windows access violation 0xC0000005. "
                "Do not retry the same crashed configuration automatically. "
                "The original native log and operating-system fault record remain available for diagnosis."
            )
        original_stderr = (result.stderr or "").strip()
        diagnostic = _safe_topaz_failure_excerpt(original_stderr) if phase.startswith(
            ("topaz_", "adaptive_enhancement_pilot", "adaptive_interpolation")
        ) else original_stderr[-4000:]
        classification = f" [{failure_class}; unsafe to auto-retry]" if failure_class else ""
        raise RuntimeError(f"{phase} failed with exit {result.returncode}:{classification} {diagnostic}")
    return result


def _topaz_native_failure_class(phase: str, exit_code: int | None) -> str:
    """Classify Windows native crashes without treating access violations as model warm-up.

    0xC0000005 can indicate an invalid memory access in native dependencies;
    it does NOT by itself prove VRAM exhaustion. The known optional warm-up
    retry is reserved for a different, specifically admitted first-use code.
    """
    native_phase = phase.startswith(("topaz_", "adaptive_enhancement_pilot",
                                    "adaptive_interpolation"))
    if native_phase and type(exit_code) is int and exit_code in (3221225477, -1073741819):
        return "windows_access_violation"
    return ""


def _safe_topaz_failure_excerpt(stderr: str) -> str:
    """Keep error messages useful without forwarding local license/account details."""
    lines: list[str] = []
    skipped_sensitive = False
    for line in stderr.splitlines():
        text = line.casefold()
        if any(marker in text for marker in (
                "license", "[aie-rlm]", "roaming", "issued to", "owned - cached")):
            skipped_sensitive = True
            continue
        lines.append(line)
    if skipped_sensitive:
        # Some preflight probes have no persistent native stderr file.
        # Do not promise that an original log exists when it may not.
        lines.append("[Topaz licensing details omitted]")
    return "\n".join(lines)[-4000:]


def run_topaz_process_with_first_use_retry(
    argv: list[str],
    writer: ProgressWriter,
    phase: str,
    *,
    duration_s: float | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
    priority: str = "normal",
    startup_timeout: float | None = None,
    stall_timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Retry one Topaz native fail-fast after first-use TensorRT engine creation."""

    kwargs = {
        "duration_s": duration_s,
        "percent_start": percent_start,
        "percent_end": percent_end,
        "priority": priority,
        **({"startup_timeout": startup_timeout, "stall_timeout": stall_timeout} if startup_timeout is not None or stall_timeout is not None else {}),
    }
    try:
        return run_process(argv, writer, phase, **kwargs)
    except RuntimeError as exc:
        if not is_topaz_native_failfast_error(exc, phase):
            raise
        _best_effort_topaz_log(writer,
            f"{phase} hit Topaz native fail-fast during model execution; retrying the exact command once "
            "to reuse any TensorRT engine created by the failed first invocation."
        )
        return run_process(argv, writer, phase, **kwargs)


def is_topaz_native_failfast_error(exc: BaseException, phase: str) -> bool:
    message = str(exc)
    return any(
        f"{phase} failed with exit {exit_code}:" in message
        for exit_code in TOPAZ_NATIVE_FIRST_USE_RETRY_EXIT_CODES
    )


def topaz_cpu_recovery_allowed(target_w: int, target_h: int) -> bool:
    return even(target_w) * even(target_h) <= TOPAZ_CPU_RECOVERY_MAX_PIXELS


def validate_topaz_filter_model(
    topaz_ffmpeg: str,
    filter_name: str,
    model: str,
    writer: ProgressWriter,
    *,
    recipe: EnhancementRecipe | None = None,
    target_fps: int | str = 60,
) -> tuple[bool, str]:
    if filter_name not in {"tvai_up", "tvai_fi"}:
        raise ValueError("Unsupported Topaz preflight filter.")
    # Model IDs are inserted inside an FFmpeg filter option string (not a
    # shell command). Disallow option separators, newlines, and filter-graph
    # syntax before they can override download=0 or device selection.
    if type(model) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", model) is None:
        raise ValueError("Topaz preflight model identifier is invalid.")
    if recipe is not None and (filter_name != "tvai_up" or recipe.up_model != model):
        raise ValueError("enhancement recipe/model preflight mismatch.")
    if filter_name == "tvai_up":
        params = recipe.params if recipe is not None else ""
        vf = f"tvai_up=model={model}:scale=0:w=320:h=180{params}:download=0:device=-2:vram=0.5:instances=0"
        timeout_seconds = MODEL_UPSCALE_SMOKE_TIMEOUT_SECONDS
    else:
        vf = f"tvai_fi=model={model}:slowmo=1:rdt=0.01:fps={frame_rate_token(target_fps)}:download=0:device=-2:vram=0.5:instances=0"
        timeout_seconds = MODEL_INTERPOLATION_SMOKE_TIMEOUT_SECONDS
    argv = [
        topaz_ffmpeg,
        "-hide_banner",
        "-y",
        "-nostdin",
        "-nostats",
        "-progress", "pipe:1",
        "-f", "lavfi",
        "-i", "testsrc2=s=160x90:d=0.1:r=30" if filter_name == "tvai_fi" else "testsrc2=s=160x90:d=1:r=30",
        "-frames:v", "2" if filter_name == "tvai_fi" else "1",
        "-vf", vf,
        "-f", "null",
        "-",
    ]
    native_retry_used = False
    for attempt in range(2):
        if getattr(writer, "cancel_requested", False):
            raise JobCancelled("Cancelled before Topaz model preflight.")
        writer.command(argv)
        if callable(getattr(writer, "update", None)):
            writer.update(phase="preflight_model", message=f"Checking local {model} ({timeout_seconds}s startup limit)")
        try:
            result = _run_cancellable_subprocess(
                argv,
                writer=writer,
                timeout=timeout_seconds,
                # The overall timeout already bounds startup. Enable the
                # existing FFmpeg frame-progress parser without extra decoding
                # or a second model invocation. A zero-exit process that
                # never outputs a frame must not qualify a production lane.
                stall_timeout=timeout_seconds,
                cwd=str(Path(topaz_ffmpeg).parent),
            )
        except subprocess.TimeoutExpired as exc:
            diagnostic = _safe_topaz_failure_excerpt(str(exc.stderr or ""))
            if diagnostic:
                _best_effort_topaz_log(writer, "Bounded model startup timed out; diagnostic tail: " + diagnostic[-2000:])
            return False, (
                f"{filter_name} model smoke test timed out after {timeout_seconds}s for {model}; "
                "refusing to launch a production Topaz "
                "render until the model/GPU preflight completes successfully"
            )
        text = (result.stderr or "") + (result.stdout or "")
        if "model not found" in text.casefold():
            return False, f"{filter_name} model assets are missing or unavailable: {model}"
        if result.returncode == 0:
            expected_frames = 2 if filter_name == "tvai_fi" else 1
            emitted_frames = getattr(result, "emitted_frames", None)
            if type(emitted_frames) is not int or emitted_frames < expected_frames:
                observed = emitted_frames if type(emitted_frames) is int else "unverified"
                return False, (
                    f"{filter_name} model smoke test exited without the required output frames "
                    f"({observed}/{expected_frames}) for {model}; refusing to launch a production Topaz render."
                )
            return True, ""
        exit_code = int(result.returncode)
        if attempt == 0 and exit_code in TOPAZ_NATIVE_FIRST_USE_RETRY_EXIT_CODES:
            native_retry_used = True
            _best_effort_topaz_log(writer,
                f"{filter_name} model preflight hit Topaz native fail-fast for {model}; retrying the "
                "exact preflight command once to reuse any TensorRT engine created by the failed "
                "first invocation."
            )
            continue
        retry_suffix = " after one exact retry" if native_retry_used else ""
        return False, (
            f"{filter_name} model smoke test failed for {model} with exit {exit_code}{retry_suffix}; "
            "refusing to launch a known-failing production Topaz render: "
            f"{_safe_topaz_failure_excerpt(text)[-1000:]}"
        )
    raise AssertionError("unreachable Topaz model preflight state")


def _adaptive_shot_sequence_to_prores(
    source: Path, dest: Path, topaz_ffmpeg: str, info: dict[str, Any],
    preset_id: str, target_w: int, target_h: int, writer: ProgressWriter,
    percent_start: float, percent_end: float, recipe: EnhancementRecipe,
    adaptive_plan: dict[str, Any],
) -> bool:
    """Run admitted contextual windows through this worker and remux once.

    All unit frames are independently counted/timestamp-checked before the
    stream-copy assembly assigns their original exact frame grid. Original
    audio is mapped once, not repeatedly trimmed and joined at scene cuts.
    """
    from adaptive_enhancement import render_unit_plans

    units = render_unit_plans(adaptive_plan)
    rate = Fraction(adaptive_plan["source_fps"])
    ffmpeg, ffprobe = standard_ffmpeg(), standard_ffprobe()
    model_dir = str(os.environ.get("TVAI_MODEL_DIR") or "")
    parts: list[Path] = []
    checkpoints: list[Path] = []
    cpu_recovered = False
    for index, unit in enumerate(units):
        if writer.cancel_requested:
            raise JobCancelled("Adaptive shot render cancelled.")
        window = unit["execution_window"]
        frame_count = window["end_frame"] - window["start_frame"]
        part = Path(writer.temp_dir) / f"{dest.stem}_shot_{index:03d}_{unit['fingerprint'][:12]}.mov"
        checkpoint = topaz_checkpoint_path(part)
        active = unit["route"] != "source_preserved"
        assets = topaz_model_asset_identity(model_dir, recipe.up_model) if active else {}
        expectation_args = dict(stage="adaptive_shot", input_path=source,
            topaz_ffmpeg=topaz_ffmpeg if active else ffmpeg, preset_id=preset_id,
            model=recipe.up_model if active else "conventional_resize",
            model_params=unit["effective_params"] if active else "lanczos",
            target_width=target_w, target_height=target_h,
            target_fps=str(rate), adaptive_fingerprint=unit["fingerprint"])
        expected = build_topaz_checkpoint_fingerprint(**expectation_args, model_asset_identity=assets)
        reusable = resolve_topaz_checkpoint(checkpoint, expected=expected) if active and assets.get("resume_safe") else None
        writer.update(phase="adaptive_shots", message=f"Processing shot group {index + 1}/{len(units)}",
            metrics={"adaptive_shot_group": index + 1, "adaptive_shot_groups": len(units),
                     "adaptive_shot_route": unit["route"], "adaptive_shot_frames": frame_count})
        recovered = False
        if reusable is None:
            recovered = topaz_upscale_to_prores(source, part, topaz_ffmpeg, info, preset_id,
                target_w, target_h, writer,
                percent_start + (percent_end - percent_start) * index / len(units),
                percent_start + (percent_end - percent_start) * (index + 1) / len(units),
                recipe=recipe, adaptive_plan=unit)
        else:
            writer.log(f"Reusing exact verified adaptive shot checkpoint: {part}")
        cpu_recovered |= recovered
        probe = probe_video(part, ffprobe, inspect_timestamps=True)
        validate_working_dimensions(probe, expected_width=target_w, expected_height=target_h,
                                    phase=f"Adaptive shot group {index + 1}")
        if (int(probe.get("video_pts_count") or 0) != frame_count
                or Fraction(str(probe.get("video_pts_start") or "-1")) != 0
                or probe.get("video_pts_uniform") is not True
                or source_frame_rate(probe) != rate
                or probe.get("audio_stream_count")):
            raise RuntimeError("Adaptive shot output changed frame custody, cadence, origin or audio isolation.")
        if active and not recovered:
            after = build_topaz_checkpoint_fingerprint(**expectation_args,
                model_asset_identity=topaz_model_asset_identity(model_dir, recipe.up_model))
            stable = validate_topaz_stage_identity_stable(expected, after, stage="Adaptive shot")
            retain_verified_topaz_checkpoint(checkpoint, output_path=part, expected=stable)
        parts.append(part)
        checkpoints.append(checkpoint)
    concat_file = Path(writer.temp_dir) / f"{dest.stem}_shots.ffconcat"
    lines = ["ffconcat version 1.0"]
    for part in parts:
        text = part.resolve().as_posix()
        if "\n" in text or "\r" in text:
            raise ValueError("Invalid line break in adaptive job path.")
        lines.append("file '" + text.replace("'", "'\\''") + "'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf8")
    # ProRes is intra-frame. Packet-grid assignment after per-unit validation
    # avoids concat-demuxer microsecond rounding without re-encoding pictures.
    packet_clock = (f"setts=pts=N*{rate.denominator}:dts=N*{rate.denominator}:"
                    f"duration={rate.denominator}:time_base=1/{rate.numerator}")
    argv = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "copy", "-bsf:v", packet_clock, "-video_track_timescale", str(rate.numerator),
            "-c:a", "pcm_s24le", "-movflags", "+write_colr", str(dest)]
    run_process(argv, writer, "adaptive_shot_assembly", duration_s=float(info.get("duration") or 0) or None)
    assembled = probe_video(dest, ffprobe, inspect_timestamps=True)
    validate_source_timing_preserved(info, assembled, phase="Adaptive shot assembly")
    validate_intermediate_audio(info, assembled, phase="Adaptive shot assembly", require_sample_extent=True)
    for path in [*parts, *checkpoints, concat_file]:
        path.unlink(missing_ok=True)
    return cpu_recovered


def _discard_new_failed_topaz_temp(dest: Path, writer: ProgressWriter, *,
                                   existed_before: bool) -> bool:
    """Discard only an output this failed attempt newly created in its own temp folder.

    Never remove a preexisting checkpoint candidate, source, symlink or
    output elsewhere. The original failure remains the caller's exception;
    cleanup is best-effort and must not hide the native crash/cancellation.
    """
    if existed_before:
        return False
    try:
        path = Path(dest)
        owned = Path(writer.temp_dir)
        if (path.suffix.lower() != ".mov" or path.is_symlink()
                or not path.is_file() or owned.is_symlink()
                or path.parent.is_symlink() or not owned.is_dir()):
            return False
        if not path.parent.samefile(owned):
            return False
        path.unlink()
    except (OSError, AttributeError, TypeError) as exc:
        try:
            writer.log(f"Failed native temp output could not be safely cleaned: {type(exc).__name__}.")
        except Exception:
            pass  # Never replace the real native crash with a logging failure.
        return False
    try:
        writer.log("Discarded incomplete newly created native movie from this job's temp folder.")
    except Exception:
        pass  # An unavailable/full log must not hide the original render error.
    return True


def topaz_upscale_to_prores(
    source: Path,
    dest: Path,
    topaz_ffmpeg: str,
    info: dict[str, Any],
    preset_id: str,
    target_w: int,
    target_h: int,
    writer: ProgressWriter,
    percent_start: float = 0.0,
    percent_end: float = 0.0,
    *,
    recipe: EnhancementRecipe | None = None,
    preserve_source_timing: bool = False,
    adaptive_plan: dict[str, Any] | None = None,
    working_geometry: dict[str, Any] | None = None,
    audition: bool = False,
    audition_timeout: float = 60.0,
) -> bool:
    resolved = recipe if recipe is not None else delivery_enhancement_recipe({}, preset_id)
    if resolved.preset_id != preset_id:
        raise ValueError("enhancement recipe/preset render mismatch.")
    if type(audition) is not bool:
        raise ValueError("audition must be a boolean.")
    if audition:
        window = (adaptive_plan or {}).get("execution_window") or {}
        if (adaptive_plan is None or adaptive_plan.get("route") == "shot_sequence"
                or type(window.get("start_frame")) is not int or type(window.get("end_frame")) is not int
                or not 1 <= window["end_frame"] - window["start_frame"] <= 8
                or type(audition_timeout) not in (int, float) or not math.isfinite(audition_timeout)
                or not 1 <= audition_timeout <= 60):
            raise ValueError("Enhancement auditions require a bounded source window and deadline.")
    if adaptive_plan is not None and adaptive_plan.get("route") == "shot_sequence":
        return _adaptive_shot_sequence_to_prores(source, dest, topaz_ffmpeg, info, preset_id,
            target_w, target_h, writer, percent_start, percent_end, resolved, adaptive_plan)
    up_model = resolved.up_model
    params = resolved.params
    if type(preserve_source_timing) is not bool:
        raise ValueError("preserve_source_timing must be a boolean.")
    if working_geometry is not None and working_geometry.get("apply_input_layout") and adaptive_plan is not None:
        raise ValueError("Custom working geometry cannot be combined with the adaptive input-layout route.")
    track_timescale = None
    if preserve_source_timing:
        if info.get("frame_rate_evidence") != "presentation-timestamps":
            raise RuntimeError("Topaz source-timing preservation requires decoded presentation timestamp evidence.")
        try:
            time_base = Fraction(str(info.get("video_time_base") or "0"))
            pts_time_base = Fraction(str(info.get("video_pts_time_base") or "0"))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise RuntimeError("Topaz source-timing preservation requires an exact source video time base.") from exc
        if time_base <= 0 or pts_time_base != time_base or int(info.get("video_pts_count") or 0) < 2 or not info.get("video_pts_sha256"):
            raise RuntimeError("Topaz source-timing preservation requires complete source PTS evidence.")
        track_timescale = video_track_timescale_for_time_base(time_base)
        writer.log(
            "Topaz source-timing policy: passthrough with exact source PTS; "
            f"MOV track timescale={track_timescale} from time_base={time_base}."
        )

    def build_argv(device: str, *, download: int | None = None) -> list[str]:
        download_arg = "" if download is None else f":download={int(download)}"
        input_layout = ""
        if working_geometry is not None and working_geometry.get("apply_input_layout"):
            source_display = working_geometry.get("source_display")
            if not isinstance(source_display, dict):
                raise ValueError("Custom working geometry is missing its source-display proof.")
            display_width = int(source_display.get("display_width") or 0)
            display_height = int(source_display.get("display_height") or 0)
            if display_width < 2 or display_height < 2:
                raise ValueError("Custom working geometry has invalid source-display dimensions.")
            input_layout = (
                f"scale={display_width}:{display_height}:flags=lanczos,setsar=1,"
                + layout_filter("fit_into_frame", display_width, display_height, even(target_w), even(target_h))
                + ","
            )
        filt = (
            f"[0:v:0]{input_layout}tvai_up=model={up_model}:scale=0:w={even(target_w)}:h={even(target_h)}"
            f"{params}{download_arg}:device={device}:vram=0.85:instances=0[topaz];"
            f"[topaz]scale=w={even(target_w)}:h={even(target_h)}:flags=lanczos,setsar=1[vout]"
        )
        if adaptive_plan is not None:
            from adaptive_enhancement import enhancement_filter_graph, verify_artifacts
            verify_artifacts(adaptive_plan)
            filt = enhancement_filter_graph(adaptive_plan, up_model, device)
        unit_window = adaptive_plan.get("execution_window") if adaptive_plan is not None else None
        unit_rate = Fraction(adaptive_plan["source_fps"]) if unit_window else None
        return [
            topaz_ffmpeg,
            "-hide_banner",
            "-y",
            *(["-nostdin", "-nostats", "-progress", "pipe:1", "-stats_period", "0.5"] if adaptive_plan is not None else []),
            *(["-copyts"] if preserve_source_timing else []),
            "-autorotate",
            "-i", str(source),
            *(["-i", str(adaptive_plan["mask_path"])]
              if adaptive_plan is not None and adaptive_plan["route"] != "source_preserved" else []),
            "-filter_complex", filt,
            "-map", "[vout]",
            *(["-an"] if unit_window else ["-map", "0:a?"]),
            "-c:v", "prores_ks",
            "-profile:v", "3",
            "-pix_fmt", "yuv422p10le",
            "-c:a", "pcm_s24le",
            "-movflags", "+write_colr",
            *(["-fps_mode", "passthrough", "-video_track_timescale", str(unit_rate.numerator)] if unit_rate else
              ["-fps_mode", "passthrough", "-video_track_timescale", str(track_timescale)] if preserve_source_timing else []),
            str(dest),
        ]

    argv = build_argv("-2", download=0)
    output_existed_before = dest.exists() or dest.is_symlink()
    if audition:
        try:
            run_process(argv, writer, "adaptive_enhancement_pilot", timeout=audition_timeout,
                        startup_timeout=min(30.0, audition_timeout), stall_timeout=min(30.0, audition_timeout),
                        priority="below_normal")
        except BaseException:
            _discard_new_failed_topaz_temp(dest, writer, existed_before=output_existed_before)
            raise
        return False
    if adaptive_plan is not None and adaptive_plan["route"] == "source_preserved":
        argv[0] = standard_ffmpeg()
        writer.log("Adaptive enhancement: no admitted detail; preserving source via conventional resize, no enhancement-model execution.")
        try:
            run_process(argv, writer, "adaptive_source_preservation", duration_s=float(info.get("duration") or 0) or None)
        except BaseException:
            _discard_new_failed_topaz_temp(dest, writer, existed_before=output_existed_before)
            raise
        return False
    if bool(info.get("has_audio")):
        writer.log(
            "Topaz ProRes audio policy: mapping every source track to lossless 24-bit PCM "
            "in the first render pass."
        )
    run_kwargs = {
        "duration_s": float(info.get("duration") or 0) or None,
        "percent_start": percent_start,
        "percent_end": percent_end,
        **({"startup_timeout": 90.0, "stall_timeout": 120.0} if adaptive_plan is not None else {}),
    }
    try:
        run_topaz_process_with_first_use_retry(
            argv,
            writer,
            "topaz_upscale",
            **run_kwargs,
            priority="normal",
        )
        return False
    except RuntimeError as exc:
        if not is_topaz_native_failfast_error(exc, "topaz_upscale"):
            _discard_new_failed_topaz_temp(dest, writer, existed_before=output_existed_before)
            raise
        if not topaz_cpu_recovery_allowed(target_w, target_h):
            writer.log(
                "Topaz upscale still hit native fail-fast after the exact GPU retry; CPU recovery is "
                f"disabled above {TOPAZ_CPU_RECOVERY_MAX_PIXELS} working pixels."
            )
            _discard_new_failed_topaz_temp(dest, writer, existed_before=output_existed_before)
            raise
        writer.log(
            "Topaz upscale still hit native fail-fast after the exact GPU retry; running the same model "
            "and parameters once on the CPU recovery backend for this qualified small-frame lane."
        )
        try:
            run_process(
                build_argv("-1", download=0),
                writer,
                "topaz_upscale_cpu_recovery",
                **run_kwargs,
                priority="below_normal",
            )
        except BaseException:
            _discard_new_failed_topaz_temp(dest, writer, existed_before=output_existed_before)
            raise
        return True
    except BaseException:
        _discard_new_failed_topaz_temp(dest, writer, existed_before=output_existed_before)
        raise


def topaz_interpolate_to_prores(
    source: Path,
    dest: Path,
    topaz_ffmpeg: str,
    target_fps: int | str,
    writer: ProgressWriter,
    duration_s: float | None = None,
    percent_start: float = 0.0,
    percent_end: float = 0.0,
    recovery_width: int | None = None,
    recovery_height: int | None = None,
    source_info: dict[str, Any] | None = None,
    guarded_startup: bool = False,
    output_dimensions: tuple[int, int] | None = None,
    source_cut_frames: list[int] | None = None,
    caption_mask_plan: dict[str, Any] | None = None,
    interpolation_model: str = "apo-8",
) -> bool:
    if interpolation_model not in {"apo-8", "apf-2"}:
        raise ValueError("Unsupported local interpolation model; arbitrary filter input is not accepted.")
    if interpolation_model != "apo-8" and source_cut_frames is None:
        raise ValueError("Alternate interpolation model requires the protected adaptive motion path.")
    if caption_mask_plan is not None:
        from adaptive_enhancement import verify_artifacts
        verify_artifacts(caption_mask_plan)
        if (source_cut_frames is None or output_dimensions is None
                or int(caption_mask_plan["frame_count"]) != int((source_info or {}).get("video_frame_count") or 0)
                or Fraction(caption_mask_plan["source_fps"]) != source_frame_rate(source_info or {})):
            raise ValueError("Caption interpolation masks must match the source cadence/frame count and output geometry.")
    explicit_cadence = isinstance(target_fps, str) or output_dimensions is not None or source_cut_frames is not None
    boundary = interpolation_boundary_contract(source_info or {}, target_fps) if explicit_cadence else None
    resize_prefix = ""
    if output_dimensions is not None:
        if (not isinstance(output_dimensions, (tuple, list)) or len(output_dimensions) != 2
                or any(type(v) is not int or v < 16 or v % 2 for v in output_dimensions)):
            raise ValueError("Interpolation output dimensions must be positive even pixels.")
        width, height = output_dimensions
        src_w, src_h = int((source_info or {}).get("width") or 0), int((source_info or {}).get("height") or 0)
        if width > src_w or height > src_h or width * src_h != height * src_w:
            raise ValueError("Interpolation resizing must preserve exact aspect without enlargement.")
        if (width, height) != (src_w, src_h):
            resize_prefix = f"scale={width}:{height}:flags=lanczos,setsar=1,"
    intervals: list[tuple[int, int]] = []
    if source_cut_frames is not None:
        if not isinstance(source_cut_frames, list) or any(type(cut) is not int for cut in source_cut_frames):
            raise ValueError("Interpolation cuts must be exact source frame ordinals.")
        source_rate = source_frame_rate(source_info or {})
        requested_rate = Fraction(frame_rate_token(target_fps))
        source_count = int((source_info or {}).get("video_frame_count") or 0)
        for cut in sorted(set(source_cut_frames)):
            if not 0 < cut < source_count:
                raise ValueError("Interpolation cut falls outside source frames.")
            first = max(0, math.ceil(Fraction(cut - 1) * requested_rate / source_rate))
            last = math.ceil(Fraction(cut + 1) * requested_rate / source_rate) - 1
            if intervals and first <= intervals[-1][1] + 1:
                intervals[-1] = (intervals[-1][0], max(last, intervals[-1][1]))
            else:
                intervals.append((first, last))
        if len(intervals) > 512:
            raise ValueError("Interpolation cut-protection expression exceeds the bounded shot budget.")
    target_fps = frame_rate_token(target_fps)
    authored_motion = source_cut_frames is not None
    output_clock = Fraction(target_fps)
    duplicate_threshold = "0" if authored_motion else "0.01"
    def build_argv(device: str, *, download: int | None = None) -> list[str]:
        download_arg = "" if download is None else f":download={int(download)}"
        initial = f"[0:v:0]{resize_prefix}"
        prefix = ""
        if intervals or caption_mask_plan is not None:
            prefix = initial + "split=2[fi_source][cut_source];"
            initial = "[fi_source]"
        if boundary is not None:
            initial += "tpad=stop_mode=clone:stop=1,"
        filt = (prefix + initial + f"tvai_fi=model={interpolation_model}:slowmo=1:rdt={duplicate_threshold}:fps={target_fps}"
                + f"{download_arg}:device={device}:vram=0.85:instances=0")
        if boundary is not None:
            filt += f",trim=end_frame={boundary['trim_end_frame']}"
        if authored_motion:
            # Native Apollo can emit CFR samples one source-clock tick early.
            # Align already-generated frames to the requested exact grid BEFORE
            # framesync, or a protected cut selects the previous source hold.
            # This neither creates frames nor conceals missing model output.
            filt += f",settb=expr=1/{output_clock.numerator},setpts=N*{output_clock.denominator}"
        if caption_mask_plan is not None:
            from adaptive_enhancement import temporal_caption_graph
            enable = "+".join(f"between(n,{first},{last})" for first, last in intervals)
            filt += "[native];" + temporal_caption_graph(caption_mask_plan, target_fps,
                output_dimensions[0], output_dimensions[1], boundary["trim_end_frame"], enable)
        elif intervals:
            enable = "+".join(f"between(n,{first},{last})" for first, last in intervals)
            filt += ("[native];[native]format=gbrp16le[fi_rgb];"
                     f"[cut_source]fps=fps={target_fps}:round=up,format=gbrp16le[cut_hold];"
                     f"[fi_rgb][cut_hold]blend=all_expr=B:enable='{enable}',"
                     f"trim=end_frame={boundary['trim_end_frame']}[vout]")
        else:
            filt += "[vout]"
        return [
            topaz_ffmpeg,
            "-hide_banner",
            "-y",
            *(["-nostdin", "-nostats", "-progress", "pipe:1", "-stats_period", "0.5"] if guarded_startup else []),
            "-i", str(source),
            *(["-i", str(caption_mask_plan["mask_path"])] if caption_mask_plan is not None else []),
            "-filter_complex", filt,
            "-map", "[vout]",
            "-map", "0:a?",
            "-c:v", "prores_ks",
            "-profile:v", "3",
            "-pix_fmt", "yuv422p10le",
            "-c:a", "copy",
            "-movflags", "+write_colr",
            *(["-fps_mode", "passthrough"] if explicit_cadence else []),
            str(dest),
        ]

    argv = build_argv("-2", download=0)
    output_existed_before = dest.exists() or dest.is_symlink()
    run_kwargs = {
        "duration_s": duration_s,
        "percent_start": percent_start,
        "percent_end": percent_end,
        **({"startup_timeout": 120.0, "stall_timeout": 120.0} if guarded_startup else {}),
    }
    try:
        run_topaz_process_with_first_use_retry(
            argv,
            writer,
            "topaz_interpolate",
            **run_kwargs,
            priority="normal",
        )
        return False
    except RuntimeError as exc:
        if not is_topaz_native_failfast_error(exc, "topaz_interpolate"):
            _discard_new_failed_topaz_temp(dest, writer, existed_before=output_existed_before)
            raise
        if (
            recovery_width is None
            or recovery_height is None
            or interpolation_model != "apo-8"
            or not topaz_cpu_recovery_allowed(recovery_width, recovery_height)
        ):
            writer.log(
                "Topaz interpolation still hit native fail-fast after the exact GPU retry; bounded CPU "
                "recovery is unavailable for this working-frame size."
            )
            _discard_new_failed_topaz_temp(dest, writer, existed_before=output_existed_before)
            raise
        writer.log(
            "Topaz interpolation still hit native fail-fast after the exact GPU retry; running Apollo v8 "
            "once on the CPU recovery backend for this qualified small-frame lane."
        )
        try:
            run_process(
                build_argv("-1", download=0),
                writer,
                "topaz_interpolate_cpu_recovery",
                **run_kwargs,
                priority="below_normal",
            )
        except BaseException:
            _discard_new_failed_topaz_temp(dest, writer, existed_before=output_existed_before)
            raise
        return True
    except BaseException:
        _discard_new_failed_topaz_temp(dest, writer, existed_before=output_existed_before)
        raise


class _InterpolationCalibrationIntegrityError(RuntimeError):
    """An identity change is not an optional model-calibration failure."""


def select_adaptive_interpolation_model(
    source: Path, source_info: dict[str, Any], plan: dict[str, Any], target_fps: int | str,
    dimensions: tuple[int, int], topaz_ffmpeg: str, model_dir: str, writer: ProgressWriter,
) -> dict[str, Any]:
    """Admit a faster LOCAL model only after a bounded per-video comparison.

    Runs inside the existing GPU admission. Missing candidate assets never
    trigger a render-time download. Failed/slow/different pilots retain Apollo.
    Nothing is selected from a model name, a global benchmark or an LLM score.
    """
    from adaptive_enhancement import (
        choose_interpolation_model, compare_interpolation_candidates,
        interpolation_pilot_windows, verify_artifacts,
    )

    verify_artifacts(plan)
    target = frame_rate_token(target_fps)
    windows = interpolation_pilot_windows(plan, target, *dimensions)
    fallback = {"model": "apo-8", "reason": "pilot_not_cost_eligible", "schema": "spiffy_fi_choice_v1"}
    if not windows:
        return fallback
    assets = {model: topaz_model_asset_identity(model_dir, model) for model in ("apo-8", "apf-2")}
    if not all(value.get("resume_safe") for value in assets.values()):
        return {**fallback, "reason": "local_candidate_assets_unavailable_no_download"}
    source_identity = sampled_file_identity(source)
    source_identity.pop("path", None)
    contract = {"schema": "spiffy_fi_choice_v1", "adaptive_fingerprint": plan["fingerprint"],
                "source": source_identity, "target_rate": target, "dimensions": list(dimensions),
                "windows": [list(value) for value in windows],
                "models": {model: value["identity_sha256"] for model, value in assets.items()}}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache = writer.temp_dir / f"{source.stem}_fi_calibration.json"
    if cache.is_file() and cache.stat().st_size < 256 * 1024:
        try:
            saved = load_json(cache)
            if saved.get("fingerprint") == fingerprint and saved.get("contract") == contract:
                expected = choose_interpolation_model(saved["comparison"], saved["reference_seconds"], saved["candidate_seconds"])
                if all(saved.get(k) == expected[k] for k in expected):
                    return {**saved, "cache_hit": True}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    writer.update(phase="adaptive_motion_calibration", message="Comparing local interpolation on bounded motion samples")
    pilot = writer.temp_dir / f"{source.stem}_fi_pilot.mov"
    paths = {model: writer.temp_dir / f"{source.stem}_fi_pilot_{model}.mov" for model in assets}
    source_rate = source_frame_rate(source_info)
    pieces = []
    if len(windows) > 1:
        pieces.append(f"[0:v:0]split={len(windows)}" + "".join(f"[p{i}]" for i in range(len(windows))))
    for i, (start, end) in enumerate(windows):
        incoming = f"[p{i}]" if len(windows) > 1 else "[0:v:0]"
        pieces.append(incoming + f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,"
                      f"scale={dimensions[0]}:{dimensions[1]}:flags=lanczos,setsar=1[s{i}]")
    joined = "".join(f"[s{i}]" for i in range(len(windows)))
    pieces.append(joined + (f"concat=n={len(windows)}:v=1:a=0," if len(windows) > 1 else "")
                  + f"settb=expr=1/{source_rate.numerator},setpts=N*{source_rate.denominator}[pilot]")
    pilot_frames = sum(end - start for start, end in windows)
    receipt: dict[str, Any] = {**fallback, "fingerprint": fingerprint, "contract": contract, "cache_hit": False}
    try:
        run_process([standard_ffmpeg(), "-hide_banner", "-v", "error", "-nostdin", "-y", "-i", str(source),
                     "-filter_complex", ";".join(pieces), "-map", "[pilot]", "-an", "-c:v", "prores_ks",
                     "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-fps_mode", "passthrough",
                     "-video_track_timescale", str(source_rate.numerator), str(pilot)],
                    writer, "adaptive_motion_pilot", timeout=45)
        info = probe_video(pilot, standard_ffprobe(), inspect_timestamps=True)
        if int(info["video_frame_count"]) != pilot_frames or source_frame_rate(info) != source_rate:
            raise RuntimeError("Interpolation pilot did not preserve the selected source clock.")
        pilot_cuts = [windows[0][1] - windows[0][0]] if len(windows) > 1 else []
        timings = {}
        for model, destination in paths.items():
            if writer.cancel_requested:
                raise JobCancelled("Cancelled during interpolation calibration.")
            started = time.monotonic()
            topaz_interpolate_to_prores(pilot, destination, topaz_ffmpeg, str(target), writer,
                source_info=info, source_cut_frames=pilot_cuts, guarded_startup=True,
                interpolation_model=model)
            timings[model] = time.monotonic() - started
            output_info = probe_video(destination, standard_ffprobe(), inspect_timestamps=True)
            validate_final_delivery(output_info, expected_width=dimensions[0], expected_height=dimensions[1],
                preset_id="", frame_rate={"mode": "cfr", "fps": target}, source_info=info)
        comparison = compare_interpolation_candidates(paths["apo-8"], paths["apf-2"], writer)
        receipt.update(choose_interpolation_model(comparison, timings["apo-8"], timings["apf-2"]))
        if any(topaz_model_asset_identity(model_dir, model) != assets[model] for model in assets):
            raise _InterpolationCalibrationIntegrityError("Interpolation model assets changed during calibration.")
        now = sampled_file_identity(source)
        now.pop("path", None)
        if now != source_identity:
            raise _InterpolationCalibrationIntegrityError("Interpolation input changed during calibration.")
        atomic_write_json(cache, receipt)
        writer.log("Adaptive interpolation selection: " + json.dumps({k: receipt[k] for k in
            ("model", "reason", "reference_seconds", "candidate_seconds")}, sort_keys=True))
        return receipt
    except (JobCancelled, GpuLockCancelled, _InterpolationCalibrationIntegrityError):
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        # A candidate qualification failure is not permission to downgrade the
        # quality contract or abort otherwise-supported reference rendering.
        writer.log("Faster-model pilot unavailable; retaining Apollo reference: " + str(exc)[-600:])
        return {**fallback, "reason": "candidate_pilot_not_qualified", "fingerprint": fingerprint,
                "contract": contract}
    finally:
        for path in [pilot, *paths.values()]:
            if path.is_file() and not path.is_symlink():
                path.unlink(missing_ok=True)


def verify_or_remux(path: Path, ffmpeg: str, ffprobe: str, writer: ProgressWriter) -> Path:
    ok, info, reason = verify_video(path, ffprobe)
    if ok:
        writer.log(f"Verified intermediate: {path} {info}")
        return path
    fixed = path.with_name(path.stem + "_fixed" + path.suffix)
    writer.log(f"Intermediate verify failed ({reason}); attempting copy remux to {fixed}")
    argv = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        str(fixed),
    ]
    run_process(argv, writer, "remux_intermediate")
    ok, info, reason = verify_video(fixed, ffprobe)
    if not ok:
        raise RuntimeError(f"Intermediate failed verification after remux: {reason}")
    writer.log(f"Verified remuxed intermediate: {fixed} {info}")
    return fixed


def delivery_movie_timescale(
    source_info: dict[str, Any], *, frame_rate: dict[str, Any] | None = None,
    preserve_source_timing: bool = False,
) -> int:
    """Keep MP4 edit-list endpoints exact on the admitted video/audio clocks.

    The default 1000 Hz movie clock rounded AAC valid duration on short NTSC
    clips even when video PTS and the audio track time base were correct.
    This changes mux timing, not audio samples or the validation tolerance.
    """
    if preserve_source_timing:
        scale = video_track_timescale_for_time_base(source_info["video_time_base"])
    else:
        rate = Fraction(frame_rate["fps"]) if frame_rate else source_frame_rate(source_info)
        scale = rate.numerator
    for audio in source_info.get("audio_streams") or []:
        sample_rate = audio.get("sample_rate")
        if isinstance(sample_rate, bool):
            raise ValueError("Movie timescale requires an exact audio sample rate.")
        try:
            exact = Fraction(str(sample_rate))
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("Movie timescale requires an exact audio sample rate.") from exc
        if exact.denominator != 1 or exact <= 0:
            raise ValueError("Movie timescale requires a positive integer audio sample rate.")
        scale = math.lcm(scale, exact.numerator)
    if not 1 <= scale <= 2_147_483_647:
        raise ValueError("Exact combined audio/video movie timescale exceeds the MP4 integer range.")
    return scale


def final_mp4(
    source: Path,
    dest: Path,
    ffmpeg: str,
    src_info: dict[str, Any],
    layout_id: str,
    frame_w: int,
    frame_h: int,
    crf: int,
    preset: str,
    writer: ProgressWriter,
    max_fps: int | None = None,
    duration_s: float | None = None,
    percent_start: float = 0.0,
    percent_end: float = 0.0,
    *,
    frame_rate: dict[str, Any] | None = None,
    original_info: dict[str, Any] | None = None,
    preserve_source_timing: bool = False,
) -> None:
    if type(preserve_source_timing) is not bool:
        raise ValueError("preserve_source_timing must be a boolean.")
    timing_reference = original_info or src_info
    track_timescale = None
    if preserve_source_timing:
        if frame_rate is not None or max_fps is not None:
            raise ValueError("Source-timing preservation cannot be combined with a CFR request or FPS ceiling.")
        validate_source_timing_preserved(timing_reference, src_info, phase="Final H264 input")
        track_timescale = video_track_timescale_for_time_base(timing_reference["video_time_base"])
    if frame_rate is not None:
        frame_rate = normalize_frame_rate(frame_rate)
        if source_frame_rate(src_info) < Fraction(frame_rate["fps"]):
            raise RuntimeError("Explicit frame_rate requires prior real interpolation; final encoding cannot manufacture missing frames.")
    movie_timescale = delivery_movie_timescale(timing_reference, frame_rate=frame_rate,
        preserve_source_timing=preserve_source_timing) if frame_rate is not None or preserve_source_timing else None
    tmp = part_path(dest)
    if tmp.exists():
        tmp.unlink()
    vf = layout_filter(layout_id, int(src_info["width"]), int(src_info["height"]), even(frame_w), even(frame_h))
    if frame_rate is not None:
        vf = f"{vf},fps=fps={frame_rate_token(frame_rate['fps'])}:round=near"
    elif max_fps:
        vf = f"{vf},fps=fps={max_fps}:round=near"
    argv = [
        ffmpeg,
        "-hide_banner",
        "-y",
        *(["-copyts"] if preserve_source_timing else []),
        "-i", str(source),
        "-vf", vf,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        *(["-movie_timescale", str(movie_timescale)] if movie_timescale is not None else []),
        *(["-fps_mode", "passthrough"] if frame_rate is not None else []),
        *(["-fps_mode", "passthrough", "-video_track_timescale", str(track_timescale)]
          if preserve_source_timing else []),
        str(tmp),
    ]
    run_process(
        argv,
        writer,
        "final_h264",
        duration_s=duration_s,
        percent_start=percent_start,
        percent_end=percent_end,
        priority="below_normal",
    )
    if preserve_source_timing:
        candidate = probe_video(tmp, standard_ffprobe(), inspect_timestamps=True)
        validate_source_timing_preserved(timing_reference, candidate, phase="Final H264 output")
        validate_final_delivery(candidate, expected_width=frame_w, expected_height=frame_h, preset_id="")
        validate_intermediate_audio(
            timing_reference,
            candidate,
            phase="Source-preserve candidate before publication",
            require_sample_extent=True,
        )
    elif frame_rate is not None:
        candidate = probe_video(tmp, standard_ffprobe())
        validate_final_delivery(candidate, expected_width=frame_w, expected_height=frame_h,
            preset_id="", frame_rate=frame_rate, source_info=original_info or src_info)
        validate_intermediate_audio(
            original_info or src_info,
            candidate,
            phase="Explicit-cadence candidate before publication",
            require_sample_extent=True,
        )
    os.replace(tmp, dest)


def _watermark_api() -> tuple[Any, Any]:
    if str(SPIFFY_WORKSPACE) not in sys.path:
        sys.path.insert(0, str(SPIFFY_WORKSPACE))
    from workflow.video_provenance_watermark import doctor, watermark_final_encode

    return doctor, watermark_final_encode


def run_videoseal_preflight(writer: ProgressWriter) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "workflow.video_provenance_watermark",
        "doctor",
        "--load-model",
    ]
    writer.command(command)
    try:
        result = subprocess.run(
            command,
            cwd=str(SPIFFY_WORKSPACE),
            capture_output=True,
            text=True,
            timeout=VIDEOSEAL_PREFLIGHT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"VideoSeal model preflight timed out after {VIDEOSEAL_PREFLIGHT_TIMEOUT_SECONDS} seconds."
        ) from exc
    if result.stderr.strip():
        writer.log("VideoSeal preflight: " + result.stderr.strip()[-4000:])
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"VideoSeal preflight returned invalid JSON: {result.stdout[-2000:]}") from exc
    if result.returncode != 0 or not report.get("ready"):
        errors = "; ".join(str(value) for value in report.get("errors") or [])
        raise RuntimeError("VideoSeal preflight failed: " + (errors or result.stderr.strip() or "unknown error"))
    return report


def final_mp4_with_optional_watermark(
    source: Path,
    identity_source: Path,
    dest: Path,
    ffmpeg: str,
    ffprobe: str,
    src_info: dict[str, Any],
    layout_id: str,
    frame_w: int,
    frame_h: int,
    crf: int,
    preset: str,
    writer: ProgressWriter,
    *,
    watermark_enabled: bool,
    watermark_strength: float,
    watermark_namespace: str = "",
    watermark_identifier: str = "",
    delivery_profile_id: str = "",
    max_fps: int | None = None,
    duration_s: float | None = None,
    percent_start: float = 0.0,
    percent_end: float = 0.0,
    receipt_path: Path | None = None,
    frame_rate: dict[str, Any] | None = None,
    original_info: dict[str, Any] | None = None,
    preserve_source_timing: bool = False,
) -> dict[str, Any] | None:
    if type(preserve_source_timing) is not bool:
        raise ValueError("preserve_source_timing must be a boolean.")
    if frame_rate is not None and watermark_enabled:
        raise ValueError("Explicit frame_rate with VideoSeal is not yet qualified.")
    if preserve_source_timing and (frame_rate is not None or watermark_enabled or max_fps is not None):
        raise ValueError("Source-PTS preservation requires an unwatermarked encode with no CFR or FPS cap.")
    if not watermark_enabled:
        final_mp4(
            source,
            dest,
            ffmpeg,
            src_info,
            layout_id,
            frame_w,
            frame_h,
            crf,
            preset,
            writer,
            max_fps,
            duration_s,
            percent_start,
            percent_end,
            **({"frame_rate": frame_rate} if frame_rate is not None else {}),
            **({"original_info": original_info} if original_info is not None else {}),
            **({"preserve_source_timing": True} if preserve_source_timing else {}),
        )
        return None

    _, watermark_final_encode = _watermark_api()
    video_filter = layout_filter(layout_id, int(src_info["width"]), int(src_info["height"]), even(frame_w), even(frame_h))
    if max_fps:
        video_filter = f"{video_filter},fps=fps={max_fps}:round=near"
    output_fps = float(max_fps or src_info.get("fps") or 0)
    if output_fps <= 0:
        raise RuntimeError(f"Cannot determine final FPS for VideoSeal export: {source}")
    last_progress_update = 0.0

    def update_watermark_progress(processed_frames: int, total_frames: int) -> None:
        nonlocal last_progress_update
        now = time.monotonic()
        if now - last_progress_update < 0.8 and processed_frames < total_frames:
            return
        last_progress_update = now
        ratio = min(1.0, processed_frames / max(1, total_frames))
        percent = percent_start + (percent_end - percent_start) * ratio if percent_end > percent_start else 0.0
        writer.update(
            phase="final_h264_watermark",
            message=f"Embedding invisible provenance ({processed_frames}/{total_frames} frames)",
            percent=round(percent, 2) if percent else None,
            metrics={"watermark_frames_processed": processed_frames, "watermark_frames_total_estimate": total_frames},
        )

    result = watermark_final_encode(
        source,
        dest,
        identity_source=identity_source,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        video_filter=video_filter,
        output_width=even(frame_w),
        output_height=even(frame_h),
        output_fps=output_fps,
        crf=crf,
        preset=preset,
        strength=watermark_strength,
        namespace=watermark_namespace,
        identifier=watermark_identifier,
        delivery_profile_id=delivery_profile_id,
        log=writer.log,
        progress=update_watermark_progress,
        receipt_path=receipt_path,
    )
    writer.log(f"Invisible provenance verified: {result}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-root", default=str(ROOT / "_jobs"))
    parser.add_argument("--job-id", default="")
    parser.add_argument("--resume-job-id", default="")
    parser.add_argument("--reconcile-completed-sources", action="store_true")
    parser.add_argument("--input-list", required=True)
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto")
    parser.add_argument("--preset-id", default="auto")
    parser.add_argument("--layout", default="keep_source_aspect")
    parser.add_argument("--crf", type=int, default=17)
    parser.add_argument("--x264-preset", default="slow")
    parser.add_argument("--watermark", choices=["on", "off"], default="off")
    parser.add_argument("--watermark-strength", type=float, default=DEFAULT_WATERMARK_STRENGTH)
    parser.add_argument("--watermark-namespace", default="")
    parser.add_argument("--watermark-identifier", default="")
    parser.add_argument("--delivery-plan", default="")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    job_id = args.job_id or make_job_id("video")
    writer = ProgressWriter(args.jobs_root, job_id, "Upscale Video", "video_upscale")
    started = time.monotonic()
    final_outputs: list[str] = []
    errors: list[str] = []
    cleanup_results: dict[str, Any] = {}
    watermark_results: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}

    receipt_cleanup = prune_temporary_videoseal_receipts(args.jobs_root)
    writer.log(
        "Temporary VideoSeal receipt cleanup: "
        f"scanned={receipt_cleanup['scanned']} removed={receipt_cleanup['removed']} "
        f"retention_days={VIDEOSEAL_RECEIPT_RETENTION_DAYS}"
    )

    try:
        retry_temp_dir = validated_resume_temp_dir(
            args.jobs_root,
            args.resume_job_id,
            current_job_id=job_id,
        )
        resume_manifest = (
            load_json(retry_temp_dir.parent / "manifest.json")
            if retry_temp_dir is not None
            else {}
        )
        raw_files = [Path(p) for p in load_input_list(args.input_list)]
        normalized_files = normalize_input_paths(raw_files, writer)
        if not normalized_files:
            writer.error("No input files were provided.", "preflight")
            return 2
        delivery_plan = load_delivery_plan(args.delivery_plan)
        frame_rate = validate_frame_rate_plan(delivery_plan)
        adaptive_policy = delivery_plan.get("adaptive_enhancement")
        recipe = delivery_enhancement_recipe(delivery_plan, args.preset_id)
        if recipe.explicit:
            args.preset_id = recipe.preset_id
            writer.log("Resolved enhancement recipe: " + json.dumps(recipe.as_dict(), sort_keys=True))
        if resume_manifest and (not enhancement_resume_matches(resume_manifest, recipe)
                                or resume_manifest.get("frame_rate") != frame_rate
                                or resume_manifest.get("adaptive_enhancement") != adaptive_policy
                                or (adaptive_policy is not None and resume_manifest.get("delivery_plan", {}).get("adaptive_engine_identity") != delivery_plan.get("adaptive_engine_identity"))):
            writer.log("Enhancement recipe or frame-rate contract changed; disabling prior child checkpoints and completed-source carry-forward.")
            resume_manifest = {}
            retry_temp_dir = None
        bad = [str(p) for p in normalized_files if p.suffix.lower() not in VIDEO_EXTS]
        if not delivery_plan:
            bad.extend(str(p) for p in normalized_files if not p.exists())
        if bad:
            writer.error("Unsupported or missing video path(s): " + "; ".join(bad), "preflight")
            return 2

        if args.reconcile_completed_sources:
            if retry_temp_dir is None and not args.resume_job_id:
                raise ValueError(
                    "--reconcile-completed-sources requires a validated --resume-job-id."
                )
            carry_forward = build_completed_source_carry_forward(
                resume_manifest,
                scheduled_files=normalized_files,
            )
        else:
            carry_forward = build_completed_source_carry_forward(
                {},
                scheduled_files=normalized_files,
            )
        carried_states = carry_forward["states"]
        carried_state_keys = carry_forward["state_keys"]
        files = carry_forward["pending_files"]
        carried_completed_sources = carry_forward["completed_sources"]
        carried_resolved_sources = carry_forward["resolved_sources"]
        carried_completed_delivery_outputs = carry_forward["completion_rows"]
        final_outputs = carry_forward["final_outputs"]
        watermark_results = carry_forward["watermark_results"]

        preset_path = ROOT / "presets" / "video" / f"{args.preset_id}.json"
        if not preset_path.exists():
            preset_path = ROOT / "presets" / "video" / "auto.json"
        layout_path = ROOT / "presets" / "video_layouts.json"
        preset_json = load_json(preset_path)
        layouts = load_json(layout_path)
        layout_json = layouts.get(args.layout) or layouts["keep_source_aspect"]
        delivery_watermark_required = bool(
            args.watermark == "on"
            or any(bool(row.get("watermark")) for row in delivery_plan.get("variants") or [] if isinstance(row, dict))
        )
        if frame_rate is not None and frame_rate.get("mode") == "preserve_source" and delivery_watermark_required:
            writer.error("Source-PTS preservation is not qualified with VideoSeal or global watermarking; use watermark off.", "preflight")
            return 3

        topaz = find_topaz_video()
        ffmpeg = standard_ffmpeg()
        ffprobe = standard_ffprobe()
        topaz_ffmpeg = topaz.get("ffmpeg") or ""
        topaz_ffprobe = topaz.get("ffprobe") or ffprobe
        topaz_model_dir = str(topaz.get("model_dir") or "").strip()

        manifest: dict[str, Any] = {
            "worker": "video_upscale",
            "selected_files": [str(p) for p in raw_files],
            "normalized_files": [str(p) for p in normalized_files],
            "mode": args.mode,
            "preset_id": args.preset_id,
            "preset_json_path": str(preset_path),
            "preset_json": preset_json,
            "layout": args.layout,
            "layout_json": layout_json,
            "delivery_plan": delivery_plan,
            "delivery_plan_path": str(Path(args.delivery_plan).expanduser().resolve()) if args.delivery_plan else "",
            "final_encode": {
                "crf": args.crf,
                "x264_preset": args.x264_preset,
                "delivery_size_policy": (
                    "explicit_delivery_plan_dimensions"
                    if delivery_plan
                    else "upscale_target_dimensions"
                ),
            },
            "invisible_provenance": {
                "enabled": delivery_watermark_required,
                "global_default_enabled": args.watermark == "on",
                "provider": "facebookresearch/videoseal",
                "strength": args.watermark_strength,
                "default_namespace": args.watermark_namespace,
                "default_identifier": args.watermark_identifier,
                "policy": "one_final_lossy_encode_sparse_embedding_sparse_post_encode_verification",
            },
            "resume_checkpoints": {
                "schema": TOPAZ_CHECKPOINT_SCHEMA,
                "policy": "same_job_or_validated_failed_retry_lineage_fail_closed_identity",
                "identity_owner": "Spiffy Workspace/workflow/workflow_hashing.py",
                "sample_bytes_per_region": CHECKPOINT_SAMPLE_BYTES,
                "sample_regions": ["first", "middle", "last"],
                "retry_of_job_id": str(args.resume_job_id or ""),
                "retry_temp_dir": str(retry_temp_dir or ""),
                "cross_job_reuse": "validated_failed_retry_lineage_only",
                "carried_completed_source_count": len(carried_states),
                "carried_completion_policy": (
                    "current_source_mapping_and_every_final_output_identity_required"
                ),
            },
            "source_app_window": "",
            "resolved_executable_paths": {
                "python": sys.executable,
                "topaz_ffmpeg": topaz_ffmpeg,
                "topaz_ffprobe": topaz_ffprobe,
                "topaz_model_dir": topaz_model_dir,
                "standard_ffmpeg": ffmpeg,
                "standard_ffprobe": ffprobe,
            },
            "topaz_video_discovery": topaz,
            "exact_commands_run": [],
            "final_outputs": list(final_outputs),
            "resolved_delivery_sources": list(carried_resolved_sources),
            "resolved_delivery_profile_output_map": {},
            "resolved_delivery_physical_output_count": 0,
            "errors": [],
            "cleanup_results": {},
            "watermark_results": list(watermark_results),
            "completed_delivery_outputs": list(carried_completed_delivery_outputs),
            "completed_sources": list(carried_completed_sources),
            "ready_sources": [],
        }
        if recipe.explicit:
            manifest["enhancement_recipe"] = recipe.as_dict()
        from delivery_insights import resource_snapshot
        manifest["resource_start"] = resource_snapshot()
        if frame_rate is not None:
            manifest["frame_rate"] = dict(frame_rate)
        if adaptive_policy is not None:
            manifest["adaptive_enhancement"] = dict(adaptive_policy)
            manifest["adaptive_plans"] = [dict(row) for row in resume_manifest.get("adaptive_plans", [])
                                          if isinstance(row, dict) and row.get("source_path")
                                          and _path_key(row["source_path"]) not in {_path_key(p) for p in files}]
            manifest["adaptive_quality"] = dict(resume_manifest.get("adaptive_quality") or {})
        if carried_resolved_sources:
            update_single_source_delivery_summary(
                manifest,
                total_source_count=len(normalized_files),
                profile_output_map=dict(
                    carried_resolved_sources[0].get("profile_output_map") or {}
                ),
                physical_output_count=int(
                    carried_resolved_sources[0].get("physical_output_count") or 0
                ),
            )
        writer.write_manifest(manifest)
        writer.log(f"Video preset: {preset_path}")
        writer.log(f"Layout: {layout_json}")
        writer.log(f"Topaz discovery: {topaz}")
        if carried_states:
            writer.log(
                "Carried forward fully revalidated completed sources from failed job "
                f"{args.resume_job_id}: {len(carried_states)} source(s), "
                f"{len(final_outputs)} final output(s)."
            )
        if not files:
            require_carried_completed_sources_current(
                resume_manifest,
                expected_state_keys=carried_state_keys,
            )
            cleanup_results["temp_cleanup"] = "not_required_all_sources_reconciled"
            manifest["cleanup_results"] = cleanup_results
            manifest["elapsed_seconds"] = round(time.monotonic() - started, 2)
            writer.write_manifest(manifest)
            writer.done("Video upscale retry reconciliation complete")
            return 0

        writer.update(
            state="running",
            phase="preflight",
            message="Checking Topaz Video and FFmpeg",
            file_total=len(files),
            metrics={"tvai_up": bool(topaz.get("tvai_up")), "tvai_fi": bool(topaz.get("tvai_fi"))},
        )

        if not topaz_ffmpeg or not topaz.get("tvai_up"):
            writer.error("Topaz bundled ffmpeg with tvai_up is required for video upscaling.", "preflight")
            return 3
        try:
            topaz_model_dir = configure_topaz_model_environment(topaz_model_dir)
        except (FileNotFoundError, OSError, ValueError) as exc:
            writer.error(str(exc), "preflight")
            return 3
        manifest["resolved_executable_paths"]["topaz_model_dir"] = topaz_model_dir
        manifest["topaz_video_discovery"]["model_dir"] = topaz_model_dir
        writer.write_manifest(manifest)
        needs_fi = bool(PRESET_MODEL.get(args.preset_id, PRESET_MODEL["auto"]).get("fps"))
        if frame_rate is not None:
            needs_fi = False
        if needs_fi and not topaz.get("tvai_fi"):
            writer.log("tvai_fi missing; source clips below 60 FPS will fail closed instead of duplicating frames.")
        if not Path(ffmpeg).exists() or not Path(ffprobe).exists():
            writer.error("Standard FFmpeg/FFprobe not found for final MP4 encode.", "preflight")
            return 3
        if not has_encoder(ffmpeg, "libx264"):
            writer.error(f"Standard FFmpeg lacks libx264: {ffmpeg}", "preflight")
            return 3
        if delivery_watermark_required:
            if not 0.01 <= float(args.watermark_strength) <= 1.0:
                writer.error("VideoSeal watermark strength must be between 0.01 and 1.0.", "preflight")
                return 3
            watermark_preflight = run_videoseal_preflight(writer)
            manifest["invisible_provenance"]["preflight"] = watermark_preflight
            writer.write_manifest(manifest)
            if not watermark_preflight.get("ready"):
                writer.error(
                    "VideoSeal preflight failed: " + "; ".join(str(value) for value in watermark_preflight.get("errors") or []),
                    "preflight",
                )
                return 3
            writer.log(f"VideoSeal preflight ready: {watermark_preflight.get('paths')}")

        ready_files: list[tuple[Path, Path, dict[str, Any]]] = []
        prepared_adaptive: dict[str, dict[str, Any]] = {}
        prepared_working_geometry: dict[str, dict[str, Any]] = {}
        if adaptive_policy is not None:
            needs_fi = False
        for idx, selected_source in enumerate(files, start=1):
            writer.update(
                phase="wait_for_render",
                message=f"Waiting for source {idx}/{len(files)}",
                percent=round((idx - 1) / len(files) * 5, 1),
                file_index=idx,
                file_total=len(files),
                current_file={"display_name": selected_source.name, "source_path": str(selected_source)},
            )
            source, info = wait_for_render_ready(selected_source, ffprobe, writer)
            validate_source_color_pipeline(info)
            if frame_rate is not None or adaptive_policy is not None:
                ok, info, reason = verify_video(source, ffprobe, inspect_timestamps=True)
                if not ok:
                    raise RuntimeError(f"Source presentation timing could not be qualified: {reason}")
                source_needs_fi = should_interpolate(info, args.preset_id, writer, topaz, frame_rate=frame_rate)
                require_topaz_interpolation_route(info, args.preset_id, source_needs_fi, frame_rate=frame_rate)
                needs_fi = needs_fi or source_needs_fi
            for variant in delivery_plan.get("variants") or []:
                if isinstance(variant, dict) and variant.get("geometry_policy") in {"source_display", "custom_final"}:
                    source_display_geometry(info)
            working_geometry = (
                resolve_delivery_working_geometry(info, args.preset_id, args.layout, delivery_plan)
                if delivery_plan.get("geometry") is not None else None
            )
            if working_geometry is not None:
                prepared_working_geometry[_path_key(source)] = working_geometry
            if adaptive_policy is not None:
                from adaptive_enhancement import prepare as prepare_adaptive
                adaptive_w, adaptive_h, _, _ = target_dimensions_for_delivery(info, args.preset_id, args.layout, delivery_plan)
                adaptive_plan = prepare_adaptive(source, info, adaptive_w, adaptive_h, writer, adaptive_policy)
                prepared_adaptive[_path_key(source)] = adaptive_plan
                manifest["adaptive_plans"].append({"source_path": str(source), "plan": adaptive_plan})
                writer.log("Adaptive decision: " + json.dumps({k: adaptive_plan[k] for k in
                    ("route", "crop", "processed_area_ratio", "effective_controls", "analysis_seconds", "cache_hit")}))
            ready_files.append((selected_source, source, info))
            ready_source = {
                "selected_source_path": str(selected_source),
                "source_path": str(source),
                "probe": dict(info),
            }
            if working_geometry is not None:
                ready_source["working_geometry"] = working_geometry
            manifest["ready_sources"].append(ready_source)
            writer.write_manifest(manifest)

        with SharedGpuLock(args.jobs_root, writer, "gpu_render", "Topaz/AI GPU"):
            up_model = recipe.up_model
            ok, reason = (True, "") if adaptive_policy is not None else validate_topaz_filter_model(
                topaz_ffmpeg, "tvai_up", up_model, writer,
                **({"recipe": recipe} if recipe.explicit else {}),
            )
            if ok and reason:
                writer.log(reason)
            if not ok:
                writer.error(reason, "preflight")
                return 3
            if needs_fi and adaptive_policy is None:
                ok, reason = validate_topaz_filter_model(topaz_ffmpeg, "tvai_fi", "apo-8", writer,
                    **({"target_fps": frame_rate["fps"]} if frame_rate is not None else {}))
                if ok and reason:
                    writer.log(reason)
                if not ok:
                    writer.error(reason, "preflight")
                    return 3

            for idx, ready in enumerate(ready_files, start=1):
                selected_source, source, info = ready
                adaptive_plan = prepared_adaptive.get(_path_key(source))
                execution_recipe = recipe
                adaptive_kwargs = {"adaptive_fingerprint": adaptive_plan["fingerprint"]} if adaptive_plan else {}
                if adaptive_plan is not None:
                    writer.log("Adaptive native admission uses the actual render's first-output-frame and stalled-output deadlines; no separate throwaway model render.")
                if writer.cancel_requested:
                    writer.cancelled()
                    return 2

                writer.update(
                    phase="probe",
                    message="Probing video",
                    percent=round((idx - 1) / len(files) * 100, 1),
                    file_index=idx,
                    file_total=len(files),
                    current_file={"display_name": source.name, "source_path": str(source)},
                )
                if source != selected_source:
                    writer.update(
                        phase="probe",
                        message=f"Using finished render: {source.name}",
                        current_file={"display_name": source.name, "source_path": str(source), "selected_path": str(selected_source)},
                    )
                writer.log(f"{source.name}: probe={info}")
                if not info["width"] or not info["height"]:
                    raise RuntimeError(f"Could not read video dimensions for {source}")
                if info["rotation"]:
                    writer.log(f"Rotation metadata detected: {info['rotation']} degrees. FFmpeg autorotation will be used.")
                if info["color_transfer"] and "smpte2084" in str(info["color_transfer"]).lower():
                    writer.log("HDR transfer detected. HDR to SDR is not fully implemented; final encode may preserve tags imperfectly.")
                if info["is_vfr"]:
                    if frame_rate is not None and frame_rate.get("mode") == "preserve_source":
                        writer.log("Variable presentation timestamps detected; this output requires exact source-PTS preservation.")
                    else:
                        writer.log("Variable frame rate suspected. Final encode may normalize timing through FFmpeg.")

                working_geometry = prepared_working_geometry.get(_path_key(source))
                if working_geometry is not None:
                    target_w = int(working_geometry["target_width"])
                    target_h = int(working_geometry["target_height"])
                    frame_w = int(working_geometry["frame_width"])
                    frame_h = int(working_geometry["frame_height"])
                else:
                    target_w, target_h, frame_w, frame_h = target_dimensions_for_delivery(
                        info, args.preset_id, args.layout, delivery_plan
                    )
                rate_kwargs = {"frame_rate": frame_rate} if frame_rate is not None else {}
                preserve_source_timing = frame_rate is not None and frame_rate["mode"] == "preserve_source"
                if preserve_source_timing:
                    interpolation_fps = None
                elif frame_rate is not None:
                    interpolation_fps = frame_rate["fps"]
                else:
                    interpolation_fps = 60
                interpolate = should_interpolate(info, args.preset_id, writer, topaz, **rate_kwargs)
                require_topaz_interpolation_route(info, args.preset_id, interpolate, **rate_kwargs)
                out_dir = source.parent / "_UpscaledVideo"
                out_dir.mkdir(exist_ok=True)
                retry_source_aliases = resume_source_aliases(
                    resume_manifest,
                    source=source,
                    selected_source=selected_source,
                )
                variants = delivery_variants_for_source(
                    source,
                    info,
                    args,
                    out_dir,
                    delivery_plan,
                    selected_source=selected_source,
                    source_aliases=retry_source_aliases,
                )
                source_display_enabled = any(
                    variant.get("geometry_policy") in {"source_display", "custom_final"} for variant in variants
                ) or working_geometry is not None
                variants, delivery_resume_origin = restore_resume_delivery_variants(
                    variants,
                    source=source,
                    selected_source=selected_source,
                    resume_manifest=resume_manifest,
                )
                if adaptive_plan is not None:
                    from adaptive_model_selection import select_for_source, execution_model, validate_selected_assets
                    adaptive_plan, enhancement_choice = select_for_source(source, info, adaptive_plan,
                        topaz, writer, recipe, delivery_variants=variants)
                    prepared_adaptive[_path_key(source)] = adaptive_plan
                    for recorded in manifest["adaptive_plans"]:
                        if _path_key(recorded["source_path"]) == _path_key(source):
                            recorded["plan"] = adaptive_plan
                    manifest.setdefault("adaptive_enhancement_calibration", {})[_path_key(source)] = enhancement_choice
                    validate_selected_assets(adaptive_plan, topaz_model_dir)
                    chosen_model = execution_model(adaptive_plan, recipe.up_model)
                    execution_recipe = EnhancementRecipe(recipe.preset_id, chosen_model,
                        adaptive_plan["effective_params"], explicit=recipe.explicit)
                    adaptive_kwargs = {"adaptive_fingerprint": adaptive_plan["fingerprint"]}
                    writer.write_manifest(manifest)
                if delivery_resume_origin:
                    writer.log(
                        "Restored delivery output paths from the validated failed-job manifest "
                        f"for {source.name}."
                    )
                if preserve_source_timing:
                    for variant in variants:
                        validate_source_timing_profile_cap(info, int(variant.get("max_fps") or 0))
                resolved_profile_map = delivery_profile_output_map(variants)
                source_identity_at_start = sampled_file_identity(source)
                resolved_source = {
                    "source_path": str(source),
                    "selected_source_path": str(selected_source),
                    "source_identity": source_identity_at_start,
                    "physical_output_count": len(variants),
                    "profile_output_map": resolved_profile_map,
                    "variants": variants,
                }
                manifest["resolved_delivery_sources"].append(resolved_source)
                update_single_source_delivery_summary(
                    manifest,
                    total_source_count=len(normalized_files),
                    profile_output_map=resolved_profile_map,
                    physical_output_count=len(variants),
                )
                writer.write_manifest(manifest)
                writer.log(
                    "Resolved delivery variants after source probe: "
                    f"{len(delivery_plan.get('variants') or variants)} planned -> {len(variants)} physical"
                )
                temp1 = writer.temp_dir / f"{source.stem}_topaz_up.mov"
                temp2 = writer.temp_dir / f"{source.stem}_topaz_fi.mov"
                temp1_checkpoint = topaz_checkpoint_path(temp1)
                temp2_checkpoint = topaz_checkpoint_path(temp2)
                max_fps = None if preserve_source_timing else final_fps_cap(info, args.preset_id)

                writer.update(
                    phase="topaz_upscale",
                    message=f"Topaz {target_w}x{target_h}; {len(variants)} final delivery variant(s)",
                    output_folder=str(Path(variants[0]["output_path"]).parent),
                    metrics={
                        "target_width": target_w,
                        "target_height": target_h,
                        "frame_width": frame_w,
                        "frame_height": frame_h,
                        "layout": args.layout,
                        "interpolate": interpolate,
                        **frame_rate_progress_metrics(variants, max_fps, source_info=info if preserve_source_timing else None),
                        "delivery_variant_count": len(variants),
                    },
                )
                file_base_percent = (idx - 1) / len(files) * 100
                file_span_percent = 100 / len(files)
                model_spec = {"up_model": execution_recipe.up_model, "params": adaptive_plan["effective_params"] if adaptive_plan else recipe.params}
                upscale_assets_before = topaz_model_asset_identity(
                    topaz_model_dir,
                    str(model_spec["up_model"]),
                )
                upscale_checkpoint_expected = build_topaz_checkpoint_fingerprint(
                    stage="topaz_upscale",
                    input_path=source,
                    topaz_ffmpeg=topaz_ffmpeg,
                    preset_id=args.preset_id,
                    model=str(model_spec["up_model"]),
                    model_params=str(model_spec.get("params") or ""),
                    target_width=target_w,
                    target_height=target_h,
                    model_asset_identity=upscale_assets_before,
                    enhancement_recipe=recipe.as_dict() if recipe.explicit else None,
                    **({"working_geometry": working_geometry} if working_geometry is not None else {}),
                    **adaptive_kwargs,
                    **rate_kwargs,
                )
                reusable_upscale, upscale_resume_origin = (
                    resolve_current_or_retry_checkpoint(
                        current_checkpoint=temp1_checkpoint,
                        current_output=temp1,
                        retry_temp_dir=retry_temp_dir,
                        expected=upscale_checkpoint_expected,
                    )
                    if upscale_assets_before["resume_safe"]
                    else (None, "")
                )
                if not upscale_assets_before["resume_safe"]:
                    writer.log(
                        f"Topaz model assets are not fully discoverable for {model_spec['up_model']}; "
                        "checkpoint reuse is disabled for this stage."
                    )
                if reusable_upscale is not None:
                    upscale_cpu_recovery_used = False
                    writer.update(
                        phase="resume_topaz_upscale",
                        message="Verifying completed Topaz upscale checkpoint",
                        percent=round(file_base_percent + file_span_percent * 0.45, 1),
                    )
                    writer.log(
                        "Reusing fingerprint-matched Topaz upscale checkpoint "
                        f"({upscale_resume_origin}): {reusable_upscale}"
                    )
                    verified_upscale = verify_or_remux(
                        reusable_upscale,
                        topaz_ffmpeg,
                        topaz_ffprobe,
                        writer,
                    )
                else:
                    upscale_cpu_recovery_used = topaz_upscale_to_prores(
                        source,
                        temp1,
                        topaz_ffmpeg,
                        info,
                        args.preset_id,
                        target_w,
                        target_h,
                        writer,
                        file_base_percent + file_span_percent * 0.05,
                        file_base_percent + file_span_percent * 0.45,
                        **({"recipe": execution_recipe} if recipe.explicit or adaptive_plan is not None else {}),
                        **({"preserve_source_timing": True} if preserve_source_timing else {}),
                        **({"adaptive_plan": adaptive_plan} if adaptive_plan is not None else {}),
                        **({"working_geometry": working_geometry} if working_geometry is not None else {}),
                    )
                    verified_upscale = verify_or_remux(
                        temp1,
                        topaz_ffmpeg,
                        topaz_ffprobe,
                        writer,
                    )
                if frame_rate is not None:
                    ok, verified_upscale_info, reason = verify_video(
                        verified_upscale, topaz_ffprobe, inspect_timestamps=True,
                    )
                    if not ok:
                        raise RuntimeError(f"Topaz upscale presentation timing could not be qualified: {reason}")
                else:
                    verified_upscale_info = probe_video(verified_upscale, topaz_ffprobe)
                if adaptive_plan is not None:
                    from adaptive_enhancement import assess_candidate, conventional_filter_graph, engine_identity
                    from adaptive_model_selection import validate_selected_assets
                    validate_selected_assets(adaptive_plan, topaz_model_dir)
                    if engine_identity() != delivery_plan["adaptive_engine_identity"]:
                        raise RuntimeError("Adaptive implementation changed during the render; output is not publishable.")
                    quality = assess_candidate(source, verified_upscale, adaptive_plan, writer)
                    if quality["status"] != "PASS_OBJECTIVE_GUARDS":
                        writer.log("Adaptive candidate rejected; retaining original appearance through conventional resize: " + json.dumps(quality["reasons"]))
                        rejected = verified_upscale.with_name(verified_upscale.stem + "_rejected.mov")
                        os.replace(verified_upscale, rejected)
                        conventional_argv = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", str(source),
                            "-filter_complex", conventional_filter_graph(adaptive_plan), "-map", "[vout]", "-map", "0:a?",
                            "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-c:a", "pcm_s24le", str(verified_upscale)]
                        run_process(conventional_argv, writer, "adaptive_preservation_fallback", duration_s=float(info.get("duration") or 0) or None)
                        fallback_quality = assess_candidate(source, verified_upscale, adaptive_plan, writer)
                        if fallback_quality["status"] != "PASS_OBJECTIVE_GUARDS":
                            raise RuntimeError("Conventional preservation also failed adaptive objective guards; no final output published.")
                        quality = {"status": "SOURCE_PRESERVATION_FALLBACK", "rejected_candidate": quality,
                                   "fallback": fallback_quality, "actual_route": "source_preserved"}
                        rejected.unlink(missing_ok=True)
                        verified_upscale_info = probe_video(verified_upscale, topaz_ffprobe, inspect_timestamps=True)
                    else:
                        quality["actual_route"] = adaptive_plan["route"]
                    quality["adaptive_fingerprint"] = adaptive_plan["fingerprint"]
                    from adaptive_enhancement import validate_quality_receipt
                    validate_quality_receipt(quality, adaptive_plan)
                    manifest["adaptive_quality"][_path_key(source)] = quality
                    writer.write_manifest(manifest)
                validate_working_dimensions(
                    verified_upscale_info,
                    expected_width=target_w,
                    expected_height=target_h,
                    phase="Topaz upscale",
                )
                if source_display_enabled:
                    validate_source_display_square_pixels(verified_upscale_info, phase="Topaz upscale")
                if preserve_source_timing:
                    validate_source_timing_preserved(info, verified_upscale_info, phase="Topaz upscale")
                validate_intermediate_audio(
                    info,
                    verified_upscale_info,
                    phase="Topaz upscale",
                    require_sample_extent=frame_rate is not None,
                )
                upscale_assets_after = topaz_model_asset_identity(
                    topaz_model_dir,
                    str(model_spec["up_model"]),
                )
                upscale_checkpoint_after = build_topaz_checkpoint_fingerprint(
                    stage="topaz_upscale",
                    input_path=source,
                    topaz_ffmpeg=topaz_ffmpeg,
                    preset_id=args.preset_id,
                    model=str(model_spec["up_model"]),
                    model_params=str(model_spec.get("params") or ""),
                    target_width=target_w,
                    target_height=target_h,
                    model_asset_identity=upscale_assets_after,
                    enhancement_recipe=recipe.as_dict() if recipe.explicit else None,
                    **({"working_geometry": working_geometry} if working_geometry is not None else {}),
                    **adaptive_kwargs,
                    **rate_kwargs,
                )
                stable_upscale_checkpoint = validate_topaz_stage_identity_stable(
                    upscale_checkpoint_expected,
                    upscale_checkpoint_after,
                    stage="Topaz upscale",
                )
                if upscale_cpu_recovery_used:
                    retained_upscale_checkpoint = None
                    writer.log(
                        "Topaz upscale used the bounded CPU recovery backend; upscale checkpoint and "
                        "downstream delivery reuse are disabled for this run because the default GPU "
                        "backend did not produce the retained intermediate."
                    )
                else:
                    retained_upscale_checkpoint = retain_verified_topaz_checkpoint(
                        temp1_checkpoint,
                        output_path=verified_upscale,
                        expected=stable_upscale_checkpoint,
                    )
                if retained_upscale_checkpoint is None and not upscale_cpu_recovery_used:
                    writer.log(
                        "Topaz upscale checkpoint and downstream delivery reuse are disabled "
                        "because the selected model assets are not fully discoverable."
                    )
                verified = verified_upscale
                cleanup_candidates = {
                    temp1,
                    temp2,
                    temp1_checkpoint,
                    temp2_checkpoint,
                    temp1.with_name(f"{temp1.stem}_fixed{temp1.suffix}"),
                    temp2.with_name(f"{temp2.stem}_fixed{temp2.suffix}"),
                    verified_upscale,
                }
                if adaptive_plan is not None:
                    cleanup_candidates.update({Path(adaptive_plan["mask_path"]), Path(adaptive_plan["mask_path"]).with_suffix(".json")})

                interpolation_w, interpolation_h = target_w, target_h
                interpolation_cuts: list[int] | None = None
                interpolation_model = "apo-8"
                if interpolate and adaptive_plan is not None:
                    from adaptive_enhancement import (
                        interpolation_geometry,
                        interpolation_has_source_protection,
                    )
                    interpolation_w, interpolation_h = interpolation_geometry(adaptive_plan, variants)
                    protect_authored = interpolation_has_source_protection(adaptive_plan)
                    interpolation_cuts = [int(shot["start_frame"]) for shot in adaptive_plan.get("shots", [])
                                          if int(shot["start_frame"]) > 0]
                    interpolation_selection = select_adaptive_interpolation_model(
                        verified_upscale, verified_upscale_info, adaptive_plan, interpolation_fps,
                        (interpolation_w, interpolation_h), topaz_ffmpeg, topaz_model_dir, writer)
                    interpolation_model = interpolation_selection["model"]
                    manifest.setdefault("adaptive_interpolation_calibration", {})[_path_key(source)] = interpolation_selection
                    manifest.setdefault("adaptive_interpolation", {})[_path_key(source)] = {
                        "input_width": interpolation_w, "input_height": interpolation_h,
                        "enhancement_width": target_w, "enhancement_height": target_h,
                        "source_cut_frames": interpolation_cuts,
                        "cut_policy": "source_hold_exact_cfr_clock_v2",
                        "authored_repeated_frames": "preserved_rdt_0",
                        "caption_protection": dict(adaptive_plan.get("caption_protection") or {}),
                        "model": interpolation_model,
                        "selection_contract": {k: interpolation_selection[k] for k in
                            ("schema", "model", "reason", "fingerprint") if k in interpolation_selection},
                        "size_policy": adaptive_plan["policy"].get("interpolation_resolution", "delivery")}
                    writer.write_manifest(manifest)
                if interpolate:
                    interpolation_parameters = "slowmo=1:rdt=0" if adaptive_plan is not None else "slowmo=1:rdt=0.01"
                    if frame_rate is not None:
                        boundary = interpolation_boundary_contract(verified_upscale_info, interpolation_fps)
                        interpolation_parameters += ":boundary=" + json.dumps(boundary, sort_keys=True, separators=(",", ":"))
                    if adaptive_plan is not None:
                        interpolation_parameters += ":adaptive_interpolation=" + json.dumps(
                            manifest["adaptive_interpolation"][_path_key(source)], sort_keys=True, separators=(",", ":"))
                    interpolation_assets_before = topaz_model_asset_identity(
                        topaz_model_dir,
                        interpolation_model,
                    )
                    interpolation_checkpoint_expected = build_topaz_checkpoint_fingerprint(
                        stage="topaz_interpolate",
                        input_path=verified_upscale,
                        topaz_ffmpeg=topaz_ffmpeg,
                        preset_id=args.preset_id,
                        model=interpolation_model,
                        model_params=interpolation_parameters,
                        target_width=interpolation_w,
                        target_height=interpolation_h,
                        target_fps=interpolation_fps,
                        model_asset_identity=interpolation_assets_before,
                        enhancement_recipe=recipe.as_dict() if recipe.explicit else None,
                        **adaptive_kwargs,
                        **rate_kwargs,
                    )
                    reusable_interpolation, interpolation_resume_origin = (
                        resolve_current_or_retry_checkpoint(
                            current_checkpoint=temp2_checkpoint,
                            current_output=temp2,
                            retry_temp_dir=retry_temp_dir,
                            expected=interpolation_checkpoint_expected,
                        )
                        if interpolation_assets_before["resume_safe"]
                        else (None, "")
                    )
                    if not interpolation_assets_before["resume_safe"]:
                        writer.log(
                            "Topaz interpolation model assets are not fully discoverable; "
                            "checkpoint reuse is disabled for this stage."
                        )
                    if reusable_interpolation is not None:
                        interpolation_cpu_recovery_used = False
                        writer.update(
                            phase="resume_topaz_interpolate",
                            message=f"Verifying completed {interpolation_fps} FPS checkpoint",
                            percent=round(file_base_percent + file_span_percent * 0.82, 1),
                        )
                        writer.log(
                            "Reusing fingerprint-matched Topaz interpolation checkpoint "
                            f"({interpolation_resume_origin}): {reusable_interpolation}"
                        )
                        verified = verify_or_remux(
                            reusable_interpolation,
                            topaz_ffmpeg,
                            topaz_ffprobe,
                            writer,
                        )
                    else:
                        writer.update(phase="topaz_interpolate", message=f"Interpolating {interpolation_w}x{interpolation_h} to {interpolation_fps} FPS",
                                      metrics={"interpolation_width": interpolation_w, "interpolation_height": interpolation_h,
                                               "protected_cut_count": len(interpolation_cuts or [])})
                        interpolation_cpu_recovery_used = topaz_interpolate_to_prores(
                            verified_upscale,
                            temp2,
                            topaz_ffmpeg,
                            interpolation_fps,
                            writer,
                            float(info.get("duration") or 0) or None,
                            file_base_percent + file_span_percent * 0.45,
                            file_base_percent + file_span_percent * 0.82,
                            recovery_width=interpolation_w,
                            recovery_height=interpolation_h,
                            **({"source_info": verified_upscale_info} if frame_rate is not None or adaptive_plan is not None else {}),
                            **({"guarded_startup": True, "output_dimensions": (interpolation_w, interpolation_h),
                                "source_cut_frames": interpolation_cuts,
                                "interpolation_model": interpolation_model,
                                "caption_mask_plan": adaptive_plan if protect_authored else None}
                               if adaptive_plan is not None else {}),
                        )
                        verified = verify_or_remux(
                            temp2,
                            topaz_ffmpeg,
                            topaz_ffprobe,
                            writer,
                        )
                    cleanup_candidates.add(verified)

                if frame_rate is not None:
                    ok, verified_info, reason = verify_video(verified, ffprobe, inspect_timestamps=True)
                    if not ok:
                        raise RuntimeError(f"Topaz working presentation timing could not be qualified: {reason}")
                else:
                    verified_info = probe_video(verified, ffprobe)
                validate_working_dimensions(
                    verified_info,
                    expected_width=interpolation_w,
                    expected_height=interpolation_h,
                    phase="Topaz working intermediate",
                )
                if source_display_enabled:
                    validate_source_display_square_pixels(verified_info, phase="Topaz working intermediate")
                validate_intermediate_audio(
                    info,
                    verified_info,
                    phase="Topaz working intermediate",
                    require_sample_extent=frame_rate is not None,
                )
                if preserve_source_timing:
                    validate_source_timing_preserved(info, verified_info, phase="Topaz working intermediate")
                if interpolate:
                    validate_interpolated_fps(verified_info, interpolation_fps)
                    if adaptive_plan is not None:
                        from adaptive_enhancement import assess_temporal_candidate, validate_quality_receipt
                        writer.update(phase="adaptive_temporal_quality", message="Checking caption motion and shot boundaries")
                        temporal_quality = assess_temporal_candidate(verified_upscale, verified, adaptive_plan,
                            str(interpolation_fps), interpolation_cuts or [], writer)
                        manifest["adaptive_quality"][_path_key(source)]["temporal_protection"] = temporal_quality
                        writer.write_manifest(manifest)
                        if temporal_quality["status"] != "PASS_AUTHORED_PROTECTION":
                            raise RuntimeError("Adaptive interpolation changed authored captions or cuts: "
                                               + "; ".join(temporal_quality["reasons"]))
                        validate_quality_receipt(manifest["adaptive_quality"][_path_key(source)],
                                                 adaptive_plan, require_temporal=True)
                    interpolation_assets_after = topaz_model_asset_identity(
                        topaz_model_dir,
                        interpolation_model,
                    )
                    interpolation_checkpoint_after = build_topaz_checkpoint_fingerprint(
                        stage="topaz_interpolate",
                        input_path=verified_upscale,
                        topaz_ffmpeg=topaz_ffmpeg,
                        preset_id=args.preset_id,
                        model=interpolation_model,
                        model_params=interpolation_parameters,
                        target_width=interpolation_w,
                        target_height=interpolation_h,
                        target_fps=interpolation_fps,
                        model_asset_identity=interpolation_assets_after,
                        enhancement_recipe=recipe.as_dict() if recipe.explicit else None,
                        **adaptive_kwargs,
                        **rate_kwargs,
                    )
                    stable_interpolation_checkpoint = validate_topaz_stage_identity_stable(
                        interpolation_checkpoint_expected,
                        interpolation_checkpoint_after,
                        stage="Topaz interpolation",
                    )
                    if interpolation_cpu_recovery_used:
                        retained_interpolation_checkpoint = None
                        writer.log(
                            "Topaz interpolation used the bounded CPU recovery backend; interpolation "
                            "checkpoint and downstream delivery reuse are disabled for this run because "
                            "the default GPU backend did not produce the retained intermediate."
                        )
                    else:
                        retained_interpolation_checkpoint = retain_verified_topaz_checkpoint(
                            temp2_checkpoint,
                            output_path=verified,
                            expected=stable_interpolation_checkpoint,
                        )
                    if retained_interpolation_checkpoint is None and not interpolation_cpu_recovery_used:
                        writer.log(
                            "Topaz interpolation checkpoint and downstream delivery reuse are disabled "
                            "because the selected model assets are not fully discoverable."
                        )
                working_checkpoint_for_delivery = select_reusable_working_checkpoint(
                    interpolate=interpolate,
                    upscale_cpu_recovery_used=upscale_cpu_recovery_used,
                    interpolation_cpu_recovery_used=(
                        interpolation_cpu_recovery_used if interpolate else False
                    ),
                    retained_upscale_checkpoint=retained_upscale_checkpoint,
                    retained_interpolation_checkpoint=(
                        retained_interpolation_checkpoint if interpolate else None
                    ),
                )
                variant_probes: list[dict[str, Any]] = []
                source_delivery_completion_fingerprints: list[str] = []
                final_span_start = file_base_percent + file_span_percent * 0.82
                final_span_end = file_base_percent + file_span_percent * 0.98
                final_span = final_span_end - final_span_start
                for variant_index, variant in enumerate(variants, start=1):
                    final = Path(variant["output_path"])
                    final.parent.mkdir(parents=True, exist_ok=True)
                    variant_start = final_span_start + final_span * (variant_index - 1) / len(variants)
                    variant_end = final_span_start + final_span * variant_index / len(variants)
                    delivery_completion_expected = build_reusable_delivery_completion_expectation(
                        variant=variant,
                        retained_working_checkpoint=working_checkpoint_for_delivery,
                    )
                    resumed_delivery = (
                        resolve_completed_delivery_output(
                            resume_manifest,
                            expected=delivery_completion_expected,
                        )
                        if delivery_completion_expected is not None
                        else None
                    )
                    if resumed_delivery is not None:
                        final, prior_completion = resumed_delivery
                        writer.update(
                            phase="resume_final_delivery",
                            message=f"Verifying completed {variant['profile_id']} ({variant_index}/{len(variants)})",
                            output_folder=str(final.parent),
                            metrics={
                                "delivery_profile_id": variant["profile_id"],
                                "delivery_width": variant["width"],
                                "delivery_height": variant["height"],
                                "delivery_layout": variant["layout"],
                                "delivery_resume_origin": str(args.resume_job_id),
                            },
                        )
                        watermark_result = (
                            dict(prior_completion.get("watermark_result") or {})
                            if isinstance(prior_completion.get("watermark_result"), dict)
                            else {}
                        )
                        writer.log(
                            "Reusing fingerprint-matched completed delivery from failed job "
                            f"{args.resume_job_id}: {final}"
                        )
                    else:
                        variant, collision_route = route_unmatched_delivery_output(
                            variant,
                            validated_retry_manifest_restored=bool(
                                args.reconcile_completed_sources
                                and delivery_resume_origin == "validated_retry_manifest"
                            ),
                        )
                        if collision_route:
                            variants[variant_index - 1] = variant
                            final = Path(variant["output_path"])
                            final.parent.mkdir(parents=True, exist_ok=True)
                            delivery_completion_expected = (
                                build_reusable_delivery_completion_expectation(
                                    variant=variant,
                                    retained_working_checkpoint=working_checkpoint_for_delivery,
                                )
                            )
                            resolved_profile_map = delivery_profile_output_map(variants)
                            resolved_source["profile_output_map"] = resolved_profile_map
                            resolved_source["physical_output_count"] = len(variants)
                            resolved_source["variants"] = list(variants)
                            update_single_source_delivery_summary(
                                manifest,
                                total_source_count=len(normalized_files),
                                profile_output_map=resolved_profile_map,
                                physical_output_count=len(variants),
                            )
                            writer.write_manifest(manifest)
                            writer.log(
                                "Prior retry delivery exists but failed completion identity; "
                                "preserving it and encoding to collision-safe path: "
                                f"{final}"
                            )
                        writer.update(
                            phase="final_h264",
                            message=f"Encoding {variant['profile_id']} ({variant_index}/{len(variants)})",
                            output_folder=str(final.parent),
                            metrics={
                                "delivery_profile_id": variant["profile_id"],
                                "delivery_width": variant["width"],
                                "delivery_height": variant["height"],
                                "delivery_layout": variant["layout"],
                            },
                        )
                        watermark_result = final_mp4_with_optional_watermark(
                            verified,
                            source,
                            final,
                            ffmpeg,
                            ffprobe,
                            verified_info,
                            variant["layout"],
                            variant["width"],
                            variant["height"],
                            variant["crf"],
                            variant["x264_preset"],
                            writer,
                            watermark_enabled=bool(variant["watermark"]),
                            watermark_strength=float(variant["watermark_strength"]),
                            watermark_namespace=str(variant["namespace"]),
                            watermark_identifier=str(variant.get("watermark_identifier") or ""),
                            delivery_profile_id=str(variant["profile_id"]),
                            max_fps=(
                                None if preserve_source_timing else
                                variant["max_fps"] or final_fps_cap(verified_info, args.preset_id) or max_fps
                            ),
                            duration_s=float(verified_info.get("duration") or info.get("duration") or 0) or None,
                            percent_start=variant_start,
                            percent_end=variant_end,
                            receipt_path=temporary_videoseal_receipt_path(writer.temp_dir, final),
                            **({"frame_rate": variant["frame_rate"]} if "frame_rate" in variant and not preserve_source_timing else {}),
                            **({"original_info": info} if "frame_rate" in variant else {}),
                            **({"preserve_source_timing": True} if preserve_source_timing else {}),
                        )
                    ok, final_info, reason = verify_video(
                        final, ffprobe, inspect_timestamps="frame_rate" in variant,
                    )
                    if not ok:
                        raise RuntimeError(f"Final MP4 failed verification for {variant['profile_id']}: {reason}")
                    final_info["size_bytes"] = final.stat().st_size
                    validate_final_delivery(
                        final_info,
                        expected_width=variant["width"],
                        expected_height=variant["height"],
                        preset_id=args.preset_id,
                        **({"frame_rate": variant["frame_rate"]} if "frame_rate" in variant else {}),
                        **({"source_info": info} if "frame_rate" in variant else {}),
                        **({"source_timing_max_fps": int(variant["max_fps"])} if preserve_source_timing else {}),
                    )
                    if variant.get("geometry_policy") in {"source_display", "custom_final"}:
                        validate_source_display_square_pixels(final_info, phase="Final Smart Finish output")
                    validate_intermediate_audio(
                        info,
                        final_info,
                        phase=f"Final delivery {variant['profile_id']}",
                        require_sample_extent="frame_rate" in variant,
                    )
                    if frame_rate is None and float(final_info.get("fps") or 0) > MAX_OUTPUT_FPS + 0.1:
                        raise RuntimeError(f"Final output FPS exceeded {MAX_OUTPUT_FPS}: {final}")
                    if bool(variant["watermark"]):
                        watermark_result = publish_delivery_watermark_receipt(final, variant, watermark_result)
                    if delivery_completion_expected is not None:
                        completion_row = build_delivery_completion_row(
                            output_path=final,
                            expected=delivery_completion_expected,
                            probe=final_info,
                            watermark_result=watermark_result,
                        )
                        append_delivery_completion(manifest, completion_row)
                        source_delivery_completion_fingerprints.append(
                            str(completion_row["fingerprint"])
                        )
                    if watermark_result:
                        watermark_results.append(watermark_result)
                    manifest["final_outputs"] = list(final_outputs) + [str(final)]
                    manifest["watermark_results"] = list(watermark_results)
                    writer.write_manifest(manifest)
                    variant_probes.append(
                        {
                            "profile_id": variant["profile_id"],
                            "profile_ids": list(variant.get("profile_ids") or [variant["profile_id"]]),
                            "planned_output_paths": list(variant.get("planned_output_paths") or [str(final)]),
                            "output": str(final),
                            "probe": final_info,
                        }
                    )
                    final_outputs.append(str(final))
                    writer.log(f"Final output: {final} {final_info}")
                source_identity_at_completion = sampled_file_identity(source)
                if _content_identity(source_identity_at_completion) != _content_identity(
                    source_identity_at_start
                ):
                    raise RuntimeError(
                        "Source content changed before all delivery variants completed. "
                        "Generated outputs remain in the review folder, but this source is not "
                        "marked complete and must be rerun from one stable file."
                    )
                completed_sources = [
                    row
                    for row in manifest.get("completed_sources") or []
                    if isinstance(row, dict)
                    and _path_key(row.get("selected_source_path") or row.get("source_path") or "")
                    != _path_key(selected_source)
                ]
                completed_sources.append(
                    {
                        "schema": SOURCE_COMPLETION_SCHEMA,
                        "selected_source_path": str(selected_source),
                        "source_path": str(source),
                        "source_identity": source_identity_at_start,
                        **({"working_geometry": working_geometry} if working_geometry is not None else {}),
                        "delivery_output_count": len(variant_probes),
                        "delivery_outputs": [
                            sampled_file_identity(Path(row["output"]))
                            for row in variant_probes
                        ],
                        "delivery_completion_fingerprints": list(
                            source_delivery_completion_fingerprints
                        ),
                    }
                )
                manifest["completed_sources"] = completed_sources
                manifest["final_outputs"] = list(final_outputs)
                manifest["watermark_results"] = list(watermark_results)
                manifest["process_timings"] = list(getattr(writer, "process_timings", []))
                writer.write_manifest(manifest)
                writer.update(
                    percent=round(idx / len(files) * 100, 1),
                    phase="complete_file",
                    message=f"Finished {source.name}",
                    output_folder=str(Path(variants[0]["output_path"]).parent),
                    metrics={"delivery_variant_probes": variant_probes},
                )

                if not args.debug:
                    for temp in sorted(cleanup_candidates, key=lambda path: str(path)):
                        try:
                            if temp.exists():
                                temp.unlink()
                        except Exception as exc:
                            writer.log(f"Temp cleanup failed for {temp}: {exc}")

        require_carried_completed_sources_current(
            resume_manifest,
            expected_state_keys=carried_state_keys,
        )
        cleanup_results["temp_cleanup"] = "kept" if args.debug else "ok"
        manifest["final_outputs"] = final_outputs
        manifest["errors"] = errors
        manifest["cleanup_results"] = cleanup_results
        manifest["watermark_results"] = watermark_results
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 2)
        manifest["process_timings"] = list(getattr(writer, "process_timings", []))
        writer.write_manifest(manifest)
        writer.done("Video upscale complete")
        return 0
    except (JobCancelled, GpuLockCancelled):
        writer.cancelled()
        return 2
    except Exception as exc:
        errors.append(str(exc))
        writer.error(str(exc), "error")
        return 1
    finally:
        if manifest:
            # Retain actual failed/cancelled attempt cost as well as successes.
            # This is not publication/completion authority.
            try:
                from delivery_insights import resource_snapshot
                manifest["elapsed_seconds"] = round(time.monotonic() - started, 4)
                manifest["process_timings"] = list(getattr(writer, "process_timings", []))
                manifest["resource_end"] = resource_snapshot()
                manifest["errors"] = list(errors)
                writer.write_manifest(manifest)
            except (OSError, RuntimeError, ValueError) as diagnostic_error:
                writer.log(f"Final diagnostic snapshot unavailable: {diagnostic_error}")


if __name__ == "__main__":
    raise SystemExit(main())
