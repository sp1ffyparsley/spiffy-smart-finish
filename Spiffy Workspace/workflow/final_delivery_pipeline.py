"""Verified post-export delivery for Premiere and other Spiffy video owners.

Premiere remains responsible for rendering Premiere-only timeline effects. This
module starts from that finished render, runs the existing Topaz worker once,
and fans the verified high-resolution intermediate into only the distinct
physical delivery files requested by the caller.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SS_SCRIPTS_ROOT = WORKSPACE_ROOT.parent
UPSCALE_WORKER = SS_SCRIPTS_ROOT / "tools" / "video_upscale" / "video_upscale.py"
DEFAULT_JOBS_ROOT = SS_SCRIPTS_ROOT / "_jobs"
PLAN_SCHEMA = "spiffy_final_delivery_plan_v1"
REQUEST_SCHEMA = "spiffy_final_delivery_request_v1"
RESULT_SCHEMA = "spiffy_final_delivery_result_v1"
ATTEMPT_EVIDENCE_SCHEMA = "spiffy_final_delivery_attempt_evidence_v1"

if str(SS_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SS_SCRIPTS_ROOT))
VIDEO_UPSCALE_ROOT = SS_SCRIPTS_ROOT / "tools" / "video_upscale"
if str(VIDEO_UPSCALE_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEO_UPSCALE_ROOT))

from shared.job_paths import ensure_job_dirs, make_job_id  # noqa: E402
from shared.progress_writer import ProgressWriter, atomic_write_json, atomic_write_text  # noqa: E402
from adaptive_enhancement import normalize_policy as normalize_adaptive_policy  # noqa: E402
from adaptive_enhancement import engine_identity as adaptive_engine_identity  # noqa: E402
from video_upscale import (  # noqa: E402
    delivery_enhancement_recipe,
    load_delivery_plan,
    resolve_enhancement_recipe,
    normalize_frame_rate,
    normalize_delivery_geometry,
    resolve_custom_final_geometry,
    resolve_delivery_working_geometry,
    validate_requested_frame_rate,
    validate_delivery_watermark_receipt,
    validated_completed_source_states,
)

from .atomic_io import artifact_publication_lease  # noqa: E402
from .channel_style_profiles import resolve_channel_style_profile  # noqa: E402
from .social_export_profiles import get_social_export_profile  # noqa: E402
from .video_provenance_watermark import inspect_signed_video_receipt, normalize_namespace  # noqa: E402
from .workflow_jobs import interprocess_lane_lease  # noqa: E402
from .workflow_paths import platform_name  # noqa: E402


BRAND_NAMESPACES: dict[str, str] = {
    "spiffy": "spiffy.default",
    "spiffy-scene": "spiffy.scenes",
    "spiffy-scenes": "spiffy.scenes",
    "spiffyscenes": "spiffy.scenes",
    "spiffy-clips": "spiffy.clips",
    "spiffy-clips-hq": "spiffy.clips",
    "spiffy-luna": "spiffy.luna",
    "spiffy-mma": "spiffy.mma",
    "spiffy-parsley": "spiffy.parsley",
}

CHANNEL_BRAND_NAMESPACES: dict[str, str] = {
    "spiffyscenes": "spiffy.scenes",
    "spiffy_clips_hq": "spiffy.clips",
    "lunas_music": "spiffy.luna",
    "spiffymma": "spiffy.mma",
    "spiffy_parsley": "spiffy.parsley",
    "lune_kr": "spiffy.lune.kr",
}

REJECTED_BRAND_ALIASES = {"chatgpt", "spiffy-chatgpt"}

DELIVERY_SETS: dict[str, tuple[str, ...]] = {
    "canonical": ("native_master",),
    "smart_finish": ("smart_finish_master",),
    "short_form": ("native_master", "spiffy_4x5", "generic_vertical_short"),
    "all_social": (
        "native_master",
        "spiffy_4x5",
        "x_portrait_4x5",
        "generic_vertical_short",
        "tiktok_vertical",
        "instagram_reels",
        "facebook_reels",
        "snapchat_story",
        "youtube_shorts",
        "youtube_horizontal",
        "dailymotion_horizontal",
    ),
}

# Quality targets, not fixed bitrates. The existing CRF 17 / slow export remains
# unchanged unless the user explicitly selects another compression profile.
COMPRESSION_PROFILES: dict[str, tuple[int, str]] = {
    "existing": (17, "slow"),
    "maximum": (14, "slow"),
    "quality": (16, "slow"),
    "balanced": (18, "slow"),
    "compact": (20, "slow"),
}


def normalize_compression_profile(value: Any) -> str:
    if type(value) is not str or value not in COMPRESSION_PROFILES:
        raise ValueError("compression_profile must be existing, maximum, quality, balanced, or compact.")
    return value

WATERMARK_VIDEO_TYPE_PRESETS: dict[str, str] = {
    "native_master": "native-master",
    "smart_finish_master": "native-master",
    "spiffy_4x5": "portrait-4x5",
    "x_portrait_4x5": "portrait-4x5",
    "generic_vertical_short": "vertical-short-9x16",
    "tiktok_vertical": "vertical-short-9x16",
    "instagram_reels": "vertical-short-9x16",
    "facebook_reels": "vertical-short-9x16",
    "snapchat_story": "vertical-short-9x16",
    "youtube_shorts": "vertical-short-9x16",
    "youtube_horizontal": "horizontal-16x9",
    "dailymotion_horizontal": "horizontal-16x9",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_brand_namespace(brand: str) -> str:
    """Resolve a channel style identity or a safe custom VideoSeal namespace."""

    key = str(brand or "spiffy-scenes").strip().lower().replace("_", "-").replace(" ", "-")
    if key in REJECTED_BRAND_ALIASES:
        raise ValueError("ChatGPT is not a Spiffy delivery brand. Choose a channel style identity.")
    if key in BRAND_NAMESPACES:
        return BRAND_NAMESPACES[key]
    resolution = resolve_channel_style_profile(str(brand or "spiffy-scenes"))
    channel_id = str(resolution.get("channel_id") or "")
    if channel_id in CHANNEL_BRAND_NAMESPACES and resolution.get("status") == "ready":
        return CHANNEL_BRAND_NAMESPACES[channel_id]
    namespace = normalize_namespace(key)
    if not namespace:
        raise ValueError("Delivery brand cannot be empty.")
    return namespace if "." in namespace else f"spiffy.{namespace}"


def profiles_for_delivery_set(delivery_set: str) -> list[str]:
    key = str(delivery_set or "canonical").strip().lower().replace("-", "_")
    if key not in DELIVERY_SETS:
        raise ValueError(f"Unknown delivery set: {delivery_set}")
    return list(DELIVERY_SETS[key])


def normalize_profile_ids(profile_ids: Iterable[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for raw in profile_ids:
        value = str(raw or "").strip().lower().replace("-", "_")
        if value and value not in seen:
            get_social_export_profile(value)
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("At least one delivery profile is required.")
    return values


def watermark_identifier_for_profile(profile_id: str) -> str:
    key = str(profile_id or "").strip().lower().replace("-", "_")
    return WATERMARK_VIDEO_TYPE_PRESETS.get(key, key.replace("_", "-"))


def _physical_name(width: int, height: int, layout: str) -> str:
    if not width or not height:
        return "native_master"
    orientation = "vertical" if height > width else "landscape" if width > height else "square"
    suffix = "_crop" if layout == "fill_crop_center" else ""
    if (width, height) == (1080, 1350):
        orientation = "portrait"
    return f"{orientation}_{width}x{height}{suffix}"


def _layout_for_profile(profile: dict[str, Any], layout_policy: str) -> str:
    if not int(profile.get("expected_width") or 0) or not int(profile.get("expected_height") or 0):
        render = dict(profile.get("delivery_render") or {})
        layout = str(render.get("layout") or "keep_source_aspect")
        if layout not in {"keep_source_aspect", "fit_into_frame", "fill_crop_center"}:
            raise ValueError(f"Unknown delivery render layout for {profile.get('profile_id')}: {layout}")
        return layout
    policy = str(layout_policy or "fit_black").strip().lower()
    if policy == "fit_black":
        return "fit_into_frame"
    if policy == "fill_crop":
        return "fill_crop_center"
    raise ValueError(f"Unknown layout policy: {layout_policy}. Use fit_black or fill_crop.")


def build_final_delivery_plan(
    source: str | Path,
    output_root: str | Path,
    *,
    profile_ids: Iterable[str],
    brand: str = "spiffy-scenes",
    layout_policy: str = "fit_black",
    watermark: bool = False,
    watermark_strength: float = 0.2,
    enhancement: dict[str, Any] | None = None,
    frame_rate: dict[str, Any] | None = None,
    adaptive_enhancement: dict[str, Any] | None = None,
    geometry: dict[str, Any] | None = None,
    compression_profile: str = "existing",
) -> dict[str, Any]:
    """Build one non-destructive, physically deduplicated delivery plan."""

    recipe = resolve_enhancement_recipe(enhancement) if enhancement is not None else None
    adaptive = normalize_adaptive_policy(adaptive_enhancement) if adaptive_enhancement is not None else None
    geometry_contract = normalize_delivery_geometry(geometry) if geometry is not None else None
    compression_profile = normalize_compression_profile(compression_profile)
    if adaptive is not None and (watermark or (recipe is not None and recipe.up_model != "prob-4")
                                 or (enhancement or {}).get("overrides")):
        raise ValueError("Adaptive natural requires local Proteus, watermark off, and no conflicting manual overrides.")
    cadence = normalize_frame_rate(frame_rate) if frame_rate is not None else None
    if cadence is not None and watermark:
        raise ValueError("Explicit frame_rate with VideoSeal is not yet qualified; use watermark off or the legacy route.")
    source_path = Path(source).expanduser().resolve()
    output_dir = Path(output_root).expanduser().resolve()
    namespace = resolve_brand_namespace(brand)
    requested = normalize_profile_ids(profile_ids)
    if geometry_contract is not None and "smart_finish_master" not in requested:
        raise ValueError("Custom working/final geometry requires the smart_finish_master output profile.")
    if geometry_contract is not None and geometry_contract.get("working") is not None and adaptive is not None:
        raise ValueError("Custom working geometry cannot be combined with adaptive enhancement until its crop owner supports it.")
    if geometry_contract is not None and geometry_contract.get("working") is not None and "native_master" in requested:
        raise ValueError("Custom working geometry cannot change the isolated legacy native_master path.")
    if not 0.01 <= float(watermark_strength) <= 1.0:
        raise ValueError("VideoSeal watermark strength must be between 0.01 and 1.0.")

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    profile_output_map: dict[str, str] = {}
    for profile_id in requested:
        profile = get_social_export_profile(profile_id)
        render = dict(profile.get("delivery_render") or {})
        width = int(profile.get("expected_width") or 0)
        height = int(profile.get("expected_height") or 0)
        layout = _layout_for_profile(profile, layout_policy)
        crf = int(render.get("crf") or 17)
        preset = str(render.get("preset") or "slow")
        if compression_profile != "existing":
            if profile_id != "smart_finish_master" or watermark:
                raise ValueError("Custom compression profiles require an unwatermarked Smart Finish master.")
            crf, preset = COMPRESSION_PROFILES[compression_profile]
        profile_max_fps = int(float(profile.get("max_fps") or 60))
        # Raising local-master capability must not change older saved requests.
        max_fps = min(profile_max_fps, 60) if cadence is None else profile_max_fps
        variant_rate = None
        frame_rate_reason = ""
        if cadence is not None:
            if cadence["mode"] == "preserve_source":
                if profile.get("supports_source_timing") is not True:
                    raise ValueError(
                        f"Profile {profile_id} does not declare exact source-PTS preservation support; "
                        "select native_master or smart_finish_master instead of silently resampling timing."
                    )
                variant_rate = cadence
                max_fps = profile_max_fps
                frame_rate_reason = "source_preserved"
            else:
                effective = min(Fraction(cadence["fps"]), Fraction(profile_max_fps))
                variant_rate = normalize_frame_rate({"mode": "cfr", "fps": str(effective)})
                max_fps = math.ceil(effective)
                frame_rate_reason = "requested" if variant_rate == cadence else "profile_ceiling"
        watermark_identifier = watermark_identifier_for_profile(profile_id)
        geometry_policy = str(render.get("geometry_policy") or "")
        custom_final = (
            geometry_contract.get("final")
            if geometry_contract is not None and profile_id == "smart_finish_master"
            else None
        )
        if isinstance(custom_final, dict) and custom_final.get("mode") in {"exact", "long_edge", "preset"}:
            width = height = 0
            layout = "fit_into_frame"
            geometry_policy = "custom_final"
            if custom_final["mode"] == "exact":
                physical_name = f"smart_finish_{custom_final['width']}x{custom_final['height']}"
            elif custom_final["mode"] == "preset":
                physical_name = f"smart_finish_{custom_final['preset']}_{custom_final['long_edge']}"
            else:
                physical_name = f"smart_finish_long_{custom_final['long_edge']}"
        else:
            physical_name = str(render.get("physical_name") or _physical_name(width, height, layout))
        signature = (
            width, height, layout, crf, preset, max_fps, bool(watermark), namespace,
            watermark_identifier, geometry_policy, physical_name,
            json.dumps(variant_rate, sort_keys=True, separators=(",", ":")) if variant_rate is not None else None,
            json.dumps(custom_final, sort_keys=True, separators=(",", ":")) if custom_final is not None else None,
        )
        if signature not in groups:
            output = output_dir / f"{source_path.stem}__{physical_name}.mp4"
            if os.path.normcase(str(output)) == os.path.normcase(str(source_path)):
                raise ValueError(f"Delivery output would overwrite its source: {source_path}")
            groups[signature] = {
                "profile_id": profile_id,
                "profile_ids": [],
                "physical_name": physical_name,
                "output_path": str(output),
                "width": width,
                "height": height,
                "layout": layout,
                "background": "black" if layout == "fit_into_frame" else "source",
                "crf": crf,
                "x264_preset": preset,
                "max_fps": max_fps,
                "watermark": bool(watermark),
                "watermark_strength": float(watermark_strength),
                "namespace": namespace,
                "watermark_identifier": watermark_identifier,
                **({"geometry_policy": geometry_policy} if geometry_policy else {}),
                **({"geometry": dict(custom_final)} if custom_final is not None and custom_final.get("mode") in {"exact", "long_edge", "preset"} else {}),
                **({"frame_rate": variant_rate, "frame_rate_reason":
                    frame_rate_reason} if variant_rate is not None else {}),
            }
        groups[signature]["profile_ids"].append(profile_id)
        profile_output_map[profile_id] = groups[signature]["output_path"]

    variants = list(groups.values())
    if cadence is not None and cadence["mode"] == "cfr" and not any(row.get("frame_rate") == cadence for row in variants):
        raise ValueError("Requested frame_rate exceeds every selected profile. Include native_master or smart_finish_master to retain that cadence.")
    return {
        "schema": "spiffy_final_delivery_plan_v4" if geometry_contract is not None else "spiffy_final_delivery_plan_v3" if adaptive is not None else "spiffy_final_delivery_plan_v2" if cadence is not None else PLAN_SCHEMA,
        **({"adaptive_enhancement": adaptive} if adaptive is not None else {}),
        **({"adaptive_engine_identity": adaptive_engine_identity()} if adaptive is not None else {}),
        **({"geometry": geometry_contract} if geometry_contract is not None else {}),
        "created_at_utc": utc_now(),
        "source_path": str(source_path),
        "output_root": str(output_dir),
        "brand": str(brand),
        "brand_namespace": namespace,
        "layout_policy": layout_policy,
        **({"compression_profile": compression_profile} if compression_profile != "existing" else {}),
        "enhance_preset": recipe.preset_id if recipe is not None else "4k_60fps",
        **({"frame_rate": cadence} if cadence is not None else {}),
        **({"frame_rate_boundary": (
            "source_pts_preserve_v1" if cadence["mode"] == "preserve_source" else "source_end_context_v1"
        )} if cadence is not None else {}),
        **({"enhancement": {"preset_id": recipe.preset_id, "overrides": dict(enhancement.get("overrides", {}))},
            "enhancement_recipe": recipe.as_dict()} if recipe is not None else {}),
        "requested_profile_ids": requested,
        "profile_output_map": profile_output_map,
        "physical_output_count": len(variants),
        "variants": variants,
        "policy": {
            "preserve_source": True,
            "never_stretch": True,
            "topaz_passes": 1,
            "lossy_encodes_per_output": 1,
            "delete_verified_intermediate_only_after_all_outputs_pass": True,
        },
    }


def request_sources(payload: dict[str, Any]) -> list[Path]:
    """Read new requests and legacy AutoExport v2 manifests without a second schema owner."""

    candidates: list[Any] = []
    for key in ("expected_files", "source_files", "inputs"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    exports = payload.get("exports")
    if isinstance(exports, list):
        for row in exports:
            if isinstance(row, dict):
                candidates.append(row.get("path") or row.get("output_path"))
    paths: list[Path] = []
    seen: set[str] = set()
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw)).expanduser().resolve()
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            paths.append(path)
    if not paths:
        raise ValueError("Final-delivery request contains no expected video files.")
    return paths


def request_delivery_options(payload: dict[str, Any]) -> dict[str, Any]:
    delivery = payload.get("final_delivery", {})
    if not isinstance(delivery, dict):
        raise ValueError("final_delivery must be an object (including enhancement settings).")
    if "enhancement" in delivery:
        resolve_enhancement_recipe(delivery["enhancement"])
    if "frame_rate" in delivery:
        normalize_frame_rate(delivery["frame_rate"])
    if "adaptive_enhancement" in delivery:
        normalize_adaptive_policy(delivery["adaptive_enhancement"])
    if "geometry" in delivery:
        normalize_delivery_geometry(delivery["geometry"])
    if "compression_profile" in delivery:
        normalize_compression_profile(delivery["compression_profile"])
    profiles = delivery.get("profile_ids") or delivery.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        delivery_set = str(delivery.get("delivery_set") or "canonical")
        profiles = profiles_for_delivery_set(delivery_set)
    return {
        "profile_ids": normalize_profile_ids(profiles),
        "brand": str(delivery.get("brand") or "spiffy-scenes"),
        "layout_policy": str(delivery.get("layout_policy") or "fit_black"),
        "watermark": bool(delivery.get("watermark", False)),
        "watermark_strength": float(delivery.get("watermark_strength") or 0.2),
        "output_root": str(delivery.get("output_root") or ""),
        **({"adaptive_enhancement": normalize_adaptive_policy(delivery["adaptive_enhancement"])} if "adaptive_enhancement" in delivery else {}),
        **({"geometry": normalize_delivery_geometry(delivery["geometry"])} if "geometry" in delivery else {}),
        **({"compression_profile": normalize_compression_profile(delivery["compression_profile"])}
           if "compression_profile" in delivery else {}),
        **({"frame_rate": normalize_frame_rate(delivery["frame_rate"])} if "frame_rate" in delivery else {}),
        **({"enhancement": {"preset_id": delivery["enhancement"]["preset_id"],
                           "overrides": dict(delivery["enhancement"].get("overrides", {}))}}
           if "enhancement" in delivery else {}),
    }


def build_worker_command(
    *,
    jobs_root: Path,
    child_job_id: str,
    input_list: Path,
    plan_path: Path,
    resume_job_id: str = "",
) -> list[str]:
    plan = load_delivery_plan(plan_path) if plan_path.exists() else {}
    recipe = delivery_enhancement_recipe(plan, "4k_60fps")
    command = [
        sys.executable,
        str(UPSCALE_WORKER),
        "--jobs-root",
        str(jobs_root),
        "--job-id",
        child_job_id,
        "--input-list",
        str(input_list),
        "--preset-id",
        recipe.preset_id,
        "--layout",
        "keep_source_aspect",
        "--delivery-plan",
        str(plan_path),
        "--watermark",
        "off",
    ]
    if resume_job_id:
        # Reuse ONLY source/recipe/model/PTS-matched checkpoints proven by
        # the previous child. The worker revalidates every imported artifact;
        # an invalid one is rerendered rather than considered complete.
        command.extend(["--resume-job-id", resume_job_id, "--reconcile-completed-sources"])
    return command


def _path_key(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).expanduser().resolve(strict=False)))


def _stable_file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    if int(before.st_size) != int(after.st_size) or int(before.st_mtime_ns) != int(after.st_mtime_ns):
        raise OSError(f"File changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
        "mtime_ns": int(after.st_mtime_ns),
    }


def _same_content_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        bool(left)
        and bool(right)
        and int(left.get("bytes") or -1) == int(right.get("bytes") or -2)
        and str(left.get("sha256") or "") == str(right.get("sha256") or "")
    )


def _plan_binding(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        **({"adaptive_enhancement": normalize_adaptive_policy(plan["adaptive_enhancement"])} if "adaptive_enhancement" in plan else {}),
        **({"adaptive_engine_identity": plan.get("adaptive_engine_identity")} if "adaptive_enhancement" in plan else {}),
        **({"frame_rate": normalize_frame_rate(plan["frame_rate"])} if "frame_rate" in plan else {}),
        **({"frame_rate_boundary": plan.get("frame_rate_boundary")} if "frame_rate" in plan else {}),
        **({"geometry": normalize_delivery_geometry(plan["geometry"])} if "geometry" in plan else {}),
        **({"enhancement_recipe": delivery_enhancement_recipe(plan, "4k_60fps").as_dict()}
           if "enhancement" in plan or "enhancement_recipe" in plan else {}),
        "schema": str(plan.get("schema") or ""),
        "source_path": str(Path(str(plan.get("source_path") or "")).expanduser().resolve(strict=False)),
        "output_root": str(Path(str(plan.get("output_root") or "")).expanduser().resolve(strict=False)),
        "brand": str(plan.get("brand") or ""),
        "brand_namespace": str(plan.get("brand_namespace") or ""),
        "layout_policy": str(plan.get("layout_policy") or ""),
        "enhance_preset": str(plan.get("enhance_preset") or ""),
        "requested_profile_ids": list(plan.get("requested_profile_ids") or []),
        "profile_output_map": dict(plan.get("profile_output_map") or {}),
        "physical_output_count": int(plan.get("physical_output_count") or 0),
        "variants": [dict(row) for row in plan.get("variants") or [] if isinstance(row, dict)],
        "policy": dict(plan.get("policy") or {}),
    }


def _plan_fingerprint(plan: dict[str, Any]) -> str:
    payload = json.dumps(_plan_binding(plan), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _delivery_target_lane(plan: dict[str, Any]) -> str:
    output_root = _path_key(str(plan.get("output_root") or ""))
    digest = hashlib.sha256(output_root.encode("utf-8")).hexdigest()[:24]
    return f"final_delivery_{digest}"


def _variant_signature(
    row: dict[str, Any],
    *,
    resolved_dimensions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_id = str(row.get("profile_id") or "")
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    if resolved_dimensions is not None and width == 0 and height == 0:
        width = int(resolved_dimensions.get("width") or 0)
        height = int(resolved_dimensions.get("height") or 0)
    signature = {
        "profile_id": profile_id,
        "profile_ids": [str(value) for value in row.get("profile_ids") or [profile_id]],
        "width": width,
        "height": height,
        "layout": str(row.get("layout") or ""),
        "crf": int(row.get("crf") or 0),
        "x264_preset": str(row.get("x264_preset") or ""),
        "watermark": bool(row.get("watermark")),
        "watermark_strength": float(row.get("watermark_strength") or 0.0),
        "namespace": str(row.get("namespace") or ""),
        "watermark_identifier": str(row.get("watermark_identifier") or ""),
        "max_fps": int(row.get("max_fps")) if row.get("max_fps") is not None else None,
    }
    if row.get("geometry_policy"):
        signature["geometry_policy"] = str(row.get("geometry_policy"))
    for field in ("geometry", "source_geometry", "working_geometry"):
        value = row.get(field)
        if value is None and resolved_dimensions is not None:
            value = resolved_dimensions.get(field)
        if isinstance(value, dict):
            signature[field] = dict(value)
    if "frame_rate" in row:
        signature["frame_rate"] = normalize_frame_rate(row["frame_rate"])
    return signature


def _validated_child_delivery(
    child_manifest: dict[str, Any],
    *,
    child_manifest_path: Path,
    source: Path,
    source_identity: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    if not child_manifest:
        issues.append("child_manifest_missing_or_malformed")
    if str(child_manifest.get("worker") or "") != "video_upscale":
        issues.append("child_worker_identity_invalid")
    if child_manifest.get("errors"):
        issues.append("child_manifest_reports_errors")
    child_plan = child_manifest.get("delivery_plan") if isinstance(child_manifest.get("delivery_plan"), dict) else {}
    if not child_plan or _plan_fingerprint(child_plan) != _plan_fingerprint(plan):
        issues.append("child_delivery_plan_binding_mismatch")
    recipe = delivery_enhancement_recipe(plan, "4k_60fps")
    if child_manifest.get("enhancement_recipe") != (recipe.as_dict() if recipe.explicit else None):
        issues.append("child_enhancement_recipe_binding_mismatch")
    if child_manifest.get("frame_rate") != plan.get("frame_rate"):
        issues.append("child_frame_rate_binding_mismatch")
    if child_manifest.get("adaptive_enhancement") != plan.get("adaptive_enhancement"):
        issues.append("child_adaptive_policy_binding_mismatch")
    if child_plan.get("geometry") != plan.get("geometry"):
        issues.append("child_geometry_contract_binding_mismatch")
    try:
        current_source = _stable_file_identity(source)
    except OSError:
        current_source = {}
        issues.append("source_missing_or_unstable")
    if current_source and not _same_content_identity(source_identity, current_source):
        issues.append("source_content_changed")

    validation = validated_completed_source_states(child_manifest) if not issues else {
        "states": [],
        "stale_or_unproven_completed_sources": [],
        "completed_source_row_count": 0,
    }
    states = [row for row in validation.get("states") or [] if isinstance(row, dict)]
    source_key = _path_key(source)
    matching = [row for row in states if str(row.get("matched_source_key") or "") == source_key]
    if len(matching) != 1:
        issues.append("child_completed_source_missing_invalid_or_ambiguous")
        matching = []
    if int(validation.get("completed_source_row_count") or 0) != 1:
        issues.append("child_completed_source_count_invalid")

    geometry_ready_source: dict[str, Any] = {}
    expected_working_geometry: dict[str, Any] | None = None
    if "geometry" in plan and matching:
        ready_sources = [
            row for row in child_manifest.get("ready_sources") or []
            if isinstance(row, dict)
            and source_key in {
                _path_key(row.get("selected_source_path") or "") if row.get("selected_source_path") else "",
                _path_key(row.get("source_path") or "") if row.get("source_path") else "",
            }
        ]
        if len(ready_sources) != 1 or not isinstance(ready_sources[0].get("probe"), dict):
            issues.append("child_working_geometry_source_probe_missing_or_ambiguous")
        else:
            geometry_ready_source = ready_sources[0]
            try:
                expected_working_geometry = resolve_delivery_working_geometry(
                    dict(geometry_ready_source["probe"]),
                    str(plan.get("enhance_preset") or "4k_60fps"),
                    "keep_source_aspect",
                    plan,
                )
                if geometry_ready_source.get("working_geometry") != expected_working_geometry:
                    issues.append("child_working_geometry_proof_invalid")
                completed_rows = [
                    row for row in child_manifest.get("completed_sources") or []
                    if isinstance(row, dict)
                    and source_key in {
                        _path_key(row.get("selected_source_path") or "") if row.get("selected_source_path") else "",
                        _path_key(row.get("source_path") or "") if row.get("source_path") else "",
                    }
                ]
                if len(completed_rows) != 1 or completed_rows[0].get("working_geometry") != expected_working_geometry:
                    issues.append("child_completed_working_geometry_binding_mismatch")
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                issues.append("child_working_geometry_proof_invalid")

    state = matching[0] if matching else {}
    resolved = state.get("resolved_source") if isinstance(state.get("resolved_source"), dict) else {}
    variants = [row for row in resolved.get("variants") or [] if isinstance(row, dict)]
    # Only plans may use unresolved sentinels. Completed child geometry must be
    # positive integer pixels before it can normalize a source-native plan.
    if matching and any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for variant in variants
        for value in (variant.get("width"), variant.get("height"))
    ):
        issues.append("child_delivery_variant_dimensions_invalid")
        matching = []

    output_proofs: list[dict[str, Any]] = []
    resolved_profile_map: dict[str, str] = {}
    resolved_count = 0
    if matching:
        resolved_by_profile = {
            str(row.get("profile_id") or ""): row
            for row in variants
            if str(row.get("profile_id") or "")
        }
        planned_signatures = sorted(
            json.dumps(
                _variant_signature(
                    row,
                    resolved_dimensions=resolved_by_profile.get(str(row.get("profile_id") or "")),
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in plan.get("variants") or []
            if isinstance(row, dict)
        )
        resolved_signatures = sorted(
            json.dumps(_variant_signature(row), sort_keys=True, separators=(",", ":"))
            for row in variants
        )
        if planned_signatures != resolved_signatures:
            issues.append("child_delivery_variant_binding_mismatch")
        resolved_profile_map = {
            str(key): str(value)
            for key, value in dict(resolved.get("profile_output_map") or {}).items()
        }
        resolved_count = int(resolved.get("physical_output_count") or 0)
        if set(resolved_profile_map) != set(plan.get("requested_profile_ids") or []):
            issues.append("child_delivery_profile_binding_mismatch")
        if resolved_count != int(plan.get("physical_output_count") or 0):
            issues.append("child_delivery_output_count_mismatch")
        output_paths = [
            str(value)
            for value in state.get("output_paths") or []
            if str(value or "").strip()
        ]
        manifest_outputs = [
            str(value)
            for value in child_manifest.get("final_outputs") or []
            if str(value or "").strip()
        ]
        if (
            len(output_paths) != int(plan.get("physical_output_count") or 0)
            or {_path_key(value) for value in output_paths}
            != {_path_key(value) for value in manifest_outputs}
        ):
            issues.append("child_final_output_membership_mismatch")
        completion_rows = [
            row for row in state.get("completion_rows") or [] if isinstance(row, dict)
        ]
        completion_by_path: dict[str, dict[str, Any]] = {}
        for row in completion_rows:
            stored_output = row.get("output") if isinstance(row.get("output"), dict) else {}
            output_path = str(stored_output.get("path") or "")
            if output_path:
                completion_by_path[_path_key(output_path)] = row
        for output_path in output_paths:
            completion = completion_by_path.get(_path_key(output_path)) or {}
            contract = completion.get("contract") if isinstance(completion.get("contract"), dict) else {}
            variant = contract.get("variant") if isinstance(contract.get("variant"), dict) else {}
            if recipe.explicit and contract.get("enhancement_recipe") != recipe.as_dict():
                issues.append("child_completion_enhancement_recipe_binding_mismatch")
            adaptive_proof = None
            adaptive_quality = None
            if "adaptive_enhancement" in plan:
                from adaptive_enhancement import plan_fingerprint as adaptive_plan_fingerprint, validate_quality_receipt
                adaptive_rows = [row for row in child_manifest.get("adaptive_plans", []) if isinstance(row, dict)
                                 and row.get("source_path") and _path_key(row["source_path"]) == source_key]
                if len(adaptive_rows) == 1 and isinstance(adaptive_rows[0].get("plan"), dict):
                    adaptive_proof = adaptive_rows[0]["plan"]
                if (not adaptive_proof or adaptive_proof.get("policy") != plan["adaptive_enhancement"]
                        or adaptive_proof.get("fingerprint") != adaptive_plan_fingerprint(adaptive_proof)
                        or contract.get("adaptive_fingerprint") != adaptive_proof.get("fingerprint")
                        or adaptive_proof.get("identity", {}).get("source_sha256") != current_source.get("sha256")):
                    issues.append("child_adaptive_analysis_binding_mismatch")
                if adaptive_proof:
                    adaptive_quality = child_manifest.get("adaptive_quality", {}).get(source_key)
                    try:
                        from fractions import Fraction
                        from video_upscale import PRESET_MODEL
                        cadence = plan.get("frame_rate") or {}
                        target_rate = (cadence.get("fps") if cadence.get("mode") == "cfr" else None
                                       if cadence.get("mode") == "preserve_source" else
                                       PRESET_MODEL.get(str(plan.get("enhance_preset") or "4k_60fps"), {}).get("fps"))
                        needs_temporal = bool(target_rate and Fraction(str(target_rate)) > Fraction(adaptive_proof["source_fps"]))
                        validate_quality_receipt(adaptive_quality, adaptive_proof, require_temporal=needs_temporal)
                    except (TypeError, ValueError, RuntimeError):
                        issues.append("child_adaptive_quality_gate_missing_or_failed")
            if "frame_rate" in plan:
                if contract.get("frame_rate") != plan["frame_rate"]:
                    issues.append("child_completion_frame_rate_binding_mismatch")
                try:
                    source_timing_reference = None
                    source_timing_cap = None
                    if plan["frame_rate"].get("mode") == "preserve_source":
                        ready_sources = [
                            row for row in child_manifest.get("ready_sources") or []
                            if isinstance(row, dict)
                            and source_key in {
                                _path_key(row.get("selected_source_path") or "") if row.get("selected_source_path") else "",
                                _path_key(row.get("source_path") or "") if row.get("source_path") else "",
                            }
                        ]
                        if len(ready_sources) != 1 or not isinstance(ready_sources[0].get("probe"), dict):
                            raise RuntimeError("Child source PTS reference is missing or ambiguous.")
                        source_timing_reference = dict(ready_sources[0]["probe"])
                        source_timing_cap = variant.get("max_fps")
                    validate_requested_frame_rate(
                        dict(completion.get("probe") or {}),
                        variant.get("frame_rate"),
                        source_info=source_timing_reference,
                        max_fps=source_timing_cap,
                    )
                except (ValueError, TypeError, RuntimeError):
                    issues.append(
                        "child_output_source_timing_invalid"
                        if plan.get("frame_rate", {}).get("mode") == "preserve_source"
                        else "child_output_frame_rate_invalid"
                    )
            try:
                identity = _stable_file_identity(output_path)
            except OSError:
                issues.append("child_final_output_missing_or_unstable")
                continue
            watermark_receipt: dict[str, Any] = {}
            if bool(variant.get("watermark")) and not issues:
                try:
                    watermark_receipt = validate_delivery_watermark_receipt(
                        Path(output_path), variant, completion.get("watermark_result")
                    )
                except (OSError, TypeError, ValueError, RuntimeError):
                    issues.append("child_watermark_receipt_missing_or_invalid")
                    continue
            output_proofs.append(
                {
                    "path": identity["path"],
                    "sha256": identity["sha256"],
                    "bytes": identity["bytes"],
                    "profile_id": str(variant.get("profile_id") or ""),
                    "profile_ids": [str(value) for value in variant.get("profile_ids") or []],
                    "width": int(variant.get("width") or 0),
                    "height": int(variant.get("height") or 0),
                    "layout": str(variant.get("layout") or ""),
                    **({"geometry_policy": str(variant.get("geometry_policy"))}
                       if variant.get("geometry_policy") else {}),
                    **({"geometry": dict(variant.get("geometry") or {})}
                       if variant.get("geometry") else {}),
                    **({"source_geometry": dict(variant.get("source_geometry") or {})}
                       if variant.get("source_geometry") else {}),
                    **({"working_geometry": dict(variant.get("working_geometry") or {})}
                       if variant.get("working_geometry") else {}),
                    "probe": dict(completion.get("probe") or {}),
                    "completion_fingerprint": str(completion.get("fingerprint") or ""),
                    **({"frame_rate": dict(variant["frame_rate"])} if "frame_rate" in variant else {}),
                    **({"enhancement_recipe": recipe.as_dict()} if recipe.explicit else {}),
                    **({"adaptive_fingerprint": adaptive_proof["fingerprint"],
                        "adaptive_route": adaptive_proof["route"],
                        "adaptive_actual_route": (adaptive_quality or {}).get("actual_route"),
                        "adaptive_quality_status": (adaptive_quality or {}).get("status"),
                        "adaptive_processed_area_ratio": adaptive_proof["processed_area_ratio"]}
                       if adaptive_proof else {}),
                    "watermark_enabled": bool(variant.get("watermark")),
                    **({"watermark_receipt": watermark_receipt} if watermark_receipt else {}),
                }
            )
    if len(output_proofs) != int(plan.get("physical_output_count") or 0):
        issues.append("child_final_output_proof_count_mismatch")
    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": list(dict.fromkeys(issues)),
        "child_manifest_path": str(child_manifest_path),
        "plan_fingerprint": _plan_fingerprint(plan),
        "source_identity": current_source,
        "outputs": [row["path"] for row in output_proofs],
        "final_output_proofs": output_proofs,
        "profile_output_map": resolved_profile_map,
        "physical_output_count": resolved_count,
    }


def _load_json_if_present(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _attempt_evidence_path(parent_dir: Path, attempt_id: str) -> Path:
    return parent_dir / "attempts" / f"{attempt_id}.json"


def _write_attempt_evidence(parent_dir: Path, payload: dict[str, Any]) -> None:
    path = _attempt_evidence_path(parent_dir, str(payload.get("attempt_id") or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _reusable_previous_child(
    parent_dir: Path,
    previous_result: dict[str, Any],
    *,
    source_index: int,
    source: Path,
    source_identity: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any] | None:
    previous_attempt = str(previous_result.get("attempt_id") or "").strip()
    if not previous_attempt:
        return None
    evidence = _load_json_if_present(_attempt_evidence_path(parent_dir, previous_attempt))
    if (
        evidence.get("schema") != ATTEMPT_EVIDENCE_SCHEMA
        or str(evidence.get("attempt_id") or "") != previous_attempt
    ):
        return None
    rows = [
        row
        for row in evidence.get("sources") or []
        if isinstance(row, dict) and int(row.get("source_index") or 0) == source_index
    ]
    recovered_child_id = ""
    if len(rows) == 1:
        row = rows[0]
        child_manifest_path = Path(str(row.get("child_manifest_path") or "")).expanduser()
    else:
        # Parent can die AFTER the child finished and wrote its verified
        # media but BEFORE the parent appended attempt-evidence. Its existing
        # running result still contains the child ID/plan/source. Revalidate
        # the child's entire delivery contract; an exit code or file alone
        # does not count as completed.
        abandoned = (
            previous_result.get("schema") == RESULT_SCHEMA
            and previous_result.get("state") == "running"
            and isinstance(previous_result.get("attempt_owner"), dict)
            and _attempt_process_active(previous_result["attempt_owner"]) is False
        )
        if not abandoned:
            return None
        prior = [
            item for item in previous_result.get("sources") or []
            if isinstance(item, dict)
            and int(item.get("source_index") or 0) == source_index
        ]
        if len(prior) != 1:
            return None
        row = prior[0]
        recovered_child_id = str(row.get("child_job_id") or "")
        if (not recovered_child_id
                or not all(char.isascii() and (char.isalnum() or char in "-_")
                           for char in recovered_child_id)
                or _child_process_active(recovered_child_id, parent_dir.parent) is not False):
            return None
        child_dir = parent_dir.parent / recovered_child_id
        if child_dir.is_symlink() or not child_dir.is_dir():
            return None
        child_manifest_path = child_dir / "manifest.json"
        if child_manifest_path.is_symlink():
            return None
    if str(row.get("plan_fingerprint") or "") != _plan_fingerprint(plan):
        return None
    if not _same_content_identity(dict(row.get("source_identity") or {}), source_identity):
        return None
    child_manifest = _load_json_if_present(child_manifest_path)
    validation = _validated_child_delivery(
        child_manifest,
        child_manifest_path=child_manifest_path,
        source=source,
        source_identity=source_identity,
        plan=plan,
    )
    if validation.get("status") == "PASS":
        if recovered_child_id:
            validation["recovered_child_job_id"] = recovered_child_id
            validation["recovered_child_plan_path"] = str(
                parent_dir.parent / recovered_child_id / "delivery_plan.json")
        return validation
    return None


def _failed_child_checkpoint_candidate(
    parent_dir: Path, previous_result: dict[str, Any], *,
    source_index: int, source_identity: dict[str, Any], plan: dict[str, Any],
) -> str:
    """Admit a past failed child for worker-level checkpoint verification only.

    Do not transfer failed output authority or adopt unproven media. New
    source/plan identity, unknown worker, incomplete temp directory, or
    still-running child must start clean.
    """
    prior_state = str(previous_result.get("state") or "")
    abandoned = (prior_state == "running"
                 and isinstance(previous_result.get("attempt_owner"), dict)
                 and _attempt_process_active(previous_result["attempt_owner"]) is False)
    if (previous_result.get("schema") != RESULT_SCHEMA
            or (prior_state not in {"failed", "cancelled"} and not abandoned)
            or str(previous_result.get("job_id") or "") != parent_dir.name):
        return ""
    previous_rows = [
        row for row in previous_result.get("sources") or []
        if isinstance(row, dict)
        and int(row.get("source_index") or 0) == source_index
    ]
    if len(previous_rows) != 1:
        return ""
    row = previous_rows[0]
    if (row.get("state") not in ({"failed", "cancelled", "running"} if abandoned
                                else {"failed", "cancelled"})
            or row.get("plan_fingerprint") != _plan_fingerprint(plan)
            or not _same_content_identity(dict(row.get("source_identity") or {}), source_identity)):
        return ""
    child_id = str(row.get("child_job_id") or "")
    if not child_id or child_id in {".", ".."} or not all(
            char.isascii() and (char.isalnum() or char in "-_") for char in child_id):
        return ""
    child = parent_dir.parent / child_id
    if (child.is_symlink() or not child.is_dir()
            or child.parent != parent_dir.parent
            or (child / "temp").is_symlink()
            or not (child / "temp").is_dir()):
        return ""
    if (child / "progress.json").is_symlink() or (child / "manifest.json").is_symlink():
        return ""
    progress = _load_json_if_present(child / "progress.json")
    manifest = _load_json_if_present(child / "manifest.json")
    if abandoned and progress.get("state") == "running":
        if _child_process_active(child_id, parent_dir.parent) is not False:
            return ""
        # This is the only recovery case without a captured subprocess exit
        # code: the original parent disappeared and the worker no longer runs.
        # Mark the stale display state before worker-owned fingerprint checks.
        _mark_exited_child_crashed(child, return_code=-1)
        progress = _load_json_if_present(child / "progress.json")
    if (progress.get("state") not in {"error", "crashed", "cancelled", "canceled"}
            or str(progress.get("worker_name") or "") != "video_upscale"
            or str(progress.get("job_id") or "") != child_id
            or str(manifest.get("worker") or "") != "video_upscale"):
        return ""
    return child_id


def _mark_exited_child_crashed(child_dir: Path, *, return_code: int) -> None:
    """Record a child that exited without publishing a terminal progress state.

    Called ONLY after Popen.wait/poll confirmed the owned child has exited.
    This allows a later verified retry while never declaring partial media
    passed. Normal worker errors/cancellations are not rewritten.
    """
    if return_code == 0:
        return
    path = child_dir / "progress.json"
    if path.is_symlink() or not path.is_file() or child_dir.is_symlink():
        return
    state = _load_json_if_present(path)
    if (state.get("state") != "running" or state.get("worker_name") != "video_upscale"
            or str(state.get("job_id") or "") != child_dir.name):
        return
    state.update(
        state="crashed", phase="crashed", can_cancel=False,
        exit_code=int(return_code),
        message="Video worker exited without a terminal result. Only independently verified checkpoints may be reused.",
        write_seq=int(state.get("write_seq") or 0) + 1,
        updated_at_utc=utc_now(),
    )
    atomic_write_json(path, state)


def _archive_previous_result(parent_dir: Path, payload: dict[str, Any]) -> None:
    attempt_id = str(payload.get("attempt_id") or "").strip()
    if not attempt_id or payload.get("schema") != RESULT_SCHEMA:
        return
    history = parent_dir / "history" / f"final_delivery_result_{attempt_id}.json"
    history.parent.mkdir(parents=True, exist_ok=True)
    if not history.exists():
        atomic_write_json(history, payload)


def _current_attempt_owner() -> dict[str, Any]:
    """Bind a running parent attempt to a process instance, not a reusable PID."""
    try:
        import psutil

        return {
            "pid": os.getpid(),
            "create_time": psutil.Process(os.getpid()).create_time(),
        }
    except (ImportError, OSError):
        # Psutil may be absent from a minimal Mac installation. Running a
        # job must still work; unverifiable crash takeover fails closed.
        return {"pid": os.getpid(), "create_time": None}


def _attempt_process_active(owner: dict[str, Any]) -> bool | None:
    """True=live, False=known exited/reused PID, None=cannot safely tell."""
    try:
        import psutil

        pid = owner.get("pid")
        created = owner.get("create_time")
        if (type(pid) is not int or pid <= 0
                or type(created) not in (float, int) or created <= 0):
            return None
        try:
            process = psutil.Process(pid)
            return abs(float(process.create_time()) - float(created)) < .01
        except psutil.NoSuchProcess:
            return False
        except (psutil.AccessDenied, OSError):
            return None
    except ImportError:
        return None


def _child_process_active(child_id: str, jobs_root: Path) -> bool | None:
    """Check the one recorded worker PID, not every process on the workstation.

    The signed-off worker manifest owns its child process identity. When
    unknown/inaccessible, refuse to take over a possibly active render.
    This avoids a costly process-wide scan under heavy background load.
    """
    try:
        import psutil

        child = jobs_root / child_id
        manifest_file = child / "manifest.json"
        if (child.is_symlink() or manifest_file.is_symlink()
                or not manifest_file.is_file()):
            return None
        manifest = _load_json_if_present(manifest_file)
        pid = manifest.get("worker_pid")
        if (manifest.get("worker") != "video_upscale"
                or type(pid) is not int or pid < 1):
            return None
        try:
            argv = psutil.Process(pid).cmdline()
        except psutil.NoSuchProcess:
            return False
        except (psutil.AccessDenied, OSError):
            return None
        if "--job-id" not in argv:
            return False
        position = argv.index("--job-id")
        return position + 1 < len(argv) and argv[position + 1] == child_id
    except ImportError:
        return None


def _reject_live_parent_attempt(result_path: Path) -> None:
    old = _load_json_if_present(result_path)
    if old.get("state") != "running":
        return
    owner = old.get("attempt_owner")
    if not isinstance(owner, dict):
        return  # Older ownerless results preserve legacy claim semantics.
    status = _attempt_process_active(owner)
    if status is not False:
        raise RuntimeError("final_delivery_parent_already_running_or_unverified")
    for row in old.get("sources") or []:
        if not isinstance(row, dict) or not row.get("child_job_id"):
            continue
        child_status = _child_process_active(str(row["child_job_id"]), result_path.parent.parent)
        if child_status is not False:
            raise RuntimeError("final_delivery_child_already_running_or_unverified")


def _claim_current_attempt(
    result_path: Path,
    writer: ProgressWriter,
    result: dict[str, Any],
) -> dict[str, Any]:
    with artifact_publication_lease(result_path):
        previous = _load_json_if_present(result_path)
        _reject_live_parent_attempt(result_path)
        if previous:
            _archive_previous_result(writer.job_dir, previous)
        writer.write_manifest(result)
        atomic_write_json(result_path, result)
        return previous


def _commit_current_result(
    result_path: Path,
    writer: ProgressWriter,
    result: dict[str, Any],
) -> None:
    attempt_id = str(result.get("attempt_id") or "")
    with artifact_publication_lease(result_path):
        current = _load_json_if_present(result_path)
        if str(current.get("attempt_id") or "") != attempt_id:
            raise RuntimeError("final_delivery_attempt_superseded")
        writer.write_manifest(result)
        # This fixed current receipt is the last completed-generation authority marker.
        atomic_write_json(result_path, result)


def _cancel_requested_for_attempt(writer: ProgressWriter, attempt_started_ns: int) -> bool:
    path = writer.job_dir / "cancel.request"
    try:
        return path.is_file() and int(path.stat().st_mtime_ns) >= int(attempt_started_ns)
    except OSError:
        return False


def _source_display_geometry_proof_valid(geometry: dict[str, Any], proof: dict[str, Any]) -> bool:
    """Validate coded source metadata, resolved display canvas and square-pixel output evidence."""
    try:
        def exact_int(value: Any) -> int:
            if isinstance(value, bool):
                raise ValueError("boolean integer")
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError("non-integer numeric value")
            return int(numeric)

        source_width = exact_int(geometry.get("source_width") or 0)
        source_height = exact_int(geometry.get("source_height") or 0)
        if source_width <= 0 or source_height <= 0:
            return False

        source_sar_text = str(geometry.get("source_sample_aspect_ratio") or "1:1")
        source_sar = Fraction(source_sar_text.replace(":", "/"))
        if source_sar <= 0:
            return False

        raw_rotation = geometry.get("source_rotation_degrees", 0)
        if isinstance(raw_rotation, bool):
            return False
        rotation_value = float(raw_rotation)
        if not math.isfinite(rotation_value) or not rotation_value.is_integer():
            return False
        source_rotation = int(rotation_value) % 360
        if source_rotation not in {0, 90, 180, 270}:
            return False

        if source_rotation in {90, 270}:
            display_width_exact = Fraction(source_height, 1) / source_sar
            expected_display_height = source_width
        else:
            display_width_exact = Fraction(source_width, 1) * source_sar
            expected_display_height = source_height
        expected_display_width = int(display_width_exact + Fraction(1, 2))
        if expected_display_width <= 0 or expected_display_height <= 0:
            return False
        expected_display_aspect = display_width_exact / expected_display_height

        display_size = (
            exact_int(geometry.get("display_width") or 0),
            exact_int(geometry.get("display_height") or 0),
        )
        output_size = (
            exact_int(geometry.get("output_width") or 0),
            exact_int(geometry.get("output_height") or 0),
        )
        proof_size = (exact_int(proof.get("width") or 0), exact_int(proof.get("height") or 0))
        padding = geometry.get("alignment_padding") if isinstance(geometry.get("alignment_padding"), dict) else {}
        padding_size = (
            exact_int(padding.get("horizontal") or 0),
            exact_int(padding.get("vertical") or 0),
        )
        expected_padding = (expected_display_width % 2, expected_display_height % 2)
        expected_output = (
            expected_display_width + expected_padding[0],
            expected_display_height + expected_padding[1],
        )

        output_sar = Fraction(str(geometry.get("sample_aspect_ratio") or "0").replace(":", "/"))
        output_rotation = geometry.get("rotation_degrees")
        if output_rotation is None or exact_int(output_rotation) % 360 != 0:
            return False
        if geometry.get("display_aspect_ratio") is not None:
            if str(geometry.get("display_aspect_ratio")) != (
                f"{expected_display_aspect.numerator}:{expected_display_aspect.denominator}"
            ):
                return False

        probe = proof.get("probe") if isinstance(proof.get("probe"), dict) else {}
        if probe.get("rotation_metadata_error"):
            return False
        probe_size = (exact_int(probe.get("width") or 0), exact_int(probe.get("height") or 0))
        probe_sar = Fraction(str(probe.get("sar") or "0").replace(":", "/"))
        probe_rotation = probe.get("rotation")
        return bool(
            geometry.get("policy") == "source_display"
            and display_size == (expected_display_width, expected_display_height)
            and output_size == expected_output
            and proof_size == expected_output
            and probe_size == expected_output
            and padding_size == expected_padding
            and output_sar == 1
            and probe_rotation is not None
            and exact_int(probe_rotation) % 360 == 0
            and probe_sar == 1
            and str(padding.get("placement") or "") == "centered"
        )
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


def _source_display_geometry_matches_probe(geometry: dict[str, Any], probe: dict[str, Any]) -> bool:
    """Bind the geometry proof's original SAR/rotation to the child's source probe."""
    try:
        def exact_int(value: Any) -> int:
            if isinstance(value, bool):
                raise ValueError("boolean integer")
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError("non-integer numeric value")
            return int(numeric)

        if probe.get("rotation_metadata_error"):
            return False
        source_width = exact_int(probe.get("width") or 0)
        source_height = exact_int(probe.get("height") or 0)
        source_sar = Fraction(str(probe.get("sar") or "0").replace(":", "/"))
        if source_width <= 0 or source_height <= 0 or source_sar <= 0:
            return False
        raw_rotation = probe.get("rotation")
        if isinstance(raw_rotation, bool):
            return False
        rotation_value = float(raw_rotation or 0)
        if not math.isfinite(rotation_value) or not rotation_value.is_integer():
            return False
        source_rotation = int(rotation_value) % 360
        if source_rotation not in {0, 90, 180, 270}:
            return False
        return bool(
            exact_int(geometry.get("source_width") or 0) == source_width
            and exact_int(geometry.get("source_height") or 0) == source_height
            and str(geometry.get("source_sample_aspect_ratio") or "1:1")
            == f"{source_sar.numerator}:{source_sar.denominator}"
            and exact_int(geometry.get("source_rotation_degrees", 0)) == source_rotation
        )
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


def validate_final_delivery_result(value: str | Path) -> dict[str, Any]:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "final_delivery_result.json"
    payload = _load_json_if_present(path)
    issues: list[str] = []
    if not payload:
        issues.append("final_delivery_result_missing_or_malformed")
    if payload.get("schema") != RESULT_SCHEMA:
        issues.append("final_delivery_result_schema_invalid")
    if str(payload.get("state") or "") != "passed":
        issues.append("final_delivery_result_not_passed")
    if not str(payload.get("attempt_id") or "").strip():
        issues.append("final_delivery_attempt_id_missing")
    if payload.get("errors"):
        issues.append("final_delivery_result_has_errors")
    outputs = [
        str(output)
        for output in payload.get("final_outputs") or []
        if str(output or "").strip()
    ]
    proofs = [
        row for row in payload.get("final_output_proofs") or [] if isinstance(row, dict)
    ]
    if not outputs or len(proofs) != len(outputs):
        issues.append("final_delivery_output_proofs_incomplete")
    proof_by_path = {
        _path_key(row.get("path") or row.get("output_path") or ""): row
        for row in proofs
        if str(row.get("path") or row.get("output_path") or "").strip()
    }
    if set(proof_by_path) != {_path_key(output) for output in outputs}:
        issues.append("final_delivery_output_proof_membership_mismatch")
    if "enhancement_recipe" in payload and any(
        proof.get("enhancement_recipe") != payload["enhancement_recipe"] for proof in proofs
    ):
        issues.append("final_delivery_enhancement_recipe_binding_mismatch")
    for proof in proofs:
        geometry_policy = str(proof.get("geometry_policy") or "")
        working_proof = proof.get("working_geometry") if isinstance(proof.get("working_geometry"), dict) else None
        if not geometry_policy and working_proof is None:
            continue
        if geometry_policy and geometry_policy not in {"source_display", "custom_final"}:
            issues.append("final_delivery_geometry_policy_invalid")
            continue
        proof_path = _path_key(proof.get("path") or proof.get("output_path") or "")
        source_owners = [
            row for row in payload.get("sources") or []
            if isinstance(row, dict)
            and proof_path in {_path_key(value) for value in row.get("outputs") or [] if str(value or "").strip()}
        ]
        if len(source_owners) != 1 or not source_owners[0].get("child_manifest_path"):
            issues.append("final_delivery_geometry_source_probe_missing_or_ambiguous")
        else:
            source_owner = source_owners[0]
            child = _load_json_if_present(Path(str(source_owner["child_manifest_path"])))
            child_plan = child.get("delivery_plan") if isinstance(child.get("delivery_plan"), dict) else {}
            if child_plan.get("geometry") != payload.get("geometry"):
                issues.append("final_delivery_geometry_result_binding_mismatch")
            source_key = _path_key(source_owner.get("source_path") or "")
            ready_sources = [
                row for row in child.get("ready_sources") or []
                if isinstance(row, dict)
                and source_key in {
                    _path_key(row.get("selected_source_path") or "") if row.get("selected_source_path") else "",
                    _path_key(row.get("source_path") or "") if row.get("source_path") else "",
                }
            ]
            if (
                len(ready_sources) != 1
                or not isinstance(ready_sources[0].get("probe"), dict)
            ):
                issues.append(
                    "final_delivery_source_geometry_source_probe_invalid"
                    if geometry_policy == "source_display"
                    else "final_delivery_geometry_source_probe_invalid"
                )
                continue
            source_probe = dict(ready_sources[0]["probe"])
            geometry = proof.get("source_geometry") if isinstance(proof.get("source_geometry"), dict) else {}
            if geometry_policy == "source_display":
                if not _source_display_geometry_proof_valid(geometry, proof):
                    issues.append("final_delivery_source_geometry_proof_invalid")
                elif not _source_display_geometry_matches_probe(geometry, source_probe):
                    issues.append("final_delivery_source_geometry_source_probe_invalid")
            elif geometry_policy == "custom_final":
                plan_geometry = child_plan.get("geometry") if isinstance(child_plan.get("geometry"), dict) else {}
                final_spec = plan_geometry.get("final") if isinstance(plan_geometry.get("final"), dict) else None
                try:
                    expected_geometry = resolve_custom_final_geometry(source_probe, final_spec or {})
                    output_probe = proof.get("probe") if isinstance(proof.get("probe"), dict) else {}
                    output_sar = Fraction(str(output_probe.get("sar") or "1:1").replace(":", "/"))
                    output_rotation = int(float(output_probe.get("rotation") or 0))
                    custom_valid = (
                        isinstance(final_spec, dict)
                        and proof.get("geometry") == final_spec
                        and geometry == expected_geometry
                        and int(proof.get("width") or 0) == expected_geometry["output_width"]
                        and int(proof.get("height") or 0) == expected_geometry["output_height"]
                        and int(output_probe.get("width") or 0) == expected_geometry["output_width"]
                        and int(output_probe.get("height") or 0) == expected_geometry["output_height"]
                        and output_sar == 1
                        and output_rotation == 0
                        and _source_display_geometry_matches_probe(geometry, source_probe)
                    )
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    custom_valid = False
                if not custom_valid:
                    issues.append("final_delivery_custom_geometry_proof_invalid")

            plan_has_geometry = "geometry" in child_plan
            if plan_has_geometry:
                try:
                    expected_working = resolve_delivery_working_geometry(
                        source_probe,
                        str(child_plan.get("enhance_preset") or "4k_60fps"),
                        "keep_source_aspect",
                        child_plan,
                    )
                    if (
                        working_proof != expected_working
                        or ready_sources[0].get("working_geometry") != expected_working
                    ):
                        issues.append("final_delivery_working_geometry_proof_invalid")
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    issues.append("final_delivery_working_geometry_proof_invalid")
            elif working_proof is not None:
                issues.append("final_delivery_working_geometry_unplanned")
    # Older parent results did not embed receipt evidence. Revalidate their
    # existing child reference when present, without rewriting old authority.
    legacy_watermarks: dict[str, list[dict[str, Any]]] = {}
    for source_row in payload.get("sources") or []:
        if not isinstance(source_row, dict) or not source_row.get("child_manifest_path"):
            continue
        child = _load_json_if_present(Path(str(source_row["child_manifest_path"])))
        for completion in child.get("completed_delivery_outputs") or []:
            if not isinstance(completion, dict):
                continue
            contract = completion.get("contract") if isinstance(completion.get("contract"), dict) else {}
            variant = contract.get("variant") if isinstance(contract.get("variant"), dict) else {}
            if variant.get("watermark") and variant.get("output_path"):
                legacy_watermarks.setdefault(_path_key(variant["output_path"]), []).append(completion)
    watermark_checks = 0
    for output in outputs:
        proof = proof_by_path.get(_path_key(output)) or {}
        try:
            observed = _stable_file_identity(output)
        except OSError:
            issues.append("final_delivery_output_missing_or_unstable")
            continue
        if (
            int(proof.get("bytes") or -1) != int(observed["bytes"])
            or str(proof.get("sha256") or "") != str(observed["sha256"])
        ):
            issues.append("final_delivery_output_identity_mismatch")
        legacy = legacy_watermarks.get(_path_key(output)) or []
        if proof.get("watermark_enabled") or legacy:
            watermark_checks += 1
            try:
                if proof.get("watermark_enabled"):
                    stored_receipt = proof.get("watermark_receipt")
                    if not isinstance(stored_receipt, dict) or not stored_receipt.get("receipt"):
                        raise ValueError("watermark_receipt_evidence_missing")
                    receipt = inspect_signed_video_receipt(output, stored_receipt["receipt"])
                    if any(receipt[field] != stored_receipt.get(field) for field in (
                        "receipt_sha256", "receipt_bytes", "asset_id", "namespace", "identifier",
                        "delivery_profile_id", "output_sha256", "output_bytes",
                    )):
                        raise ValueError("watermark_receipt_evidence_changed")
                else:
                    if "watermark_enabled" in proof or len(legacy) != 1:
                        raise ValueError("watermark_receipt_contract_ambiguous")
                    completion = legacy[0]
                    if completion.get("fingerprint") != proof.get("completion_fingerprint"):
                        raise ValueError("watermark_receipt_completion_binding_mismatch")
                    receipt = validate_delivery_watermark_receipt(
                        Path(output), completion["contract"]["variant"], completion.get("watermark_result")
                    )
                if receipt["output_sha256"] != proof.get("sha256") or receipt["output_bytes"] != proof.get("bytes"):
                    raise ValueError("watermark_receipt_output_binding_mismatch")
            except (OSError, TypeError, ValueError, RuntimeError):
                issues.append("final_delivery_watermark_receipt_missing_or_invalid")
    for row in payload.get("sources") or []:
        if not isinstance(row, dict) or str(row.get("state") or "") != "passed":
            issues.append("final_delivery_source_not_passed")
            continue
        source_path = str(row.get("source_path") or "")
        stored_source = dict(row.get("source_identity") or {})
        try:
            observed_source = _stable_file_identity(source_path)
        except OSError:
            issues.append("final_delivery_source_missing_or_unstable")
            continue
        if not _same_content_identity(stored_source, observed_source):
            issues.append("final_delivery_source_identity_mismatch")
    return {
        "schema": "spiffy_final_delivery_result_validation_v1",
        "status": "PASS" if not issues else "FAIL",
        "current": not issues,
        "result_path": str(path),
        "attempt_id": str(payload.get("attempt_id") or ""),
        "watermark_receipts_checked": watermark_checks,
        "issues": list(dict.fromkeys(issues)),
    }


def _run_child(
    command: list[str],
    child_dir: Path,
    parent: ProgressWriter,
    file_index: int,
    file_total: int,
    *,
    attempt_started_ns: int,
) -> int:
    stdout_path = child_dir / "launcher_stdout.txt"
    stderr_path = child_dir / "launcher_stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
        last_seq = -1
        while process.poll() is None:
            if _cancel_requested_for_attempt(parent, attempt_started_ns):
                (child_dir / "cancel.request").write_text("cancel\n", encoding="utf-8")
            progress = _load_json_if_present(child_dir / "progress.json")
            seq = int(progress.get("write_seq") or 0)
            if seq != last_seq:
                child_percent = float(progress.get("percent") or 0.0)
                overall = ((file_index - 1) + child_percent / 100.0) / max(1, file_total) * 100.0
                parent.update(
                    state="running",
                    phase=str(progress.get("phase") or "delivery"),
                    message=str(progress.get("message") or "Running final delivery"),
                    percent=round(overall, 2),
                    file_index=file_index,
                    file_total=file_total,
                    metrics={"child_job_id": child_dir.name, "child_percent": child_percent},
                )
                last_seq = seq
            time.sleep(0.5)
        code = int(process.returncode or 0)
        if code:
            try:
                _mark_exited_child_crashed(child_dir, return_code=code)
            except (OSError, ValueError, TypeError) as exc:
                # A progress update must not obscure the original native exit.
                parent.log(f"Child progress recovery marker unavailable: {type(exc).__name__}.")
        return code


def sleep_system() -> None:
    system = platform_name()
    if system == "windows":
        subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=True)
    elif system == "macos":
        subprocess.run(["pmset", "sleepnow"], check=True)
    else:
        raise RuntimeError("Automatic sleep is supported only on Windows and macOS.")


def run_final_delivery(
    sources: Iterable[str | Path],
    *,
    jobs_root: str | Path = DEFAULT_JOBS_ROOT,
    job_id: str = "",
    output_root: str | Path | None = None,
    profile_ids: Iterable[str] = DELIVERY_SETS["canonical"],
    brand: str = "spiffy-scenes",
    layout_policy: str = "fit_black",
    watermark: bool = False,
    watermark_strength: float = 0.2,
    execute: bool = False,
    sleep_when_done: bool = False,
    enhancement: dict[str, Any] | None = None,
    frame_rate: dict[str, Any] | None = None,
    adaptive_enhancement: dict[str, Any] | None = None,
    geometry: dict[str, Any] | None = None,
    compression_profile: str = "existing",
) -> tuple[int, dict[str, Any]]:
    if adaptive_enhancement is not None:
        adaptive_enhancement = normalize_adaptive_policy(adaptive_enhancement)
        if watermark or (enhancement or {}).get("overrides"):
            raise ValueError("Adaptive natural requires watermark off and no conflicting manual overrides.")
    if frame_rate is not None:
        frame_rate = normalize_frame_rate(frame_rate)
        if watermark:
            raise ValueError("Explicit frame_rate with VideoSeal is not yet qualified.")
    if geometry is not None:
        geometry = normalize_delivery_geometry(geometry)
    compression_profile = normalize_compression_profile(compression_profile)
    # Validate and snapshot before creating/claiming any job or result authority.
    if enhancement is not None:
        recipe = resolve_enhancement_recipe(enhancement)
        enhancement = {"preset_id": recipe.preset_id, "overrides": dict(enhancement.get("overrides", {}))}
    source_paths = [Path(value).expanduser().resolve() for value in sources]
    if not source_paths:
        raise ValueError("At least one source video is required.")
    jobs_path = Path(jobs_root).expanduser().resolve()
    parent_id = job_id or make_job_id("final_delivery")
    # Check before constructing ProgressWriter, since that object also writes
    # a parent progress file. An active same-ID job must not lose its UI state.
    _reject_live_parent_attempt(jobs_path / parent_id / "final_delivery_result.json")
    writer = ProgressWriter(jobs_path, parent_id, "Spiffy Final Delivery", "final_delivery")
    result_path = writer.job_dir / "final_delivery_result.json"
    attempt_id = make_job_id("attempt")
    attempt_started_ns = time.time_ns()
    normalized_profiles = normalize_profile_ids(profile_ids)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "job_id": parent_id,
        "attempt_id": attempt_id,
        "attempt_owner": _current_attempt_owner(),
        "created_at_utc": utc_now(),
        "state": "planned" if not execute else "running",
        "execute": execute,
        "brand": brand,
        "brand_namespace": resolve_brand_namespace(brand),
        "profile_ids": normalized_profiles,
        **({"geometry": geometry} if geometry is not None else {}),
        **({"compression_profile": compression_profile} if compression_profile != "existing" else {}),
        "source_count": len(source_paths),
        "sources": [],
        "final_outputs": [],
        "final_output_proofs": [],
        "commands": commands,
        "errors": [],
        "sleep_requested": bool(sleep_when_done),
        "sleep_executed": False,
        **({"enhancement_recipe": recipe.as_dict()} if enhancement is not None else {}),
        **({"adaptive_enhancement": adaptive_enhancement} if adaptive_enhancement is not None else {}),
        "authority_policy": {
            "current_result_path": str(result_path),
            "current_attempt_only": True,
            "older_history_satisfies_current_readiness": False,
            "child_exit_code_alone_satisfies_delivery": False,
            "final_result_is_last_completion_authority": True,
        },
    }
    previous_result = _claim_current_attempt(result_path, writer, result)
    attempt_evidence: dict[str, Any] = {
        "schema": ATTEMPT_EVIDENCE_SCHEMA,
        "attempt_id": attempt_id,
        "parent_job_id": parent_id,
        "created_at_utc": result["created_at_utc"],
        "state": "planned" if not execute else "running",
        "authority": False,
        "purpose": "media_recovery_only_not_delivery_readiness",
        "sources": [],
    }
    _write_attempt_evidence(writer.job_dir, attempt_evidence)
    writer.update(state="running", phase="plan", message="Building final-delivery plans", file_total=len(source_paths))

    for index, source in enumerate(source_paths, start=1):
        target_root = Path(output_root).expanduser().resolve() if output_root else source.parent / "_SpiffyDelivery"
        plan = build_final_delivery_plan(
            source,
            target_root,
            profile_ids=normalized_profiles,
            brand=brand,
            layout_policy=layout_policy,
            watermark=watermark,
            watermark_strength=watermark_strength,
            enhancement=enhancement,
            frame_rate=frame_rate,
            adaptive_enhancement=adaptive_enhancement,
            geometry=geometry,
            compression_profile=compression_profile,
        )
        row = {
            "source_index": index,
            "source_path": str(source),
            "source_identity": {},
            "plan_fingerprint": _plan_fingerprint(plan),
            "child_job_id": "",
            "child_manifest_path": "",
            "plan_path": "",
            "profile_output_map": plan["profile_output_map"],
            "planned_physical_output_count": plan["physical_output_count"],
            "physical_output_count": plan["physical_output_count"],
            "state": "planned",
            "outputs": [],
            "final_output_proofs": [],
            "media_reused": False,
        }
        rows.append(row)
        if not execute:
            child_id = make_job_id("final_delivery_child")
            child_dir = ensure_job_dirs(jobs_path, child_id)
            input_list = child_dir / "inputs.txt"
            plan_path = child_dir / "delivery_plan.json"
            atomic_write_text(input_list, str(source) + "\n")
            atomic_write_json(plan_path, plan)
            command = build_worker_command(
                jobs_root=jobs_path,
                child_job_id=child_id,
                input_list=input_list,
                plan_path=plan_path,
            )
            row["child_job_id"] = child_id
            row["plan_path"] = str(plan_path)
            commands.append(command)
            writer.command(command)
            continue

        if _cancel_requested_for_attempt(writer, attempt_started_ns):
            row["state"] = "cancelled"
            result["errors"].append(
                {
                    "source_path": str(source),
                    "message": "Final delivery was cancelled before launching the next child.",
                    "code": "cancelled",
                }
            )
            break
        try:
            source_identity = _stable_file_identity(source)
        except OSError as exc:
            row["state"] = "failed"
            result["errors"].append(
                {
                    "source_path": str(source),
                    "message": f"Source is missing or unstable before delivery: {exc}",
                    "code": "source_identity_invalid",
                }
            )
            break
        row["source_identity"] = source_identity

        validation = _reusable_previous_child(
            writer.job_dir,
            previous_result,
            source_index=index,
            source=source,
            source_identity=source_identity,
            plan=plan,
        )
        if validation is not None:
            previous_attempt = str(previous_result.get("attempt_id") or "")
            previous_evidence = _load_json_if_present(
                _attempt_evidence_path(writer.job_dir, previous_attempt)
            )
            previous_row = next(
                (
                    item
                    for item in previous_evidence.get("sources") or []
                    if isinstance(item, dict)
                    and int(item.get("source_index") or 0) == index
                ),
                {},
            )
            row["child_job_id"] = str(
                previous_row.get("child_job_id")
                or validation.get("recovered_child_job_id") or "")
            row["child_manifest_path"] = str(validation.get("child_manifest_path") or "")
            row["plan_path"] = str(previous_row.get("plan_path")
                                   or validation.get("recovered_child_plan_path") or "")
            row["media_reused"] = True
            row["reused_from_attempt_id"] = previous_attempt
        else:
            child_id = make_job_id("final_delivery_child")
            child_dir = ensure_job_dirs(jobs_path, child_id)
            input_list = child_dir / "inputs.txt"
            plan_path = child_dir / "delivery_plan.json"
            atomic_write_text(input_list, str(source) + "\n")
            atomic_write_json(plan_path, plan)
            resume_id = _failed_child_checkpoint_candidate(
                writer.job_dir, previous_result, source_index=index,
                source_identity=source_identity, plan=plan)
            command = build_worker_command(
                jobs_root=jobs_path,
                child_job_id=child_id,
                input_list=input_list,
                plan_path=plan_path,
                resume_job_id=resume_id,
            )
            commands.append(command)
            writer.command(command)
            row["child_job_id"] = child_id
            row["plan_path"] = str(plan_path)
            row["child_manifest_path"] = str(child_dir / "manifest.json")
            if resume_id:
                row["resumed_from_child_job_id"] = resume_id
            # Persist the active child identity in the existing parent result.
            # A hard parent-process kill must not erase which child/checkpoints
            # belong to this attempt. State remains RUNNING, never delivered.
            row["state"] = "running"
            result["sources"] = rows
            _commit_current_result(result_path, writer, result)

            writer.update(
                phase="launch_child",
                message=f"Starting delivery {index}/{len(source_paths)}: {source.name}",
                file_index=index,
                current_file={"display_name": source.name, "source_path": str(source)},
            )
            delivery_lane = _delivery_target_lane(plan)
            try:
                with interprocess_lane_lease(
                    delivery_lane,
                    workflow="final_delivery",
                    operation="render_delivery_target",
                    cancel_requested=lambda: _cancel_requested_for_attempt(
                        writer,
                        attempt_started_ns,
                    ),
                ) as lane_receipt:
                    row["delivery_target_lane"] = {
                        "lane": delivery_lane,
                        "status": str(lane_receipt.get("status") or ""),
                        "state": str(lane_receipt.get("state") or ""),
                    }
                    return_code = _run_child(
                        command,
                        child_dir,
                        writer,
                        index,
                        len(source_paths),
                        attempt_started_ns=attempt_started_ns,
                    )
            except RuntimeError as exc:
                row["state"] = (
                    "cancelled"
                    if _cancel_requested_for_attempt(writer, attempt_started_ns)
                    else "failed"
                )
                result["errors"].append(
                    {
                        "source_path": str(source),
                        "child_job_id": child_id,
                        "message": f"Delivery target lane could not be acquired: {exc}",
                        "code": (
                            "cancelled"
                            if row["state"] == "cancelled"
                            else "delivery_target_lane_unavailable"
                        ),
                    }
                )
                break
            row["return_code"] = return_code
            if _cancel_requested_for_attempt(writer, attempt_started_ns):
                row["state"] = "cancelled"
                result["errors"].append(
                    {
                        "source_path": str(source),
                        "child_job_id": child_id,
                        "message": "Final delivery was cancelled.",
                        "code": "cancelled",
                    }
                )
                break
            if return_code != 0:
                row["state"] = "failed"
                progress = _load_json_if_present(child_dir / "progress.json")
                message = str(
                    progress.get("message")
                    or f"Child delivery failed with exit code {return_code}"
                )
                result["errors"].append(
                    {
                        "source_path": str(source),
                        "child_job_id": child_id,
                        "message": message,
                        "code": "child_failed",
                    }
                )
                break
            child_manifest = _load_json_if_present(child_dir / "manifest.json")
            validation = _validated_child_delivery(
                child_manifest,
                child_manifest_path=child_dir / "manifest.json",
                source=source,
                source_identity=source_identity,
                plan=plan,
            )

        if not validation or validation.get("status") != "PASS":
            row["state"] = "failed"
            row["validation_issues"] = list(
                (validation or {}).get("issues") or ["child_delivery_validation_missing"]
            )
            result["errors"].append(
                {
                    "source_path": str(source),
                    "child_job_id": row.get("child_job_id") or "",
                    "message": "Child delivery did not produce one current, fully proven output generation.",
                    "code": "child_delivery_not_current",
                    "issues": row["validation_issues"],
                }
            )
            break

        row["state"] = "passed"
        row["source_identity"] = dict(validation.get("source_identity") or source_identity)
        row["outputs"] = list(validation.get("outputs") or [])
        row["final_output_proofs"] = [
            dict(item)
            for item in validation.get("final_output_proofs") or []
            if isinstance(item, dict)
        ]
        row["profile_output_map"] = dict(validation.get("profile_output_map") or {})
        row["physical_output_count"] = int(validation.get("physical_output_count") or 0)
        result["final_outputs"].extend(row["outputs"])
        result["final_output_proofs"].extend(row["final_output_proofs"])
        attempt_evidence["sources"] = [
            *[
                item
                for item in attempt_evidence.get("sources") or []
                if isinstance(item, dict)
                and int(item.get("source_index") or 0) != index
            ],
            {
                "source_index": index,
                "source_path": str(source),
                "source_identity": dict(row["source_identity"]),
                "plan_fingerprint": row["plan_fingerprint"],
                "plan_path": row["plan_path"],
                "child_job_id": row["child_job_id"],
                "child_manifest_path": row["child_manifest_path"],
                "outputs": list(row["outputs"]),
                "final_output_proofs": [
                    dict(item) for item in row["final_output_proofs"]
                ],
                "media_reused": bool(row["media_reused"]),
            },
        ]
        attempt_evidence["state"] = "media_validated"
        _write_attempt_evidence(writer.job_dir, attempt_evidence)

    result["sources"] = rows
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    cancelled = any(
        isinstance(item, dict) and str(item.get("code") or "") == "cancelled"
        for item in result["errors"]
    )
    result["state"] = (
        "cancelled"
        if cancelled
        else "failed"
        if result["errors"]
        else "passed"
        if execute
        else "planned"
    )
    attempt_evidence["state"] = result["state"]
    _write_attempt_evidence(writer.job_dir, attempt_evidence)
    # The result, CLI and review HTML consume one measured projection. Reporting
    # never upgrades failed media to PASS or re-counts a resumed historical render.
    try:
        from delivery_insights import attach_delivery_insights, render_report_html
        result["performance"] = attach_delivery_insights(result, job_dir=writer.job_dir)
        report_path = writer.job_dir / "delivery_report.html"
        atomic_write_text(report_path, render_report_html(result["performance"]["summary"],
            title=f"Smart Finish · {result['state']} · {job_id or writer.job_id}"))
        result["performance_report_path"] = str(report_path)
    except (OSError, ValueError, TypeError, KeyError) as report_error:
        result["performance"] = {"status": "unavailable", "reason": str(report_error)}
        writer.log(f"Performance report unavailable; media authority unchanged: {report_error}")
    if execute and sleep_when_done and result["state"] == "passed":
        result["sleep_executed"] = True
    try:
        _commit_current_result(result_path, writer, result)
    except Exception as exc:  # noqa: BLE001 - current authority must remain fail-closed.
        result["state"] = (
            "superseded"
            if str(exc) == "final_delivery_attempt_superseded"
            else "failed"
        )
        result["errors"].append(
            {
                "message": f"Final delivery result publication failed: {exc}",
                "code": (
                    "attempt_superseded"
                    if result["state"] == "superseded"
                    else "final_result_publication_failed"
                ),
            }
        )
        writer.error(
            str(result["errors"][-1]["message"]),
            "result_publication",
        )
        if result["state"] == "failed":
            # A failed PASS publication must not strand a live-looking RUNNING
            # authority after this process already terminated its children.
            # Retry publishing only the FAILED state under the same attempt
            # fence. This never converts unverified outputs to completion.
            try:
                _commit_current_result(result_path, writer, result)
            except Exception as cleanup_error:  # noqa: BLE001 - preserve original publication failure
                writer.log(
                    "Failed-result recovery marker unavailable; prior attempt remains "
                    f"unverified: {type(cleanup_error).__name__}"
                )
        return (2 if result["state"] == "superseded" else 1), result
    if result["state"] == "cancelled":
        writer.cancelled("Final delivery cancelled")
        return 1, result
    if result["errors"]:
        writer.error(str(result["errors"][0]["message"]), "child_failed")
        return 1, result
    writer.done("Final delivery complete" if execute else "Final delivery dry-run planned")
    if execute and sleep_when_done:
        sleep_system()
    return 0, result


__all__ = [
    "BRAND_NAMESPACES",
    "CHANNEL_BRAND_NAMESPACES",
    "DEFAULT_JOBS_ROOT",
    "DELIVERY_SETS",
    "WATERMARK_VIDEO_TYPE_PRESETS",
    "ATTEMPT_EVIDENCE_SCHEMA",
    "PLAN_SCHEMA",
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "build_final_delivery_plan",
    "build_worker_command",
    "normalize_profile_ids",
    "profiles_for_delivery_set",
    "request_delivery_options",
    "request_sources",
    "resolve_brand_namespace",
    "run_final_delivery",
    "sleep_system",
    "validate_final_delivery_result",
    "watermark_identifier_for_profile",
]
