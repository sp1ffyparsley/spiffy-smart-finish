"""Read-only, source-specific Smart Finish resolution and backend advice.

This is deliberately not another render owner.  A larger output raster is NOT
evidence that a neural model inferred detail at that resolution.  Report the
existing Topaz working limit and ComfyUI eligibility independently, without
claiming model readiness on the basis of a node name or installed GUI.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import urllib.error
import urllib.request


ADVICE_SCHEMA = "spiffy_smart_finish_quality_advice_v1"
PROFILES = ((2560, "2K"), (3072, "3K"), (3840, "4K"), (5760, "6K"), (7680, "8K"))
POLICIES = ("best_quality", "balanced", "fastest_safe")
KNOWN_WEIGHT_MARKERS = {
    "relayvsr": ("relayvsr", "relay_vsr"),
    "flashvsr": ("flashvsr", "flash_vsr"),
    "seedvr2": ("seedvr2", "seed_vr2"),
}


def _even_aspect_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Choose an even, source-aspect canvas without stretching the image."""
    if min(width, height) < 1 or long_edge < 64:
        raise ValueError("Quality Advisor needs positive source geometry and target long edge.")
    if width >= height:
        return long_edge, max(2, 2 * round(height * long_edge / width / 2))
    return max(2, 2 * round(width * long_edge / height / 2)), long_edge


def _weight_markers(root: Path | None) -> dict[str, bool]:
    result = {name: False for name in KNOWN_WEIGHT_MARKERS}
    if root is None or not root.is_dir():
        return result
    models = root / "models"
    if not models.is_dir():
        return result
    # Only two levels of model files; no recursive scan over other workspace
    # projects or arbitrary user-selected folders, and no model downloads.
    for folder in (models / "diffusion_models", models / "unet", models / "vae",
                   models / "upscale_models", models / "relayvsr", models / "seedvr2",
                   models / "flashvsr"):
        if not folder.is_dir():
            continue
        try:
            children = list(folder.iterdir())[:2048]
        except OSError:
            continue
        for file in children:
            if not file.is_file() or file.suffix.casefold() not in {".safetensors", ".ckpt", ".pth", ".pt", ".gguf"}:
                continue
            lowered = file.name.casefold()
            for name, markers in KNOWN_WEIGHT_MARKERS.items():
                if any(marker in lowered for marker in markers):
                    result[name] = True
    return result


def inspect_comfyui(*, root: Path | None = None, endpoint: str = "http://127.0.0.1:8188",
                    timeout_seconds: float = 1.0) -> dict[str, object]:
    """Bounded loopback-only health check; never submits a model inference job.

    Server availability and checkpoint filenames are diagnostics, not proof of
    node registration, CUDA compatibility, quantization or visual quality.
    """
    from urllib.parse import urlsplit

    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("ComfyUI discovery permits only a plain localhost HTTP endpoint.")
    if root is None:
        configured = os.environ.get("SPIFFY_COMFYUI_ROOT", "")
        root = Path(configured) if configured else (Path.home() / "Documents" / "AutoHotkey" /
            "SS Scripts External Assets" / "AI Media" / "tools" / "ComfyUI")
    root = Path(root)
    weights = _weight_markers(root)
    # A private server is started by the runner as needed. Model readiness is
    # not inferred from a GUI process or the names of untrusted checkpoints.
    from comfyui_model_catalog import inspect_models
    models = inspect_models(root)
    executable = root / ".venv" / "Scripts" / "python.exe"
    ready_model = models["realesrgan_x4plus"]["installed_and_verified"]
    local_gan = {
        "model": "realesrgan_x4plus",
        "pinned_checkpoint_verified": ready_model,
        "isolated_runtime_present": executable.is_file() and (root / "main.py").is_file(),
        "needs_per_source_runtime_and_quality_checks": True,
        "temporal_video_model": False,
    }
    local_gan["available_for_source_preflight"] = (
        bool(local_gan["pinned_checkpoint_verified"]) and bool(local_gan["isolated_runtime_present"]))
    live = False
    reason = "local_server_unavailable"
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 5:
        raise ValueError("ComfyUI discovery timeout must be between 0 and 5 seconds.")
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, headers, newurl):
            return None

    try:
        request = urllib.request.Request(endpoint.rstrip("/") + "/system_stats", headers={"Accept": "application/json"})
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout_seconds) as response:
            payload = response.read(262144)
            stats = json.loads(payload)
            live = isinstance(stats, dict) and isinstance(stats.get("system"), dict)
        reason = "available_not_inference_qualified" if live else "invalid_server_response"
    except (OSError, ValueError, urllib.error.URLError):
        pass
    return {"server_running": live, "reason": reason, "checkpoint_filename_evidence": weights,
            "local_gan": local_gan,
            "models": models,
            "models_production_qualified": False,
            "note": "API health and checkpoint names do not prove that a VSR workflow can render on this GPU."}


def _source_dimensions(info: dict[str, object]) -> tuple[int, int]:
    width = info.get("display_width", info.get("width"))
    height = info.get("display_height", info.get("height"))
    if type(width) is not int or type(height) is not int or min(width, height) < 64:
        raise ValueError("Quality Advisor needs valid video display width and height.")
    if int(info.get("rotation", 0) or 0) % 180 == 90 and "display_width" not in info:
        return height, width
    return width, height


def _unqualified_source_reasons(info: dict[str, object], width: int, height: int) -> list[str]:
    """Avoid recommending a route the incumbent adaptive worker rejects."""
    reasons: list[str] = []
    if info.get("has_alpha") is True:
        reasons.append("alpha_video_not_supported_in_adaptive_delivery")
    transfer = str(info.get("color_transfer") or "").casefold()
    if transfer in {"smpte2084", "arib-std-b67", "pq", "hlg"}:
        reasons.append("hdr_requires_qualified_color_management")
    field = str(info.get("field_order") or "").casefold()
    if field and field not in {"progressive", "unknown"}:
        reasons.append("interlaced_source_requires_qualified_deinterlacing")
    if info.get("is_vfr") is True:
        reasons.append("variable_frame_rate_requires_source_timing_qualification")
    if "fps" in info:
        rate = info["fps"]
        if type(rate) not in (float, int) or not math.isfinite(rate) or rate <= 0:
            reasons.append("invalid_source_frame_rate")
    if "duration" in info:
        duration = info["duration"]
        if type(duration) not in (float, int) or not math.isfinite(duration) or duration <= 0:
            reasons.append("invalid_source_duration")
    sar = str(info.get("sar") or "1:1").replace("/", ":")
    if sar != "1:1":
        reasons.append("non_square_pixel_aspect_requires_separate_admission")
    rotation = info.get("rotation", 0)
    try:
        rotated = int(rotation) % 360
    except (TypeError, ValueError, OverflowError):
        rotated = -1
    if rotated not in {0, 180} and (rotated != 90 and rotated != 270 or width != height):
        reasons.append("rotation_requires_separate_geometry_qualification")
    return reasons


def build_advice(info: dict[str, object], *, policy: str = "best_quality",
                 comfyui: dict[str, object] | None = None,
                 working_limit: int = 4096, working_pixel_limit: int = 16_777_216,
                 final_limit: int = 8192, final_pixel_limit: int = 33_554_432,
                 observations: list[dict[str, object]] | None = None) -> dict[str, object]:
    """Rank real output routes; estimates exist ONLY for matching observations."""
    if policy not in POLICIES:
        raise ValueError("Unknown Smart Finish quality policy.")
    width, height = _source_dimensions(info)
    from comfyui_video_backend import MAX_NATIVE_GAN_PIXELS
    from comfyui_model_catalog import select_model
    source_issues = _unqualified_source_reasons(info, width, height)
    if not all(type(value) is int and value >= 64 for value in
               (working_limit, working_pixel_limit, final_limit, final_pixel_limit)):
        raise ValueError("Quality geometry bounds are invalid.")
    choices: list[dict[str, object]] = []
    for edge, label in PROFILES:
        out_width, out_height = _even_aspect_size(width, height, edge)
        allowed_final = (max(out_width, out_height) <= final_limit
                         and out_width * out_height <= final_pixel_limit)
        allowed_working = (not source_issues and edge >= max(width, height)
                           and max(out_width, out_height) <= working_limit
                           and out_width * out_height <= working_pixel_limit
                           and max(width, height) <= working_limit
                           and width * height <= working_pixel_limit)
        is_real_working = allowed_working and allowed_final
        status = ("source_requires_review" if source_issues else
                  "native_model_path_eligible_not_quality_verified" if is_real_working else
                  "would_downscale_source" if edge < max(width, height) else
                  "output_raster_only_no_neural_detail_at_target" if allowed_final
                  else "exceeds_final_pixel_safety_limit")
        # A past exact-run timing is not an estimate for different footage or a
        # changed runtime.  Caller must supply the source and runtime identities.
        # Never extrapolate per-pixel scaling to 6K, another device or clip.
        measured = None
        for row in observations or []:
            if (isinstance(info.get("source_fingerprint"), str) and info["source_fingerprint"]
                    and isinstance(info.get("runtime_fingerprint"), str) and info["runtime_fingerprint"]
                    and row.get("source_fingerprint") == info["source_fingerprint"]
                    and row.get("runtime_fingerprint") == info["runtime_fingerprint"]
                    and row.get("label") == label and row.get("width") == width
                    and row.get("height") == height and row.get("model") == "topaz"
                    and row.get("fps") == info.get("fps")
                    and row.get("verified") is True and type(row.get("elapsed_seconds")) in (int, float)
                    and math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0):
                measured = round(float(row["elapsed_seconds"]), 2)
                break
        choices.append({
            "label": label, "long_edge": edge, "width": out_width, "height": out_height,
            "final_canvas_allowed": allowed_final, "native_working_canvas_allowed": allowed_working,
            "neural_detail_at_output_resolution": is_real_working,
            "status": status,
            "seconds": measured,
            "time_basis": "verified_same_configuration_observation" if measured is not None else "unknown_no_matching_measurement",
            "recommended": False,
        })
        gan = (comfyui or {}).get("local_gan")
        reported_models = (comfyui or {}).get("models")
        try:
            selected_gan = select_model("auto-quality", source_long_edge=max(width, height),
                                        target_long_edge=edge)
        except ValueError:
            selected_gan = None
        if isinstance(reported_models, dict) and selected_gan is not None:
            selected_row = reported_models.get(selected_gan.id)
            installed = (isinstance(selected_row, dict)
                         and selected_row.get("installed_and_verified") is True
                         and bool((comfyui or {}).get("local_gan", {}).get("isolated_runtime_present")))
        else:
            # Previous/standalone calls still report the known x4 candidate;
            # never invent x2 or fast readiness without registry evidence.
            installed = (isinstance(gan, dict) and gan.get("available_for_source_preflight") is True
                         and selected_gan is not None and selected_gan.id == "realesrgan_x4plus")
        duration = info.get("duration")
        rate = info.get("fps")
        safe_duration = (type(duration) in (int, float) and math.isfinite(duration)
                         and 0 < duration <= 60)
        safe_rate = (type(rate) in (int, float) and math.isfinite(rate)
                     and 1 <= rate <= 60)
        eligible_gan = bool(
            installed and selected_gan is not None and not source_issues and safe_duration and safe_rate
            and out_width * out_height <= final_pixel_limit
            and width * height * (selected_gan.native_scale**2) <= MAX_NATIVE_GAN_PIXELS
            and max(width, height) <= edge <= 4 * max(width, height)
            and edge <= max(width, height) * selected_gan.native_scale
            and edge <= min(final_limit, 5760)
            and duration * rate <= 1800
        )
        choices[-1]["comfyui_gan"] = {
            "candidate": eligible_gan,
            "status": ("source_preflight_and_gpu_qualification_required" if eligible_gan
                       else "unavailable_or_not_qualified"),
            "model": selected_gan.id if selected_gan else None,
            "native_scale": selected_gan.native_scale if selected_gan else None,
            "temporal_video_model": False,
            "seconds": None,
            "time_basis": "unknown_no_matching_verified_model_measurement",
        }
    eligible = [row for row in choices if row["native_working_canvas_allowed"]]
    if policy == "best_quality":
        choice = eligible[-1] if eligible else None
    elif policy == "balanced":
        choice = next((row for row in eligible if row["label"] == "3K"), eligible[-1] if eligible else None)
    else:
        choice = eligible[0] if eligible else None
    if choice is not None:
        choice["recommended"] = True
    return {
        "schema": ADVICE_SCHEMA, "quality_policy": policy,
        "source": {"display_width": width, "display_height": height,
                   "fps": info.get("fps"), "duration_seconds": info.get("duration")},
        "recommended": choice["label"] if choice else None,
        "choices": choices,
        "source_admission_issues": source_issues,
        "comfyui": comfyui if comfyui is not None else {"reason": "not_probed", "models_production_qualified": False},
        "limits": {"topaz_working_long_edge": working_limit, "topaz_working_pixels": working_pixel_limit,
                   "final_long_edge": final_limit, "final_pixels": final_pixel_limit},
        "notes": [
            "4K and smaller choices use only the existing Topaz working path; a prior native 4K crash is still unresolved.",
            "The Topaz working model remains bounded at 4K; eligible 6K footage may use the separate local Real-ESRGAN GAN route after source/runtime checks.",
            "ComfyUI's GAN upscales individual frames, not trained temporal VSR. Quality superiority, grain stability and long-video speed remain unproven.",
            "Missing render times are unknown, never fabricated from resolution ratios.",
        ],
    }
