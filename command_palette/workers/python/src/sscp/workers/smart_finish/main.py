"""Palette adapter only: Final Delivery continues to own rendering and retries."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

from sscp.shared.jobs import JobContext
from sscp.shared.selected_files import load_selected_files

COMMAND_ID = "media.smart_finish"
ADVISOR_COMMAND_ID = "media.smart_finish_advisor"
BEST_QUALITY_COMMAND_ID = "media.smart_finish_best_quality"
COMFYUI_COMMAND_ID = "media.smart_finish_comfyui"
COMFYUI_6K_COMMAND_ID = "media.smart_finish_comfyui_6k"
COMFYUI_FAST_COMMAND_ID = "media.smart_finish_comfyui_fast"
COMFYUI_MEASURED_COMMAND_ID = "media.smart_finish_comfyui_measured"
COMFYUI_COMMAND_IDS = frozenset({
    COMFYUI_COMMAND_ID, COMFYUI_6K_COMMAND_ID,
    COMFYUI_FAST_COMMAND_ID, COMFYUI_MEASURED_COMMAND_ID,
})
SUPPORTED_COMMAND_IDS = frozenset({
    COMMAND_ID, ADVISOR_COMMAND_ID, BEST_QUALITY_COMMAND_ID, *COMFYUI_COMMAND_IDS,
})
ROOT = Path(__file__).resolve().parents[7]
MAX_SELECTIONS = 32
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".webm", ".avi", ".mxf"}


def _owner():
    workspace = ROOT / "Spiffy Workspace"
    for path in (ROOT, workspace):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from workflow import final_delivery_pipeline
    return final_delivery_pipeline


def _run_delivery(sources: list[Path], **kwargs: Any) -> tuple[int, dict[str, Any]]:
    return _owner().run_final_delivery(sources, **kwargs)


def _verify_delivery(job_dir: Path) -> dict[str, Any]:
    return _owner().validate_final_delivery_result(job_dir)


def run_job(job_dir: Path) -> int:
    from shared.job_paths import validate_job_directory
    from shared.progress_writer import atomic_write_json

    job_dir = Path(job_dir).absolute()
    if job_dir.parent.name.casefold() != "_jobs":
        raise ValueError("Smart Finish must use the existing _jobs owner.")
    validate_job_directory(job_dir, job_dir.parent)
    for name in ("manifest.json", "selected_files.json"):
        path = job_dir / name
        if path.is_file() and path.stat().st_size > 1024 * 1024:
            raise ValueError(f"Smart Finish {name} exceeds its bounded input size.")
    context = JobContext.from_job_dir(job_dir)
    command_id = context.manifest.get("command_id")
    title = ("Smart Finish — Quality Options" if command_id == ADVISOR_COMMAND_ID
             else "Smart Finish — Measured Compression" if command_id == COMFYUI_MEASURED_COMMAND_ID
             else "Smart Finish — ComfyUI AI" if command_id in COMFYUI_COMMAND_IDS
             else "Smart Finish — Best Available Detail" if command_id == BEST_QUALITY_COMMAND_ID
             else "Smart Finish — Automatic Upscale")

    def fail(message: str) -> int:
        context.write_result({"schema": "spiffy_smart_finish_palette_result_v1", "state": "failed",
                              "command_id": command_id, "errors": [message], "final_outputs": []})
        context.progress.error(message, job_id=job_dir.name, command_id=command_id,
                               title=title, can_cancel=False)
        return 1

    if context.cancel_requested:
        context.write_result({"schema": "spiffy_smart_finish_palette_result_v1", "state": "cancelled",
                              "command_id": command_id, "errors": [], "final_outputs": []})
        context.progress.cancelled("Smart Finish cancelled before rendering.", job_id=job_dir.name,
                                   command_id=command_id, title=title, can_cancel=False)
        return 2
    try:
        if command_id not in SUPPORTED_COMMAND_IDS:
            raise ValueError("Smart Finish received another command's job manifest.")
        arguments = context.manifest.get("arguments", {})
        allowed = {"output_fps", "working_long_edge", "quality_policy", "apply_recommendation",
                   "final_long_edge", "allow_raster_only", "compression_profile", "model_id",
                   "source_preservation", "backend_id", "compression_strategy",
                   "region_mode", "motioncoach_pose_root", "motioncoach_source_binding_id"}
        if not isinstance(arguments, dict) or set(arguments) - allowed:
            raise ValueError("Smart Finish received unsupported quality or geometry preferences.")
        quality_policy = arguments.get("quality_policy", "best_quality")
        if quality_policy not in ("best_quality", "balanced", "fastest_safe"):
            raise ValueError("quality_policy must be best_quality, balanced or fastest_safe.")
        model_id = arguments.get("model_id", "auto-fast" if command_id == COMFYUI_FAST_COMMAND_ID else "auto-quality")
        if command_id in COMFYUI_COMMAND_IDS:
            _owner()
            from comfyui_model_catalog import MODEL_POLICIES
            from video_backend_router import get_backend
            if type(model_id) is not str or model_id not in MODEL_POLICIES:
                raise ValueError("ComfyUI model policy does not refer to a pinned qualified model.")
            backend_id = arguments.get("backend_id", "comfyui-gan")
            selected_backend = get_backend(backend_id)
        elif (set(arguments) & {"model_id", "source_preservation", "backend_id", "compression_strategy",
                                "region_mode", "motioncoach_pose_root", "motioncoach_source_binding_id"}):
            raise ValueError("ComfyUI model selection is not supported by Topaz or the read-only advisor.")
        region_mode = arguments.get("region_mode", "full")
        pose_root_value = arguments.get("motioncoach_pose_root", "")
        pose_binding = arguments.get("motioncoach_source_binding_id", "")
        if region_mode not in {"full", "motioncoach_face"} or type(region_mode) is not str:
            raise ValueError("Region mode must be full or motioncoach_face.")
        if type(pose_root_value) is not str or type(pose_binding) is not str:
            raise ValueError("Published MotionCoach evidence and source binding must be strings.")
        if region_mode == "motioncoach_face" and (not pose_root_value or not pose_binding):
            raise ValueError("Face-only ComfyUI mode requires a published MotionCoach pose path and exact source-binding ID.")
        if region_mode == "full" and (pose_root_value or pose_binding):
            raise ValueError("MotionCoach evidence is not used without explicit face-only mode.")
        if len(pose_root_value) > 2048 or len(pose_binding) > 128:
            raise ValueError("MotionCoach evidence path or source binding exceeds safe length.")
        source_preservation = arguments.get("source_preservation", 0.0)
        if (type(source_preservation) not in (int, float) or not math.isfinite(source_preservation)
                or not 0 <= source_preservation <= 0.25):
            raise ValueError("Source preservation must be a finite fraction between 0 and 0.25.")
        compression_strategy = arguments.get(
            "compression_strategy", "measured" if command_id == COMFYUI_MEASURED_COMMAND_ID else "fixed")
        if type(compression_strategy) is not str or compression_strategy not in {"fixed", "measured"}:
            raise ValueError("Compression strategy must be fixed or measured.")
        if command_id == COMFYUI_MEASURED_COMMAND_ID and compression_strategy != "measured":
            raise ValueError("The Measured Compression command must execute its actual compression calibration.")
        if command_id in COMFYUI_COMMAND_IDS:
            if source_preservation and not selected_backend.supports_source_preservation:
                raise ValueError("Selected AI backend has no qualified source-preservation adapter.")
        compression_profile = arguments.get("compression_profile")
        if compression_profile is not None:
            _owner()
            from workflow.final_delivery_pipeline import normalize_compression_profile
            compression_profile = normalize_compression_profile(compression_profile)
        apply_recommendation = arguments.get("apply_recommendation", command_id == BEST_QUALITY_COMMAND_ID)
        if command_id == BEST_QUALITY_COMMAND_ID and apply_recommendation is not True:
            raise ValueError("Best Available Detail must use the actual quality recommendation.")
        allow_raster_only = arguments.get("allow_raster_only", False)
        if type(apply_recommendation) is not bool or type(allow_raster_only) is not bool:
            raise ValueError("Smart Finish quality permission flags must be booleans.")
        rate = arguments.get("output_fps", 60)
        from fractions import Fraction
        if type(rate) not in (int, str) or (isinstance(rate, str) and len(rate) > 30):
            raise ValueError("output_fps must be an integer or rational rate from 1 through 120.")
        parsed = Fraction(rate)
        if parsed < 1 or parsed > 120 or parsed.denominator > 1_000_000:
            raise ValueError("output_fps must be a bounded rational rate from 1 through 120.")
        adaptive_policy: dict[str, Any] = {"mode": "natural"}
        if "working_long_edge" in arguments:
            edge = arguments["working_long_edge"]
            if type(edge) is not int or not 128 <= edge <= 4096 or edge % 2:
                raise ValueError("working_long_edge must be an even integer from 128 through the present 4096 Topaz limit.")
            adaptive_policy["working_long_edge"] = edge
        final_edge = arguments.get("final_long_edge")
        if final_edge is not None and (type(final_edge) is not int or not 64 <= final_edge <= 8192 or final_edge % 2):
            raise ValueError("final_long_edge must be an even integer from 64 through 8192.")
        if final_edge is not None and final_edge > 4096 and command_id not in COMFYUI_COMMAND_IDS and not allow_raster_only:
            raise ValueError("A final canvas beyond 4096 is not native 6K/8K neural detail; explicit allow_raster_only is required.")
        if final_edge is not None and final_edge > 4096 and command_id not in COMFYUI_COMMAND_IDS:
            adaptive_policy["interpolation_resolution"] = "working"
        if final_edge is not None and apply_recommendation:
            raise ValueError("Choose either automatic recommendation or manual final_long_edge.")
        sources = [path.resolve() for path in load_selected_files(job_dir)]
        if not 1 <= len(sources) <= MAX_SELECTIONS:
            raise ValueError("Select 1–32 finished video files for Smart Finish.")
        for source in sources:
            if source.suffix.lower() not in VIDEO_EXTENSIONS or not source.is_file():
                raise ValueError(f"Selected item is not an available finished video: {source}")
        if command_id in COMFYUI_COMMAND_IDS:
            if len(sources) != 1 or "output_fps" in arguments or "working_long_edge" in arguments or apply_recommendation:
                raise ValueError("ComfyUI AI accepts one video and preserves its actual frame rate; no Topaz working size or interpolation override.")
            _owner()
            from workflow.final_delivery_pipeline import COMPRESSION_PROFILES
            from video_backend_router import run_backend
            crf, preset = COMPRESSION_PROFILES[compression_profile or "quality"]
            code, result = run_backend(backend_id, sources[0], jobs_root=job_dir.parent, job_id=job_dir.name,
                                       target_long_edge=final_edge or 3840,
                                       crf=crf, preset=preset, model_id=model_id,
                                       source_preservation=float(source_preservation),
                                       compression_strategy=compression_strategy,
                                       region_mode=region_mode,
                                       motioncoach_pose_root=Path(pose_root_value) if pose_root_value else None,
                                       motioncoach_source_binding_id=pose_binding)
            context.write_result({**result, "command_id": command_id,
                                  "final_outputs": result.get("final_outputs", [])})
            if code == 0:
                context.progress.done(message="ComfyUI AI upscale, compression and verification complete", can_cancel=False)
            return code
        if command_id == ADVISOR_COMMAND_ID or apply_recommendation:
            _owner()  # Reuse incumbent media probe and authoritative geometry limits.
            from shared.ffmpeg_utils import standard_ffprobe
            from shared.media_probe import probe_video
            from quality_advisor import build_advice, inspect_comfyui
            if apply_recommendation and len(sources) != 1:
                raise ValueError("Automatic source-specific resolution requires one selected video.")
            comfy = inspect_comfyui() if command_id == ADVISOR_COMMAND_ID else None
            options = []
            for source in sources:
                info = probe_video(source, standard_ffprobe())
                options.append({"source": str(source),
                                "advice": build_advice(info, policy=quality_policy, comfyui=comfy)})
            if command_id == ADVISOR_COMMAND_ID:
                labels = []
                for row in options[:3]:
                    topaz = row["advice"]["recommended"] or "none"
                    gan6k = next((choice for choice in row["advice"]["choices"]
                                  if choice["label"] == "6K" and choice["comfyui_gan"]["candidate"]), None)
                    labels.append(f"{Path(row['source']).name}: Topaz {topaz}"
                                  + (" / ComfyUI 6K possible" if gan6k else ""))
                recommendations = "; ".join(labels)
                if len(options) > 3:
                    recommendations += f"; +{len(options) - 3} more"
                context.write_result({"schema": "spiffy_smart_finish_quality_advisor_result_v1",
                                      "state": "passed", "command_id": command_id, "items": options,
                                      "errors": [], "final_outputs": [], "rendered": False})
                context.progress.done(message=f"Quality choices ready (no render): {recommendations}", can_cancel=False)
                return 0
            chosen = next((row for row in options[0]["advice"]["choices"] if row["recommended"]), None)
            if not chosen or "working_long_edge" in arguments:
                raise ValueError("Cannot apply recommendation to this source or combine it with manual working size.")
            adaptive_policy["working_long_edge"] = int(chosen["long_edge"])
            final_edge = int(chosen["long_edge"])
        policy = {"mode": "cfr", "fps": f"{parsed.numerator}/{parsed.denominator}"}
        # This is the existing request contract, so jobs remain inspectable and
        # reusable by the normal CLI/AutoExport route after the palette closes.
        request = {"schema": "spiffy_final_delivery_request_v1", "inputs": [str(p) for p in sources],
                   "final_delivery": {"delivery_set": "smart_finish", "adaptive_enhancement": adaptive_policy,
                                      "frame_rate": policy, "watermark": False,
                                      **({"compression_profile": compression_profile} if compression_profile is not None else {}),
                                      **({"geometry": {"final": {"mode": "long_edge", "long_edge": final_edge}}}
                                         if final_edge is not None else {})}}
        atomic_write_json(job_dir / "request.json", request)
        code, result = _run_delivery(sources, jobs_root=job_dir.parent, job_id=job_dir.name,
            profile_ids=["smart_finish_master"], adaptive_enhancement=adaptive_policy,
            frame_rate=policy, watermark=False, execute=True,
            **({"compression_profile": compression_profile} if compression_profile is not None else {}),
            **({"geometry": {"final": {"mode": "long_edge", "long_edge": final_edge}}}
               if final_edge is not None else {}))
        if code == 0:
            verification = _verify_delivery(job_dir)
            if result.get("state") != "passed" or verification.get("status") != "PASS":
                return fail("Final Delivery did not publish a current verified output: " + json.dumps(verification.get("issues", [])))
        context.write_result({**result, "command_id": command_id,
                              "delivery_result_path": str(job_dir / "final_delivery_result.json")})
        # Do not replace the owner progress lifecycle with a fake adapter success.
        return code
    except (OSError, TypeError, ValueError, RuntimeError, ZeroDivisionError) as exc:
        return fail(str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Automatic natural enhancement through Spiffy Final Delivery.")
    parser.add_argument("--job-dir", required=True)
    return run_job(Path(parser.parse_args(argv).job_dir))


if __name__ == "__main__":
    raise SystemExit(main())
