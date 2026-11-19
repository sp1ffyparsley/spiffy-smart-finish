from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow.final_delivery_pipeline import (  # noqa: E402
    DEFAULT_JOBS_ROOT,
    profiles_for_delivery_set,
    request_delivery_options,
    request_sources,
    run_final_delivery,
    normalize_compression_profile,
)
from shared.job_paths import job_dir as resolve_job_dir  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or execute verified Spiffy post-export delivery.")
    parser.add_argument("--input", action="append", default=[], help="Finished render path. Repeat for multiple files.")
    parser.add_argument("--request", default="", help="AutoExport final-delivery request JSON.")
    parser.add_argument("--jobs-root", default=str(DEFAULT_JOBS_ROOT))
    parser.add_argument("--job-id", default="")
    parser.add_argument("--inspect-job", default="",
                        help="Read an existing final-delivery job status without creating or starting any work.")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--delivery-set", choices=["canonical", "smart_finish", "short_form", "all_social"], default="canonical")
    parser.add_argument("--profiles", default="", help="Comma-separated social export profile IDs.")
    parser.add_argument("--brand", default="spiffy-scenes")
    parser.add_argument("--layout-policy", choices=["fit_black", "fill_crop"], default="fit_black")
    parser.add_argument("--watermark", choices=["on", "off"], default=None)
    parser.add_argument("--watermark-strength", type=float, default=0.2)
    parser.add_argument("--enhancement-preset", default=None, help="Existing enhancement preset; overrides request selection.")
    parser.add_argument("--enhancement-overrides", default=None, help='Proteus numeric controls as JSON, e.g. {"noise":0.2}.')
    parser.add_argument("--adaptive-enhancement", action="store_true", help="Measured natural enhancement with protected source detail and automatic contextual cropping.")
    parser.add_argument("--subject-refinement", choices=["auto", "off"], default=None,
                        help="Use existing source-bound subject roto only in admitted shot/resource windows, or keep natural protection only.")
    parser.add_argument("--enhancement-model-search", choices=["auto", "off"], default=None,
                        help="Compare installed enhancement models only when bounded shot, cost and resource gates admit it.")
    parser.add_argument("--geometry", default=None,
                        help='Typed working/final geometry JSON, e.g. {"working":{"mode":"preset","preset":"4k"},"final":{"mode":"exact","width":1080,"height":1920,"layout":"fit_into_frame"}.')
    parser.add_argument("--working-long-edge", type=int, default=None,
                        help="Adaptive working long edge in pixels (e.g. 1920, 2560, 3072, 3840); final canvas stays separately selected.")
    parser.add_argument("--quality-advice", action="store_true",
                        help="Read-only source-specific 2K/3K/4K/6K/8K advice. No job, model inference, output file or HTML is created.")
    parser.add_argument("--quality-policy", choices=["best_quality", "balanced", "fastest_safe"],
                        default="best_quality", help="Rank quality choices; normal exports are unchanged unless --apply-quality-advice is set.")
    parser.add_argument("--apply-quality-advice", action="store_true",
                        help="Apply the admitted Topaz resolution recommendation to one source, including final canvas.")
    parser.add_argument("--final-long-edge", type=int, default=None,
                        help="Explicit final output long edge, including 5760 (6K) or 7680 (8K). Does not increase native Topaz neural detail.")
    sys.path.insert(0, str(ROOT.parent / "tools" / "video_upscale"))
    from video_backend_router import BACKENDS
    parser.add_argument("--ai-backend", choices=["topaz", *BACKENDS], default="topaz",
                        help="Explicit local AI engine. ComfyUI GAN retains source FPS and uses only verified local checkpoints.")
    parser.add_argument("--model-id", default="auto-quality",
                        help="Pinned ComfyUI model or policy: auto-quality, auto-fast, realesrgan_x4plus, realesrgan_x2plus, realesrgan_general_x4v3.")
    parser.add_argument("--source-preservation", type=float, default=0.0,
                        help="Optional natural-look blend of 0–25%% original source pixels after AI enhancement, currently limited to 4K; default 0 preserves model output.")
    parser.add_argument("--region-mode", choices=["full", "motioncoach_face"], default="full",
                        help="Optional source-bound face-only neural crop driven by published MotionCoach pose evidence; background remains a conventional upscale.")
    parser.add_argument("--motioncoach-pose-root", default="",
                        help="Explicit path to a published MotionCoach pose bundle. Only used with --region-mode motioncoach_face.")
    parser.add_argument("--motioncoach-source-binding-id", default="",
                        help="Exact published MotionCoach source_binding_id; the video content hash and full pose clock must also match.")
    parser.add_argument("--compression-profile", choices=["existing", "maximum", "quality", "balanced", "compact"],
                        default=None, help="Quality-first CRF/slow H.264 settings after enhancement; normal legacy exports remain unchanged.")
    parser.add_argument("--compression-strategy", choices=["fixed", "measured"], default="fixed",
                        help="Optional measured H.264 compression pilots for eligible short source windows; defaults to the established fixed-quality encode.")
    parser.add_argument("--allow-raster-only", action="store_true",
                        help="Acknowledge that output beyond the current AI working limit is conventional upsampling, not native 6K/8K model inference.")
    frame_rate_group = parser.add_mutually_exclusive_group()
    frame_rate_group.add_argument("--output-fps", default=None, help="Explicit constant output rate, e.g. 90, 120, or 60000/1001. No slow motion.")
    frame_rate_group.add_argument("--preserve-source-timing", action="store_true", help="Preserve the exact decoded source presentation timestamps; only declared source-timing-capable master profiles are supported.")
    parser.add_argument("--execute", action="store_true", help="Run Topaz, FFmpeg, VideoSeal, and verification.")
    parser.add_argument("--sleep-when-done", action="store_true", help="Sleep only after every executed output passes.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inspect_job:
        if args.execute or args.input or args.request or args.job_id:
            raise SystemExit("--inspect-job is read-only; do not combine it with execution/input/job creation.")
        existing = resolve_job_dir(args.jobs_root, args.inspect_job) / "final_delivery_result.json"
        if not existing.is_file() or existing.is_symlink():
            raise SystemExit("No existing Final Delivery result was found for that job ID.")
        payload = json.loads(existing.read_text(encoding="utf-8"))
        if payload.get("schema") != "spiffy_final_delivery_result_v1" or payload.get("job_id") != args.inspect_job:
            raise SystemExit("Existing job result is not a recognized Final Delivery receipt.")
        print(json.dumps({
            "job_id": payload["job_id"],
            "state": payload.get("state"),
            "attempt_id": payload.get("attempt_id"),
            "source_count": payload.get("source_count"),
            "sources": [
                {
                    "index": row.get("source_index"), "state": row.get("state"),
                    "child_job_id": row.get("child_job_id"),
                    "media_reused": bool(row.get("media_reused")),
                    "resume_from_child": row.get("resumed_from_child_job_id") or "",
                }
                for row in payload.get("sources") or []
                if isinstance(row, dict)
            ],
            "final_outputs": list(payload.get("final_outputs") or []),
            "errors": list(payload.get("errors") or []),
            "recovery": (
                "If the prior job is no longer running, repeat its original inputs/options "
                f"with --job-id {args.inspect_job}. Only current, verified media/checkpoints are reused."
                if payload.get("state") in {"failed", "cancelled", "running"} else
                "No retry needed for a completed job; use the normal result verifier before publishing."
            ),
            "inspection_only": True,
        }, indent=2))
        return 0
    sources = [Path(value) for value in args.input]
    profile_ids = [value.strip() for value in args.profiles.split(",") if value.strip()] or profiles_for_delivery_set(args.delivery_set)
    brand = args.brand
    layout_policy = args.layout_policy
    watermark = args.watermark == "on"
    watermark_strength = args.watermark_strength
    output_root = args.output_root or None
    enhancement = None
    frame_rate = None
    adaptive_enhancement = None
    geometry = None
    compression_profile = "existing"

    if args.request:
        request_path = Path(args.request).expanduser().resolve()
        payload = json.loads(request_path.read_text(encoding="utf-8-sig"))
        sources.extend(request_sources(payload))
        options = request_delivery_options(payload)
        enhancement = options.get("enhancement")
        frame_rate = options.get("frame_rate")
        adaptive_enhancement = options.get("adaptive_enhancement")
        geometry = options.get("geometry")
        compression_profile = options.get("compression_profile", "existing")
        if not args.profiles and args.delivery_set == "canonical":
            profile_ids = options["profile_ids"]
        if args.brand == "spiffy-scenes":
            brand = options["brand"]
        if args.layout_policy == "fit_black":
            layout_policy = options["layout_policy"]
        if args.watermark is None:
            watermark = options["watermark"]
        if args.watermark_strength == 0.2:
            watermark_strength = options["watermark_strength"]
        if output_root is None and options["output_root"]:
            output_root = options["output_root"]

    if args.enhancement_preset is not None or args.enhancement_overrides is not None:
        enhancement = {
            "preset_id": args.enhancement_preset if args.enhancement_preset is not None else
                (enhancement["preset_id"] if enhancement is not None else "4k_60fps"),
            "overrides": json.loads(args.enhancement_overrides) if args.enhancement_overrides is not None else
                (dict(enhancement.get("overrides", {})) if enhancement is not None else {}),
        }
    if not sources:
        raise SystemExit("Provide --input or --request.")
    if args.preserve_source_timing:
        frame_rate = {"mode": "preserve_source"}
    elif args.output_fps is not None:
        frame_rate = {"mode": "cfr", "fps": args.output_fps}
    if args.adaptive_enhancement:
        adaptive_enhancement = adaptive_enhancement or {"mode": "natural"}
    if args.subject_refinement is not None:
        adaptive_enhancement = {**(adaptive_enhancement or {"mode": "natural"}),
                                "subject_refinement": args.subject_refinement}
    if args.enhancement_model_search is not None:
        adaptive_enhancement = {**(adaptive_enhancement or {"mode": "natural"}),
                                "model_search": args.enhancement_model_search}
    if args.working_long_edge is not None:
        adaptive_enhancement = {**(adaptive_enhancement or {"mode": "natural"}),
                                "working_long_edge": args.working_long_edge}
    if args.geometry is not None:
        geometry = json.loads(args.geometry)
    if args.compression_profile is not None:
        compression_profile = normalize_compression_profile(args.compression_profile)
    if args.ai_backend == "topaz" and args.model_id != "auto-quality":
        raise SystemExit("--model-id is only supported by the explicit ComfyUI GAN backend.")
    if args.ai_backend == "topaz" and args.source_preservation:
        raise SystemExit("--source-preservation is currently supported only by the explicit ComfyUI GAN backend.")
    if args.ai_backend == "topaz" and (args.region_mode != "full" or args.motioncoach_pose_root or args.motioncoach_source_binding_id):
        raise SystemExit("Pose-guided face region processing is only supported by the explicit ComfyUI backend.")
    if args.ai_backend == "topaz" and args.compression_strategy != "fixed":
        raise SystemExit("--compression-strategy measured currently requires the explicit ComfyUI GAN backend.")
    if args.ai_backend != "topaz":
        sys.path.insert(0, str(ROOT.parent / "tools" / "video_upscale"))
        from comfyui_model_catalog import MODEL_POLICIES
        from video_backend_router import get_backend, run_backend
        backend = get_backend(args.ai_backend)
        if args.model_id not in MODEL_POLICIES:
            raise SystemExit("Unrecognized or unqualified ComfyUI model ID.")
        if args.source_preservation and not backend.supports_source_preservation:
            raise SystemExit("Selected AI backend has no qualified source-preservation function.")
        if args.region_mode == "motioncoach_face" and not (args.motioncoach_pose_root and args.motioncoach_source_binding_id):
            raise SystemExit("Pose-guided face enhancement requires the MotionCoach bundle path AND exact source-binding ID.")
        if args.region_mode == "full" and (args.motioncoach_pose_root or args.motioncoach_source_binding_id):
            raise SystemExit("MotionCoach pose evidence may be used only when explicitly selecting the face-region mode.")
        if args.request or len(sources) != 1 or args.job_id or args.inspect_job or args.sleep_when_done:
            raise SystemExit("ComfyUI GAN currently takes exactly one --input and cannot use Topaz job retry, saved requests or sleep.")
        if args.quality_advice or args.apply_quality_advice or args.geometry is not None or args.working_long_edge is not None:
            raise SystemExit("ComfyUI GAN has its own qualified output geometry; do not combine it with Topaz advice or geometry.")
        if args.preserve_source_timing or args.output_fps is not None or args.watermark == "on":
            raise SystemExit("ComfyUI GAN preserves existing CFR and has no verified interpolation or VideoSeal path.")
        if any(value is not None for value in (args.enhancement_preset, args.enhancement_overrides,
                                               args.subject_refinement, args.enhancement_model_search)):
            raise SystemExit("ComfyUI GAN cannot inherit Topaz model controls or subject refinement.")
        if not args.execute:
            print(json.dumps({"backend": "comfyui-gan", "execution": "not_requested",
                              "source": str(sources[0]), "model_policy": args.model_id,
                              "selected_checkpoint": "not_resolved_until_source_probe",
                              "source_preservation": args.source_preservation,
                              "region_mode": args.region_mode,
                              "compression_strategy": args.compression_strategy,
                              "note": "Model selection and GPU qualification happen after source probing when --execute is used. No background job was started."}, indent=2))
            return 0
        from workflow.final_delivery_pipeline import COMPRESSION_PROFILES
        from shared.media_probe import probe_video
        from shared.ffmpeg_utils import standard_ffprobe
        info = probe_video(sources[0], standard_ffprobe())
        edge = args.final_long_edge or max(3840, int(info.get("width") or 0), int(info.get("height") or 0))
        crf, preset = COMPRESSION_PROFILES[compression_profile]
        code, result = run_backend(args.ai_backend, sources[0], jobs_root=Path(args.jobs_root),
                                   output_root=Path(output_root) if output_root else None,
                                   target_long_edge=edge, crf=crf, preset=preset,
                                   model_id=args.model_id,
                                   source_preservation=args.source_preservation,
                                   compression_strategy=args.compression_strategy,
                                   region_mode=args.region_mode,
                                   motioncoach_pose_root=Path(args.motioncoach_pose_root) if args.motioncoach_pose_root else None,
                                   motioncoach_source_binding_id=args.motioncoach_source_binding_id)
        print(json.dumps({"state": result["state"], "job_id": result["job_id"],
                          "final_outputs": result.get("final_outputs", []),
                          "backend": result["backend"], "errors": result.get("errors", []),
                          "elapsed_seconds": result.get("elapsed_seconds")}, indent=2))
        return code
    if args.quality_advice or args.apply_quality_advice:
        from shared.ffmpeg_utils import standard_ffprobe
        from shared.media_probe import probe_video
        from quality_advisor import build_advice, inspect_comfyui
        if args.quality_advice and (args.execute or args.sleep_when_done or args.job_id or args.inspect_job):
            raise SystemExit("--quality-advice is read-only; do not combine with execution or an existing job.")
        if args.apply_quality_advice and len(sources) != 1:
            raise SystemExit("--apply-quality-advice requires exactly one input so every choice is source-specific.")
        comfyui = inspect_comfyui() if args.quality_advice else None
        advice_rows = []
        for source in sources:
            if not source.is_file():
                raise SystemExit(f"Missing video for quality advice: {source}")
            info = probe_video(source, standard_ffprobe())
            advice = build_advice(info, policy=args.quality_policy, comfyui=comfyui)
            advice_rows.append({"source": str(source), "advice": advice})
        if args.quality_advice:
            print(json.dumps({"schema": "spiffy_quality_advice_batch_v1", "items": advice_rows,
                              "rendered": False, "outputs_created": False}, indent=2))
            return 0
        chosen = next((row for row in advice_rows[0]["advice"]["choices"] if row["recommended"]), None)
        if not chosen:
            raise SystemExit("No native-working resolution was admitted. Choose a qualified model or keep the original.")
        if geometry is not None or args.final_long_edge is not None or args.working_long_edge is not None:
            raise SystemExit("--apply-quality-advice cannot be combined with manual geometry or working-long-edge.")
        adaptive_enhancement = {**(adaptive_enhancement or {"mode": "natural"}),
                                "working_long_edge": chosen["long_edge"]}
        geometry = {"final": {"mode": "long_edge", "long_edge": chosen["long_edge"]}}
    if args.final_long_edge is not None:
        if args.apply_quality_advice or geometry is not None:
            raise SystemExit("--final-long-edge cannot be combined with another final geometry choice.")
        if (type(args.final_long_edge) is not int or args.final_long_edge < 64
                or args.final_long_edge > 8192 or args.final_long_edge % 2):
            raise SystemExit("--final-long-edge must be an even number from 64 through 8192.")
        if args.final_long_edge > 4096 and not args.allow_raster_only:
            raise SystemExit("The native AI working limit is 4096. Use --allow-raster-only to request a larger final raster without claiming native 6K/8K neural detail.")
        if args.final_long_edge > 4096:
            # Keep AI frame synthesis at the current qualified working canvas.
            # The final raster enlargement is explicitly conventional only.
            adaptive_enhancement = {**(adaptive_enhancement or {"mode": "natural"}),
                                    "interpolation_resolution": "working"}
        geometry = {"final": {"mode": "long_edge", "long_edge": args.final_long_edge}}
    code, result = run_final_delivery(
        sources,
        jobs_root=args.jobs_root,
        job_id=args.job_id,
        output_root=output_root,
        profile_ids=profile_ids,
        brand=brand,
        layout_policy=layout_policy,
        watermark=watermark,
        watermark_strength=watermark_strength,
        execute=args.execute,
        sleep_when_done=args.sleep_when_done,
        enhancement=enhancement,
        frame_rate=frame_rate,
        adaptive_enhancement=adaptive_enhancement,
        geometry=geometry,
        compression_profile=compression_profile,
    )
    print(json.dumps({
        "state": result["state"],
        "job_id": result["job_id"],
        "brand_namespace": result["brand_namespace"],
        "source_count": result["source_count"],
        "final_outputs": result["final_outputs"],
        "errors": result["errors"],
        "elapsed_seconds": result.get("elapsed_seconds"),
        "performance_report_path": result.get("performance_report_path"),
    }, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
