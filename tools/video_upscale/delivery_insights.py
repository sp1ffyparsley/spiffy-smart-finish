"""Measured diagnostics shared by Final Delivery and its offline comparison UI.

This is a projection of the existing job/child manifests, never completion
authority, a scheduler, or a second metrics database. Missing evidence stays
unknown; first-frame startup is a subset of process time, not an extra stage.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import html
import json
import math
import platform
import shutil
import subprocess
import sys
from typing import Any

SCHEMA = "spiffy_delivery_performance_v1"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(Fraction(value)) if isinstance(value, str) else float(value)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def resource_snapshot(*, platform_name: str | None = None) -> dict[str, Any]:
    """Two cheap optional observations per job, not a peak sampler or daemon."""
    host = platform_name or sys.platform
    result: dict[str, Any] = {
        "platform": host,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_percent": None,
        "ram_available_bytes": None,
        "ram_total_bytes": None,
        "gpu_memory_free_mib": None,
        "gpu_memory_total_mib": None,
        "gpu_utilization_percent": None,
        "gpu_measurement": "not_sampled_on_this_platform",
        "scope": "instantaneous_observation_not_peak",
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        result.update(
            ram_available_bytes=int(memory.available),
            ram_total_bytes=int(memory.total),
            cpu_percent=float(psutil.cpu_percent(interval=0.05)),
        )
    except (ImportError, OSError, RuntimeError):
        pass
    if host not in {"win32", "linux"}:
        return result
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            proc = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,memory.free,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if proc.returncode == 0:
                name, free, total, busy = (
                    proc.stdout.strip().splitlines()[0].rsplit(",", 3)
                )
                values = [int(v.strip()) for v in (free, total, busy)]
                if min(values) >= 0 and values[0] <= values[1] and values[2] <= 100:
                    result.update(
                        gpu_name=name.strip(),
                        gpu_memory_free_mib=values[0],
                        gpu_memory_total_mib=values[1],
                        gpu_utilization_percent=values[2],
                        gpu_measurement="nvidia_smi_instantaneous",
                    )
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
            result["gpu_measurement"] = "unavailable"
    else:
        result["gpu_measurement"] = "unavailable"
    return result


def _stage_name(phase: str) -> str:
    if phase == "adaptive_enhancement_pilot":
        return "Enhancement model trials"
    if phase.startswith("topaz_upscale"):
        return "Enhancement / compositing"
    if phase.startswith("topaz_interpolate"):
        return "Interpolation (including model trials)"
    if phase in {"final_h264", "final_encode"}:
        return "Final encoding"
    if phase.startswith("adaptive_motion"):
        return "Model-comparison preparation"
    if phase.startswith("adaptive_source"):
        return "Source-preservation fallback"
    if phase.startswith("adaptive_shot"):
        return "Shot assembly"
    return phase.replace("_", " ").capitalize() or "Other measured process"


def _shot_execution(plan: dict[str, Any], shot: dict[str, Any], quality: Any) -> dict[str, Any]:
    """Describe the delivered recipe, never promote analysis suggestions to proof."""
    unknown = {"status": "not_verified", "scope": "analysis_only", "units": []}
    if not isinstance(quality, dict):
        return unknown
    if plan.get("fingerprint") and quality.get("adaptive_fingerprint") != plan["fingerprint"]:
        return unknown
    if quality.get("status") == "SOURCE_PRESERVATION_FALLBACK" and quality.get("actual_route") == "source_preserved":
        return {"status": "rendered", "scope": "source_preservation_fallback", "units": [
            {"route": "source_preserved", "effective_controls": {}, "crop": None}]}
    if quality.get("status") != "PASS_OBJECTIVE_GUARDS" or quality.get("actual_route") != plan.get("route"):
        return unknown
    model = (plan.get("enhancement_model_choice") or {}).get("model", "prob-4")
    automatic_model = bool(plan.get("enhancement_model_choice"))
    if plan.get("route") != "shot_sequence":
        return {"status": "rendered", "scope": "batched_source_recipe", "units": [
            {"route": plan.get("route"), "effective_controls": plan.get("effective_controls", {})
                if plan.get("route") != "source_preserved" and not automatic_model else {},
             "model": model if plan.get("route") != "source_preserved" else "none", "crop": plan.get("crop")}]}
    units = []
    try:
        for unit in plan.get("render_units", []):
            if unit["start_frame"] < shot["end_frame"] and unit["end_frame"] > shot["start_frame"]:
                executed = {k: unit.get(k) for k in
                    ("start_frame", "end_frame", "route", "crop", "effective_controls")}
                executed["model"] = model if unit.get("route") != "source_preserved" else "none"
                if automatic_model or unit.get("route") == "source_preserved":
                    executed["effective_controls"] = {}
                units.append(executed)
    except (KeyError, TypeError):
        return unknown
    return {"status": "rendered" if units else "not_verified", "scope": "verified_render_units", "units": units}


def summarize_run(
    manifest: dict[str, Any], *, elapsed_seconds: Any = None
) -> dict[str, Any]:
    wall = _number(
        manifest.get("elapsed_seconds") if elapsed_seconds is None else elapsed_seconds
    )
    stages: dict[str, float] = defaultdict(float)
    startup = 0.0
    process_details = []
    for row in manifest.get("process_timings") or []:
        if not isinstance(row, dict):
            continue
        seconds = _number(row.get("seconds"))
        if seconds is None:
            continue
        phase = str(row.get("phase") or "")
        stages[_stage_name(phase)] += seconds
        first = _number(row.get("first_frame_seconds"))
        if first is not None and first <= seconds:
            startup += first
        process_details.append(
            {
                "phase": phase,
                "seconds": seconds,
                "first_frame_seconds": first,
                "emitted_frames": row.get("emitted_frames"),
                "status": row.get("status", "not_recorded"),
                "exit_code": row.get("exit_code"),
            }
        )
    shots, plans, cache_hits = [], [], 0
    for item in manifest.get("adaptive_plans") or []:
        if not isinstance(item, dict) or not isinstance(item.get("plan"), dict):
            continue
        plan = item["plan"]
        source_key = str(item.get("source_path") or "").replace("\\", "/").casefold()
        qualities = [value for key, value in (manifest.get("adaptive_quality") or {}).items()
            if source_key and (str(key).replace("\\", "/").casefold() == source_key
                or str(key).replace("\\", "/").casefold().endswith(":" + source_key))]
        quality = qualities[0] if len(qualities) == 1 else None
        cache_hits += int(plan.get("cache_hit") is True)
        seconds = _number(plan.get("analysis_seconds"))
        if seconds is not None and plan.get("cache_hit") is not True:
            stages["Scene analysis / protection / subject masks"] += seconds
        rate = _number(plan.get("source_fps"))
        plans.append(
            {
                "source_path": item.get("source_path"),
                "schema": plan.get("schema"),
                "fingerprint": plan.get("fingerprint"),
                "route": plan.get("route"),
                "processed_area_ratio": plan.get("processed_area_ratio"),
                "subject_refinement": plan.get("subject_refinement"),
                "caption_protection": plan.get("caption_protection"),
                "enhancement_model_choice": plan.get("enhancement_model_choice"),
                "analysis_efficiency": plan.get("analysis_efficiency"),
                "fresh_analysis": plan.get("cache_hit") is not True,
                "engine": plan.get("identity", {})
                .get("dependencies", {})
                .get("engine", {}),
            }
        )
        for index, shot in enumerate(plan.get("shots") or []):
            start, end = (
                _number(shot.get("start_frame")),
                _number(shot.get("end_frame")),
            )
            shots.append(
                {
                    "source_path": item.get("source_path"),
                    "shot": index + 1,
                    "start_frame": shot.get("start_frame"),
                    "end_frame": shot.get("end_frame"),
                    "start_seconds": start / rate
                    if start is not None and rate
                    else None,
                    "end_seconds": end / rate if end is not None and rate else None,
                    "route": shot.get("route"),
                    "crop": shot.get("crop"),
                    "processed_area_ratio": shot.get("processed_area_ratio"),
                    "effective_controls": shot.get("effective_controls", {}),
                    "spatial_quality": shot.get("spatial_quality", {"scope": "not_recorded"}),
                    "visual_treatment": shot.get("visual_treatment", {"scope": "not_recorded"}),
                    "motion_evidence": shot.get("motion_evidence", {"scope": "not_recorded"}),
                    "execution": _shot_execution(plan, shot, quality),
                    "subject_refinement": shot.get(
                        "subject_refinement", {"status": "not_recorded"}
                    ),
                }
            )
    for quality in (manifest.get("adaptive_quality") or {}).values():
        if not isinstance(quality, dict):
            continue
        for candidate in [
            quality,
            quality.get("fallback", {}),
            quality.get("rejected_candidate", {}),
            quality.get("temporal_protection", {}),
        ]:
            seconds = (
                _number(candidate.get("seconds"))
                if isinstance(candidate, dict)
                else None
            )
            if seconds is not None:
                stages["Image / caption / cut quality checks"] += seconds
    duration, sources = 0.0, []
    for source in manifest.get("ready_sources") or []:
        info = source.get("probe") or {}
        value = _number(info.get("video_pts_end_exclusive"))
        origin = _number(info.get("video_pts_start"))
        seconds = (
            value - origin
            if value is not None and origin is not None
            else _number(info.get("duration"))
        )
        if seconds is not None and seconds > 0:
            duration += seconds
        sources.append(
            {
                "path": source.get("source_path"),
                "duration_seconds": seconds,
                "width": info.get("width"),
                "height": info.get("height"),
                "fps": info.get("fps_ratio"),
                "frames": info.get("video_frame_count"),
            }
        )
    outputs, output_frames, seen_outputs = [], 0, set()
    for completed in manifest.get("completed_delivery_outputs") or []:
        info = completed.get("probe") or {}
        path = str(info.get("path") or "")
        if path and path in seen_outputs:
            continue
        seen_outputs.add(path)
        frames = _number(info.get("video_frame_count"))
        if frames is not None and frames.is_integer():
            output_frames += int(frames)
        outputs.append(
            {
                k: info.get(k)
                for k in (
                    "path",
                    "width",
                    "height",
                    "fps_ratio",
                    "video_frame_count",
                    "size_bytes",
                )
            }
        )
    accounted = sum(stages.values())
    valid_accounting = wall is None or accounted <= wall + max(0.1, wall * 0.005)
    stage_rows = [
        {"name": name, "seconds": round(seconds, 4)} for name, seconds in stages.items()
    ]
    return {
        "schema": SCHEMA,
        "elapsed_seconds": wall,
        "source_duration_seconds": duration or None,
        "source_count": len(sources),
        "output_frames": output_frames or None,
        "seconds_per_video_second": wall / duration
        if wall is not None and duration
        else None,
        "realtime_multiple": duration / wall if wall and duration else None,
        "output_frames_per_wall_second": output_frames / wall
        if wall and output_frames
        else None,
        "stages": stage_rows,
        "processes": process_details,
        "enhancement_model_trials": manifest.get("adaptive_enhancement_calibration", {}),
        "accounted_seconds": round(accounted, 4),
        "unclassified_seconds": round(max(0, wall - accounted), 4)
        if wall is not None and valid_accounting
        else None,
        "accounting_status": "measured_partial"
        if valid_accounting
        else "overlapping_or_inconsistent_do_not_sum",
        "startup_seconds_included_in_stages": round(startup, 4),
        "analysis_cache_hits": cache_hits,
        "sources": sources,
        "outputs": outputs,
        "shots": shots,
        "plans": plans,
        "resources": {
            "start": manifest.get("resource_start"),
            "end": manifest.get("resource_end"),
            "peak_gpu_memory_mib": None,
            "peak_ram_bytes": None,
            "measurement_scope": "snapshots_only_not_peaks",
        },
        "runtime": manifest.get("resolved_executable_paths", {}),
        "limits": [
            "Wall time includes startup, load contention and orchestration.",
            "Startup-to-first-frame is already included in process stages.",
            "Unknown stages/resources are not zero; no idle-machine or universal speed claim.",
            "Scene settings and accepted masks are evidence, not perceptual-quality approval.",
        ],
    }


def attach_delivery_insights(result: dict[str, Any], *, job_dir: Any) -> dict[str, Any]:
    """Bounded current-job projection; no scanning of the global job directory."""
    from pathlib import Path

    root = Path(job_dir).resolve().parent
    children = []
    combined: dict[str, Any] = {
        "process_timings": [],
        "adaptive_plans": [],
        "ready_sources": [],
        "completed_delivery_outputs": [],
        "adaptive_quality": {},
        "adaptive_enhancement_calibration": {},
    }
    for row in result.get("sources") or []:
        if not isinstance(row, dict):
            continue
        # Reused media did not incur its historical render cost on this attempt.
        if row.get("media_reused"):
            children.append(
                {
                    "source_path": row.get("source_path"),
                    "media_reused": True,
                    "summary": None,
                    "reason": "historical_compute_not_counted_in_current_attempt",
                }
            )
            continue
        path = Path(str(row.get("child_manifest_path") or ""))
        if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
            continue
        try:
            path.resolve().relative_to(root)
        except ValueError:
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            continue
        for key in (
            "process_timings",
            "adaptive_plans",
            "ready_sources",
            "completed_delivery_outputs",
        ):
            combined[key].extend(manifest.get(key) or [])
        combined["adaptive_quality"].update(
            {
                f"{path.parent.name}:{k}": v
                for k, v in (manifest.get("adaptive_quality") or {}).items()
            }
        )
        combined["adaptive_enhancement_calibration"].update({
            f"{path.parent.name}:{key}": value
            for key, value in (manifest.get("adaptive_enhancement_calibration") or {}).items()
        })
        if "resource_start" not in combined:
            combined["resource_start"] = manifest.get("resource_start")
        combined["resource_end"] = manifest.get("resource_end")
        combined["resolved_executable_paths"] = manifest.get(
            "resolved_executable_paths", {}
        )
        children.append(
            {
                "source_path": row.get("source_path"),
                "manifest_path": str(path),
                "media_reused": False,
                "summary": summarize_run(manifest),
            }
        )
    return {
        "schema": SCHEMA,
        "status": result.get("state"),
        "total_elapsed_seconds": _number(result.get("elapsed_seconds")),
        "summary": summarize_run(
            combined, elapsed_seconds=result.get("elapsed_seconds")
        ),
        "children": children,
        "authority": "diagnostic_projection_not_completion_authority",
    }


def render_report_html(
    summary: dict[str, Any], *, title: str = "Smart Finish · Run report"
) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    def fmt(value: Any) -> str:
        return "Not measured" if value is None else f"{value:,.3f}"
    metrics = [
        ("Total elapsed seconds", summary.get("elapsed_seconds")),
        ("Video length (seconds)", summary.get("source_duration_seconds")),
        ("Seconds per video second", summary.get("seconds_per_video_second")),
        (
            "Delivered frames per wall second",
            summary.get("output_frames_per_wall_second"),
        ),
    ]
    cards = "".join(
        f"<div><small>{esc(label)}</small><b>{esc(fmt(value))}</b></div>"
        for label, value in metrics
    )
    stages = list(summary.get("stages") or []) + [
        {
            "name": "Unclassified / orchestration",
            "seconds": summary.get("unclassified_seconds"),
        }
    ]
    rows = "".join(
        f"<tr><td>{esc(s['name'])}</td><td>{esc(fmt(s['seconds']))} s</td></tr>"
        for s in stages
    )
    shots = "".join(
        "<tr>"
        + "".join(
            f"<td>{esc(s.get(k))}</td>"
            for k in (
                "shot",
                "start_seconds",
                "end_seconds",
                "route",
                "effective_controls",
                "execution",
                "subject_refinement",
            )
        )
        + "</tr>"
        for s in summary.get("shots") or []
    )
    efficiency_rows = []
    for plan in summary.get("plans") or []:
        efficiency = plan.get("analysis_efficiency")
        if not isinstance(efficiency, dict):
            continue
        efficiency_rows.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in (
            plan.get("source_path"),
            "This attempt" if plan.get("fresh_analysis") else "Cached artifact; historical counts",
            efficiency.get("decoded_frames", "Not recorded"),
            efficiency.get("exact_repeat_hits", "Not recorded"),
            efficiency.get("detail_mask_evaluations", "Not recorded"),
            efficiency.get("caption_mask_evaluations", "Not recorded"),
        )) + "</tr>")
    efficiency_html = (
        "<h2>Analysis work avoided</h2><p>Exact repeats reuse spatial analysis only. Every input frame and timestamp is retained; no interpolation or GPU speedup is inferred.</p>"
        "<div class=scroll><table><tr><th>Source</th><th>Measurement</th><th>Frames decoded</th><th>Exact repeats reused</th><th>Detail evaluations</th><th>Caption evaluations</th></tr>"
        + "".join(efficiency_rows) + "</table></div>"
    ) if efficiency_rows else ""
    return (
        "<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)}</title><style>body{{background:#11151a;color:#e9eef4;font:15px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:24px}}"
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.cards div{background:#1b232c;padding:18px;border:1px solid #394451;border-radius:12px}"
        "b{display:block;font-size:28px}small,p{color:#b9c5d1}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #394451;padding:10px;overflow-wrap:anywhere}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}.scroll{overflow-x:auto}</style>"
        f"<h1>{esc(title)}</h1><div class=cards>{cards}</div><h2>Stage breakdown</h2><table>{rows}</table>"
        f"<p>Time to first frame: {esc(fmt(summary.get('startup_seconds_included_in_stages')))} s, already included above—not added again.</p>"
        + efficiency_html +
        "<h2>Per-shot decisions</h2><p>Suggested settings are analysis, not an execution claim. Rendered settings below come from the verified output route; short shots may share one conservative model call.</p><div class=scroll><table><tr><th>Shot</th><th>Start</th><th>End</th><th>Suggested route</th><th>Suggested settings</th><th>Verified execution</th><th>Subject mask</th></tr>"
        f"{shots}</table></div><p>Resource values below are snapshots, not peaks. Missing measurements are unknown. No aesthetic or benchmark guarantee is inferred.</p>"
        f"<details><summary>Source-bound evidence</summary><pre>{esc(json.dumps(summary, indent=2))}</pre></details></html>"
    )
