"""Bounded local enhancement auditions inside the incumbent Topaz pipeline.

Frame/shot measurements nominate a candidate; actual moving outputs decide.
No model downloads, external AI calls, watcher, new queue or model repository.
The reference remains authoritative when cost, image or runtime evidence fails.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any

SCHEMA = "spiffy_enhancement_audition_v1"
REFERENCE = "prob-4"
CANDIDATE = "ahq-12"
PILOT_FRAMES = 8
MAX_WINDOWS = 2
MAX_TOTAL_SECONDS = 120.0


class ModelSelectionIntegrityError(RuntimeError):
    """Changed source/runtime/model evidence must not become an optional fallback."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _cancel(writer: Any) -> None:
    from shared.gpu_lock import GpuLockCancelled
    if getattr(writer, "cancel_requested", False):
        raise GpuLockCancelled("Enhancement audition cancelled.")


def plan_audition(plan: dict[str, Any]) -> dict[str, Any]:
    """Nominate at most two source-bound motion witnesses, before GPU work.

    Every photographic shot must be represented in this initially qualified
    scope. Larger montages keep Proteus rather than generalizing from one face.
    These thresholds bound experimentation, not proof of recoverable detail.
    """
    def no(reason: str) -> dict[str, Any]:
        return {"eligible": False, "reason": reason, "windows": []}
    if plan.get("policy", {}).get("model_search", "auto") == "off":
        return no("explicitly_disabled")
    if plan.get("route") == "source_preserved":
        return no("no_enhancement_needed")
    count = int(plan["frame_count"])
    # No model decision from a partial shot list: every original frame must
    # belong to one contiguous, non-overlapping original shot, including holds.
    cursor = 0
    for row in plan.get("shots", []):
        start, end = row.get("start_frame"), row.get("end_frame")
        if (type(start) is not int or type(end) is not int
                or start != cursor or not start < end <= count):
            return no("shot_coverage_incomplete")
        cursor = end
    if cursor != count:
        return no("shot_coverage_incomplete")
    shots = [s for s in plan.get("shots", []) if s.get("route") != "source_preserved"]
    if not shots:
        return no("no_enhancement_needed")
    if any(isinstance(shot.get("visual_treatment"), dict)
           and shot["visual_treatment"].get("schema") == "spiffy_visual_treatment_v1"
           and shot["visual_treatment"].get("mode") == "graphics_edge_guard"
           for shot in shots):
        # A single global model choice is not source-safe when some source
        # frames are authored text/graphics and others are photography. Save
        # two expensive starts and retain tested Proteus/source protection.
        return no("graphic_content_preserve_existing_model")
    if len(shots) > MAX_WINDOWS:
        return no("too_many_distinct_shots")
    rate = Fraction(plan["source_fps"])
    active_frames = sum(s["end_frame"] - s["start_frame"] for s in shots)
    if (rate <= 0 or active_frames < 2 * PILOT_FRAMES * len(shots) * 5
            or Fraction(active_frames, 1) / rate < 10):
        # Two cold model starts recently cost 15-21s for a four-second
        # Snowfall clip, yet startup-aware evidence projected <1s savings.
        # Short scenes keep the known Proteus route instead of paying more
        # to choose a model than to process the whole Short. The minimum
        # is a conservative source-duration admission gate, not a universal
        # per-model speed claim; longer clips still require actual pilots.
        return no("pilot_cost_budget")
    # A two-model 4K trial can spend more time and VRAM than an entire Short.
    # The regular 4K enhancement route remains available; only the optional
    # unqualified model contest is skipped until real 4K timing/quality evidence.
    if (plan["target_width"] * plan["target_height"] > 1920 * 1080
            or max(plan["target_width"], plan["target_height"]) > 4096):
        return no("high_resolution_model_pilot_not_qualified")
    windows = []
    for shot in shots:
        a, b = shot["start_frame"], shot["end_frame"]
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= count:
            raise ValueError("Invalid model-audition source window.")
        if b - a < PILOT_FRAMES:
            return no("short_unrepresented_shot")
        measurements = shot.get("measurements") or []
        if not measurements:
            return no("missing_pixel_evidence")
        values = []
        for row in measurements:
            block = row.get("blockiness")
            if type(block) not in (int, float) or not math.isfinite(block) or block < 0:
                return no("missing_pixel_evidence")
            values.append(float(block))
        if max(values) > 1.5:
            return no("retain_tunable_model_for_damage")
        peak = int(shot.get("motion_peak_frame", (a + b) // 2))
        spatial = shot.get("spatial_quality")
        if isinstance(spatial, dict) and spatial.get("schema") == "spiffy_shot_spatial_summary_v1":
            representative = spatial.get("representative_frame")
            score = spatial.get("priority_score")
            if (type(representative) is int and a <= representative < b
                    and type(score) in (int, float) and math.isfinite(score) and score >= .05):
                peak = representative
        first = max(a, min(peak - PILOT_FRAMES // 2, b - PILOT_FRAMES))
        windows.append([first, first + PILOT_FRAMES])
    return {"eligible": True, "reason": "bounded_clean_source_comparison", "windows": windows,
            "active_frames": active_frames,
            "sample_policy": "measured_spatial_priority_with_motion_fallback",
            "candidate": CANDIDATE, "reference": REFERENCE,
            "evidence": "actual_source_shot_and_frame_measurements_not_vlm_quality_labels"}


def resource_admission(snapshot: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    # Serial native Topaz only: do not also load a multimodal/segmentation model.
    pixels = int(plan["target_width"]) * int(plan["target_height"])
    needed = 3 * 1024**3 + pixels * 64
    free_ram = snapshot.get("ram_available_bytes")
    reason = ""
    if type(free_ram) is not int or free_ram < needed:
        reason = "system_memory_headroom"
    elif snapshot.get("platform") not in {"win32", "linux"}:
        reason = "native_model_audition_not_qualified_on_platform"
    elif (type(snapshot.get("gpu_memory_free_mib")) not in (int, float)
            or not math.isfinite(snapshot["gpu_memory_free_mib"]) or snapshot["gpu_memory_free_mib"] < 4608):
        reason = "gpu_memory_headroom"
    elif (type(snapshot.get("gpu_utilization_percent")) not in (int, float)
            or not math.isfinite(snapshot["gpu_utilization_percent"]) or not 0 <= snapshot["gpu_utilization_percent"] <= 40):
        reason = "gpu_currently_busy"
    return {"allowed": not reason, "reason": reason or "bounded_serial_trial",
            "required_ram_bytes": needed, "snapshot": snapshot}


def choose_model(comparison: dict[str, Any], reference_seconds: float,
                 candidate_seconds: float, *, pilot_frames: int, total_frames: int,
                 reference_first_frame_seconds: float | None = None,
                 candidate_first_frame_seconds: float | None = None,
                 pilot_starts: int = 1,
                 production_starts: int = 1) -> dict[str, Any]:
    for number in (reference_seconds, candidate_seconds):
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError("Model selection requires positive finite measured times.")
    if type(pilot_frames) is not int or type(total_frames) is not int or not 0 < pilot_frames <= total_frames:
        raise ValueError("Model selection requires valid frame coverage.")
    if type(comparison.get("frames")) is not int or comparison["frames"] != pilot_frames:
        raise ValueError("Model decision must cover the exact compared pilot frames.")
    if type(pilot_starts) is not int or not 0 < pilot_starts < pilot_frames:
        raise ValueError("Model selection requires complete measured startup/frame coverage.")
    if type(production_starts) is not int or not 1 <= production_starts <= total_frames:
        raise ValueError("Model selection requires valid production model invocation count.")
    evidence_present = (reference_first_frame_seconds is not None
                        or candidate_first_frame_seconds is not None)
    if evidence_present != (reference_first_frame_seconds is not None
                           and candidate_first_frame_seconds is not None):
        raise ValueError("Model startup timing evidence must be a complete measured pair.")
    reason = "quality_not_qualified"
    projected = (reference_seconds - candidate_seconds) * (total_frames / pilot_frames - 1)
    testing_cost = reference_seconds + candidate_seconds
    metric_details: dict[str, Any] = {}
    if evidence_present:
        # Pilot totals contain expensive model initialization once per
        # separate process. Do not multiply those cold starts by the number
        # of remaining output frames, which could make a slower model look
        # hundreds of seconds faster on a long source.
        for number, total in ((reference_first_frame_seconds, reference_seconds),
                              (candidate_first_frame_seconds, candidate_seconds)):
            if (type(number) not in (int, float) or not math.isfinite(number)
                    or not 0 < number < total):
                raise ValueError("Invalid native first-frame startup timing evidence.")
        rest = pilot_frames - pilot_starts
        reference_steady = (reference_seconds - reference_first_frame_seconds) / rest
        candidate_steady = (candidate_seconds - candidate_first_frame_seconds) / rest
        reference_start = reference_first_frame_seconds / pilot_starts
        candidate_start = candidate_first_frame_seconds / pilot_starts
        # Native shot-sequence delivery may start a new enhancement model
        # process for each admitted render unit, not once for the source.
        # Keep the true verified execution count in the cost projection.
        projected = ((reference_start - candidate_start) * production_starts
                     + (reference_steady - candidate_steady)
                     * max(0, total_frames - production_starts))
        metric_details = {
            "reference_first_frame_seconds": reference_first_frame_seconds,
            "candidate_first_frame_seconds": candidate_first_frame_seconds,
            "reference_steady_seconds_per_frame": reference_steady,
            "candidate_steady_seconds_per_frame": candidate_steady,
            "pilot_starts": pilot_starts,
            "production_starts_assumed": production_starts,
            "projection_scope": "measured_first_frame_and_steady_state_estimate_not_full_export_proof",
        }
    if comparison.get("status") == "PASS_SOURCE_AND_REFERENCE_GUARDS" and not comparison.get("reasons"):
        if evidence_present:
            if (candidate_steady >= reference_steady * 0.95
                    and candidate_start >= reference_start * 0.95):
                reason = "no_material_measured_speed_gain"
            elif projected < testing_cost * 2:
                reason = "startup_aware_speed_not_repaid"
            else:
                reason = "qualified_local_candidate_with_cost_margin"
        elif production_starts > 1:
            reason = "missing_per_unit_startup_evidence"
        elif candidate_seconds > reference_seconds * 0.85:
            reason = "no_material_measured_speed_gain"
        elif projected < testing_cost * 2:
            reason = "pilot_not_repaid"
        else:
            reason = "qualified_local_candidate_with_cost_margin"
    accepted = reason == "qualified_local_candidate_with_cost_margin"
    return {"schema": SCHEMA, "model": CANDIDATE if accepted else REFERENCE,
            "reason": reason, "reference_seconds": reference_seconds,
            "candidate_seconds": candidate_seconds, "pilot_frames": pilot_frames,
            "total_frames": total_frames, "projected_remaining_savings_seconds": projected,
            "production_starts_assumed": production_starts,
            "projection_scope": "linear_admission_estimate_not_measured_complete_export",
            **metric_details}


def _production_model_starts(plan: dict[str, Any]) -> int:
    """Count actual Topaz launches in canonical shot-render units."""
    if plan.get("route") != "shot_sequence":
        return 1
    units = plan.get("render_units")
    if (not isinstance(units, list) or not units
            or any(not isinstance(unit, dict)
                   or unit.get("route") not in {
                       "source_preserved", "masked_crop", "masked_full_frame"}
                   for unit in units)):
        raise ValueError("Model selection needs valid production render units.")
    count = sum(unit["route"] != "source_preserved" for unit in units)
    if not count:
        raise ValueError("Model selection has no active production render units.")
    return count


def _source_frames(path: Path, start: int, end: int):
    import av
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        for ordinal, frame in enumerate(container.decode(video=0)):
            if ordinal >= end:
                break
            if ordinal >= start:
                yield frame


def _native_resolution_detail_witness(reference_frame: Any, candidate_frame: Any,
                                      detail_mask: Any, caption_mask: Any,
                                      previous_residual: Any) -> tuple[dict[str, Any], Any]:
    """Detect gross hallucinated/suppressed native pixels before downsampling.

    A fine checkerboard can cancel during source-size reduction, creating a
    fake quality PASS. This guard keeps the real delivered image dimensions:
    high-frequency luma, protected glyph pixels, and source-aligned
    model-difference flicker. Candidate quality is *relative* to the
    incumbent Proteus model, not an unsupported assertion of ground truth.
    Only the already decoded eight pilot frames are inspected; no new model
    pass, model download or third-party library is needed.
    """
    import cv2
    import numpy as np

    ref = reference_frame.to_ndarray(format="rgb24")
    cand = candidate_frame.to_ndarray(format="rgb24")
    if ref.shape != cand.shape or ref.dtype != np.uint8 or cand.dtype != np.uint8:
        raise ValueError("Native pilot frames changed pixel geometry or type.")
    height, width = ref.shape[:2]
    if detail_mask.shape != caption_mask.shape or detail_mask.ndim != 2:
        raise ValueError("Native pilot masks do not agree with each other.")
    active = cv2.resize(detail_mask, (width, height),
                        interpolation=cv2.INTER_NEAREST) > 8
    captions = cv2.resize(caption_mask, (width, height),
                          interpolation=cv2.INTER_NEAREST) > 0
    eligible = active & ~captions
    eligible_count = int(np.count_nonzero(eligible))
    # Assess authored glyph cores SEPARATELY from the huge protected
    # background. A one-percent video title can be entirely destroyed
    # while p99 over the whole unenhanced frame remains zero.
    protected_delta = np.abs(cand.astype(np.int16) - ref.astype(np.int16))
    glyph_p99 = (float(np.percentile(protected_delta[captions], 99))
                 if captions.any() else 0.0)
    untouched = ~active & ~captions
    backdrop_p99 = (float(np.percentile(protected_delta[untouched], 99))
                    if untouched.any() else 0.0)
    native_p99 = max(glyph_p99, backdrop_p99)
    reasons = []
    if glyph_p99 > 16.0 or backdrop_p99 > 16.0:
        reasons.append("native_protected_pixels_changed")
    # Keep source-bound mask admission. Do not manufacture an ROI from
    # high-frequency noise in a frame which the detail system protected.
    if eligible_count < 128:
        return ({"eligible_pixels": eligible_count,
                 "native_detail_energy_ratio": None,
                 "native_reference_energy": None,
                 "native_candidate_energy": None,
                 "native_protected_p99": round(native_p99, 6),
                 "native_glyph_p99": round(glyph_p99, 6),
                 "native_backdrop_p99": round(backdrop_p99, 6),
                 "native_residual_flicker": None,
                 "status": "REJECT" if reasons else "no_qualified_detail_region",
                 "reasons": reasons}, None)

    rg = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cg = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY).astype(np.float32)
    native_ref = cv2.Laplacian(rg, cv2.CV_32F, ksize=1)
    native_cand = cv2.Laplacian(cg, cv2.CV_32F, ksize=1)
    ref_energy = float(np.mean(np.abs(native_ref[eligible])))
    cand_energy = float(np.mean(np.abs(native_cand[eligible])))
    ratio = cand_energy / max(ref_energy, 0.5)

    # Detect extreme local spikes too: small hallucinated eye/skin regions
    # must not be hidden by a large visually unchanged background.
    largest_tile_gain = 0.0
    for row in range(4):
        y0, y1 = height * row // 4, height * (row + 1) // 4
        for col in range(4):
            x0, x1 = width * col // 4, width * (col + 1) // 4
            area = eligible[y0:y1, x0:x1]
            if np.count_nonzero(area) < 64:
                continue
            e1 = float(np.mean(np.abs(native_ref[y0:y1, x0:x1][area])))
            e2 = float(np.mean(np.abs(native_cand[y0:y1, x0:x1][area])))
            largest_tile_gain = max(largest_tile_gain, e2 - e1)

    # The signed high-frequency difference between models should not
    # drastically flicker relative to its own preceding difference. Since
    # both models see the same PTS, real picture motion largely cancels.
    residual = native_cand - native_ref
    flicker = (
        float(np.mean(np.abs(residual - previous_residual)[eligible]))
        if previous_residual is not None else 0.0
    )
    if cand_energy > ref_energy * 1.65 + 5.0 or largest_tile_gain > 18.0:
        reasons.append("native_high_frequency_artifacts")
    if ref_energy >= 8.0 and cand_energy < ref_energy * 0.55 - 3.0:
        reasons.append("native_texture_suppressed")
    if flicker > 25.0:
        reasons.append("native_temporal_detail_instability")
    return ({
        "eligible_pixels": eligible_count,
        "native_detail_energy_ratio": round(ratio, 6),
        "native_reference_energy": round(ref_energy, 6),
        "native_candidate_energy": round(cand_energy, 6),
        "native_local_energy_gain": round(largest_tile_gain, 6),
        "native_protected_p99": round(native_p99, 6),
        "native_glyph_p99": round(glyph_p99, 6),
        "native_backdrop_p99": round(backdrop_p99, 6),
        "native_residual_flicker": round(flicker, 6),
        "status": "REJECT" if reasons else "PASS_RELATIVE_NATIVE_GUARDS",
        "reasons": reasons,
        "scope": "native_working_scale_relative_reference_not_true_high_resolution_ground_truth",
    }, residual)


def compare_pilot(source: Path, reference: Path, candidate: Path, plan: dict[str, Any],
                  window: tuple[int, int] | list[int], writer: Any, *, deadline: float | None = None) -> dict[str, Any]:
    """Compare all pilot frames at source display size, with regional guards.

    The incumbent is a preservation reference, NOT high-resolution ground truth.
    Whole-image agreement cannot hide a smoothed face: local tiles, detail energy,
    protected source pixels and temporal differences have independent rejections.
    """
    import av
    import cv2
    import numpy as np
    from adaptive_enhancement import caption_protection_mask, detail_protection_mask

    a, b = window
    rate = Fraction(plan["source_fps"])
    sw, sh = int(plan["source_width"]), int(plan["source_height"])
    target = (int(plan["target_width"]), int(plan["target_height"]))
    rows, reasons = [], set()
    previous = None
    previous_native_residual = None
    native_screened = 0
    native_skipped = 0
    worst_native_ratio = 0.0
    worst_native_flicker = 0.0
    started = time.monotonic()
    with av.open(str(reference)) as ref, av.open(str(candidate)) as cand:
        ref.streams.video[0].codec_context.thread_count = 1
        cand.streams.video[0].codec_context.thread_count = 1
        for index, trio in enumerate(zip_longest(_source_frames(source, a, b), ref.decode(video=0), cand.decode(video=0))):
            _cancel(writer)
            if (index >= PILOT_FRAMES or time.monotonic() - started > 45
                    or (deadline is not None and time.monotonic() >= deadline)):
                raise RuntimeError("Model comparison exceeded the bounded pilot extent.")
            if any(frame is None or frame.pts is None for frame in trio):
                raise RuntimeError("Model pilot frame count/clock mismatch.")
            original, one, two = trio
            if (Fraction(original.pts) * original.time_base != Fraction(a + index) / rate
                    or Fraction(one.pts) * one.time_base != Fraction(index) / rate
                    or Fraction(two.pts) * two.time_base != Fraction(index) / rate):
                raise RuntimeError("Model pilot presentation clock mismatch.")
            if (one.width, one.height) != target or (two.width, two.height) != target:
                raise RuntimeError("Model pilot working geometry mismatch.")
            rgb = original.to_ndarray(format="rgb24")
            x = one.reformat(width=sw, height=sh, format="rgb24", interpolation="LANCZOS").to_ndarray()
            y = two.reformat(width=sw, height=sh, format="rgb24", interpolation="LANCZOS").to_ndarray()
            detail, _ = detail_protection_mask(rgb, collect_metrics=False)
            glyph, _ = caption_protection_mask(rgb)
            native, previous_native_residual = _native_resolution_detail_witness(
                one, two, detail, glyph, previous_native_residual)
            if native["native_detail_energy_ratio"] is None:
                native_skipped += 1
            else:
                native_screened += 1
                worst_native_ratio = max(worst_native_ratio, native["native_detail_energy_ratio"])
                worst_native_flicker = max(worst_native_flicker, native["native_residual_flicker"])
            reasons.update(native.get("reasons", []))
            protected = (detail <= 2) | (glyph > 0)
            signed = y.astype(np.float32) - x.astype(np.float32)
            delta = np.abs(signed)
            tiles = [float(tile.mean()) for band in np.array_split(delta, 8, axis=0)
                     for tile in np.array_split(band, 8, axis=1)]
            gx, gy = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) for image in (x, y)]
            hx = np.abs(gx - cv2.GaussianBlur(gx, (0, 0), 1.0))
            hy = np.abs(gy - cv2.GaussianBlur(gy, (0, 0), 1.0))
            active = detail > 2
            reference_energy = float(hx[active].mean()) if active.any() else 0.0
            ratio = float(hy[active].mean()) / reference_energy if reference_energy >= 1 else None
            source_error = np.abs(y.astype(np.float32) - rgb.astype(np.float32))
            reference_error = np.abs(x.astype(np.float32) - rgb.astype(np.float32))
            q = float(np.percentile(source_error[protected], 99)) if protected.any() else 0.0
            qref = float(np.percentile(reference_error[protected], 99)) if protected.any() else 0.0
            temporal = float(np.abs(signed - previous).mean()) if previous is not None else 0.0
            row = {"frame": index, "source_frame": a + index, "mae": float(delta.mean()),
                   "p99": float(np.percentile(delta, 99)), "max_tile_mae": max(tiles),
                   "large_error_fraction": float(np.mean(delta > 24)), "texture_ratio": ratio,
                   "protected_source_p99": q, "reference_protected_p99": qref,
                   "temporal_error_change": temporal,
                   "native_detail_energy_ratio": native["native_detail_energy_ratio"],
                   "native_protected_p99": native["native_protected_p99"],
                   "native_residual_flicker": native["native_residual_flicker"],
                   "native_detail_status": native["status"]}
            rows.append(row)
            if row["mae"] > 1.5 or row["p99"] > 16 or row["max_tile_mae"] > 4 or row["large_error_fraction"] > .003:
                reasons.add("local_detail_changed")
            if ratio is not None and not .90 <= ratio <= 1.10:
                reasons.add("texture_loss_or_oversharpening")
            if q > max(8, qref + 1):
                reasons.add("protected_source_pixels_changed")
            if temporal > 1.5:
                reasons.add("temporal_reference_disagreement")
            previous = signed
    if len(rows) != b - a or len(rows) < 8:
        raise RuntimeError("Model comparison lacks complete temporal evidence.")
    if (target[0] > sw or target[1] > sh) and native_screened < max(1, len(rows) // 2):
        # A second model cannot win on downsampled agreement alone if the
        # source provides almost no admitted native-detail evidence.
        reasons.add("native_detail_evidence_insufficient")
    return {"status": "REJECT" if reasons else "PASS_SOURCE_AND_REFERENCE_GUARDS",
            "reasons": sorted(reasons), "frames": len(rows), "samples": rows,
            "native_detail": {"screened_frames": native_screened, "skipped_no_active_mask": native_skipped,
                              "worst_energy_ratio": round(worst_native_ratio, 6),
                              "worst_residual_flicker": round(worst_native_flicker, 6),
                              "scale": list(target),
                              "reference": REFERENCE, "candidate": CANDIDATE,
                              "policy": "source_masked_native_frequency_and_protected_pixel_guard"},
            "seconds": round(time.monotonic() - started, 4),
            "scope": "bounded native-resolution and source/local/temporal preservation, not recovered-detail or human taste approval"}


def _pilot_plan(plan: dict[str, Any], window: list[int], model: str) -> dict[str, Any]:
    from adaptive_enhancement import plan_fingerprint
    matching = [s for s in plan["shots"] if s["start_frame"] <= window[0] < window[1] <= s["end_frame"]]
    if len(matching) != 1:
        raise ValueError("A model witness must belong to exactly one shot.")
    if plan.get("route") == "shot_sequence":
        units = [u for u in plan.get("render_units", [])
                 if u["start_frame"] <= window[0] < window[1] <= u["end_frame"]]
        if len(units) != 1:
            raise ValueError("A model witness must belong to one executed render unit.")
        actual = units[0]
        params = "".join(f":{k}={v}" for k, v in actual["effective_controls"].items())
    else:
        actual = plan
        params = plan["effective_params"]
    pilot = {**plan, "route": actual["route"], "crop": list(actual["crop"]),
             "render_units": [], "execution_window": {"start_frame": window[0], "end_frame": window[1]},
             "effective_controls": dict(actual.get("effective_controls", {})) if model == REFERENCE else {},
             "effective_params": params if model == REFERENCE else ""}
    pilot.pop("enhancement_model_choice", None)
    pilot["fingerprint"] = plan_fingerprint(pilot)
    return pilot


def run_pilot_pair(source: Path, info: dict[str, Any], plan: dict[str, Any], window: list[int],
                   topaz: dict[str, Any], writer: Any, preset_id: str,
                   *, deadline: float | None = None) -> dict[str, Any]:
    """Two serial, capped native calls through the existing worker; caller owns GPU lock."""
    import video_upscale as worker
    from adaptive_enhancement import verify_artifacts
    verify_artifacts(plan)
    paths = {m: Path(writer.temp_dir) / f"model_pilot_{plan['fingerprint'][:12]}_{window[0]}_{m}.mov"
             for m in (REFERENCE, CANDIDATE)}
    timings = {}
    first_frames: dict[str, float] = {}
    deadline = deadline if deadline is not None else time.monotonic() + MAX_TOTAL_SECONDS
    try:
        for model, destination in paths.items():
            _cancel(writer)
            remaining = deadline - time.monotonic()
            if remaining < 1:
                raise RuntimeError("Enhancement audition exhausted its total deadline.")
            pilot = _pilot_plan(plan, window, model)
            recipe = worker.EnhancementRecipe(preset_id, model, pilot["effective_params"], explicit=False)
            started = time.monotonic()
            timed_before = len(getattr(writer, "process_timings", []))
            worker.topaz_upscale_to_prores(source, destination, topaz["ffmpeg"], info,
                preset_id, plan["target_width"], plan["target_height"], writer,
                recipe=recipe, adaptive_plan=pilot, audition=True, audition_timeout=min(60.0, remaining))
            timings[model] = time.monotonic() - started
            # The real worker already records first emitted frame and finished
            # frames for each native process. Bind the *current* pilot's
            # measurement; never reuse a previous job's startup time or infer
            # startup from a model name/default benchmark.
            fresh_timings = getattr(writer, "process_timings", [])[timed_before:]
            admitted = [
                row for row in fresh_timings if isinstance(row, dict)
                and row.get("phase") == "adaptive_enhancement_pilot"
                and row.get("status") == "completed"
                and type(row.get("emitted_frames")) is int
                and row["emitted_frames"] == window[1] - window[0]
                and type(row.get("first_frame_seconds")) in (int, float)
                and math.isfinite(row["first_frame_seconds"])
                and 0 < row["first_frame_seconds"] < timings[model]
            ]
            if not admitted:
                raise RuntimeError("Native enhancement pilot lacks measured first-frame/steady-state evidence.")
            first_frames[model] = float(admitted[-1]["first_frame_seconds"])
        comparison = compare_pilot(source, paths[REFERENCE], paths[CANDIDATE], plan, window, writer, deadline=deadline)
        return {"window": list(window), "reference_seconds": timings[REFERENCE],
                "candidate_seconds": timings[CANDIDATE],
                "reference_first_frame_seconds": first_frames[REFERENCE],
                "candidate_first_frame_seconds": first_frames[CANDIDATE],
                "comparison": comparison}
    finally:
        for path in paths.values():
            if path.is_file() and not path.is_symlink():
                path.unlink(missing_ok=True)


def _aggregate(trials: list[dict[str, Any]]) -> tuple[dict[str, Any], float, float, int]:
    if not trials:
        raise ValueError("No model-audition evidence.")
    comparisons = [t["comparison"] for t in trials]
    reasons = {r for c in comparisons for r in c.get("reasons", [])}
    speed_rows: list[dict[str, Any]] = []
    measured_count = 0
    for trial in trials:
        first_ref, first_cand = (trial.get("reference_first_frame_seconds"),
                                 trial.get("candidate_first_frame_seconds"))
        if first_ref is None and first_cand is None:
            continue
        measured_count += 1
        count = trial["comparison"].get("frames")
        if (type(count) is not int or count < 2
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in (first_ref, first_cand))
                or not 0 < first_ref < trial["reference_seconds"]
                or not 0 < first_cand < trial["candidate_seconds"]):
            raise ValueError("A selected model trial has invalid per-shot native timing evidence.")
        ref_tail = trial["reference_seconds"] - first_ref
        cand_tail = trial["candidate_seconds"] - first_cand
        speed_rows.append({
            "window": list(trial["window"]),
            "frames": count,
            "reference_steady_seconds_per_frame": ref_tail / (count - 1),
            "candidate_steady_seconds_per_frame": cand_tail / (count - 1),
            "steady_candidate_over_reference": cand_tail / max(ref_tail, 0.000001),
            "measured_tail_seconds_difference": cand_tail - ref_tail,
        })
    if 0 < measured_count < len(trials):
        reasons.add("incomplete_per_shot_native_timing_evidence")
    # Each production source uses one chosen enhancement model. Where two
    # different photographic shots are represented, don't generalize the
    # faster aggregate average if one demonstrably renders materially slower.
    # Require both >=20% regression and >=0.2s tail difference to avoid
    # rejecting based on a few milliseconds of first-frame/timing jitter.
    if len(speed_rows) == len(trials) > 1 and any(
            row["steady_candidate_over_reference"] >= 1.20
            and row["measured_tail_seconds_difference"] >= 0.20
            for row in speed_rows):
        reasons.add("per_shot_steady_speed_regression")
    passed = (not reasons and all(
        c.get("status") == "PASS_SOURCE_AND_REFERENCE_GUARDS" and not c.get("reasons")
        for c in comparisons))
    count = sum(int(c["frames"]) for c in comparisons)
    return ({"status": "PASS_SOURCE_AND_REFERENCE_GUARDS" if passed else "REJECT",
             "reasons": sorted(reasons), "frames": count, "shot_speed_evidence": speed_rows,
             "scope": "measured_native_pilot_tails_one_global_model_not_a_cross_genre_speed_guarantee"},
            sum(t["reference_seconds"] for t in trials),
            sum(t["candidate_seconds"] for t in trials), count)


def _apply(plan: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    from adaptive_enhancement import plan_fingerprint
    if receipt["decision"]["model"] == REFERENCE:
        return plan
    chosen = deepcopy(plan)
    chosen["enhancement_model_choice"] = {
        "schema": SCHEMA, "model": CANDIDATE, "params": "",
        "basis": receipt["fingerprint"], "base_adaptive_fingerprint": plan["fingerprint"],
        "candidate_assets": receipt["contract"]["models"][CANDIDATE],
        "scope": "all_admitted_photographic_shots_witnessed_one_model_per_source",
    }
    chosen["effective_params"] = ""
    chosen["fingerprint"] = plan_fingerprint(chosen)
    return chosen


def execution_model(plan: dict[str, Any], fallback: str = REFERENCE) -> str:
    choice = plan.get("enhancement_model_choice")
    if choice is None:
        return fallback
    if (not isinstance(choice, dict) or choice.get("schema") != SCHEMA or choice.get("model") != CANDIDATE
            or choice.get("params") != "" or not isinstance(choice.get("candidate_assets"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", choice["candidate_assets"])
            or not re.fullmatch(r"[0-9a-f]{64}", str(choice.get("basis", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(choice.get("base_adaptive_fingerprint", "")))
            or plan.get("effective_params") != ""):
        raise ValueError("Invalid adaptive model-choice contract.")
    return CANDIDATE


def validate_selected_assets(plan: dict[str, Any], model_dir: str) -> None:
    import video_upscale as worker
    if plan.get("enhancement_model_choice"):
        choice = plan["enhancement_model_choice"]
        actual = worker.topaz_model_asset_identity(model_dir, execution_model(plan))["identity_sha256"]
        if actual != choice["candidate_assets"]:
            raise ModelSelectionIntegrityError("Selected enhancement assets changed after qualification.")


def select_for_source(source: Path, info: dict[str, Any], plan: dict[str, Any],
                      topaz: dict[str, Any], writer: Any, recipe: Any,
                      *, delivery_variants: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the existing job's temp/cache/progress and serial GPU lane.

    A user-chosen manual recipe is never silently replaced. Short/unsupported
    inputs incur no experiment. Valid cached evidence is considered before
    resource admission, but only exact source/runtime/model/plan matches reuse.
    """
    import video_upscale as worker
    from adaptive_enhancement import _hash_file, verify_artifacts
    from delivery_insights import resource_snapshot
    from shared.progress_writer import atomic_write_json
    from shared.gpu_lock import GpuLockCancelled

    fallback = {"schema": SCHEMA, "model": REFERENCE, "cache_hit": False}
    if recipe.explicit or recipe.up_model != REFERENCE:
        return plan, {**fallback, "model": recipe.up_model, "reason": "manual_recipe_preserved"}
    nomination = plan_audition(plan)
    if not nomination["eligible"]:
        return plan, {**fallback, "reason": nomination["reason"]}
    sw, sh = plan["source_width"], plan["source_height"]
    if (not delivery_variants or any(
            type(v.get("width")) is not int or type(v.get("height")) is not int
            or not 0 < v["width"] <= sw or not 0 < v["height"] <= sh
            or v["width"] * sh != v["height"] * sw or v.get("layout") == "fill_crop_center"
            for v in delivery_variants)):
        return plan, {**fallback, "reason": "delivery_geometry_not_qualified_for_source_scale_comparison"}
    verify_artifacts(plan)
    model_dir = str(topaz["model_dir"])
    assets = {m: worker.topaz_model_asset_identity(model_dir, m) for m in (REFERENCE, CANDIDATE)}
    if not all(v.get("resume_safe") for v in assets.values()):
        return plan, {**fallback, "reason": "local_candidate_assets_unavailable_no_download"}
    contract = {"schema": SCHEMA, "base_adaptive_fingerprint": plan["fingerprint"],
                "source_sha256": _hash_file(source), "runtime_sha256": _hash_file(Path(topaz["ffmpeg"])),
                "selector_sha256": _hash_file(Path(__file__)), "windows": nomination["windows"],
                "models": {m: a["identity_sha256"] for m, a in assets.items()},
                "frame_count": plan["frame_count"], "active_frames": nomination["active_frames"],
                "deliveries": [{k: v.get(k) for k in ("width", "height", "layout")} for v in delivery_variants]}
    if contract["source_sha256"] != plan.get("identity", {}).get("source_sha256"):
        raise ModelSelectionIntegrityError("The enhancement source no longer matches its analyzed masks.")
    def check_identity() -> None:
        try:
            verify_artifacts(plan)
            changed = (_hash_file(source) != contract["source_sha256"]
                or _hash_file(Path(topaz["ffmpeg"])) != contract["runtime_sha256"]
                or _hash_file(Path(__file__)) != contract["selector_sha256"]
                or any(worker.topaz_model_asset_identity(model_dir, m) != assets[m] for m in assets))
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelSelectionIntegrityError("Enhancement evidence disappeared or changed during qualification.") from exc
        if changed:
            raise ModelSelectionIntegrityError("Enhancement source/runtime/assets changed during qualification.")

    def decision_for(trials: list[dict[str, Any]]) -> dict[str, Any]:
        if not 0 < len(trials) <= len(nomination["windows"]) or [t["window"] for t in trials] != nomination["windows"][:len(trials)]:
            raise ValueError("Cached model trials do not match admitted shot witnesses.")
        comparison, r, c, n = _aggregate(trials)
        # All current real pilots must carry first-frame timing. Historical
        # synthetic fixture receipts without it preserve the compatibility
        # calculation; exact source/runtime/selector fingerprints prevent
        # older real results from entering this new source revision.
        first_evidence = all(
            type(t.get("reference_first_frame_seconds")) in (int, float)
            and type(t.get("candidate_first_frame_seconds")) in (int, float)
            for t in trials
        )
        timing_args = (
            {
                "reference_first_frame_seconds": sum(t["reference_first_frame_seconds"] for t in trials),
                "candidate_first_frame_seconds": sum(t["candidate_first_frame_seconds"] for t in trials),
                "pilot_starts": len(trials),
            } if first_evidence else {}
        )
        expected = choose_model(
            comparison, r, c, pilot_frames=n, total_frames=nomination["active_frames"],
            production_starts=_production_model_starts(plan),
            **timing_args)
        if len(trials) != len(nomination["windows"]):
            if trials[-1]["comparison"]["status"] != "REJECT":
                raise ValueError("Incomplete passing model evidence is not reusable.")
            expected.update(model=REFERENCE, reason="not_all_shots_qualified")
        return expected
    fingerprint = _digest(contract)
    cache = Path(writer.temp_dir) / f"enhancement_choice_{fingerprint[:24]}.json"
    if cache.is_file() and not cache.is_symlink() and cache.stat().st_size < 512 * 1024:
        try:
            saved = json.loads(cache.read_text(encoding="utf8"))
            checksum = saved.pop("receipt_sha256", None)
            if checksum == _digest(saved) and saved.get("contract") == contract and saved.get("fingerprint") == fingerprint:
                expected = decision_for(saved["trials"])
                if saved["decision"] == expected:
                    check_identity()
                    return _apply(plan, saved), {**saved, "cache_hit": True}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    admission = resource_admission(resource_snapshot(), plan)
    if not admission["allowed"]:
        return plan, {**fallback, "reason": admission["reason"], "admission": admission}
    started = time.monotonic()
    trials = []
    try:
        for window in nomination["windows"]:
            _cancel(writer)
            if time.monotonic() - started > MAX_TOTAL_SECONDS:
                check_identity()
                return plan, {**fallback, "reason": "bounded_audition_time_budget", "trials": trials,
                              "seconds": time.monotonic() - started}
            live = resource_admission(resource_snapshot(), plan)
            if not live["allowed"]:
                check_identity()
                return plan, {**fallback, "reason": live["reason"], "admission": live, "trials": trials}
            writer.update(phase="adaptive_enhancement_audition", message="Testing installed enhancement models on a short shot witness")
            trials.append(run_pilot_pair(source, info, plan, window, topaz, writer, recipe.preset_id,
                                        deadline=started + MAX_TOTAL_SECONDS))
            if trials[-1]["comparison"]["status"] != "PASS_SOURCE_AND_REFERENCE_GUARDS":
                break  # Do not pay for more pilots after a visible preservation failure.
        check_identity()
        decision = decision_for(trials)
        receipt = {"schema": SCHEMA, "fingerprint": fingerprint, "contract": contract,
                   "trials": trials, "decision": decision, "model": decision["model"],
                   "reason": decision["reason"], "seconds": round(time.monotonic() - started, 4),
                   "cache_hit": False, "admission": admission}
        receipt["receipt_sha256"] = _digest(receipt)
        atomic_write_json(cache, receipt)
        return _apply(plan, receipt), receipt
    except (worker.JobCancelled, GpuLockCancelled, ModelSelectionIntegrityError):
        raise
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        # Catching optional model failures must not conceal changed evidence.
        check_identity()
        writer.log("Optional enhancement candidate not qualified; retaining Proteus: " + str(exc)[-500:])
        return plan, {**fallback, "reason": "candidate_trial_not_qualified", "error": str(exc)[-500:],
                      "seconds": round(time.monotonic() - started, 4), "trials": trials}
