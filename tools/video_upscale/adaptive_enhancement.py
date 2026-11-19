"""Measured, source-preserving enhancement inside the existing Topaz worker.

No queue, model downloader, OCR, or second renderer lives here. This adapter
creates a bounded job-temp mask and a stable contextual crop, chooses restrained
Proteus controls, and supplies the existing FFmpeg worker with its filtergraph.
The mask is an enhancement-confidence heuristic, NOT a semantic alpha matte.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA = "spiffy_adaptive_natural_v12_native_depth_temporal_holds"
MAX_MASK_BYTES = 512 * 1024 * 1024
MAX_ANALYSIS_SECONDS = 600
MAX_FRAMES = 72000


def normalize_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("mode") != "natural" or set(value) - {"mode", "working_long_edge", "interpolation_resolution", "subject_refinement", "model_search"}:
        raise ValueError('adaptive_enhancement requires mode="natural" and only declared options.')
    result: dict[str, Any] = {"mode": "natural"}
    if "working_long_edge" in value:
        edge = value["working_long_edge"]
        if type(edge) is not int or not 128 <= edge <= 8192 or edge % 2:
            raise ValueError("adaptive working_long_edge must be an even integer from 128 through 8192.")
        result["working_long_edge"] = edge
    if "interpolation_resolution" in value:
        choice = value["interpolation_resolution"]
        if choice not in ("delivery", "working"):
            raise ValueError("adaptive interpolation_resolution must be delivery or working.")
        result["interpolation_resolution"] = choice
    if "subject_refinement" in value:
        if value["subject_refinement"] not in ("auto", "off"):
            raise ValueError("adaptive subject_refinement must be auto or off.")
        result["subject_refinement"] = value["subject_refinement"]
    if "model_search" in value:
        if value["model_search"] not in ("auto", "off"):
            raise ValueError("adaptive model_search must be auto or off.")
        result["model_search"] = value["model_search"]
    return result


def _hash_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def engine_identity() -> dict[str, str]:
    """Bind parent retry decisions to the actual implementation, not just a label."""
    root = Path(__file__).resolve().parents[2]
    return {"adapter_sha256": _hash_file(Path(__file__)),
            "model_selection_sha256": _hash_file(Path(__file__).with_name("adaptive_model_selection.py")),
            "metrics_sha256": _hash_file(root / "Spiffy Workspace/music_edit_builder/metrics.py"),
            "scene_engine_sha256": _hash_file(root / "Spiffy Workspace/scene/scene_detection_engine.py"),
            "scene_tracker_sha256": _hash_file(root / "Spiffy Workspace/reframe/reframe_tracker.py"),
            "scene_fast_paths_sha256": _hash_file(root / "Spiffy Workspace/reframe/reframe_fast_paths.py"),
            "subject_bridge_sha256": _hash_file(Path(__file__).with_name("adaptive_subjects.py")),
            "semantic_roto_sha256": _hash_file(root / "Spiffy Workspace/workflow/semantic_roto.py"),
            "specialist_runtime_sha256": _hash_file(root / "Spiffy Workspace/workflow/bounded_specialist_runtime.py"),
            "subject_localizer_sha256": _hash_file(root / "Spiffy Workspace/workflow/sparse_media_specialists.py"),
            "subject_prompt_owner_sha256": _hash_file(root / "Spiffy Workspace/workflow/thumbnail_mask_generator.py"),
            "worker_sha256": _hash_file(Path(__file__).with_name("video_upscale.py"))}


def plan_fingerprint(plan: dict[str, Any]) -> str:
    stable = {k: v for k, v in plan.items() if k not in {"fingerprint", "cache_hit", "analysis_seconds", "mask_path", "analysis_efficiency"}}
    if isinstance(stable.get("scene_analysis"), dict):
        stable["scene_analysis"] = {k: v for k, v in stable["scene_analysis"].items() if k != "cache"}
    # Measured cost/load and cache locations are not photographic decisions.
    def subject_contract(row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k,v in row.items() if k not in {"seconds", "admission", "cache_state", "manifest_path",
            "attempted_shots", "cached_shots"}}
    if isinstance(stable.get("subject_refinement"), dict):
        original = stable["subject_refinement"]
        stable["subject_refinement"] = {**subject_contract(original),
            "shots": [subject_contract(row) for row in original.get("shots", [])]}
    if isinstance(stable.get("shots"), list):
        stable["shots"] = [{**shot, **({"subject_refinement": subject_contract(shot["subject_refinement"])}
            if isinstance(shot.get("subject_refinement"), dict) else {})} for shot in stable["shots"]]
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _cancel(writer: Any) -> None:
    if getattr(writer, "cancel_requested", False):
        from shared.gpu_lock import GpuLockCancelled
        raise GpuLockCancelled("Adaptive enhancement cancelled before publication.")


def _exact_uniform_rgb(rgb: Any) -> bool:
    """Prove flat color across ALL channels, never by a sampled similarity score.

    Corner/center checks reject most real footage without a full-frame scan.
    Only a possible solid card pays for OpenCV's native mean/stddev pass.
    A changed glyph or single RGB code keeps the full protection pipeline.
    """
    import cv2
    import numpy as np

    if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or
            rgb.dtype != np.uint8 or rgb.shape[2] != 3 or
            min(rgb.shape[:2]) < 1):
        return False
    height, width = rgb.shape[:2]
    reference = rgb[0, 0]
    for y, x in ((height - 1, 0), (0, width - 1),
                 (height - 1, width - 1), (height // 2, width // 2)):
        if not np.array_equal(rgb[y, x], reference):
            return False
    mean, stddev = cv2.meanStdDev(rgb)
    return bool(np.all(stddev == 0) and
                np.array_equal(np.rint(mean[:, 0]).astype(np.uint8), reference))


def detail_protection_mask(rgb: Any, *, collect_metrics: bool = True) -> tuple[Any, dict[str, Any]]:
    """Return bounded enhancement contribution; protect flatness and sharp edges.

    A broad hard-edge guard also protects many burned-in caption strokes without
    recognizing/replacing text. It cannot certify arbitrary typography/effects.
    Instantaneous protection always wins; no temporal trail fills protected gaps.
    """
    import cv2
    import numpy as np

    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Adaptive mask requires an RGB uint8 frame.")
    if _exact_uniform_rgb(rgb):
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        # OpenCV's float32 Gaussian kernels produce a tiny numeric residue on
        # some EXACTLY flat colors (e.g. gray 128). Reproduce that metric on a
        # constant 8x8 patch so old calibrated reports remain byte-equal,
        # without building full-resolution blur/Sobel/contour intermediates.
        flat_metric = 0.0
        if collect_metrics:
            gray_patch = cv2.cvtColor(rgb[:1, :1], cv2.COLOR_RGB2GRAY)
            patch = np.full((min(8, rgb.shape[0]), min(8, rgb.shape[1])),
                            float(gray_patch[0, 0]), dtype=np.float32)
            smooth = cv2.GaussianBlur(patch, (0, 0), 1.2)
            texture = cv2.GaussianBlur(np.abs(patch - smooth), (0, 0), 2.0)
            flat_metric = round(float(np.median(texture)), 6)
        return mask, {} if not collect_metrics else {
            "eligible_fraction": 0.0,
            "contribution_mean": 0.0,
            "protected_edge_fraction": 0.0,
            "texture_residual": flat_metric,
            "semantic_caption_detection": False,
            "policy": "measured_texture_with_instantaneous_hard_edge_and_flatness_protection",
        }
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Same float32 arithmetic and thresholds as the reference implementation.
    # Reuse owned temporary arrays; never modify the decoded source pixels.
    residual = gray.astype(np.float32)
    smooth = cv2.GaussianBlur(residual, (0, 0), 1.2)
    np.subtract(residual, smooth, out=residual)
    np.abs(residual, out=residual)
    del smooth
    texture = cv2.GaussianBlur(residual, (0, 0), 2.0)
    del residual
    texture_median = round(float(np.median(texture)), 6) if collect_metrics else None
    # This selects existing recoverable texture, never invents detail in bokeh.
    amount = texture
    amount -= 1.5
    amount /= 6.0
    np.clip(amount, 0.0, 1.0, out=amount)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    np.abs(gradient_x, out=gradient_x)
    np.abs(gradient_y, out=gradient_y)
    np.maximum(gradient_x, gradient_y, out=gradient_x)
    hard = (gradient_x > 160).astype(np.uint8)
    del gradient_x, gradient_y
    # Join glyph-like strokes into protected line regions, while preserving any
    # other high-contrast object edge too. False positives reduce enhancement.
    joined = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((3, 13), dtype=np.uint8))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(joined, contours, -1, 1, thickness=cv2.FILLED)
    # Include the local antialias/glow halo, not merely the bright glyph edge.
    hard = cv2.dilate(np.maximum(hard, joined), np.ones((15, 15), dtype=np.uint8))
    protection = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 2.0)
    protection *= 1.6
    np.subtract(1.0, protection, out=protection)
    np.clip(protection, 0.0, 1.0, out=protection)
    amount *= protection
    del protection
    amount[hard != 0] = 0.0
    # Retain at least 65% of the conventional source even in admitted interiors.
    amount *= 89.0
    np.rint(amount, out=amount)
    mask = amount.astype(np.uint8)
    mask[mask < 3] = 0
    return mask, {} if not collect_metrics else {
        "eligible_fraction": round(float(np.mean(mask > 0)), 6),
        "contribution_mean": round(float(mask.mean()) / 255.0, 6),
        "protected_edge_fraction": round(float(hard.mean()), 6),
        "texture_residual": texture_median,
        "semantic_caption_detection": False,
        "policy": "measured_texture_with_instantaneous_hard_edge_and_flatness_protection",
    }


def measured_blockiness(rgb: Any) -> float:
    import cv2
    import numpy as np
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    evidence = []
    for axis in (0, 1):
        delta = np.abs(np.diff(gray, axis=axis))
        if delta.shape[axis] < 24:
            continue
        means = delta.mean(axis=1 - axis)
        boundary = means[7::8]
        ordinary = np.delete(means, np.arange(7, len(means), 8))
        evidence.append(max(0.0, float(np.median(boundary) - np.median(ordinary))))
    return max(evidence, default=0.0)


def _caption_baseline_index(values: Any) -> tuple[Any, Any]:
    """Sort contour baseline coordinates once, preserving original box indices.

    The candidate helper below returns the exact same original index order as
    the previous all-contour scan. No detection thresholds or glyph groups
    are changed, and an arbitrary caption must never be omitted to save time.
    """
    import numpy as np

    baselines = values[:, 1] + values[:, 3]
    order = np.argsort(baselines, kind="stable")
    return order, baselines[order]


def _caption_row_candidates(values: Any, order: Any, sorted_baselines: Any,
                            bottom: int, height: int) -> Any:
    """Exact baseline/height search without rechecking every distant contour."""
    import numpy as np

    tolerance = max(2, height * 0.16)
    start = np.searchsorted(sorted_baselines, bottom - tolerance, side="left")
    stop = np.searchsorted(sorted_baselines, bottom + tolerance, side="right")
    candidates = order[start:stop]
    heights = values[candidates, 3]
    within_height = ((heights >= height * 0.82) & (heights <= height * 1.22))
    return np.sort(candidates[within_height])


def _conservative_caption_edge_protection(edges: Any, reason: str) -> tuple[Any, dict[str, Any]]:
    """Protect actual visible edge pixels when glyph grouping is overwhelmed.

    Returning a ZERO mask after finding too many plausible text shapes can
    silently feed animated captions into Topaz/temporal interpolation. A
    bounded halo around source edges is less specific but source-preserving.
    This does not pretend to recognize text or to recover an object matte.
    """
    import cv2
    import numpy as np

    hard = cv2.dilate(edges, np.ones((5, 9), dtype=np.uint8))
    protection = np.maximum(hard, cv2.GaussianBlur(hard, (0, 0), 2.0))
    protection[protection < 3] = 0
    return protection, {
        "line_count": 0, "semantic_recognition": False,
        "status": "conservative_edge_protection",
        "fallback_reason": reason,
        "coverage": round(float(np.mean(protection > 0)), 6),
        "scope": "ambiguous_caption_shapes_preserve_visible_source_edges",
    }


def caption_protection_mask(rgb: Any) -> tuple[Any, dict[str, Any]]:
    """Locate strong aligned glyph groups without OCR or guessed screen boxes.

    This is deliberately a visible-shape heuristic, not semantic recognition.
    Only repeated, similarly sized/aligned components form an admitted line.
    Protection retains actual source pixels, including their existing blur;
    it never reconstructs words or changes the typography.
    """
    import cv2
    import numpy as np

    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Caption protection requires RGB uint8 pixels.")
    height, width = rgb.shape[:2]
    if _exact_uniform_rgb(rgb):
        return np.zeros((height, width), dtype=np.uint8), {
            "line_count": 0, "semantic_recognition": False, "status": "no_glyph_groups",
        }
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 75, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # A busy 4K picture can contain tens of thousands of irrelevant tiny
    # contours. Never build a massive Python box set before discovering the
    # glyph classifier has exceeded its bounded evaluation budget.
    if len(contours) > 10000:
        return _conservative_caption_edge_protection(edges, "raw_contour_budget")
    candidate_boxes: set[tuple[int, int, int, int]] = set()
    for contour in contours:
        if len(contour) < 6:
            continue
        box = cv2.boundingRect(contour)
        if not (7 <= box[3] <= min(180, height * 0.22)
                and 2 <= box[2] <= box[3] * 2
                and box[2] >= box[3] * 0.08):
            continue
        candidate_boxes.add(box)
        if len(candidate_boxes) > 2048:
            return _conservative_caption_edge_protection(edges, "bounded_component_budget")
    boxes = sorted(candidate_boxes)
    protection = np.zeros((height, width), dtype=np.uint8)
    if not boxes:
        return protection, {"line_count": 0, "semantic_recognition": False,
                            "status": "no_glyph_groups"}
    values = np.asarray(boxes, dtype=np.int32)
    baseline_order, sorted_baselines = _caption_baseline_index(values)
    admitted: set[tuple[int, int, int, int]] = set()
    seen_groups: set[tuple[int, ...]] = set()
    for seed in values:
        x, y, w, h = (int(v) for v in seed)
        # Baseline and height agreement reject arbitrary neighboring texture.
        same_row = _caption_row_candidates(values, baseline_order, sorted_baselines, y + h, h)
        if len(same_row) < 4:
            continue
        groups: list[list[int]] = []
        for index in same_row:
            box = values[index]
            if groups:
                previous = values[groups[-1][-1]]
                gap = int(box[0] - previous[0] - previous[2])
                # Nested contours of one glyph do not count as extra letters.
                if gap < -min(int(box[2]), int(previous[2])) * 0.35:
                    continue
                if gap <= h * 2:
                    groups[-1].append(int(index))
                    continue
            groups.append([int(index)])
        for group in groups:
            signature = tuple(group)
            if len(group) < 4 or signature in seen_groups:
                continue
            seen_groups.add(signature)
            chosen = values[group]
            left, top = int(chosen[:, 0].min()), int(chosen[:, 1].min())
            right = int((chosen[:, 0] + chosen[:, 2]).max())
            bottom = int((chosen[:, 1] + chosen[:, 3]).max())
            if right - left < 3 * h or bottom - top > h * 1.4:
                continue
            # A glyph group's high-contrast strokes must occupy a meaningful,
            # bounded fraction; neither a solid rectangle nor noise qualifies.
            density = float(np.mean(edges[top:bottom, left:right] != 0))
            if not 0.08 <= density <= 0.55:
                continue
            for box in chosen:
                admitted.add(tuple(int(v) for v in box))
    if not admitted:
        return protection, {"line_count": 0, "semantic_recognition": False, "status": "no_glyph_groups"}
    # Small context around actual glyph boxes retains outline/glow without
    # freezing an entire guessed subtitle band or the rest of the picture.
    for x, y, w, h in admitted:
        pad = max(2, round(h * 0.16))
        cv2.rectangle(protection, (max(0, x - pad), max(0, y - pad)),
                      (min(width - 1, x + w + pad), min(height - 1, y + h + pad)), 255, cv2.FILLED)
    hard = protection.copy()
    protection = np.maximum(hard, cv2.GaussianBlur(hard, (0, 0), 2.0))
    protection[protection < 3] = 0
    # Excessive detections are not trustworthy caption localization.
    coverage = float(np.mean(protection > 0))
    if coverage > 0.25:
        return _conservative_caption_edge_protection(edges, "ambiguous_dense_edges")
    return protection, {"line_count": len(seen_groups), "glyph_count": len(admitted),
                        "coverage": coverage, "semantic_recognition": False, "status": "measured_glyph_groups"}


def choose_visual_treatment(measurements: list[dict[str, Any]], *,
                            motion_evidence: dict[str, Any] | None = None,
                            caption_detected_fraction: float = 0.0) -> dict[str, Any]:
    """Route measured *risks*, not guessed footage genres or subject identities.

    Flat graphics, high-speed action and interviews may all contain people
    or text. Without a verified semantic result, pixel features cannot
    legitimately distinguish basketball from a fast camera pan or a podcast
    from a stationary film scene. Safe source-preserving choices are better
    than inventing a VLM classification and starting a costly model.
    """
    import statistics

    def median(key: str) -> float | None:
        values = [float(row[key]) for row in measurements
                  if isinstance(row, dict) and type(row.get(key)) in (int, float)
                  and math.isfinite(row[key]) and row[key] >= 0
                  and (key not in {"text_density", "edge_density", "luma_entropy",
                                   "sharpness"} or row[key] <= 1)]
        return statistics.median(values) if values else None

    m = motion_evidence if isinstance(motion_evidence, dict) else {}
    caption = (float(caption_detected_fraction)
               if type(caption_detected_fraction) in (int, float)
               and math.isfinite(caption_detected_fraction)
               and 0 <= caption_detected_fraction <= 1 else 0.0)
    text_density = median("text_density")
    edge_density = median("edge_density")
    entropy = median("luma_entropy")
    sharpness = median("sharpness")
    texture = median("texture_residual")
    spatial = [row["spatial_quality"] for row in measurements
               if isinstance(row, dict) and isinstance(row.get("spatial_quality"), dict)
               and row["spatial_quality"].get("schema") == "spiffy_spatial_quality_v1"]

    def spatial_median(key: str) -> float | None:
        values = [float(s[key]) for s in spatial
                  if type(s.get(key)) in (int, float)
                  and math.isfinite(s[key]) and 0 <= s[key] <= 1]
        return statistics.median(values) if values else None

    soft = spatial_median("soft_non_enhanced_fraction")
    eligible = spatial_median("eligible_fraction")
    observed = m.get("observed_transitions", 0)
    if type(observed) is not int or observed < 0:
        observed = 0
    fast_fraction = m.get("high_motion_fraction")
    if (type(fast_fraction) not in (int, float) or not math.isfinite(fast_fraction)
            or not 0 <= fast_fraction <= 1):
        fast_fraction = None
    motion_mean = m.get("mean_abs_luma_change")
    if (type(motion_mean) not in (int, float) or not math.isfinite(motion_mean)
            or not 0 <= motion_mean <= 255):
        motion_mean = None
    sustained_motion = (
        type(observed) is int and observed >= 8
        and type(fast_fraction) in (float, int) and math.isfinite(fast_fraction)
        and fast_fraction >= .35 and fast_fraction <= 1
        and type(motion_mean) in (float, int) and math.isfinite(motion_mean)
        and motion_mean >= 7
    )
    # High-contrast basketball floors, sports jerseys and moving patterned
    # footage can resemble vector graphics in a flat color histogram. Do
    # not skip subject detection or change model selection based on palette
    # alone during sustained picture motion.
    # A scoreboard or lower third is not proof the ENTIRE shot is graphic
    # design. Never classify an active sports/interview picture as artwork
    # just because a title remains overlaid for much of its duration.
    # Strong flat-palette title cards can still qualify with smaller glyphs.
    caption_graphics = (
        not sustained_motion
        and caption >= .25 and entropy is not None and entropy <= .50
        and edge_density is not None and edge_density >= .01
        and text_density is not None
        and (text_density >= .10 or (entropy <= .12 and text_density >= .008))
    )
    static_graphics = (
        not sustained_motion
        and text_density is not None and text_density >= .09
        and edge_density is not None and edge_density >= .07
        and entropy is not None and entropy <= .45
        and sharpness is not None and sharpness >= .65
    )
    graphics = caption_graphics or static_graphics
    if graphics:
        mode, reason = "graphics_edge_guard", "flat_palette_text_and_edges"
    elif sustained_motion:
        mode, reason = "fast_motion_guard", "sustained_measured_frame_changes"
    elif soft is not None and eligible is not None and soft >= .625 and eligible <= .18:
        mode, reason = "soft_detail_guard", "mostly_soft_or_protected_area"
    elif texture is not None and texture >= 4:
        mode, reason = "grain_retention", "strong_residual_texture_unknown_intent"
    else:
        mode, reason = "balanced_photographic", "insufficient_evidence_for_specialist_treatment"
    return {
        "schema": "spiffy_visual_treatment_v1", "mode": mode,
        "reason": reason, "genre": "not_inferred",
        "scope": "measured_risk_not_semantic_genre",
        "basis": {"caption_detected_fraction": round(caption, 6),
                  "text_density": text_density, "edge_density": edge_density,
                  "luma_entropy": entropy, "texture_residual": texture,
                  "soft_fraction": soft, "eligible_fraction": eligible,
                  "motion_transition_count": observed,
                  "motion_high_fraction": fast_fraction,
                  "motion_mean_luma_change": motion_mean},
    }


def choose_natural_controls(measurements: list[dict[str, Any]], *,
                            visual_treatment: dict[str, Any] | None = None) -> dict[str, float]:
    import statistics
    def median(key: str, default: float = 0.0) -> float:
        values = [float(row[key]) for row in measurements
                  if isinstance(row, dict) and type(row.get(key)) in (int, float)
                  and math.isfinite(row[key]) and row[key] >= 0
                  and (key != "sharpness" or row[key] <= 1)]
        return statistics.median(values) if values else default
    sharpness = median("sharpness", 0.5)
    texture = median("texture_residual")
    compression = min(0.2, median("blockiness") / 40.0)
    # Noise/grain intent is unknown: no positive denoise or deblur is inferred.
    controls = {
        "details": round(0.08 if texture > 4 else 0.12 if sharpness > 0.35 else 0.18, 3),
        "compression": round(compression, 3), "blur": 0.0, "noise": -0.15,
        "halo": 0.0, "preblur": 0.0,
        "blend": 0.65 if texture > 4 else 0.5,
    }
    # If photographic contribution is concentrated in a small detailed region
    # while most tiles are soft and excluded, use LESS generative correction.
    # This cannot identify artistic defocus; it is a conservative protection
    # when the source itself offers little trustworthy texture for enhancement.
    spatial = [row["spatial_quality"] for row in measurements
               if isinstance(row, dict) and isinstance(row.get("spatial_quality"), dict)
               and row["spatial_quality"].get("schema") == "spiffy_spatial_quality_v1"]
    if spatial:
        safe = [(float(s["soft_non_enhanced_fraction"]), float(s["eligible_fraction"]))
                for s in spatial
                if type(s.get("soft_non_enhanced_fraction")) in (int, float)
                and type(s.get("eligible_fraction")) in (int, float)
                and math.isfinite(s["soft_non_enhanced_fraction"])
                and math.isfinite(s["eligible_fraction"])
                and 0 <= s["soft_non_enhanced_fraction"] <= 1
                and 0 <= s["eligible_fraction"] <= 1]
        if (safe and statistics.median(v[0] for v in safe) >= .625
                and statistics.median(v[1] for v in safe) <= .18):
            controls["details"] = min(controls["details"], .10)
            controls["blend"] = max(controls["blend"], .70)
    if visual_treatment is not None:
        if (not isinstance(visual_treatment, dict) or
                visual_treatment.get("schema") != "spiffy_visual_treatment_v1"):
            raise ValueError("Visual treatment must come from measured local evidence.")
        mode = visual_treatment.get("mode")
        limits = {"graphics_edge_guard": (.08, .75),
                  "fast_motion_guard": (.10, .70),
                  "soft_detail_guard": (.10, .70),
                  "grain_retention": (.08, .70),
                  "balanced_photographic": (None, None)}
        if mode not in limits:
            raise ValueError("Unknown automatic visual treatment.")
        max_details, min_blend = limits[mode]
        if max_details is not None:
            controls["details"] = min(controls["details"], max_details)
            controls["blend"] = max(controls["blend"], min_blend)
    return controls


def spatial_quality_evidence(rgb: Any, detail_mask: Any, caption_mask: Any) -> dict[str, Any]:
    """Measure small aspect-correct tiles without another decode or GPU call.

    Only the full-resolution source masks determine pixels for enhancement.
    These low-cost thumbnails nominate detail witnesses for model comparison;
    they do not locate faces or certify that a blurred background is intentional.
    """
    import cv2
    import numpy as np

    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or
            rgb.ndim != 3 or rgb.shape[2] != 3):
        raise ValueError("Spatial quality requires RGB uint8 source pixels.")
    height, width = rgb.shape[:2]
    if (min(height, width) < 16 or
            any(not isinstance(mask, np.ndarray) or mask.dtype != np.uint8
                or mask.shape != (height, width)
                for mask in (detail_mask, caption_mask))):
        raise ValueError("Spatial quality requires matching nonempty uint8 masks.")

    ratio = min(1.0, 384.0 / max(width, height))
    small_width, small_height = max(16, round(width * ratio)), max(16, round(height * ratio))
    shape = (small_width, small_height)
    small = cv2.resize(rgb, shape, interpolation=cv2.INTER_AREA)
    detail = cv2.resize(detail_mask, shape, interpolation=cv2.INTER_AREA)
    captions = cv2.resize(caption_mask, shape, interpolation=cv2.INTER_AREA)
    eligible = (detail > 2) & (captions < 128)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    residual = cv2.absdiff(gray, cv2.GaussianBlur(gray, (3, 3), 0))

    tiles: list[dict[str, Any]] = []
    soft_area = 0
    for row in range(4):
        y0, y1 = small_height * row // 4, small_height * (row + 1) // 4
        sy0, sy1 = height * row // 4, height * (row + 1) // 4
        for col in range(4):
            x0, x1 = small_width * col // 4, small_width * (col + 1) // 4
            sx0, sx1 = width * col // 4, width * (col + 1) // 4
            part = eligible[y0:y1, x0:x1]
            fraction = float(np.mean(part))
            edge_density = float(np.mean(edges[y0:y1, x0:x1] != 0))
            texture = float(np.mean(residual[y0:y1, x0:x1]))
            if fraction < .02 and edge_density < .01 and texture < 1.6:
                soft_area += (y1 - y0) * (x1 - x0)
            score = min(1.0, fraction * (0.7 + .3 * min(1.0, 12.0 * edge_density)))
            tiles.append({
                "grid": [row, col],
                "source_rect": [sx0, sy0, sx1 - sx0, sy1 - sy0],
                "eligible_fraction": round(fraction, 6),
                "caption_fraction": round(float(np.mean(captions[y0:y1, x0:x1] >= 128)), 6),
                "edge_density": round(edge_density, 6),
                "texture_residual": round(texture, 4),
                "priority_score": round(score, 6),
            })
    candidates = sorted((t for t in tiles if t["eligible_fraction"] >= .015),
                        key=lambda t: (-t["priority_score"], t["grid"]))
    return {
        "schema": "spiffy_spatial_quality_v1",
        "source_size": [width, height], "sample_size": [small_width, small_height],
        "aspect": str(Fraction(width, height)),
        "eligible_fraction": round(float(np.mean(eligible)), 6),
        "caption_fraction": round(float(np.mean(captions >= 128)), 6),
        "soft_non_enhanced_fraction": round(soft_area / (small_width * small_height), 6),
        "priority_score": candidates[0]["priority_score"] if candidates else 0.,
        "priority_regions": [dict(t) for t in candidates[:3]],
        "tiles": tiles,
        "scope": "local_advisory_not_object_semantics_or_quality_truth",
    }


def summarize_shot_spatial_quality(measurements: list[dict[str, Any]],
                                  start_frame: int, end_frame: int) -> dict[str, Any]:
    """Choose a detailed representative without re-decoding the shot."""
    import statistics

    rows = [(int(row["frame"]), row["spatial_quality"])
            for row in measurements if isinstance(row.get("spatial_quality"), dict)]
    if not rows:
        return {"schema": "spiffy_shot_spatial_summary_v1", "sample_count": 0,
                "representative_frame": None, "priority_score": 0.,
                "priority_regions": [], "scope": "no_spatial_sample"}
    if any(not start_frame <= frame < end_frame for frame, _ in rows):
        raise ValueError("Spatial witness belongs to the wrong shot.")
    selected = max(rows, key=lambda row: (row[1].get("priority_score", 0.), -row[0]))
    return {"schema": "spiffy_shot_spatial_summary_v1", "sample_count": len(rows),
            "representative_frame": selected[0], "priority_score": selected[1]["priority_score"],
            "priority_score_median": round(statistics.median(
                value["priority_score"] for _, value in rows), 6),
            "soft_non_enhanced_fraction_median": round(statistics.median(
                value["soft_non_enhanced_fraction"] for _, value in rows), 6),
            "priority_regions": selected[1]["priority_regions"],
            "scope": "aspect_safe_sampled_evidence_not_semantic_roi_or_quality_truth"}


def choose_processing_crop(union: Any, target_width: int, target_height: int) -> list[int]:
    import numpy as np
    height, width = union.shape
    full = [0, 0, width, height]
    # A rational isotropic resize has a repeating pixel phase too. For reduced
    # p/q, source boundaries at 2*q map to even destination boundaries at 2*p.
    # This preserves both pixel phase and chroma alignment without restricting
    # the common 1080x1350 -> 3072x3840 route to full-frame enhancement.
    scale = Fraction(target_width, width)
    if scale < 1 or scale != Fraction(target_height, height):
        return full
    alignment = 2 * scale.denominator
    # Materializing every nonzero X/Y coordinate creates two int64 arrays
    # (potentially >100 MiB at 4K). We only need the first/last occupied
    # rows and columns; scan bool reductions instead, without changing crop.
    rows = np.flatnonzero(np.any(union, axis=1))
    if rows.size == 0:
        return full
    columns = np.flatnonzero(np.any(union, axis=0))
    pad = max(64, int(round(min(width, height) * 0.04)))
    x0, y0 = max(0, int(columns[0]) - pad), max(0, int(rows[0]) - pad)
    x1, y1 = min(width, int(columns[-1]) + pad + 1), min(height, int(rows[-1]) + pad + 1)
    x0, y0 = x0 // alignment * alignment, y0 // alignment * alignment
    x1 = min(width, math.ceil(Fraction(x1, alignment)) * alignment)
    y1 = min(height, math.ceil(Fraction(y1, alignment)) * alignment)
    if any(value % alignment for value in (x0, y0, x1, y1)):
        return full
    if (x1 - x0) < 64 or (y1 - y0) < 64 or (x1 - x0) * (y1 - y0) > width * height * 0.70:
        return full
    return [x0, y0, x1 - x0, y1 - y0]


def scene_frame_windows(shots: list[dict[str, Any]], rate: Fraction, count: int) -> list[tuple[int, int]]:
    """Translate the incumbent detector's millisecond starts onto source frames.

    The last endpoint is always the decoded frame count, never a detector's
    rounded millisecond duration. Every frame belongs to exactly one interval.
    """
    if rate <= 0 or type(count) is not int or count < 1:
        raise ValueError("Scene windows require positive exact cadence and frame count.")
    cuts = {0, count}
    for shot in shots:
        if not isinstance(shot, dict):
            raise ValueError("Scene owner returned a malformed shot.")
        value = shot.get("start_ms", 0)
        if isinstance(value, bool):
            raise ValueError("Scene start must be finite milliseconds.")
        try:
            ordinal = math.floor(Fraction(str(value)) * rate / 1000 + Fraction(1, 2))
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError("Scene start must be finite milliseconds.") from exc
        if 0 < ordinal < count:
            cuts.add(ordinal)
    ordered = sorted(cuts)
    return list(zip(ordered, ordered[1:]))


def select_render_units(shots: list[dict[str, Any]], width: int, height: int,
                        rate: int | Fraction) -> tuple[list[dict[str, Any]], str]:
    """Admit temporal splitting only when it avoids meaningful model work.

    This is a conservative setup-cost heuristic, not a measured time estimate.
    Short busy cuts stay in one model invocation. Adjacent identical recipes
    share a unit; protected intervals do not initialize an enhancement model.
    """
    if len(shots) < 2:
        return [], "single_shot"
    units: list[dict[str, Any]] = []
    for shot in shots:
        item = {k: shot[k] for k in ("start_frame", "end_frame", "crop", "route", "effective_controls")}
        if units and units[-1]["end_frame"] == item["start_frame"] and (
            (units[-1]["route"] == item["route"] == "source_preserved") or
            all(units[-1][k] == item[k] for k in ("crop", "route", "effective_controls"))
        ):
            units[-1]["end_frame"] = item["end_frame"]
        else:
            units.append(dict(item))
    # An edited burst may contain several very short photographic shots. Keep
    # those adjacent shots in one conservative model call rather than paying
    # startup again at each cut; their mask still resets at the true boundaries.
    batched: list[dict[str, Any]] = []
    short_limit = max(8, math.ceil(Fraction(rate) / 2))
    for unit in units:
        previous = batched[-1] if batched else None
        if (previous and previous["route"] != "source_preserved" and unit["route"] != "source_preserved"
                and min(previous["end_frame"] - previous["start_frame"], unit["end_frame"] - unit["start_frame"]) < short_limit):
            a, b = previous["crop"], unit["crop"]
            x, y = min(a[0], b[0]), min(a[1], b[1])
            w = max(a[0] + a[2], b[0] + b[2]) - x
            h = max(a[1] + a[3], b[1] + b[3]) - y
            previous["crop"] = [x, y, w, h] if w * h < width * height * 0.70 else [0, 0, width, height]
            previous["route"] = "masked_crop" if previous["crop"][2:] != [width, height] else "masked_full_frame"
            previous["end_frame"] = unit["end_frame"]
            previous["effective_controls"] = {
                key: (max if key == "blend" else min)(value, unit["effective_controls"][key])
                for key, value in previous["effective_controls"].items()}
        else:
            batched.append(dict(unit))
    units = batched
    active = [u for u in units if u["route"] != "source_preserved"]
    if not active:
        return [], "all_source_preserved"
    total = sum(u["end_frame"] - u["start_frame"] for u in units)
    skipped = sum(u["end_frame"] - u["start_frame"] for u in units if u["route"] == "source_preserved")
    if len(units) <= 16 and len(active) == 1 and skipped >= 8 and skipped / total >= 0.10:
        return units, "skip_protected_intervals"
    # Multiple model starts need substantial area-time reduction and useful
    # temporal extent. Otherwise retain the whole-clip conservative recipe.
    left = min(u["crop"][0] for u in active)
    top = min(u["crop"][1] for u in active)
    right = max(u["crop"][0] + u["crop"][2] for u in active)
    bottom = max(u["crop"][1] + u["crop"][3] for u in active)
    union_area = min(width * height, (right - left) * (bottom - top))
    work = sum(u["crop"][2] * u["crop"][3] * (u["end_frame"] - u["start_frame"]) for u in active)
    saved_fraction = 1.0 - work / max(1, union_area * total)
    if (len(units) <= 16 and len(active) <= 4 and saved_fraction >= 0.35
            and min(u["end_frame"] - u["start_frame"] for u in active) >= max(8, math.ceil(Fraction(rate) / 2))):
        return units, "spatial_temporal_work_reduction"
    return [], "batch_short_or_similar_shots"


def interpolation_geometry(plan: dict[str, Any], variants: list[dict[str, Any]]) -> tuple[int, int]:
    """Avoid synthesizing pixels that every requested delivery then discards.

    Resizing happens after enhancement but before frame synthesis, inside the
    same native filter graph. Keep working resolution when a requested master
    needs it, a crop requires it, or exact aspect cannot be retained.
    """
    width, height = int(plan["target_width"]), int(plan["target_height"])
    if plan.get("policy", {}).get("interpolation_resolution", "delivery") == "working":
        return width, height
    scales = []
    for item in variants:
        w, h = int(item["width"]), int(item["height"])
        if min(w, h) <= 0:
            raise ValueError("Interpolation sizing requires resolved delivery dimensions.")
        sx, sy = Fraction(w, width), Fraction(h, height)
        scales.append(max(sx, sy) if item["layout"] == "fill_crop_center" else min(sx, sy))
    factor = min(Fraction(1), max(scales, default=Fraction(1)))
    out_w, out_h = math.ceil(width * factor / 2) * 2, math.ceil(height * factor / 2) * 2
    if out_w * height != out_h * width or min(out_w, out_h) < 64:
        return width, height
    return out_w, out_h


def interpolation_pilot_windows(plan: dict[str, Any], target_rate: str, width: int, height: int) -> list[tuple[int, int]]:
    """Two eight-frame motion witnesses at most, never across a source cut.

    Very short jobs cannot repay two model startups. High-rate or larger-than-
    1080p-class delivery remains on the incumbent reference until qualified.
    """
    rate, target = Fraction(plan["source_fps"]), Fraction(target_rate)
    count = int(plan["frame_count"])
    if (not 20 <= rate <= 30 or not rate < target <= 60 or width * height > 1920 * 1080
            or Fraction(count, 1) / rate < 4):
        return []
    shots = [s for s in plan.get("shots", [])
             if s["end_frame"] - s["start_frame"] >= 8
             and s.get("route") != "source_preserved"
             and not (type(s.get("exact_repeat_pairs")) is int
                      and s["exact_repeat_pairs"] >= s["end_frame"] - s["start_frame"] - 1)
             and not (isinstance(s.get("visual_treatment"), dict)
                      and s["visual_treatment"].get("schema") == "spiffy_visual_treatment_v1"
                      and s["visual_treatment"].get("mode") == "graphics_edge_guard")]

    def motion_priority(shot: dict[str, Any]) -> tuple[float, float, int]:
        evidence = shot.get("motion_evidence")
        sustained = 0.0
        if isinstance(evidence, dict) and evidence.get("schema") == "spiffy_shot_motion_risk_v1":
            observed = evidence.get("observed_transitions")
            fraction = evidence.get("high_motion_fraction")
            mean = evidence.get("mean_abs_luma_change")
            if (type(observed) is int and observed >= 8
                    and type(fraction) in (int, float) and math.isfinite(fraction) and 0 <= fraction <= 1
                    and type(mean) in (int, float) and math.isfinite(mean) and mean >= 0):
                sustained = float(fraction) * float(mean)
        # Without verified continuous-motion evidence, preserve the incumbent
        # measured-peak ranking. A single cut-like spike cannot outrank known
        # sustained picture motion for a model trial.
        return (-sustained, -float(shot.get("motion_peak_delta", 0)), shot["start_frame"])

    shots.sort(key=motion_priority)
    windows = []
    for shot in shots[:2]:
        peak = int(shot.get("motion_peak_frame", (shot["start_frame"] + shot["end_frame"]) // 2))
        start = max(shot["start_frame"], min(peak - 4, shot["end_frame"] - 8))
        windows.append((start, start + 8))
    # BOTH models process each pilot. Cap their combined source-frame workload
    # at 20% of the complete job; the former one-model budget underestimated
    # calibration cost and admitted 32 pilot frames for a 96-frame short.
    # This is an admission bound, not a claim of linear wall-clock speedup.
    return sorted(windows) if 2 * sum(b - a for a, b in windows) * 5 <= count else []


def compare_interpolation_candidates(reference: Path, candidate: Path, writer: Any) -> dict[str, Any]:
    """Compare a bounded native pilot against its incumbent model, regionally.

    Reference agreement is not ground-truth recovered motion. Local tiles keep
    a damaged face/hand from hiding behind a low whole-image average.
    """
    import av
    import numpy as np
    from itertools import zip_longest

    rows = []
    previous_error = None
    started = time.monotonic()
    with av.open(str(reference)) as one, av.open(str(candidate)) as two:
        for index, pair in enumerate(zip_longest(one.decode(video=0), two.decode(video=0))):
            _cancel(writer)
            if index >= 64 or time.monotonic() - started > 60:
                raise RuntimeError("Interpolation pilot comparison exceeded its bounded budget.")
            f, g = pair
            if f is None or g is None or f.pts is None or g.pts is None:
                raise RuntimeError("Interpolation pilot frame coverage is inconsistent.")
            if ((f.width, f.height) != (g.width, g.height)
                    or f.width * f.height > 1920 * 1080
                    or Fraction(f.pts) * f.time_base != Fraction(g.pts) * g.time_base):
                raise RuntimeError("Interpolation pilot geometry or presentation clocks disagree.")
            x, y = f.to_ndarray(format="rgb24"), g.to_ndarray(format="rgb24")
            error = y.astype(np.float32) - x.astype(np.float32)
            delta = np.abs(error)
            tile_means = [float(tile.mean()) for band in np.array_split(delta, 8, axis=0)
                          for tile in np.array_split(band, 8, axis=1)]
            rows.append({"frame": index, "mae": float(delta.mean()), "p99": float(np.percentile(delta, 99)),
                         "max_tile_mae": max(tile_means), "large_error_fraction": float(np.mean(delta > 32)),
                         "temporal_error_change": float(np.mean(np.abs(error - previous_error)))
                             if previous_error is not None else 0.0})
            previous_error = error
    if len(rows) < 8:
        raise RuntimeError("Interpolation pilot has insufficient moving-frame evidence.")
    reasons = []
    if any(r["mae"] > 2.5 or r["p99"] > 24 for r in rows):
        reasons.append("reference_pixels_changed")
    if any(r["max_tile_mae"] > 8 or r["large_error_fraction"] > 0.005 for r in rows):
        reasons.append("local_detail_changed")
    if any(r["temporal_error_change"] > 3 for r in rows):
        reasons.append("temporal_reference_disagreement")
    return {"status": "REJECT" if reasons else "PASS_REFERENCE_AGREEMENT", "reasons": reasons,
            "frames": len(rows), "samples": rows,
            "scope": "bounded local/temporal agreement with Apollo reference, not universal visual acceptance"}


def choose_interpolation_model(comparison: dict[str, Any], reference_seconds: float, candidate_seconds: float) -> dict[str, Any]:
    for value in (reference_seconds, candidate_seconds):
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError("Interpolation timing evidence must be finite positive seconds.")
    agreement = (comparison.get("status") == "PASS_REFERENCE_AGREEMENT"
                 and not comparison.get("reasons") and type(comparison.get("frames")) is int
                 and comparison["frames"] >= 8)
    faster = candidate_seconds <= reference_seconds * 0.9
    return {"model": "apf-2" if agreement and faster else "apo-8",
            "reason": "pilot_quality_and_speed_pass" if agreement and faster else
                      "pilot_quality_rejected" if not agreement else "pilot_no_measured_speed_win",
            "reference_seconds": round(reference_seconds, 4), "candidate_seconds": round(candidate_seconds, 4),
            "comparison": comparison}


def motion_thumbnail_size(width: int, height: int) -> tuple[int, int]:
    """Exact source-aspect motion sampling, limited to 96 pixels long-edge.

    A forced 96x54 rectangle previously distorted portrait and square
    source frames and could bias the measured frame-change decision. This
    is the same decoded image, only a source-proportional small score plane.
    """
    if (type(width) is not int or type(height) is not int or
            width < 1 or height < 1):
        raise ValueError("Motion sampling needs positive integer dimensions.")
    if width >= height:
        return 96, max(1, round(Fraction(height * 96, width)))
    return max(1, round(Fraction(width * 96, height))), 96


def _same_rgb_frame(first: Any, second: Any) -> bool:
    """Exact RGB repeat check with cheap moving-frame rejection on large clips.

    Sparse probes NEVER approve a repeat: matching probes still require a
    complete exact NORM_INF comparison across all RGB codes. Ordinary motion
    near the image center/quadrants avoids the large full-frame scan.
    """
    import cv2
    import numpy as np

    if (not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray)
            or first.dtype != np.uint8 or second.dtype != np.uint8
            or first.shape != second.shape or first.ndim != 3
            or first.shape[2] != 3):
        return False
    if first.size >= 2_000_000:
        height, width = first.shape[:2]
        for y, x in ((height // 2, width // 2),
                     (height // 4, width // 4),
                     (height // 4, width * 3 // 4),
                     (height * 3 // 4, width // 4),
                     (height * 3 // 4, width * 3 // 4),
                     (height // 2, width // 4),
                     (height // 2, width * 3 // 4),
                     (height // 4, width // 2),
                     (height * 3 // 4, width // 2)):
            if not np.array_equal(first[y, x], second[y, x]):
                return False
    return bool(cv2.norm(first, second, cv2.NORM_INF) == 0)


@lru_cache(maxsize=1)
def _native_memcmp_function() -> Any:
    """Use the host C runtime's exact byte comparison, with a safe fallback.

    Comparing PyAV plane memoryviews with Python's elementwise rich-compare
    was expensive for large identical 10-bit source frames. The host C
    comparison checks every byte without allocating another image or
    converting 10-bit source samples to RGB8. If the system C runtime is
    unavailable, callers retain the old exact pixel comparison.
    """
    try:
        import ctypes

        runtime = ctypes.CDLL("msvcrt.dll" if os.name == "nt" else None)
        compare = runtime.memcmp
        compare.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
        compare.restype = ctypes.c_int
        return compare
    except (ImportError, AttributeError, OSError):
        return None


def _native_decoded_frames_equal(previous: Any, current: Any) -> bool:
    """Prove a source hold at its ORIGINAL decoded pixel depth.

    An 8-bit RGB mask-analysis cache hit is not enough to freeze output:
    distinct 10/12-bit luma values can quantize to identical RGB24.
    Compare every decoded native plane, including low-order sample bits.
    Retaining a single previous PyAV frame avoids another decode. The
    full plane includes padding; different padding can cause a conservative
    false negative (no hold), never a false positive.
    """
    if previous is None or current is None:
        return False
    try:
        if ((previous.width, previous.height, previous.format.name)
                != (current.width, current.height, current.format.name)):
            return False
        for attribute in ("color_range", "color_primaries", "color_trc",
                          "colorspace", "sample_aspect_ratio"):
            if getattr(previous, attribute, None) != getattr(current, attribute, None):
                return False
        a, b = previous.planes, current.planes
        if not a or len(a) != len(b):
            return False
        compare_bytes = _native_memcmp_function()
        for left, right in zip(a, b):
            if ((left.width, left.height, left.line_size, left.buffer_size)
                    != (right.width, right.height, right.line_size, right.buffer_size)):
                return False
            # Plane.buffer_ptr is owned by these live PyAV VideoFrame objects
            # for the duration of the call. Never use raw pointers from
            # external data, and never compare beyond buffer_size.
            left_pointer = getattr(left, "buffer_ptr", None)
            right_pointer = getattr(right, "buffer_ptr", None)
            if (compare_bytes is not None and type(left_pointer) is int
                    and type(right_pointer) is int and left_pointer > 0
                    and right_pointer > 0 and 0 < left.buffer_size <= 512 * 1024 * 1024):
                if left_pointer != right_pointer and compare_bytes(
                        left_pointer, right_pointer, left.buffer_size) != 0:
                    return False
            elif memoryview(left) != memoryview(right):
                return False
        return True
    except (AttributeError, BufferError, TypeError, ValueError):
        return False


def _admit_parallel_masks(width: int, height: int) -> bool:
    """Use two independent CPU mask workers only with adequate memory.

    Do not increase CPU concurrency for thumbnails, on busy/low-memory hosts
    or on a Mac where the available-memory probe is unavailable. Two already
    necessary full-frame calculations overlap; no source frame is skipped,
    no approximate result is reused, and no GPU work is introduced.
    """
    if width * height < 640 * 360:
        return False
    try:
        import psutil

        available = psutil.virtual_memory().available
        return available >= max(4 * 1024**3, width * height * 128)
    except (ImportError, OSError, AttributeError, RuntimeError):
        return False


class _FrameAnalysisCache:
    """One preceding RGB frame, scoped to a shot and a single analysis pass.

    Exact uint8 RGB equality is required. No approximate image hash, gray-only
    match, sparse frame skipping, motion prediction, or persistent frame store.
    Only raw spatial analysis is reused; temporal smoothing and output timestamps
    still run for every frame. Returned arrays/dicts cannot mutate the cache.
    """

    def __init__(self, *, parallel_analysis: bool = False) -> None:
        self._counts = {"decoded_frames": 0, "exact_repeat_hits": 0,
                        "detail_mask_evaluations": 0, "caption_mask_evaluations": 0,
                        "metric_evaluations": 0, "scene_resets": 0,
                        "parallel_mask_evaluations": 0,
                        "native_rgb_conversion_skips": 0}
        self._parallel_analysis = bool(parallel_analysis)
        self._executor = None
        self._closed = False
        self.reset()

    def disable_parallel(self) -> None:
        """Release bounded worker threads when memory headroom disappears."""
        self._parallel_analysis = False
        pool, self._executor = self._executor, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def close(self) -> None:
        self.disable_parallel()
        self._closed = True

    def reset(self) -> None:
        self._rgb = None
        self._detail = None
        self._caption = None
        self._caption_evidence = None
        self._detail_evidence = None
        self._measured = None
        self._gray_small = None
        self.last_repeated = False
        self._counts["scene_resets"] += 1

    def _rgb_for_verified_native_hold(self) -> Any | None:
        """Read-only internal reuse after the caller proves native plane equality.

        RGB from the previous decoded frame is already a private copy. The
        caller must first establish equality across *all* native source planes;
        quantized RGB8 similarity alone never qualifies as a source hold.
        """
        return self._rgb if not self._closed else None

    def analyze(self, rgb: Any, *, collect_metrics: bool = False,
                _verified_native_hold: bool = False) -> tuple[Any, Any, dict[str, Any], dict[str, Any], Any]:
        import cv2
        import numpy as np
        from music_edit_builder.metrics import frame_quality_metrics

        if self._closed:
            raise RuntimeError("Frame-analysis cache is closed.")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Frame analysis requires RGB uint8 source pixels.")
        if _verified_native_hold and (self._rgb is None or rgb is not self._rgb):
            raise ValueError("Verified native repeat must reuse the cache's exact preceding RGB buffer.")
        self._counts["decoded_frames"] += 1
        repeated = self._rgb is not None and (
            _verified_native_hold or _same_rgb_frame(rgb, self._rgb))
        if _verified_native_hold:
            self._counts["native_rgb_conversion_skips"] += 1
        self.last_repeated = repeated
        if repeated:
            self._counts["exact_repeat_hits"] += 1
        else:
            if self._parallel_analysis:
                if self._executor is None:
                    from concurrent.futures import ThreadPoolExecutor

                    self._executor = ThreadPoolExecutor(
                        max_workers=2, thread_name_prefix="spiffy-mask")
                detail_work = self._executor.submit(
                    detail_protection_mask, rgb, collect_metrics=collect_metrics)
                caption_work = self._executor.submit(caption_protection_mask, rgb)
                self._detail, evidence = detail_work.result()
                self._caption, self._caption_evidence = caption_work.result()
                self._counts["parallel_mask_evaluations"] += 1
            else:
                self._detail, evidence = detail_protection_mask(rgb, collect_metrics=collect_metrics)
                self._caption, self._caption_evidence = caption_protection_mask(rgb)
            self._counts["detail_mask_evaluations"] += 1
            self._detail_evidence = evidence if collect_metrics else None
            self._counts["caption_mask_evaluations"] += 1
            self._gray_small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                                          motion_thumbnail_size(rgb.shape[1], rgb.shape[0]))
            self._measured = None
            self._rgb = rgb.copy()
        if collect_metrics:
            if self._detail_evidence is None:
                _mask, self._detail_evidence = detail_protection_mask(rgb, collect_metrics=True)
                self._counts["detail_mask_evaluations"] += 1
            if self._measured is None:
                stats = frame_quality_metrics(rgb)
                self._measured = {**{k: stats[k] for k in
                    ("sharpness", "laplacian_variance", "edge_density", "text_density", "luma_entropy")},
                    **self._detail_evidence, "blockiness": round(measured_blockiness(rgb), 6)}
                self._counts["metric_evaluations"] += 1
        return (self._detail.copy(), self._caption.copy(), dict(self._caption_evidence),
                dict(self._measured) if collect_metrics else {}, self._gray_small.copy())

    def stats(self) -> dict[str, Any]:
        return {**self._counts, "cached_rgb_frames": int(self._rgb is not None),
                "retained_rgb_bytes": int(self._rgb.nbytes) if self._rgb is not None else 0,
                "policy": "exact_rgb_equality_with_per_shot_reset_no_skipped_frames"}


def _scene_windows(source: Path, rate: Fraction, count: int, root: Path,
                   writer: Any) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    from scene.scene_detection_engine import detect_scene_shots, ENGINE_VERSION

    def progress(amount: float, message: str) -> None:
        _cancel(writer)
        writer.update(phase="adaptive_scenes", message=message,
                      metrics={"adaptive_scene_progress": float(amount)})

    _cancel(writer)
    result = detect_scene_shots(source, strategy="v2", cache_dir=root / "adaptive_scene_cache",
                               progress_cb=progress)
    _cancel(writer)
    windows = scene_frame_windows(result.shots, rate, count)
    return windows, {"owner": "scene.scene_detection_engine", "engine_version": ENGINE_VERSION,
                     "strategy": result.strategy, "detector_version": result.detector_version,
                     "shot_count": len(windows), "cache": result.metadata.get("cache", {})}


def prepare(source: Path, info: dict[str, Any], target_width: int, target_height: int,
            writer: Any, policy: dict[str, Any]) -> dict[str, Any]:
    """Decode all source frames once, retain only a lossless confidence-mask video.

    This dense pass prevents sparse sampling from missing briefly animated text
    or a subject entering the frame. Reuse is confined to the existing job temp.
    """
    import av
    import cv2
    import numpy as np
    from shared.progress_writer import atomic_write_json
    from shared.ffmpeg_utils import standard_ffmpeg
    from adaptive_subjects import prepare_subjects, verify_cached_subjects

    normalize_policy(policy)
    source = Path(source).resolve()
    _cancel(writer)
    from shared.media_probe import pixel_format_has_alpha
    if info.get("has_alpha") is True or pixel_format_has_alpha(info.get("pix_fmt")):
        raise ValueError(
            "Adaptive natural cannot preserve source alpha/transparency through "
            "the opaque Topaz/MP4 path. Composite transparencies explicitly first."
        )
    if info.get("is_vfr") or int(info.get("rotation") or 0) != 0 or str(info.get("sar") or "1:1") not in {"1:1", "1/1"}:
        raise ValueError("Adaptive natural currently requires CFR, square-pixel, zero-rotation source; use the existing timing/geometry route for other sources.")
    if str(info.get("color_transfer") or "").lower() in {"smpte2084", "arib-std-b67"}:
        raise ValueError("Adaptive natural has no qualified HDR compositing path.")
    if str(info.get("field_order") or "").lower() not in {"", "unknown", "progressive"}:
        raise ValueError("Adaptive natural does not silently deinterlace a finished edit.")
    width, height = int(info["width"]), int(info["height"])
    count = int(info.get("video_frame_count") or info.get("video_pts_count") or 0)
    if width % 2 or height % 2 or width < 16 or height < 16 or width * height > 3840 * 2160:
        raise ValueError("Adaptive natural requires an even source canvas from 16 pixels through UHD pixel count.")
    if count < 2 or count > MAX_FRAMES or float(info.get("duration") or 0) > MAX_ANALYSIS_SECONDS:
        raise ValueError("Adaptive natural requires 2..72000 frames and a clip no longer than 600 seconds.")
    rate = Fraction(str(info.get("fps_ratio") or info.get("fps") or 0))
    if rate <= 0 or rate > 120:
        raise ValueError("Adaptive natural requires a source cadence in (0,120].")
    started = time.monotonic()
    stat = source.stat()
    source_sha = _hash_file(source)
    dependencies = {
        "adapter": _hash_file(Path(__file__)),
        "metrics": _hash_file(Path(__file__).resolve().parents[2] / "Spiffy Workspace/music_edit_builder/metrics.py"),
        "av": av.__version__, "cv2": cv2.__version__, "numpy": np.__version__,
        "standard_ffmpeg": _hash_file(Path(standard_ffmpeg())),
        "engine": engine_identity(),
    }
    identity = {"source_sha256": source_sha, "source_bytes": stat.st_size, "schema": SCHEMA,
                "target": [target_width, target_height], "policy": policy, "dependencies": dependencies}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(writer.temp_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path, mask_path = root / f"adaptive_{key}.json", root / f"adaptive_{key}.nut"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf8"))
            if (cached["identity"] == identity and cached["fingerprint"] == plan_fingerprint(cached)
                    and Path(cached["mask_path"]).resolve() == mask_path.resolve()
                    and not mask_path.is_symlink() and _hash_file(mask_path) == cached["mask_sha256"]):
                verify_cached_subjects(cached.get("subject_refinement"))
                _cancel(writer)
                return {**cached, "cache_hit": True}
        except (OSError, KeyError, TypeError, ValueError, RuntimeError):
            pass
    temporary_mask = mask_path.with_suffix(".part.nut")
    windows, scene_analysis = _scene_windows(source, rate, count, root, writer)
    union = np.zeros((height, width), dtype=np.uint8)
    shot_union = np.zeros_like(union)
    shot_measurements: list[dict[str, Any]] = []
    shots: list[dict[str, Any]] = []
    window_index = 0
    measurements: list[dict[str, Any]] = []
    previous_gray = None
    previous_mask = None
    frames = 0
    pts_digest = hashlib.sha256()
    mask_total = 0.0
    cut_resets = 0
    caption_detected = 0
    caption_coverage = 0.0
    caption_fallback_frames = 0
    caption_fallback_reasons: dict[str, int] = {}
    exact_repeat_pairs = 0
    exact_repeat_mask_frames = 0
    exact_repeat_hold_ranges: list[list[int]] = []
    exact_repeat_ranges_truncated = False
    previous_caption = None
    previous_caption_stamp = None
    previous_source_frame_was_repeat = False
    previous_native_frame = None
    motion_peak_delta = 0.0
    motion_peak_frame = 0
    shot_motion_sum = 0.0
    shot_motion_observed = 0
    shot_motion_fast = 0
    shot_motion_cutlike = 0
    shot_caption_detected = 0
    shot_exact_repeat_pairs = 0
    frame_analysis = _FrameAnalysisCache(
        parallel_analysis=_admit_parallel_masks(width, height))
    try:
        with av.open(str(source)) as input_container, av.open(str(temporary_mask), "w", format="nut") as output:
            source_stream = input_container.streams.video[0]
            stream = output.add_stream("ffv1", rate=rate)
            stream.width, stream.height, stream.pix_fmt = width, height, "gray"
            stream.time_base = source_stream.time_base
            stream.codec_context.time_base = source_stream.time_base
            caption_stream = output.add_stream("ffv1", rate=rate)
            caption_stream.width, caption_stream.height, caption_stream.pix_fmt = width, height, "gray"
            caption_stream.time_base = source_stream.time_base
            caption_stream.codec_context.time_base = source_stream.time_base
            for frame in input_container.decode(source_stream):
                _cancel(writer)
                if frames >= MAX_FRAMES or time.monotonic() - started > MAX_ANALYSIS_SECONDS:
                    raise RuntimeError("Adaptive analysis exceeded its bounded frame/time budget.")
                if frame.pts is None or frame.time_base is None or frame.width != width or frame.height != height:
                    raise RuntimeError("Adaptive source frame geometry/timestamp changed.")
                timestamp = Fraction(frame.pts) * frame.time_base
                if timestamp != Fraction(frames, 1) / rate:
                    raise ValueError("Adaptive masking requires zero-origin exact CFR; source timestamps will not be normalized silently.")
                shot_start, shot_end = windows[window_index]
                if frames == shot_start:
                    frame_analysis.reset()
                    previous_native_frame = None
                    previous_mask = None
                    previous_gray = None
                    motion_peak_delta, motion_peak_frame = 0.0, shot_start
                    shot_motion_sum = 0.0
                    shot_motion_observed = 0
                    shot_motion_fast = 0
                    shot_motion_cutlike = 0
                    shot_caption_detected = 0
                    shot_exact_repeat_pairs = 0
                    if frames:
                        cut_resets += 1
                if frames and frames % 120 == 0 and frame_analysis._parallel_analysis:
                    # Concurrent editors can consume RAM midway through a
                    # longer shot. Downgrade to the bit-identical serial
                    # route; never launch extra model work to compensate.
                    if not _admit_parallel_masks(width, height):
                        frame_analysis.disable_parallel()
                sample_step = max(1, math.ceil((shot_end - shot_start) / 8))
                collect_metrics = (frames - shot_start) % sample_step == 0
                # A compact four-still contact-sheet equivalent is sufficient
                # for model-pilot nomination. The full RGB/caption/detail
                # protection passes still run on EVERY unique source frame.
                collect_spatial = collect_metrics and ((frames - shot_start) // sample_step) % 2 == 0
                exact_source_repeat = (
                    frames > shot_start
                    and _native_decoded_frames_equal(previous_native_frame, frame))
                previous_native_frame = frame
                # A verified *native-depth* hold has identical decoded RGB,
                # including any caption/texture pixels. Reuse the already
                # copied RGB plane BEFORE PyAV conversion and bypass the
                # redundant full RGB equality scan. Moving/10-bit-different
                # frames still go through their complete independent analysis.
                cached_rgb = (
                    frame_analysis._rgb_for_verified_native_hold()
                    if exact_source_repeat else None)
                rgb = cached_rgb if cached_rgb is not None else frame.to_ndarray(format="rgb24")
                if cached_rgb is None:
                    mask, caption, caption_evidence, measured_values, gray_small = frame_analysis.analyze(
                        rgb, collect_metrics=collect_metrics)
                else:
                    mask, caption, caption_evidence, measured_values, gray_small = frame_analysis.analyze(
                        rgb, collect_metrics=collect_metrics, _verified_native_hold=True)
                if exact_source_repeat:
                    # Exact RGB equality is verified over every source pixel.
                    # Never infer holds from a visually similar thumbnail:
                    # moving faces, balls and animation must retain real FI.
                    exact_repeat_pairs += 1
                    shot_exact_repeat_pairs += 1
                    if exact_repeat_hold_ranges and exact_repeat_hold_ranges[-1][1] == frames - 1:
                        exact_repeat_hold_ranges[-1][1] = frames
                    elif len(exact_repeat_hold_ranges) < 32:
                        exact_repeat_hold_ranges.append([frames - 1, frames])
                    else:
                        exact_repeat_ranges_truncated = True
                caption_detected += bool(caption_evidence["line_count"])
                shot_caption_detected += bool(caption_evidence["line_count"])
                caption_coverage += float(np.mean(caption > 0))
                if caption_evidence.get("status") == "conservative_edge_protection":
                    caption_fallback_frames += 1
                    reason = str(caption_evidence.get("fallback_reason") or "unknown")
                    caption_fallback_reasons[reason] = caption_fallback_reasons.get(reason, 0) + 1
                mask[caption > 0] = 0
                if previous_caption is not None:
                    # A target sample between source frames may contain a model
                    # ghost of either caption position. Protect their union, but
                    # never carry a new shot's lettering back across a cut.
                    if exact_source_repeat or (frames == shot_start and previous_source_frame_was_repeat):
                        # An exact source hold has no intermediate motion to
                        # reconstruct. Preserve the entire authored frame at
                        # output 60 FPS rather than allowing FI to shimmer
                        # over a still graphic, face or grain texture. At a
                        # shot boundary this applies ONLY to the preceding
                        # shot's final source interval, never the new shot.
                        held_caption = np.full_like(previous_caption, 255)
                        exact_repeat_mask_frames += 1
                    else:
                        held_caption = previous_caption if frames == shot_start else np.maximum(previous_caption, caption)
                    caption_frame = av.VideoFrame.from_ndarray(held_caption, format="gray")
                    caption_frame.pts, caption_frame.time_base = previous_caption_stamp
                    for packet in caption_stream.encode(caption_frame):
                        output.mux(packet)
                previous_caption = caption
                previous_caption_stamp = (frame.pts, frame.time_base)
                previous_source_frame_was_repeat = exact_source_repeat
                if previous_gray is not None:
                    change = float(np.mean(np.abs(gray_small.astype(np.float32) - previous_gray.astype(np.float32))))
                    if change > motion_peak_delta:
                        motion_peak_delta, motion_peak_frame = change, frames
                    if change > 32:
                        previous_mask = None
                        cut_resets += 1
                        shot_motion_cutlike += 1
                    else:
                        # Real source-frame mean luma change. Large cutlike
                        # discontinuities do not by themselves imply sustained
                        # sports/action motion and are excluded from this risk.
                        shot_motion_observed += 1
                        shot_motion_sum += change
                        shot_motion_fast += int(change >= 8)
                if previous_mask is not None:
                    # Smoothing can only REDUCE the current admitted contribution.
                    mask = np.minimum(mask, np.rint(0.75 * mask + 0.25 * previous_mask).astype(np.uint8))
                previous_mask, previous_gray = mask, gray_small
                np.maximum(union, mask, out=union)
                np.maximum(shot_union, mask, out=shot_union)
                mask_total += float(mask.mean()) / 255.0
                if collect_metrics:
                    measured = {"frame": frames, **measured_values}
                    if collect_spatial:
                        spatial = spatial_quality_evidence(rgb, mask, caption)
                        # Do not persist sixteen tile dictionaries per sample
                        # in a potentially long finished montage. Keep the
                        # ranked region and reliable scalar source evidence.
                        measured["spatial_quality"] = {
                            k: spatial[k] for k in (
                                "schema", "priority_score", "eligible_fraction",
                                "soft_non_enhanced_fraction", "sample_size")}
                        measured["spatial_quality"]["priority_regions"] = spatial["priority_regions"][:1]
                    shot_measurements.append(measured)
                    measurements.append(measured)
                mask_frame = av.VideoFrame.from_ndarray(mask, format="gray")
                mask_frame.pts, mask_frame.time_base = frame.pts, frame.time_base
                for packet in stream.encode(mask_frame):
                    output.mux(packet)
                pts_digest.update(f"{timestamp.numerator}/{timestamp.denominator}\n".encode("ascii"))
                frames += 1
                if frames == shot_end:
                    shot_crop = choose_processing_crop(shot_union, target_width, target_height)
                    shot_active = bool(np.any(shot_union))
                    shot_ratio = shot_crop[2] * shot_crop[3] / (width * height) if shot_active else 0.0
                    motion_evidence = {
                        "schema": "spiffy_shot_motion_risk_v1",
                        "observed_transitions": shot_motion_observed,
                        "cutlike_discontinuities": shot_motion_cutlike,
                        "mean_abs_luma_change": round(
                            shot_motion_sum / max(1, shot_motion_observed), 6),
                        "high_motion_fraction": round(
                            shot_motion_fast / max(1, shot_motion_observed), 6),
                        "thumbnail_size": list(motion_thumbnail_size(width, height)),
                        "scope": "source_frame_luma_changes_not_optical_flow_or_sports_semantics",
                    }
                    treatment = choose_visual_treatment(
                        shot_measurements, motion_evidence=motion_evidence,
                        caption_detected_fraction=shot_caption_detected / (shot_end - shot_start))
                    shots.append({"start_frame": shot_start, "end_frame": shot_end,
                                  "crop": shot_crop, "processed_area_ratio": shot_ratio,
                                  "route": "source_preserved" if not shot_active else
                                           "masked_crop" if shot_ratio < 1 else "masked_full_frame",
                                  "effective_controls": choose_natural_controls(
                                      shot_measurements, visual_treatment=treatment),
                                  "visual_treatment": treatment,
                                  "motion_evidence": motion_evidence,
                                  "spatial_quality": summarize_shot_spatial_quality(
                                      shot_measurements, shot_start, shot_end),
                                  "motion_peak_delta": round(motion_peak_delta, 6), "motion_peak_frame": motion_peak_frame,
                                  "exact_repeat_pairs": shot_exact_repeat_pairs,
                                  "measurements": shot_measurements})
                    shot_union.fill(0)
                    shot_measurements = []
                    window_index += 1
                if frames % 30 == 0:
                    if temporary_mask.stat().st_size > MAX_MASK_BYTES:
                        raise RuntimeError("Adaptive mask exceeded the 512 MiB job-temp budget.")
                    writer.update(phase="adaptive_analysis", message=f"Analyzing detail/protection {frames}/{count}",
                                  metrics={"adaptive_frames_analyzed": frames, "adaptive_total_frames": count})
            for packet in stream.encode():
                output.mux(packet)
            if previous_caption is not None:
                if previous_source_frame_was_repeat:
                    # EOF extension is an exact hold too; do not let the last
                    # cloned presentation interval acquire invented motion.
                    previous_caption = np.full_like(previous_caption, 255)
                    exact_repeat_mask_frames += 1
                caption_frame = av.VideoFrame.from_ndarray(previous_caption, format="gray")
                caption_frame.pts, caption_frame.time_base = previous_caption_stamp
                for packet in caption_stream.encode(caption_frame):
                    output.mux(packet)
            for packet in caption_stream.encode():
                output.mux(packet)
        if frames != count or (info.get("video_pts_sha256") and pts_digest.hexdigest() != info["video_pts_sha256"]):
            raise RuntimeError("Adaptive mask is not bound to the complete source PTS sequence.")
        # Only after the full cheap detail/protection pass do optional subject
        # models run. A flat/protected shot never needs detector/SAM2 startup.
        subject_summary, subject_frames = prepare_subjects(
            source, info, windows, rate, writer, policy, shot_evidence=shots)
        if subject_frames:
            union, mask_total = refine_subject_stream(
                temporary_mask, subject_frames, shots, rate, width, height,
                target_width, target_height, frames, writer)
        for shot, evidence in zip(shots, subject_summary["shots"]):
            shot["subject_refinement"] = dict(evidence)
        if source.stat().st_size != stat.st_size or source.stat().st_mtime_ns != stat.st_mtime_ns or _hash_file(source) != source_sha:
            raise RuntimeError("Source changed during adaptive analysis.")
        _cancel(writer)
        os.replace(temporary_mask, mask_path)
    except BaseException:
        temporary_mask.unlink(missing_ok=True)
        raise
    finally:
        frame_analysis.close()
    crop = choose_processing_crop(union, target_width, target_height)
    ratio = crop[2] * crop[3] / (width * height)
    controls = choose_natural_controls(measurements)
    # When the setup-cost policy batches several shots into one native call,
    # its single model recipe must not become more aggressive than any
    # source-bound shot's guarded recipe. Split render units retain their
    # own controls; the global values are the conservative safe common set.
    active_controls = [shot["effective_controls"] for shot in shots
                       if shot["route"] != "source_preserved"]
    if active_controls:
        controls["details"] = min(controls["details"],
                                  *(item["details"] for item in active_controls))
        controls["blend"] = max(controls["blend"],
                                *(item["blend"] for item in active_controls))
    params = "".join(f":{k}={v}" for k, v in controls.items())
    active = bool(np.any(union))
    units, unit_reason = select_render_units(shots, width, height, rate)
    if units:
        ratio = sum(u["crop"][2] * u["crop"][3] * (u["end_frame"] - u["start_frame"])
                    for u in units if u["route"] != "source_preserved") / (width * height * frames)
    plan = {
        "schema": SCHEMA, "identity": identity, "policy": normalize_policy(policy),
        "source_width": width, "source_height": height, "target_width": target_width, "target_height": target_height,
        "frame_count": frames, "source_fps": str(rate), "source_pts_sha256": pts_digest.hexdigest(),
        "mask_path": str(mask_path), "mask_sha256": _hash_file(mask_path),
        "caption_protection": {"stream_index": 1, "policy": "authored_glyph_and_exact_native_hold_v3",
                               "detected_frames": caption_detected,
                               "mean_coverage": caption_coverage / max(1, frames),
                               "conservative_fallback_frames": caption_fallback_frames,
                               "conservative_fallback_reasons": dict(sorted(caption_fallback_reasons.items())),
                               "exact_repeat_pairs": exact_repeat_pairs,
                               "exact_repeat_mask_frames": exact_repeat_mask_frames,
                               "exact_repeat_hold_ranges": exact_repeat_hold_ranges,
                               "exact_repeat_ranges_truncated": exact_repeat_ranges_truncated,
                               "exact_repeat_policy": "full_native_decoded_pixel_planes_equal_within_same_shot",
                               "semantic_recognition": False},
        "crop": crop, "processed_area_ratio": ratio if active else 0.0,
        "route": "shot_sequence" if units else "source_preserved" if not active else "masked_crop" if ratio < 1 else "masked_full_frame",
        "scene_analysis": scene_analysis, "shots": shots, "render_units": units,
        "subject_refinement": subject_summary,
        "render_unit_decision": unit_reason,
        "effective_controls": controls, "effective_params": params,
        "mean_enhancement_contribution": mask_total / max(1, frames),
        "measurements": measurements, "temporal_protection_resets": cut_resets,
        "quality_claim": "bounded measured heuristic; not semantic caption/matte or perceptual acceptance",
        "analysis_seconds": round(time.monotonic() - started, 4), "cache_hit": False,
        "analysis_efficiency": frame_analysis.stats(),
    }
    plan["fingerprint"] = plan_fingerprint(plan)
    atomic_write_json(manifest_path, plan)
    return plan


def refine_subject_stream(mask_path: Path, subject_frames: dict[int, dict[str, str]],
                          shots: list[dict[str, Any]], rate: Fraction, width: int, height: int,
                          target_width: int, target_height: int, count: int, writer: Any) -> tuple[Any, float]:
    """Restrict lossless masks after admission, without decoding source video again.

    Only a few mask frames are resident. Existing glyph masks are copied exactly;
    accepted subject masks cannot add enhancement to a protected source pixel.
    """
    import av
    import numpy as np
    from itertools import zip_longest
    from adaptive_subjects import load_subject_frame, restrict_to_subject

    refined = mask_path.with_suffix(".refined.nut")
    union = np.zeros((height, width), dtype=np.uint8)
    shot_union = np.zeros_like(union)
    total = 0.0
    index = 0
    shot_index = 0
    started = time.monotonic()
    try:
        with av.open(str(mask_path)) as a, av.open(str(mask_path)) as b, av.open(str(refined), "w", format="nut") as output:
            streams = []
            for original in (a.streams.video[0], b.streams.video[1]):
                stream = output.add_stream("ffv1", rate=rate)
                stream.width, stream.height, stream.pix_fmt = width, height, "gray"
                stream.time_base = original.time_base
                stream.codec_context.time_base = original.time_base
                streams.append(stream)
            for index, (detail, caption) in enumerate(zip_longest(a.decode(video=0), b.decode(video=1)), start=1):
                _cancel(writer)
                if index > count or time.monotonic() - started > MAX_ANALYSIS_SECONDS:
                    raise RuntimeError("Subject mask refinement exceeded its bounded extent.")
                ordinal = index - 1
                for frame in (detail, caption):
                    if (frame is None or frame.pts is None or (frame.width, frame.height) != (width, height)
                            or Fraction(frame.pts) * frame.time_base != Fraction(ordinal, 1) / rate):
                        raise RuntimeError("Subject mask stream is not bound to the exact source frames.")
                mask = detail.to_ndarray(format="gray")
                if ordinal in subject_frames:
                    alpha = load_subject_frame(subject_frames[ordinal], (height, width))
                    mask = restrict_to_subject(mask, alpha)
                np.maximum(union, mask, out=union)
                np.maximum(shot_union, mask, out=shot_union)
                total += float(mask.mean()) / 255.0
                for stream, pixels, original in ((streams[0], mask, detail),
                                                (streams[1], caption.to_ndarray(format="gray"), caption)):
                    frame = av.VideoFrame.from_ndarray(pixels, format="gray")
                    frame.pts, frame.time_base = original.pts, original.time_base
                    for packet in stream.encode(frame):
                        output.mux(packet)
                if index == shots[shot_index]["end_frame"]:
                    crop = choose_processing_crop(shot_union, target_width, target_height)
                    active = bool(np.any(shot_union))
                    ratio = crop[2] * crop[3] / (width * height) if active else 0.0
                    shots[shot_index].update(crop=crop, processed_area_ratio=ratio,
                        route="source_preserved" if not active else "masked_crop" if ratio < 1 else "masked_full_frame")
                    shot_union.fill(0)
                    shot_index += 1
                if index % 30 == 0 and refined.stat().st_size > MAX_MASK_BYTES:
                    raise RuntimeError("Refined mask exceeded its bounded disk budget.")
            for stream in streams:
                for packet in stream.encode():
                    output.mux(packet)
        if index != count or shot_index != len(shots) or refined.stat().st_size > MAX_MASK_BYTES:
            raise RuntimeError("Refined mask has incomplete source coverage or excessive size.")
        _cancel(writer)
        os.replace(refined, mask_path)
        return union, total
    finally:
        refined.unlink(missing_ok=True)


def assess_candidate(source: Path, candidate: Path, plan: dict[str, Any], writer: Any) -> dict[str, Any]:
    """Bounded objective damage rejection, not a claim of perceptual equivalence.

    Decode the whole candidate to check frame custody; inspect up to 16 spatial
    samples at source dimensions. In between those samples, inspect ONLY
    tiny aspect-preserving images for catastrophic color/frame corruption.
    Missing or reordered frames cannot pass on a good-looking first frame,
    and one-frame render glitches cannot hide between sampled details.
    Input files remain immutable. This is NOT a fine-grained per-frame
    perceptual metric: local texture/caption checks still use original masks.
    """
    import av
    import cv2
    import numpy as np
    from itertools import zip_longest
    sample_step = max(1, math.ceil(int(plan["frame_count"]) / 16))
    rows = []
    # Keep only the worst source-relative preview, not one Python dict for
    # every otherwise-unsampled frame. Long videos can have tens of
    # thousands of previews; their full history adds no reporting value.
    coarse_extra_frames = 0
    worst_coarse: dict[str, float | int] | None = None
    coarse_severe = False
    source_size = (int(plan["source_width"]), int(plan["source_height"]))
    coarse_width, coarse_height = motion_thumbnail_size(*source_size)
    count = 0
    previous_source_preview = None
    previous_candidate_preview = None
    frozen_motion_frames = 0
    worst_motion_freeze_frame = None
    worst_frozen_source_change = 0.0
    exact_hold_ranges = (plan.get("caption_protection") or {}).get("exact_repeat_hold_ranges") or []
    documented_exact_holds = [
        (int(pair[0]), int(pair[1])) for pair in exact_hold_ranges
        if (isinstance(pair, (list, tuple)) and len(pair) == 2
            and all(type(v) is int for v in pair) and 0 <= pair[0] < pair[1] <= int(plan["frame_count"]))
    ]
    exact_hold_checked = 0
    exact_hold_shimmer = 0
    worst_exact_hold_shimmer_frame = None
    worst_exact_hold_shimmer_amount = 0.0
    localized_damage_frames = 0
    worst_localized_damage_frame = None
    worst_localized_damage_score = 0.0
    started = time.monotonic()
    with av.open(str(source)) as src, av.open(str(candidate)) as out, av.open(plan["mask_path"]) as masks:
        for index, trio in enumerate(zip_longest(src.decode(video=0), out.decode(video=0), masks.decode(video=0))):
            _cancel(writer)
            if time.monotonic() - started > MAX_ANALYSIS_SECONDS:
                raise RuntimeError("Adaptive candidate assessment exceeded the bounded validation budget.")
            if any(frame is None for frame in trio):
                raise RuntimeError("Adaptive candidate/source/mask frame counts differ.")
            original, rendered, mask_frame = trio
            stamps = [Fraction(frame.pts) * frame.time_base for frame in trio]
            if len(set(stamps)) != 1:
                raise RuntimeError("Adaptive candidate/source/mask timestamps differ.")
            if rendered.width != plan["target_width"] or rendered.height != plan["target_height"]:
                raise RuntimeError("Adaptive candidate has incorrect working dimensions.")
            count += 1
            # Screen all source/candidate frame transitions. A single exact
            # repeated model-output frame can be invisible to source-size
            # detail sampling (and below gross pixel-error thresholds) while
            # the authored source is clearly moving. Use only two tiny
            # aspect-preserving previews, never an extra source decode or
            # full-frame output history. An authored source hold is safe.
            before = original.reformat(
                width=coarse_width, height=coarse_height,
                format="rgb24", interpolation="LANCZOS").to_ndarray()
            after = rendered.reformat(
                width=coarse_width, height=coarse_height,
                format="rgb24", interpolation="LANCZOS").to_ndarray()
            if previous_source_preview is not None:
                source_change = float(np.mean(cv2.absdiff(before, previous_source_preview)))
                candidate_change = float(np.mean(cv2.absdiff(after, previous_candidate_preview)))
                if source_change >= 8.0 and candidate_change <= 0.15:
                    frozen_motion_frames += 1
                    if source_change > worst_frozen_source_change:
                        worst_frozen_source_change = source_change
                        worst_motion_freeze_frame = index
                if any(start <= index - 1 < end for start, end in documented_exact_holds):
                    # A full-RGB source hold was proven during source analysis.
                    # Native enhancement must not produce shimmer on a frozen
                    # original frame (even if global MAE stays below tolerance).
                    exact_hold_checked += 1
                    if candidate_change >= 8.0:
                        exact_hold_shimmer += 1
                        if candidate_change > worst_exact_hold_shimmer_amount:
                            worst_exact_hold_shimmer_amount = candidate_change
                            worst_exact_hold_shimmer_frame = index
            previous_source_preview, previous_candidate_preview = before, after
            if index % sample_step and index != int(plan["frame_count"]) - 1:
                # A small, differently colored BAD frame might sit between
                # otherwise beautiful full-frame samples. Compare corresponding
                # original and rendered RGB thumbnail pixels at the SAME PTS.
                # Source-authored flashes and cuts are safe when both match.
                difference = cv2.absdiff(after, before)
                # A one-frame defect covering only ~1% of the picture can
                # disappear in both global MAE and global p95. Inspect the
                # SAME decoded source-relative difference using 12x12 local
                # regions. INTER_AREA bounds the region's mean color error,
                # not a model-generated texture metric or a new AI pass.
                tiles = cv2.resize(difference, (12, 12),
                                   interpolation=cv2.INTER_AREA)
                local_error = float(np.max(tiles))
                if local_error >= 75.0:
                    localized_damage_frames += 1
                    if local_error > worst_localized_damage_score:
                        worst_localized_damage_score = local_error
                        worst_localized_damage_frame = index
                preview_evidence = {
                    "frame": index,
                    "mean_abs_rgb": round(float(difference.mean()), 6),
                    "p95_abs_rgb": round(float(np.percentile(difference, 95)), 6),
                }
                coarse_extra_frames += 1
                coarse_severe |= bool(
                    preview_evidence["mean_abs_rgb"] > 24
                    or preview_evidence["p95_abs_rgb"] > 80)
                if (worst_coarse is None or
                        preview_evidence["mean_abs_rgb"] > worst_coarse["mean_abs_rgb"]):
                    # Stable ties favor the earliest frame, just like the
                    # previous max(mean, -frame) summary.
                    worst_coarse = preview_evidence
                continue
            rgb = original.to_ndarray(format="rgb24")
            resized = rendered.reformat(width=original.width, height=original.height,
                                        format="rgb24", interpolation="LANCZOS").to_ndarray()
            # Both decoded images are RGB uint8. Native absdiff produces the
            # same integer errors without two full-frame float conversions
            # and a third temporary difference array (expensive at 4K).
            delta = cv2.absdiff(resized, rgb)
            mask = mask_frame.to_ndarray(format="gray")
            protected = mask <= 2
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            changed = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            rows.append({"frame": index, "mean_abs_rgb": float(delta.mean()),
                         "p99_abs_rgb": float(np.percentile(delta, 99)),
                         "protected_p99": float(np.percentile(delta[protected], 99)) if protected.any() else 0.0,
                         # Mean(A-B) == Mean(A)-Mean(B), with OpenCV's
                         # double-precision accumulation and no float images.
                         "signed_luma_shift": float(cv2.mean(changed)[0] - cv2.mean(gray)[0])})
    if count != int(plan["frame_count"]) or not rows:
        raise RuntimeError("Adaptive candidate validation did not cover the complete source.")
    reasons = []
    if max(row["mean_abs_rgb"] for row in rows) > 6:
        reasons.append("excessive_source_deviation")
    if max(row["p99_abs_rgb"] for row in rows) > 40:
        reasons.append("large_local_changes")
    if max(row["protected_p99"] for row in rows) > 24:
        reasons.append("protected_region_changed")
    if max(abs(row["signed_luma_shift"]) for row in rows) > 3:
        reasons.append("grade_brightness_drift")
    shifts = [row["signed_luma_shift"] for row in rows]
    if shifts and max(shifts) - min(shifts) > 4:
        reasons.append("temporal_brightness_instability")
    # Deliberately higher than ordinary model/source pixel differences.
    # This is a gross-frame corruption guard, not an image-quality score.
    if coarse_severe:
        reasons.append("unsampled_severe_frame_damage")
    if frozen_motion_frames:
        reasons.append("unexpected_frozen_motion_frame")
    if exact_hold_shimmer:
        reasons.append("unexpected_shimmer_on_exact_hold")
    if localized_damage_frames:
        reasons.append("unsampled_localized_patch_damage")
    return {"status": "REJECT" if reasons else "PASS_OBJECTIVE_GUARDS", "reasons": reasons,
            "frames_decoded": count, "sample_count": len(rows), "samples": rows,
            "coarse_screened_frames": count,
            "coarse_extra_frames": coarse_extra_frames,
            "worst_coarse_frame": worst_coarse["frame"] if worst_coarse else None,
            "worst_coarse_mean_abs_rgb": worst_coarse["mean_abs_rgb"] if worst_coarse else None,
            "coarse_screen_policy": "all_unsampled_frames_screened_at_aspect_preserving_thumbnail",
            "freeze_screened_transitions": max(0, count - 1),
            "unexpected_frozen_motion_frames": frozen_motion_frames,
            "worst_motion_freeze_frame": worst_motion_freeze_frame,
            "worst_frozen_source_change": round(worst_frozen_source_change, 6),
            "freeze_policy": "only_source_motion_ge_8_and_output_change_le_0_15; authored_source_holds_admitted",
            "exact_hold_transitions_screened": exact_hold_checked,
            "unexpected_exact_hold_shimmer_frames": exact_hold_shimmer,
            "worst_exact_hold_shimmer_frame": worst_exact_hold_shimmer_frame,
            "worst_exact_hold_shimmer_change": round(worst_exact_hold_shimmer_amount, 6),
            "exact_hold_shimmer_policy": "source_exact_RGB_pair_and_output_preview_change_ge_8",
            "localized_damage_frames": localized_damage_frames,
            "worst_localized_damage_frame": worst_localized_damage_frame,
            "worst_localized_damage_score": round(worst_localized_damage_score, 6),
            "localized_damage_policy": "12x12_source_relative_coarse_region_mean_channel_error_ge_75",
            "seconds": round(time.monotonic() - started, 4),
            "visual_acceptance": "not_established_by_objective_metrics"}


def assess_temporal_candidate(source: Path, candidate: Path, plan: dict[str, Any],
                              target_rate: str, cut_frames: list[int], writer: Any) -> dict[str, Any]:
    """Check authored glyph samples and cut holds AFTER native interpolation.

    Photographic motion between source frames is not compared to a frozen
    original. Only admitted glyph cores and explicit cut holds have that
    reference. Between detailed samples, tiny source-bound caption previews
    reject catastrophic one-frame text loss without penalizing interpolation
    of the photographic picture. Three tiny output/source previews also
    reject localized invented black/white and saturated-color tears,
    including strict one-neighbor checks on the first and last output frames.
    Not a general taste or motion quality score.
    """
    import av
    import cv2
    import numpy as np
    from itertools import zip_longest

    def localized_flash_scores(candidate_pixels: Any, source_pixels: Any,
                               neighbors: tuple[Any, ...]) -> tuple[float, float]:
        """Bounded 96px-long-edge source-relative tear check (blank, chromatic).

        Only uniform extreme-color areas with a material footprint count.
        Every suspicious pixel must disagree with the authored source AND
        each available output neighbor; legitimate source graphics survive.
        Boundary frames deliberately have one neighbor instead of two.
        """
        size = candidate_pixels.shape[0] * candidate_pixels.shape[1]
        required = max(8, math.ceil(size * .01))

        def peak_coverage(mask: Any) -> float:
            if cv2.countNonZero(mask) < required:
                return 0.0
            tiles = cv2.resize(mask, (12, 12), interpolation=cv2.INTER_AREA)
            return float(tiles.max()) / 255.0

        blank_peak = 0.0
        for low, high, normal_low, normal_high in (
                (0, 8, 65, 255), (247, 255, 0, 195)):
            mask = cv2.inRange(candidate_pixels, (low,) * 3, (high,) * 3)
            if cv2.countNonZero(mask) < required:
                continue
            for reference in (source_pixels, *neighbors):
                normal = cv2.inRange(
                    reference, (normal_low,) * 3, (normal_high,) * 3)
                cv2.bitwise_and(mask, normal, dst=mask)
            blank_peak = max(blank_peak, peak_coverage(mask))

        # A single saturated blue/green/magenta block has a normal-looking
        # global brightness. Restrict to extreme RGB pixels and test each
        # location against source plus adjacent outputs (not a whole-frame
        # mean that moving objects can easily satisfy).
        saturated = ((candidate_pixels.max(axis=2) >= 247)
                     & (candidate_pixels.min(axis=2) <= 8))
        color_peak = 0.0
        if int(np.count_nonzero(saturated)) >= required:
            color_mask = saturated.astype(np.uint8) * 255
            for reference, threshold in ((source_pixels, 100), *(
                    (neighbor, 90) for neighbor in neighbors)):
                difference = cv2.absdiff(candidate_pixels, reference)
                disagrees = (difference.max(axis=2) >= threshold).astype(np.uint8) * 255
                cv2.bitwise_and(color_mask, disagrees, dst=color_mask)
            color_peak = peak_coverage(color_mask)
        return blank_peak, color_peak

    verify_artifacts(plan)
    rate, source_rate = Fraction(target_rate), Fraction(plan["source_fps"])
    source_count = int(plan["frame_count"])
    if not source_rate < rate <= 120:
        raise ValueError("Temporal candidate assessment requires a genuine up-rate path.")
    expected_count = math.ceil(Fraction(source_count) * rate / source_rate)
    cut_ordinals: set[int] = set()
    for cut in cut_frames:
        if type(cut) is not int or not 0 < cut < source_count:
            raise ValueError("Temporal assessment cut is outside source frames.")
        cut_ordinals.update(range(max(0, math.ceil(Fraction(cut - 1) * rate / source_rate)),
                                  min(expected_count, math.ceil(Fraction(cut + 1) * rate / source_rate))))
    stride = max(1, math.ceil(expected_count / 24))
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    decoded = 0
    current_index = -1
    current = None
    coarse_caption_screened = 0
    worst_caption_frame = None
    worst_caption_mae = 0.0
    coarse_caption_failed = False
    preview_width, preview_height = motion_thumbnail_size(
        int(plan["source_width"]), int(plan["source_height"]))
    preview_reference = None
    previous_render_preview = None
    source_interval_output_count = 0
    source_interval_max_render_motion = 0.0
    motion_intervals_screened = 0
    flatline_source_intervals = 0
    worst_flatline_source_interval = None
    source_preview_mean = None
    flash_history: list[tuple[int, float, float, bool,
                              tuple[float, float, float], tuple[float, float, float]]] = []
    # Keep tiny source/output pixels alongside the existing scalar history.
    # Bound both histories to three frames, regardless of clip duration.
    flash_pixel_history: list[tuple[Any, Any]] = []
    first_flash_pair = None
    first_flash_pixels = None
    flash_screened_frames = 0
    unexpected_black_frames = 0
    unexpected_white_frames = 0
    unexpected_saturated_color_frames = 0
    localized_blank_frames = 0
    worst_localized_blank_frame = None
    worst_localized_blank_fraction = 0.0
    localized_color_frames = 0
    worst_localized_color_frame = None
    worst_localized_color_fraction = 0.0
    worst_unexpected_flash_frame = None
    worst_unexpected_flash_contrast = 0.0
    worst_unexpected_color_frame = None
    worst_unexpected_color_contrast = 0.0
    source_preview_pixels = None
    source_preview_channels = None
    # A source cut can legitimately be held on either side; don't grade it
    # as interpolated motion. This includes explicit cuts and large unowned
    # discontinuities. A video with near-60 source cadence also won't have
    # enough synthesized samples to trigger this check.
    authored_cuts = set(cut_frames)
    with av.open(str(source)) as src, av.open(str(candidate)) as out, av.open(plan["mask_path"]) as masks:
        originals = iter(zip_longest(src.decode(video=0), masks.decode(video=1)))
        for index, rendered in enumerate(out.decode(video=0)):
            _cancel(writer)
            if index >= expected_count or time.monotonic() - started > MAX_ANALYSIS_SECONDS:
                raise RuntimeError("Temporal candidate exceeded its frame/time verification bound.")
            if rendered.pts is None or rendered.time_base is None or Fraction(rendered.pts) * rendered.time_base != Fraction(index) / rate:
                raise RuntimeError("Temporal candidate is not on the exact authored output clock.")
            source_index = min(source_count - 1, math.floor(Fraction(index) * source_rate / rate))
            while current_index < source_index:
                following = next(originals, None)
                if following is None or any(frame is None for frame in following):
                    raise RuntimeError("Temporal protection reference ended early.")
                # The new source preview is needed for every output in its
                # interval. Reuse the old preview for motion comparison instead
                # of decoding/resizing source images more than once.
                next_source_image = following[0].reformat(
                    width=preview_width, height=preview_height,
                    format="rgb24", interpolation="LANCZOS").to_ndarray()
                if current is not None:
                    previous_source_image = source_preview_pixels
                    if previous_source_image is None:
                        previous_source_image = current[0].reformat(
                            width=preview_width, height=preview_height,
                            format="rgb24", interpolation="LANCZOS").to_ndarray()
                    source_change = float(np.mean(cv2.absdiff(
                        previous_source_image, next_source_image)))
                    motion_intervals_screened += 1
                    if (current_index + 1 not in authored_cuts
                            and 8.0 <= source_change <= 64.0
                            and source_interval_output_count >= 2
                            and source_interval_max_render_motion <= 0.25):
                        flatline_source_intervals += 1
                        worst_flatline_source_interval = current_index
                current = following
                current_index += 1
                preview_reference = None
                source_preview_pixels = next_source_image
                source_preview_channels = tuple(
                    float(value) for value in cv2.mean(next_source_image)[:3])
                source_preview_mean = sum(source_preview_channels) / 3.0
                previous_render_preview = None
                source_interval_output_count = 0
                source_interval_max_render_motion = 0.0
                if any(frame.pts is None or Fraction(frame.pts) * frame.time_base != Fraction(current_index) / source_rate
                       for frame in current):
                    raise RuntimeError("Temporal protection mask/reference timestamp mismatch.")
            decoded += 1
            at_cut = index in cut_ordinals
            # A CFR header can look perfect while the generated "60 FPS"
            # video contains only duplicated 24/30 FPS motion. Check the
            # synthesized samples inside each source-frame interval using
            # tiny current-output previews, then wait for the NEXT source
            # frame to establish whether actual motion existed. We never
            # demand motion across source-authored static holds or cuts.
            rendered_preview = rendered.reformat(
                width=preview_width, height=preview_height,
                format="rgb24", interpolation="LANCZOS").to_ndarray()
            if source_preview_mean is None:
                source_preview_pixels = current[0].reformat(
                    width=preview_width, height=preview_height,
                    format="rgb24", interpolation="LANCZOS").to_ndarray()
                source_preview_channels = tuple(
                    float(value) for value in cv2.mean(source_preview_pixels)[:3])
                source_preview_mean = sum(source_preview_channels) / 3.0
            output_channels = tuple(
                float(value) for value in cv2.mean(rendered_preview)[:3])
            flash_history.append((
                index, sum(output_channels) / 3.0,
                source_preview_mean, at_cut, output_channels,
                source_preview_channels))
            flash_pixel_history.append((rendered_preview, source_preview_pixels))
            if index == 1:
                # The ordinary three-frame window cannot score output frame
                # zero. Hold only its source-bound sample and immediate
                # neighbor; the last two samples already survive in history.
                first_flash_pair = (flash_history[0], flash_history[1])
                first_flash_pixels = (flash_pixel_history[0], flash_pixel_history[1])
            if len(flash_history) == 3:
                (_, previous_mean, _, previous_cut, previous_color, _), (
                    middle_idx, middle_mean, middle_source_mean, middle_cut,
                    middle_color, middle_source_color), (
                    _, next_mean, _, next_cut, next_color, _) = flash_history
                flash_screened_frames += 1
                # Single-frame near-black/white inventions are not valid
                # interpolation. Source-authored flashes are exempt using
                # their exact source-frame brightness; explicit cut guards
                # remain authoritative. We do not infer damage from ordinary
                # color shifts, action motion or camera exposure changes.
                if not (previous_cut or middle_cut or next_cut):
                    unexpected_black = (
                        middle_mean <= 6.0
                        and min(previous_mean, next_mean, middle_source_mean) >= 65.0)
                    unexpected_white = (
                        middle_mean >= 249.0
                        and max(previous_mean, next_mean, middle_source_mean) <= 195.0)
                    if unexpected_black or unexpected_white:
                        unexpected_black_frames += int(unexpected_black)
                        unexpected_white_frames += int(unexpected_white)
                        contrast = abs(middle_mean - (previous_mean + next_mean) / 2.0)
                        if contrast > worst_unexpected_flash_contrast:
                            worst_unexpected_flash_contrast = contrast
                            worst_unexpected_flash_frame = middle_idx
                    # A one-frame solid magenta/green/blue native failure may
                    # have an ordinary average brightness, evading the black
                    # and white checks above. Reject only an extreme isolated
                    # chromatic flash unlike BOTH its source and neighbors.
                    # Actual source-authored color strobes are not anomalies.
                    saturated = min(middle_color) <= 8.0 and max(middle_color) >= 247.0
                    source_disagreement = max(
                        abs(a - b) for a, b in zip(middle_color, middle_source_color))
                    previous_disagreement = max(
                        abs(a - b) for a, b in zip(middle_color, previous_color))
                    next_disagreement = max(
                        abs(a - b) for a, b in zip(middle_color, next_color))
                    neighbors_agree = max(
                        abs(a - b) for a, b in zip(previous_color, next_color)) <= 16.0
                    if (saturated and source_disagreement >= 100.0
                            and previous_disagreement >= 90.0
                            and next_disagreement >= 90.0 and neighbors_agree):
                        unexpected_saturated_color_frames += 1
                        if source_disagreement > worst_unexpected_color_contrast:
                            worst_unexpected_color_contrast = source_disagreement
                            worst_unexpected_color_frame = middle_idx
                    # Localized extreme colors evade the whole-frame means.
                    # Reuse tiny decoded previews, with exact source ownership
                    # and both output neighbors to avoid invented glitches.
                    previous_pixels = flash_pixel_history[0][0]
                    middle_pixels, middle_source_pixels = flash_pixel_history[1]
                    next_pixels = flash_pixel_history[2][0]
                    blank, chromatic = localized_flash_scores(
                        middle_pixels, middle_source_pixels,
                        (previous_pixels, next_pixels))
                    if blank >= .55:
                        localized_blank_frames += 1
                        if blank > worst_localized_blank_fraction:
                            worst_localized_blank_fraction = blank
                            worst_localized_blank_frame = middle_idx
                    if chromatic >= .55:
                        localized_color_frames += 1
                        if chromatic > worst_localized_color_fraction:
                            worst_localized_color_fraction = chromatic
                            worst_localized_color_frame = middle_idx
                flash_history.pop(0)
                flash_pixel_history.pop(0)
            if previous_render_preview is not None:
                delta = float(np.mean(cv2.absdiff(
                    rendered_preview, previous_render_preview)))
                source_interval_max_render_motion = max(
                    source_interval_max_render_motion, delta)
            previous_render_preview = rendered_preview
            source_interval_output_count += 1
            if index % stride and not at_cut and index != expected_count - 1:
                # The source/next-caption union is already the authoritative
                # mask for this PTS. Sample its strong core at a tiny fixed
                # aspect-safe size; photographic motion outside that core is
                # never compared to an artificially held source frame.
                if preview_reference is None:
                    original, protected = current
                    tiny_mask = protected.reformat(
                        width=preview_width, height=preview_height,
                        format="gray", interpolation="LANCZOS").to_ndarray()
                    core = tiny_mask >= 220
                    if int(np.count_nonzero(core)) >= 8:
                        # Reuse the exact decoded source thumbnail already
                        # needed for gross-flash validation. No second resize
                        # or additional source-frame decode is necessary.
                        preview_reference = (source_preview_pixels, core)
                    else:
                        preview_reference = (None, None)
                tiny_original, core = preview_reference
                if core is not None:
                    delta = np.abs(
                        rendered_preview.astype(np.int16) - tiny_original.astype(np.int16))
                    difference = delta[core]
                    mae = float(difference.mean())
                    coarse_caption_screened += 1
                    if worst_caption_frame is None or mae > worst_caption_mae:
                        worst_caption_frame, worst_caption_mae = index, mae
                    if mae > 24 or float(np.percentile(difference, 95)) > 80:
                        coarse_caption_failed = True
                continue
            original, mask_frame = current
            rgb = rendered.to_ndarray(format="rgb24")
            reference = original.reformat(width=rendered.width, height=rendered.height,
                format="rgb24", interpolation="LANCZOS").to_ndarray()
            mask = cv2.resize(mask_frame.to_ndarray(format="gray"),
                              (rendered.width, rendered.height), interpolation=cv2.INTER_LINEAR)
            core = mask >= 250
            delta = np.abs(rgb.astype(np.int16) - reference.astype(np.int16))
            samples.append({"frame": index, "source_frame": source_index, "cut_hold": at_cut,
                "caption_pixels": int(np.count_nonzero(core)),
                "caption_p99": float(np.percentile(delta[core], 99)) if core.any() else 0.0,
                "cut_mean": float(delta.mean()) if at_cut else 0.0,
                "cut_p99": float(np.percentile(delta, 99)) if at_cut else 0.0})
        if decoded != expected_count or current_index != source_count - 1:
            raise RuntimeError("Temporal candidate/reference did not cover the complete source extent.")
        if next(originals, None) is not None:
            raise RuntimeError("Temporal candidate/reference contains unowned trailing frames.")
    boundary_flash_screened_frames = 0
    if decoded >= 3 and first_flash_pair is not None and len(flash_history) == 2:
        # Single-frame decoder/model failures can also occur at the very
        # first or last output frame. They have only ONE output neighbor, so
        # require strong disagreement with the exact authored source frame
        # as well; source-authored opening/closing flashes remain legitimate.
        boundary_pixels = (first_flash_pixels, (flash_pixel_history[-1], flash_pixel_history[-2]))
        for (boundary, neighbor), (pixels, neighbor_pixels) in zip(
                (first_flash_pair, (flash_history[-1], flash_history[-2])), boundary_pixels):
            frame, brightness, source_brightness, at_cut, color, source_color = boundary
            _, neighbor_brightness, _, neighbor_cut, neighbor_color, _ = neighbor
            if at_cut or neighbor_cut:
                continue
            boundary_flash_screened_frames += 1
            black = brightness <= 6.0 and min(neighbor_brightness, source_brightness) >= 65.0
            white = brightness >= 249.0 and max(neighbor_brightness, source_brightness) <= 195.0
            if black or white:
                unexpected_black_frames += int(black)
                unexpected_white_frames += int(white)
                contrast = abs(brightness - neighbor_brightness)
                if contrast > worst_unexpected_flash_contrast:
                    worst_unexpected_flash_contrast = contrast
                    worst_unexpected_flash_frame = frame
            saturated = min(color) <= 8.0 and max(color) >= 247.0
            source_disagreement = max(abs(a - b) for a, b in zip(color, source_color))
            neighbor_disagreement = max(abs(a - b) for a, b in zip(color, neighbor_color))
            if saturated and source_disagreement >= 100.0 and neighbor_disagreement >= 90.0:
                unexpected_saturated_color_frames += 1
                if source_disagreement > worst_unexpected_color_contrast:
                    worst_unexpected_color_contrast = source_disagreement
                    worst_unexpected_color_frame = frame
            blank, chromatic = localized_flash_scores(
                pixels[0], pixels[1], (neighbor_pixels[0],))
            if blank >= .55:
                localized_blank_frames += 1
                if blank > worst_localized_blank_fraction:
                    worst_localized_blank_fraction = blank
                    worst_localized_blank_frame = frame
            if chromatic >= .55:
                localized_color_frames += 1
                if chromatic > worst_localized_color_fraction:
                    worst_localized_color_fraction = chromatic
                    worst_localized_color_frame = frame
    reasons: list[str] = []
    if any(row["caption_pixels"] >= 8 and row["caption_p99"] > 24 for row in samples):
        reasons.append("authored_caption_changed")
    if any(row["cut_hold"] and (row["cut_mean"] > 6 or row["cut_p99"] > 32) for row in samples):
        reasons.append("authored_cut_changed")
    if coarse_caption_failed:
        reasons.append("unsampled_authored_caption_damage")
    if flatline_source_intervals:
        reasons.append("interpolated_motion_flatline")
    if unexpected_black_frames:
        reasons.append("unexpected_interpolated_black_frame")
    if unexpected_white_frames:
        reasons.append("unexpected_interpolated_white_frame")
    if unexpected_saturated_color_frames:
        reasons.append("unexpected_interpolated_saturated_color_frame")
    if localized_blank_frames:
        reasons.append("unexpected_interpolated_localized_blank_patch")
    if localized_color_frames:
        reasons.append("unexpected_interpolated_localized_color_patch")
    return {"status": "REJECT" if reasons else "PASS_AUTHORED_PROTECTION", "reasons": reasons,
            "adaptive_fingerprint": plan["fingerprint"], "target_rate": str(rate),
            "frames_decoded": decoded, "sample_count": len(samples),
            "caption_samples": sum(row["caption_pixels"] >= 8 for row in samples),
            "cut_samples": sum(row["cut_hold"] for row in samples), "samples": samples,
            "coarse_caption_screened_frames": coarse_caption_screened,
            "worst_coarse_caption_frame": worst_caption_frame,
            "worst_coarse_caption_mean_abs_rgb": round(worst_caption_mae, 6),
            "coarse_caption_policy": "unsampled_strong_source_glyph_cores_only_not_photographic_motion",
            "motion_intervals_screened": motion_intervals_screened,
            "flatline_source_intervals": flatline_source_intervals,
            "worst_flatline_source_interval": worst_flatline_source_interval,
            "motion_continuity_policy": "source_motion_8_to_64_and_two_generated_samples_without_motion; skip_cuts_and_static",
            "flash_screened_frames": flash_screened_frames,
            "boundary_flash_screened_frames": boundary_flash_screened_frames,
            "unexpected_black_frames": unexpected_black_frames,
            "unexpected_white_frames": unexpected_white_frames,
            "unexpected_saturated_color_frames": unexpected_saturated_color_frames,
            "localized_blank_frames": localized_blank_frames,
            "worst_localized_blank_frame": worst_localized_blank_frame,
            "worst_localized_blank_tile_coverage": round(worst_localized_blank_fraction, 6),
            "localized_blank_policy": "source_neighbor_bound_previews_extreme_rgb_1pct_and_tile_coverage_ge_55pct_skip_cuts_boundary_single_neighbor",
            "localized_color_frames": localized_color_frames,
            "worst_localized_color_frame": worst_localized_color_frame,
            "worst_localized_color_tile_coverage": round(worst_localized_color_fraction, 6),
            "localized_color_policy": "source_neighbor_bound_previews_saturated_rgb_1pct_and_tile_coverage_ge_55pct_skip_cuts_boundary_single_neighbor",
            "worst_unexpected_flash_frame": worst_unexpected_flash_frame,
            "worst_unexpected_color_frame": worst_unexpected_color_frame,
            "flash_protection_policy": "three_output_previews_plus_source_bound_first_last_neighbor_checks_skip_cuts",
            "seconds": round(time.monotonic() - started, 4),
            "scope": "sampled authored glyph/core and cut protection; not perceptual motion acceptance"}


def validate_quality_receipt(receipt: Any, plan: dict[str, Any], *, require_temporal: bool = False) -> None:
    if not isinstance(receipt, dict) or receipt.get("adaptive_fingerprint") != plan.get("fingerprint"):
        raise RuntimeError("Adaptive quality receipt is absent or bound to another plan.")
    status = receipt.get("status")
    if status == "SOURCE_PRESERVATION_FALLBACK":
        admitted = receipt.get("fallback", {})
        if receipt.get("actual_route") != "source_preserved" or receipt.get("rejected_candidate", {}).get("status") != "REJECT":
            raise RuntimeError("Adaptive fallback receipt is inconsistent.")
    elif status == "PASS_OBJECTIVE_GUARDS":
        admitted = receipt
        if receipt.get("actual_route") != plan.get("route"):
            raise RuntimeError("Adaptive actual route disagrees with its analyzed plan.")
    else:
        raise RuntimeError("Adaptive candidate lacks a passing quality gate.")
    if (admitted.get("status") != "PASS_OBJECTIVE_GUARDS" or admitted.get("reasons")
            or admitted.get("frames_decoded") != plan.get("frame_count")
            or type(admitted.get("sample_count")) is not int or admitted["sample_count"] < 1):
        raise RuntimeError("Adaptive candidate verification is incomplete.")
    if require_temporal:
        temporal = receipt.get("temporal_protection")
        if (not isinstance(temporal, dict) or temporal.get("status") != "PASS_AUTHORED_PROTECTION"
                or temporal.get("reasons") or temporal.get("adaptive_fingerprint") != plan.get("fingerprint")
                or type(temporal.get("frames_decoded")) is not int or temporal["frames_decoded"] < plan["frame_count"]
                or type(temporal.get("sample_count")) is not int or temporal["sample_count"] < 1):
            raise RuntimeError("Adaptive interpolation lacks its authored protection validation.")


def conventional_filter_graph(plan: dict[str, Any]) -> str:
    return f"[0:v:0]{_window_prefix(plan)}scale={int(plan['target_width'])}:{int(plan['target_height'])}:flags=lanczos,setsar=1[vout]"


def render_unit_plans(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Freeze bounded executable units under one verified analyzed source."""
    verify_artifacts(plan)
    units = plan.get("render_units")
    if plan.get("route") != "shot_sequence" or not isinstance(units, list) or not 2 <= len(units) <= 16:
        raise ValueError("Invalid adaptive shot-sequence plan.")
    current = 0
    rendered = []
    sw, sh = int(plan["source_width"]), int(plan["source_height"])
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("Malformed adaptive shot unit.")
        start, end = unit.get("start_frame"), unit.get("end_frame")
        if type(start) is not int or type(end) is not int or start != current or not start < end <= plan["frame_count"]:
            raise ValueError("Adaptive shots must cover the source exactly without gaps or overlaps.")
        crop = unit.get("crop")
        if not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int or v % 2 for v in crop):
            raise ValueError("Invalid adaptive unit crop.")
        x, y, w, h = crop
        if min(x, y) < 0 or min(w, h) < 2 or x + w > sw or y + h > sh:
            raise ValueError("Adaptive unit crop exceeds source bounds.")
        controls = unit.get("effective_controls")
        if not isinstance(controls, dict):
            raise ValueError("Adaptive unit is missing its measured recipe.")
        child = {**plan, "parent_adaptive_fingerprint": plan["fingerprint"],
                 "route": unit["route"], "crop": list(crop),
                 "effective_controls": dict(controls),
                 "effective_params": "" if plan.get("enhancement_model_choice") else "".join(f":{key}={value}" for key, value in controls.items()),
                 "execution_window": {"start_frame": start, "end_frame": end},
                 "render_units": []}
        # The same validated graph builder supplies both execution and tests.
        from adaptive_model_selection import execution_model
        enhancement_filter_graph(child, execution_model(child), "-2")
        child["fingerprint"] = plan_fingerprint(child)
        rendered.append(child)
        current = end
    if current != plan["frame_count"]:
        raise ValueError("Adaptive shot sequence ends before the source does.")
    return rendered


def _window_prefix(plan: dict[str, Any]) -> str:
    window = plan.get("execution_window")
    if window is None:
        return ""
    if not isinstance(window, dict) or set(window) != {"start_frame", "end_frame"}:
        raise ValueError("Invalid adaptive temporal execution window.")
    start, end = window["start_frame"], window["end_frame"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= MAX_FRAMES:
        raise ValueError("Invalid adaptive temporal frame boundaries.")
    rate = Fraction(plan["source_fps"])
    if not 0 < rate <= 120:
        raise ValueError("Invalid adaptive temporal cadence.")
    return (f"settb=expr=1/{rate.numerator},trim=start_pts={start * rate.denominator}:"
            f"end_pts={end * rate.denominator},setpts=PTS-STARTPTS,")


def enhancement_filter_graph(plan: dict[str, Any], model: str, device: str) -> str:
    if model not in {"prob-4", "ahq-12"} or device not in {"-2", "-1"}:
        raise ValueError("Adaptive natural requires an admitted local enhancement model.")
    if model == "ahq-12" and plan.get("effective_params") != "":
        raise ValueError("Artemis uses its own default controls, not Proteus controls.")
    # Generated controls are validated again at the executable boundary. No raw
    # filter grammar from a saved artifact is an executable authority.
    entries = str(plan["effective_params"]).split(":")[1:]
    allowed = {"details", "compression", "blur", "noise", "halo", "preblur", "blend"}
    seen = set()
    for entry in entries:
        pieces = entry.split("=")
        if len(pieces) != 2 or pieces[0] not in allowed or pieces[0] in seen:
            raise ValueError("Invalid adaptive executable controls.")
        seen.add(pieces[0])
        try:
            value = float(pieces[1])
        except ValueError as exc:
            raise ValueError("Invalid adaptive executable value.") from exc
        if not math.isfinite(value) or not (-1 <= value <= 1) or (pieces[0] == "blend" and value < 0):
            raise ValueError("Invalid adaptive executable bounds.")
    sizes = [plan[k] for k in ("source_width", "source_height", "target_width", "target_height")]
    if any(type(value) is not int or value < 2 or value % 2 for value in sizes):
        raise ValueError("Adaptive geometry must use positive even pixel dimensions.")
    sw, sh, tw, th = sizes
    if plan["route"] == "source_preserved":
        return conventional_filter_graph(plan)
    if plan["route"] not in {"masked_crop", "masked_full_frame"}:
        raise ValueError("Shot sequences must be executed through the bounded existing worker.")
    crop = plan["crop"]
    if not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int or v % 2 for v in crop):
        raise ValueError("Adaptive crop must contain four even integer coordinates.")
    x, y, w, h = crop
    if min(x, y) < 0 or min(w, h) < 2 or x + w > sw or y + h > sh:
        raise ValueError("Adaptive crop exceeds the source geometry.")
    if [x, y, w, h] == [0, 0, sw, sh]:
        cw, ch, dx, dy = tw, th, 0, 0
    else:
        scale = Fraction(tw, sw)
        if scale < 1 or scale != Fraction(th, sh):
            raise ValueError("Adaptive crop requires an isotropic, non-downscaling geometry.")
        mapped = [Fraction(v) * scale for v in (w, h, x, y)]
        if any(v.denominator != 1 or v.numerator % 2 for v in mapped):
            raise ValueError("Adaptive crop is not phase/chroma aligned to the final geometry.")
        cw, ch, dx, dy = [int(v) for v in mapped]
    return (
        f"[0:v:0]{_window_prefix(plan)}split=2[base_in][detail_in];"
        f"[base_in]scale={tw}:{th}:flags=lanczos,setsar=1,format=gbrp16le[base];"
        f"[detail_in]crop={w}:{h}:{x}:{y},tvai_up=model={model}:scale=0:w={cw}:h={ch}"
        f"{plan['effective_params']}:download=0:device={device}:vram=0.70:instances=0,"
        f"scale={cw}:{ch}:flags=lanczos,setsar=1,format=gbrp16le,"
        f"pad={tw}:{th}:{dx}:{dy}:color=black[detail];"
        f"[1:v:0]{_window_prefix(plan)}scale={tw}:{th}:flags=bilinear:in_range=full:out_range=full,format=gbrp16le[mask];"
        "[base][detail][mask]maskedmerge=planes=7[vout]"
    )


def verify_artifacts(plan: dict[str, Any]) -> None:
    if plan.get("schema") != SCHEMA or plan.get("fingerprint") != plan_fingerprint(plan):
        raise RuntimeError("Adaptive execution plan changed after analysis.")
    mask = Path(plan["mask_path"])
    if mask.is_symlink() or not mask.is_file() or _hash_file(mask) != plan["mask_sha256"]:
        raise RuntimeError("Adaptive source-protection mask is missing or changed.")


def interpolation_has_source_protection(plan: dict[str, Any]) -> bool:
    """Attach the existing protection stream for glyphs, uncertain glyphs or exact stills.

    Fallback-only captions previously had zero detected text lines and their
    source edge mask was silently omitted from native interpolation.
    """
    protection = plan.get("caption_protection")
    if not isinstance(protection, dict) or protection.get("stream_index") != 1:
        return False
    if protection.get("policy") not in {
            "authored_glyph_hold_v1", "authored_glyph_and_exact_rgb_hold_v2",
            "authored_glyph_and_exact_native_hold_v3"}:
        return False
    for name in ("detected_frames", "conservative_fallback_frames", "exact_repeat_pairs"):
        value = protection.get(name, 0)
        if type(value) is int and value > 0:
            return True
    return False


def temporal_caption_graph(plan: dict[str, Any], target_rate: str, width: int, height: int,
                           frame_count: int, cut_expression: str) -> str:
    """Complete native FI with source-bound authored glyph samples, not OCR.

    [native] is on the exact output clock and [cut_source] on the source clock.
    Input 1 is the existing two-stream job mask. Photographic motion outside
    admitted caption/context pixels remains genuine model interpolation.
    """
    verify_artifacts(plan)
    protection = plan.get("caption_protection") or {}
    if protection.get("stream_index") != 1 or protection.get("policy") not in {
            "authored_glyph_hold_v1", "authored_glyph_and_exact_rgb_hold_v2",
            "authored_glyph_and_exact_native_hold_v3"}:
        raise ValueError("Temporal caption protection requires its source-bound mask stream.")
    rate = Fraction(target_rate)
    if rate <= 0 or any(type(v) is not int or v <= 0 for v in (width, height, frame_count)):
        raise ValueError("Temporal caption output clock and geometry are invalid.")
    if cut_expression and any(c not in "between(n,0123456789)+" for c in cut_expression):
        raise ValueError("Temporal caption cut expression contains unsupported syntax.")
    clock = f"trim=end_frame={frame_count},settb=expr=1/{rate.numerator},setpts=N*{rate.denominator}"
    cut = f",drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{cut_expression}'" if cut_expression else ""
    return ("[native]format=gbrp16le[fi_rgb];"
            f"[cut_source]tpad=stop_mode=clone:stop=1,fps=fps={target_rate}:round=up,{clock},format=gbrp16le[authored_hold];"
            f"[1:v:1]tpad=stop_mode=clone:stop=1,fps=fps={target_rate}:round=up,{clock},"
            f"scale={width}:{height}:flags=bilinear:in_range=full:out_range=full{cut},format=gbrp16le[caption_mask];"
            f"[fi_rgb][authored_hold][caption_mask]maskedmerge=planes=7,trim=end_frame={frame_count}[vout]")
