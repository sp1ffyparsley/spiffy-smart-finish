"""Smart Finish consumer of the incumbent semantic-roto owner.

No new detector, model store or inference runtime. A bounded optional subject
mask can ONLY reduce natural enhancement contribution. Ambiguous subjects,
busy hardware and unavailable backends keep the conservative natural route.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from delivery_insights import resource_snapshot

SCHEMA = "spiffy_adaptive_subject_bridge_v1"
MAX_SUBJECT_WINDOWS = 2
MAX_TOTAL_SUBJECT_SECONDS = 120.0


def _subject_cache_root() -> Path:
    from workflow.workflow_paths import local_cache_root

    return local_cache_root() / "semantic_roto"


def _reference_contract(source_identity: dict[str, Any], row: dict[str, Any],
                        rate: Fraction, width: int, height: int, runtime: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "Spiffy Workspace/workflow/semantic_roto.py",
             root / "Spiffy Workspace/workflow/bounded_specialist_runtime.py",
             root / "Spiffy Workspace/workflow/sparse_media_specialists.py",
             root / "Spiffy Workspace/workflow/thumbnail_mask_generator.py"]
    owners = {}
    for path in paths:
        with path.open("rb") as handle:
            owners[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"schema": "spiffy_semantic_roto_consumer_reference_v1",
            "source_sha256": source_identity["digest"], "source_bytes": source_identity.get("file_size"),
            "frames": [row["start_frame"], row["end_frame"]], "rate": str(rate),
            "source_size": [width, height], "runtime": runtime, "owners": owners}


def _reference_path(contract: dict[str, Any]) -> Path:
    key = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return _subject_cache_root() / "_smart_finish_refs" / f"{key}.json"


def remember_subject_reference(source_identity: dict[str, Any], row: dict[str, Any],
                               rate: Fraction, width: int, height: int) -> None:
    """Point to the incumbent immutable mask artifact; never duplicate its PNGs."""
    from workflow.semantic_roto import validate_semantic_roto_manifest
    from shared.progress_writer import atomic_write_json

    runtime = runtime_identity()
    if row.get("status") != "ready" or row.get("runtime") != runtime:
        raise ValueError("Only current accepted canonical subjects can be referenced.")
    root = _subject_cache_root().resolve()
    manifest_path = Path(row["manifest_path"])
    if manifest_path.is_symlink():
        raise ValueError("Canonical subject manifest cannot be a link.")
    relative = manifest_path.resolve().relative_to(root)
    manifest = validate_semantic_roto_manifest(manifest_path, expected_source_identity=source_identity)
    if manifest["manifest_sha256"] != row["manifest_sha256"]:
        raise ValueError("Accepted canonical subject manifest changed.")
    bind_mask_frames(manifest, manifest_path, row["start_frame"], row["end_frame"], rate)
    contract = _reference_contract(source_identity, row, rate, width, height, runtime)
    path = _reference_path(contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("Subject-reference owner must be an ordinary directory/file.")
    packet = {"contract": contract, "manifest_relative": relative.as_posix(),
              "manifest_sha256": row["manifest_sha256"], "geometry": row.get("geometry"),
              "confidence": row.get("confidence")}
    packet["sha256"] = hashlib.sha256(json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    atomic_write_json(path, packet)


def _reuse_subject_reference(source_identity: dict[str, Any], row: dict[str, Any],
                             rate: Fraction, width: int, height: int, runtime: dict[str, Any]) -> tuple[dict[str, Any], dict[int, dict[str, str]]] | None:
    from workflow.semantic_roto import validate_semantic_roto_manifest

    contract = _reference_contract(source_identity, row, rate, width, height, runtime)
    path = _reference_path(contract)
    if not path.is_file() or path.is_symlink() or path.parent.is_symlink() or path.stat().st_size > 65536:
        return None
    try:
        packet = json.loads(path.read_text(encoding="utf8"))
        digest = packet.pop("sha256", None)
        if digest != hashlib.sha256(json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest():
            return None
        if packet["contract"] != contract:
            return None
        root = _subject_cache_root().resolve()
        manifest_path = root / packet["manifest_relative"]
        manifest_path.resolve().relative_to(root)
        if manifest_path.is_symlink():
            return None
        manifest = validate_semantic_roto_manifest(manifest_path, expected_source_identity=source_identity)
        if manifest["manifest_sha256"] != packet["manifest_sha256"]:
            return None
        frames = bind_mask_frames(manifest, manifest_path, row["start_frame"], row["end_frame"], rate)
        return ({"status": "ready", "reason": "canonical_subject_masks_accepted",
                 "manifest_path": str(manifest_path), "manifest_sha256": manifest["manifest_sha256"],
                 "runtime": runtime, "geometry": packet.get("geometry"), "confidence": packet.get("confidence"),
                 "cache_state": "validated_reference_hit"}, frames)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return None


def _cancel(writer: Any) -> None:
    if getattr(writer, "cancel_requested", False):
        from shared.gpu_lock import GpuLockCancelled

        raise GpuLockCancelled("Cancelled during source-bound subject refinement.")


def runtime_identity() -> dict[str, Any]:
    from workflow.bounded_specialist_runtime import bounded_specialist_status

    status = bounded_specialist_status(usage_profile="commercial")
    result = {}
    for task in ("semantic_roto_sam2", "semantic_roto_person_localizer"):
        value = status.get("models", {}).get(task, {})
        result[task] = {
            "ready": value.get("ready") is True,
            "assets": [
                {
                    "sha256": row.get("sha256") or row.get("expected_sha256"),
                    "bytes": row.get("bytes"),
                }
                for row in value.get("assets", [])
            ],
        }
    return result


def plan_subject_windows(
    windows: list[tuple[int, int]], rate: Fraction, width: int, height: int
) -> list[dict[str, Any]]:
    """Inspect every shot; only admitted complete shots receive new inference.

    Do not join different shots for SAM2 or increase its canonical four-second /
    120-frame bound. Long shots stay natural until full chunk-continuity is proven.
    """
    from workflow.semantic_roto import MAX_AUTO_ROTO_DURATION_MS, MAX_AUTO_ROTO_FRAMES

    result = []
    for start, end in windows:
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end
            or rate <= 0
        ):
            raise ValueError(
                "Subject windows require valid frame ordinals and cadence."
            )
        start_ms = round(Fraction(start) * 1000 / rate)
        end_ms = round(Fraction(end - 1) * 1000 / rate)
        reason = (
            "canonical_window_budget"
            if end - start > MAX_AUTO_ROTO_FRAMES
            or end_ms - start_ms > MAX_AUTO_ROTO_DURATION_MS
            else ""
        )
        if end - start < 8 or max(width, height) < 640:
            reason = "small_window_preserve_natural"
        result.append(
            {
                "start_frame": start,
                "end_frame": end,
                "source_start_ms": start_ms,
                "source_end_ms": end_ms,
                "status": "deferred" if reason else "eligible",
                "reason": reason,
            }
        )
    return result


def resource_admission(
    snapshot: dict[str, Any], width: int, height: int, frames: int
) -> dict[str, Any]:
    # The existing worker explicitly requires CUDA. Never label an untested MPS
    # substitution qualified; the CPU natural masks remain cross-platform.
    if snapshot.get("platform") not in {"win32", "linux"}:
        return {
            "allowed": False,
            "reason": "canonical_roto_backend_not_qualified_on_this_platform",
            "snapshot": snapshot,
        }
    required_ram = 4 * 1024**3 + width * height * frames * 12
    available = snapshot.get("ram_available_bytes")
    free = snapshot.get("gpu_memory_free_mib")
    busy = snapshot.get("gpu_utilization_percent")
    reason = ""
    if not isinstance(available, int) or available < required_ram:
        reason = "system_memory_headroom"
    elif not isinstance(free, (float, int)) or free < 4608:
        reason = "gpu_memory_headroom"
    elif not isinstance(busy, (float, int)) or busy > 40:
        reason = "gpu_currently_busy"
    return {
        "allowed": not reason,
        "reason": reason or "bounded_headroom_available",
        "required_ram_bytes": required_ram,
        "snapshot": snapshot,
    }


def choose_subject(detections: Any) -> dict[str, Any] | None:
    if not isinstance(detections, list):
        return None
    rows = []
    for row in detections:
        try:
            box = row["bbox_norm"]
            score = float(row["score"])
            if (
                len(box) != 4
                or any(isinstance(v, bool) or not math.isfinite(float(v)) for v in box)
                or not math.isfinite(score)
                or not 0.5 <= score <= 1
            ):
                continue
            x, y, r, b = [float(v) for v in box]
            if not (0 <= x < r <= 1 and 0 <= y < b <= 1) or (r - x) * (b - y) < 0.02:
                continue
            rows.append(
                {
                    "bbox": {
                        "x": x,
                        "y": y,
                        "width": r - x,
                        "height": b - y,
                        "coordinate_space": "normalized",
                    },
                    "confidence": score,
                    "capability": "semantic_subject",
                    "source": "workflow.bounded_specialist_runtime.semantic_roto_person_localizer",
                }
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    rows.sort(key=lambda row: -row["confidence"])
    if not rows:
        return None
    if len(rows) > 1 and rows[1]["confidence"] >= rows[0]["confidence"] * 0.85:
        return None
    return rows[0]


def bind_mask_frames(
    manifest: dict[str, Any], manifest_path: Path, start: int, end: int, rate: Fraction
) -> dict[int, dict[str, str]]:
    mapping = {}
    root = manifest_path.parent.resolve()
    for row in manifest.get("frames") or []:
        try:
            ordinal = (
                Fraction(row["pts"])
                * Fraction(row["time_base_num"], row["time_base_den"])
                * rate
            )
            if (
                ordinal.denominator != 1
                or not start <= ordinal < end
                or int(ordinal) in mapping
            ):
                raise ValueError("mask timestamp outside exact shot")
            path = (root / row["mask_file"]).resolve()
            path.relative_to(root)
            mapping[int(ordinal)] = {
                "path": str(path),
                "sha256": str(row["mask_sha256"]),
            }
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise RuntimeError(
                "Semantic mask timestamp/coverage is inconsistent with its shot."
            ) from exc
    if set(mapping) != set(range(start, end)):
        raise RuntimeError("Semantic mask coverage does not contain every shot frame.")
    return mapping


def restrict_to_subject(contribution: Any, alpha: Any) -> Any:
    import cv2
    import numpy as np

    if (
        not isinstance(alpha, np.ndarray)
        or alpha.dtype != np.uint8
        or alpha.ndim != 2
        or alpha.shape != contribution.shape
        or contribution.dtype != np.uint8
    ):
        raise RuntimeError("Semantic mask dimensions/type do not match source pixels.")
    # Preserve source edges/motion blur. Binary segmentation is NOT promoted to
    # physical fractional-opacity matting by feathering it.
    inside = (alpha >= 128).astype(np.uint8)
    distance = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    radius = max(4.0, min(alpha.shape) * 0.006)
    weight = np.clip((distance - 1.0) / radius, 0, 1)
    return np.minimum(
        contribution, np.rint(contribution.astype(np.float32) * weight).astype(np.uint8)
    )


def load_subject_frame(row: dict[str, str], shape: tuple[int, int]) -> Any:
    import cv2
    import numpy as np

    path = Path(row["path"])
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("A bound subject mask disappeared or became a link.")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise RuntimeError("Bound semantic mask changed during rendering.")
    mask = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape:
        raise RuntimeError("Bound semantic mask has incorrect source geometry.")
    return mask


def prepare_subjects(
    source: Path,
    info: dict[str, Any],
    windows: list[tuple[int, int]],
    rate: Fraction,
    writer: Any,
    policy: dict[str, Any],
    *,
    shot_evidence: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, str]]]:
    from workflow.semantic_roto import (
        build_semantic_roto,
        validate_semantic_roto_manifest,
    )
    from workflow.bounded_specialist_runtime import run_bounded_specialist
    from shared.gpu_lock import SharedGpuLock, GpuLockCancelled

    started = time.monotonic()
    width, height = int(info["width"]), int(info["height"])
    rows = plan_subject_windows(windows, rate, width, height)
    if shot_evidence is not None:
        if len(shot_evidence) != len(rows) or any(
            not isinstance(evidence, dict)
            or evidence.get("start_frame") != row["start_frame"]
            or evidence.get("end_frame") != row["end_frame"]
            or evidence.get("route") not in {"source_preserved", "masked_crop", "masked_full_frame"}
            for row, evidence in zip(rows, shot_evidence)
        ):
            raise ValueError("Subject shot evidence must match the complete analyzed shot windows.")
        for row, evidence in zip(rows, shot_evidence):
            if evidence["route"] == "source_preserved":
                row.update(status="deferred", reason="no_enhancement_contribution_in_shot")
            elif (isinstance(evidence.get("visual_treatment"), dict)
                  and evidence["visual_treatment"].get("schema") == "spiffy_visual_treatment_v1"
                  and evidence["visual_treatment"].get("mode") == "graphics_edge_guard"):
                # Measured high-contrast graphic/title shapes are already
                # protected by source-edge/caption masks. Avoid a speculative
                # expensive person-localizer + SAM2 startup with no reliable
                # photographic subject to refine. No claim of true graphics
                # semantics or visual-object identity follows from this.
                row.update(status="deferred", reason="measured_graphics_preserve_source_edges")
    summary = {
        "schema": SCHEMA,
        "owner": "workflow.semantic_roto",
        "mode": policy.get("subject_refinement", "auto"),
        "shots": rows,
        "seconds": 0.0,
        "successful_shots": 0,
        "attempted_shots": 0,
        "cached_shots": 0,
        "alpha_kind": "binary_segmentation_with_source_edge_preservation_not_soft_matting",
    }
    mapping: dict[int, dict[str, str]] = {}
    source_identity = None
    runtime_stop = ""
    lookup_runtime = None
    for row in rows:
        _cancel(writer)
        if summary["mode"] == "off":
            row.update(status="deferred", reason="explicitly_disabled")
        if row["status"] != "eligible":
            continue
        lookup_start = time.monotonic()
        if lookup_runtime is None:
            lookup_runtime = runtime_identity()
        # An exact known artifact can be revalidated without a CUDA context or
        # idle GPU. Full source identity and each PNG still have to match.
        if (_subject_cache_root() / "_smart_finish_refs").is_dir():
            if source_identity is None:
                from workflow.media_identity import full_media_identity
                source_identity = full_media_identity(source, check_cancel=lambda: _cancel(writer))
            reused = _reuse_subject_reference(source_identity, row, rate, width, height, lookup_runtime)
            if reused is not None:
                cached_row, cached_frames = reused
                if set(mapping) & set(cached_frames):
                    raise RuntimeError("Cached subject masks overlap another shot.")
                mapping.update(cached_frames)
                row.update(cached_row, seconds=round(time.monotonic() - lookup_start, 4))
                summary["successful_shots"] += 1
                summary["cached_shots"] += 1
                continue
        if runtime_stop:
            row.update(status="deferred", reason=runtime_stop)
            continue
        if (
            summary["attempted_shots"] >= MAX_SUBJECT_WINDOWS
            or time.monotonic() - started > MAX_TOTAL_SUBJECT_SECONDS
        ):
            row.update(status="deferred", reason="bounded_optional_refinement_budget")
            continue
        admission = resource_admission(
            resource_snapshot(), width, height, row["end_frame"] - row["start_frame"]
        )
        row["admission"] = admission
        if not admission["allowed"]:
            row.update(status="deferred", reason=admission["reason"])
            continue
        stage_start = time.monotonic()
        acquired = False
        try:
            with SharedGpuLock(
                Path(writer.job_dir).parent,
                writer,
                "gpu_render",
                "Smart Finish subject refinement",
                timeout_s=2,
            ):
                acquired = True
                identity_before = runtime_identity()
                if any(
                    isinstance(v, dict) and not v.get("ready")
                    for v in identity_before.values()
                ):
                    row.update(
                        status="deferred",
                        reason="canonical_subject_runtime_unavailable",
                        runtime=identity_before,
                    )
                    continue
                if source_identity is None:
                    from workflow.media_identity import full_media_identity

                    source_identity = full_media_identity(
                        source, check_cancel=lambda: _cancel(writer)
                    )
                summary["attempted_shots"] += 1
                writer.update(
                    phase="adaptive_subject_localize",
                    message=f"Locating subject in shot {rows.index(row) + 1}",
                )
                remaining = MAX_TOTAL_SUBJECT_SECONDS - (time.monotonic() - started)
                if remaining < 10:
                    row.update(status="deferred", reason="bounded_optional_refinement_budget")
                    continue
                localized = run_bounded_specialist(
                    "semantic_roto_person_localizer",
                    {
                        "source_path": str(source),
                        "source_start_ms": row["source_start_ms"],
                        "source_end_ms": row["source_end_ms"],
                    },
                    timeout_s=min(45, max(5, int(remaining))),
                    usage_profile="commercial",
                    check_cancel=lambda: _cancel(writer),
                )
                _cancel(writer)
                geometry = (
                    choose_subject(localized.get("evidence", {}).get("detections", []))
                    if localized.get("status") == "done"
                    else None
                )
                if geometry is None:
                    if localized.get("status") in {"timeout", "failed", "unavailable"}:
                        # A cold/runtime failure is not a subject-specific miss.
                        # Do not pay the same failing optional startup per shot.
                        runtime_stop = "prior_optional_runtime_" + str(localized["status"])
                    row.update(
                        status="deferred",
                        reason="no_unambiguous_subject" if localized.get("status") == "done" else
                               "subject_localizer_" + str(localized.get("status") or "unavailable"),
                        localizer_status=localized.get("status"),
                    )
                    continue
                writer.update(
                    phase="adaptive_subject_roto",
                    message=f"Tracking source-bound subject in shot {rows.index(row) + 1}",
                )
                remaining = MAX_TOTAL_SUBJECT_SECONDS - (time.monotonic() - started)
                if remaining < 10:
                    row.update(status="deferred", reason="bounded_optional_refinement_budget")
                    continue
                result = build_semantic_roto(
                    source_path=source,
                    source_start_ms=row["source_start_ms"],
                    source_end_ms=row["source_end_ms"],
                    geometry=geometry,
                    source_identity=source_identity,
                    # Reuse the source/model-bound canonical cache shared with
                    # other editor consumers, not a duplicated per-export store.
                    cache_root=None,
                    check_cancel=lambda: _cancel(writer),
                    timeout_s=min(60, max(5, int(remaining))),
                    max_frames=row["end_frame"] - row["start_frame"],
                )
                _cancel(writer)
                if result.get("status") == "invalid":
                    raise RuntimeError(
                        "Canonical subject evidence failed integrity: "
                        + str(result.get("reason_code"))
                    )
                if result.get("status") != "ready":
                    row.update(
                        status="deferred",
                        reason=result.get("reason_code", "canonical_subject_not_ready"),
                    )
                    continue
                path = Path(result["manifest_path"])
                manifest = validate_semantic_roto_manifest(
                    path, expected_source_identity=source_identity
                )
                current = bind_mask_frames(
                    manifest, path, row["start_frame"], row["end_frame"], rate
                )
                if runtime_identity() != identity_before:
                    raise RuntimeError(
                        "Subject runtime/assets changed during the admitted refinement."
                    )
                if set(current) & set(mapping):
                    raise RuntimeError(
                        "Subject masks overlap another shot's frame ownership."
                    )
                mapping.update(current)
                row.update(
                    status="ready",
                    reason="canonical_subject_masks_accepted",
                    manifest_path=str(path),
                    manifest_sha256=manifest["manifest_sha256"],
                    runtime=identity_before,
                    geometry=geometry,
                    confidence=result.get("confidence"),
                    cache_state=result.get("cache_state"),
                )
                summary["successful_shots"] += 1
                # This is a small lookup reference in the existing semantic
                # cache. It carries no duplicate masks or negative timeout cache.
                remember_subject_reference(source_identity, row, rate, width, height)
        except GpuLockCancelled:
            raise
        except RuntimeError:
            if acquired:
                raise
            row.update(status="deferred", reason="shared_gpu_lane_busy")
        finally:
            row["seconds"] = round(time.monotonic() - stage_start, 4)
    summary["seconds"] = round(time.monotonic() - started, 4)
    return summary, mapping


def verify_cached_subjects(summary: Any) -> None:
    if not isinstance(summary, dict):
        raise RuntimeError("Cached subject decision evidence is missing.")
    ready = [r for r in summary.get("shots", []) if r.get("status") == "ready"]
    if not ready:
        return
    from workflow.semantic_roto import validate_semantic_roto_manifest

    identity = runtime_identity()
    for row in ready:
        if row.get("runtime") != identity:
            raise RuntimeError("Cached subject model identity changed.")
        manifest = validate_semantic_roto_manifest(row["manifest_path"])
        if manifest["manifest_sha256"] != row["manifest_sha256"]:
            raise RuntimeError(
                "Cached subject manifest no longer matches the analyzed plan."
            )
