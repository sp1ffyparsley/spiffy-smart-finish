"""Optional face-region guidance from *published* MotionCoach pose evidence.

This reader does not run another face detector, infer identity, download a
model or edit MotionCoach data. It delegates validation to MotionCoach's
canonical published pose-bundle reader and requires a byte-identical source
and complete source-frame coverage before allowing any face-only enhancement.
Pose landmarks are a spatial hint, never identity or face-recognition proof.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import math
from pathlib import Path
import sys
from typing import Any


SCHEMA = "spiffy_motioncoach_face_roi_v1"
ROOT = Path(__file__).resolve().parents[2]
MOTIONCOACH_SRC = ROOT / "Spiffy Workspace" / "MotionCoach" / "src"
REQUIRED_FACE_INDICES = (0, 2, 5, 9, 10)  # nose, two eyes, two mouth corners
OPTIONAL_EARS = (7, 8)
CONFIDENCE_FLOOR = 0.72
MAX_BUNDLE_ARRAY_BYTES = 32 * 1024 * 1024


def full_source_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("MotionCoach face evidence requires an ordinary local source file.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _face_box(landmarks: Any, visibility: Any, presence: Any,
              *, width: int, height: int) -> tuple[int, int, int, int]:
    """Conservative, square landmark-derived crop with head/hair margin."""
    import numpy as np

    xy = np.asarray(landmarks)
    vis, pres = np.asarray(visibility), np.asarray(presence)
    if xy.shape != (33, 3) or vis.shape != (33,) or pres.shape != (33,):
        raise ValueError("MotionCoach pose topology is not MediaPipe-33.")
    if not all(np.isfinite(xy[i, :2]).all() and
               min(float(vis[i]), float(pres[i])) >= CONFIDENCE_FLOOR
               for i in REQUIRED_FACE_INDICES):
        raise ValueError("Face landmark confidence or observability is insufficient.")
    eyes = xy[[2, 5], :2].astype(float)
    mouth = xy[[9, 10], :2].astype(float)
    nose = xy[0, :2].astype(float)
    if any(v < 0 or v > 1 for v in (*eyes.flatten(), *mouth.flatten(), *nose)):
        raise ValueError("Face landmarks leave the normalized image canvas.")
    eye_span = abs(float(eyes[0, 0] - eyes[1, 0]))
    eye_to_mouth = float(mouth[:, 1].mean() - eyes[:, 1].mean())
    if eye_span < 0.012 or eye_to_mouth < 0.012:
        raise ValueError("Face landmarks do not establish reliable eye/mouth geometry.")
    ear_x = [float(xy[i, 0]) for i in OPTIONAL_EARS
             if np.isfinite(xy[i, :2]).all()
             and min(float(vis[i]), float(pres[i])) >= CONFIDENCE_FLOOR]
    ear_span = (max(ear_x) - min(ear_x)) if len(ear_x) == 2 else 0.0
    face_width = max(eye_span * 2.75, ear_span * 1.55, eye_to_mouth * 2.7)
    face_height = max(face_width, eye_to_mouth * 3.4)
    # A tiny face is not safely recoverable; a huge crop is no longer a
    # facial ROI and should use the normal full-frame AI path instead.
    if face_width > 0.56 or face_height > 0.56:
        raise ValueError("Pose face ROI is too large to be selective.")
    center_x = float((eyes[:, 0].mean() + nose[0]) / 2)
    center_y = float(eyes[:, 1].mean() + eye_to_mouth * 0.28)
    side = max(face_width * width, face_height * height, 40.0)
    if side > min(width, height) * 0.60:
        raise ValueError("Pose face ROI exceeds selective enhancement bounds.")
    x0 = round(center_x * width - side / 2)
    y0 = round(center_y * height - side / 2)
    x0 = max(0, min(x0, width - 40))
    y0 = max(0, min(y0, height - 40))
    x1 = min(width, round(x0 + side))
    y1 = min(height, round(y0 + side))
    # Use divisible-by-2 source crop geometry for exact GPU output mapping.
    x1 -= (x1 - x0) % 2
    y1 -= (y1 - y0) % 2
    if min(x1 - x0, y1 - y0) < 32:
        raise ValueError("Face crop has too few supported source pixels.")
    return int(x0), int(y0), int(x1), int(y1)


def load_published_face_plan(
    source: Path, pose_root: Path, *,
    source_binding_id: str, info: dict[str, Any],
) -> dict[str, Any]:
    """Return one source-bound ROI per frame or fail without partial render."""
    if type(source_binding_id) is not str or not 1 <= len(source_binding_id) <= 128:
        raise ValueError("Explicit MotionCoach source-binding ID is required.")
    arrays = Path(pose_root) / "pose_arrays.npz"
    if not arrays.is_file() or arrays.is_symlink() or arrays.stat().st_size > MAX_BUNDLE_ARRAY_BYTES:
        raise ValueError("MotionCoach canonical pose arrays are unavailable or unexpectedly large.")
    if str(MOTIONCOACH_SRC) not in sys.path:
        sys.path.insert(0, str(MOTIONCOACH_SRC))
    from motioncoach.pose import load_pose_playback_bundle

    digest = full_source_sha256(Path(source))
    seq = load_pose_playback_bundle(
        pose_root, expected_source_binding_id=source_binding_id,
        expected_source_sha256=digest,
        source_duration_seconds=float(info["duration"]),
    )
    width, height = int(info["width"]), int(info["height"])
    count = int(info["video_pts_count"])
    if (seq.image_width, seq.image_height, seq.rotation_degrees) != (width, height, 0):
        raise ValueError("MotionCoach pose and current video geometry/rotation differ.")
    if seq.frame_count != count or count < 2 or count > 1800:
        raise ValueError("Selective face enhancement requires one published pose for every source frame.")
    rate = Fraction(str(info["fps_ratio"]))
    if any(int(seq.frame_indices[i]) != i or not bool(seq.valid[i])
           or abs(float(seq.timestamps[i]) - float(Fraction(i, 1) / rate)) > 0.025
           for i in range(count)):
        raise ValueError("MotionCoach pose timeline must match every decoded source-frame index and clock.")
    boxes: list[tuple[int, int, int, int]] = []
    frame_tolerance = min(0.005, 0.20 / float(rate))
    for i in range(count):
        if abs(float(seq.timestamps[i]) - float(Fraction(i, 1) / rate)) > frame_tolerance:
            raise ValueError("MotionCoach published face sample left the exact source-frame timeline.")
        box = _face_box(seq.image_landmarks[i], seq.visibility[i], seq.presence[i],
                        width=width, height=height)
        if boxes:
            prev = boxes[-1]
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            old = ((prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2)
            if math.hypot((center[0] - old[0]) / width,
                          (center[1] - old[1]) / height) > 0.16:
                raise ValueError("MotionCoach face ROI jumps too far between adjacent frames.")
            if any(
                not (0.65 <= current / previous <= 1.55)
                for current, previous in (
                    (box[2] - box[0], prev[2] - prev[0]),
                    (box[3] - box[1], prev[3] - prev[1]),
                )
            ):
                raise ValueError("MotionCoach face ROI scale jumps unexpectedly between frames.")
        boxes.append(box)
    return {
        "schema": SCHEMA, "source_sha256": digest, "source_binding_id": source_binding_id,
        "pose_source": "MotionCoach.canonical_published_pose_playback",
        "frame_count": count, "width": width, "height": height,
        "boxes_xyxy": boxes,
        "identification_claim": "none_pose_landmarks_are_spatial_evidence_only",
        "policy": "explicit_opt_in_every_frame_accepted_no_interpolation",
    }
