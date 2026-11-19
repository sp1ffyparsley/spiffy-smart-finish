"""Real MotionCoach published-pose binding and selective face compositing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "tools" / "video_upscale"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from motioncoach_face_bridge import (  # noqa: E402
    _face_box, full_source_sha256, load_published_face_plan,
)
from selective_face_enhancement import (  # noqa: E402
    prepare_face_crop, compose_face_roi,
)
from comfyui_video_backend import _validate_source  # noqa: E402
from comfyui_model_catalog import get_model  # noqa: E402


def _face_landmarks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.full((33, 3), (0.5, 0.5, 0.0), dtype=np.float32)
    for i, x, y in ((0, .50, .31), (2, .45, .28), (5, .55, .28),
                    (7, .40, .30), (8, .60, .30), (9, .47, .38),
                    (10, .53, .38)):
        coords[i, :2] = (x, y)
    return coords, np.ones(33, dtype=np.float32), np.ones(33, dtype=np.float32)


def _published_pose(tmp_path: Path, source: Path, *, count: int = 8) -> Path:
    """Write a published-shape *internal synthetic* MotionCoach pose fixture."""
    root = tmp_path / "published_pose"
    root.mkdir()
    landmarks, vis, pres = _face_landmarks()
    timeline = np.arange(count, dtype=np.float64) / 12.0
    np.savez_compressed(
        root / "pose_arrays.npz",
        timestamps=timeline, source_pts=np.arange(count, dtype=np.int64),
        frame_indices=np.arange(count, dtype=np.int64),
        image_landmarks=np.repeat(landmarks[None], count, axis=0),
        world_landmarks=np.zeros((count, 33, 3), dtype=np.float32),
        visibility=np.repeat(vis[None], count, axis=0),
        presence=np.repeat(pres[None], count, axis=0),
        valid=np.ones(count, dtype=np.uint8),
    )
    payload = {
        "schema": "motioncoach.pose-evidence.v1",
        "source_binding_id": "source.test_face",
        "span": {"timebase_id": "timebase.test", "start": 0.0,
                 "end": float(timeline[-1]), "source_binding_id": "source.test_face"},
        "coordinate_frame_id": "coordinate.test",
        "array_artifacts": [{"role": "pose_arrays",
                             "locator": {"kind": "artifact_ref", "value": "pose_arrays.npz"}}],
        "provenance": {"source_refs": [{"source_binding_id": "source.test_face",
                                       "sha256": full_source_sha256(source), "role": "source_video"}]},
        "extensions": {"image_width": 128, "image_height": 72, "rotation_degrees": 0},
    }
    (root / "pose_evidence.json").write_text(json.dumps(payload), encoding="utf8")
    (root / "model_execution.json").write_text(json.dumps({"status": "succeeded"}), encoding="utf8")
    (root / "topology.json").write_text(json.dumps({"landmark_count": 33}), encoding="utf8")
    integrity = {}
    for name in ("pose_arrays.npz", "pose_evidence.json", "model_execution.json", "topology.json"):
        file = root / name
        data = file.read_bytes()
        integrity[name] = {"sha256": hashlib.sha256(data).hexdigest(), "byte_length": len(data)}
    (root / "bundle_integrity.json").write_text(json.dumps(integrity), encoding="utf8")
    return root


def _probe_info() -> dict:
    return {"width": 128, "height": 72, "duration": 8 / 12,
            "fps": 12, "fps_ratio": "12/1",
            "video_pts_count": 8, "rotation": 0, "sar": "1:1",
            "frame_rate_evidence": "presentation-timestamps", "video_pts_uniform": True,
            "video_pts_start": "0", "video_pts_average_interval": "1/12",
            "video_pts_end_exclusive": "2/3", "video_pts_time_base": "1/12000"}


def test_face_box_requires_confident_mouth_eyes_nose_and_admitted_area():
    points, visible, present = _face_landmarks()
    rect = _face_box(points, visible, present, width=128, height=72)
    assert len(rect) == 4
    assert rect[2] > rect[0] and rect[3] > rect[1]
    visible[2] = .1
    with pytest.raises(ValueError, match="confidence"):
        _face_box(points, visible, present, width=128, height=72)
    visible[2] = 1
    points[5, 0] = points[2, 0]
    with pytest.raises(ValueError, match="eye/mouth"):
        _face_box(points, visible, present, width=128, height=72)


def test_motioncoach_published_pose_is_read_through_original_loader(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic-internal-video-identity")
    bundle = _published_pose(tmp_path, source)
    plan = load_published_face_plan(
        source, bundle, source_binding_id="source.test_face", info=_probe_info())
    assert plan["frame_count"] == 8
    assert plan["source_sha256"] == full_source_sha256(source)
    assert len(plan["boxes_xyxy"]) == 8
    assert all(rect == plan["boxes_xyxy"][0] for rect in plan["boxes_xyxy"])
    assert plan["identification_claim"].startswith("none")


def test_motioncoach_rejects_different_source_binding_stale_or_corrupt_pose(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"first-source-identity")
    bundle = _published_pose(tmp_path, source)
    with pytest.raises(Exception, match="binding"):
        load_published_face_plan(source, bundle,
                                 source_binding_id="source.wrong", info=_probe_info())
    source.write_bytes(b"mutated-source-identity")
    with pytest.raises(Exception, match="digest"):
        load_published_face_plan(source, bundle,
                                 source_binding_id="source.test_face", info=_probe_info())
    source.write_bytes(b"first-source-identity")
    (bundle / "pose_arrays.npz").write_bytes(b"tampered-array")
    with pytest.raises(Exception, match="integrity"):
        load_published_face_plan(source, bundle,
                                 source_binding_id="source.test_face", info=_probe_info())


def test_source_bound_face_patch_enhances_only_requested_region(tmp_path):
    source = tmp_path / "original.png"
    crop = tmp_path / "face_input.png"
    enhanced = tmp_path / "face_neural.png"
    final = tmp_path / "rendered.png"
    Image.new("RGB", (128, 72), (70, 90, 110)).save(source)
    bbox = _face_box(*_face_landmarks(), width=128, height=72)
    rectangle = prepare_face_crop(
        source, bbox, source_size=(128, 72),
        output_size=(512, 288), crop_path=crop)
    assert crop.is_file()
    Image.new("RGB", (rectangle[2] - rectangle[0], rectangle[3] - rectangle[1]),
              (95, 110, 105)).save(enhanced)
    result = compose_face_roi(
        source, enhanced, final, rectangle, output_size=(512, 288))
    assert result["foreground"] == "pinned_comfyui_neural_face_crop"
    with Image.open(final) as image:
        assert image.size == (512, 288)
        assert image.getpixel((0, 0)) == (70, 90, 110)
        cx = (rectangle[0] + rectangle[2]) // 2
        cy = (rectangle[1] + rectangle[3]) // 2
        assert image.getpixel((cx, cy))[0] > 85
    with pytest.raises(FileExistsError):
        compose_face_roi(source, enhanced, final, rectangle, output_size=(512, 288))
    abnormal = tmp_path / "abnormal.png"
    Image.new("RGB", (rectangle[2] - rectangle[0], rectangle[3] - rectangle[1]),
              (235, 30, 30)).save(abnormal)
    with pytest.raises(RuntimeError, match="local color"):
        compose_face_roi(source, abnormal, tmp_path / "bad.png",
                         rectangle, output_size=(512, 288))


def test_roi_mode_can_admit_source_native_tensor_that_full_model_must_reject():
    info = {**_probe_info(), "width": 3840, "height": 2160,
            "video_pts_count": 8}
    model = get_model("realesrgan_x4plus")
    with pytest.raises(ValueError, match="native tensor"):
        _validate_source(info, 5760, model, region_mode="full")
    assert _validate_source(info, 5760, model, region_mode="motioncoach_face") == (
        5760, 3240, "12/1")


def test_private_comfyui_context_does_not_mask_downstream_file_error(tmp_path, monkeypatch):
    import comfyui_video_backend as backend

    comfy = tmp_path / "comfy"
    exe = comfy / ".venv" / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"fake-internal-test-runtime")
    (comfy / "main.py").write_text("# internal fixture", encoding="utf8")
    lifecycle = {"terminated": 0}

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            lifecycle["terminated"] += 1

        def wait(self, timeout=None):
            return 0

        def kill(self):
            lifecycle["terminated"] += 1

    monkeypatch.setattr(backend.subprocess, "Popen", lambda *args, **kw: FakeProcess())

    def api(_base, suffix, **kwargs):
        if suffix == "/system_stats":
            return {"system": {"comfyui_version": "test"}}
        if suffix == "/object_info/UpscaleModelLoader":
            return {"UpscaleModelLoader": {
                "input": {"required": {"model_name": [["RealESRGAN_x4plus.pth"]]}}}}
        raise AssertionError("Unexpected local ComfyUI API endpoint")

    monkeypatch.setattr(backend, "_json_http", api)
    with pytest.raises(OSError, match="real-face-frame-read-error"):
        with backend._owned_comfyui(comfy, tmp_path / "isolated", timeout=2):
            raise OSError("real-face-frame-read-error")
    assert lifecycle["terminated"] == 1


def test_real_video_delivery_with_motioncoach_roi_and_mocked_inference(tmp_path, monkeypatch):
    """Full FFmpeg/MP4/audio pipeline; synthetic neural crop, no GPU requirement."""
    from contextlib import contextmanager
    from shared.ffmpeg_utils import standard_ffmpeg, standard_ffprobe
    from shared.media_probe import probe_video
    from comfyui_video_backend import run_comfyui_delivery
    import comfyui_video_backend as backend
    import delivery_insights
    import shared.gpu_lock as gpu_lock

    ffmpeg = standard_ffmpeg()
    source = tmp_path / "source.mp4"
    backend._run([
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=128x72:rate=12",
        "-f", "lavfi", "-i", "sine=frequency=480:sample_rate=48000",
        "-t", "0.6666667", "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-b:a", "192k", "-shortest", str(source),
    ], timeout=30)
    evidence = _published_pose(tmp_path, source)
    input_info = probe_video(source, standard_ffprobe(), inspect_timestamps=True)
    assert input_info["video_pts_count"] == 8
    assert load_published_face_plan(
        source, evidence, source_binding_id="source.test_face",
        info=input_info)["frame_count"] == 8

    @contextmanager
    def no_gpu_owner(*args, **kwargs):
        yield object()

    @contextmanager
    def no_server(_root, work, **kwargs):
        (work / "output").mkdir(parents=True, exist_ok=True)
        yield "mocked-local-only-ComfyUI"

    calls: list[tuple[int, int]] = []

    def mocked_patch(base, cropped, *, output_root, output_width,
                     output_height, prefix, timeout, cancel, model_name):
        assert cropped.name.startswith("face_") and not cancel()
        assert cropped.parent.name == "decoded"  # Actual private ComfyUI input-directory.
        result = output_root / f"{prefix}.png"
        with Image.open(cropped) as img:
            img.convert("RGB").resize(
                (output_width, output_height), Image.Resampling.LANCZOS,
            ).save(result, format="PNG")
        calls.append((output_width, output_height))
        return result

    monkeypatch.setattr(backend, "_qualified_checkpoint", lambda *args: None)
    monkeypatch.setattr(backend, "_owned_comfyui", no_server)
    monkeypatch.setattr(backend, "_process_frame", mocked_patch)
    monkeypatch.setattr(gpu_lock, "SharedGpuLock", no_gpu_owner)
    monkeypatch.setattr(delivery_insights, "resource_snapshot", lambda: {
        "gpu_memory_free_mib": 8000, "ram_available_bytes": 16 * 1024**3,
    })
    original_hash = full_source_sha256(source)
    code, receipt = run_comfyui_delivery(
        source, jobs_root=tmp_path / "_jobs",
        output_root=tmp_path / "outputs", job_id="face_source_bound_test",
        target_long_edge=512, region_mode="motioncoach_face",
        motioncoach_pose_root=evidence, motioncoach_source_binding_id="source.test_face",
    )
    assert code == 0, receipt
    assert receipt["state"] == "passed" and receipt["region"]["mode"] == "motioncoach_face"
    assert receipt["verification"]["final_full_decode"] == "pass"
    assert receipt["video"]["frames"] == 8
    assert receipt["processing_strategy"]["unique_frames_inferred"] >= 1
    assert len(calls) == receipt["processing_strategy"]["unique_frames_inferred"]
    assert all(min(width, height) >= 64 for width, height in calls)
    assert Path(receipt["output"]).is_file()
    assert "faceonly" in Path(receipt["output"]).name
    assert full_source_sha256(source) == original_hash
    assert not (tmp_path / "_jobs" / "face_source_bound_test" / "temp" / "comfyui_gan").exists()
