"""Source, model, timing and geometry safeguards for real ComfyUI video jobs."""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UPSCALE = ROOT / "tools" / "video_upscale"
WORKSPACE = ROOT / "Spiffy Workspace"
for path in (ROOT, UPSCALE, WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from comfyui_video_backend import (  # noqa: E402
    _qualified_checkpoint,
    _upscale_graph,
    _validate_source,
    _verify_reconstructed_clock,
    _exact_audio_stream_args,
    _verify_full_decode,
    _publish_verified_candidate,
    _run,
    _resource_admission,
    run_comfyui_delivery,
    _preview_guard,
    _temporal_guard,
    _preflight_frame_budget,
)
from workflow.final_delivery_pipeline import (  # noqa: E402
    COMPRESSION_PROFILES,
    build_final_delivery_plan,
    normalize_compression_profile,
)


def test_comfyui_has_real_core_upscaler_graph_no_fake_quality_node():
    graph = _upscale_graph("frame_000001.png", output_width=3840, output_height=2160,
                           output_prefix="Spiffy_2026_some_run_000001")
    assert [graph[k]["class_type"] for k in ("1", "2", "3", "4", "5")] == [
        "LoadImage", "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale", "SaveImage"]
    assert graph["4"]["inputs"]["width"] == 3840
    assert graph["4"]["inputs"]["height"] == 2160


@pytest.mark.parametrize("filename", ["../x.png", "C:/x.png", "..\\x.png", "x.txt", ".hidden.png"])
def test_comfyui_rejects_untrusted_local_video_frame_names(filename):
    with pytest.raises(ValueError):
        _upscale_graph(filename, output_width=512, output_height=288,
                       output_prefix="Spiffy_2026_test_000001")


@pytest.mark.parametrize("width,height", [(0, 512), (100, -1), (8192, 8192), (513, 288)])
def test_comfyui_rejects_unsafe_output_size(width, height):
    with pytest.raises(ValueError):
        _upscale_graph("frame_000001.png", output_width=width, output_height=height,
                       output_prefix="Spiffy_2026_test_000001")


def _info(**kwargs):
    return {"width": 1920, "height": 1080, "fps_ratio": "24000/1001", "fps": 24000 / 1001,
            "duration": 10.01, "sar": "1:1", "rotation": 0, "has_alpha": False,
            "field_order": "progressive", "color_transfer": "bt709", "is_vfr": False,
            "frame_rate_evidence": "presentation-timestamps", "video_pts_uniform": True,
            "video_pts_count": 240, "video_pts_start": "0",
            "video_pts_average_interval": "1001/24000", "video_pts_time_base": "1/24000",
            "video_pts_end_exclusive": "1001/100", **kwargs}


def test_comfyui_1080p_source_can_request_native_model_6k_and_preserve_fractional_fps():
    assert _validate_source(_info(), 5760) == (5760, 3240, "24000/1001")


def test_comfyui_720p_source_cannot_falsely_claim_6k_from_4x_model():
    with pytest.raises(ValueError, match="cannot produce native AI detail"):
        _validate_source({**_info(), "width": 1280, "height": 720}, 5760)


def test_comfyui_refuses_4k_source_due_to_x4_tensor_memory_even_at_6k_delivery():
    with pytest.raises(ValueError, match="native tensor"):
        _validate_source({**_info(), "width": 3840, "height": 2160}, 5760)


def test_comfyui_resource_admission_prevents_disk_or_gpu_overcommit():
    source = _info()
    snapshot = {"gpu_memory_free_mib": 7600, "ram_available_bytes": 16 * 1024**3}
    accepted = _resource_admission(source, 5760, 3240,
                                   available_disk_bytes=150 * 1024**3, snapshot=snapshot)
    assert accepted["required_free_gpu_mib"] == 6144
    with pytest.raises(RuntimeError, match="disk headroom"):
        _resource_admission(source, 5760, 3240,
                            available_disk_bytes=1024**3, snapshot=snapshot)
    with pytest.raises(RuntimeError, match="GPU memory"):
        _resource_admission(source, 5760, 3240,
                            available_disk_bytes=150 * 1024**3,
                            snapshot={**snapshot, "gpu_memory_free_mib": 5120})
    x2 = _resource_admission(source, 3840, 2160, native_scale=2,
                            available_disk_bytes=150 * 1024**3, snapshot=snapshot)
    x4 = _resource_admission(source, 3840, 2160, native_scale=4,
                            available_disk_bytes=150 * 1024**3, snapshot=snapshot)
    assert x2["required_free_gpu_mib"] == 3584
    assert x4["required_free_gpu_mib"] == 6144
    assert x2["native_tensor_pixels"] < x4["native_tensor_pixels"]


@pytest.mark.parametrize("updates", [
    {"frame_rate_evidence": "header-heuristic"},
    {"video_pts_uniform": False}, {"video_pts_start": "1/2"},
    {"video_pts_average_interval": "1/10"},
    {"video_pts_end_exclusive": "10"},
    {"audio_streams": [{"sample_rate": 48000, "start_time": "1/10"}]},
])
def test_comfyui_rejects_unqualified_source_clock_or_audio_offset(updates):
    with pytest.raises(ValueError, match="clock|offset|timestamps"):
        _validate_source(_info(**updates), 3840)


def test_comfyui_clock_accepts_exact_fractional_rate_but_rejects_drift():
    source = _info()
    encoded = {**source, "video_pts_time_base": "1/60000"}
    _verify_reconstructed_clock(source, encoded, "24000/1001", phase="final")
    with pytest.raises(RuntimeError, match="changed the source frame clock"):
        _verify_reconstructed_clock(source, {**encoded, "video_pts_end_exclusive": "103/10"},
                                    "24000/1001", phase="final")


def test_comfyui_audio_assembly_trims_every_track_without_padding():
    source = {"audio_stream_count": 2, "audio_streams": [
        {"sample_count": 31968, "sample_rate": 48000, "channels": 2, "start_time": "0"},
        {"sample_count": 29400, "sample_rate": 44100, "channels": 1, "start_time": "0"},
    ]}
    args = _exact_audio_stream_args(source)
    assert args[:2] == ["-filter_complex", (
        "[1:a:0]atrim=start_sample=0:end_sample=31968,asetpts=N/SR/TB[a0];"
        "[1:a:1]atrim=start_sample=0:end_sample=29400,asetpts=N/SR/TB[a1]"
    )]
    assert args[2:] == ["-map", "0:v:0", "-map", "[a0]", "-map", "[a1]"]
    assert _exact_audio_stream_args({"audio_stream_count": 0}) == ["-map", "0:v:0", "-an"]


@pytest.mark.parametrize("track", [
    {"sample_rate": 48000, "channels": 1, "start_time": "0"},
    {"sample_count": 0, "sample_rate": 48000, "channels": 1, "start_time": "0"},
    {"sample_count": 44100, "sample_rate": 48000, "channels": 1, "start_time": "1/10"},
])
def test_comfyui_audio_assembly_rejects_ambiguous_or_shifted_audio(track):
    with pytest.raises(ValueError, match="sample|offset|evidence"):
        _exact_audio_stream_args({"audio_stream_count": 1, "audio_streams": [track]})


def test_comfyui_publication_is_collision_safe_and_preserves_original(tmp_path):
    source = tmp_path / "source.mp4"
    original = b"original-original-source"
    source.write_bytes(original)
    folder = tmp_path / "published"
    folder.mkdir()
    target = folder / "source__smart_finish_comfyui_512x288.mp4"
    target.write_bytes(b"pre-existing-delivery")
    candidate = tmp_path / "candidate.mp4"
    candidate.write_bytes(b"verified-compressed-output")
    actual = _publish_verified_candidate(candidate, target, source, "comfy_test_1", lambda: False)
    assert actual.name == "source__smart_finish_comfyui_512x288 (2).mp4"
    assert actual.read_bytes() == b"verified-compressed-output"
    assert source.read_bytes() == original
    assert target.read_bytes() == b"pre-existing-delivery"
    assert not candidate.exists()


def test_comfyui_publication_cancel_does_not_consume_or_delete_candidate(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    candidate = tmp_path / "candidate.mp4"
    candidate.write_bytes(b"verified")
    target = tmp_path / "output.mp4"
    with pytest.raises(RuntimeError, match="cancelled"):
        _publish_verified_candidate(candidate, target, source, "comfy_test_2", lambda: True)
    assert candidate.read_bytes() == b"verified"
    assert not target.exists()


def test_comfyui_process_cancel_stops_owned_child_quickly():
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        return calls >= 3
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="cancelled"):
        _run([sys.executable, "-c", "import time;time.sleep(30)"], timeout=15, cancel=cancel)
    assert time.monotonic() - started < 5


def test_full_decode_mandatory_gate_reads_video_and_all_audio_tracks(tmp_path):
    import shutil
    from shared.ffmpeg_utils import standard_ffmpeg

    if shutil.which("ffmpeg") is None:
        pytest.skip("Local FFmpeg not available")
    ffmpeg = standard_ffmpeg()
    fixture = tmp_path / "private_test.mp4"
    _run([
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=128x72:rate=12:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(fixture),
    ], timeout=20)
    elapsed = _verify_full_decode(fixture, ffmpeg, duration_seconds=1, cancel=lambda: False)
    assert elapsed >= 0
    with pytest.raises(RuntimeError, match="cancelled"):
        _verify_full_decode(fixture, ffmpeg, duration_seconds=1, cancel=lambda: True)
    fixture.write_bytes(b"not a valid video container at all")
    with pytest.raises(RuntimeError, match="Media stage failed"):
        _verify_full_decode(fixture, ffmpeg, duration_seconds=1, cancel=lambda: False)


def test_comfyui_failed_model_preflight_cleans_own_temps_and_retains_result(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"nonvideo-private-source-is-untouched")
    jobs = tmp_path / "_jobs"
    status, receipt = run_comfyui_delivery(source, jobs_root=jobs, job_id="comfy_failure_case",
                                          comfy_root=tmp_path / "missing-comfyui")
    assert status == 1 and receipt["state"] == "failed"
    directory = jobs / "comfy_failure_case"
    assert json.loads((directory / "comfyui_delivery_result.json").read_text())["state"] == "failed"
    assert not (directory / "temp" / "comfyui_gan").exists()
    assert source.read_bytes() == b"nonvideo-private-source-is-untouched"
    with pytest.raises(RuntimeError, match="already claimed"):
        run_comfyui_delivery(source, jobs_root=jobs, job_id="comfy_failure_case",
                             comfy_root=tmp_path / "missing-comfyui")


def test_comfyui_frame_budget_is_checked_before_any_gpu_or_png_work():
    info = {**_info(), "video_frame_count": "1801"}
    with pytest.raises(ValueError, match="frame count exceeds"):
        _preflight_frame_budget(info, "60/1", 1800)
    with pytest.raises(ValueError, match="duration and frame rate"):
        _preflight_frame_budget({**_info(), "video_frame_count": ""}, "60/1", 100)
    _preflight_frame_budget({**_info(), "video_frame_count": "240"}, "24/1", 300)


@pytest.mark.parametrize("updates", [
    {"color_transfer": "smpte2084"}, {"has_alpha": True}, {"field_order": "tt"},
    {"is_vfr": True}, {"sar": "4:3"}, {"rotation": 90},
])
def test_comfyui_rejects_unqualified_media(updates):
    with pytest.raises(ValueError, match="separate qualification"):
        _validate_source({**_info(), **updates}, 3840)


def test_comfyui_refuses_unpinned_checkpoint(tmp_path):
    checkpoint = tmp_path / "models" / "upscale_models"
    checkpoint.mkdir(parents=True)
    (checkpoint / "RealESRGAN_x4plus.pth").write_bytes(b"incomplete-or-untrusted-weights")
    with pytest.raises(RuntimeError, match="checksum/identity"):
        _qualified_checkpoint(tmp_path)


def test_comfyui_guards_luma_and_preserves_exact_temporal_holds(tmp_path):
    from PIL import Image

    source, result = tmp_path / "source.png", tmp_path / "out.png"
    Image.new("RGB", (128, 72), (72, 100, 120)).save(source)
    Image.new("RGB", (512, 288), (75, 102, 121)).save(result)
    current = _preview_guard(source, result, 512, 288)
    _temporal_guard(current, current)
    Image.new("RGB", (512, 288), (150, 30, 40)).save(result)
    with pytest.raises(RuntimeError, match="global color"):
        _preview_guard(source, result, 512, 288)
    with pytest.raises(RuntimeError, match="frame-to-frame|nearly static"):
        _temporal_guard(current, (current[0], b"\xff" * (80 * 80 * 3)))


def test_comfyui_rejects_new_texture_flicker_in_nearly_static_original():
    still = b"\x64" * (80 * 80 * 3)
    previous = (still, still)
    current = (still, b"\x71" * (80 * 80 * 3))
    with pytest.raises(RuntimeError, match="nearly static"):
        _temporal_guard(previous, current)


def test_compression_profiles_use_quality_crf_and_keep_existing_default(tmp_path):
    source = tmp_path / "src.mp4"
    source.write_bytes(b"fixture-placeholder")
    plan = build_final_delivery_plan(source, tmp_path / "delivery",
                                     profile_ids=["smart_finish_master"])
    assert plan["variants"][0]["crf"] == 17
    quality = build_final_delivery_plan(source, tmp_path / "delivery",
                                        profile_ids=["smart_finish_master"],
                                        compression_profile="quality")
    assert quality["compression_profile"] == "quality"
    assert quality["variants"][0]["crf"] == 16
    assert quality["variants"][0]["x264_preset"] == "slow"
    assert COMPRESSION_PROFILES["maximum"][0] < COMPRESSION_PROFILES["quality"][0]
    assert COMPRESSION_PROFILES["compact"][0] > COMPRESSION_PROFILES["balanced"][0]


@pytest.mark.parametrize("bad", [True, 20, "lossless", "ultra_quality", ""])
def test_compression_profile_rejects_unrecognized_modes(bad):
    with pytest.raises(ValueError):
        normalize_compression_profile(bad)
