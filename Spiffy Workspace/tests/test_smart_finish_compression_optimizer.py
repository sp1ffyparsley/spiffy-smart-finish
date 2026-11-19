"""Measured compression pilots: conservative selection and failure-safe cleanup."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
UPSCALE = ROOT / "tools" / "video_upscale"
for path in (ROOT, UPSCALE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compression_optimizer import (  # noqa: E402
    ADMITTED_CRFS, GATES, SCHEMA, calibrate_h264, meets_high_quality_gate,
    parse_measurements, sample_windows,
)


def _valid_vmaf(frames: int, *, score: float = 99.0) -> dict:
    return {
        "frames": [{"frameNum": i} for i in range(frames)],
        "pooled_metrics": {
            "vmaf": {"mean": score, "min": score - 2},
            "float_ssim": {"mean": 0.998},
            "psnr_y": {"mean": 49.5},
        },
    }


def test_sample_windows_are_small_separate_and_never_claim_6k_pilot():
    assert sample_windows(2.5, "24/1", output_pixels=640 * 360) == []
    assert sample_windows(8, "24/1", output_pixels=640 * 360)[0]["expected_frames"] >= 8
    assert sample_windows(30, "24/1", output_pixels=640 * 360).__len__() == 2
    assert sample_windows(30, "24/1", output_pixels=5760 * 3240) == []
    assert sample_windows(30, "6/1", output_pixels=640 * 360) == []
    for window in sample_windows(30, "24000/1001", output_pixels=640 * 360):
        assert window["duration_seconds"] <= 1.1
        assert window["expected_frames"] <= 24


@pytest.mark.parametrize("rate,duration", [("0/1", 9), ("garbage", 9), ("24/1", float("nan")), ("24/1", -4)])
def test_sample_windows_reject_invalid_timing(rate, duration):
    with pytest.raises(ValueError):
        sample_windows(duration, rate, output_pixels=640 * 360)


def test_vmaf_parser_rejects_missing_nonfinite_and_incorrect_frame_coverage():
    good = parse_measurements(_valid_vmaf(16), 16)
    assert good["vmaf_mean"] == 99
    assert meets_high_quality_gate(good)
    with pytest.raises(ValueError, match="frame count"):
        parse_measurements(_valid_vmaf(15), 16)
    with pytest.raises(ValueError, match="missing"):
        parse_measurements({"frames": [{}] * 16, "pooled_metrics": {}}, 16)
    invalid = _valid_vmaf(16)
    invalid["pooled_metrics"]["vmaf"]["mean"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        parse_measurements(invalid, 16)
    assert not meets_high_quality_gate({**good, "vmaf_min": GATES["vmaf_min"] - 0.1})
    assert not meets_high_quality_gate({**good, "psnr_y_mean_db": 42.0})


def test_pilot_chooses_18_when_20_does_not_pass_and_cleans_own_files(tmp_path):
    reference = tmp_path / "lossless.mov"
    reference.write_bytes(b"internal-simulated-reference")
    calls = []

    def run(argv, *, timeout, cancel, cwd=None):
        calls.append(argv)
        assert not cancel() and timeout <= 90
        if "libvmaf" in " ".join(argv):
            crf = int(Path(argv[argv.index("-i") + 1]).stem.split("_")[1])
            expected = 16
            (Path(cwd) / f"vmaf_{crf}_0.json").write_text(
                json.dumps(_valid_vmaf(expected, score=98.8 if crf == 18 else 92.0)), encoding="utf8")
        else:
            Path(argv[-1]).write_bytes(b"fake-small-pilot-for-unit-test")

    result = calibrate_h264(
        reference, ffmpeg="ffmpeg", temp_root=tmp_path,
        duration=8.0, fps="24/1", width=640, height=360,
        baseline_crf=16, cancel=lambda: False, run_command=run)
    assert result["schema"] == SCHEMA
    assert result["selected_crf"] == 18
    assert result["candidate_results"][0]["passed"]
    assert not result["candidate_results"][1]["passed"]
    assert len(calls) == 4
    assert not (tmp_path / "compression_calibration").exists()
    assert reference.read_bytes() == b"internal-simulated-reference"


def test_pilot_errors_keep_16_and_remove_intermediates(tmp_path):
    reference = tmp_path / "lossless.mov"
    reference.write_bytes(b"private-reference")

    def failure(*args, **kwargs):
        raise RuntimeError("VMAF library unavailable")

    result = calibrate_h264(
        reference, ffmpeg="ffmpeg", temp_root=tmp_path,
        duration=8.0, fps="24/1", width=640, height=360,
        baseline_crf=16, cancel=lambda: False, run_command=failure)
    assert result["status"] == "pilot_unavailable_baseline_preserved"
    assert result["selected_crf"] == 16
    assert not (tmp_path / "compression_calibration").exists()


def test_short_and_large_sources_do_not_launch_pilot(tmp_path):
    for duration, width, height in ((1.0, 640, 360), (15.0, 5760, 3240)):
        result = calibrate_h264(
            tmp_path / "not_necessary.mov", ffmpeg="ffmpeg",
            temp_root=tmp_path, duration=duration, fps="24/1",
            width=width, height=height, baseline_crf=16,
            cancel=lambda: False,
            run_command=lambda *args, **kwargs: pytest.fail("Unneeded compression pilot launched"))
        assert result["selected_crf"] == 16
        assert result["status"].startswith("skipped")
    with pytest.raises(RuntimeError, match="cancelled"):
        calibrate_h264(tmp_path / "reference.mov", ffmpeg="ffmpeg",
                       temp_root=tmp_path, duration=8, fps="24/1",
                       width=640, height=360, baseline_crf=16,
                       cancel=lambda: True, run_command=lambda *a, **k: None)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="Local FFmpeg unavailable")
def test_actual_libvmaf_h264_pilot_returns_source_bound_evidence(tmp_path):
    from comfyui_video_backend import _run
    from shared.ffmpeg_utils import standard_ffmpeg

    ffmpeg = standard_ffmpeg()
    reference = tmp_path / "internal_prores.mov"
    _run([ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
          "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=6",
          "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
          "-an", str(reference)], timeout=50)
    result = calibrate_h264(
        reference, ffmpeg=ffmpeg, temp_root=tmp_path,
        duration=6.0, fps="24/1", width=320, height=180, baseline_crf=16,
        cancel=lambda: False, run_command=_run)
    assert result["status"] in {"measured_high_quality_candidate", "baseline_retained"}
    assert result["selected_crf"] in (16, *ADMITTED_CRFS)
    assert result["full_resolution_measurements"] is True
    assert result["candidate_results"]
    assert result["candidate_results"][0]["samples"][0]["scores"]["vmaf_mean"] > 0
    assert not (tmp_path / "compression_calibration").exists()
