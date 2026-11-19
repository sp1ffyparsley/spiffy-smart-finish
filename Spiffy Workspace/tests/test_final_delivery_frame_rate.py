"""Explicit Smart Finish cadence contracts; no model download or production media."""
from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import workflow.final_delivery_pipeline as pipeline
import video_upscale as worker
from tools import run_spiffy_final_delivery as runner


@pytest.mark.parametrize("rate", [30, 60, 90, 120, "24000/1001", "30000/1001", "60000/1001"])
def test_requested_rate_reaches_request_plan_variant_and_binding(tmp_path, rate):
    spec = {"mode": "cfr", "fps": rate}
    expected = {"mode": "cfr", "fps": str(Fraction(rate).numerator) + "/" + str(Fraction(rate).denominator)}
    options = pipeline.request_delivery_options({"final_delivery": {"delivery_set": "smart_finish", "frame_rate": spec}})
    assert options["frame_rate"] == expected
    plan = pipeline.build_final_delivery_plan(tmp_path / "in.mp4", tmp_path / "out",
        profile_ids=options["profile_ids"], frame_rate=options["frame_rate"])
    assert plan["frame_rate"] == expected
    assert plan["schema"] == "spiffy_final_delivery_plan_v2"
    assert plan["variants"][0]["frame_rate"] == expected
    assert plan["variants"][0]["max_fps"] >= Fraction(rate)
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan), encoding="utf8")
    assert worker.load_delivery_plan(p)["frame_rate"] == expected
    legacy = pipeline.build_final_delivery_plan(tmp_path / "in.mp4", tmp_path / "out", profile_ids=["smart_finish_master"])
    assert "frame_rate" not in legacy
    assert legacy["variants"][0]["max_fps"] == 60
    assert pipeline._plan_fingerprint(plan) != pipeline._plan_fingerprint(legacy)
    legacy_path = tmp_path / "legacy-plan.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf8")
    assert "frame_rate" not in worker.load_delivery_plan(legacy_path)


@pytest.mark.parametrize("spec", [None, [], True, {}, {"mode": "cfr"},
    {"mode": "preserve"}, {"mode": "cfr", "fps": 90, "extra": 1},
    *[{"mode": "cfr", "fps": x} for x in [True, None, [], {}, 0, -1, 121, "1/0", "0/1", "NaN", "90:device=-1", "1e5000", float("nan"), float("inf")]]])
def test_invalid_rate_request_fails_before_job(spec):
    with pytest.raises(ValueError, match="frame_rate"):
        pipeline.request_delivery_options({"final_delivery": {"frame_rate": spec}})


def test_master_retains_120_while_platform_derivatives_are_explicitly_capped(tmp_path):
    plan = pipeline.build_final_delivery_plan(tmp_path / "in.mp4", tmp_path / "out",
        profile_ids=["native_master", "generic_vertical_short"], frame_rate={"mode": "cfr", "fps": 120})
    by_id = {row["profile_id"]: row for row in plan["variants"]}
    assert by_id["native_master"]["frame_rate"]["fps"] == "120/1"
    assert by_id["generic_vertical_short"]["frame_rate"]["fps"] == "60/1"
    assert by_id["generic_vertical_short"]["frame_rate_reason"] == "profile_ceiling"
    with pytest.raises(ValueError, match="master"):
        pipeline.build_final_delivery_plan(tmp_path / "in.mp4", tmp_path / "out",
            profile_ids=["generic_vertical_short"], frame_rate={"mode": "cfr", "fps": 120})


@pytest.mark.parametrize("profile_id,rate,max_fps,reason", [
    ("native_master", "90/1", 90, "requested"),
    ("generic_vertical_short", "30/1", 30, "profile_ceiling"),
    ("generic_vertical_short", "60/1", 120, "profile_ceiling"),
])
def test_worker_rejects_variant_rate_that_disagrees_with_profile_contract(
    tmp_path, profile_id, rate, max_fps, reason,
):
    plan = pipeline.build_final_delivery_plan(
        tmp_path / "in.mp4", tmp_path / "out",
        profile_ids=["native_master", "smart_finish_master", "generic_vertical_short"],
        frame_rate={"mode": "cfr", "fps": 120},
    )
    variant = next(row for row in plan["variants"] if row["profile_id"] == profile_id)
    variant["frame_rate"] = {"mode": "cfr", "fps": rate}
    variant["max_fps"] = max_fps
    variant["frame_rate_reason"] = reason
    path = tmp_path / "tampered-plan.json"
    path.write_text(json.dumps(plan), encoding="utf8")

    with pytest.raises(ValueError, match="profile frame-rate contract"):
        worker.load_delivery_plan(path)


def test_worker_rejects_incorrect_per_output_frame_rate_reason(tmp_path):
    plan = pipeline.build_final_delivery_plan(
        tmp_path / "in.mp4", tmp_path / "out",
        profile_ids=["native_master", "generic_vertical_short"],
        frame_rate={"mode": "cfr", "fps": 120},
    )
    derivative = next(row for row in plan["variants"] if row["profile_id"] == "generic_vertical_short")
    assert derivative["frame_rate"] == {"mode": "cfr", "fps": "60/1"}
    derivative["frame_rate_reason"] = "requested"
    path = tmp_path / "tampered-reason.json"
    path.write_text(json.dumps(plan), encoding="utf8")

    with pytest.raises(ValueError, match="frame_rate_reason"):
        worker.load_delivery_plan(path)


def test_source_pts_preserve_policy_reaches_request_plan_worker_and_master_capabilities(tmp_path):
    policy = {"mode": "preserve_source"}
    options = pipeline.request_delivery_options({
        "final_delivery": {"profile_ids": ["native_master", "smart_finish_master"], "frame_rate": policy}
    })
    assert options["frame_rate"] == policy
    plan = pipeline.build_final_delivery_plan(
        tmp_path / "source.mov", tmp_path / "out",
        profile_ids=options["profile_ids"], frame_rate=options["frame_rate"],
    )
    assert plan["schema"] == "spiffy_final_delivery_plan_v2"
    assert plan["frame_rate"] == policy
    assert plan["frame_rate_boundary"] == "source_pts_preserve_v1"
    assert {row["profile_id"] for row in plan["variants"]} == {"native_master", "smart_finish_master"}
    for row in plan["variants"]:
        assert row["frame_rate"] == policy
        assert row["frame_rate_reason"] == "source_preserved"
        assert row["max_fps"] == 120
    path = tmp_path / "source-preserve-plan.json"
    path.write_text(json.dumps(plan), encoding="utf8")
    assert worker.load_delivery_plan(path)["frame_rate"] == policy
    assert pipeline._plan_fingerprint(plan) != pipeline._plan_fingerprint(
        pipeline.build_final_delivery_plan(tmp_path / "source.mov", tmp_path / "out",
            profile_ids=["native_master", "smart_finish_master"])
    )
    with pytest.raises(ValueError, match="does not declare exact source-PTS"):
        pipeline.build_final_delivery_plan(
            tmp_path / "source.mov", tmp_path / "out",
            profile_ids=["generic_vertical_short"], frame_rate=policy,
        )


def test_source_pts_preserve_plan_rejects_tampered_capability_or_watermark(tmp_path):
    policy = {"mode": "preserve_source"}
    plan = pipeline.build_final_delivery_plan(
        tmp_path / "source.mov", tmp_path / "out", profile_ids=["native_master"], frame_rate=policy,
    )
    variant = plan["variants"][0]
    variant["max_fps"] = 60
    path = tmp_path / "bad-source-cap.json"
    path.write_text(json.dumps(plan), encoding="utf8")
    with pytest.raises(ValueError, match="profile frame-rate|capabilit"):
        worker.load_delivery_plan(path)

    plan = pipeline.build_final_delivery_plan(
        tmp_path / "source.mov", tmp_path / "out", profile_ids=["native_master"], frame_rate=policy,
    )
    plan["variants"][0]["watermark"] = True
    path.write_text(json.dumps(plan), encoding="utf8")
    with pytest.raises(ValueError, match="VideoSeal"):
        worker.load_delivery_plan(path)


def test_source_pts_preserve_cli_option_is_explicit_and_mutually_exclusive(tmp_path):
    args = runner.parse_args([
        "--input", str(tmp_path / "pending.mov"), "--profiles", "native_master",
        "--preserve-source-timing",
    ])
    assert args.preserve_source_timing is True
    assert args.output_fps is None
    with pytest.raises(SystemExit):
        runner.parse_args([
            "--input", str(tmp_path / "pending.mov"), "--preserve-source-timing", "--output-fps", "90",
        ])


def test_cli_source_pts_preserve_option_is_persisted_in_the_delivery_plan(tmp_path, capsys):
    jobs = tmp_path / "jobs"
    assert runner.main([
        "--input", str(tmp_path / "pending.mov"), "--profiles", "native_master",
        "--jobs-root", str(jobs), "--job-id", "source-pts-preserve", "--preserve-source-timing",
    ]) == 0
    result = json.loads((jobs / "source-pts-preserve" / "final_delivery_result.json").read_text(encoding="utf8"))
    plan = worker.load_delivery_plan(result["sources"][0]["plan_path"])
    assert plan["frame_rate"] == {"mode": "preserve_source"}
    assert plan["frame_rate_boundary"] == "source_pts_preserve_v1"


def test_source_pts_preserve_progress_and_profile_cap_use_minimum_exact_pts_interval():
    source = {
        "frame_rate_evidence": "presentation-timestamps", "video_pts_count": 4,
        "video_pts_sha256": "a" * 64, "video_time_base": "1/120",
        "video_pts_min_interval": "1/120",
    }
    assert worker.validate_source_timing_profile_cap(source, 120) == 120
    with pytest.raises(RuntimeError, match="above this output profile"):
        worker.validate_source_timing_profile_cap(source, 90)
    metrics = worker.frame_rate_progress_metrics([{
        "profile_id": "native_master", "profile_ids": ["native_master"],
        "frame_rate": {"mode": "preserve_source"}, "frame_rate_reason": "source_preserved",
    }], default_max_fps=None, source_info=source)
    assert metrics["max_output_fps"] == 120.0
    assert metrics["output_frame_rates"][0]["max_instantaneous_fps"] == "120"


def test_source_pts_preserve_skips_interpolation_and_requires_exact_pts_evidence():
    source = {
        "frame_rate_evidence": "presentation-timestamps", "video_pts_count": 4,
        "video_pts_sha256": "a" * 64, "video_pts_min_interval": "1/30",
        "field_order": "progressive", "is_vfr": True,
    }
    writer = SimpleNamespace(log=lambda _: None)
    policy = {"mode": "preserve_source"}
    assert worker.should_interpolate(source, "ss_4k", writer, {"tvai_fi": False}, frame_rate=policy) is False
    worker.require_topaz_interpolation_route(source, "ss_4k", False, frame_rate=policy)
    with pytest.raises(RuntimeError, match="presentation timestamp evidence"):
        worker.should_interpolate({"is_vfr": True}, "ss_4k", writer, {}, frame_rate=policy)


def test_source_pts_preserve_final_validation_compares_source_digest_and_cap():
    source = {
        "frame_rate_evidence": "presentation-timestamps", "video_pts_count": 4,
        "video_pts_sha256": "a" * 64, "video_time_base": "1/30", "video_pts_time_base": "1/30",
        "video_pts_start": "1/10", "video_pts_end": "1/5", "video_pts_span": "1/10",
        "video_pts_min_interval": "1/30",
        "video_pts_last_duration": "1/30", "video_pts_end_exclusive": "7/30",
        "video_pts_endpoint_evidence": "decoded-last-frame-duration",
    }
    candidate = {**source, "width": 64, "height": 48}
    worker.validate_final_delivery(
        candidate, expected_width=64, expected_height=48, preset_id="",
        frame_rate={"mode": "preserve_source"}, source_info=source, source_timing_max_fps=120,
    )
    with pytest.raises(RuntimeError, match="timestamps changed"):
        worker.validate_final_delivery(
            {**candidate, "video_pts_sha256": "b" * 64}, expected_width=64, expected_height=48,
            preset_id="", frame_rate={"mode": "preserve_source"}, source_info=source,
            source_timing_max_fps=120,
        )


def test_parent_child_receipt_revalidates_source_pts_against_ready_source_probe(tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source-identity")
    policy = {"mode": "preserve_source"}
    plan = pipeline.build_final_delivery_plan(
        source, tmp_path / "out", profile_ids=["native_master"], frame_rate=policy,
    )
    planned_variant = plan["variants"][0]
    output = Path(planned_variant["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"final-output")
    resolved_variant = {**planned_variant, "width": 64, "height": 48}
    source_probe = {
        "frame_rate_evidence": "presentation-timestamps", "video_pts_count": 4,
        "video_pts_sha256": "a" * 64, "video_time_base": "1/30", "video_pts_time_base": "1/30",
        "video_pts_start": "1/10", "video_pts_end": "1/5", "video_pts_span": "1/10",
        "video_pts_min_interval": "1/30",
        "video_pts_last_duration": "1/30", "video_pts_end_exclusive": "7/30",
        "video_pts_endpoint_evidence": "decoded-last-frame-duration",
    }

    def reconcile(candidate_digest, **endpoint_overrides):
        candidate_probe = {**source_probe, "video_pts_sha256": candidate_digest,
            "width": 64, "height": 48, **endpoint_overrides}
        completion = {
            "output": {"path": str(output)},
            "probe": candidate_probe,
            "fingerprint": "completion-fingerprint",
            "contract": {"variant": resolved_variant, "frame_rate": policy},
        }
        state = {
            "matched_source_key": pipeline._path_key(source),
            "resolved_source": {
                "variants": [resolved_variant],
                "profile_output_map": dict(plan["profile_output_map"]),
                "physical_output_count": plan["physical_output_count"],
            },
            "output_paths": [str(output)],
            "completion_rows": [completion],
        }
        child_manifest = {
            "worker": "video_upscale", "errors": [], "delivery_plan": plan,
            "enhancement_recipe": None, "frame_rate": policy,
            "ready_sources": [{
                "selected_source_path": str(source), "source_path": str(source), "probe": source_probe,
            }],
            "final_outputs": [str(output)],
        }
        with patch.object(pipeline, "validated_completed_source_states", return_value={
            "states": [state], "completed_source_row_count": 1,
        }):
            return pipeline._validated_child_delivery(
                child_manifest, child_manifest_path=tmp_path / "child-manifest.json",
                source=source, source_identity=pipeline._stable_file_identity(source), plan=plan,
            )

    passed = reconcile("a" * 64)
    assert passed["status"] == "PASS"
    failed = reconcile("b" * 64)
    assert failed["status"] == "FAIL"
    assert "child_output_source_timing_invalid" in failed["issues"]
    missing = reconcile("a" * 64, video_pts_endpoint_evidence="unavailable")
    assert missing["status"] == "FAIL"
    assert "child_output_source_timing_invalid" in missing["issues"]
    shortened = reconcile("a" * 64, video_pts_last_duration="1/60", video_pts_end_exclusive="13/60")
    assert shortened["status"] == "FAIL"
    assert "child_output_source_timing_invalid" in shortened["issues"]


def test_progress_metrics_report_per_output_rates_instead_of_the_legacy_preset_cap():
    metrics = worker.frame_rate_progress_metrics([
        {"profile_id": "smart_finish_master", "profile_ids": ["smart_finish_master"],
         "frame_rate": {"mode": "cfr", "fps": "120/1"}, "frame_rate_reason": "requested"},
        {"profile_id": "generic_vertical_short", "profile_ids": ["generic_vertical_short"],
         "frame_rate": {"mode": "cfr", "fps": "60/1"}, "frame_rate_reason": "profile_ceiling"},
    ], default_max_fps=60)
    assert metrics["max_output_fps"] == 120.0
    assert [row["fps"] for row in metrics["output_frame_rates"]] == ["120/1", "60/1"]
    assert metrics["output_frame_rates"][1]["reason"] == "profile_ceiling"
    fractional = worker.frame_rate_progress_metrics([
        {"profile_id": "smart_finish_master", "frame_rate": {"mode": "cfr", "fps": "60000/1001"}}
    ], default_max_fps=60)
    assert fractional["output_frame_rates"][0]["fps"] == "60000/1001"
    assert fractional["max_output_fps"] == pytest.approx(60000 / 1001)
    assert worker.frame_rate_progress_metrics([{"profile_id": "native_master"}], 60) == {"max_output_fps": 60}


def test_explicit_cadence_audio_validation_requires_exact_per_track_sample_extent():
    source = {"audio_stream_count": 1, "audio_streams": [{
        "sample_rate": 48000, "channels": 2, "sample_count": 14400, "start_time": "0"
    }]}
    worker.validate_intermediate_audio(source, {"audio_stream_count": 1, "audio_streams": [{
        "sample_rate": 48000, "channels": 2, "sample_count": 14400, "start_time": "0"
    }]}, phase="test", require_sample_extent=True)
    with pytest.raises(RuntimeError, match="sample extent"):
        worker.validate_intermediate_audio(source, {"audio_stream_count": 1, "audio_streams": [{
            "sample_rate": 48000, "channels": 2, "sample_count": 15360, "start_time": "0"
        }]}, phase="test", require_sample_extent=True)
    with pytest.raises(RuntimeError, match="cannot prove"):
        worker.validate_intermediate_audio(source, {"audio_stream_count": 1},
            phase="test", require_sample_extent=True)


def test_rate_is_not_a_float_tolerance_or_a_fake_frame_generator(tmp_path):
    writer = SimpleNamespace(log=lambda _: None)
    policy = {"mode": "cfr", "fps": "60/1"}
    info = {"fps": 60000 / 1001, "fps_ratio": "60000/1001", "field_order": "progressive", "width": 64, "height": 48}
    assert worker.should_interpolate(info, "ss_4k", writer, {"tvai_fi": True}, frame_rate=policy)
    with pytest.raises(RuntimeError, match="interpolation"):
        worker.require_topaz_interpolation_route(info, "ss_4k", False, frame_rate=policy)
    with pytest.raises(RuntimeError, match="frame rate|FPS"):
        worker.validate_final_delivery(info, expected_width=64, expected_height=48, preset_id="4k_60fps", frame_rate=policy)
    worker.validate_final_delivery({**info, "fps": 120.0, "fps_ratio": "120/1"}, expected_width=64,
        expected_height=48, preset_id="4k_60fps", frame_rate={"mode": "cfr", "fps": "120/1"})


def test_fractional_checkpoint_and_preflight_use_exact_requested_rate(monkeypatch, tmp_path):
    source, runtime = tmp_path / "source", tmp_path / "runtime"
    source.write_bytes(b"source")
    runtime.write_bytes(b"runtime")
    common = dict(stage="topaz_interpolate", input_path=source, topaz_ffmpeg=runtime,
        preset_id="4k_60fps", model="apo-8", target_width=64, target_height=48)
    a = worker.build_topaz_checkpoint_fingerprint(**common, target_fps="60000/1001")
    b = worker.build_topaz_checkpoint_fingerprint(**common, target_fps="60/1")
    assert a["contract"]["target_fps"] == "60000/1001"
    assert a["fingerprint"] != b["fingerprint"]
    commands = []
    writer = SimpleNamespace(command=commands.append, log=lambda _: None, cancel_requested=False)
    monkeypatch.setattr(worker, "_run_cancellable_subprocess", lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr="", emitted_frames=2))
    assert worker.validate_topaz_filter_model(str(runtime), "tvai_fi", "apo-8", writer, target_fps="60000/1001")[0]
    assert ":fps=60000/1001:" in commands[0][commands[0].index("-vf") + 1]


def test_plan_rejects_unbound_or_tampered_variant_rate(tmp_path):
    plan = pipeline.build_final_delivery_plan(tmp_path / "in.mp4", tmp_path / "out",
        profile_ids=["smart_finish_master"], frame_rate={"mode": "cfr", "fps": 90})
    plan["variants"][0]["frame_rate"] = {"mode": "cfr", "fps": "120/1"}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan), encoding="utf8")
    with pytest.raises(ValueError, match="frame_rate"):
        worker.load_delivery_plan(p)


def test_cli_rate_override_is_connected(tmp_path):
    assert runner.main(["--input", str(tmp_path / "pending.mp4"), "--delivery-set", "smart_finish",
        "--jobs-root", str(tmp_path / "jobs"), "--job-id", "fps90", "--output-fps", "90"]) == 0
    result = json.loads((tmp_path / "jobs/fps90/final_delivery_result.json").read_text(encoding="utf8"))
    plan = worker.load_delivery_plan(result["sources"][0]["plan_path"])
    assert plan["frame_rate"]["fps"] == "90/1"


@pytest.mark.parametrize("rate", ["30/1", "60/1", "90/1", "120/1", "30000/1001", "60000/1001"])
def test_real_cpu_encode_has_exact_cadence_and_no_sixty_cap(tmp_path, rate):
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    source = tmp_path / "source.mkv"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s=64x48:r={rate}:d=0.2",
        "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=48000:duration=0.2", "-c:v", "ffv1", "-c:a", "pcm_s16le", str(source)],
        check=True, capture_output=True, timeout=20)
    info = worker.probe_video(source, ffprobe, inspect_timestamps=True)
    dest = tmp_path / "out.mp4"
    writer = worker.ProgressWriter(tmp_path / "jobs", "encode", "CFR delivery test", "test")
    policy = {"mode": "cfr", "fps": rate}
    worker.final_mp4(source, dest, ffmpeg, info, "keep_source_aspect", 64, 48, 17, "ultrafast", writer, frame_rate=policy)
    evidence = json.loads(subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=avg_frame_rate,time_base,duration,nb_frames:frame=best_effort_timestamp_time", "-of", "json", str(dest)],
        check=True, capture_output=True, text=True, timeout=20).stdout)
    assert Fraction(evidence["streams"][0]["avg_frame_rate"]) == Fraction(rate)
    stamps = [Fraction(row["best_effort_timestamp_time"]) for row in evidence["frames"]]
    assert len(stamps) >= 5
    assert all(abs((b - a) - 1 / Fraction(rate)) <= Fraction(1, 1000000) for a, b in zip(stamps, stamps[1:]))
    output_info = worker.probe_video(dest, ffprobe, inspect_timestamps=True)
    assert output_info["audio_stream_count"] == 1
    assert output_info["audio_streams"][0]["sample_count"] == info["audio_streams"][0]["sample_count"]
    worker.validate_intermediate_audio(info, output_info, phase="CFR test output", require_sample_extent=True)
    worker.validate_final_delivery(output_info, expected_width=64, expected_height=48,
        preset_id="4k_60fps", frame_rate=policy)


def test_plain_encode_cannot_impersonate_interpolation(tmp_path):
    writer = SimpleNamespace(log=lambda _: None)
    with pytest.raises(RuntimeError, match="interpolation"):
        worker.final_mp4(tmp_path / "in", tmp_path / "out.mp4", "not-executed",
            {"width": 64, "height": 48, "fps": 30, "fps_ratio": "30/1"}, "keep_source_aspect", 64, 48, 17, "fast", writer,
            frame_rate={"mode": "cfr", "fps": 120})


@pytest.mark.parametrize("rate,frames", [("24000/1001", 8), ("30000/1001", 5), ("60000/1001", 5)])
def test_fractional_end_audio_samples_survive_mp4_movie_clock(tmp_path, rate, frames):
    """Real AAC/MP4 muxing: the default 1 kHz movie clock lost 32 samples."""
    duration = Fraction(frames, 1) / Fraction(rate)
    sample_count = duration * 48000
    assert sample_count.denominator == 1
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    source = tmp_path / "fractional_audio.mov"
    subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
        "-f", "lavfi", "-i", f"testsrc2=s=64x48:r={rate}:d={float(duration):.12f}",
        "-f", "lavfi", "-i", f"sine=frequency=700:sample_rate=48000:duration={float(duration):.12f}",
        "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-threads", "2",
        "-c:a", "pcm_s24le", "-video_track_timescale", str(Fraction(rate).numerator),
        "-movie_timescale", "240000", str(source)], check=True, capture_output=True, timeout=20)
    info = worker.probe_video(source, ffprobe, inspect_timestamps=True)
    assert info["audio_streams"][0]["sample_count"] == sample_count
    writer = worker.ProgressWriter(tmp_path / "jobs", "fractional-audio", "Movie clock", "test")
    target = tmp_path / "final.mp4"
    worker.final_mp4(source, target, ffmpeg, info, "keep_source_aspect", 64, 48, 17, "ultrafast", writer,
        frame_rate={"mode": "cfr", "fps": rate}, original_info=info)
    final = worker.probe_video(target, ffprobe, inspect_timestamps=True)
    assert final["audio_streams"][0]["sample_count"] == sample_count
    assert final["video_pts_count"] == frames
    assert Fraction(final["fps_ratio"]) == Fraction(rate)
    subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(target), "-f", "null", "-"],
        check=True, capture_output=True, timeout=20)


def test_movie_clock_represents_multiple_audio_rates_and_rejects_unbounded_clock():
    source = {"fps_ratio":"60/1", "audio_streams":[{"sample_rate":48000},{"sample_rate":44100}]}
    assert worker.delivery_movie_timescale(source,frame_rate={"fps":"60/1"}) == 7_056_000
    with pytest.raises(ValueError,match="integer range"):
        worker.delivery_movie_timescale(source,frame_rate={"fps":"2147483647/100000000"})
    with pytest.raises(ValueError,match="audio sample rate"):
        worker.delivery_movie_timescale({**source,"audio_streams":[{"sample_rate":True}]},frame_rate={"fps":"60"})


def test_vfr_and_interlaced_sources_are_not_silently_normalized():
    writer = SimpleNamespace(log=lambda _: None)
    policy = {"mode": "cfr", "fps": 120}
    with pytest.raises(RuntimeError, match="VFR"):
        worker.should_interpolate({"fps": 30, "is_vfr": True}, "4k_60fps", writer, {"tvai_fi": True}, frame_rate=policy)
    info = {"fps": 30, "field_order": "tt"}
    assert not worker.should_interpolate(info, "4k_60fps", writer, {"tvai_fi": True}, frame_rate=policy)
    with pytest.raises(RuntimeError, match="interpolation"):
        worker.require_topaz_interpolation_route(info, "4k_60fps", False, frame_rate=policy)


def test_real_vfr_source_is_detected_and_rejected_for_explicit_cfr(tmp_path):
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    source = tmp_path / "variable_rate_same_headers.mkv"
    subprocess.run([
        ffmpeg, "-hide_banner", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=s=64x48:r=30:d=0.3",
        "-vf", r"settb=1/1000,setpts=N*1000/30+if(gte(N\,4)\,20\,0)",
        "-fps_mode", "vfr", "-c:v", "ffv1", "-an", str(source),
    ], check=True, capture_output=True, timeout=20)
    info = worker.probe_video(source, ffprobe, inspect_timestamps=True)
    assert info["video_frame_count"] == "9"
    assert info["video_r_frame_rate"] == info["video_avg_frame_rate"] == "30/1"
    assert info["video_pts_count"] == 9
    assert info["is_vfr"] is True
    with pytest.raises(RuntimeError, match="VFR"):
        worker.should_interpolate(
            info,
            "4k_60fps",
            SimpleNamespace(log=lambda _: None),
            {"tvai_fi": True},
            frame_rate={"mode": "cfr", "fps": "90/1"},
        )


def test_unqualified_watermark_combination_rejected_before_jobs(tmp_path):
    with pytest.raises(ValueError, match="VideoSeal"):
        pipeline.run_final_delivery([tmp_path / "in.mp4"], jobs_root=tmp_path / "jobs", watermark=True,
            frame_rate={"mode": "cfr", "fps": 90})
    assert not (tmp_path / "jobs").exists()


def test_exact_rate_header_does_not_excuse_changed_duration_or_missing_frames():
    policy = {"mode": "cfr", "fps": 90}
    source = {"duration": 1, "video_duration": "1", "fps": 30, "fps_ratio": "30/1",
        "video_frame_count": "30", "video_pts_count": 30, "video_time_base": "1/30",
        "frame_rate_evidence": "presentation-timestamps", "video_pts_uniform": True}
    good = {"width": 64, "height": 48, "fps": 90, "fps_ratio": "90/1",
        "video_duration": "1", "video_frame_count": "90", "video_time_base": "1/90"}
    worker.validate_final_delivery(good, expected_width=64, expected_height=48,
        preset_id="4k_60fps", frame_rate=policy, source_info=source)
    for broken in [{**good, "video_duration": "0.5"}, {**good, "video_frame_count": "45"}]:
        with pytest.raises(RuntimeError, match="duration|frame count"):
            worker.validate_final_delivery(broken, expected_width=64, expected_height=48,
                preset_id="4k_60fps", frame_rate=policy, source_info=source)


def test_fractional_up_rate_requires_ceil_endpoint_and_exact_video_grid_duration():
    source = {
        "video_duration": "0.300000", "fps_ratio": "30/1", "video_frame_count": "9",
        "video_pts_count": 9, "video_time_base": "1/15360",
        "frame_rate_evidence": "presentation-timestamps", "video_pts_uniform": True,
    }
    policy = {"mode": "cfr", "fps": "60000/1001"}
    good = {
        "video_duration": "0.300300", "video_frame_count": "18",
        "video_time_base": "1/60000", "fps_ratio": "60000/1001",
    }
    worker.validate_cadence_extent(source, good, policy)

    with pytest.raises(RuntimeError, match="expected 18"):
        worker.validate_cadence_extent(source, {
            **good, "video_duration": "0.283617", "video_frame_count": "17",
        }, policy)
    with pytest.raises(RuntimeError, match="duration"):
        worker.validate_cadence_extent(source, {
            **good, "video_duration": "0.300000",
        }, policy)


def test_fractional_up_rate_final_encode_keeps_audio_samples_and_ceil_video_endpoint(tmp_path):
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    source = tmp_path / "source_30fps.mkv"
    subprocess.run([
        ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
        "-f", "lavfi", "-i", "testsrc2=s=64x48:r=30:d=0.3",
        "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=48000:duration=0.3",
        "-map", "0:v:0", "-map", "1:a:0", "-frames:v", "9",
        "-c:v", "ffv1", "-c:a", "pcm_s16le", str(source),
    ], check=True, capture_output=True, timeout=20)
    source_info = worker.probe_video(source, ffprobe, inspect_timestamps=True)
    assert source_info["video_frame_count"] == "9"
    assert source_info["audio_streams"][0]["sample_count"] == 14400

    intermediate = tmp_path / "interpolated_60000_1001.mov"
    subprocess.run([
        ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
        "-f", "lavfi", "-i", "testsrc2=s=64x48:r=60000/1001",
        "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=48000:duration=0.3",
        "-map", "0:v:0", "-map", "1:a:0", "-frames:v", "18",
        "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
        "-c:a", "pcm_s24le", str(intermediate),
    ], check=True, capture_output=True, timeout=20)
    intermediate_info = worker.probe_video(intermediate, ffprobe, inspect_timestamps=True)
    assert intermediate_info["video_frame_count"] == "18"
    assert Fraction(intermediate_info["video_duration"]) == Fraction(3003, 10000)

    output = tmp_path / "fractional_endpoint.mp4"
    writer = worker.ProgressWriter(tmp_path / "jobs", "fractional-endpoint", "Fractional endpoint CPU test", "test")
    policy = {"mode": "cfr", "fps": "60000/1001"}
    worker.final_mp4(
        intermediate, output, ffmpeg, intermediate_info, "keep_source_aspect", 64, 48,
        17, "ultrafast", writer, frame_rate=policy, original_info=source_info,
    )

    output_info = worker.probe_video(output, ffprobe, inspect_timestamps=True)
    assert output_info["video_frame_count"] == "18"
    assert Fraction(output_info["video_duration"]) == Fraction(3003, 10000)
    assert output_info["audio_streams"][0]["sample_count"] == 14400
    worker.validate_intermediate_audio(source_info, output_info, phase="Fractional endpoint CPU output", require_sample_extent=True)
    decoded = subprocess.run([
        ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-",
    ], capture_output=True, text=True, timeout=20)
    assert decoded.returncode == 0, decoded.stderr


def test_source_preserve_rejects_shortened_final_hold_with_identical_pts(tmp_path):
    """Remux only the last packet duration; image PTS and all audio stay intact."""
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    source = tmp_path / "complete.mp4"
    subprocess.run([
        ffmpeg, "-v", "error", "-nostdin",
        "-f", "lavfi", "-i", "testsrc2=s=64x48:r=30:d=0.3",
        "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=48000:duration=0.3",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "ultrafast",
        "-bf", "0", "-c:a", "aac", "-video_track_timescale", "30000", str(source),
    ], check=True, capture_output=True, timeout=20)
    changed = tmp_path / "shortened-last-hold.mp4"
    subprocess.run([
        ffmpeg, "-v", "error", "-nostdin", "-copyts", "-i", str(source),
        "-map", "0", "-c", "copy", "-bsf:v", r"setts=duration=if(eq(N\,8)\,1\,DURATION)",
        "-video_track_timescale", "30000", str(changed),
    ], check=True, capture_output=True, timeout=20)
    original = worker.probe_video(source, ffprobe, inspect_timestamps=True)
    shortened = worker.probe_video(changed, ffprobe, inspect_timestamps=True)
    assert original["video_pts_sha256"] == shortened["video_pts_sha256"]
    assert original["video_pts_count"] == shortened["video_pts_count"] == 9
    assert original["audio_streams"][0]["sample_count"] == shortened["audio_streams"][0]["sample_count"]
    assert Fraction(shortened["video_duration"]) < Fraction(original["video_duration"])
    with pytest.raises(RuntimeError, match="endpoint|last.frame"):
        worker.validate_source_timing_preserved(original, shortened, phase="Final hold regression")


def test_source_preserve_does_not_accept_unknown_or_inconsistent_final_endpoint():
    source = {
        "frame_rate_evidence": "presentation-timestamps", "video_time_base": "1/1000",
        "video_pts_time_base": "1/1000", "video_pts_count": 3, "video_pts_sha256": "a" * 64,
        "video_pts_start": "1/10", "video_pts_end": "167/1000", "video_pts_span": "67/1000",
        "video_pts_last_duration": "33/1000", "video_pts_end_exclusive": "1/5",
        "video_pts_endpoint_evidence": "decoded-last-frame-duration",
    }
    worker.validate_source_timing_preserved(source, dict(source), phase="Complete endpoint")
    for changes in [
        {"video_pts_endpoint_evidence": "unavailable"},
        {"video_pts_last_duration": "0"},
        {"video_pts_end_exclusive": "201/1000"},
    ]:
        with pytest.raises(RuntimeError, match="endpoint|last.frame"):
            worker.validate_source_timing_preserved(source, {**source, **changes}, phase="Invalid endpoint")


def test_interpolation_uses_source_end_context_not_post_render_missing_frame_padding(monkeypatch, tmp_path):
    captured = []
    writer = SimpleNamespace(log=lambda _: None)
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry", lambda argv, *a, **k: captured.append(argv))
    source_info = {"fps": 30.0, "fps_ratio": "30/1", "video_frame_count": "9",
        "video_duration": "0.300000", "video_start_time": "0", "video_time_base": "1/15360"}
    worker.topaz_interpolate_to_prores(tmp_path / "in.mov", tmp_path / "out.mov", "ffmpeg", "90/1", writer,
        source_info=source_info)
    vf = captured[0][captured[0].index("-filter_complex") + 1]
    assert "tpad=stop_mode=clone:stop=1" in vf
    assert vf.index("tpad=") < vf.index("tvai_fi=") < vf.index("trim=end_frame=27")
    assert worker.interpolation_boundary_contract(source_info, "120/1")["trim_end_frame"] == 36
    assert worker.interpolation_boundary_contract(source_info, "60000/1001")["trim_end_frame"] == 18
    for bad in [{**source_info, "video_frame_count": "8"}, {**source_info, "video_start_time": "0.2"},
                {**source_info, "is_vfr": True}]:
        with pytest.raises(ValueError, match="endpoint"):
            worker.interpolation_boundary_contract(bad, "90/1")


def test_explicit_rate_in_old_plan_schema_is_rejected(tmp_path):
    plan = pipeline.build_final_delivery_plan(tmp_path / "in.mp4", tmp_path / "out",
        profile_ids=["smart_finish_master"], frame_rate={"mode": "cfr", "fps": 90})
    plan["schema"] = "spiffy_final_delivery_plan_v1"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf8")
    with pytest.raises(ValueError, match="plan v2"):
        worker.load_delivery_plan(path)


def test_invalid_cadence_candidate_is_not_published(monkeypatch, tmp_path):
    dest = tmp_path / "final.mp4"
    part = worker.part_path(dest)
    writer = SimpleNamespace(log=lambda _: None)
    info = {"width": 64, "height": 48, "fps": 90.0, "fps_ratio": "90/1", "duration": 1.0}
    monkeypatch.setattr(worker, "run_process", lambda *args, **kw: part.write_bytes(b"candidate-not-accepted"))
    monkeypatch.setattr(worker, "probe_video", lambda *args: {**info, "fps": 60, "fps_ratio": "60/1"})
    with pytest.raises(RuntimeError, match="frame rate"):
        worker.final_mp4(tmp_path / "source", dest, "unused", info, "keep_source_aspect", 64, 48,
            17, "fast", writer, frame_rate={"mode": "cfr", "fps": 90})
    assert not dest.exists()
    assert part.read_bytes() == b"candidate-not-accepted"


def test_topaz_upscale_source_preserve_builds_passthrough_and_exact_track_timescale(monkeypatch, tmp_path):
    captured = []
    writer = SimpleNamespace(log=lambda _: None)
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry",
        lambda argv, *args, **kwargs: captured.append(list(argv)))
    info = {
        "duration": 0.3,
        "has_audio": True,
        "video_time_base": "1001/30000",
        "video_pts_time_base": "1001/30000",
        "video_pts_count": 9,
        "video_pts_sha256": "a" * 64,
        "frame_rate_evidence": "presentation-timestamps",
    }
    worker.topaz_upscale_to_prores(tmp_path / "in.mkv", tmp_path / "out.mov", "topaz-ffmpeg",
        info, "auto", 128, 72, writer, preserve_source_timing=True)
    argv = captured[0]
    assert argv.index("-autorotate") < argv.index("-i")
    assert argv.index("-copyts") < argv.index("-i")
    assert argv[argv.index("-fps_mode") + 1] == "passthrough"
    assert argv[argv.index("-video_track_timescale") + 1] == "30000"


def test_source_pts_preserve_final_wrapper_forwards_passthrough_without_fps_cap(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(worker, "final_mp4", lambda *args, **kwargs: captured.update({"args": args, "kwargs": kwargs}))
    source_info = {"video_time_base": "1/30"}
    writer = SimpleNamespace(log=lambda _: None)
    common = (
        tmp_path / "source.mov", tmp_path / "identity.mov", tmp_path / "out.mp4", "ffmpeg", "ffprobe",
        source_info, "keep_source_aspect", 64, 48, 17, "ultrafast", writer,
    )
    worker.final_mp4_with_optional_watermark(
        *common, watermark_enabled=False, watermark_strength=0.2,
        original_info=source_info, preserve_source_timing=True,
    )
    assert captured["args"][10] is None
    assert captured["kwargs"]["preserve_source_timing"] is True
    with pytest.raises(ValueError, match="Source-PTS preservation"):
        worker.final_mp4_with_optional_watermark(
            *common, watermark_enabled=False, watermark_strength=0.2,
            original_info=source_info, max_fps=60, preserve_source_timing=True,
        )


def test_topaz_upscale_source_preserve_fails_closed_without_pts_evidence(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry",
        lambda *args, **kwargs: called.append(True))
    with pytest.raises(RuntimeError, match="presentation timestamp evidence"):
        worker.topaz_upscale_to_prores(tmp_path / "in.mkv", tmp_path / "out.mov", "topaz-ffmpeg",
            {"duration": 0.3, "video_time_base": "1/1000"}, "auto", 128, 72,
            SimpleNamespace(log=lambda _: None), preserve_source_timing=True)
    assert not called


def test_source_timing_validation_compares_exact_pts_across_different_time_bases():
    source = {
        "frame_rate_evidence": "presentation-timestamps", "video_time_base": "1/1000",
        "video_pts_time_base": "1/1000", "video_pts_count": 9, "video_pts_sha256": "same",
        "video_pts_start": "0", "video_pts_end": "3/10", "video_pts_span": "3/10",
        "video_pts_last_duration": "33/1000", "video_pts_end_exclusive": "333/1000",
        "video_pts_endpoint_evidence": "decoded-last-frame-duration",
    }
    candidate = {
        **source, "video_time_base": "1/30000", "video_pts_time_base": "1/30000",
    }
    worker.validate_source_timing_preserved(source, candidate, phase="test")
    with pytest.raises(RuntimeError, match="timestamps changed"):
        worker.validate_source_timing_preserved(source, {**candidate, "video_pts_sha256": "different"}, phase="test")


@pytest.mark.skipif(not worker.standard_ffmpeg() or not worker.standard_ffprobe(), reason="local FFmpeg tools unavailable")
def test_final_mp4_source_preserve_keeps_vfr_pts_and_audio_extent(tmp_path):
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    source = tmp_path / "vfr_source_with_audio.mkv"
    subprocess.run([
        ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
        "-f", "lavfi", "-i", "testsrc2=s=64x48:r=30:d=0.3",
        "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=48000:duration=0.3",
        "-vf", r"settb=1/1000,setpts=N*1000/30+if(gte(N\,4)\,20\,0)+100",
        "-fps_mode", "vfr", "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "ffv1", "-c:a", "pcm_s16le", str(source),
    ], check=True, capture_output=True, timeout=20)
    source_info = worker.probe_video(source, ffprobe, inspect_timestamps=True)
    assert source_info["is_vfr"] is True
    assert Fraction(source_info["video_pts_start"]) > 0
    assert source_info["audio_streams"][0]["sample_count"] == 14400

    dest = tmp_path / "vfr_preserved.mp4"
    writer = worker.ProgressWriter(tmp_path / "jobs", "source-preserve", "Source PTS preservation CPU test", "test")
    worker.final_mp4(source, dest, ffmpeg, source_info, "keep_source_aspect", 64, 48, 17, "ultrafast",
        writer, duration_s=float(source_info["duration"]), original_info=source_info,
        preserve_source_timing=True)

    output_info = worker.probe_video(dest, ffprobe, inspect_timestamps=True)
    worker.validate_source_timing_preserved(source_info, output_info, phase="CPU H264 regression")
    assert output_info["is_vfr"] is True
    assert output_info["video_pts_count"] == source_info["video_pts_count"] == 9
    assert output_info["audio_streams"][0]["sample_count"] == 14400
    decoded = subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-i", str(dest), "-f", "null", "-"],
        capture_output=True, text=True, timeout=20)
    assert decoded.returncode == 0, decoded.stderr
