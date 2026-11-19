"""Connected adaptive enhancement tests; native Topaz is never impersonated."""
from __future__ import annotations

import copy
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import workflow.final_delivery_pipeline as pipeline
import video_upscale as worker
from tools import run_spiffy_final_delivery as runner


def adaptive():
    import adaptive_enhancement
    return adaptive_enhancement


def detail_fixture():
    image = np.full((288, 512, 3), 110, dtype=np.uint8)
    yy, xx = np.mgrid[:96, :128]
    texture = 110 + 15 * np.sin(xx * 1.1) + 12 * np.cos(yy * 0.7)
    image[80:176, 192:320] = np.repeat(texture.astype(np.uint8)[..., None], 3, axis=2)
    return image


@pytest.mark.parametrize("value", [None, True, [], {}, {"mode": "anything"},
    {"mode": "natural", "cloud": True}, {"mode": "natural", "strength": float("nan")}])
def test_policy_rejects_malformed_or_unimplemented_options(value):
    with pytest.raises(ValueError, match="adaptive"):
        adaptive().normalize_policy(value)


def test_transparent_motion_graphics_fail_before_adaptive_frame_analysis(tmp_path):
    source = tmp_path / "transparent_graphics.mov"
    evidence = {
        "width": 512, "height": 288, "video_frame_count": "24",
        "fps_ratio": "24/1", "duration": 1.0,
        "pix_fmt": "yuva444p10le", "has_alpha": True,
        "sar": "1:1", "rotation": 0, "color_transfer": "bt709",
        "field_order": "progressive", "is_vfr": False,
    }
    with pytest.raises(ValueError, match="alpha"):
        adaptive().prepare(
            source, evidence, 1024, 576,
            SimpleNamespace(cancel_requested=False),
            {"mode": "natural", "subject_refinement": "off"})


def test_adaptive_request_cli_and_worker_plan_are_connected(tmp_path):
    policy = {"mode": "natural"}
    options = pipeline.request_delivery_options({"final_delivery": {
        "delivery_set": "smart_finish", "adaptive_enhancement": policy}})
    assert options["adaptive_enhancement"] == policy
    plan = pipeline.build_final_delivery_plan(tmp_path / "input.mp4", tmp_path / "out",
        profile_ids=options["profile_ids"], adaptive_enhancement=policy)
    assert plan["schema"] == "spiffy_final_delivery_plan_v3"
    saved = tmp_path / "plan.json"
    saved.write_text(json.dumps(plan), encoding="utf8")
    assert worker.load_delivery_plan(saved)["adaptive_enhancement"] == policy
    assert pipeline._plan_fingerprint(plan) != pipeline._plan_fingerprint(
        pipeline.build_final_delivery_plan(tmp_path / "input.mp4", tmp_path / "out",
            profile_ids=options["profile_ids"]))
    assert runner.main(["--input", str(tmp_path / "input.mp4"), "--jobs-root", str(tmp_path / "jobs"),
        "--job-id", "adaptive-plan", "--delivery-set", "smart_finish", "--adaptive-enhancement"]) == 0
    result = json.loads((tmp_path / "jobs/adaptive-plan/final_delivery_result.json").read_text())
    assert worker.load_delivery_plan(result["sources"][0]["plan_path"])["adaptive_enhancement"] == policy
    old = copy.deepcopy(plan)
    old["schema"] = "spiffy_final_delivery_plan_v2"
    saved.write_text(json.dumps(old), encoding="utf8")
    with pytest.raises(ValueError, match="v3"):
        worker.load_delivery_plan(saved)


def test_measured_mask_leaves_flat_background_and_hard_caption_edges_alone():
    frame = detail_fixture()
    mask, evidence = adaptive().detail_protection_mask(frame)
    assert mask.shape == frame.shape[:2]
    assert mask.dtype == np.uint8
    assert mask[:50].max() == 0
    assert mask[95:155, 210:295].mean() > 1
    assert mask.max() <= 90
    frame[230:250, 125:380] = 250
    caption_mask, _ = adaptive().detail_protection_mask(frame)
    assert caption_mask[228:253, 123:382].max() == 0
    assert evidence["semantic_caption_detection"] is False
    assert adaptive().detail_protection_mask(np.zeros_like(frame))[0].max() == 0


def test_crop_is_padded_stable_and_not_used_with_fractional_scale():
    union = np.zeros((288, 512), dtype=np.uint8)
    union[80:176, 192:320] = 80
    crop = adaptive().choose_processing_crop(union, 1024, 576)
    x, y, w, h = crop
    assert x < 192 and y < 80 and x + w > 320 and y + h > 176
    assert (w * h) / (512 * 288) < 0.7
    assert all(value % 2 == 0 for value in crop)
    assert adaptive().choose_processing_crop(union, 1000, 562) == [0, 0, 512, 288]


def test_crop_graph_reduces_model_input_before_enhancement():
    plan = {"schema": adaptive().SCHEMA, "route": "masked_crop", "source_width": 512,
        "source_height": 288, "target_width": 1024, "target_height": 576,
        "crop": [160, 48, 192, 160], "effective_params": ":details=0.1:blend=0.5"}
    graph = adaptive().enhancement_filter_graph(plan, "prob-4", "-2")
    assert graph.index("crop=") < graph.index("tvai_up=")
    assert "w=384:h=320" in graph
    assert "maskedmerge" in graph
    assert "download=0" in graph


@pytest.mark.parametrize("source_size,target_size,region", [
    ((512, 288), (768, 432), (192, 80, 320, 176)),
    ((1080, 1350), (3072, 3840), (390, 480, 630, 780)),
])
def test_rational_crop_keeps_source_and_destination_on_the_same_chroma_grid(source_size, target_size, region):
    width, height = source_size
    target_width, target_height = target_size
    union = np.zeros((height, width), dtype=np.uint8)
    left, top, right, bottom = region
    union[top:bottom, left:right] = 80
    crop = adaptive().choose_processing_crop(union, target_width, target_height)
    x, y, w, h = crop
    assert w * h < width * height * 0.7
    assert x <= left - 64 and y <= top - 64
    assert x + w >= right + 64 and y + h >= bottom + 64
    scale = Fraction(target_width, width)
    assert scale == Fraction(target_height, height)
    assert all(v % 2 == 0 and (Fraction(v) * scale).denominator == 1
               and int(Fraction(v) * scale) % 2 == 0 for v in crop)
    plan = {"schema": adaptive().SCHEMA, "route": "masked_crop",
        "source_width": width, "source_height": height,
        "target_width": target_width, "target_height": target_height,
        "crop": crop, "effective_params": ":details=0.1:blend=0.5"}
    graph = adaptive().enhancement_filter_graph(plan, "prob-4", "-2")
    assert f"w={int(w * scale)}:h={int(h * scale)}" in graph
    assert f"pad={target_width}:{target_height}:{int(x * scale)}:{int(y * scale)}" in graph


@pytest.mark.parametrize("crop", [[162, 48, 192, 160], [-4, 48, 192, 160], [160, 48, 800, 160]])
def test_rational_crop_graph_rejects_misalignment_or_out_of_frame_regions(crop):
    plan = {"schema": adaptive().SCHEMA, "route": "masked_crop", "source_width": 512,
        "source_height": 288, "target_width": 768, "target_height": 432,
        "crop": crop, "effective_params": ":details=0.1:blend=0.5"}
    with pytest.raises(ValueError, match="crop|aligned|geometry"):
        adaptive().enhancement_filter_graph(plan, "prob-4", "-2")


def test_effective_controls_are_measured_but_do_not_claim_ground_truth():
    clean = adaptive().choose_natural_controls([{"sharpness": 0.7, "blockiness": 0.0, "texture_residual": 9.0}])
    damaged = adaptive().choose_natural_controls([{"sharpness": 0.1, "blockiness": 9.0, "texture_residual": 1.0}])
    assert clean != damaged
    assert clean["noise"] <= 0 and damaged["noise"] <= 0
    assert clean["details"] <= 0.2 and damaged["details"] <= 0.2
    assert clean["blend"] >= 0.5


def test_mask_analysis_generates_real_frame_aligned_cache_and_reuses_it(tmp_path):
    import av
    source = tmp_path / "detail.mov"
    with av.open(str(source), "w") as output:
        stream = output.add_stream("png", rate=30)
        stream.width, stream.height, stream.pix_fmt = 512, 288, "rgb24"
        for index in range(6):
            frame = av.VideoFrame.from_ndarray(detail_fixture(), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    info = worker.probe_video(source, worker.standard_ffprobe(), inspect_timestamps=True)
    writer = worker.ProgressWriter(tmp_path / "jobs", "adaptive", "Adaptive test", "test")
    first = adaptive().prepare(source, info, 1024, 576, writer, {"mode": "natural"})
    assert first["route"] == "masked_crop"
    assert first["frame_count"] == 6
    assert first["processed_area_ratio"] < 0.7
    assert Path(first["mask_path"]).is_file()
    with av.open(first["mask_path"]) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 6
    assert [Fraction(f.pts) * f.time_base for f in frames] == [Fraction(i, 30) for i in range(6)]
    assert adaptive().prepare(source, info, 1024, 576, writer, {"mode": "natural"})["cache_hit"] is True


def test_requested_adaptive_mode_is_not_silently_lost_in_reuse(tmp_path):
    kwargs = dict(stage="topaz_upscale", input_path=tmp_path / "source", topaz_ffmpeg=tmp_path / "runtime",
        preset_id="auto", model="prob-4", target_width=128, target_height=72)
    kwargs["input_path"].write_bytes(b"source")
    kwargs["topaz_ffmpeg"].write_bytes(b"runtime")
    original = worker.build_topaz_checkpoint_fingerprint(**kwargs)
    changed = worker.build_topaz_checkpoint_fingerprint(**kwargs, adaptive_fingerprint="a" * 64)
    assert original["fingerprint"] != changed["fingerprint"]


def write_fixture(path, frames=6):
    import av
    with av.open(str(path), "w") as output:
        stream = output.add_stream("png", rate=30)
        stream.width, stream.height, stream.pix_fmt = 512, 288, "rgb24"
        for index in range(frames):
            rgb = detail_fixture()
            rgb[230:250, 120 + index:370 + index] = 250
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def test_real_cpu_compositor_preserves_masked_pixels_and_frame_alignment(tmp_path):
    """Use a conspicuously wrong black detail layer, never call it Topaz proof."""
    import av
    import re
    source = tmp_path / "input.mov"
    write_fixture(source)
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    info = worker.probe_video(source, ffprobe, inspect_timestamps=True)
    writer = worker.ProgressWriter(tmp_path / "jobs", "compositor", "CPU mask graph", "test")
    plan = adaptive().prepare(source, info, 1024, 576, writer, {"mode": "natural"})
    assert plan["route"] == "masked_crop"
    native_graph = adaptive().enhancement_filter_graph(plan, "prob-4", "-2")
    graph = re.sub(r"tvai_up=[^,;]+", "lutrgb=r=0:g=0:b=0", native_graph)
    assert "tvai_up" not in graph
    output = tmp_path / "composite.nut"
    result = subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-i", str(source),
        "-i", plan["mask_path"], "-filter_complex", graph, "-map", "[vout]", "-an",
        "-c:v", "ffv1", "-pix_fmt", "gbrp16le", str(output)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    base = tmp_path / "base.nut"
    subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-i", str(source),
        "-vf", "scale=1024:576:flags=lanczos,setsar=1,format=gbrp16le", "-an",
        "-c:v", "ffv1", "-pix_fmt", "gbrp16le", str(base)], check=True, capture_output=True, timeout=30)
    with av.open(str(output)) as container:
        actual = [frame.to_ndarray(format="rgb48le") for frame in container.decode(video=0)]
    with av.open(str(base)) as container:
        reference = [frame.to_ndarray(format="rgb48le") for frame in container.decode(video=0)]
    assert len(actual) == len(reference) == 6
    for index, (a, b) in enumerate(zip(actual, reference)):
        np.testing.assert_array_equal(a[:120], b[:120])
        np.testing.assert_array_equal(a[456:505, (120 + index) * 2:(370 + index) * 2],
                                      b[456:505, (120 + index) * 2:(370 + index) * 2])
        assert np.mean(np.abs(a[200:300, 420:570].astype(float) - b[200:300, 420:570])) > 1
    checked = worker.probe_video(output, ffprobe, inspect_timestamps=True)
    assert checked["video_pts_sha256"] == info["video_pts_sha256"]


def test_rational_crop_cpu_resampling_matches_full_frame_phase_and_preserves_captions(tmp_path):
    """Actual fractional crop/composite decode; no claim of native model parity."""
    import av
    import re
    source = tmp_path / "rational_source.mov"
    write_fixture(source, 3)
    info = worker.probe_video(source, worker.standard_ffprobe(), inspect_timestamps=True)
    writer = worker.ProgressWriter(tmp_path / "jobs", "rational", "Rational crop proof", "test")
    plan = adaptive().prepare(source, info, 768, 432, writer, {"mode": "natural"})
    assert plan["route"] == "masked_crop"
    full_plan = {**plan, "route": "masked_full_frame", "crop": [0, 0, 512, 288]}
    rendered = []
    for label, candidate in [("cropped", plan), ("full", full_plan)]:
        graph = re.sub(r"tvai_up=[^,;]+", "null", adaptive().enhancement_filter_graph(candidate, "prob-4", "-2"))
        destination = tmp_path / f"{label}.nut"
        run = subprocess.run([worker.standard_ffmpeg(), "-hide_banner", "-v", "error", "-nostdin",
            "-i", str(source), "-i", plan["mask_path"], "-filter_complex", graph,
            "-map", "[vout]", "-an", "-c:v", "ffv1", "-pix_fmt", "gbrp16le", str(destination)],
            capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stderr
        with av.open(str(destination)) as decoded:
            rendered.append([(Fraction(frame.pts) * frame.time_base, frame.to_ndarray(format="rgb48le"))
                             for frame in decoded.decode(video=0)])
    assert len(rendered[0]) == len(rendered[1]) == 3
    for index, ((time_a, image_a), (time_b, image_b)) in enumerate(zip(*rendered)):
        assert time_a == time_b == Fraction(index, 30)
        delta = np.abs(image_a.astype(np.int32) - image_b.astype(np.int32))
        # Different scaler widths use different fixed-point rounding paths.
        # Their native 16-bit result need not be byte-identical: require less
        # than one 16-bit code RMS and at most one delivered 8-bit code, while
        # protected lettering/background remain exactly unchanged. A phase
        # shift cannot pass this on the high-frequency detail fixture.
        assert np.mean(delta.astype(np.float64) ** 2) < 1.0
        delivered_a = np.rint(image_a.astype(np.float64) / 257)
        delivered_b = np.rint(image_b.astype(np.float64) / 257)
        assert np.max(np.abs(delivered_a - delivered_b)) <= 1
        np.testing.assert_array_equal(image_a[:100], image_b[:100])
        np.testing.assert_array_equal(image_a[342:380], image_b[342:380])


def test_adaptive_mask_tampering_is_rejected(tmp_path):
    source = tmp_path / "input.mov"
    write_fixture(source, 2)
    info = worker.probe_video(source, worker.standard_ffprobe(), inspect_timestamps=True)
    writer = worker.ProgressWriter(tmp_path / "jobs", "tamper", "Mask custody", "test")
    plan = adaptive().prepare(source, info, 1024, 576, writer, {"mode": "natural"})
    adaptive().verify_artifacts(plan)
    with Path(plan["mask_path"]).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(RuntimeError, match="mask"):
        adaptive().verify_artifacts(plan)


def test_explicit_adaptive_conflicts_fail_before_creating_jobs(tmp_path):
    for kwargs in [{"watermark": True}, {"enhancement": {"preset_id": "auto", "overrides": {"noise": 0.1}}}]:
        with pytest.raises(ValueError, match="Adaptive"):
            pipeline.run_final_delivery([tmp_path / "pending.mov"], jobs_root=tmp_path / "jobs",
                adaptive_enhancement={"mode": "natural"}, **kwargs)
        assert not (tmp_path / "jobs").exists()


def test_adaptive_cancel_prevents_analysis_and_publication(tmp_path):
    from shared.gpu_lock import GpuLockCancelled
    writer = SimpleNamespace(cancel_requested=True, temp_dir=tmp_path / "temp")
    with pytest.raises(GpuLockCancelled):
        adaptive().prepare(tmp_path / "absent.mov", {}, 128, 72, writer, {"mode": "natural"})
    assert not writer.temp_dir.exists()


def test_objective_guard_rejects_a_changed_grade_without_claiming_visual_acceptance(tmp_path):
    source = tmp_path / "input.mov"
    write_fixture(source, 3)
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    info = worker.probe_video(source, ffprobe, inspect_timestamps=True)
    writer = worker.ProgressWriter(tmp_path / "jobs", "quality", "Quality gate", "test")
    plan = adaptive().prepare(source, info, 1024, 576, writer, {"mode": "natural"})
    for label, vf, expected in [
        ("preservation", "scale=1024:576:flags=lanczos", "PASS_OBJECTIVE_GUARDS"),
        ("damaged", "eq=brightness=0.3,scale=1024:576:flags=lanczos", "REJECT"),
    ]:
        candidate = tmp_path / f"{label}.mov"
        subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", str(source), "-vf", vf,
            "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-an", str(candidate)],
            check=True, capture_output=True, timeout=30)
        result = adaptive().assess_candidate(source, candidate, plan, writer)
        assert result["status"] == expected
        assert result["frames_decoded"] == 3
        assert result["visual_acceptance"] == "not_established_by_objective_metrics"


def test_adaptive_plan_pins_engine_revision_and_rejects_source_drift(tmp_path):
    plan = pipeline.build_final_delivery_plan(tmp_path / "source.mov", tmp_path / "out",
        profile_ids=["smart_finish_master"], adaptive_enhancement={"mode": "natural"})
    assert plan["adaptive_engine_identity"] == adaptive().engine_identity()
    path = tmp_path / "plan.json"
    plan["adaptive_engine_identity"]["adapter_sha256"] = "0" * 64
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="engine changed"):
        worker.load_delivery_plan(path)


def test_flat_content_avoids_enhancement_model_execution(tmp_path, monkeypatch):
    import av
    source = tmp_path / "flat.mov"
    with av.open(str(source), "w") as output:
        stream = output.add_stream("png", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "rgb24"
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 112, np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    info = worker.probe_video(source, worker.standard_ffprobe(), inspect_timestamps=True)
    writer = worker.ProgressWriter(tmp_path / "jobs", "flat", "No useful detail", "test")
    plan = adaptive().prepare(source, info, 128, 96, writer, {"mode": "natural"})
    assert plan["route"] == "source_preserved"
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry",
        lambda *a, **k: pytest.fail("No enhancement model should execute for an all-protected clip"))
    dest = tmp_path / "preserved.mov"
    worker.topaz_upscale_to_prores(source, dest, "must-not-execute", info, "auto", 128, 96, writer,
                                  adaptive_plan=plan)
    result = worker.probe_video(dest, worker.standard_ffprobe(), inspect_timestamps=True)
    assert result["video_pts_count"] == 3
    assert (result["width"], result["height"]) == (128, 96)


def test_parent_worker_quality_failure_uses_real_cpu_preservation_not_a_fake_success(tmp_path, monkeypatch):
    """Controlled bad model output; real owner, compositor, fallback and encoding.

    The injected candidate is deliberately generated by standard FFmpeg. This
    test claims no Topaz visual fidelity; separate native receipts cover that.
    """
    import hashlib
    import sys
    source = tmp_path / "input.mov"
    write_fixture(source, 3)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    info = worker.probe_video(source, ffprobe, inspect_timestamps=True)
    runtime = tmp_path / "identity_only_topaz.exe"
    runtime.write_bytes(b"contract fixture, never executed")
    monkeypatch.setattr(worker, "find_topaz_video", lambda: {"ffmpeg": str(runtime), "ffprobe": ffprobe,
        "tvai_up": True, "tvai_fi": True, "model_dir": str(tmp_path)})
    monkeypatch.setattr(worker, "configure_topaz_model_environment", lambda _p: str(tmp_path))
    monkeypatch.setattr(worker, "wait_for_render_ready", lambda *a: (source, info))
    monkeypatch.setattr(worker, "topaz_model_asset_identity", lambda *a: {
        "resume_safe": True, "identity_sha256": "f" * 64, "descriptor_present": True,
        "weight_or_engine_count": 1, "asset_files": [], "model": "prob-4"})
    native_attempts = []
    def damaged_candidate(src, dest, executable, src_info, preset, width, height, writer, *args, **kwargs):
        native_attempts.append(kwargs["adaptive_plan"]["fingerprint"])
        subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-i", str(src),
            "-vf", f"eq=brightness=0.35,scale={width}:{height}:flags=lanczos,setsar=1",
            "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-an", str(dest)],
            check=True, capture_output=True, timeout=30)
        return False
    monkeypatch.setattr(worker, "topaz_upscale_to_prores", damaged_candidate)
    def in_process_worker(command, child_dir, *args, **kwargs):
        with monkeypatch.context() as scoped:
            scoped.setattr(sys, "argv", command[1:])
            return worker.main()
    monkeypatch.setattr(pipeline, "_run_child", in_process_worker)
    code, result = pipeline.run_final_delivery([source], jobs_root=tmp_path / "jobs", job_id="fallback",
        output_root=tmp_path / "out", profile_ids=["smart_finish_master"],
        frame_rate={"mode": "cfr", "fps": 30}, adaptive_enhancement={"mode": "natural"}, execute=True)
    assert code == 0, result.get("errors")
    assert len(native_attempts) == 1
    assert result["state"] == "passed"
    assert result["performance"]["schema"] == "spiffy_delivery_performance_v1"
    assert Path(result["performance_report_path"]).is_file()
    assert result["performance"]["summary"]["elapsed_seconds"] > 0
    proof = result["final_output_proofs"][0]
    assert proof["adaptive_actual_route"] == "source_preserved"
    assert proof["adaptive_quality_status"] == "SOURCE_PRESERVATION_FALLBACK"
    child = json.loads(Path(result["sources"][0]["child_manifest_path"]).read_text())
    quality = next(iter(child["adaptive_quality"].values()))
    assert quality["rejected_candidate"]["status"] == "REJECT"
    assert quality["fallback"]["status"] == "PASS_OBJECTIVE_GUARDS"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    final = Path(result["final_outputs"][0])
    subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(final), "-f", "null", "-"],
        check=True, capture_output=True, timeout=20)
    assert pipeline.validate_final_delivery_result(tmp_path / "jobs/fallback")["status"] == "PASS"


def test_shared_scene_windows_cover_exact_frame_extent_not_truncated_milliseconds():
    shots = [{"start_ms": 0, "end_ms": 750}, {"start_ms": 750, "end_ms": 1000}]
    windows = adaptive().scene_frame_windows(shots, Fraction(24000, 1001), 24)
    assert windows == [(0, 18), (18, 24)]
    assert adaptive().scene_frame_windows([], Fraction(30), 90) == [(0, 90)]
    assert adaptive().scene_frame_windows([{"start_ms": 2000}], Fraction(30), 30) == [(0, 30)]


def test_shot_selection_skips_flat_intervals_without_extra_model_startup():
    controls = adaptive().choose_natural_controls([])
    raw = [
        {"start_frame": 0, "end_frame": 60, "crop": [0, 0, 512, 288],
         "route": "source_preserved", "effective_controls": controls},
        {"start_frame": 60, "end_frame": 120, "crop": [160, 48, 192, 160],
         "route": "masked_crop", "effective_controls": controls},
    ]
    units, reason = adaptive().select_render_units(raw, 512, 288, 30)
    assert len(units) == 2
    assert reason == "skip_protected_intervals"
    assert sum(u["end_frame"] - u["start_frame"] for u in units if u["route"] != "source_preserved") == 60


def test_short_busy_shots_are_batched_instead_of_reloading_model_per_cut():
    controls = adaptive().choose_natural_controls([])
    shots = [{"start_frame": i * 10, "end_frame": (i + 1) * 10,
              "crop": [0, 0, 512, 288], "route": "masked_full_frame",
              "effective_controls": {**controls, "details": 0.08 + 0.01 * i}}
             for i in range(8)]
    units, reason = adaptive().select_render_units(shots, 512, 288, 30)
    assert units == []
    assert reason == "batch_short_or_similar_shots"


def test_temporal_window_is_trimmed_before_model_and_uses_exact_ticks():
    plan = {"schema": adaptive().SCHEMA, "route": "masked_crop", "source_width": 512,
        "source_height": 288, "target_width": 1024, "target_height": 576,
        "crop": [160, 48, 192, 160], "effective_params": ":details=0.1:blend=0.5",
        "source_fps": "24000/1001", "execution_window": {"start_frame": 18, "end_frame": 48}}
    graph = adaptive().enhancement_filter_graph(plan, "prob-4", "-2")
    assert "settb=expr=1/24000,trim=start_pts=18018:end_pts=48048,setpts=PTS-STARTPTS" in graph
    assert graph.index("trim=") < graph.index("tvai_up=")
    assert graph.count("trim=start_pts=18018:end_pts=48048") == 2


def test_scene_analysis_consumes_existing_owner_and_keeps_shot_controls(tmp_path, monkeypatch):
    import av
    import scene.scene_detection_engine as scenes
    source = tmp_path / "two_shots.mov"
    with av.open(str(source), "w") as output:
        stream = output.add_stream("png", rate=30)
        stream.width, stream.height, stream.pix_fmt = 512, 288, "rgb24"
        for i in range(60):
            image = np.full((288, 512, 3), 110, dtype=np.uint8) if i < 30 else detail_fixture()
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 30)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    observed = []
    def existing_owner(path, **kwargs):
        observed.append((path, kwargs))
        return SimpleNamespace(shots=[{"start_ms": 0}, {"start_ms": 1000}],
                               strategy="v2", detector_version="test-owner", metadata={})
    monkeypatch.setattr(scenes, "detect_scene_shots", existing_owner)
    info = worker.probe_video(source, worker.standard_ffprobe(), inspect_timestamps=True)
    writer = worker.ProgressWriter(tmp_path / "jobs", "shots", "Shot adaptive", "test")
    plan = adaptive().prepare(source, info, 1024, 576, writer, {"mode": "natural"})
    assert len(observed) == 1
    assert plan["scene_analysis"]["owner"] == "scene.scene_detection_engine"
    assert len(plan["shots"]) == 2
    assert plan["shots"][0]["route"] == "source_preserved"
    assert plan["shots"][1]["route"] == "masked_crop"
    assert plan["route"] == "shot_sequence"
    assert plan["processed_area_ratio"] < 0.4
    assert len(plan["render_units"]) == 2

    # Exercise actual temporal trims, compositing, ProRes assembly and audio-free
    # output using a CPU identity detail stand-in; this is NOT Topaz inference.
    import re
    native_calls = []
    def cpu_detail(argv, log_writer, phase, **kwargs):
        native_calls.append(list(argv))
        cpu_argv = list(argv)
        idx = cpu_argv.index("-filter_complex") + 1
        cpu_argv[idx] = re.sub(r"tvai_up=[^,;]+", "null", cpu_argv[idx])
        worker.run_process(cpu_argv, log_writer, "cpu_identity_detail", timeout=30)
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry", cpu_detail)
    result_path = tmp_path / "shot_sequence.mov"
    worker.topaz_upscale_to_prores(source, result_path, worker.standard_ffmpeg(), info,
                                  "auto", 1024, 576, writer, adaptive_plan=plan)
    result_info = worker.probe_video(result_path, worker.standard_ffprobe(), inspect_timestamps=True)
    worker.validate_source_timing_preserved(info, result_info, phase="CPU shot assembly")
    assert result_info["video_pts_count"] == 60
    assert len(native_calls) == 1  # the flat first shot must never initialize AI
    filtergraph = native_calls[0][native_calls[0].index("-filter_complex") + 1]
    assert "trim=start_pts=30:end_pts=60" in filtergraph
    assert not list(writer.temp_dir.glob("*_shot_*.mov"))


def test_render_unit_plan_rejects_gaps_and_overlaps(tmp_path):
    source = tmp_path / "source.mov"
    write_fixture(source, 3)
    info = worker.probe_video(source, worker.standard_ffprobe(), inspect_timestamps=True)
    writer = worker.ProgressWriter(tmp_path / "jobs", "bounds", "Unit contract", "test")
    plan = adaptive().prepare(source, info, 1024, 576, writer, {"mode": "natural"})
    unit = {"crop": plan["crop"], "route": "masked_crop", "effective_controls": plan["effective_controls"]}
    plan.update(route="shot_sequence", render_units=[
        {**unit, "start_frame": 0, "end_frame": 1}, {**unit, "start_frame": 2, "end_frame": 3}])
    plan["fingerprint"] = adaptive().plan_fingerprint(plan)
    with pytest.raises(ValueError, match="gaps or overlaps"):
        adaptive().render_unit_plans(plan)


@pytest.mark.parametrize("edge", [1024, 1920, 2560, 3072, 3840])
def test_explicit_working_size_is_independent_of_final_source_canvas(tmp_path, edge):
    policy = {"mode": "natural", "working_long_edge": edge}
    plan = pipeline.build_final_delivery_plan(tmp_path / "input.mp4", tmp_path / "out",
        profile_ids=["smart_finish_master"], adaptive_enhancement=policy)
    info = {"width": 1920, "height": 1080, "sar": "1:1", "rotation": 0}
    tw, th, fw, fh = worker.target_dimensions_for_delivery(info, "4k_60fps", "keep_source_aspect", plan)
    assert max(tw, th) == max(1920, edge)
    assert (fw, fh) == (tw, th)
    assert plan["variants"][0]["geometry_policy"] == "source_display"
    assert plan["variants"][0]["width"] == plan["variants"][0]["height"] == 0


@pytest.mark.parametrize("edge", [True, "3840", None, -1, 127, 8194, 1025])
def test_bad_working_size_is_rejected_before_execution(edge):
    with pytest.raises(ValueError, match="adaptive"):
        adaptive().normalize_policy({"mode": "natural", "working_long_edge": edge})


def test_native_master_delivers_the_requested_adaptive_working_canvas(tmp_path):
    plan = pipeline.build_final_delivery_plan(tmp_path / "in.mov", tmp_path / "out",
        profile_ids=["native_master"], adaptive_enhancement={"mode": "natural", "working_long_edge": 3072})
    args = worker.parse_args(["--input-list", str(tmp_path / "inputs.txt"), "--watermark", "off", "--preset-id", "4k_60fps"])
    info = {"width": 1920, "height": 1080, "fps": 60, "fps_ratio": "60/1", "sar": "1:1", "rotation": 0}
    variants = worker.delivery_variants_for_source(tmp_path / "in.mov", info, args, tmp_path / "out", plan)
    assert (variants[0]["width"], variants[0]["height"]) == (3072, 1728)


def test_optional_statistics_do_not_change_dense_protection_pixels():
    measured, evidence = adaptive().detail_protection_mask(detail_fixture())
    fast, omitted = adaptive().detail_protection_mask(detail_fixture(), collect_metrics=False)
    np.testing.assert_array_equal(fast, measured)
    assert evidence and omitted == {}


def test_scene_cache_location_and_hit_do_not_invalidate_identical_render_work():
    plan = {"schema": adaptive().SCHEMA, "source_sha256": "a" * 64,
            "scene_analysis": {"owner": "scene.scene_detection_engine", "strategy": "v2",
                "cache": {"hit": False, "path": "old_job/scenes.json", "elapsed_s": 2.0}},
            "shots": [{"start_frame": 0, "end_frame": 24}]}
    other = copy.deepcopy(plan)
    other["scene_analysis"]["cache"] = {"hit": True, "path": "retry_job/scenes.json", "elapsed_s": 0.01}
    assert adaptive().plan_fingerprint(plan) == adaptive().plan_fingerprint(other)
    other["shots"][0]["end_frame"] = 25
    assert adaptive().plan_fingerprint(plan) != adaptive().plan_fingerprint(other)


def test_cli_working_long_edge_is_saved_in_existing_request_path(tmp_path):
    assert runner.main(["--input", str(tmp_path / "pending.mov"), "--jobs-root", str(tmp_path / "jobs"),
        "--job-id", "adaptive-size", "--working-long-edge", "3072", "--delivery-set", "smart_finish"]) == 0
    saved = json.loads((tmp_path / "jobs/adaptive-size/final_delivery_result.json").read_text())
    plan = worker.load_delivery_plan(saved["sources"][0]["plan_path"])
    assert plan["adaptive_enhancement"] == {"mode": "natural", "working_long_edge": 3072}


def test_interpolation_uses_required_delivery_canvas_without_later_enlargement():
    plan = {"target_width": 3840, "target_height": 2160, "policy": {"mode": "natural"}}
    variants = [{"width": 1920, "height": 1080, "layout": "fit_into_frame"}]
    assert adaptive().interpolation_geometry(plan, variants) == (1920, 1080)
    assert adaptive().interpolation_geometry(plan, variants + [
        {"width": 3840, "height": 2160, "layout": "keep_source_aspect"}]) == (3840, 2160)
    assert adaptive().interpolation_geometry({**plan, "policy": {"mode": "natural", "interpolation_resolution": "working"}}, variants) == (3840, 2160)
    assert adaptive().interpolation_geometry(plan, [
        {"width": 1080, "height": 1920, "layout": "fill_crop_center"}])[1] >= 1920


@pytest.mark.parametrize("native_clock_bias", [False, True])
@pytest.mark.parametrize("source_rate,source_count,cut", [(Fraction(30), 30, 15), (Fraction(24000, 1001), 48, 24)])
def test_cut_guard_does_not_show_next_shot_early(tmp_path, monkeypatch, source_rate, source_count, cut, native_clock_bias):
    import av
    import re
    source = tmp_path / "hard_cut.mov"
    with av.open(str(source), "w") as c:
        stream = c.add_stream("png", rate=source_rate)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "rgb24"
        for i in range(source_count):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 30 if i < cut else 230, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, 1 / source_rate
            for packet in stream.encode(frame):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)
    info = worker.probe_video(source, worker.standard_ffprobe(), inspect_timestamps=True)
    writer = worker.ProgressWriter(tmp_path / "jobs", "cut", "Cut protection CPU proof", "test")
    captured = []
    def deliberately_wrong_interpolation(argv, log_writer, phase, **kwargs):
        cpu = list(argv)
        at = cpu.index("-filter_complex") + 1
        captured.append(cpu[at])
        # Actual installed Apollo emits PTS one 1/24000 tick early. Default
        # framesync then chooses the previous hold despite matching FPS headers.
        timing = "fps=60,settb=expr=1/24000,setpts=PTS-1" if native_clock_bias else "fps=60"
        cpu[at] = re.sub(r"tvai_fi=[^,;\[]+", timing + ",lutrgb=r=0:g=0:b=0", cpu[at])
        worker.run_process(cpu, log_writer, "cpu_cut_guard", timeout=20)
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry", deliberately_wrong_interpolation)
    output = tmp_path / "cut_guard.mov"
    worker.topaz_interpolate_to_prores(source, output, worker.standard_ffmpeg(), "60/1", writer,
        source_info=info, source_cut_frames=[cut], output_dimensions=(64, 48))
    with av.open(str(output)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    import math
    assert len(frames) == math.ceil(source_count * 60 / source_rate)
    first, change, last = [math.ceil(Fraction(value) * 60 / source_rate) for value in (cut - 1, cut, cut + 1)]
    for index in range(first, last):
        assert np.mean(frames[index]) == pytest.approx(30 if index < change else 230, abs=2)
    assert np.mean(frames[first - 1]) < 3 and np.mean(frames[last]) < 3
    assert "round=up" in captured[0]
    # Native Apollo + framesync can emit an extra EOF event after the blend.
    # The source-derived endpoint must bound the completed graph, not only FI.
    assert captured[0].rindex("trim=end_frame=") > captured[0].index("blend=")


def test_authored_duplicate_frames_are_not_reclassified_as_missing_motion(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry", lambda argv, *a, **k: captured.append(argv))
    info = {"fps_ratio": "24000/1001", "video_frame_count": "48", "video_duration": "2.002000",
            "video_time_base": "1/24000", "video_start_time": "0", "width": 96, "height": 64}
    worker.topaz_interpolate_to_prores(tmp_path / "in.mov", tmp_path / "out.mov", "unused", "60/1",
        SimpleNamespace(log=lambda _: None), source_info=info, source_cut_frames=[24])
    graph = captured[0][captured[0].index("-filter_complex") + 1]
    assert ":rdt=0:" in graph
    assert "settb=expr=1/60,setpts=N*1" in graph


def test_interpolation_protection_activates_when_only_conservative_caption_fallback_exists():
    from adaptive_enhancement import interpolation_has_source_protection

    base = {"stream_index": 1, "policy": "authored_glyph_hold_v1",
            "detected_frames": 0, "conservative_fallback_frames": 5,
            "exact_repeat_pairs": 0}
    plan = {"caption_protection": base}
    assert interpolation_has_source_protection(plan) is True
    assert interpolation_has_source_protection({
        "caption_protection": {**base, "conservative_fallback_frames": 0}
    }) is False
    assert interpolation_has_source_protection({
        "caption_protection": {**base, "conservative_fallback_frames": 0,
                               "exact_repeat_pairs": 7}
    }) is True
    assert interpolation_has_source_protection({
        "caption_protection": {**base, "policy": "invalid"}
    }) is False


def test_exact_source_duplicates_are_held_for_60fps_without_warping_or_borrowing_from_cut(tmp_path, monkeypatch):
    import av
    import cv2
    import re

    source = tmp_path / "static_title_then_moving_30.mov"
    frames_rgb = []
    background = np.full((72, 128, 3), (60, 80, 100), np.uint8)
    cv2.putText(background, "STILL", (14, 39),
                cv2.FONT_HERSHEY_SIMPLEX, .72, (225, 225, 225), 2, cv2.LINE_AA)
    for i in range(8):
        if i < 3:
            rgb = background.copy()
        elif i < 5:
            rgb = np.roll(background, 6, axis=1)
        else:
            rgb = np.roll(background, 2 * (i - 5) + 24, axis=1)
        frames_rgb.append(rgb)
    with av.open(str(source), "w") as output:
        video = output.add_stream("png", rate=30)
        video.width, video.height, video.pix_fmt = 128, 72, "rgb24"
        for index, rgb in enumerate(frames_rgb):
            f = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            f.pts, f.time_base = index, Fraction(1, 30)
            for packet in video.encode(f):
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
    info = worker.probe_video(source, worker.standard_ffprobe(), inspect_timestamps=True)
    # Actual source cut starts at 5; the exact repeat between 3 and 4 must
    # never carry the hold mask into source frame 5.
    monkeypatch.setattr(adaptive(), "_scene_windows", lambda *args: (
        [(0, 5), (5, 8)], {"owner": "exact_shot_fixture"}))
    writer = worker.ProgressWriter(tmp_path / "jobs", "exact_holds", "Native-temporal protection", "test")
    plan = adaptive().prepare(source, info, 256, 144, writer,
                              {"mode": "natural", "subject_refinement": "off", "model_search": "off"})
    evidence = plan["caption_protection"]
    assert evidence["exact_repeat_pairs"] == 3
    assert evidence["exact_repeat_mask_frames"] == 4
    assert evidence["exact_repeat_hold_ranges"] == [[0, 2], [3, 4]]
    assert adaptive().interpolation_has_source_protection(plan)
    with av.open(plan["mask_path"]) as container:
        protected_frames = [f.to_ndarray(format="gray") for f in container.decode(video=1)]
    assert len(protected_frames) == 8
    for index in (0, 1, 3, 4):
        assert np.all(protected_frames[index] == 255)
    assert not np.all(protected_frames[5] == 255)

    graphs = []
    def wrong_fi(argv, parent_writer, stage, **kwargs):
        modified = list(argv)
        at = modified.index("-filter_complex") + 1
        graphs.append(modified[at])
        modified[at] = re.sub(
            r"tvai_fi=[^,;\[]+", "fps=fps=60,lutrgb=r=0:g=0:b=0", modified[at])
        worker.run_process(modified, parent_writer, "test_black_fi", timeout=30)
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry", wrong_fi)
    final = tmp_path / "final_60.mov"
    worker.topaz_interpolate_to_prores(
        source, final, worker.standard_ffmpeg(), "60/1", writer,
        source_info=info, output_dimensions=(128, 72),
        source_cut_frames=[5], caption_mask_plan=plan)
    with av.open(str(final)) as output:
        actual = [f.to_ndarray(format="rgb24") for f in output.decode(video=0)]
    assert len(actual) == 16
    for interval in (0, 1, 2, 3, 6, 7):
        expected = frames_rgb[interval // 2]
        assert np.mean(np.abs(actual[interval].astype(np.int16)
                              - expected.astype(np.int16))) < 7
    assert np.mean(actual[10]) > 10  # Source-cut protection uses shot 2, not shot 1.
    assert "[1:v:1]" in graphs[0] and "maskedmerge" in graphs[0]
    verified = worker.probe_video(final, worker.standard_ffprobe(), inspect_timestamps=True)
    assert verified["fps_ratio"] == "60/1"
    assert verified["video_pts_uniform"] is True
    assert int(verified["video_frame_count"]) == 16


@pytest.mark.parametrize("source_rate,target_rate", [
    (Fraction(24), Fraction(60)),
    (Fraction(24000, 1001), Fraction(60)),
    (Fraction(30000, 1001), Fraction(60000, 1001)),
    (Fraction(25), Fraction(60)),
])
def test_exact_still_region_protection_respects_fractional_60fps_clocks(
        tmp_path, monkeypatch, source_rate, target_rate):
    import av
    import math
    import re

    source = tmp_path / "still_then_motion.mov"
    pixels = np.full((72, 128, 3), (67, 103, 154), np.uint8)
    for y in range(8, 63, 9):
        pixels[y:y + 3, 28:110] = (205, 225, 190)
    with av.open(str(source), "w") as output:
        stream = output.add_stream("png", rate=source_rate)
        stream.width, stream.height, stream.pix_fmt = 128, 72, "rgb24"
        for index in range(8):
            rgb = pixels.copy() if index < 3 else np.roll(pixels, 4 * (index - 2), axis=1)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1) / source_rate
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    info = worker.probe_video(source, worker.standard_ffprobe(), inspect_timestamps=True)
    monkeypatch.setattr(adaptive(), "_scene_windows",
                        lambda *args: ([(0, 8)], {"owner": "fractional_rate_fixture"}))
    writer = worker.ProgressWriter(tmp_path / "jobs", "still_fps", "Static hold cadence", "test")
    plan = adaptive().prepare(source, info, 256, 144, writer,
                              {"mode": "natural", "subject_refinement": "off", "model_search": "off"})
    assert plan["caption_protection"]["exact_repeat_pairs"] == 2
    assert adaptive().interpolation_has_source_protection(plan)

    def deliberately_wrong_model(argv, parent_writer, phase, **kwargs):
        modified = list(argv)
        graph_index = modified.index("-filter_complex") + 1
        modified[graph_index] = re.sub(
            r"tvai_fi=[^,;\[]+", f"fps=fps={target_rate},lutrgb=r=0:g=0:b=0",
            modified[graph_index])
        worker.run_process(modified, parent_writer, "test_fractional_hold", timeout=30)
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry", deliberately_wrong_model)
    rendered = tmp_path / "result.mov"
    worker.topaz_interpolate_to_prores(
        source, rendered, worker.standard_ffmpeg(), str(target_rate), writer,
        source_info=info, source_cut_frames=[], output_dimensions=(128, 72),
        caption_mask_plan=plan)
    result_info = worker.probe_video(rendered, worker.standard_ffprobe(), inspect_timestamps=True)
    assert Fraction(result_info["fps_ratio"]) == target_rate
    assert result_info["video_pts_uniform"] is True
    assert int(result_info["video_frame_count"]) == math.ceil(Fraction(8) * target_rate / source_rate)
    with av.open(str(rendered)) as output:
        frames = list(output.decode(video=0))
    assert all(Fraction(frame.pts) * frame.time_base == Fraction(index) / target_rate
               for index, frame in enumerate(frames))
    held_target_count = math.ceil(Fraction(2) * target_rate / source_rate)
    for index in range(held_target_count):
        frame = frames[index].to_ndarray(format="rgb24")
        assert np.mean(np.abs(frame.astype(np.int16) - pixels.astype(np.int16))) < 8


def test_60fps_artifact_check_rejects_repeated_frames_during_authored_motion(tmp_path, monkeypatch):
    import av

    source = tmp_path / "moving_source_30fps.mov"
    colors = [42 + 12 * index for index in range(8)]
    with av.open(str(source), "w") as container:
        stream = container.add_stream("png", rate=30)
        stream.width, stream.height, stream.pix_fmt = 128, 72, "rgb24"
        for index, color in enumerate(colors):
            frame = av.VideoFrame.from_ndarray(
                np.full((72, 128, 3), color, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    info = worker.probe_video(source, worker.standard_ffprobe(), inspect_timestamps=True)
    monkeypatch.setattr(adaptive(), "_scene_windows",
                        lambda *args: ([(0, 8)], {"owner": "one_fade_shot"}))
    writer = worker.ProgressWriter(tmp_path / "jobs", "fps_judder", "FPS continuity QA", "test")
    plan = adaptive().prepare(source, info, 256, 144, writer,
                              {"mode": "natural", "subject_refinement": "off", "model_search": "off"})
    moving, repeated = tmp_path / "interpolated.mov", tmp_path / "duplicated.mov"
    for destination, interpolate in ((moving, True), (repeated, False)):
        with av.open(str(destination), "w") as container:
            stream = container.add_stream("png", rate=60)
            stream.width, stream.height, stream.pix_fmt = 128, 72, "rgb24"
            for index in range(16):
                src = min(7, index // 2)
                color = colors[src]
                if interpolate and index % 2 and src < 7:
                    color = (colors[src] + colors[src + 1]) // 2
                frame = av.VideoFrame.from_ndarray(
                    np.full((72, 128, 3), color, dtype=np.uint8), format="rgb24")
                frame.pts, frame.time_base = index, Fraction(1, 60)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    good = adaptive().assess_temporal_candidate(source, moving, plan, "60/1", [], writer)
    assert good["status"] == "PASS_AUTHORED_PROTECTION", good["reasons"]
    bad = adaptive().assess_temporal_candidate(source, repeated, plan, "60/1", [], writer)
    assert bad["status"] == "REJECT"
    assert "interpolated_motion_flatline" in bad["reasons"]
    assert bad["flatline_source_intervals"] >= 6
    assert bad["motion_intervals_screened"] == 7


def test_60fps_motion_check_never_treats_an_authored_hard_cut_as_failed_interpolation(
        tmp_path, monkeypatch):
    import av

    source, output = tmp_path / "two_still_shots.mov", tmp_path / "authored_holds_60.mov"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("png", rate=30)
        stream.width, stream.height, stream.pix_fmt = 128, 72, "rgb24"
        for index in range(8):
            frame = av.VideoFrame.from_ndarray(
                np.full((72, 128, 3), 70 if index < 4 else 110, dtype=np.uint8),
                format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    info = worker.probe_video(source, worker.standard_ffprobe(), inspect_timestamps=True)
    monkeypatch.setattr(adaptive(), "_scene_windows",
                        lambda *args: ([(0, 4), (4, 8)], {"owner": "two_still_shots"}))
    writer = worker.ProgressWriter(tmp_path / "jobs", "cut_holds", "Hard cut QA", "test")
    plan = adaptive().prepare(source, info, 256, 144, writer,
                              {"mode": "natural", "subject_refinement": "off", "model_search": "off"})
    with av.open(str(output), "w") as container:
        stream = container.add_stream("png", rate=60)
        stream.width, stream.height, stream.pix_fmt = 128, 72, "rgb24"
        for index in range(16):
            frame = av.VideoFrame.from_ndarray(
                np.full((72, 128, 3), 70 if index < 8 else 110, dtype=np.uint8),
                format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 60)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    evidence = adaptive().assess_temporal_candidate(source, output, plan, "60/1", [4], writer)
    assert evidence["status"] == "PASS_AUTHORED_PROTECTION", evidence["reasons"]
    assert evidence["motion_intervals_screened"] == 7
    assert evidence["flatline_source_intervals"] == 0
    assert evidence["cut_samples"] > 0


def test_caption_protection_detects_glyph_groups_without_guessing_a_subtitle_box():
    import cv2
    image = detail_fixture()
    glyphs = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.putText(glyphs, "HELLO WORLD", (115, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2, cv2.LINE_AA)
    image[glyphs > 200] = 250
    protection, evidence = adaptive().caption_protection_mask(image)
    assert protection.dtype == np.uint8 and protection.shape == glyphs.shape
    assert np.mean(protection[glyphs > 200] > 200) >= 0.98
    assert np.count_nonzero(protection[:60]) == 0
    assert np.mean(protection > 0) < 0.2
    assert evidence["semantic_recognition"] is False
    assert evidence["line_count"] >= 1
    rectangle = np.full_like(image, 110)
    rectangle[220:245, 100:400] = 250
    assert not np.any(adaptive().caption_protection_mask(rectangle)[0])


def test_interpolation_pilots_are_bounded_to_measured_motion_inside_shots():
    plan = {"frame_count": 240, "source_fps": "24000/1001", "shots": [
        {"start_frame": 0, "end_frame": 100, "motion_peak_frame": 98, "motion_peak_delta": 8},
        {"start_frame": 100, "end_frame": 106, "motion_peak_frame": 103, "motion_peak_delta": 30},
        {"start_frame": 106, "end_frame": 240, "motion_peak_frame": 200, "motion_peak_delta": 12},
    ]}
    windows = adaptive().interpolation_pilot_windows(plan, "60/1", 1080, 1350)
    assert len(windows) == 2 and sum(b - a for a, b in windows) == 16
    assert all(any(s["start_frame"] <= a < b <= s["end_frame"] for s in plan["shots"]) for a, b in windows)
    assert any(a <= 200 < b for a, b in windows)
    assert not adaptive().interpolation_pilot_windows({**plan, "frame_count": 48}, "60/1", 1080, 1350)
    assert not adaptive().interpolation_pilot_windows(plan, "120/1", 1080, 1350)
    assert not adaptive().interpolation_pilot_windows(plan, "60/1", 3840, 2160)


def test_exact_still_shots_skip_extra_interpolation_model_audition_but_moving_shot_can_run():
    static = {"start_frame": 0, "end_frame": 120, "route": "masked_full_frame",
              "motion_peak_frame": 40, "motion_peak_delta": 0.0,
              "exact_repeat_pairs": 119}
    moving = {"start_frame": 120, "end_frame": 360, "route": "masked_full_frame",
              "motion_peak_frame": 220, "motion_peak_delta": 16.0,
              "exact_repeat_pairs": 0}
    plan = {"frame_count": 360, "source_fps": "24/1", "shots": [static]}
    assert adaptive().interpolation_pilot_windows(plan, "60/1", 1080, 1350) == []
    plan["shots"] = [static, moving]
    assert adaptive().interpolation_pilot_windows(plan, "60/1", 1080, 1350) == [(216, 224)]


def test_short_export_keeps_reference_instead_of_spending_third_of_work_on_model_comparison():
    plan = {"frame_count": 96, "source_fps": "24000/1001", "shots": [
        {"start_frame": 0, "end_frame": 48, "motion_peak_frame": 20, "motion_peak_delta": 8},
        {"start_frame": 48, "end_frame": 96, "motion_peak_frame": 70, "motion_peak_delta": 10},
    ]}
    assert adaptive().interpolation_pilot_windows(plan, "60/1", 1080, 1350) == []
    long_plan = {**plan, "frame_count": 240, "shots": [
        {**plan["shots"][0], "end_frame": 120},
        {**plan["shots"][1], "start_frame": 120, "end_frame": 240, "motion_peak_frame": 180},
    ]}
    windows = adaptive().interpolation_pilot_windows(long_plan, "60/1", 1080, 1350)
    assert len(windows) == 2
    assert 2 * sum(end - start for start, end in windows) * 5 <= long_plan["frame_count"]


def test_fast_interpolation_needs_both_pixel_agreement_and_measured_speed():
    accepted = {"status": "PASS_REFERENCE_AGREEMENT", "frames": 20, "reasons": []}
    assert adaptive().choose_interpolation_model(accepted, 20, 10)["model"] == "apf-2"
    assert adaptive().choose_interpolation_model(accepted, 20, 21)["model"] == "apo-8"
    assert adaptive().choose_interpolation_model({**accepted, "status": "REJECT", "reasons": ["local_detail_changed"]}, 20, 1)["model"] == "apo-8"
    for invalid in [0, -1, True, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="timing"):
            adaptive().choose_interpolation_model(accepted, invalid, 1)


def test_interpolator_comparison_rejects_local_damage_not_only_global_average(tmp_path):
    import av
    def video(path, patch):
        with av.open(str(path), "w") as c:
            stream = c.add_stream("png", rate=60)
            stream.width, stream.height, stream.pix_fmt = 256, 144, "rgb24"
            for i in range(8):
                pixels = np.full((144,256,3), 90 + i, np.uint8)
                if patch:
                    pixels[50:64, 80:94] = 240
                f = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                f.pts, f.time_base = i, Fraction(1,60)
                for packet in stream.encode(f):
                    c.mux(packet)
            for packet in stream.encode():
                c.mux(packet)
    reference, candidate = tmp_path / "reference.mov", tmp_path / "candidate.mov"
    video(reference, False)
    writer = SimpleNamespace(cancel_requested=False)
    assert adaptive().compare_interpolation_candidates(reference, reference, writer)["status"] == "PASS_REFERENCE_AGREEMENT"
    video(candidate, True)
    result = adaptive().compare_interpolation_candidates(reference, candidate, writer)
    assert result["status"] == "REJECT" and "local_detail_changed" in result["reasons"]


def test_selected_local_interpolation_model_reaches_real_filter_without_arbitrary_injection(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry", lambda argv, *a, **k: captured.append(argv))
    info = {"fps_ratio": "24/1", "video_frame_count": "8", "video_duration": "0.333333",
            "video_time_base": "1/24000", "video_start_time": "0", "width": 96, "height": 64}
    writer = SimpleNamespace(log=lambda _: None)
    worker.topaz_interpolate_to_prores(tmp_path / "in", tmp_path / "out", "unused", "60/1", writer,
        source_info=info, source_cut_frames=[], interpolation_model="apf-2")
    graph = captured[0][captured[0].index("-filter_complex") + 1]
    assert "tvai_fi=model=apf-2:" in graph and "download=0" in graph
    with pytest.raises(ValueError, match="interpolation model"):
        worker.topaz_interpolate_to_prores(tmp_path / "in", tmp_path / "out", "unused", "60/1", writer,
            source_info=info, source_cut_frames=[], interpolation_model="apf-2:download=1")


def test_per_video_calibration_is_connected_cached_and_never_downloads_models(tmp_path, monkeypatch):
    import subprocess
    source = tmp_path / "moving.mp4"
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=s=128x72:r=24:d=4",
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0", str(source)],
        check=True, capture_output=True, timeout=20)
    info = worker.probe_video(source, ffprobe, inspect_timestamps=True)
    writer = worker.ProgressWriter(tmp_path / "jobs", "calibration", "Calibration CPU contract", "test")
    plan = adaptive().prepare(source, info, 256, 144, writer, {"mode": "natural"})
    calls = []
    monkeypatch.setattr(worker, "topaz_model_asset_identity", lambda directory, model:
        {"resume_safe": True, "identity_sha256": model + "-local-fixture"})
    def cpu_standin(src, dest, exe, rate, log, **kwargs):
        calls.append(kwargs["interpolation_model"])
        boundary = worker.interpolation_boundary_contract(kwargs["source_info"], rate)
        subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", str(src), "-an", "-vf",
            f"tpad=stop_mode=clone:stop=1,fps={rate},trim=end_frame={boundary['trim_end_frame']}",
            "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", str(dest)],
            check=True, capture_output=True, timeout=20)
        return False
    monkeypatch.setattr(worker, "topaz_interpolate_to_prores", cpu_standin)
    first = worker.select_adaptive_interpolation_model(source, info, plan, "60/1", (128,72), "unused", "unused", writer)
    assert first["comparison"]["status"] == "PASS_REFERENCE_AGREEMENT"
    assert calls == ["apo-8", "apf-2"]
    second = worker.select_adaptive_interpolation_model(source, info, plan, "60/1", (128,72), "unused", "unused", writer)
    assert second["cache_hit"] is True and calls == ["apo-8", "apf-2"]
    assert second["model"] == first["model"]
    assert not list(writer.temp_dir.glob("*_fi_pilot*.mov"))
    monkeypatch.setattr(worker, "topaz_model_asset_identity", lambda *args:
        {"resume_safe": False, "identity_sha256": "absent"})
    missing = worker.select_adaptive_interpolation_model(source, info, plan, "60/1", (128,72), "unused", "unused", writer)
    assert missing["model"] == "apo-8" and missing["reason"] == "local_candidate_assets_unavailable_no_download"
    assert calls == ["apo-8", "apf-2"]


def test_temporal_caption_stream_is_exactly_bound_and_covers_next_authored_position(tmp_path, monkeypatch):
    import av
    import cv2
    source = tmp_path / "animated_caption.mov"
    with av.open(str(source), "w") as c:
        stream = c.add_stream("png", rate=30)
        stream.width, stream.height, stream.pix_fmt = 512, 288, "rgb24"
        for i in range(3):
            image = detail_fixture()
            cv2.putText(image, "HELLO WORLD", (90 + i * 12, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (250, 250, 250), 2, cv2.LINE_AA)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1,30)
            for packet in stream.encode(frame):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)
    info = worker.probe_video(source, worker.standard_ffprobe(), inspect_timestamps=True)
    writer = worker.ProgressWriter(tmp_path / "jobs", "captions", "Temporal captions", "test")
    plan = adaptive().prepare(source, info, 1024, 576, writer, {"mode": "natural"})
    assert plan["caption_protection"]["stream_index"] == 1
    assert plan["caption_protection"]["detected_frames"] == 3
    with av.open(plan["mask_path"]) as c:
        frames = list(c.decode(video=1))
    assert len(frames) == 3
    assert [Fraction(f.pts)*f.time_base for f in frames] == [Fraction(i,30) for i in range(3)]
    # Pixel near the next frame's glyph must already suppress a synthesized trail.
    with av.open(str(source)) as c:
        originals = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    next_mask, _ = adaptive().caption_protection_mask(originals[1])
    first_mask = frames[0].to_ndarray(format="gray")
    assert np.all(first_mask[next_mask == 255] == 255)
    # Deliberately wrong model pixels prove the production interpolation graph
    # protects actual caption shapes; this stand-in is not native AI evidence.
    import re
    captured = []
    def black_model(argv, log_writer, phase, **kwargs):
        cpu = list(argv)
        at = cpu.index("-filter_complex") + 1
        captured.append(cpu[at])
        cpu[at] = re.sub(r"tvai_fi=[^,;\[]+", "fps=60,lutrgb=r=0:g=0:b=0", cpu[at])
        worker.run_process(cpu, log_writer, "CPU caption graph", timeout=30)
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry", black_model)
    output = tmp_path / "protected_60.mov"
    worker.topaz_interpolate_to_prores(source, output, worker.standard_ffmpeg(), "60/1", writer,
        source_info=info, output_dimensions=(512,288), source_cut_frames=[], caption_mask_plan=plan)
    with av.open(str(output)) as c:
        output_frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    assert len(output_frames) == 6
    for index, rendered in enumerate(output_frames):
        expected = originals[index // 2]
        glyph = np.zeros(expected.shape[:2], dtype=np.uint8)
        cv2.putText(glyph, "HELLO WORLD", (90 + (index // 2) * 12, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2, cv2.LINE_AA)
        delta = np.abs(rendered.astype(float) - expected.astype(float))
        assert np.quantile(delta[glyph > 250], .99) < 12
        assert np.mean(rendered[:50]) < 3
    assert "[1:v:1]" in captured[0] and "maskedmerge" in captured[0]
    quality = adaptive().assess_temporal_candidate(source, output, plan, "60/1", [], writer)
    assert quality["status"] == "PASS_AUTHORED_PROTECTION"
    assert quality["caption_samples"] > 0 and quality["frames_decoded"] == 6
    bad = tmp_path / "destroyed_caption.mov"
    with av.open(str(bad), "w") as c:
        stream = c.add_stream("png", rate=60)
        stream.width, stream.height, stream.pix_fmt = 512, 288, "rgb24"
        for i in range(6):
            f = av.VideoFrame.from_ndarray(np.zeros((288,512,3),np.uint8), format="rgb24")
            f.pts, f.time_base = i, Fraction(1,60)
            for packet in stream.encode(f):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)
    rejected = adaptive().assess_temporal_candidate(source, bad, plan, "60/1", [], writer)
    assert rejected["status"] == "REJECT" and "authored_caption_changed" in rejected["reasons"]


def test_dense_analysis_precedes_optional_subject_models_on_a_flat_shot(tmp_path, monkeypatch):
    import av
    import adaptive_subjects
    import workflow.bounded_specialist_runtime as runtime
    source=tmp_path/'flat_24.mov'
    with av.open(str(source),'w') as out:
        stream=out.add_stream('png',rate=24)
        stream.width,stream.height,stream.pix_fmt=640,360,'rgb24'
        for i in range(24):
            frame=av.VideoFrame.from_ndarray(np.full((360,640,3),110,np.uint8),format='rgb24')
            frame.pts,frame.time_base=i,Fraction(1,24)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    monkeypatch.setattr(adaptive(),'_scene_windows',lambda *a: ([(0,24)],{'owner':'test scene boundary'}))
    monkeypatch.setattr(adaptive_subjects,'resource_snapshot',lambda:pytest.fail('Flat footage should not request optional inference'))
    monkeypatch.setattr(runtime,'run_bounded_specialist',lambda *a,**kw:pytest.fail('No useful detail; no subject model'))
    info=worker.probe_video(source,worker.standard_ffprobe(),inspect_timestamps=True)
    writer=worker.ProgressWriter(tmp_path/'jobs','flat','Flat first','test')
    plan=adaptive().prepare(source,info,1280,720,writer,{'mode':'natural'})
    assert plan['route']=='source_preserved'
    assert plan['subject_refinement']['shots'][0]['reason']=='no_enhancement_contribution_in_shot'


def test_streamed_subject_refinement_restricts_real_masks_without_changing_caption_stream(tmp_path,monkeypatch):
    """Actual mask encode/decode with synthetic subject alpha, not SAM2 proof."""
    import av
    import cv2
    import hashlib
    import adaptive_subjects
    source=tmp_path/'source.mov'
    write_fixture(source,6)
    monkeypatch.setattr(adaptive(),'_scene_windows',lambda *a: ([(0,6)],{'owner':'test scene boundary'}))
    writer=worker.ProgressWriter(tmp_path/'jobs','subject-stream','Streamed alpha test','test')
    info=worker.probe_video(source,worker.standard_ffprobe(),inspect_timestamps=True)
    original=adaptive().prepare(source,info,1024,576,writer,{'mode':'natural','subject_refinement':'off'})
    mapping={}
    for i in range(6):
        alpha=np.zeros((288,512),np.uint8)
        alpha[80:175,220:295]=255
        path=tmp_path/f'mask_{i}.png'
        cv2.imwrite(str(path),alpha)
        mapping[i]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def subject(*args,**kwargs):
        assert kwargs['shot_evidence'][0]['route']!='source_preserved'
        return {'schema':'synthetic-alpha-contract','shots':[{'start_frame':0,'end_frame':6,'status':'ready'}],
            'seconds':0,'successful_shots':1,'attempted_shots':1},mapping
    monkeypatch.setattr(adaptive_subjects,'prepare_subjects',subject)
    refined=adaptive().prepare(source,info,1024,576,writer,{'mode':'natural'})
    for index in (0,1):
        with av.open(original['mask_path']) as base,av.open(refined['mask_path']) as changed:
            a=list(base.decode(video=index))
            b=list(changed.decode(video=index))
        assert len(a)==len(b)==6
        for x,y in zip(a,b):
            assert Fraction(x.pts)*x.time_base==Fraction(y.pts)*y.time_base
            old,new=x.to_ndarray(format='gray'),y.to_ndarray(format='gray')
            if index==1:
                np.testing.assert_array_equal(old,new)
            else:
                assert np.all(new<=old)
                assert np.count_nonzero(new)<np.count_nonzero(old)
                assert np.count_nonzero(new)>0
    assert not list(writer.temp_dir.glob('*.refined.nut'))
