from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "command_palette/workers/python/src"
for path in (ROOT, PACKAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def worker():
    import sscp.workers.smart_finish.main
    return sscp.workers.smart_finish.main


def make_job(tmp_path, arguments=None):
    root = tmp_path / "workspace"
    job = root / "_jobs" / "smart-finish"
    job.mkdir(parents=True)
    source = root / "Finished (1).mp4"
    source.write_bytes(b"owned-selection-fixture")
    (job / "manifest.json").write_text(json.dumps({
        "job_id": job.name, "command_id": "media.smart_finish", "title": "Smart Finish",
        "arguments": {"output_fps": 60} if arguments is None else arguments,
    }), encoding="utf8")
    (job / "selected_files.json").write_text(json.dumps({"items": [{"index": 0, "path": str(source)}]}), encoding="utf8")
    return job, source


def test_palette_selection_uses_existing_delivery_owner_and_does_not_hold_child_gpu_lock(tmp_path, monkeypatch):
    module = worker()
    job, source = make_job(tmp_path)
    before = source.read_bytes()
    calls = []
    outcome = {"state": "passed", "final_outputs": [str(tmp_path / "derived.mp4")],
               "final_output_proofs": [{"adaptive_actual_route": "masked_crop"}], "errors": []}
    def run(sources, **kwargs):
        calls.append((sources, kwargs))
        return 0, outcome
    monkeypatch.setattr(module, "_run_delivery", run)
    monkeypatch.setattr(module, "_verify_delivery", lambda _job: {"status": "PASS"})
    assert module.run_job(job) == 0
    sources, kwargs = calls[0]
    assert sources == [source.resolve()]
    assert kwargs["jobs_root"] == job.parent
    assert kwargs["job_id"] == job.name
    assert kwargs["profile_ids"] == ["smart_finish_master"]
    assert kwargs["adaptive_enhancement"] == {"mode": "natural"}
    assert kwargs["frame_rate"] == {"mode": "cfr", "fps": "60/1"}
    assert kwargs["execute"] is True and kwargs["watermark"] is False
    result = json.loads((job / "result.json").read_text())
    assert result["state"] == "passed" and result["final_outputs"] == outcome["final_outputs"]
    assert source.read_bytes() == before
    manifest = json.loads((ROOT / "command_palette/manifests/commands/media/smart_finish.json").read_text())
    assert manifest["queue_behavior"]["mode"] == "single_command"
    assert manifest["worker_entry"] == "sscp.workers.smart_finish.main"


@pytest.mark.parametrize("fault", ["missing", "folder", "empty", "unknown_argument", "bad_rate", "bad_command"])
def test_bad_palette_request_never_reaches_renderer(tmp_path, monkeypatch, fault):
    module = worker()
    job, source = make_job(tmp_path)
    if fault == "missing":
        source.unlink()
    elif fault == "folder":
        source.unlink()
        source.mkdir()
    elif fault == "empty":
        (job / "selected_files.json").write_text('{"items":[]}')
    else:
        path = job / "manifest.json"
        manifest = json.loads(path.read_text())
        if fault == "unknown_argument":
            manifest["arguments"]["cloud"] = True
        elif fault == "bad_rate":
            manifest["arguments"]["output_fps"] = True
        else:
            manifest["command_id"] = "another.owner"
        path.write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "_run_delivery", lambda *a, **k: pytest.fail("Must not start renderer"))
    assert module.run_job(job) == 1
    assert json.loads((job / "result.json").read_text())["state"] == "failed"


def test_palette_cancel_and_native_failure_are_not_reported_as_success(tmp_path, monkeypatch):
    module = worker()
    job, _ = make_job(tmp_path)
    (job / "cancel.request").touch()
    monkeypatch.setattr(module, "_run_delivery", lambda *a, **k: pytest.fail("Cancelled before render"))
    assert module.run_job(job) == 2
    assert json.loads((job / "result.json").read_text())["state"] == "cancelled"
    (job / "cancel.request").unlink()
    monkeypatch.setattr(module, "_run_delivery", lambda *a, **k: (1, {"state": "failed", "errors": ["model unavailable"], "final_outputs": []}))
    assert module.run_job(job) == 1
    result = json.loads((job / "result.json").read_text())
    assert result["state"] == "failed" and result["errors"] == ["model unavailable"]


def test_palette_rechecks_owner_output_before_accepting_exit_zero(tmp_path, monkeypatch):
    module = worker()
    job, _ = make_job(tmp_path)
    monkeypatch.setattr(module, "_run_delivery", lambda *a, **k: (0, {"state": "passed", "errors": [], "final_outputs": []}))
    monkeypatch.setattr(module, "_verify_delivery", lambda _job: {"status": "FAIL", "issues": ["output missing"]})
    assert module.run_job(job) == 1
    assert json.loads((job / "result.json").read_text())["state"] == "failed"


def test_new_manifest_is_searchable_without_restarting_stale_ahk(tmp_path, monkeypatch):
    import sscp.command_palette_runtime as runtime
    manifest = json.loads((ROOT / "command_palette/manifests/commands/media/smart_finish.json").read_text())
    compiled = tmp_path / "command_palette/manifests/registry/commands.compiled.json"
    compiled.parent.mkdir(parents=True)
    compiled.write_text(json.dumps({"schema_version": 1, "commands": [manifest]}))
    snapshot = tmp_path / "_cache/command_palette/registry_snapshot.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps({"schema_version": 1, "source": "ahk", "commands": []}))
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    catalog = runtime.RegistryCatalog(tmp_path, runtime.PreferenceStore(tmp_path / "config/preferences.json"))
    record = catalog.record("media.smart_finish")
    assert record is not None and record.palette_visible
    assert record.raw["_runtime_dispatch_owner"] == "portable_manifest"


def test_palette_explicit_working_size_is_forwarded_without_changing_source_delivery(tmp_path, monkeypatch):
    module = worker()
    job, _source = make_job(tmp_path, {"output_fps": "60000/1001", "working_long_edge": 3072})
    observed = {}
    def run(sources, **kwargs):
        observed.update(kwargs)
        return 0, {"state": "passed", "errors": [], "final_outputs": [str(tmp_path / "derived.mp4")]}
    monkeypatch.setattr(module, "_run_delivery", run)
    monkeypatch.setattr(module, "_verify_delivery", lambda _job: {"status": "PASS"})
    assert module.run_job(job) == 0
    assert observed["adaptive_enhancement"] == {"mode": "natural", "working_long_edge": 3072}
    assert observed["frame_rate"] == {"mode": "cfr", "fps": "60000/1001"}
    assert observed["profile_ids"] == ["smart_finish_master"]


@pytest.mark.parametrize("edge", [True, "3072", 1025, -1, 16384])
def test_palette_invalid_working_size_never_reaches_renderer(tmp_path, monkeypatch, edge):
    module = worker()
    job, _source = make_job(tmp_path, {"working_long_edge": edge})
    monkeypatch.setattr(module, "_run_delivery", lambda *a, **kw: pytest.fail("Invalid request must stop before processing"))
    assert module.run_job(job) == 1


def test_quality_advisor_command_inspects_selected_video_without_rendering(tmp_path, monkeypatch):
    module = worker()
    job, source = make_job(tmp_path, {"quality_policy": "best_quality"})
    manifest_path = job / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["command_id"] = module.ADVISOR_COMMAND_ID
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    module._owner()
    from shared import media_probe
    import quality_advisor
    monkeypatch.setattr(media_probe, "probe_video", lambda *_args: {"width": 1920, "height": 1080, "fps": 24, "duration": 8.0})
    monkeypatch.setattr(quality_advisor, "inspect_comfyui", lambda: {"server_running": False, "reason": "test"})
    monkeypatch.setattr(module, "_run_delivery", lambda *a, **k: pytest.fail("Advisor must not launch renderer"))
    before = source.read_bytes()
    assert module.run_job(job) == 0
    result = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert result["command_id"] == module.ADVISOR_COMMAND_ID
    assert result["rendered"] is False and result["final_outputs"] == []
    assert result["items"][0]["advice"]["recommended"] == "4K"
    assert source.read_bytes() == before


def test_palette_manual_6k_is_explicitly_raster_only_and_interpolates_at_working_resolution(tmp_path, monkeypatch):
    module = worker()
    job, _source = make_job(tmp_path, {"final_long_edge": 5760, "allow_raster_only": True})
    captured = {}
    def run(sources, **kwargs):
        captured.update(kwargs)
        return 0, {"state": "passed", "errors": [], "final_outputs": [str(tmp_path / "result.mp4")]}
    monkeypatch.setattr(module, "_run_delivery", run)
    monkeypatch.setattr(module, "_verify_delivery", lambda *_args: {"status": "PASS"})
    assert module.run_job(job) == 0
    assert captured["geometry"] == {"final": {"mode": "long_edge", "long_edge": 5760}}
    assert captured["adaptive_enhancement"]["interpolation_resolution"] == "working"


def test_palette_6k_without_acknowledgment_fails_before_render(tmp_path, monkeypatch):
    module = worker()
    job, _source = make_job(tmp_path, {"final_long_edge": 5760})
    monkeypatch.setattr(module, "_run_delivery", lambda *a, **k: pytest.fail("Must not render"))
    assert module.run_job(job) == 1
    error = json.loads((job / "result.json").read_text(encoding="utf-8"))["errors"]
    assert "not native 6K/8K neural detail" in error[0]


def test_best_quality_command_passes_source_admitted_working_and_final_geometry(tmp_path, monkeypatch):
    module = worker()
    job, _source = make_job(tmp_path, {"quality_policy": "best_quality", "apply_recommendation": True})
    manifest_path = job / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["command_id"] = module.BEST_QUALITY_COMMAND_ID
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    module._owner()
    from shared import media_probe
    monkeypatch.setattr(media_probe, "probe_video", lambda *_args: {"width": 1920, "height": 1080, "fps": 24, "duration": 8.0})
    observed = {}
    def run(sources, **kwargs):
        observed.update(kwargs)
        return 0, {"state": "passed", "errors": [], "final_outputs": [str(tmp_path / "result.mp4")]}
    monkeypatch.setattr(module, "_run_delivery", run)
    monkeypatch.setattr(module, "_verify_delivery", lambda *_args: {"status": "PASS"})
    assert module.run_job(job) == 0
    assert observed["adaptive_enhancement"]["working_long_edge"] == 3840
    assert observed["geometry"] == {"final": {"mode": "long_edge", "long_edge": 3840}}
    result = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert result["command_id"] == module.BEST_QUALITY_COMMAND_ID


def test_comfyui_palette_dispatches_one_local_source_with_quality_encode(tmp_path, monkeypatch):
    module = worker()
    job, source = make_job(tmp_path, {"compression_profile": "quality", "final_long_edge": 3840})
    manifest_path = job / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["command_id"] = module.COMFYUI_COMMAND_ID
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    module._owner()
    import comfyui_video_backend
    observed = {}

    def run(source_path, **kwargs):
        observed.update(kwargs)
        observed["source"] = source_path
        return 0, {"schema": "spiffy_comfyui_delivery_result_v1",
                   "state": "passed", "backend": "comfyui_frame_gan_v1",
                   "errors": [], "final_outputs": [str(tmp_path / "rendered.mp4")]}

    monkeypatch.setattr(comfyui_video_backend, "run_comfyui_delivery", run)
    monkeypatch.setattr(module, "_run_delivery", lambda *a, **k: pytest.fail("Topaz must not start"))
    assert module.run_job(job) == 0
    assert observed["source"] == source.resolve()
    assert observed["target_long_edge"] == 3840
    assert observed["crf"] == 16 and observed["preset"] == "slow"
    assert observed["job_id"] == job.name
    result = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert result["command_id"] == module.COMFYUI_COMMAND_ID
    assert result["state"] == "passed"


def test_comfyui_palette_failure_remains_failed_without_topaz_fallback(tmp_path, monkeypatch):
    module = worker()
    job, _ = make_job(tmp_path, {"compression_profile": "quality"})
    manifest_path = job / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["command_id"] = module.COMFYUI_COMMAND_ID
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    module._owner()
    import comfyui_video_backend
    monkeypatch.setattr(comfyui_video_backend, "run_comfyui_delivery", lambda *_args, **_kwargs:
                        (1, {"schema": "spiffy_comfyui_delivery_result_v1",
                             "state": "failed", "backend": "comfyui_frame_gan_v1",
                             "errors": ["GPU admission declined"], "final_outputs": []}))
    monkeypatch.setattr(module, "_run_delivery", lambda *a, **k: pytest.fail("No silent Topaz fallback"))
    assert module.run_job(job) == 1
    result = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert result["state"] == "failed" and result["final_outputs"] == []


def test_measured_compression_palette_routes_bounded_quality_calibration(tmp_path, monkeypatch):
    module = worker()
    job, source = make_job(tmp_path, {
        "backend_id": "comfyui-gan",
        "model_id": "auto-quality",
        "compression_profile": "quality",
        "compression_strategy": "measured",
        "final_long_edge": 3840,
    })
    path = job / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["command_id"] = module.COMFYUI_MEASURED_COMMAND_ID
    path.write_text(json.dumps(payload), encoding="utf-8")
    module._owner()
    import comfyui_video_backend
    seen = {}

    def run_backend_mock(source_path, **kwargs):
        seen.update(kwargs)
        assert source_path == source.resolve()
        return 0, {
            "schema": "spiffy_comfyui_delivery_result_v1",
            "state": "passed", "errors": [],
            "final_outputs": [str(tmp_path / "measured_result.mp4")],
        }

    monkeypatch.setattr(comfyui_video_backend, "run_comfyui_delivery", run_backend_mock)
    monkeypatch.setattr(module, "_run_delivery", lambda *_a, **_kw: pytest.fail("Unexpected Topaz route"))
    assert module.run_job(job) == 0
    assert seen["compression_strategy"] == "measured"
    assert seen["crf"] == 16
    assert seen["model_id"] == "auto-quality"
    assert seen["target_long_edge"] == 3840
    result = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert result["state"] == "passed"
    assert result["command_id"] == module.COMFYUI_MEASURED_COMMAND_ID


@pytest.mark.parametrize("invalid", ["fixed", True, 15, "guess"])
def test_measured_compression_command_rejects_false_or_invalid_modes(tmp_path, monkeypatch, invalid):
    module = worker()
    job, _source = make_job(tmp_path, {"compression_strategy": invalid})
    path = job / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["command_id"] = module.COMFYUI_MEASURED_COMMAND_ID
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(module, "_run_delivery", lambda *_a, **_kw: pytest.fail("Invalid request reached Topaz"))
    assert module.run_job(job) == 1
    result = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert result["state"] == "failed"


def test_palette_explicit_motioncoach_face_region_routes_to_existing_backend(tmp_path, monkeypatch):
    module = worker()
    pose_dir = tmp_path / "published_motioncoach_pose"
    job, source = make_job(tmp_path, {
        "region_mode": "motioncoach_face",
        "motioncoach_pose_root": str(pose_dir),
        "motioncoach_source_binding_id": "source.selected_performer",
        "model_id": "auto-quality",
        "final_long_edge": 3840,
    })
    path = job / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["command_id"] = module.COMFYUI_COMMAND_ID
    path.write_text(json.dumps(payload), encoding="utf-8")
    module._owner()
    import comfyui_video_backend

    observed = {}

    def run_mock(source_path, **kwargs):
        assert source_path == source.resolve()
        observed.update(kwargs)
        return 0, {
            "schema": "spiffy_comfyui_delivery_result_v1",
            "state": "passed", "errors": [], "final_outputs": [str(tmp_path / "verified_face.mp4")],
        }

    monkeypatch.setattr(comfyui_video_backend, "run_comfyui_delivery", run_mock)
    monkeypatch.setattr(module, "_run_delivery", lambda *a, **k: pytest.fail("Not a Topaz job"))
    assert module.run_job(job) == 0
    assert observed["region_mode"] == "motioncoach_face"
    assert observed["motioncoach_pose_root"] == pose_dir
    assert observed["motioncoach_source_binding_id"] == "source.selected_performer"
    assert observed["source_preservation"] == 0.0


def test_palette_rejects_face_mode_without_complete_published_binding(tmp_path, monkeypatch):
    module = worker()
    job, _source = make_job(tmp_path, {"region_mode": "motioncoach_face"})
    path = job / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["command_id"] = module.COMFYUI_COMMAND_ID
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "_run_delivery", lambda *a, **k: pytest.fail("No Topaz fallback"))
    assert module.run_job(job) == 1
    failure = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert "published MotionCoach pose" in failure["errors"][0]
