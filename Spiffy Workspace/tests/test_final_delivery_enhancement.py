"""Smart Finish B1 contract tests; no Topaz inference or production media."""
from __future__ import annotations

import copy
import hashlib
import json
import importlib.util
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from shared import media_probe
import workflow.final_delivery_pipeline as pipeline
import video_upscale as worker
_runner_spec = importlib.util.spec_from_file_location(
    "final_delivery_cli_b1", Path(pipeline.WORKSPACE_ROOT) / "tools/run_spiffy_final_delivery.py")
runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(runner)


def test_request_plan_command_recipe_connected(tmp_path):
    spec = {"preset_id": "detail_boost", "overrides": {"noise": -0.25, "blend": 0.4}}
    options = pipeline.request_delivery_options({"final_delivery": {"enhancement": spec}})
    assert options["enhancement"] == spec
    plan = pipeline.build_final_delivery_plan(tmp_path / "in.mp4", tmp_path / "out",
        profile_ids=["native_master"], enhancement=options["enhancement"])
    assert plan["enhance_preset"] == "detail_boost"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    command = pipeline.build_worker_command(jobs_root=tmp_path, child_job_id="child",
        input_list=tmp_path / "inputs.txt", plan_path=path)
    assert command[command.index("--preset-id") + 1] == "detail_boost"
    loaded = worker.load_delivery_plan(path)
    recipe = worker.delivery_enhancement_recipe(loaded, "auto")
    assert recipe.as_dict() == plan["enhancement_recipe"]
    assert ":noise=-0.25" in recipe.params


@pytest.mark.parametrize("spec", [None, [], True, {}, {"preset_id": "missing"},
    {"preset_id": True}, {"preset_id": "auto", "raw": "x"},
    {"preset_id": "auto", "overrides": None},
    {"preset_id": "auto", "overrides": []},
    {"preset_id": "clean_default", "overrides": {"blend": 0.2}},
    *[{"preset_id": "auto", "overrides": {"details": value}}
      for value in [True, "0.2", float("nan"), float("inf"), -1.01, 1.01, {}, []]],
    {"preset_id": "auto", "overrides": {"blend": -0.01}},
    {"preset_id": "auto", "overrides": {"blend": 1.01}},
    {"preset_id": "auto", "overrides": {"device": 0}},
])
def test_invalid_request_rejected(spec):
    with pytest.raises(ValueError, match="enhancement"):
        pipeline.request_delivery_options({"final_delivery": {"enhancement": spec}})


def test_isolation_and_canonical_numeric_params():
    before = copy.deepcopy(worker.PRESET_MODEL)
    a = worker.resolve_enhancement_recipe({"preset_id": "auto", "overrides": {"details": 1, "noise": -0.0}})
    b = worker.resolve_enhancement_recipe({"overrides": {"noise": 0, "details": 1.0}, "preset_id": "auto"})
    assert a == b
    assert worker.resolve_enhancement_recipe({"preset_id": "auto"}).params == before["auto"]["params"]
    assert worker.PRESET_MODEL == before
    with pytest.raises((AttributeError, TypeError)):
        a.params = "injected"


@pytest.mark.parametrize("control", ["details", "compression", "blur", "noise", "halo", "preblur", "blend"])
def test_all_control_bounds(control):
    low = 0 if control == "blend" else -1
    for value in [low, 1]:
        assert worker.resolve_enhancement_recipe({"preset_id": "auto", "overrides": {control: value}})
    for value in [low - 0.001, 1.001, True, None, float("-inf")]:
        with pytest.raises(ValueError, match="enhancement"):
            worker.resolve_enhancement_recipe({"preset_id": "auto", "overrides": {control: value}})


def test_concurrent_job_recipes_are_isolated(tmp_path):
    before = copy.deepcopy(worker.PRESET_MODEL)
    def plan_job(index):
        spec = {"preset_id": "ss_4k", "overrides": {"noise": index / 10}}
        code, result = pipeline.run_final_delivery([tmp_path / "pending.mp4"],
            jobs_root=tmp_path / "jobs", job_id=f"isolated_{index}", enhancement=spec)
        assert code == 0
        loaded = worker.load_delivery_plan(result["sources"][0]["plan_path"])
        assert loaded["enhancement_recipe"] == worker.resolve_enhancement_recipe(spec).as_dict()
        return loaded["enhancement_recipe"]["params"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        params = list(pool.map(plan_job, [1, 2]))
    assert len(set(params)) == 2
    assert worker.PRESET_MODEL == before


def test_exact_recipe_preflight_render_and_failure_propagation(monkeypatch, tmp_path):
    recipe = worker.resolve_enhancement_recipe({"preset_id": "compression_fix", "overrides": {"preblur": -1, "halo": 1}})
    commands = []
    writer = SimpleNamespace(command=commands.append, log=lambda _: None, cancel_requested=False)
    monkeypatch.setattr(worker, "_run_cancellable_subprocess", lambda argv, **kw: SimpleNamespace(returncode=0, stderr="", stdout="", emitted_frames=1))
    assert worker.validate_topaz_filter_model("ffmpeg", "tvai_up", recipe.up_model, writer, recipe=recipe) == (True, "")
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry", lambda argv, *a, **kw: commands.append(argv))
    worker.topaz_upscale_to_prores(tmp_path / "in", tmp_path / "out", "ffmpeg", {}, recipe.preset_id, 64, 64, writer, recipe=recipe)
    assert commands[-1].index("-autorotate") < commands[-1].index("-i")
    for argv in commands:
        filt = argv[argv.index("-vf") + 1] if "-vf" in argv else argv[argv.index("-filter_complex") + 1]
        assert recipe.params in filt
        assert ":download=0" in filt
    writer.cancel_requested = True
    with pytest.raises(worker.JobCancelled):
        worker.validate_topaz_filter_model("ffmpeg", "tvai_up", recipe.up_model, writer, recipe=recipe)
    writer.cancel_requested = False
    def fail(*a, **kw):
        raise RuntimeError("encoder failed")
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry", fail)
    with pytest.raises(RuntimeError, match="encoder failed"):
        worker.topaz_upscale_to_prores(tmp_path / "in", tmp_path / "out", "ffmpeg", {}, recipe.preset_id, 64, 64, writer, recipe=recipe)


def test_checkpoint_and_parent_binding_invalidation(tmp_path):
    source = tmp_path / "source"
    runtime = tmp_path / "ffmpeg"
    source.write_bytes(b"source")
    runtime.write_bytes(b"runtime")
    kwargs = dict(stage="topaz_upscale", input_path=source, topaz_ffmpeg=runtime,
        preset_id="auto", model="prob-4", target_width=64, target_height=64)
    legacy = worker.build_topaz_checkpoint_fingerprint(**kwargs)
    assert "enhancement_recipe" not in legacy["contract"]
    assert legacy["fingerprint"] == hashlib.sha256(json.dumps(legacy["contract"], sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    a = worker.resolve_enhancement_recipe({"preset_id": "auto", "overrides": {"noise": 0.1}})
    b = worker.resolve_enhancement_recipe({"preset_id": "auto", "overrides": {"noise": 0.2}})
    first = worker.build_topaz_checkpoint_fingerprint(**kwargs, enhancement_recipe=a.as_dict())
    second = worker.build_topaz_checkpoint_fingerprint(**kwargs, enhancement_recipe=b.as_dict())
    assert len({legacy["fingerprint"], first["fingerprint"], second["fingerprint"]}) == 3
    plans = [pipeline.build_final_delivery_plan(source, tmp_path / "out", profile_ids=["native_master"],
        enhancement={"preset_id": "auto", "overrides": {"noise": n}}) for n in [0.1, 0.2]]
    assert pipeline._plan_fingerprint(plans[0]) != pipeline._plan_fingerprint(plans[1])
    assert worker.enhancement_resume_matches({"enhancement_recipe": a.as_dict()}, a)
    assert not worker.enhancement_resume_matches({"enhancement_recipe": a.as_dict()}, b)
    assert not worker.enhancement_resume_matches({}, a)
    assert not worker.enhancement_resume_matches({"enhancement_recipe": a.as_dict()},
        worker.delivery_enhancement_recipe({}, "auto"))
    for stage in ["topaz_upscale", "topaz_interpolate"]:
        left = worker.build_topaz_checkpoint_fingerprint(**{**kwargs, "stage": stage}, enhancement_recipe=a.as_dict())
        right = worker.build_topaz_checkpoint_fingerprint(**{**kwargs, "stage": stage}, enhancement_recipe=b.as_dict())
        assert left["fingerprint"] != right["fingerprint"]


def test_cli_request_forwarding_and_plan_only(tmp_path, capsys):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"inputs": [str(tmp_path / "pending.mp4")], "final_delivery": {
        "enhancement": {"preset_id": "ss_4k", "overrides": {"blend": 0.3}}}}), encoding="utf-8")
    assert runner.main(["--request", str(request), "--jobs-root", str(tmp_path / "jobs"), "--job-id", "b1"]) == 0
    result = json.loads((tmp_path / "jobs/b1/final_delivery_result.json").read_text())
    plan = worker.load_delivery_plan(result["sources"][0]["plan_path"])
    assert plan["enhance_preset"] == "ss_4k"
    assert ":blend=0.3" in plan["enhancement_recipe"]["params"]
    assert runner.main(["--input", str(tmp_path / "pending.mp4"), "--jobs-root", str(tmp_path / "jobs"),
        "--job-id", "b2", "--enhancement-preset", "compression_fix", "--enhancement-overrides", '{"noise":0.2}']) == 0

    request.write_text(json.dumps({"inputs": [str(tmp_path / "pending.mp4")], "final_delivery": {
        "delivery_set": "smart_finish"}}), encoding="utf-8")
    assert runner.main(["--request", str(request), "--jobs-root", str(tmp_path / "jobs"), "--job-id", "b3"]) == 0
    smart_result = json.loads((tmp_path / "jobs/b3/final_delivery_result.json").read_text())
    smart_plan = worker.load_delivery_plan(smart_result["sources"][0]["plan_path"])
    assert smart_plan["requested_profile_ids"] == ["smart_finish_master"]
    assert smart_plan["variants"][0]["geometry_policy"] == "source_display"


def test_tampered_plan_rejected(tmp_path):
    plan = pipeline.build_final_delivery_plan(tmp_path / "in", tmp_path / "out", profile_ids=["native_master"],
        enhancement={"preset_id": "auto"})
    plan["enhancement_recipe"]["params"] = ":details=100"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="enhancement"):
        worker.load_delivery_plan(path)


def test_unknown_geometry_policy_rejected_before_worker_discovery(tmp_path):
    plan = pipeline.build_final_delivery_plan(
        tmp_path / "in",
        tmp_path / "out",
        profile_ids=["smart_finish_master"],
    )
    plan["variants"][0]["geometry_policy"] = "guess_source_geometry"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported delivery geometry policy"):
        worker.load_delivery_plan(path)


def test_parent_result_validates_source_geometry_proof(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"verified source identity")
    output = tmp_path / "final.mp4"
    output.write_bytes(b"verified final output identity")
    identity = pipeline._stable_file_identity(output)
    geometry = worker.source_display_geometry({"width": 720, "height": 576, "sar": "16:15", "rotation": 90})
    child_manifest_path = tmp_path / "child_manifest.json"
    child_manifest_path.write_text(json.dumps({
        "ready_sources": [{
            "source_path": str(source),
            "selected_source_path": str(source),
            "probe": {"width": 720, "height": 576, "sar": "16:15", "rotation": 90},
        }],
    }), encoding="utf-8")
    payload = {
        "schema": pipeline.RESULT_SCHEMA,
        "state": "passed",
        "attempt_id": "smart-finish-proof",
        "errors": [],
        "final_outputs": [str(output)],
        "final_output_proofs": [{
            "path": str(output),
            "bytes": identity["bytes"],
            "sha256": identity["sha256"],
            "width": geometry["output_width"],
            "height": geometry["output_height"],
            "geometry_policy": "source_display",
            "source_geometry": geometry,
            "probe": {
                "width": geometry["output_width"],
                "height": geometry["output_height"],
                "sar": "1:1",
                "rotation": 0,
            },
            "watermark_enabled": False,
        }],
        "sources": [{
            "source_path": str(source),
            "source_identity": pipeline._stable_file_identity(source),
            "child_manifest_path": str(child_manifest_path),
            "outputs": [str(output)],
            "state": "passed",
        }],
    }
    result_path = tmp_path / "final_delivery_result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    assert pipeline.validate_final_delivery_result(result_path)["status"] == "PASS"

    payload["final_output_proofs"][0]["source_geometry"]["output_width"] = 640
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    invalid = pipeline.validate_final_delivery_result(result_path)
    assert invalid["status"] == "FAIL"
    assert "final_delivery_source_geometry_proof_invalid" in invalid["issues"]

    payload["final_output_proofs"][0]["source_geometry"]["source_rotation_degrees"] = 180
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    invalid_rotation = pipeline.validate_final_delivery_result(result_path)
    assert invalid_rotation["status"] == "FAIL"
    assert "final_delivery_source_geometry_proof_invalid" in invalid_rotation["issues"]

    geometry = worker.source_display_geometry({"width": 720, "height": 576, "sar": "16:15", "rotation": 90})
    payload["final_output_proofs"][0]["source_geometry"] = geometry
    payload["final_output_proofs"][0]["probe"]["sar"] = "16:15"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    invalid_output_sar = pipeline.validate_final_delivery_result(result_path)
    assert invalid_output_sar["status"] == "FAIL"
    assert "final_delivery_source_geometry_proof_invalid" in invalid_output_sar["issues"]

    payload["final_output_proofs"][0]["probe"]["sar"] = "1:1"
    payload["final_output_proofs"][0]["probe"]["rotation"] = 90
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    invalid_output_rotation = pipeline.validate_final_delivery_result(result_path)
    assert invalid_output_rotation["status"] == "FAIL"
    assert "final_delivery_source_geometry_proof_invalid" in invalid_output_rotation["issues"]

    payload["final_output_proofs"][0]["probe"]["rotation"] = 0
    payload["final_output_proofs"][0]["probe"]["rotation_metadata_error"] = "unreadable"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    invalid_output_rotation_metadata = pipeline.validate_final_delivery_result(result_path)
    assert invalid_output_rotation_metadata["status"] == "FAIL"
    assert "final_delivery_source_geometry_proof_invalid" in invalid_output_rotation_metadata["issues"]
    payload["final_output_proofs"][0]["probe"].pop("rotation_metadata_error")

    child_manifest_path.write_text(json.dumps({
        "ready_sources": [{
            "source_path": str(source),
            "selected_source_path": str(source),
            "probe": {"width": 720, "height": 576, "sar": "1:1", "rotation": 90},
        }],
    }), encoding="utf-8")
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    invalid_source_reference = pipeline.validate_final_delivery_result(result_path)
    assert invalid_source_reference["status"] == "FAIL"
    assert "final_delivery_source_geometry_source_probe_invalid" in invalid_source_reference["issues"]

    child_manifest_path.write_text(json.dumps({
        "ready_sources": [{
            "source_path": str(source),
            "selected_source_path": str(source),
            "probe": {
                "width": 720, "height": 576, "sar": "16:15", "rotation": 90,
                "rotation_metadata_error": "unreadable",
            },
        }],
    }), encoding="utf-8")
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    invalid_source_rotation_metadata = pipeline.validate_final_delivery_result(result_path)
    assert invalid_source_rotation_metadata["status"] == "FAIL"
    assert "final_delivery_source_geometry_source_probe_invalid" in invalid_source_rotation_metadata["issues"]


def test_delivery_geometry_contract_normalizes_presets_and_fails_closed_on_crop():
    geometry = worker.normalize_delivery_geometry({
        "working": {"mode": "preset", "preset": "4k"},
        "final": {"mode": "exact", "width": 1081, "height": 1921, "layout": "fit_into_frame"},
    })
    assert geometry == {
        "schema": "spiffy_final_delivery_geometry_v1",
        "working": {"mode": "preset", "preset": "4k", "long_edge": 3840},
        "final": {"mode": "exact", "width": 1081, "height": 1921, "layout": "fit_into_frame"},
    }

    for invalid in (
        {"final": {"mode": "exact", "width": True, "height": 1920, "layout": "fit_into_frame"}},
        {"final": {"mode": "exact", "width": 1080, "height": 1920, "layout": "fill_crop_center"}},
        {"working": {"mode": "preset", "preset": "cinema-2k"}},
        {"working": {"mode": "exact", "width": 1920, "height": 1080, "layout": "fit_into_frame", "allow_downscale": "yes"}},
    ):
        with pytest.raises(ValueError):
            worker.normalize_delivery_geometry(invalid)
    with pytest.raises(ValueError, match="caption-collision proof"):
        worker.normalize_delivery_geometry({
            "final": {"mode": "exact", "width": 1080, "height": 1920, "layout": "fill_crop_center"},
        })
    with pytest.raises(ValueError, match="smart_finish_master"):
        pipeline.build_final_delivery_plan(
            "source.mp4", "delivery", profile_ids=["native_master"],
            geometry={"working": {"mode": "preset", "preset": "2k"}},
        )
    with pytest.raises(ValueError, match="isolated legacy native_master"):
        pipeline.build_final_delivery_plan(
            "source.mp4", "delivery", profile_ids=["smart_finish_master", "native_master"],
            geometry={"working": {"mode": "preset", "preset": "2k"}},
        )


def test_final_delivery_geometry_plan_and_autoexport_request_round_trip(tmp_path, capsys):
    geometry = {
        "working": {"mode": "preset", "preset": "4k"},
        "final": {"mode": "exact", "width": 1081, "height": 1921, "layout": "fit_into_frame"},
    }
    source = tmp_path / "pending.mp4"
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "inputs": [str(source)],
        "final_delivery": {"delivery_set": "smart_finish", "geometry": geometry},
    }), encoding="utf-8")

    assert runner.main([
        "--request", str(request), "--jobs-root", str(tmp_path / "jobs"), "--job-id", "geometry-request",
    ]) == 0
    result = json.loads((tmp_path / "jobs/geometry-request/final_delivery_result.json").read_text())
    plan = worker.load_delivery_plan(result["sources"][0]["plan_path"])
    assert plan["schema"] == "spiffy_final_delivery_plan_v4"
    assert plan["geometry"]["working"]["long_edge"] == 3840
    assert plan["variants"][0]["geometry_policy"] == "custom_final"
    assert plan["variants"][0]["width"] == plan["variants"][0]["height"] == 0
    assert plan["variants"][0]["geometry"] == plan["geometry"]["final"]

    original_fingerprint = pipeline._plan_fingerprint(plan)
    changed = dict(plan)
    changed["geometry"] = worker.normalize_delivery_geometry({
        "working": {"mode": "preset", "preset": "3k"},
        "final": geometry["final"],
    })
    assert pipeline._plan_fingerprint(changed) != original_fingerprint

    tampered_plan = dict(plan)
    tampered_plan["geometry"] = json.loads(json.dumps(plan["geometry"]))
    tampered_plan["geometry"]["working"]["long_edge"] = 4000
    tampered_path = tmp_path / "tampered-plan.json"
    tampered_path.write_text(json.dumps(tampered_plan), encoding="utf-8")
    with pytest.raises(ValueError, match="long_edge"):
        worker.load_delivery_plan(tampered_path)

    cli_geometry = {"working": {"mode": "long_edge", "long_edge": 2560}}
    assert runner.main([
        "--input", str(source), "--delivery-set", "smart_finish", "--geometry", json.dumps(cli_geometry),
        "--jobs-root", str(tmp_path / "jobs"), "--job-id", "geometry-cli",
    ]) == 0
    cli_result = json.loads((tmp_path / "jobs/geometry-cli/final_delivery_result.json").read_text())
    cli_plan = worker.load_delivery_plan(cli_result["sources"][0]["plan_path"])
    assert cli_plan["geometry"]["working"] == {"mode": "long_edge", "long_edge": 2560}
    assert capsys.readouterr().out


def test_working_geometry_uses_display_orientation_and_requires_downscale_opt_in(tmp_path):
    preset_plan = pipeline.build_final_delivery_plan(
        tmp_path / "source.mp4", tmp_path / "delivery", profile_ids=["smart_finish_master"],
        geometry={"working": {"mode": "preset", "preset": "2k"}},
    )
    anamorphic_rotated = {"width": 720, "height": 576, "sar": "16:15", "rotation": 90}
    working = worker.resolve_delivery_working_geometry(
        anamorphic_rotated, "4k_60fps", "keep_source_aspect", preset_plan
    )
    assert (working["target_width"], working["target_height"]) == (1920, 2560)
    assert (working["source_display"]["display_width"], working["source_display"]["display_height"]) == (540, 720)
    assert working["layout"] == "fit_into_frame"

    higher_resolution = worker.resolve_delivery_working_geometry(
        {"width": 4096, "height": 2160, "sar": "1:1", "rotation": 0},
        "4k_60fps", "keep_source_aspect", preset_plan,
    )
    assert (higher_resolution["target_width"], higher_resolution["target_height"]) == (4096, 2160)

    odd_plan = pipeline.build_final_delivery_plan(
        tmp_path / "source.mp4", tmp_path / "delivery", profile_ids=["smart_finish_master"],
        geometry={"working": {
            "mode": "exact", "width": 1281, "height": 721, "layout": "fit_into_frame",
        }},
    )
    odd_working = worker.resolve_delivery_working_geometry(
        {"width": 768, "height": 576, "sar": "1:1", "rotation": 0},
        "4k_60fps", "keep_source_aspect", odd_plan,
    )
    assert (odd_working["target_width"], odd_working["target_height"]) == (1282, 722)
    assert odd_working["requested_canvas"] == {"width": 1281, "height": 721}
    assert odd_working["alignment_padding"] == {"horizontal": 1, "vertical": 1, "placement": "centered"}

    no_downscale = pipeline.build_final_delivery_plan(
        tmp_path / "source.mp4", tmp_path / "delivery", profile_ids=["smart_finish_master"],
        geometry={"working": {
            "mode": "exact", "width": 1280, "height": 720,
            "layout": "fit_into_frame",
        }},
    )
    with pytest.raises(ValueError, match="allow_downscale"):
        worker.resolve_delivery_working_geometry(
            {"width": 1920, "height": 1080, "sar": "1:1", "rotation": 0},
            "4k_60fps", "keep_source_aspect", no_downscale,
        )

    allow_downscale = pipeline.build_final_delivery_plan(
        tmp_path / "source.mp4", tmp_path / "delivery", profile_ids=["smart_finish_master"],
        geometry={"working": {
            "mode": "exact", "width": 1280, "height": 720,
            "layout": "fit_into_frame", "allow_downscale": True,
        }},
    )
    resized = worker.resolve_delivery_working_geometry(
        {"width": 1920, "height": 1080, "sar": "1:1", "rotation": 0},
        "4k_60fps", "keep_source_aspect", allow_downscale,
    )
    assert (resized["target_width"], resized["target_height"]) == (1280, 720)
    assert resized["downscale_authorized"] is True


def test_custom_final_geometry_resolves_exact_padding_and_long_edge_presets(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    source_info = {"width": 720, "height": 576, "sar": "16:15", "rotation": 90}
    inputs = tmp_path / "inputs.txt"
    inputs.write_text(str(source), encoding="utf-8")
    args = worker.parse_args(["--input-list", str(inputs), "--watermark", "off", "--preset-id", "4k_60fps"])

    exact_plan = pipeline.build_final_delivery_plan(
        source, tmp_path / "delivery", profile_ids=["smart_finish_master"],
        geometry={"final": {
            "mode": "exact", "width": 1081, "height": 1921, "layout": "fit_into_frame",
        }},
    )
    exact_path = tmp_path / "exact-plan.json"
    exact_path.write_text(json.dumps(exact_plan), encoding="utf-8")
    exact_loaded = worker.load_delivery_plan(exact_path)
    exact_variant = worker.delivery_variants_for_source(
        source, source_info, args, tmp_path, exact_loaded,
    )[0]
    assert (exact_variant["width"], exact_variant["height"]) == (1082, 1922)
    assert exact_variant["layout"] == "fit_into_frame"
    assert exact_variant["geometry_policy"] == "custom_final"
    assert exact_variant["source_geometry"]["requested_canvas"] == {"width": 1081, "height": 1921}
    assert exact_variant["source_geometry"]["alignment_padding"] == {
        "horizontal": 1, "vertical": 1, "placement": "centered",
    }

    preset_plan = pipeline.build_final_delivery_plan(
        source, tmp_path / "delivery", profile_ids=["smart_finish_master"],
        geometry={"final": {"mode": "preset", "preset": "2k"}},
    )
    preset_variant = worker.delivery_variants_for_source(
        source, source_info, args, tmp_path, preset_plan,
    )[0]
    assert (preset_variant["width"], preset_variant["height"]) == (1920, 2560)
    assert preset_variant["layout"] == "fit_into_frame"


def test_custom_working_geometry_is_applied_before_topaz_and_bound_to_checkpoint(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    topaz_ffmpeg = tmp_path / "topaz-ffmpeg.exe"
    topaz_ffmpeg.write_bytes(b"topaz")
    plan = pipeline.build_final_delivery_plan(
        source, tmp_path / "delivery", profile_ids=["smart_finish_master"],
        geometry={"working": {
            "mode": "exact", "width": 1280, "height": 720, "layout": "fit_into_frame",
        }},
    )
    info = {"width": 720, "height": 576, "sar": "4:3", "rotation": 90, "has_audio": False}
    working = worker.resolve_delivery_working_geometry(info, "4k_60fps", "keep_source_aspect", plan)
    captured = []
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry",
                        lambda argv, *args, **kwargs: captured.append(list(argv)))
    writer = worker.ProgressWriter(tmp_path / "jobs", "geometry-filter", "Geometry filter", "test")
    worker.topaz_upscale_to_prores(
        source, tmp_path / "working.mov", "topaz-ffmpeg", info,
        "4k_60fps", 1280, 720, writer, working_geometry=working,
    )
    argv = captured[0]
    graph = argv[argv.index("-filter_complex") + 1]
    source_display = working["source_display"]
    assert graph.index(
        f"scale={source_display['display_width']}:{source_display['display_height']}:flags=lanczos,setsar=1"
    ) < graph.index("tvai_up=")
    assert "force_original_aspect_ratio=decrease" in graph
    assert "pad=1280:720" in graph

    fingerprint = worker.build_topaz_checkpoint_fingerprint(
        stage="topaz_upscale", input_path=source, topaz_ffmpeg=str(topaz_ffmpeg),
        preset_id="4k_60fps", model="prob-4", target_width=1280, target_height=720,
        working_geometry=working,
    )
    changed = dict(working)
    changed["layout"] = "keep_source_aspect"
    changed_fingerprint = worker.build_topaz_checkpoint_fingerprint(
        stage="topaz_upscale", input_path=source, topaz_ffmpeg=str(topaz_ffmpeg),
        preset_id="4k_60fps", model="prob-4", target_width=1280, target_height=720,
        working_geometry=changed,
    )
    assert fingerprint["fingerprint"] != changed_fingerprint["fingerprint"]


def test_worker_main_records_custom_geometry_and_builds_fit_filter_before_topaz(monkeypatch, tmp_path):
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    assert Path(ffmpeg).is_file() and Path(ffprobe).is_file()
    source = tmp_path / "source.mp4"
    subprocess.run([
        ffmpeg, "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=s=720x576:r=30:d=0.1", "-vf", "setsar=16/15",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], check=True, capture_output=True, timeout=20)
    source_info = worker.probe_video(source, ffprobe)
    plan = pipeline.build_final_delivery_plan(
        source, tmp_path / "delivery", profile_ids=["smart_finish_master"],
        geometry={
            "working": {"mode": "exact", "width": 1280, "height": 720, "layout": "fit_into_frame"},
            "final": {"mode": "exact", "width": 1081, "height": 1921, "layout": "fit_into_frame"},
        },
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    inputs = tmp_path / "inputs.txt"
    inputs.write_text(str(source), encoding="utf-8")
    runtime = tmp_path / "topaz-not-executed.exe"
    runtime.write_bytes(b"identity only; never executed")
    monkeypatch.setattr(worker, "find_topaz_video", lambda: {
        "ffmpeg": str(runtime), "ffprobe": worker.standard_ffprobe(),
        "tvai_up": True, "tvai_fi": True, "model_dir": str(tmp_path),
    })
    class NoopGpuLock:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(worker, "SharedGpuLock", NoopGpuLock)
    monkeypatch.setattr(worker, "configure_topaz_model_environment", lambda path: path)
    monkeypatch.setattr(worker, "wait_for_render_ready", lambda *args: (source, source_info))
    checkpoints = []
    real_fingerprint = worker.build_topaz_checkpoint_fingerprint

    def capture_fingerprint(**kwargs):
        checkpoints.append(kwargs)
        return real_fingerprint(**kwargs)

    monkeypatch.setattr(worker, "build_topaz_checkpoint_fingerprint", capture_fingerprint)
    monkeypatch.setattr(worker, "validate_topaz_filter_model", lambda *args, **kwargs: (True, "test preflight only"))
    argv_seen = []

    def stop_before_inference(argv, *args, **kwargs):
        argv_seen.append(list(argv))
        raise RuntimeError("geometry contract test: Topaz inference deliberately not executed")

    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry", stop_before_inference)
    monkeypatch.setattr(sys, "argv", [
        "video_upscale", "--jobs-root", str(tmp_path / "jobs"), "--job-id", "geometry-worker",
        "--input-list", str(inputs), "--delivery-plan", str(plan_path),
        "--preset-id", "4k_60fps", "--watermark", "off",
    ])
    assert worker.main() == 1

    manifest = json.loads((tmp_path / "jobs/geometry-worker/manifest.json").read_text(encoding="utf-8"))
    expected_working = worker.resolve_delivery_working_geometry(
        source_info, "4k_60fps", "keep_source_aspect", plan,
    )
    assert manifest["ready_sources"][0]["working_geometry"] == expected_working
    assert checkpoints[0]["working_geometry"] == expected_working
    assert len(argv_seen) == 1
    graph = argv_seen[0][argv_seen[0].index("-filter_complex") + 1]
    source_display = expected_working["source_display"]
    assert graph.index(
        f"scale={source_display['display_width']}:{source_display['display_height']}:flags=lanczos,setsar=1"
    ) < graph.index("tvai_up=")
    assert "pad=1280:720" in graph
    assert manifest["final_outputs"] == []


def test_parent_revalidates_custom_final_geometry_against_source_probe(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"verified source")
    output = tmp_path / "final.mp4"
    output.write_bytes(b"verified final")
    source_info = {"width": 720, "height": 576, "sar": "16:15", "rotation": 90}
    plan = pipeline.build_final_delivery_plan(
        source, tmp_path / "delivery", profile_ids=["smart_finish_master"],
        geometry={"final": {
            "mode": "exact", "width": 1081, "height": 1921, "layout": "fit_into_frame",
        }},
    )
    geometry_proof = worker.resolve_custom_final_geometry(source_info, plan["geometry"]["final"])
    working_proof = worker.resolve_delivery_working_geometry(
        source_info, plan["enhance_preset"], "keep_source_aspect", plan,
    )
    child_manifest = tmp_path / "child_manifest.json"
    child_manifest.write_text(json.dumps({
        "delivery_plan": plan,
        "ready_sources": [{
            "source_path": str(source), "selected_source_path": str(source), "probe": source_info,
            "working_geometry": working_proof,
        }],
    }), encoding="utf-8")
    identity = pipeline._stable_file_identity(output)
    payload = {
        "schema": pipeline.RESULT_SCHEMA,
        "state": "passed",
        "attempt_id": "custom-geometry-proof",
        "geometry": plan["geometry"],
        "errors": [],
        "final_outputs": [str(output)],
        "final_output_proofs": [{
            "path": str(output), "bytes": identity["bytes"], "sha256": identity["sha256"],
            "width": geometry_proof["output_width"], "height": geometry_proof["output_height"],
            "geometry_policy": "custom_final", "geometry": plan["geometry"]["final"],
            "source_geometry": geometry_proof,
            "working_geometry": working_proof,
            "probe": {
                "width": geometry_proof["output_width"], "height": geometry_proof["output_height"],
                "sar": "1:1", "rotation": 0,
            },
            "watermark_enabled": False,
        }],
        "sources": [{
            "source_path": str(source), "source_identity": pipeline._stable_file_identity(source),
            "child_manifest_path": str(child_manifest), "outputs": [str(output)], "state": "passed",
        }],
    }
    result_path = tmp_path / "final_delivery_result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    validation = pipeline.validate_final_delivery_result(result_path)
    assert validation["status"] == "PASS", validation

    payload["final_output_proofs"][0]["source_geometry"]["output_width"] += 2
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    invalid = pipeline.validate_final_delivery_result(result_path)
    assert invalid["status"] == "FAIL"
    assert "final_delivery_custom_geometry_proof_invalid" in invalid["issues"]

@pytest.mark.parametrize("preset_id", list(worker.PRESET_MODEL))
def test_existing_presets_and_legacy_plan_identity(tmp_path, preset_id):
    recipe = worker.resolve_enhancement_recipe({"preset_id": preset_id})
    assert recipe.params == worker.PRESET_MODEL[preset_id]["params"]
    assert recipe.up_model == worker.PRESET_MODEL[preset_id]["up_model"]
    legacy = pipeline.build_final_delivery_plan(tmp_path / "in", tmp_path / "out", profile_ids=["native_master"])
    assert "enhancement" not in legacy
    assert "enhancement_recipe" not in legacy
    assert "enhancement_recipe" not in pipeline._plan_binding(legacy)


@pytest.mark.parametrize("container", [None, [], True])
def test_malformed_delivery_container(container):
    with pytest.raises(ValueError, match="enhancement"):
        pipeline.request_delivery_options({"final_delivery": container})


def test_worker_main_recipe_reaches_real_argv_without_running_topaz(monkeypatch, tmp_path):
    """Real CPU source/decode and final encode; Topaz attempt deliberately stopped.

    No fabricated Topaz output or successful enhancement claim is used here.
    """
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    assert Path(ffmpeg).is_file() and Path(ffprobe).is_file()
    source = tmp_path / "source.mp4"
    subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=s=64x64:r=60:d=0.1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
        check=True, capture_output=True, timeout=20)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    info = worker.probe_video(source, ffprobe)
    plan = pipeline.build_final_delivery_plan(source, tmp_path / "out", profile_ids=["native_master"],
        enhancement={"preset_id": "ss_4k", "overrides": {"noise": -0.2, "blend": 0.3}})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    inputs = tmp_path / "inputs.txt"
    inputs.write_text(str(source))
    runtime = tmp_path / "topaz-not-executed.exe"
    runtime.write_bytes(b"identity-only: never executed")
    monkeypatch.setattr(worker, "find_topaz_video", lambda: {"ffmpeg": str(runtime), "ffprobe": ffprobe,
        "tvai_up": True, "tvai_fi": True, "model_dir": str(tmp_path)})
    monkeypatch.setattr(worker, "configure_topaz_model_environment", lambda p: p)
    monkeypatch.setattr(worker, "wait_for_render_ready", lambda *a: (source, info))
    checkpoints = []
    real_fingerprint = worker.build_topaz_checkpoint_fingerprint
    def capture_fingerprint(**kw):
        checkpoints.append(kw)
        return real_fingerprint(**kw)
    monkeypatch.setattr(worker, "build_topaz_checkpoint_fingerprint", capture_fingerprint)
    seen = []
    def preflight(exe, filter_name, model, writer, **kw):
        seen.append(kw["recipe"].as_dict())
        return True, "contract test: Topaz not executed"
    monkeypatch.setattr(worker, "validate_topaz_filter_model", preflight)
    argv_seen = []
    def stop_topaz(argv, *args, **kw):
        argv_seen.append(argv)
        raise RuntimeError("B1 deliberate stop before Topaz execution")
    monkeypatch.setattr(worker, "run_topaz_process_with_first_use_retry", stop_topaz)
    monkeypatch.setattr(sys, "argv", ["video_upscale", "--jobs-root", str(tmp_path / "jobs"),
        "--job-id", "worker", "--input-list", str(inputs), "--delivery-plan", str(plan_path),
        "--preset-id", "4k_60fps", "--watermark", "off"])
    assert worker.main() == 1
    assert seen == [plan["enhancement_recipe"]]
    assert checkpoints[0]["enhancement_recipe"] == seen[0]
    assert checkpoints[0]["model_params"] == seen[0]["params"]
    assert len(argv_seen) == 1
    assert plan["enhancement_recipe"]["params"] in argv_seen[0][argv_seen[0].index("-filter_complex") + 1]
    manifest = json.loads((tmp_path / "jobs/worker/manifest.json").read_text())
    assert manifest["enhancement_recipe"] == plan["enhancement_recipe"]
    assert manifest["preset_id"] == "ss_4k"
    progress = json.loads((tmp_path / "jobs/worker/progress.json").read_text())
    assert progress["state"] == "error"
    assert manifest["final_outputs"] == []

    # A changed retry cannot reconcile old completed sources or checkpoints.
    changed_plan = pipeline.build_final_delivery_plan(source, tmp_path / "out", profile_ids=["native_master"],
        enhancement={"preset_id": "ss_4k", "overrides": {"noise": 0.4}})
    plan_path.write_text(json.dumps(changed_plan))
    seen_carry = []
    real_carry = worker.build_completed_source_carry_forward
    def capture_carry(manifest, **kw):
        seen_carry.append(manifest)
        return real_carry(manifest, **kw)
    monkeypatch.setattr(worker, "build_completed_source_carry_forward", capture_carry)
    retry_argv = [*sys.argv]
    retry_argv[retry_argv.index("--job-id") + 1] = "retry"
    retry_argv.extend(["--resume-job-id", "worker", "--reconcile-completed-sources"])
    monkeypatch.setattr(sys, "argv", retry_argv)
    assert worker.main() == 1
    assert seen_carry == [{}]
    assert seen[-1] == changed_plan["enhancement_recipe"]
    assert checkpoints[-1]["enhancement_recipe"] == changed_plan["enhancement_recipe"]

    # Cancellation and rejected preflight publish no successful output.
    def cancel_preflight(*a, **kw):
        raise worker.JobCancelled("test cancellation before inference")
    monkeypatch.setattr(worker, "validate_topaz_filter_model", cancel_preflight)
    assert worker.main() == 2
    assert json.loads((tmp_path / "jobs/retry/progress.json").read_text())["state"] == "cancelled"
    monkeypatch.setattr(worker, "validate_topaz_filter_model", lambda *a, **kw: (False, "model unavailable"))
    assert worker.main() == 3
    assert json.loads((tmp_path / "jobs/retry/progress.json").read_text())["state"] == "error"
    assert len(argv_seen) == 2

    # The real standard-FFmpeg downstream encode consumes this plan's variant;
    # source is explicitly used here, never labelled an enhanced intermediate.
    args = worker.parse_args(["--input-list", str(inputs), "--watermark", "off", "--preset-id", "ss_4k"])
    variants = worker.delivery_variants_for_source(source, info, args, tmp_path, plan)
    variant = variants[0]
    dest = tmp_path / "cpu-delivery-contract.mp4"
    writer = worker.ProgressWriter(tmp_path / "jobs", "cpu_encode", "B1 CPU encode", "test")
    worker.final_mp4(source, dest, ffmpeg, info, variant["layout"], variant["width"], variant["height"],
        variant["crf"], "ultrafast", writer, max_fps=variant["max_fps"])
    subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(dest), "-f", "null", "-"],
        check=True, capture_output=True, timeout=20)
    decoded = worker.probe_video(dest, ffprobe)
    assert (decoded["width"], decoded["height"]) == (variant["width"], variant["height"])
    assert decoded["fps"] == 60
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_invalid_worker_plan_fails_before_discovery(monkeypatch, tmp_path):
    plan = {"schema": worker.DELIVERY_PLAN_SCHEMA, "variants": [{}],
            "enhancement": {"preset_id": "auto", "overrides": {"noise": True}}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    inputs = tmp_path / "inputs.txt"
    inputs.write_text(str(tmp_path / "pending.mp4"))
    def forbidden():
        pytest.fail("Malformed enhancement must fail before runtime/model discovery")
    monkeypatch.setattr(worker, "find_topaz_video", forbidden)
    monkeypatch.setattr(sys, "argv", ["worker", "--jobs-root", str(tmp_path / "jobs"), "--job-id", "invalid",
        "--input-list", str(inputs), "--delivery-plan", str(path)])
    assert worker.main() == 1
    assert json.loads((tmp_path / "jobs/invalid/progress.json").read_text())["state"] == "error"


def test_smart_finish_source_geometry_resolves_sar_rotation_and_even_padding():
    even = worker.source_display_geometry({"width": 1920, "height": 1080, "sar": "1:1", "rotation": 0})
    assert (even["output_width"], even["output_height"]) == (1920, 1080)
    assert (even["display_width"], even["display_height"]) == (1920, 1080)
    assert even["alignment_padding"] == {"horizontal": 0, "vertical": 0, "placement": "centered"}

    odd = worker.source_display_geometry({"width": 641, "height": 359, "sar": "1:1", "rotation": 0})
    assert (odd["output_width"], odd["output_height"]) == (642, 360)
    assert odd["alignment_padding"] == {"horizontal": 1, "vertical": 1, "placement": "centered"}

    anamorphic = worker.source_display_geometry({"width": 720, "height": 576, "sar": "16:15", "rotation": 0})
    assert (anamorphic["display_width"], anamorphic["display_height"]) == (768, 576)
    assert anamorphic["source_sample_aspect_ratio"] == "16:15"
    assert anamorphic["display_aspect_ratio"] == "4:3"

    rotated = worker.source_display_geometry({"width": 1920, "height": 1080, "sar": "1:1", "rotation": 90})
    assert (rotated["display_width"], rotated["display_height"]) == (1080, 1920)
    assert rotated["source_rotation_degrees"] == 90

    anamorphic_rotated = worker.source_display_geometry(
        {"width": 720, "height": 576, "sar": "16:15", "rotation": -90}
    )
    assert (anamorphic_rotated["display_width"], anamorphic_rotated["display_height"]) == (540, 720)
    assert anamorphic_rotated["source_rotation_degrees"] == 270
    assert anamorphic_rotated["display_aspect_ratio"] == "3:4"

    odd_rotated = worker.source_display_geometry({"width": 641, "height": 359, "sar": "1:1", "rotation": 90})
    assert (odd_rotated["display_width"], odd_rotated["display_height"]) == (359, 641)
    assert (odd_rotated["output_width"], odd_rotated["output_height"]) == (360, 642)

    with pytest.raises(ValueError, match="cannot interpret source SAR"):
        worker.source_display_geometry({"width": 1920, "height": 1080, "sar": "N/A", "rotation": 0})
    with pytest.raises(ValueError, match="must be positive"):
        worker.source_display_geometry({"width": 1920, "height": 1080, "sar": "0:1", "rotation": 0})
    with pytest.raises(ValueError, match="orthogonal"):
        worker.source_display_geometry({"width": 1080, "height": 1920, "sar": "1:1", "rotation": 45})
    worker.validate_source_display_square_pixels({"sar": "1:1", "rotation": 0}, phase="test output")
    with pytest.raises(RuntimeError, match="square pixels"):
        worker.validate_source_display_square_pixels({"sar": "4:3", "rotation": 0}, phase="test output")
    with pytest.raises(RuntimeError, match="rotation metadata"):
        worker.validate_source_display_square_pixels({"sar": "1:1", "rotation": 90}, phase="test output")
    with pytest.raises(RuntimeError, match="unreadable rotation evidence"):
        worker.validate_source_display_square_pixels(
            {"sar": "1:1", "rotation": 0, "rotation_metadata_error": "unreadable"},
            phase="test output",
        )


@pytest.mark.parametrize("raw_rotation", ["90.5", "N/A"])
def test_source_display_geometry_rejects_unreliable_rotation_from_ffprobe(
    tmp_path, monkeypatch, raw_rotation,
):
    payload = {
        "format": {"duration": "1.0"},
        "streams": [{
            "codec_type": "video",
            "width": 720,
            "height": 576,
            "r_frame_rate": "25/1",
            "avg_frame_rate": "25/1",
            "sample_aspect_ratio": "16:15",
            "side_data_list": [{"rotation": raw_rotation}],
        }],
    }
    monkeypatch.setattr(media_probe, "run_json", lambda _argv: payload)
    info = worker.probe_video(tmp_path / "source.mp4", "ffprobe")

    with pytest.raises(ValueError, match="cannot interpret source rotation metadata"):
        worker.source_display_geometry(info)


def test_smart_finish_working_dimensions_use_display_orientation_only_for_source_display_plan(tmp_path):
    plan = pipeline.build_final_delivery_plan(
        tmp_path / "source.mp4", tmp_path / "delivery", profile_ids=["smart_finish_master"]
    )
    info = {"width": 72, "height": 48, "sar": "4:3", "rotation": 90}
    dimensions = worker.target_dimensions_for_delivery(info, "4k_60fps", "keep_source_aspect", plan)
    assert dimensions == (144, 288, 144, 288)
    assert worker.target_dimensions(info, "4k_60fps", "keep_source_aspect") == (288, 192, 288, 192)
    legacy_plan = pipeline.build_final_delivery_plan(
        tmp_path / "source.mp4", tmp_path / "delivery", profile_ids=["native_master"]
    )
    assert worker.target_dimensions_for_delivery(info, "4k_60fps", "keep_source_aspect", legacy_plan) == (
        288, 192, 288, 192
    )


def test_source_display_mixed_plan_fails_closed_only_when_working_dimensions_change(tmp_path):
    plan = pipeline.build_final_delivery_plan(
        tmp_path / "source.mp4",
        tmp_path / "delivery",
        profile_ids=["smart_finish_master", "native_master"],
    )
    path = tmp_path / "mixed-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded = worker.load_delivery_plan(path)
    square = {"width": 72, "height": 48, "sar": "1:1", "rotation": 0}
    assert worker.target_dimensions_for_delivery(square, "4k_60fps", "keep_source_aspect", loaded) == (
        288, 192, 288, 192
    )
    rotated_180 = {"width": 72, "height": 48, "sar": "1:1", "rotation": 180}
    assert worker.target_dimensions_for_delivery(rotated_180, "4k_60fps", "keep_source_aspect", loaded) == (
        288, 192, 288, 192
    )
    square_rotated_90 = {"width": 72, "height": 72, "sar": "1:1", "rotation": 90}
    assert worker.target_dimensions_for_delivery(
        square_rotated_90, "4k_60fps", "keep_source_aspect", loaded
    ) == worker.target_dimensions(square_rotated_90, "4k_60fps", "keep_source_aspect")
    with pytest.raises(ValueError, match="dedicated smart_finish_master-only"):
        worker.target_dimensions_for_delivery(
            {"width": 72, "height": 48, "sar": "4:3", "rotation": 90},
            "4k_60fps",
            "keep_source_aspect",
            loaded,
        )


def test_worker_rejects_unreadable_sar_before_topaz_preflight(monkeypatch, tmp_path):
    ffprobe = worker.standard_ffprobe()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture source identity")
    inputs = tmp_path / "inputs.txt"
    inputs.write_text(str(source), encoding="utf-8")
    plan = pipeline.build_final_delivery_plan(
        source,
        tmp_path / "delivery",
        profile_ids=["smart_finish_master"],
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    runtime = tmp_path / "topaz-not-executed.exe"
    runtime.write_bytes(b"identity only")
    monkeypatch.setattr(worker, "find_topaz_video", lambda: {
        "ffmpeg": str(runtime), "ffprobe": ffprobe, "tvai_up": True, "tvai_fi": True,
        "model_dir": str(tmp_path),
    })
    monkeypatch.setattr(worker, "configure_topaz_model_environment", lambda path: path)
    monkeypatch.setattr(worker, "wait_for_render_ready", lambda *args: (
        source, {"width": 1920, "height": 1080, "sar": "N/A", "rotation": 0}
    ))
    preflight_calls = []
    monkeypatch.setattr(worker, "validate_topaz_filter_model", lambda *args, **kwargs: preflight_calls.append(args))
    monkeypatch.setattr(sys, "argv", [
        "worker", "--jobs-root", str(tmp_path / "jobs"), "--job-id", "sar-rejected",
        "--input-list", str(inputs), "--delivery-plan", str(plan_path), "--preset-id", "4k_60fps",
        "--watermark", "off",
    ])
    assert worker.main() == 1
    assert preflight_calls == []
    progress = json.loads((tmp_path / "jobs/sar-rejected/progress.json").read_text(encoding="utf-8"))
    assert progress["state"] == "error"
    assert "cannot interpret source SAR" in json.dumps(progress)


def test_smart_finish_cpu_geometry_normalizes_anamorphic_rotated_input(tmp_path):
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    assert Path(ffmpeg).is_file() and Path(ffprobe).is_file()
    base = tmp_path / "anamorphic-source.mp4"
    source = tmp_path / "anamorphic-rotated-source.mp4"
    working = tmp_path / "normalized-working.mp4"
    output = tmp_path / "smart-finish-display.mp4"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=s=72x48:r=25:d=0.12", "-vf", "setsar=4/3", "-c:v", "libx264",
            "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an", str(base),
        ],
        check=True, capture_output=True, timeout=20,
    )
    subprocess.run(
        [ffmpeg, "-hide_banner", "-v", "error", "-display_rotation", "90", "-i", str(base),
         "-c", "copy", str(source)],
        check=True, capture_output=True, timeout=20,
    )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    source_info = worker.probe_video(source, ffprobe)
    assert (source_info["width"], source_info["height"]) == (72, 48)
    assert source_info["sar"] == "4:3"
    assert source_info["rotation"] == 90

    geometry = worker.source_display_geometry(source_info)
    assert (geometry["display_width"], geometry["display_height"]) == (36, 72)
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-v", "error", "-autorotate", "-i", str(source),
            "-vf", f"scale={geometry['output_width']}:{geometry['output_height']}:flags=lanczos,setsar=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an", str(working),
        ],
        check=True, capture_output=True, timeout=20,
    )
    working_info = worker.probe_video(working, ffprobe)
    assert (working_info["width"], working_info["height"]) == (36, 72)
    assert working_info["sar"] == "1:1"
    assert working_info["rotation"] == 0

    writer = worker.ProgressWriter(tmp_path / "jobs", "smart-finish-display", "Smart Finish display geometry", "test")
    worker.final_mp4(
        working, output, ffmpeg, working_info, "fit_into_frame", geometry["output_width"],
        geometry["output_height"], 17, "ultrafast", writer,
    )
    valid, output_info, reason = worker.verify_video(output, ffprobe)
    assert valid, reason
    assert (output_info["width"], output_info["height"]) == (36, 72)
    assert output_info["sar"] == "1:1"
    assert output_info["rotation"] == 0
    subprocess.run(
        [ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
        check=True, capture_output=True, timeout=20,
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_smart_finish_master_cpu_delivery_restores_source_geometry_and_audio(tmp_path):
    ffmpeg, ffprobe = worker.standard_ffmpeg(), worker.standard_ffprobe()
    assert Path(ffmpeg).is_file() and Path(ffprobe).is_file()
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=s=64x48:r=60:d=0.2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.2",
            "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", str(source),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    source_info = worker.probe_video(source, ffprobe)
    assert source_info["audio_stream_count"] == 1
    child_manifest_path = tmp_path / "child_manifest.json"
    child_manifest_path.write_text(json.dumps({
        "ready_sources": [{
            "source_path": str(source),
            "selected_source_path": str(source),
            "probe": source_info,
        }],
    }), encoding="utf-8")

    plan = pipeline.build_final_delivery_plan(
        source,
        tmp_path / "delivery",
        profile_ids=pipeline.profiles_for_delivery_set("smart_finish"),
    )
    plan_path = tmp_path / "delivery_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    loaded_plan = worker.load_delivery_plan(plan_path)
    inputs = tmp_path / "inputs.txt"
    inputs.write_text(str(source), encoding="utf-8")
    args = worker.parse_args(["--input-list", str(inputs), "--watermark", "off", "--preset-id", "4k_60fps"])
    variant = worker.delivery_variants_for_source(source, source_info, args, tmp_path, loaded_plan)[0]
    assert (variant["width"], variant["height"]) == (64, 48)
    assert variant["geometry_policy"] == "source_display"
    assert variant["source_geometry"]["policy"] == "source_display"

    working = tmp_path / "working.mp4"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-v", "error", "-i", str(source),
            "-vf", "scale=128:96:flags=lanczos,setsar=1",
            "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "0", "-pix_fmt", "yuv420p", "-c:a", "copy", str(working),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    working_info = worker.probe_video(working, ffprobe)
    writer = worker.ProgressWriter(tmp_path / "jobs", "smart_finish_geometry", "Smart Finish geometry", "test")
    output = tmp_path / "smart_finish_master.mp4"
    worker.final_mp4(
        working,
        output,
        ffmpeg,
        working_info,
        variant["layout"],
        variant["width"],
        variant["height"],
        variant["crf"],
        "ultrafast",
        writer,
        max_fps=variant["max_fps"],
    )
    valid, output_info, reason = worker.verify_video(output, ffprobe)
    assert valid, reason
    assert (output_info["width"], output_info["height"]) == (64, 48)
    assert output_info["fps"] == pytest.approx(60.0, abs=0.1)
    assert output_info["sar"] == "1:1"
    assert output_info["rotation"] == 0
    assert output_info["audio_stream_count"] == source_info["audio_stream_count"]
    subprocess.run(
        [ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
        check=True,
        capture_output=True,
        timeout=20,
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    output_identity = pipeline._stable_file_identity(output)
    result_path = tmp_path / "final_delivery_result.json"
    result_path.write_text(json.dumps({
        "schema": pipeline.RESULT_SCHEMA,
        "state": "passed",
        "attempt_id": "smart-finish-cpu-output",
        "errors": [],
        "final_outputs": [str(output)],
        "final_output_proofs": [{
            "path": str(output),
            "sha256": output_identity["sha256"],
            "bytes": output_identity["bytes"],
            "width": output_info["width"],
            "height": output_info["height"],
            "geometry_policy": variant["geometry_policy"],
            "source_geometry": variant["source_geometry"],
            "probe": output_info,
            "watermark_enabled": False,
        }],
        "sources": [{
            "source_path": str(source),
            "source_identity": pipeline._stable_file_identity(source),
            "child_manifest_path": str(child_manifest_path),
            "outputs": [str(output)],
            "state": "passed",
        }],
    }), encoding="utf-8")
    assert pipeline.validate_final_delivery_result(result_path)["status"] == "PASS"
