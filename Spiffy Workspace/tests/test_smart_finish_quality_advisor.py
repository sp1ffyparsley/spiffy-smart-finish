"""Pure, non-rendering checks for Smart Finish's quality recommendations."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
UPSCALE = ROOT / "tools" / "video_upscale"
if str(UPSCALE) not in sys.path:
    sys.path.insert(0, str(UPSCALE))

from quality_advisor import build_advice, inspect_comfyui, _weight_markers  # noqa: E402


def test_quality_first_never_treats_6k_raster_as_native_ai():
    result = build_advice({"width": 1920, "height": 1080, "duration": 10.0, "fps": 24.0})
    assert result["recommended"] == "4K"
    choices = {row["label"]: row for row in result["choices"]}
    assert (choices["4K"]["width"], choices["4K"]["height"]) == (3840, 2160)
    assert (choices["6K"]["width"], choices["6K"]["height"]) == (5760, 3240)
    assert choices["6K"]["final_canvas_allowed"] is True
    assert choices["6K"]["neural_detail_at_output_resolution"] is False
    assert choices["8K"]["neural_detail_at_output_resolution"] is False
    assert not choices["6K"]["recommended"] and not choices["8K"]["recommended"]
    assert all(row["seconds"] is None for row in result["choices"])


@pytest.mark.parametrize("policy,expected", [("best_quality", "4K"), ("balanced", "3K"), ("fastest_safe", "2K")])
def test_policies_only_select_within_existing_working_limit(policy, expected):
    result = build_advice({"width": 1080, "height": 1920}, policy=policy)
    assert result["recommended"] == expected
    assert sum(int(row["recommended"]) for row in result["choices"]) == 1


def test_source_above_working_limit_does_not_fake_a_recommendation():
    result = build_advice({"width": 6000, "height": 3376})
    assert result["recommended"] is None
    assert not any(row["native_working_canvas_allowed"] for row in result["choices"])


def test_output_pixel_budget_rejects_square_8k_but_not_6k():
    result = build_advice({"width": 1000, "height": 1000})
    choices = {row["label"]: row for row in result["choices"]}
    assert choices["6K"]["final_canvas_allowed"] is True
    assert choices["8K"]["final_canvas_allowed"] is False


def test_measured_time_requires_explicit_same_source_geometry_and_model():
    source = {"width": 1920, "height": 1080, "fps": 24.0,
              "source_fingerprint": "source-one", "runtime_fingerprint": "topaz-build-one"}
    invalid = [{"label": "4K", "width": 1920, "height": 1080, "model": "other", "verified": True, "elapsed_seconds": 19},
               {"label": "6K", "width": 1280, "height": 720, "model": "topaz", "verified": True, "elapsed_seconds": 30}]
    result = build_advice(source, observations=invalid)
    assert all(row["seconds"] is None for row in result["choices"])
    valid = invalid + [{"label": "4K", "width": 1920, "height": 1080,
                        "model": "topaz", "verified": True, "elapsed_seconds": 21.75,
                        "fps": 24.0, "source_fingerprint": "source-one", "runtime_fingerprint": "topaz-build-one"}]
    result = build_advice(source, observations=valid)
    assert next(row for row in result["choices"] if row["label"] == "4K")["seconds"] == 21.75
    assert all(row["seconds"] is None for row in build_advice(
        {**source, "source_fingerprint": "another-video"}, observations=valid)["choices"])


def test_faster_mode_does_not_downscale_a_larger_original():
    result = build_advice({"width": 3840, "height": 2160}, policy="fastest_safe")
    assert result["recommended"] == "4K"
    assert next(row for row in result["choices"] if row["label"] == "2K")["status"] == "would_downscale_source"


@pytest.mark.parametrize("unsafe,reason", [
    ({"has_alpha": True}, "alpha_video_not_supported_in_adaptive_delivery"),
    ({"color_transfer": "smpte2084"}, "hdr_requires_qualified_color_management"),
    ({"field_order": "tt"}, "interlaced_source_requires_qualified_deinterlacing"),
    ({"is_vfr": True}, "variable_frame_rate_requires_source_timing_qualification"),
    ({"sar": "4:3"}, "non_square_pixel_aspect_requires_separate_admission"),
    ({"rotation": 90}, "rotation_requires_separate_geometry_qualification"),
])
def test_advisor_does_not_recommend_an_incompatible_adaptive_route(unsafe, reason):
    result = build_advice({"width": 1920, "height": 1080, **unsafe})
    assert result["recommended"] is None
    assert reason in result["source_admission_issues"]
    assert all(row["status"] == "source_requires_review" for row in result["choices"])


@pytest.mark.parametrize("bad", ["http://example.com:8188", "https://127.0.0.1:8188", "http://localhost:8188/redirect", "file:///etc/passwd"])
def test_backend_check_is_loopback_only(bad):
    with pytest.raises(ValueError):
        inspect_comfyui(endpoint=bad)


def test_backend_uses_weight_evidence_without_claiming_inference(tmp_path):
    diffusion = tmp_path / "models" / "diffusion_models"
    diffusion.mkdir(parents=True)
    (diffusion / "relayvsr_x4_int8_convrot.safetensors").write_bytes(b"tiny-test-file")
    assert _weight_markers(tmp_path)["relayvsr"] is True
    result = inspect_comfyui(root=tmp_path, endpoint="http://127.0.0.1:1", timeout_seconds=.05)
    assert result["server_running"] is False
    assert result["checkpoint_filename_evidence"]["relayvsr"] is True
    assert result["models_production_qualified"] is False


def test_real_gan_local_readiness_does_not_require_an_already_running_server(tmp_path, monkeypatch):
    import comfyui_model_catalog as catalog
    from comfyui_video_backend import MODEL_NAME, MODEL_SHA256
    model = tmp_path / "models" / "upscale_models" / MODEL_NAME
    model.parent.mkdir(parents=True)
    model.write_bytes(b"test-checkpoint-file-not-actual-weights")
    python = tmp_path / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    (tmp_path / "main.py").touch()
    monkeypatch.setattr(catalog.hashlib, "file_digest",
                        lambda *_args, **_kwargs: SimpleNamespace(hexdigest=lambda: MODEL_SHA256))
    result = inspect_comfyui(root=tmp_path, endpoint="http://127.0.0.1:1", timeout_seconds=.05)
    assert result["server_running"] is False
    assert result["local_gan"]["pinned_checkpoint_verified"] is True
    assert result["local_gan"]["available_for_source_preflight"] is True
    assert result["models_production_qualified"] is False


def test_advisor_shows_real_6k_gan_candidate_without_faking_auto_model_choice():
    comfy = {"local_gan": {"available_for_source_preflight": True}}
    result = build_advice({"width": 1920, "height": 1080, "duration": 4,
                           "fps": 24}, comfyui=comfy)
    choices = {row["label"]: row for row in result["choices"]}
    assert choices["6K"]["comfyui_gan"]["candidate"] is True
    assert choices["6K"]["native_working_canvas_allowed"] is False
    assert choices["6K"]["comfyui_gan"]["temporal_video_model"] is False
    assert choices["6K"]["comfyui_gan"]["seconds"] is None
    assert choices["8K"]["comfyui_gan"]["candidate"] is False
    assert result["recommended"] == "4K"
    assert build_advice({"width": 1920, "height": 1080, "duration": 61,
                         "fps": 24}, comfyui=comfy)["choices"][3]["comfyui_gan"]["candidate"] is False
    assert build_advice({"width": 3840, "height": 2160, "duration": 3,
                         "fps": 24}, comfyui=comfy)["choices"][3]["comfyui_gan"]["candidate"] is False


def test_advisor_manifest_is_read_only_command_not_renderer():
    path = ROOT / "command_palette/manifests/commands/media/smart_finish_advisor.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["id"] == "media.smart_finish_advisor"
    assert manifest["worker_entry"] == "sscp.workers.smart_finish.main"
    assert manifest["arguments"] == {"quality_policy": "best_quality"}
