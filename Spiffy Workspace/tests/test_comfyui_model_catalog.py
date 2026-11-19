"""Versioned, pinned ComfyUI models and bounded frame-batch adapter contracts."""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace

import pytest

ROOT = Path(__file__).resolve().parents[2]
UPSCALE = ROOT / "tools" / "video_upscale"
if str(UPSCALE) not in sys.path:
    sys.path.insert(0, str(UPSCALE))

from comfyui_model_catalog import (  # noqa: E402
    DEFAULT_MODEL_POLICY, MODELS, MODEL_CONTRACT,
    FRAME_GAN_ADAPTER, get_model, get_model_by_filename, inspect_models,
    select_model, verify_model_file,
)
from comfyui_video_backend import (  # noqa: E402
    _batch_size, _exact_frame_key, _upscale_batch_graph, _upscale_graph,
    _process_batch, _validate_source,
)
from video_backend_router import (  # noqa: E402
    BACKEND_CONTRACT, BACKENDS, get_backend, run_backend,
)


def test_catalog_version_and_three_distinct_pinned_real_models():
    assert MODEL_CONTRACT == "spiffy_model_registry_v1"
    assert set(MODELS) == {
        "realesrgan_x4plus", "realesrgan_x2plus", "realesrgan_general_x4v3"}
    assert len({model.filename for model in MODELS.values()}) == 3
    assert len({model.sha256 for model in MODELS.values()}) == 3
    assert all(len(model.sha256) == 64 and model.adapter == FRAME_GAN_ADAPTER
               and model.temporal_context is False and model.source_url.startswith("https://github.com/xinntao/")
               for model in MODELS.values())
    assert all(get_model_by_filename(model.filename) == model for model in MODELS.values())
    with pytest.raises(TypeError):
        MODELS["untrusted"] = get_model("realesrgan_x4plus")


def test_backend_owner_contract_is_explicit_and_not_a_fictitious_temporal_model():
    assert BACKEND_CONTRACT == "spiffy_video_backend_v1"
    assert set(BACKENDS) == {"comfyui-gan"}
    chosen = get_backend("comfyui-gan")
    assert chosen.adapter_id == FRAME_GAN_ADAPTER
    assert chosen.temporal_context is False
    assert chosen.source_audio_preserved and chosen.lossless_working_intermediate
    with pytest.raises(ValueError, match="no installed"):
        get_backend("comfyui-relayvsr")


def test_backend_dispatch_requires_verifiable_result_contract(monkeypatch, tmp_path):
    import comfyui_video_backend as runner

    def success(source, **kwargs):
        assert source == tmp_path / "source.mp4" and kwargs["model_id"] == "auto-fast"
        return 0, {"schema": "spiffy_comfyui_delivery_result_v1",
                   "state": "passed", "final_outputs": [str(tmp_path / "enhanced.mp4")]}

    monkeypatch.setattr(runner, "run_comfyui_delivery", success)
    code, record = run_backend("comfyui-gan", tmp_path / "source.mp4", model_id="auto-fast")
    assert code == 0 and record["state"] == "passed"

    def invalid(source, **kwargs):
        return 0, {"schema": "unknown_result", "state": "passed", "final_outputs": []}

    monkeypatch.setattr(runner, "run_comfyui_delivery", invalid)
    with pytest.raises(RuntimeError, match="unrecognized result schema"):
        run_backend("comfyui-gan", tmp_path / "source.mp4")


def test_pinned_three_model_files_are_installed_in_the_existing_comfyui_root():
    from comfyui_video_backend import DEFAULT_COMFY_ROOT
    # Local-only optional evidence. On another developer machine the pure
    # tests above still exercise the catalog without downloading weights.
    if not (DEFAULT_COMFY_ROOT / "models" / "upscale_models" / "RealESRGAN_x4plus.pth").is_file():
        pytest.skip("Local ComfyUI model install not present on this machine.")
    ready = inspect_models(DEFAULT_COMFY_ROOT)
    assert {model_id for model_id, row in ready.items()
            if row["installed_and_verified"]} == set(MODELS)


def test_quality_model_switch_preserves_native_training_scale():
    assert DEFAULT_MODEL_POLICY == "auto-quality"
    assert select_model("auto-quality", source_long_edge=1920, target_long_edge=3840).id == "realesrgan_x2plus"
    assert select_model("auto-quality", source_long_edge=1920, target_long_edge=5760).id == "realesrgan_x4plus"
    assert select_model("auto-fast", source_long_edge=1920, target_long_edge=3840).id == "realesrgan_general_x4v3"
    assert select_model("realesrgan_x4plus", source_long_edge=1920, target_long_edge=3840).id == "realesrgan_x4plus"
    with pytest.raises(ValueError, match="native 2x"):
        select_model("realesrgan_x2plus", source_long_edge=1920, target_long_edge=5760)
    with pytest.raises(ValueError):
        select_model("not-a-local-model", source_long_edge=1920, target_long_edge=3840)
    with pytest.raises(ValueError):
        get_model_by_filename("../untrusted.pth")


def test_untrusted_model_specs_and_missing_weights_fail_closed(tmp_path):
    authentic = get_model("realesrgan_x4plus")
    fake = replace(authentic, sha256="0" * 64)
    with pytest.raises(ValueError, match="Untrusted"):
        verify_model_file(tmp_path, fake)
    with pytest.raises(FileNotFoundError, match="not installed"):
        verify_model_file(tmp_path, authentic)
    found = inspect_models(tmp_path)
    assert len(found) == 3
    assert not any(row["installed_and_verified"] for row in found.values())


def test_x2_native_tensor_admits_geometry_which_x4_tensor_rejects():
    info = {"width": 3840, "height": 2160, "fps_ratio": "24/1", "fps": 24,
            "duration": 2.0, "sar": "1:1", "rotation": 0,
            "video_pts_uniform": True, "frame_rate_evidence": "presentation-timestamps",
            "video_pts_count": 48, "video_pts_start": "0",
            "video_pts_average_interval": "1/24", "video_pts_end_exclusive": "2",
            "video_pts_time_base": "1/24000"}
    with pytest.raises(ValueError, match="native tensor"):
        _validate_source(info, 5760, get_model("realesrgan_x4plus"))
    assert _validate_source(info, 5760, get_model("realesrgan_x2plus")) == (5760, 3240, "24/1")


def test_batch_graph_preserves_input_order_and_one_single_model_node():
    graph = _upscale_batch_graph(["frame_000000.png", "frame_000001.png", "frame_000002.png"],
                                output_width=512, output_height=288,
                                output_prefix="Spiffy_2026_batch_000000",
                                model_name=get_model("realesrgan_general_x4v3").filename)
    assert graph["2"]["inputs"]["model_name"] == "realesr-general-x4v3.pth"
    assert [graph[node]["class_type"] for node in ("11", "12", "21", "22")] == [
        "LoadImage", "LoadImage", "ImageBatch", "ImageBatch"]
    assert graph["21"]["inputs"] == {"image1": ["1", 0], "image2": ["11", 0]}
    assert graph["22"]["inputs"] == {"image1": ["21", 0], "image2": ["12", 0]}
    assert graph["3"]["inputs"]["image"] == ["22", 0]
    assert sum(node["class_type"] == "UpscaleModelLoader" for node in graph.values()) == 1


def test_natural_source_preservation_is_optional_bounded_and_uses_core_nodes():
    args = {"output_width": 512, "output_height": 288,
            "output_prefix": "Spiffy_2026_natural_0000", "source_preservation": 0.10}
    graph = _upscale_graph("frame_000000.png", **args)
    assert graph["6"]["class_type"] == "ImageScale"
    assert graph["7"]["class_type"] == "ImageBlend"
    assert graph["7"]["inputs"]["blend_factor"] == 0.10
    assert graph["7"]["inputs"]["image1"] == ["4", 0]
    assert graph["7"]["inputs"]["image2"] == ["6", 0]
    assert graph["5"]["inputs"]["images"] == ["7", 0]
    batch = _upscale_batch_graph(["frame_000000.png", "frame_000001.png"], **args)
    assert batch["6"]["inputs"]["image"] == ["21", 0]
    default = _upscale_graph("frame_000000.png", **{k: v for k, v in args.items()
                                                   if k != "source_preservation"})
    assert "6" not in default and "7" not in default


@pytest.mark.parametrize("bad", [-0.1, 0.26, True, float("nan"), float("inf")])
def test_natural_source_preservation_rejects_out_of_range_and_nonfinite_values(bad):
    with pytest.raises(ValueError, match="Source-preservation blend"):
        _upscale_graph("frame_000000.png", output_width=512, output_height=288,
                       output_prefix="Spiffy_2026_natural_0000", source_preservation=bad)


def test_natural_source_preservation_is_memory_gated_over_4k():
    with pytest.raises(ValueError, match="beyond 4K"):
        _upscale_graph("frame_000000.png", output_width=5760, output_height=3240,
                       output_prefix="Spiffy_2026_natural_0000", source_preservation=0.08)


@pytest.mark.parametrize("invalid", [[], ["frame_000000.png"] * 2,
                                     ["frame_000000.png"] * 5,
                                     ["frame_000000.png", "../outside.png"]])
def test_batch_graph_rejects_missing_duplicate_oversized_or_untrusted_input(invalid):
    with pytest.raises(ValueError):
        _upscale_batch_graph(invalid, output_width=512, output_height=288,
                            output_prefix="Spiffy_2026_test_000000")


def test_batch_sizes_prefer_conservative_gpu_resource_bounds():
    assert _batch_size(128, 72, 512, 288, available_gpu_mib=3800, native_scale=4) == 3
    assert _batch_size(128, 72, 512, 288, available_gpu_mib=2900, native_scale=4) == 1
    assert _batch_size(1280, 720, 3840, 2160, available_gpu_mib=7200, native_scale=2) == 2
    assert _batch_size(1920, 1080, 5760, 3240, available_gpu_mib=9000, native_scale=4) == 1


def test_exact_frame_key_does_not_confuse_similar_pixels(tmp_path):
    from PIL import Image

    source = tmp_path / "first.png"
    copy = tmp_path / "second.png"
    changed = tmp_path / "changed.png"
    Image.new("RGB", (128, 72), (120, 90, 30)).save(source)
    copy.write_bytes(source.read_bytes())
    Image.new("RGB", (128, 72), (121, 90, 30)).save(changed)
    assert _exact_frame_key(source) == _exact_frame_key(copy)
    assert _exact_frame_key(source) != _exact_frame_key(changed)
    with pytest.raises(RuntimeError):
        _exact_frame_key(tmp_path / "missing.png")


def test_batch_result_checks_exact_coverage_and_order(tmp_path, monkeypatch):
    from PIL import Image
    import comfyui_video_backend as backend

    root = tmp_path / "server_output"
    root.mkdir()
    frames = [tmp_path / f"frame_{i:06d}.png" for i in range(3)]
    for i, item in enumerate(frames):
        Image.new("RGB", (128, 72), (i * 10, 60, 80)).save(item)
    saved = [f"Spiffy_2026_batch_00000{i}.png" for i in range(3)]
    for i, name in enumerate(saved):
        Image.new("RGB", (512, 288), (i * 10, 60, 80)).save(root / name)

    def fake_api(_base, suffix, **kwargs):
        if suffix == "/prompt":
            return {"prompt_id": "test-prompt"}
        if suffix == "/history/test-prompt":
            return {"test-prompt": {"status": {"completed": True},
                                    "outputs": {"5": {"images": [
                                        {"filename": name, "subfolder": "", "type": "output"}
                                        for name in saved]}}}}
        raise AssertionError(f"Unexpected API endpoint: {suffix}")

    monkeypatch.setattr(backend, "_json_http", fake_api)
    files = _process_batch("http://127.0.0.1:1", frames, output_root=root,
                           output_width=512, output_height=288,
                           prefix="Spiffy_2026_test_000000", timeout=2, cancel=lambda: False)
    assert files == [root / name for name in saved]
    saved.pop()
    with pytest.raises(RuntimeError, match="coverage"):
        _process_batch("http://127.0.0.1:1", frames, output_root=root,
                       output_width=512, output_height=288,
                       prefix="Spiffy_2026_test_000000", timeout=2, cancel=lambda: False)
