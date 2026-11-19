"""Pinned local AI model identities and an explicit, versioned backend contract.

Only the frame-GAN adapter is executable today. A temporal VSR implementation
must implement the same source-clock / full-coverage / audio / verification
contract but may NOT be routed through this image-by-image graph.  Model files
live in the existing external ComfyUI install, never inside SS Scripts source.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Literal


MODEL_CONTRACT = "spiffy_model_registry_v1"
FRAME_GAN_ADAPTER = "comfyui_frame_gan_v1"
SUPPORTED_ADAPTERS = frozenset({FRAME_GAN_ADAPTER})
DEFAULT_MODEL_POLICY = "auto-quality"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    filename: str
    sha256: str
    native_scale: int
    adapter: str
    media_style: Literal["photographic", "illustration"]
    processing_tier: Literal["high_detail", "fast"]
    source_url: str
    temporal_context: bool = False

    def public_info(self) -> dict[str, object]:
        return {**asdict(self), "schema": MODEL_CONTRACT,
                "source_license_check_required_for_redistribution": True}


_MODELS: dict[str, ModelSpec] = {
    "realesrgan_x4plus": ModelSpec(
        id="realesrgan_x4plus", filename="RealESRGAN_x4plus.pth",
        sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        native_scale=4, adapter=FRAME_GAN_ADAPTER,
        media_style="photographic", processing_tier="high_detail",
        source_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    "realesrgan_x2plus": ModelSpec(
        id="realesrgan_x2plus", filename="RealESRGAN_x2plus.pth",
        sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        native_scale=2, adapter=FRAME_GAN_ADAPTER,
        media_style="photographic", processing_tier="high_detail",
        source_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    ),
    "realesrgan_general_x4v3": ModelSpec(
        id="realesrgan_general_x4v3", filename="realesr-general-x4v3.pth",
        sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        native_scale=4, adapter=FRAME_GAN_ADAPTER,
        media_style="photographic", processing_tier="fast",
        source_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    ),
}
MODELS = MappingProxyType(_MODELS)
MODEL_POLICIES = frozenset({"auto-quality", "auto-fast", *MODELS})


def get_model(model_id: str) -> ModelSpec:
    if type(model_id) is not str or model_id not in MODELS:
        raise ValueError("Unrecognized/unqualified local model identity.")
    model = MODELS[model_id]
    if model.adapter not in SUPPORTED_ADAPTERS or model.temporal_context:
        raise RuntimeError("No qualified adapter exists for this model family.")
    return model


def get_model_by_filename(filename: str) -> ModelSpec:
    if type(filename) is not str:
        raise ValueError("Model filename must be a pinned catalog name.")
    for model in MODELS.values():
        if model.filename == filename:
            return get_model(model.id)
    raise ValueError("ComfyUI model filename is not in the pinned registry.")


def select_model(model_id: str, *, source_long_edge: int,
                 target_long_edge: int) -> ModelSpec:
    """Select using only source geometry and policy, never a fabricated score."""
    if type(source_long_edge) is not int or type(target_long_edge) is not int:
        raise ValueError("Model selection requires exact integer geometry.")
    if source_long_edge < 64 or target_long_edge < source_long_edge:
        raise ValueError("Upscaling cannot downscale or operate on invalid source geometry.")
    if type(model_id) is not str or model_id not in MODEL_POLICIES:
        raise ValueError("Unknown model or model policy.")
    selected = ("realesrgan_general_x4v3" if model_id == "auto-fast" else
                "realesrgan_x2plus" if model_id == "auto-quality"
                and target_long_edge <= source_long_edge * 2 else
                "realesrgan_x4plus" if model_id == "auto-quality" else model_id)
    result = get_model(selected)
    if target_long_edge > source_long_edge * result.native_scale:
        raise ValueError(
            f"{result.id} is a native {result.native_scale}x model; cannot claim "
            f"{target_long_edge}-pixel AI output from {source_long_edge}-pixel source.")
    return result


def verify_model_file(comfy_root: Path, model: ModelSpec) -> Path:
    """Fail closed on missing/linked/mutated weights. No downloads in a render."""
    if type(comfy_root) is not Path:
        comfy_root = Path(comfy_root)
    if get_model(model.id) != model:
        raise ValueError("Untrusted model specification or substituted checkpoint.")
    path = comfy_root / "models" / "upscale_models" / model.filename
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Pinned model not installed: {path}")
    before = path.stat()
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest().lower()
    after = path.stat()
    if digest != model.sha256 or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{model.id} checksum/identity changed; rendering is blocked.")
    return path


def inspect_models(comfy_root: Path) -> dict[str, dict[str, object]]:
    """Discovery only. Present checkpoint evidence as readiness, not visual QA."""
    ready: dict[str, dict[str, object]] = {}
    for spec in MODELS.values():
        try:
            verify_model_file(comfy_root, spec)
            status = "pinned_checkpoint_available_not_source_qualified"
        except (OSError, ValueError, RuntimeError):
            status = "not_verified"
        ready[spec.id] = {**spec.public_info(), "status": status,
                          "installed_and_verified": status.startswith("pinned")}
    return ready
