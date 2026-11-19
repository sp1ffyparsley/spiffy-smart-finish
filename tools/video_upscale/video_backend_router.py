"""Stable, allowlisted Smart Finish AI-backend dispatch boundary.

Backends return the incumbent job/result tuple and own only enhancement
inference. The Final Delivery / shared progress / GPU lock / media validation
contracts remain unchanged. A new model can reuse an existing adapter; a
genuinely temporal VSR family needs its own fully validated adapter, not a
pretend one-frame GAN graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from comfyui_model_catalog import FRAME_GAN_ADAPTER


BACKEND_CONTRACT = "spiffy_video_backend_v1"


@dataclass(frozen=True, slots=True)
class BackendSpec:
    id: str
    adapter_id: str
    inference_family: str
    temporal_context: bool
    source_audio_preserved: bool
    lossless_working_intermediate: bool
    supports_source_preservation: bool


BACKENDS = MappingProxyType({
    "comfyui-gan": BackendSpec(
        id="comfyui-gan", adapter_id=FRAME_GAN_ADAPTER,
        inference_family="local_pinned_frame_model",
        temporal_context=False, source_audio_preserved=True,
        lossless_working_intermediate=True, supports_source_preservation=True,
    ),
})


def get_backend(backend_id: str) -> BackendSpec:
    if type(backend_id) is not str or backend_id not in BACKENDS:
        raise ValueError("Requested AI backend has no installed and verified execution adapter.")
    return BACKENDS[backend_id]


def run_backend(backend_id: str, source: Path, **kwargs: Any) -> tuple[int, dict[str, Any]]:
    """Dispatch only to code that implements this versioned delivery contract."""
    spec = get_backend(backend_id)
    if spec.adapter_id == FRAME_GAN_ADAPTER:
        from comfyui_video_backend import run_comfyui_delivery

        code, result = run_comfyui_delivery(source, **kwargs)
        if not isinstance(result, dict) or result.get("schema") not in {
            "spiffy_comfyui_delivery_result_v1",
        }:
            raise RuntimeError("AI backend returned an unrecognized result schema; no success is authorized.")
        if code == 0 and (result.get("state") != "passed" or not result.get("final_outputs")):
            raise RuntimeError("AI backend claimed success without a verified published output.")
        return code, result
    raise RuntimeError("Requested AI backend adapter is not implemented and qualified.")
