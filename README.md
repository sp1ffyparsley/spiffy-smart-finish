# Spiffy Smart Finish

**Public, scoped research mirror of the Smart Finish video-enhancement and delivery workflow.**

This repository makes the actual Smart Finish implementation available for technical review and independent research. It is **not** a full copy of the owner's private `SS-Scripts` monorepo and is **not** presented as a self-contained application, redistributable Topaz package, or ready-to-run ComfyUI installation. The owner maintains the operational Windows workspace separately.

## Where to start

- [Research prompt 1 — performance, fidelity, and reliability audit](research/01_TECHNICAL_AUDIT_AND_OPTIMIZATION.md)
- [Research prompt 2 — new AI functions, model discovery, and product architecture](research/02_NEW_CAPABILITIES_AND_MODEL_ROADMAP.md)
- [Current architecture and code navigation](docs/ARCHITECTURE.md)
- [Research rules and evidence bar](docs/RESEARCH_PROTOCOL.md)
- [Source snapshot checksum manifest](SOURCE_SNAPSHOT_MANIFEST.json)

## What is mirrored

| Code area | Where to look |
| --- | --- |
| Main enhancement / Topaz worker | [tools/video_upscale/video_upscale.py](tools/video_upscale/video_upscale.py) |
| Adaptive natural enhancement, model selection, and subject protection | [tools/video_upscale](tools/video_upscale) |
| ComfyUI inference, safe local model catalog, backend adapter | [tools/video_upscale/comfyui_video_backend.py](tools/video_upscale/comfyui_video_backend.py), [comfyui_model_catalog.py](tools/video_upscale/comfyui_model_catalog.py), [video_backend_router.py](tools/video_upscale/video_backend_router.py) |
| Source-bound MotionCoach face-region bridge | [motioncoach_face_bridge.py](tools/video_upscale/motioncoach_face_bridge.py), [selective_face_enhancement.py](tools/video_upscale/selective_face_enhancement.py) |
| Compression calibration | [compression_optimizer.py](tools/video_upscale/compression_optimizer.py) |
| Read-only quality / resolution advisor | [quality_advisor.py](tools/video_upscale/quality_advisor.py) |
| Delivery orchestration and CLI | [final_delivery_pipeline.py](Spiffy%20Workspace/workflow/final_delivery_pipeline.py), [run_spiffy_final_delivery.py](Spiffy%20Workspace/tools/run_spiffy_final_delivery.py) |
| Shared FFmpeg, source probes, GPU/job protections | [shared](shared) |
| Command Palette worker and command definitions | [smart_finish/main.py](command_palette/workers/python/src/sscp/workers/smart_finish/main.py), [manifests](command_palette/manifests/commands/media) |
| Regression tests | [Spiffy Workspace/tests](Spiffy%20Workspace/tests), [Palette tests](command_palette/tests) |

## Operational constraints

- Windows desktop; primary GPU observed as NVIDIA RTX 3080 **10 GiB VRAM**. CPU / RAM load and available GPU memory may vary because other local jobs run concurrently.
- Topaz is a separately installed commercial dependency. No Topaz binary, model weights, activation mechanisms, or proprietary software are in this mirror.
- ComfyUI is an existing separately installed local runtime. Real-ESRGAN x2plus, x4plus, and general x4v3 are locally tested, pinned/checksummed weights in the owner's private model store. **No model weights are redistributed here**.
- Real-ESRGAN is an **image** upscaler applied to video frames, not a temporal video-restoration system. Dedicated temporal VSR models are future candidates, **not** certified in this repository.
- Default completion requires exact frame/audio timing, source checks, GPU/disk admission, a lossless intermediate, one final encode and final-output verification.
- Very high-resolution rendering (6K / 8K), face identity fidelity, grain and temporal consistency require actual model/source-specific quality tests; larger pixel counts are not automatically better.
- ComfyUI face-region enhancement optionally consumes published source-bound MotionCoach pose evidence from the separately maintained system. Spatial pose tracking is **not** proof of face identity.
- A prior native Topaz 4K OpenCV crash is unresolved. An observed FFmpeg/NVIDIA NVENC API mismatch also prevents assuming GPU HEVC encoding will work on the current driver.

## Scope and honesty

This is a source-code research snapshot with selected support modules and tests from a larger codebase. Some imports, plugins, and integration owners remain in the private parent workspace (for example, the complete MotionCoach application, Premiere/CEP platform, and larger workflow runtime). **Do not claim this mirror can be installed, built, or integration-tested independently unless you have actually resolved those dependencies.**

The mirror does not include personal footage, audio libraries, customer data, machine-specific secrets, virtual environments, caches, model checkpoints, binaries, Adobe/Topaz software or the private monorepo's Git history. It also does not automatically track subsequent private-workspace changes; [the manifest](SOURCE_SNAPSHOT_MANIFEST.json) identifies this revision.

## Research and reuse

Use the two linked research prompts for independent, evidence-backed analysis. Prefer concrete findings linked to exact source files, realistic hardware measurements, new narrow open-source research and an implementation order that preserves the existing architecture.

**License:** Public visibility does **not** grant permission to copy, redistribute, modify or commercially use this code. No open-source license has been granted for this snapshot. Third-party projects mentioned here retain their own licenses. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md).

**Security:** Do not send secrets, model weights, private source media or customer footage in an issue. See [SECURITY.md](SECURITY.md).
