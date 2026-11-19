# Smart Finish — current architecture and review map

This document describes the published implementation boundaries. It does not claim that the research mirror is a complete production package.

## System flow

```text
User-selected finished video / Command Palette / CLI
  -> input, source-content identity, metadata and decoded PTS preflight
  -> source-specific resolution and model policy
      -> local Topaz owner (adaptive model selection, motion/FPS policy)
      -> ComfyUI allowlisted backend router + pinned image-model registry
          -> adaptive bounded batching / exact-byte frame reuse
          -> optional source-preservation blend
          -> optional MotionCoach-published face ROI (explicit only)
  -> enhancement frames / source-clock-preserving, lossless working media
  -> audio sample-extent and frame-timing checks
  -> optional bounded native-resolution compression pilots
  -> one final H.264 delivery encode
  -> full decoding + picture/audio/timing/identity verification
  -> collision-safe atomic publication and cleanup
```

## Core components

| Component | Responsibility | Relevant paths |
| --- | --- | --- |
| Command entry / user controls | Separate Topaz, 4K/6K ComfyUI, fast model, advisor and measured compression routes | `command_palette/workers/python/src/sscp/workers/smart_finish/main.py`, `command_palette/manifests/commands/media/smart_finish*.json` |
| Final Delivery orchestration | Single job/result ownership, source preflight, format policy and publication requirements | `Spiffy Workspace/workflow/final_delivery_pipeline.py`, `Spiffy Workspace/tools/run_spiffy_final_delivery.py` |
| Topaz AI worker | Existing enhancement, adaptive working geometry, model choice, temporal interpolation and H.264 encoding | `tools/video_upscale/video_upscale.py`, `adaptive_enhancement.py`, `adaptive_model_selection.py` |
| Pinned image-model catalog | Model IDs, hashes, native 2×/4× limits and explicit policies | `tools/video_upscale/comfyui_model_catalog.py` |
| Alternate backend contract | Allowlisted adapters, fail-closed result schema | `tools/video_upscale/video_backend_router.py` |
| ComfyUI frame-GAN execution | Isolated offline server, image batches, conservative resource gates, per-frame QA and delivery handoff | `tools/video_upscale/comfyui_video_backend.py` |
| Selective face enhancement | Reads canonical MotionCoach pose, checks full source hash, localizes a high-confidence crop, composites into original background | `tools/video_upscale/motioncoach_face_bridge.py`, `selective_face_enhancement.py` |
| Compression calibration | Bounded CRF 18/20 pilots with full-resolution VMAF, SSIM and PSNR | `tools/video_upscale/compression_optimizer.py` |
| Read-only advice | 2K/3K/4K/6K/8K canvas feasibility, separated from model readiness and actual render-time evidence | `tools/video_upscale/quality_advisor.py` |
| Job and safety primitives | FFmpeg, media probing, progress, job IDs and GPU locking | `shared/*.py` |

## Known constraints worth challenging

1. **Cold-start performance:** Full isolated ComfyUI startup can dominate very short clips. The private server avoids unknown custom nodes and competing resident GPU services, but startup can vary with other workloads. Investigate a direct pinned Spandrel/PyTorch inference adapter, narrower startup, and safe short-lived worker reuse without creating a second scheduler.
2. **Frame-model temporal artifacts:** Real-ESRGAN is not trained for video. The current frame-to-frame heuristic detects some flashes but does not guarantee motion, face or grain consistency.
3. **10 GiB VRAM:** Source geometry and native 2×/4× tensor area, not just final raster dimensions, determine feasibility. Validate tiled inference, quantized model memory, CPU/RAM offloading and clean GPU contention behavior.
4. **Real model quality:** Compare authentic temporal VSR, face-detail restoration, color/grain preservation and optical-flow-aided consistency on identical footage. Do not infer superiority from model popularity, output resolution or mismatched hardware benchmarks.
5. **MotionCoach reuse:** The face bridge uses separately published pose files, not a private identity-recognition model. The user explicitly wanted cross-workspace reuse, but false face identity/poor tracking must fail closed.
6. **Compression:** The existing H.264 baseline and measured-pilot path are useful but objective quality metrics are not proof of subjective grain/skin texture quality. Identify better source-aware bitrate, grain/film tuning, color and codec tests without another lossy round trip.
7. **Unresolved native Topaz crash:** A previously observed 4K OpenCV access violation needs root-cause proof before changing global defaults.

## What the mirror intentionally cannot prove

- Packaged user interface, Adobe/CEP integration and complete MotionCoach consumer app are outside the scope of this mirror.
- Several original internal workflow modules and external installations are not vendored. Code that imports them documents a real interface but is not a standalone runnable build.
- No authentic retained user footage or benchmark output is distributed publicly. Qualifying tests with user-owned footage must stay local, private and source-bound.
- Research proposals must not introduce untrusted node auto-downloads, unrestricted filesystem access, silent source edits, automatic model installation, new background agents or multiple render/GPU queue owners.

For the requested external audits, follow [research protocol](RESEARCH_PROTOCOL.md) and the [two specific briefs](../research).
