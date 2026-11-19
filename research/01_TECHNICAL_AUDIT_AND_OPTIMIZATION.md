# Gemini Deep Research Prompt 1 — Audit Smart Finish for speed, fidelity, reliability, and architecture

You are conducting an independent, evidence-backed **engineering research audit of an existing Windows AI-video enhancement system**. This is a real code review and external technical investigation, not a product brainstorm or a request to recreate the application.

## Your primary source

**Public GitHub repository:** https://github.com/sp1ffyparsley/spiffy-smart-finish

Read the current default branch at the beginning of your investigation. Start with `README.md`, `docs/ARCHITECTURE.md`, `docs/RESEARCH_PROTOCOL.md`, and `SOURCE_SNAPSHOT_MANIFEST.json`, then inspect the actual implementation. Open individual files directly when GitHub search is not indexed. Do not base recommendations only on README claims or a cached summary.

The relevant code is under `tools/video_upscale/`, `Spiffy Workspace/workflow/final_delivery_pipeline.py`, `Spiffy Workspace/tools/run_spiffy_final_delivery.py`, `shared/`, `command_palette/`, and the selected test files. Particularly inspect `video_upscale.py`, `adaptive_enhancement.py`, `adaptive_model_selection.py`, `adaptive_subjects.py`, `comfyui_video_backend.py`, `comfyui_model_catalog.py`, `video_backend_router.py`, `motioncoach_face_bridge.py`, `selective_face_enhancement.py`, `quality_advisor.py`, `compression_optimizer.py` and the current tests. This mirror is a scoped snapshot of a larger, private, active workspace: **do not invent missing dependencies, configuration, implementations, test results or private files**. Flag missing integration evidence clearly.

## Goal and constraints

Find the **highest-value ways to make the existing workflow faster, more visually accurate, safer, more reliable, easier to maintain and more efficient with limited hardware**. Optimize the entire video from input to verified delivery, not only neural inference FPS. Prioritize photographic realism, natural skin and hair, stable faces, preserved motion blur, authentic source grain, readable typography, audio correctness and temporal consistency over artificially sharp or enlarged pixels.

Target hardware: **NVIDIA GeForce RTX 3080 with 10 GiB VRAM**, Windows desktop, approximately 64 GiB system RAM; assume the user may also be running Premiere Pro, After Effects, other local AI models, encoding jobs and multiple coding agents. Offline/local inference is strongly preferred. Do not casually recommend paid APIs, new cloud dependencies, a 16–24 GiB-only default or replacing the entire application. Consider genuine 4K, 6K and above when the image-quality benefit and hardware feasibility justify it, without treating conventional upscaling as native neural 6K reconstruction.

The existing system has one established Final Delivery owner, progress/job receipts, source checks, GPU lock, isolated ComfyUI runtime, allowlisted model registry, exact-frame reuse, bounded batching, optional MotionCoach-based face ROI, audio-sample/frame-clock verification, lossless intermediate and one final delivery encode. Keep those proven boundaries unless you can document a decisive reason to change them. Never recommend silently deleting or modifying original source footage or user project files.

## Required audit investigations

### 1. Reconstruct the actual execution graph

Trace real calls from Command Palette/CLI to source probing, model selection, Topaz or ComfyUI execution, frame decoding, PNG transfer, optional ROI, compositing, ProRes/PCM assembly, H.264 encoding, quality evaluation, final full decode, atomic publication and temporary cleanup. Identify all sources of duplicate work, redundant decoding, repeated intermediate encodes, hidden disk round-trips, repeated GPU model loads, repeated ffprobe calls, serialization bottlenecks and cross-process data copies. Map who owns retries, cancellation, GPU admission, interprocess locks, provenance, temporal source identity and output publishing. Provide exact file/function references.

### 2. Quantify the REAL speed bottlenecks

Separate cold startup, Python/Torch import, ComfyUI server readiness, model-weight load, GPU warmup, input decoding, PNG write/read, model inference, tiling, batch effects, CPU image composites, RAM/PCIe transfers, optical-flow or motion analysis, ProRes/PCM assembly, H.264 compression, VMAF pilots, FFprobe and full decoding. Analyze whether a trusted **direct Spandrel/PyTorch inference adapter** can bypass most of ComfyUI's startup cost while sharing the exact model/backend contract, without duplicating the scheduler or silently changing visual output. Investigate model hot reuse *only when it can be made safe under concurrent GPU workloads*, lazy loading, persistent CPU-only model metadata, bounded job pools, zero-copy/pipe alternatives, lossless FFmpeg pipes, memory-mapped images, optimized PNG/EXR transfer, frame batches and on-demand GPU model unloading.

Treat existing local observations as hints, not universal proof. For example, some eight-frame runs had 4–80 seconds of server startup; a same-server x4plus benchmark recorded 6.366 seconds for single-frame inference versus 3.998 seconds for three-frame batches with pixel-identical results. Explain what must be held constant before claiming an end-to-end speedup. Propose a reproducible ABBA/alternating benchmark plan with GPU load and driver measurements, cold versus warm isolation, power/VRAM telemetry and a clear statistical acceptance threshold.

### 3. Find actual reliability bugs and architectural debt

Audit exception and context-manager boundaries, FFmpeg subprocess termination, process-tree ownership, timeout/cancellation semantics, input symlink/path traversal protections, check-then-use races, file checksum verification, lock acquisition/release, collision-safe rename/cross-drive copy, concurrent agents, source changes mid-render, incomplete model downloads, malformed ComfyUI graph output, duplicate/missing frames, source PTS rounding, AAC priming/padding, audio-track offsets, signal-based termination, disk-full behavior, high-resolution pixel bounds, job resume and partial artifact recovery. Check for error masking, false success, overbroad catches, corrupted outputs, leaked subprocesses and race conditions. Distinguish proven bugs from hypotheses, and provide minimum negative-before test cases.

### 4. Improve temporal consistency and naturalness

Evaluate how a per-frame GAN can create flicker, unstable facial details, inconsistent film grain, warped textures, haloing, facial identity changes, hair and clothing shimmer, amplified compression artifacts, ghosting at scene cuts and artificial sharpened edges. Research inexpensive **temporal validation** based on existing optical-flow, scene boundaries, motion vectors, DOVER/VMAF, alignment/warp residuals, landmark confidence, source-matched identity and regions of interest. Separate *detection and gating* from *alteration*: identify when the safe action is retaining the original frame, reducing AI strength, choosing another model, preserving grain or waiting for a true temporal video model. Define realistic thresholds, real negatives and false positives.

### 5. Re-evaluate models against the 10 GiB GPU

Research and date-check actively maintained niche repositories, official checkpoints, forks and reproducible optimizations for Real-ESRGAN, BSRGAN, SwinIR, HAT, OmniSR, SUPIR, DPIR, BasicVSR++, RealBasicVSR, RVRT, VRT, RelayVSR, FlashVSR and SeedVR2 **only where relevant to the current path**. Distinguish pure image SR, temporal VSR, face restoration and diffusion. For each serious candidate: exact upstream URL, license, last meaningful update, source weights, checkpoint format, native scale, video timing support, Windows support, CUDA/Torch/ComfyUI versions, 10 GiB VRAM evidence, fp16/int8/quantization compatibility, tiling/block swapping and real runtime on 1080p, 4K and 6K. Never invent performance or claim a 16 GiB official benchmark proves 10 GiB compatibility.

### 6. Challenge compression decisions scientifically

Audit H.264 CRF14/16/17/18/20 slow, source-matched grain handling, adaptive quality by shot or scene, 8-bit versus 10-bit pipelines, source color metadata, conversion between RGB, 4:2:0, 4:2:2 and HDR paths, resize/chroma filters, audio remuxing and AAC re-encoding. Evaluate x264 tune grain, x265/HEVC, AV1/SVT-AV1, perceptual encoding, grain synthesis, 2-pass/quality-targeted size control and platform-specific YouTube Shorts/TikTok exports. Identify when hardware encoders may actually lose quality at a given size. The observed NVIDIA driver 560.94 / NVENC API mismatch means **codec availability in FFmpeg -encoders is not proof of a working HEVC NVENC path**. Propose explicit runtime admission and verification. Review `compression_optimizer.py`: full-resolution sampling validity, VMAF/SSIM/PSNR aggregation, failure fallback, false positive cases, the 45-second cap, duration edge cases, test coverage and whether extra pilots cost more time than they save.

### 7. Make 6K/8K choices truthful and useful

Inspect geometry contracts and model-native tensor area against final output raster. Determine whether native higher-resolution model inference, tiled inference, iterative 2× passes, region-specific 6K reconstruction or conventional resizing yields materially better **viewed** detail. Require actual comparative experiments and honest render-time/VRAM estimates, with an easy quality/speed ladder. Show when a 6K intermediate merely multiplies cost and adds no perceptual benefit. Propose automatic advice only after model/source-specific runtime and quality evidence; otherwise surface uncertainty.

### 8. Reuse existing MotionCoach and semantic-mask work

Inspect the exposed face ROI contract and the current `adaptive_subjects.py` bridge. Evaluate full-SHA/source binding, timestamp maps, landmark confidence, crop alignment, mask transitions, small-region color QA, identity preservation and failure to track a person. Check which existing data could be shared safely across reframe, edit tracking, selective detail, masking, AI enhancement and video QA, without creating another detector, tracker, cache, database or different ownership policy. Do not assume the private MotionCoach code is available beyond exposed interface evidence.

### 9. Simplify without regression

Identify extra code, unnecessary abstractions, duplicate paths, stale flags, repeat model discovery, wasteful logging, avoidable JSON/PNG operations, unbounded loops or overprotective checks whose measured cost is high and whose safety benefit is minimal. Distinguish useful fault tolerance from bloat. Suggest specific safe simplifications with exact diffs/contracts and regression tests. Do not advise disabling validation just to claim a speedup.

### 10. Testing, packaging, portability and security

Propose unit, property, integration, real short-video, long-video, corrupted-source, 4K/6K, GPU contention, codec and cancellation tests. Discuss standalone package boundaries versus the current private monorepo, dependency-locking, reproducible model manifests, no executable model downloads, GitHub Actions limited to publicly mirrored dependencies, and the Windows 10 → later Windows path. Flag licensing and third-party model redistribution risks before suggesting any public weight/source mirroring.

## External research expectations

Use current information as of your research date. Search GitHub, GitHub issues/discussions, official docs, arXiv/papers, model cards, release tags, benchmark repositories, engineering posts and credible user repros. Prefer official claims verified by local or independent benchmarks. Critically assess community claims (including Reddit) for hardware differences, unsupported measurements, test-footage quality, model revisions, GPU contention and cherry-picking. Look for **narrow reusable libraries and optimized kernels**, not only full upscaler apps.

## Required deliverables and format

1. **Executive findings:** ten or fewer decisive findings ranked by expected real-world value, with current architecture bottlenecks and what not to change.
2. **Evidence-based architecture map:** exact source paths, current responsibility, missing private dependencies, traced bottlenecks and call paths.
3. **Ranked engineering opportunities table:** priority P0–P3, implementation area/file, current behavior, change, improvement mechanism, source links, estimated speed/quality gain *with confidence and assumptions*, VRAM/RAM/disk requirements, implementation effort, safety risk and hard acceptance tests.
4. **Deep model/codec comparison:** only viable candidates, grounded official links, required weights/runtime/licenses, realistic 10 GiB feasibility, motion and identity risks, throughput information, quality tradeoffs.
5. **Benchmark and QA protocol:** exact fixtures/classes, same-source controlled comparisons, performance breakdowns, no external disclosure of private media, thresholds for promotion/rejection and rollback triggers.
6. **Concrete implementation sequence:** dependency-aware small-to-large stages, names of existing modules to change, interfaces to reuse, necessary new adapter contracts, what code can be deleted or consolidated, risks of concurrent agents and a complete test matrix. This is an actionable engineering blueprint **not an AI-coding prompt**.
7. **Final decisions:** five improvements to implement first, five ideas to reject or postpone, and exact unresolved questions requiring real local evidence.

**Non-negotiables:** No filler, generic “try a newer model” advice, speculative FPS claims, instructions to make the entire private monorepo public, imaginary existing functionality, invented open-source licenses or private user footage. Separate **verified in the public source**, **verified externally**, **plausible but untested**, and **unsupported**. Cite exact GitHub file links and upstream research for every major assertion. I want conclusions that a Windows engineer can implement and falsify.
