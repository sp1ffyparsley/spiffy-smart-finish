# Gemini Deep Research Prompt 2 — New AI video capabilities, radical quality gains, and a future-proof Smart Finish roadmap

Act as an independent **AI video research lead and product/architecture investigator**. Your job is to challenge the limits of a working local video upscaling system and discover what it could become over the next several years. I want new **useful functions, specialized models, novel combinations of existing capabilities and genuine improvements to the rendered video**, not a second generic upscaler app or a giant library of dormant code.

## Read the actual implementation first

**Public project:** https://github.com/sp1ffyparsley/spiffy-smart-finish

Start at `README.md`, `docs/ARCHITECTURE.md`, `docs/RESEARCH_PROTOCOL.md`, and the implementation under `tools/video_upscale/`. Inspect the actual backend router, pinned model catalog, ComfyUI worker, selective MotionCoach face bridge, adaptive subject/geometry logic, quality advisor, compression module, shared job/GPU locks, Final Delivery/CLI and Command Palette integration. Read the published tests to understand what has genuinely been verified. This GitHub repository is a **scoped public research mirror** of a larger private active Windows workspace. Do not assume that missing private files are included; do not request account credentials or privately owned media.

## The creative vision

The user creates cinematic short-form edits, documentary/TV scene edits, VFX and motion graphics. The target is **the most realistic, temporally stable, natural and detailed finished video that can be produced sensibly on the available hardware**. The system must preserve faces, skin texture, hair, hands, expressions, source motion, eyes, lip movements, camera moves, film grain, cinematic color and text/graphics. It must offer genuine 4K, 6K and optionally higher quality when useful, while avoiding meaningless enlargement, waxy faces, invented text and scene regeneration errors.

The main machine has **RTX 3080 / 10 GiB VRAM** and approximately 64 GiB RAM. Other local editing, AI and automation workloads run concurrently; avoiding GPU contention and disk bloat is part of the product requirement. Favor **offline, open-source, no recurring API fees**, with licensed model checkpoints, bounded resource use and a swappable backend contract. Keep original footage read-only and make new creative/generative behavior an **explicit, reversible derivative**. Existing Topaz, FFmpeg, ComfyUI and Final Delivery capabilities should be reused rather than rebuilt.

Think beyond the current model list. Investigate underexplored narrow tools, inference kernels, quantization techniques, tracking/segmentation packages, research prototypes, active forks, fast video generative pipelines and temporal restoration work. Propose the **strongest compatible ideas**, including ambitious ones, but clearly separate what works on 10 GiB now from longer-term possibilities that require different hardware or a new model breakthrough.

## Research track A — True temporal video super-resolution

Find and compare credible open implementations of **temporal VSR**, not just individually upscaled frames. Research the current state of FlashVSR, RelayVSR, SeedVR2, BasicVSR++, RealBasicVSR, VRT, RVRT, EDVR, IconVSR and relevant 2025–2026 entrants or specialized forks. Determine whether each can meaningfully improve *moving* footage over Real-ESRGAN and Topaz, whether it supports legitimate 2×/4× and higher output, where its temporal window introduces artifacts, what its checkpoint/license/runtime actually requires and whether **10 GiB VRAM** can handle it using quantization, frame windows, tiling, CPU/RAM block swapping, GGUF/int8/fp8/fp16 or shared model state. Report real processing rates and memory footprints **only from comparable evidence**, never mismatched 4090/5070 Ti benchmarks treated as RTX3080 proof.

Look for acceleration opportunities that can be developed as a **separate adapter** inside `video_backend_router.py`: streaming frames, caches, flash attention, quantized weights, chunk overlap, deterministic frame-clock reconstruction and a consistent output contract. Explain the safe fallback to the current image-GAN path when no qualified VSR is available.

## Research track B — Real face, eyes, hair and skin detail without identity drift

Investigate narrow-source face restoration and enhancement: CodeFormer, GFPGAN, RestoreFormer++, GPEN, FaceXLib, RetinaFace, InsightFace-based *detection only*, face-preserving super-resolution, expression/landmark consistency, hair and skin fidelity, and any better recent open alternatives. Cover why some face restoration models change eye shape, teeth, expression, identity, facial age, skin grain or lip timing.

Suggest a **selective face/subject detail engine** using existing published MotionCoach pose, SAM2/SAM3 segmentation, optical flow, current `adaptive_subjects.py` and per-frame confidence. It should spend GPU time on the face, eyes, mouth, hair, hands, important objects or a damaged region only when quality measurements justify it, while retaining source background, motion blur and grain. Handle multiple people, occlusion, turnarounds, tiny faces, profile views, uncertain landmarks, fast cuts, mask flicker, tracking loss and misidentified targets. Distinguish low-risk detail enhancement from models that hallucinate facial identity. Propose per-region quality checks (including user-visible identity-consistency guard results) rather than global VMAF alone.

For every face model provide link, license, weights availability, Windows/PyTorch/ComfyUI support, realistic GPU/CPU memory cost, inference speed, input/crop size, aliasing/soft-matte issues, color/motion artifacts and what actual test would permit production activation. Do not activate automatic face-swapping or identity-altering pipelines by default.

## Research track C — Reference-guided video-to-video AI regeneration

Explore the bigger possibility: **produce a newly synthesized, visually higher-quality video using the actual original video as a strict reference**. Treat it as a separate optional creative mode, not as normal upscaling. Research relevant current technologies: video-to-video diffusion/transformers, image/video-conditioned restoration, ControlNet-style pose/depth/edge/motion guidance, latent video editing, reference-based relighting, identity-consistent generation, frame-to-frame and shot-level denoising, and candidates in Wan video/VACE, LTX Video, HunyuanVideo, CogVideoX, AnimateDiff, SVD and active 2026 alternatives where warranted.

Specify how to anchor the recreated video to the **original camera motion, pacing, exact source FPS/PTS, acting, facial identity, lip synchronization, blocking, background geometry, objects and scene boundaries**. Explore combining sparse real keyframes, Pose/CoTracker/optical-flow/depth guides, source color/grain analysis, reference embeddings and constrained diffusion strength. Discuss what “enhance” means when generated content cannot be considered a faithful copy of the source. Identify reasonable use cases—damaged low-resolution backgrounds, severe compression, VFX matte repair, stylized remastering—and cases where *not* to use generation, such as dialogue closeups with subtle expressions or legally exact documentary footage.

Most video generators may not fit 10 GiB. Give a candid feasibility ladder: (1) verified practical on the current GPU, (2) plausible with short clips / quantization / offload but not certified, (3) research-only or requires larger GPU. Do not present remote inference or an API bill as an invisible requirement. Recommend explicit provenance labels and source-preserving fallback whenever generative output drifts.

## Research track D — Quality- and content-aware scene orchestration

Imagine Smart Finish making **different technical decisions within the same film**, with scene boundaries and source identity kept exact. Investigate local scene classification based on observed texture, motion magnitude, noise, faces, camera shake, low-light grain, captions, CGI, anime/cartoon versus live action, digital zoom, focus depth and compression damage. This classification should not depend on a generic LLM guessing image quality from names; use measurable evidence and confidence.

Propose how a decision engine could choose among (a) unchanged original, (b) natural source-preservation enhancement, (c) high-detail 2×/4× model, (d) temporal VSR, (e) specialized face/hair/object ROI, (f) denoising/deblocking/deblurring, (g) grain-aware restoration and (h) separate generative reconstruction. Do not permit model switching mid-shot to cause flicker, luminance jumps or mismatched texture. Include motion-aware seam smoothing, cross-frame identity checks and adjacent-shot thresholds. Preserve captions, graphics, film grain and deliberate motion blur rather than “cleaning up” everything.

## Research track E — Reuse existing MotionCoach, tracked assets and internal systems

The owner's private ecosystem includes MotionCoach pose analysis, source-bound tracking and acceptance, semantic subject masks, reframe, VFX tracking, media indexing, MMA scene workflows, adaptive color, audio archives and other editing intelligence. The public code exposes several actual interfaces, especially `motioncoach_face_bridge.py` and `adaptive_subjects.py`. **Do not invent access to the rest of the private workspace.**

Find cross-domain opportunities that save genuine engineering effort: a single source-timebase and tracking-evidence contract, source revision and model checksum cache, ROI landmarks for faces/hands/foreground items, crop trajectories for reframe and localized super-resolution, confidence/occlusion masks, optical flow and depth, motion- and camera-aware temporal QA, user-approved subject selection, caption/text protection and common artifact storage. Specify ownership: MotionCoach owns pose evidence, Final Delivery owns completed outputs, ComfyUI adapters own inference, and neither should create a duplicate global job queue. Show exactly how existing **read-only evidence can be consumed** and where a new versioned adapter is necessary. No speculative destructive cleanup or deletion of “unused” modules that might be needed elsewhere.

## Research track F — Better-than-basic film restoration

Find targeted workflows for difficult real footage:

- Night scenes: chroma noise, banding, shadow detail, film-grain retention, dark skin and lighting fidelity.
- Motion: denoising without ghosting, deblurring with source-correct shutter artifacts, restoration of motion-blurred faces, camera shake stabilization without warping.
- Old compressed clips: ringing, mosquito noise, macroblocking, edge artifacts and detail ambiguity.
- Skin/hair: pore realism, teeth/eyes, complexion and temporal identity; avoid wax, reshaped faces or artificial eye highlights.
- VFX/graphics: alpha-aware restoration, edge/caption protection, native text remaster, motion tracking, preserved typography and logos.
- Animation/CGI: style-aware paths that avoid injecting photographic textures.
- Objects/scene: selective background details, important props or textures, depth-aware regions and source color consistency.

Research viable open-source narrow implementations for each where justified. Require proof that an extra pass helps more than it harms. Identify combined pipelines that are genuinely better than naive model stacking, including the correct order of debanding, denoising, VSR, grain restoration, resizing, localized enhancement, stabilization and final encoding.

## Research track G — Smarter compression and publishing

Explore a **quality-targeted delivery engine** that chooses codecs, bit depth, encoder settings, quantizer/grain policies, keyframe strategy and final bitrate based on real source/scene evidence and destination platform. Compare compatibility versus file size and encoding time for H.264/x264, H.265/x265, AV1/SVT-AV1 and actual GPU encoder admission. Validate YouTube Shorts/TikTok/Instagram compression realities rather than assuming “higher bitrate always wins.” Design source-specific 4K/6K alternatives with real time and quality estimates, not arbitrary fixed numbers. Avoid accidentally recompressing the user's final video twice.

Consider pixel-accurate independent QA versus native Premiere/Media Encoder; sample and full-clip perceptual metrics, temporal alignment validation, grain-aware objectives and visual audit. Preserve all original source and PCM/audio timing properties through the current Final Delivery owner.

## Research track H — User interface and practical workflow

Propose a **simple, non-bloated expert UI** within the existing Smart Finish/Command Palette system, such as:

- An easy **Best Natural Quality**, **Best Detail**, **Fast**, **Selective Face**, **Temporal VSR** and **Experimental AI Remaster** tier, with conditional availability rather than inert buttons.
- A genuine source-specific **2K / 4K / 6K / higher** choice supported by local model readiness, VRAM and measured render-time ranges; clear distinctions between actual AI detail and output-canvas resizing.
- An optional before/after/ghost/temporal-flicker inspection of internal small representative frames, without forcing the creator to manage test media or HTML comparisons.
- A visual map of exactly what regions the AI will touch, including faces, hands, captions and protected original areas, with explicit user authorization on ambiguous identities.
- Reliable job status, GPU/memory warnings, render estimation with uncertainty, resume/cancel, safe fallback, verified output folder and automatic scratch cleanup.

Favor straightforward controls, advanced settings when genuinely useful, minimal user intervention for well-supported cases, and zero bloat. Do not design imaginary screens that require replacing the entire editor.

## Research track I — Future-proof architecture and model lifecycle

Evaluate the existing versioned `video_backend_router.py` and `comfyui_model_catalog.py` against a model-agnostic plugin interface. How should adapters declare checkpoint hashes, licenses, image/video conditioning, temporal context, native scale, model precision, inference device, memory limits, tile/window requirements, output color depth, audio/source preservation, expected startup and fallback behavior?

Research model swapping without rewriting the whole pipeline, content-addressed/validated cache reuse, locally available weights with no render-time downloads, model hot-swapping and graceful unloading, GPU budgeting with multiple agents, and constrained deterministic execution. Think about a future **plug-in host** for genuinely new temporal and generative models, while leaving its owner and state architecture stable. Compare implementing such abstractions now versus delaying until at least two real compatible model families exist. Reject scaffolding, empty features, extra queues and meaningless adapter layers.

## Research track J — What is actually novel?

After surveying public projects, challenge yourself to identify **five non-obvious, high-value combinations** that most upscalers do not execute well. Examples to investigate—not predetermined answers—include:

- **Identity-constrained face ROI VSR** with full source-frame verification and anti-flicker control.
- **Content-addressed, source-aware GPU remaster cache** that reuses models/ROI across real edits without stealing another tool's authoritative data.
- **Per-scene neural budget allocator** that spends processing only where subjective improvement is measurable.
- **Automatic grain/film-look consistency guard** that prevents AI from making cinema look plastic across cuts.
- **Multimodal motion/pose/depth guidance for local reference-video regeneration** on short, consented segments.

Your better ideas may be different. Prioritize actual novelty, owner reuse, measurable visual benefit and realistic implementation on 10 GiB.

## External research protocol

Find real **official GitHub repositories, niche forks, model cards, weights, research papers, live demos, issue discussions, optimizer libraries and 2025–2026 implementations**. For every notable option verify URL, last meaningful code/model release, license and any redistribution restrictions, model size, Windows runtime requirements, precision, VRAM/RAM/runtime evidence, maintained ComfyUI nodes or direct PyTorch adapter, and relevant comparisons. Look for narrow reusable components, not whole applications to copy. Clearly identify claims you could not verify. Community benchmark anecdotes must not become authoritative GPU requirements.

Do not assume source fidelity from FID/CLIP score alone. For faces require perceptual and identity-change evaluation with human review and source-bound temporal tests. For audio, maintain exact synchronization. For new diffusion paths, disclose hallucination and copyright/provenance risks. Propose genuine source-preserving fallbacks.

## Required output — deliver a concrete research blueprint

1. **New capability opportunity map:** twelve or fewer strongest functions ranked by creative value, technical feasibility on RTX 3080, quality gain, speed impact, ownership reuse, user complexity and maintenance burden. Keep a separate list of speculative bets.
2. **Model/technology matrix:** exact official links for serious candidates, model type and actual purpose, native resolution, temporal context, weights/license, last update, 10 GiB support evidence, quality risks, estimated compute cost with uncertainty and integration path.
3. **Three product directions:** (A) source-faithful high-quality remaster, (B) intelligent selective restoration, (C) explicitly generative reference-video reconstruction. Explain when each should be chosen and what source/identity invariants cannot be guaranteed.
4. **Architectural connection map:** show how the best innovations plug into existing backend/model/Final Delivery/MotionCoach contracts without a second app, tracker, database, scheduler, source reader or publishing owner. Identify specific code files and where private integration boundaries prevent certainty.
5. **Prototype acceptance recipes:** concrete source types, GPU/VRAM instrumentation, expected visual artifacts, motion/identity/grain checks, audio/timing tests, comparative metrics, user review criteria, safe fallback/rollback and *pass/fail* gates. No invented benchmark footage.
6. **Practical roadmap:** a dependency-ordered sequence with easy wins, medium engineering investments and longer-term R&D; mark **ready to integrate**, **must be locally tested**, or **not feasible on current hardware**. Explain which exploratory ideas should be killed early and how to detect a bad investment.
7. **Top five surprising recommendations:** ideas that are not already in Smart Finish, that have compelling evidence, and that materially improve creator experience or image quality. Be explicit about *why* they outperform a simpler solution and how the integration stays clean.

Finish with a **single recommended next-generation architecture**, a precise shortlist of the five best open-source modules to evaluate first, and an ordered experimental plan. This is a **research document, not a coding prompt or speculative implementation claim**.

**Non-negotiables:** No vague brainstorms, fake 6K detail, made-up GPU performance, unverified forks, one-click downloads of untrusted weights, mandatory paid APIs, uncontrolled background GPU services, destructive changes to private user projects, additional lossy encode generations or generative face hallucination passed off as faithful restoration. Cite the GitHub source and credible external references throughout, clearly mark unknowns, and explain what **not** to build as carefully as what to build.
