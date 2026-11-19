# Video Upscale Worker

## MotionCoach face ROI → ComfyUI selective enhancement — November 18, 2026

An **explicitly selected** `--region-mode motioncoach_face` path now reuses
**MotionCoach's existing immutable, published Pose Landmarker evidence**
instead of creating another person tracker, new face detector, model store or
background scheduler. It is an optional branch of the current
`comfyui-gan` backend and reuses its pinned x2/x4 Real-ESRGAN models,
GPU ownership, job receipts, source audio, frame clocks and final
H.264/compression publication contract.

### What it does

1. MotionCoach's existing `load_pose_playback_bundle()` validates
   `bundle_integrity.json`, source evidence, model artifacts and safe
   `pose_arrays.npz` contents. Smart Finish independently calculates
   the **full SHA-256 of the actual selected video**, checks the exact
   `source_binding_id`, image size/orientation, each decoded frame index,
   presentation clock and per-frame landmark confidence. Every frame must
   have accepted face landmarks; missing/ambiguous evidence blocks the
   selective route before GPU inference.
2. The existing MediaPipe-33 nose, eye, ear and mouth landmarks define a
   conservative square face crop with forehead/chin margin. Rapid track
   jumps, missing/weak facial features, huge crops and incomplete pose
   coverage are rejected. No face identity is inferred from MotionCoach.
3. Only the source face crop enters the already-pinned ComfyUI GAN model,
   **not** the whole scene. The rest of the image remains source-based
   Lanczos enlargement, without AI-generated background detail. A soft
   feather mask composites the AI-enhanced face crop back onto the full
   original-derived frame. The optional source-preservation slider blends
   a small amount of the original face patch into AI output and retains
   the same 0–25% safety bounds.
4. Face regions are short-lived per-job PNGs, processed one at a time
   for predictable memory. Exact byte-identical source face crops and
   equal target patch geometry reuse their verified neural result within
   the job; moving faces are processed normally. Composited frames still
   must pass frame-to-frame, source color, exact source audio/PTS and
   complete final MP4 decode checks. Scratch is deleted after completion
   or failure.

### Advanced opt-in entry

```powershell
python "Spiffy Workspace/tools/run_spiffy_final_delivery.py" --input "C:/renders/finished.mp4" --ai-backend comfyui-gan --model-id auto-quality --region-mode motioncoach_face --motioncoach-pose-root "C:/MotionCoach/published_pose_bundle" --motioncoach-source-binding-id "source.exact_identifier_from_MotionCoach" --final-long-edge 3840 --compression-profile quality --execute
```

The MotionCoach path and binding ID are *real current published evidence*
inputs; the illustrative path above is not an already-existing bundle.
The source selection must actually be the same file whose SHA-256 is
declared by MotionCoach. This mode is also supported through typed
Command Palette job arguments, but **is not silently enabled by the
existing default ComfyUI commands**, as they have no guaranteed
published pose bundle for a selected video. If the source has no
accepted published pose, keep the normal full-frame AI workflow.

This is **regional upscaling using existing general-purpose Real-ESRGAN**,
not specialized CodeFormer/GFPGAN identity restoration, face swapping or
full generative video-to-video re-creation. Third-party custom nodes
which download unfamiliar models automatically are deliberately not
activated in the current private offline ComfyUI server. Face identity
and temporal consistency across long scenes still require real-world
validation; human pose evidence never proves a particular face identity.
Generative reference-video reconstruction is a separate opt-in model
family, not something safely simulated by a frame image upscaler.

Internal verification at implementation time includes real canonical
MotionCoach bundle-reader/sha/timeline rejection tests, conservative
ROI crop/composite checks, and a complete **real FFmpeg video/audio/final
decode** round trip using a synthetic face-model result (no claim of
actual GPU model quality from that mocked neural stage). A real
ComfyUI face-ROI render must be separately accepted on an available GPU
before claiming production-level neural ROI reliability.

**Live GPU admission result:** a real face-only ComfyUI run initially
exposed two genuine integration defects: the face crop was placed in a
directory outside ComfyUI's isolated `--input-directory`, and an overly
broad readiness `except OSError` caught the downstream exception after
the context manager's `yield`, hiding it with `generator didn't stop
after throw()`. Both were corrected. Crops are now written directly
into the registered private input directory, and errors from actual
inference/compositing propagate unchanged, with bounded ComfyUI HTTP
failure diagnostics and private process teardown. The later live
attempt declined at GPU preflight because free RTX 3080 VRAM was below
the existing 3584 MiB minimum: **the neural face-ROI result remains
unqualified on actual live GPU** until an accepted run completes.
The worker also rechecks free GPU/scratch capacity while holding
the incumbent GPU lock, just before starting ComfyUI, to protect
against concurrent agents. New local face-color/exposure checks
catch large face-only changes that might be hidden in full-frame metrics.

## Measured quality compression and full-decode publication gate — November 18, 2026

The optional `--compression-strategy measured` path now runs **inside the
existing ComfyUI Smart Finish/Final Delivery backend**, after its source-clock
and sample-exact-audio ProRes intermediate is verified. It does **not** create
another user-facing encoder application, database, output watcher, or
lossy-generation chain.

- The default is still `fixed` compression and the requested quality profile.
  The current quality profile uses **H.264 CRF16 slow**, and the existing
  Topaz smart-finish/export profiles remain unchanged. The new Command
  Palette action is **Smart Finish — Measured Compression**
  (`media.smart_finish_comfyui_measured`).
- For supported output canvases up to 3840×2160, progressive exact-CFR
  inputs of at least five seconds at 12–60 FPS, bounded job-private FFmpeg
  pilots evaluate CRF18 and then CRF20. The pilots use the real full-resolution
  lossless master as the comparison reference. Each candidate must pass
  **VMAF mean ≥98.2, minimum frame VMAF ≥94.0, float SSIM ≥0.996 and
  luma PSNR ≥47.5 dB** for every tested window. The selector uses no
  downscaled 1080p score to authorize a 4K/6K final encode, and does not
  claim these objective measures guarantee grain or facial microtexture.
- One centered sample is checked for shorter eligible clips; two nonoverlapping
  samples for ≥15-second clips. Each sample contains 8–24 original-rate frames
  and is at most approximately one second. The total pilot wall budget is
  **45 seconds**, with cancellation, hard subprocess deadlines and a graceful
  return to the previously verified CRF. Above 4K, on insufficient evidence,
  with an inaccessible VMAF build, or when `maximum` CRF14 was requested,
  the baseline quality settings remain in force. The pilot never publishes
  its temporary MP4 or its JSON metric files.
- The original final MP4 still receives **one** full-length H.264/AAC lossy
  encode from ProRes. Its CRF and pilot evidence are recorded in the existing
  job result. The new encoder mode does not silently change audio, FPS, shot
  timing, intermediate/master pixels, source filenames or output locations.
- Beyond container/decoded-PTS checks, **full FFmpeg decode of every final
  picture and all audio tracks** is now required before publishing ComfyUI
  results. A corrupt frame, audio decoder failure, unexpected frame count
  or mismatched sample extent blocks final publication.

Internal validation: a native 640×360 eight-second lossless sample selected
CRF20 after real FFmpeg/libvmaf/SSIM/PSNR pilots passed; in an unrelated
six-second 128×72→512×288 full ComfyUI test, a CRF18 pilot scored
VMAF **99.53** but only SSIM **0.991593** and luma PSNR **45.29 dB**.
It was rejected and the full verified MP4 retained **CRF16**. This
demonstrates a real fail-closed quality choice, not a blanket compression
speedup. The six-second test took 122.71s total under the observed host load:
80.662s startup and 39.392s neural processing/validation; the bounded
compression pilot took just 0.407s. These separate timings do not predict
future clips. A separate real internal final H.264/AAC output passed
the new all-frames/all-audio decoder gate.

**GPU encoding is not silently enabled:** the standalone FFmpeg exposes
`hevc_nvenc`, but on this machine the real encoder fails because
NVIDIA driver **560.94** provides NVENC API 12.2 while FFmpeg requires
API 13.0 (driver 570+). The workflow keeps its verified software H.264
path rather than shipping a broken hardware HEVC option or changing
drivers while other GPU work is active.

## Swappable local ComfyUI models and bounded batch processing — November 18, 2026

**Additional validated speed shortcut:** the frame-GAN adapter now computes
the complete SHA-256 of each job-local decoded PNG (plus its exact byte length)
before deciding whether the GPU must process that frame. Identical inputs
under the *same* model, blend and output geometry reuse the already-verified
AI PNG through a job-local NTFS hardlink, with a verified copy fallback.
No similarity heuristics, interpolation, guessed scene continuity, or changed
frame count are involved. The cache exists only for the current job and
is discarded with its disposable outputs. An actual 8-frame constant-video
GPU run processed **one unique frame and reused seven exact frames**; full
source/audio/clock/final-MP4 validation passed at **9.67s** total, of which
8.101s was isolated ComfyUI startup and 0.79s was the model/validation
portion. This is a synthetic hold-heavy case, not a moving-video speed claim.

Private ComfyUI startup cost can still dominate very short videos. The
lightweight/offline server eliminates unrelated custom-node startup while
retaining strict security, but a long-running background GPU server was
intentionally not introduced: it would compete with Premiere, Topaz and
other active RTX 3080 workloads. Models are reusable within one job and
selectable across jobs through a stable adapter/catalog.
Resource admission also checks the selected model's *native* 2× or 4×
intermediate tensor, not just the requested final raster. A 1080p-to-4K
job using native 4× can require the higher GPU-memory threshold even
though its final output is 4K; native 2× can be admitted at lower memory
when its output is genuinely within the scale. This is a conservative
gate, not a measured peak-VRAM guarantee.

The ComfyUI engine now has a **single allowlisted dispatch contract** in
`video_backend_router.py`, a **versioned, immutable model catalog** in
`comfyui_model_catalog.py`, and one fully implemented frame-GAN execution
adapter (`comfyui_video_backend.py`). These are separate parts of the existing
Smart Finish / Command Palette / Final Delivery system, **not** a new app,
daemon, database, or independent GPU scheduler. A future temporal VSR backend
can implement the same result/progress/audio/publication contracts without
pretending its model can be run through a per-frame GAN graph. The only
production-admitted alternative backend today is `comfyui-gan`.

Three upstream Real-ESRGAN checkpoint files were installed and fully hashed in
the existing ComfyUI `models/upscale_models` directory (no duplicate model
weights in the Git workspace):

| Model ID | Official checkpoint SHA-256 | Purpose |
| --- | --- | --- |
| `realesrgan_x2plus` | `49FAFD45F8FD7AA8D31AB2A22D14D91B536C34494A5CFE31EB5D89C2FA266ABB` | Photographic native 2×, for outputs up to twice the source long edge |
| `realesrgan_x4plus` | `4FA0D38905F75AC06EB49A7951B426670021BE3018265FD191D2125DF9D682F1` | Photographic higher-detail 4× baseline |
| `realesrgan_general_x4v3` | `8DC7EDB9AC80CCDC30C3A5DCA6616509367F05FBC184AD95B731F05BECE96292` | Small, faster 4× general model; may restore less fine detail |

Source URLs and native scales are recorded in the pinned Python catalog.
`--model-id auto-quality` chooses native x2 for a <=2× target and x4 for a
larger target. `--model-id auto-fast` explicitly chooses the small x4v3
model. Any model ID can be chosen explicitly, but the native-scale, signed-off
checkpoint hash and 10 GB GPU limits must all pass. An unexpected or mutated
model file blocks rendering. **No user job automatically downloads or
executes new community nodes.** The private GAN server now starts offline with
custom nodes/partner APIs disabled; it uses only built-in ComfyUI nodes.

The existing Command Palette includes a third **Smart Finish — ComfyUI Fast
Model** entry in addition to the quality-first and 6K commands. All dispatch
through the same router and return the same verified Final Delivery result
contract. The existing Topaz route, render defaults and VideoSeal behavior are
preserved. The read-only Quality Advisor lists each verified local checkpoint,
shows native 2× versus 4× source eligibility, and does not invent times or
claim that temporal video models are installed.

ComfyUI's **ImageBatch** node now groups up to three small input frames into
one graph/API request under conservative GPU/pixel limits. Large 4K/6K model
tensors still run one frame at a time; an absent/deprecated ImageBatch node
causes a safe one-frame fallback. Model weights stay cached within that
private job's server. Exact input/output frames remain in order; the worker
verifies every output, preserves the source/audio clock and deletes processed
source PNGs promptly. Optional `--source-preservation 0.08` uses the built-in
`ImageScale` and `ImageBlend` nodes to mix 8% of a conventional original
resize back into the AI-enhanced result for source identity/color continuity.
Default 0 makes **no** modification, and blending is rejected above 4K until
separately qualified. The strength is capped at 25%; source blending may
reduce sharpness and is not claimed to automatically improve any footage.

**Internal same-server batch check (8 frames, RealESRGAN x4plus):**
single-frame inference 6.366 s versus three-frame batching 3.998 s,
approximately **1.59× faster for measured inference**, with eight
pixel-identical outputs. This excludes server launch, decode, encoding and
long-video GPU-pressure effects. In separate complete 8-frame tests with
distinct cold starts: x2plus at 256×144 took 13.77 s, tiny x4v3 at 512×288
took 15.05 s, and x4plus at 512×288 took 22.91 s. Those are **not**
controlled complete-job benchmarks because output sizes and startup times
vary. An optional 8% source-blend job also passed real local GPU inference,
sample-exact audio and final MP4 checks. Test media is internal only and
must be removed after verification.

### Local entrypoints

```powershell
# Stable engine interface and explicit model choice; only an --execute makes video.
python "Spiffy Workspace/tools/run_spiffy_final_delivery.py" --input "C:/renders/clip.mp4" --ai-backend comfyui-gan --model-id auto-quality --final-long-edge 3840 --compression-profile quality --execute
python "Spiffy Workspace/tools/run_spiffy_final_delivery.py" --input "C:/renders/clip.mp4" --ai-backend comfyui-gan --model-id auto-fast --final-long-edge 3840 --compression-profile quality --execute
python "Spiffy Workspace/tools/run_spiffy_final_delivery.py" --input "C:/renders/clip.mp4" --ai-backend comfyui-gan --model-id realesrgan_x4plus --source-preservation 0.08 --final-long-edge 3840 --execute
```

**Not yet certified:** RelayVSR/FlashVSR/SeedVR2 temporal video inference,
long-form VRAM/scratch behavior, 6K output on every source, faithful
AI-generated face details in all scenes, and superiority to Topaz. Models that
need a different or larger GPU remain behind a separate, fail-closed adapter.

## Smart Finish ComfyUI production hardening — November 18, 2026

**Latest source-bound validation:** a separate 1920×1080 two-frame source
completed an actual 5760×3240 Real-ESRGAN/ComfyUI delivery in **39.65s**
including final H.264/AAC, with the stricter source PTS and original audio
sample-count checks passing. No source media or installed model was modified.
The internal test footage was removed after verification; only these compact
implementation notes remain. This is not a forecast for long 6K videos.

The existing read-only Quality Advisor now distinguishes the Topaz 4096-pixel
working limit from the **explicit ComfyUI GAN 6K candidate**. It verifies
the pinned checkpoint and installed private-server runtime without expecting
an always-running server, and lists the 6K candidate only for bounded eligible
short clips. It leaves timing unknown and does **not** automatically prefer
the GAN over Topaz without source-specific quality evidence.

Before decoding any frames, the GAN path also checks the *native x4 tensor*
pixel area (not only the final output dimensions), GPU-memory headroom,
available system RAM and conservative scratch-disk capacity. On this 10 GB
RTX 3080, the current preflight requires at least 6144 MiB free GPU memory
for larger-than-4K results and 3584 MiB for smaller targets. Source frames
whose native x4 processing would exceed 33,554,432 pixels are rejected with
a clear reason, avoiding possible multi-gigabyte OOM/scratch crashes. These
are conservative admission thresholds, not peak-memory benchmarks.

The optional local ComfyUI GAN path now validates the **decoded presentation
clock**, not just the FPS header: it requires a uniform rational CFR grid,
source start and last-frame endpoint, correct output frame coverage and
unshifted reconstructed presentation timing after both the lossless ProRes
intermediate and final H.264 encode. Unsupported VFR, nonzero start offsets,
HDR/alpha/interlace and ambiguous media fail before inference.

For every source audio track, the worker uses the source's exact valid sample
count to trim decoder padding during the lossless PCM assembly. Both stages
must pass the incumbent sample-rate/channel-count/sample-extent/start checks;
an export with an extra decoded AAC padding tail is no longer accepted.
Audio remains one final AAC encode from the lossless intermediate.

The FFmpeg decode and intermediate-assembly processes now honor cancellation
with a hard timeout and terminate **only their own child processes**. The
private ComfyUI process is also terminated when its job exits. Every job
removes its own disposable frame sequences and large ProRes intermediates,
even after failure, retaining progress/result and a small stderr tail. Do
not delete shared model files, other ComfyUI sessions or source footage.

Final MP4 bytes are encoded and checked **inside the job first**; only
verified media is published to the output directory. Publication reuses
the existing artifact-lease owner, chooses a collision-safe name under that
lock, uses same-volume atomic rename or checksum-verified cross-volume copy,
and rejects source changes immediately before publication. Existing output
files are never silently overwritten. A receipt failure after publication
records the already-published file for reconciliation rather than deleting
that otherwise valid user data.

Focused and related regressions passed **379 tests plus 15 subtests** across
Smart Finish/ComfyUI, the Command Palette, Final Delivery, rational frame
rates, adaptive model selection and adaptive video enhancement. A complete
internal 8-frame 512x288 ComfyUI+H.264 run passed after sample-exact audio
handling (22.87s under observed local load). These are code/short-media
acceptance claims, not long-form visual-comparison, Topaz crash-resolution,
or automatic superiority claims.
The later readiness/memory/cleanup guards also passed 88 focused tests.

## Smart Finish live ComfyUI GPU route and quality-first compression — November 18, 2026

Three explicit **ComfyUI** palette entries now use the existing job/progress/GPU
lock and source/final-delivery verification components:

- `media.smart_finish_comfyui`: Auto quality-model selection, 3840-long-edge default,
  original frame rate, one high-quality H.264 encode.
- `media.smart_finish_comfyui_6k`: the same pinned local GAN checkpoint,
  5760-long-edge target where source scale and pixel area qualify. Do not
  mislabel 4K stretching as native 6K neural processing. The 6K entry has only
  a bounded two-frame full-resolution local acceptance, **not** long-form
  6K visual/throughput certification.
- `media.smart_finish_comfyui_fast`: fast, small x4v3 checkpoint and the same
  audio, source identity, frame-clock and final-output checks.

`comfyui_video_backend.py` starts a private local-only ComfyUI process with the
installed external-assets `.venv` and registered `UpscaleModelLoader` /
`ImageUpscaleWithModel` / `ImageScale` / `SaveImage` nodes, checks the exact
official RealESRGAN_x4plus.pth SHA256, decodes real input frames into the job's
isolated input directory, and submits frames sequentially over the real local
ComfyUI API. Generated PNG files are moved within the isolated job directories
without redundant HTTP upload/download copies. Unlike trained video-temporal
VSR, this is an image model applied to video frames: the worker checks
per-frame consistency and large AI-only temporal discontinuities, but **does
not** guarantee no texture flicker on all natural motion. RelayVSR, FlashVSR,
SeedVR2 and other uninstalled video-model runtimes remain unavailable.

Then existing FFmpeg assembles the real model frames into a lossless ProRes
422HQ / 24-bit PCM intermediate, retaining original frame count, exact CFR and
all original audio streams. **One final lossy** H.264/AAC pass uses the shared
Final Delivery encoder (`libx264`, slow, CRF 16 in the quality profile) and
current full decode, audio/geometry/frame-count guards. Source media is never
modified. ComfyUI video route currently admits only short, opaque SDR,
progressive exact-CFR footage at 1–60 source FPS with a bounded 1800-frame / 60s
budget. No interpolation, VideoSeal, HDR or variable-frame-rate support is
claimed. A failed local Comfy job reports a failure; it does not silently
substitute a Topaz/CPU result.

The incumbent unwatermarked `smart_finish_master` delivery plan now optionally
accepts `compression_profile` values `existing` (CRF17 slow, unchanged
default), `maximum` (CRF14 slow), `quality` (CRF16 slow), `balanced` (CRF18
slow), or `compact` (CRF20 slow). These are subjective quality/size tradeoffs,
not guaranteed VMAF thresholds or time estimates. All profiles encode **once
from the verified AI intermediate**, rather than round-tripping another
compressed Media Encoder file. Existing social and VideoSeal formats keep the
current presets. `media.smart_finish_best_quality` opts into `quality` without
changing `media.smart_finish`'s established default.

```powershell
python "Spiffy Workspace/tools/run_spiffy_final_delivery.py" --input "C:/renders/finished.mp4" --ai-backend comfyui-gan --compression-profile quality --final-long-edge 3840 --execute
python "Spiffy Workspace/tools/run_spiffy_final_delivery.py" --input "C:/renders/finished.mp4" --ai-backend comfyui-gan --compression-profile quality --final-long-edge 5760 --execute
```

**Local validation, not a universal speed claim:** an internal 128×72 8-frame
source with AAC completed and fully verified at 512×288 in 21.75s, with an
identical source/output-frame check, followed by a transfer-optimized run in
18.98s. The same protected workflow including its temporal checks completed
in 23.13s under a different load. A 1920×1080 two-frame AAC source then
completed a native-model 5760×3240 delivery in 43.54s including startup,
lossless intermediate, and compression. Output was probed and fully decoded.
These are tiny fixtures with different durations/resolutions/load, not proof
of an overall performance gain over Topaz or visual acceptance on long media.
The native Topaz 4K OpenCV access-violation issue remains separately unresolved.


## Smart Finish quality choices and ComfyUI readiness — November 17, 2026

The existing Command Palette now includes **Smart Finish — Quality Options
(No Render)** (`media.smart_finish_advisor`). It uses source metadata and the
incumbent geometry limits to rank 2K/3K/4K/6K/8K canvas choices under
`best_quality`, `balanced`, and `fastest_safe`; no video is rendered and no
HTML or test footage is generated. The existing `media.smart_finish` command
keeps its prior automatic delivery defaults. A separate explicit **Smart Finish
— Best Available Detail** command (`media.smart_finish_best_quality`) processes
one selected source at its highest admitted Topaz working resolution and retains
that final canvas, without pretending that an uninstalled 6K model is usable.
The separate `quality_advisor.py`
does **not** own rendering, scheduling, media or model downloads.

Read-only CLI example (does not claim or create a Final Delivery job):

```powershell
python "Spiffy Workspace/tools/run_spiffy_final_delivery.py" --input "C:/renders/finished.mp4" --quality-advice
```

For one video, `--apply-quality-advice --execute` applies the admitted local
Topaz working resolution and same-sized final canvas. `--quality-policy`
accepts `best_quality`, `balanced`, or `fastest_safe`. Unmeasured times are
reported as unknown; comparisons are not extrapolated from unrelated GPUs or
sample clips. The current 4096-pixel Topaz working-edge / 16,777,216-pixel
working-area limits remain in effect. Source frames larger than the working
limit are not silently downscaled to qualify.

A larger **final** canvas is separately available with `--final-long-edge 5760`
or `7680` where the existing final-pixel budget admits it. Above the 4096
working-edge limit, explicit `--allow-raster-only` acknowledgment is required:
this uses the existing 4K-or-smaller neural working stage and then ordinary
resizing, **not genuine 6K/8K model detail reconstruction**. AI interpolation
stays at the working resolution for these manual larger-canvas requests.
These are non-destructive geometry options, not guarantees that the known
Topaz native 4K crash has been resolved.

`quality_advisor.inspect_comfyui()` checks only the loopback ComfyUI health API
and locally existing checkpoint filenames. A GUI install, responding API or
named checkpoint is **not** production-quality inference proof; RelayVSR,
FlashVSR and SeedVR2 remain unqualified unless actual model, node, GPU-memory,
time, image, temporal, audio and final-delivery checks pass. Do not route
production jobs to an unqualified backend or silently install model weights.


## Automatic exact-hold CPU shortcut and lower-memory quality QA — November 16, 2026

The dense adaptive source pass now checks the **original decoded native pixel
planes** before performing RGB conversion. When every native plane and color
attribute matches the preceding frame **within the same scene**, it reuses
the private preceding RGB analysis rather than converting RGB again or
repeating a full RGB comparison. Moving frames, shot boundaries, and 10-bit
differences that disappear in RGB8 still take the normal path. Every source
frame retains its mask, caption protection, exact clock and output coverage;
no frames are skipped and no lossy similarity threshold is introduced.

On a controlled 96-frame 1280x720 two-shot fixture containing 92 proven
native holds, the adaptive analysis ran **2.393s without** versus **2.315s
with** the shortcut (3.3% lower observed wall time). Both outputs had the
**same mask SHA-256, plan fingerprint, shot decisions, measured pixel
statistics and caption protection**. This is a controlled hold-heavy CPU
case, not a guaranteed speedup on moving footage.

The detailed *sampled* upscaling-quality checker now uses OpenCV native
uint8 absolute RGB differences instead of creating multiple full-size
float32 images. Source-relative mean/p99/protected-region errors and signed
brightness shift retain their original decision thresholds. Standalone
sample-metric medians measured **95.9 -> 48.9 ms at 1920x1080** and
**322.9 -> 128.5 ms at 3840x2160**, with numerical results matching the
previous formulas within floating-point rounding. Actual encoded-video QA
and corruption regressions pass. These are **per-sample calculations**, not
full-clip Topaz render speed claims.

Scoped Windows regressions: **687 passed** (157 analysis/spatial, 119
mixed-content/enhancement, 219 model/final-delivery, 93 process/preflight,
47 subject/scene/color and 52 root-compatibility tests). No new model,
service, persistent cache, or output bundle was introduced. The native
4K Topaz crash, real long-video quality review and physical Mac tests remain
unqualified; historical crash evidence and original media stay intact.

## Frame-evidenced model preflight and safe shared GPU locking — November 16, 2026

The existing one/two-frame Topaz preflight now uses native FFmpeg
`-nostdin -nostats -progress pipe:1` output to require **one actual
upscale output frame or two interpolation output frames** before
approving the configured model. Exit 0 without complete frame evidence,
or a reported `Model not found` even with exit 0, fails closed before
production rendering. This is the same bounded single native preflight:
no second model run, new decoder, cache, or download is added.
Unsupported filter names and model identifiers containing FFmpeg option
or filter-graph separators are rejected before process launch, preserving
the forced `download=0` and intended GPU settings.

The incumbent shared `SharedGpuLock` now treats diagnostic log failures
as non-authoritative, including acquisition, nested reuse, cancellation,
release and stale-lock cleanup. A failed/short owner-marker write after
exclusive lock creation closes its descriptor and removes **only that
same verified file identity**, preventing an incomplete lock from
stranding the GPU lane. Tests verify subsequent acquisition is immediate
and the original render exception is preserved.

The installed Topaz-bundled FFmpeg advertised both native filters and
passed an **actual two-frame CPU-only null-filter progress test**.
Separate actual standard-FFmpeg tests passed the complete/incomplete
one/two-frame acceptance checks. These qualify the progress contract,
**not** native Topaz AI inference or output appearance. The known 4K
`opencv_world4120.dll` access violation was not blindly rerun.

Previous isolated Windows verification: **639 tests passed** across
disjoint modules: 159 preflight/process/final-enhancement, 120 adaptive
enhancement/model selection, 157 spatial/analysis, 42 mixed-content QA,
109 final-delivery/frame-rate and 52 root compatibility. The native
Topaz 4K failure, physical Mac qualification and representative long
real-media quality remain open. Existing original media, installed
models, and external jobs were not modified.

## Native process failure integrity and preflight privacy — November 16, 2026

Render diagnostics are now best-effort: a missing/full/blocked `stderr.txt`
or interrupted log stream does not conceal the actual native exit code,
watchdog timeout or user cancellation. The bounded process-output tail still
supplies failure details; `process_timings` records `diagnostic_log_error`
when writing fails. A successful render is not aborted just because its
optional log stopped accepting data. The known first-use `0xC0000409`
retry and preflight timeout decisions no longer depend on writable logs.
The separate `0xC0000005` access-violation path still **never auto-retries**.

All model-preflight error returns and timeout log excerpts now use the
existing licensing redaction, including the `owned - cached` marker.
Invalid process, startup and stall deadlines (zero, negative, NaN, infinity,
boolean or text) fail **before** launching an external process. Two actual
CPU-only FFmpeg 3840x2160/four-frame tests passed with normal and deliberately
blocked logging; neither runs or qualifies Topaz's native 4K model.

Previous isolated regression: **615 passed** (42 mixed content, 76 adaptive
enhancement, 157 spatial/analysis, 110 model/final-delivery, 109
delivery pipeline/frame-rate, 69 process lifecycle/model preflight and
52 root compatibility). These are disjoint test modules. The vendor's
previously recorded native 4K OpenCV access violation and physical Mac
qualification are still unresolved; no installed model, GPU runtime, source
media or new service was changed.

## Source-bound localized 60 FPS QA and lower-overhead render logs — November 16, 2026

The temporal validator now rejects one-frame **partial black/white and
saturated-color tears** (blue/green/magenta) that pass whole-frame brightness
checks, including damage at the very first and last output frames. Small
aspect-preserving previews compare affected output pixels against the exact
authored source and both available output neighbors (one at video boundaries).
Source-authored graphics, cuts and flashes remain protected; a photo's normal
interpolated motion is never required to equal an unmoving source frame.
This uses the existing 96px previews and decoded frames with no extra model
or decoding pass. The separate spatial-upscale regional guard remains intact.

Each process now writes FFmpeg output through one buffered log handle instead
of reopening the diagnostic file for every progress key/value line. The first
native diagnostic is flushed immediately, later records are flushed at most
0.5s apart while output arrives and at each FFmpeg progress boundary, with
final flush on success, cancellation or failure. A real CPU-only FFmpeg
3840x2160/4-frame progress smoke passed exact output-frame accounting, visible
stage percent and complete logs; it is **not** a native Topaz 4K model pass.
On the local Windows filesystem, a four-run, 5,000-line logging-only median
measured 1,039.31 ms for the old per-line file open versus 1.76 ms for
buffered writing. This measures **log I/O only**, not overall rendering speed.

Previous isolated Windows tests: **460 passed** in the targeted cohorts
(42 mixed-content, 76 enhancement, 157 spatial/analysis, 110 model/final
delivery, 23 process lifecycle and 52 root compatibility); Ruff and syntax
checks passed. These certify focused objective contracts, not universal
visual quality or overall render-speed improvement. The native 4K Topaz
access violation and physical Mac qualification remain unresolved.

## Current native-hold speed and 60 FPS artifact protection — November 16

Smart Finish now compares decoded high-bit-depth source planes with a
zero-copy, **full-byte** host C-runtime comparison. If unavailable,
the previous exact Python comparison remains the fallback. No 10-bit
detail is rounded away and no hash or sparse sample can authorize a
source hold. On separately allocated Snowfall-derived YUV422 10-bit
frames, the comparison measured **15.10 -> 0.70ms at 1080p** and
**60.51 -> 3.11ms at diagnostic 4K**; moving-frame checks were
already microseconds and remain in that range. A complete 24-frame
1080p identical-source analysis ran **1.85 -> 1.54s** with identical
mask bytes and plan fingerprint. These are CPU comparisons/prepass,
NOT whole-video Topaz render speed claims.

The existing 60 FPS temporal validator also rejects single invented
black/white frames in caption-free shots, while allowing source-authored
flashes and protected cuts. It reuses the already-required small source
preview for caption and exposure screening, avoiding a duplicate
resize/decode. No new model, daemon, queue, database or persistent
benchmark files were added. The bounded motion, exact source PTS,
caption/shot protection, crash resume and Mac-safe fallback contracts
remain. Production-native 4K model crash/quality and physical Mac
qualification are still separate open gates.

## Source-safe selective 60 FPS — November 16, 2026

One consistent exact rational output frame rate (e.g. 60/1 or
60000/1001) is preserved across an edited video; its individual
regions/intervals are NOT misleadingly marked as variable FPS.
Native Apollo remains the interpolation owner for actual moving
photographic content. The same already-generated protection-mask
stream now also holds **exactly repeated decoded source frames**
and protects fallback-only caption/graphic edges at the source
presentation clock, without morphing typography or inserting frames
from the next shot across a cut.

Repeat status compares all ORIGINAL decoded native pixel planes,
including 10-bit low-order bits, not merely equality after RGB24
quantization. This decision is independent of any optional analysis
cache. Full-source still shots also skip optional *model-comparison
pilots*; the existing native interpolation graph is still used for
moving portions. The source-hash/cache schema was updated so earlier
RGB8-only mask receipts cannot be silently reused.

The temporal QA now rejects a video that merely repeats existing
frames during measurable source motion, while allowing intentional
holds, hard cuts and near-native-rate outputs. CPU FFmpeg graph
substitutes verified exact timestamps, source masking and output
frame counts at 24->60, 23.976->60, 25->60, 30->60 and
29.97->59.94. Separate 10-bit YUV422/YUV420 tests verified the
native-depth hold boundary. These are CPU/source-identity tests,
NOT new native Topaz quality or universal 60 FPS smoothness claims.

Current isolated regression: 526 workspace tests+15 subtests and
127 root tests+18 subtests, lint/syntax/diff/source guards PASS.
Two tests in the separately edited shared scene detector failed
while that owner was changing concurrently; do not claim those
tests were resolved by these FPS changes. This work created no
extra service, model store or persistent report bundle.

## Current measured CPU shortcut — November 16, 2026

The same source-bound detail and authored-caption masks now run in
parallel **only when both are needed**, input pixels are large enough
and there is at least 4 GiB of free system memory. The analyzer owns a
maximum of two mask-worker threads, reuses them within the source
analysis and shuts them down on completion or failure. If memory falls
during a long clip, it switches to the identical serial path; small
clips and hosts without a trustworthy memory probe stay serial.
No source frames, cut boundaries, glyphs, mask pixels, audio samples
or timestamps are skipped or approximated.

On eight existing real 1080p frames, the independent full-frame mask
stage measured about 0.593s serial versus 0.449s parallel (roughly
24% faster) with all source masks and evidence identical. This is not
a 24% overall Topaz render gain. End-to-end mask-stream comparison
passed identical pixels, cadence and shot settings; current regression
passed 537 workspace tests+15 subtests and 127 root+18 subtests,
plus Ruff, syntax, diff and guarded source checks. No extra service,
model, persistent report or background scheduler was created.
Vendor native 4K visual and physical Mac qualifications remain open.

## Current output-quality safeguards — November 16, 2026

The existing objective enhancer now screens **every source/output frame
transition** using two tiny, aspect-preserving RGB previews. A repeated
output frame during clearly moving source footage is rejected, while
intentional source holds stay valid. The new freeze rule caught a real
48-frame test failure that had previously passed; a retained 96-frame
Snowfall-derived export still passed with no false freeze warning.
This is a bounded artifact check, not proof that AI detail looks ideal.

The metadata probe now identifies alpha-capable video pixel formats.
Before GPU work, the worker rejects transparent motion graphics,
HDR PQ/HLG without qualified tone mapping, and interlaced footage
without a qualified deinterlace path. The current opaque SDR
Topaz/MP4 route must not silently destroy source transparency,
color or field structure. Normal SDR/progressive media is unaffected.
Latest source-hash-stable Windows checks: **528 workspace tests +
15 subtests, 127 root compatibility tests +18 subtests**, lint/syntax
and diff PASS. Native 4K vendor-crash cause, optional HDR/alpha
conversion, semantic long-shot quality and physical Mac rendering
remain unqualified.

## Current crash continuation — November 16, 2026

Final Delivery now journals its active child in the existing parent
result before launching it. A rerun with the same parent job ID reuses
only revalidated completed media or forwards the matching failed child
through existing worker checkpoints; source, plan, model and media
changes fail closed. Abandoned parent/child process identities are
checked by their recorded PIDs, and a live attempt cannot be superseded.
Native child exits without a terminal progress state become crashed;
an interrupted parent after a completed child can recover that output
only after the normal full delivery proof is rechecked. Unknown process
identity never authorizes a takeover.

Read status without rendering by running run_spiffy_final_delivery.py
with --jobs-root "_jobs" --inspect-job JOB_ID. To continue later, rerun
the original inputs/options with the same parent --job-id after the
old process has exited. Verified completed work is reused; unproven
work is rerendered. Ordinary native first-use recovery stays bounded;
Windows access violations are not blindly retried. Current contract
verification: 526 workspace tests+15 subtests, 111 root tests+12
subtests, Ruff/syntax/diff PASS, source hashes stable. No new service
or persistent report files. Native 4K and physical Mac qualification
remain open.

## V19 true render-unit cost and native failure containment

The live 2-frame Snowfall 1920x1080 -> native Topaz 3840x2160
trial crashed in vendor opencv_world4120.dll with Windows 0xC0000005
before generating frames; the cause is not proven to be VRAM.
The worker now classifies these signed/unsigned access violations,
forbids automatic replay of the same crashed configuration, and
omits license/account details from user-facing exceptions without
destroying original job logs. A separate 720p two-frame control
was deferred under GPU load and must NOT count as a render.

Model selection uses actual active shot-render-unit launch counts,
not an assumed one startup per video, and refuses a global model
switch when any measured shot is materially slower in steady
processing. Manual selection, source masks, scene cuts and resume
protection still use the existing canonical owners.

Subsequent maintenance removed another failure-mode: upscale and
interpolation (including CPU recovery) now discard only *newly made,
incomplete MOV files in their own job temp folder*. Source files,
preexisting outputs, foreign folders and links are never cleaned.
Even a full disk or broken log cannot replace the original failure
exception. Current source verification: **516 workspace tests +15
subtests, 111 root tests +12 subtests**, Ruff, syntax, diff and guarded
source identity PASS. Six task-owned pytest scratch roots from the first
pass (791 files /18.57 MiB) and four from final verification (874 files
/19.3 MiB) were cleared via Windows Recycle Bin calls; retained native
comparison and 4K crash evidence were not touched. Earlier V19 native
pilot receipts support 24 *modeled* cost scenarios, not complete
long-form measured exports. The vendor's actual 4K OpenCV crash is
still unqualified; this cleanup does not fix that underlying crash.

## V18: actual local Topaz model comparisons and startup-aware quality selection

Actual current-workspace models **available offline**: prob-4 (Proteus)
and ahq-12 (Artemis). Iris/Rhea/etc. descriptors WITHOUT downloaded
weights do not qualify as runnable assets; no models were installed
or downloaded by this pass.

The existing model selector now evaluates the **real working-resolution**
render, rather than approving two downscaled images which may hide
checkerboard hallucinations, lost texture or caption damage. Native
detail energy, protected glyph/background pixels, small local regions
and model-difference temporal instability all have source-bound rejection
checks. Two full native 8-frame Snowfall pilots passed this stronger
gate: 1280x720 Proteus 10.543s vs Artemis 8.524s, and 640x800
Proteus 9.399s vs Artemis 10.833s. There is no universal winner.

Cost prediction now uses the actual time of the first emitted native
frame separately from subsequent frame throughput, rather than
extrapolating model cold startup on every frame. These measurements
are pulled from the CURRENT pilot subprocess; a cached/stale timing
entry cannot substitute. A source with <10s active enhancement
no longer spends up to 20s running two extra models. The parent
still uses the same Topaz worker, GPU lock, masks and delivery output.

Native full public Smart Finish export on a retained 4.004s
Snowfall-derived video: **18.078s complete**, actual prob-4 Topaz
working enhance + final encode, all 96 frames and 192192 stereo
audio samples, full decode and parent acceptance PASS, extra model
comparison skipped as intended. A prior same-source export was
29.064s under DIFFERENT load; not a controlled whole-job speedup.
Source-relative pixels and temporal residuals matched closely
across both 96-frame outputs.

Current source-guarded verification after one obsolete pilot fixture
was corrected: **500 workspace tests +15 subtests**, **111 root
tests +12 subtests**, scoped Ruff/diff PASS. Full receipts and exact
limits: Spiffy Workspace/.test_tmp/smart_finish_first_20261113/
native_model_v18b_final_checks.json and docs/FINAL_DELIVERY_PIPELINE.md.
An additional actual two-frame 3840x2160 model render was DEFERRED
before execution when GPU utilization jumped above the bounded
safe-load threshold. Do not claim positive-detail 4K GPU model
acceptance, physical Mac or subjective image perfection.

## V16 canonical multi-shot scene reliability and full-frame quality guards

The existing measured V15 visual-risk selector now also benefits from
two V16 image-acceptance safeguards: a thumbnail-scale catastrophic
frame-damage screen on ALL previously-unsampled enhancement frames,
and a source-glyph-only screen on unsampled interpolated caption frames.
They preserve authored flashes and photographic motion; neither
pretends to establish subjective output quality. A historical genuine
four-second Snowfall-derived Topaz output passed all 96-frame current
checks in ~0.545s for the damage assessment, with source/output unchanged.

Shared scene V2 could find real cuts with FFmpeg but lose them during
OpenCV/MOV frame scoring or snapping. The incumbent canonical reframe
owner now uses bounded PyAV BGR decoding **only for those same candidate
frames** when OpenCV cannot decode. No second scene detector or new
model was added. It fails instead of silently merging unknown shots
when neither decoder can verify any candidate. PNG software codecs
no longer take a doomed CUDA decoder startup; H.264 keeps its previous
backend. Recovered 4/4 real 48-frame synthetic 16:9/9:16/4:5/square
scene sources, and a PNG vs H.264 29.97fps codec pair had identical
three-cut boundaries. Real canonical handoff into Smart Finish masks,
per-shot treatment/report and exact frame PTS passed on all four.

The scene cache changes identity only for affected V2/hybrid scorer
revisions, preserving unrelated V1 caches. Smart Finish also binds its
plan identity to scene_engine, reframe_tracker and now fast_paths
source hashes. Last full **pre-final-identity** proof:
scene_cache_v16_final_checks.json, 490 workspace+15 subtests,
111 root+12 subtests, scoped lint/syntax/diff all PASS. Its new
fast_paths identity test still needs a Windows rerun; don't relabel
the previous count as proof of that last source edit. See
docs/FINAL_DELIVERY_PIPELINE.md and the existing task56 receipts.
Native 4K visual/A-B, long mattes and physical Mac remain open.

## V15 automatic visual-risk routing and clean comparison custody

The production Smart Finish analysis now uses measured source detail,
text/edge evidence, caption frequency, 16-region picture assessment and
**sustained** shot-local motion. It safeguards existing motion graphics
and hard-edged authored text, photographic soft background, strong grain,
and high-motion footage with source-preserving Topaz controls rather
than claiming to recognize a podcast, basketball or actual scene
semantics. Portrait, square, landscape and 4:5 motion thumbnails
preserve their display aspect. A real two-shot encoded test caught and
fixed moving patterned action being treated as graphics. Sparse/
malformed/nonfinite evidence remains safe. Current per-shot choices
enter the existing normal render, interpolation candidate, optional
subject-model admission, checkpoints and delivery report; no second
video application or cloud/VLM is created.

The current crop bbox uses occupied row/column reductions instead of
huge per-pixel index arrays. All 56 source-vs-prior crop geometries
match exactly. Dense synthetic 4K crop CPU 36–39ms -> ~4.5ms;
tracemalloc temporary allocation ~39.8MB -> 0.03MB for this function,
not whole-export or GPU VRAM performance.

Current full source verification: **452 workspace tests + 15
subtests**, **111 root +12 subtests**, source hashes, Ruff and diff
PASS (visual_adaptive_v15_final_checks.json). Three-frame source
preservation 3840x3840 working -> 480x480 final via normal public
command passed in 6.218s with 6000 valid stereo samples; no Topaz
AI occurred on the flat test. Retained real Snowfall eight-frame
analysis selected balanced_photographic, retained all mask frames,
source PTS and report values, with fresh CPU analysis0.8686s and
cached0.1359s. These are not native 4K perceptual quality tests.

Baseline-to-current proof is indexed with 19 SHA-256 anchors in
baseline_to_current_reference_v15.json. The successful reversible
cleanup moved 20 obsolete test directories (5,143 files; 82.45MiB)
to Windows Recycle Bin, retaining the original reference outputs.
A later follow-up batch was partially audited; its separate receipt
distinguishes three recycle-bin-confirmed folders from five missing
but destination-unverified, never assuming a permanent deletion or
verified Recycle Bin placement. See FINAL_DELIVERY_PIPELINE.md for
exact scopes and remaining native Topaz/Mac/semantic/HDR gates.

## V14 dense-caption safety and report integration

Previously an overloaded caption detector could return NO caption mask
after exceeding a contour/grouping limit, exposing moving text to Topaz
and frame interpolation. The existing video worker now conservatively
protects actual high-contrast source edges with a bounded halo when
there are too many candidate shapes, more than 10000 raw contours or
ambiguous excessive glyph coverage. It still never recognizes/retypes
captions. Normal source masks remain exactly as before; ambiguous
regions may receive less AI alteration in exchange for preserving the
original authored appearance.

Caption evidence now includes conservative_fallback_frames and
conservative_fallback_reasons inside the SAME source-bound analysis plan;
the existing measured delivery report consumes that field. A connected
three-frame moving-caption test preserved lossless caption mask PTS,
source pixels and recorded the fallback reasons. Current source-hash-
stable acceptance: 374 workspace tests+15 subtests, 111 root+12
subtests, Ruff and diff checks PASS. Exact receipt:
Spiffy Workspace/.test_tmp/smart_finish_first_20261113/
caption_overflow_v14_final_checks.json. This is CPU/source pipeline
verification, not a new native Topaz 4K quality benchmark. Mac,
HDR/VFR and long soft-alpha video mattes remain open.

## V12 exact large-frame repeat checks and optimized caption row matching

The current automatic detector avoids a complete 4K RGB equality scan
when fixed source-pixel probes already demonstrate that frames differ.
Sparse probes NEVER certify reuse: an exact cache hit still requires
full OpenCV RGB pixel comparison. No skipped frames, rounded timing,
weakened color identity or cross-cut reuse. On a retained real moving
frame resized to 3072x3840 the equality operation measured 3.465ms
old versus 0.086ms new, NOT whole-video time. Exact holds were slightly
slower but remained exact.

Caption protection now indexes contour baselines and restores original
box order, without changing OCR semantics (there is no OCR in this path),
original glyph pixels or mask thresholds. A 4K resized authored frame
measured 36.9ms previous versus 32.0ms indexed, while some smaller source
frames were slightly slower. Keep both results and do not claim a
universal speedup.

Current source-hash-stable V12 acceptance: 370 workspace tests+15 subtests,
111 root+12 subtests, Ruff/diff all passed. The existing task56 proof root
contains caption_v12_final_checks.json, caption_rows_v12_cpu_receipt.json
and repeat_v12_cpu_receipt.json. No new 4K Topaz AI inference was started.
Native positive-detail 4K image quality, Mac/HDR/VFR and long-shot
matting remain unqualified.

## V11 update — actual 4K mask composition and square source-size delivery

Current guarded tests: 361 workspace +15 subtests, 111 root +12 subtests,
scoped Ruff and diff PASS. Real standard-FFmpeg masked rendering at
3072x3840 preserved exact caption/background pixels, changed admitted
subject pixels, and kept both source frame timestamps. Only the Topaz AI
filter was replaced with a deliberately wrong CPU effect for verification;
this does NOT qualify native 4K Topaz appearance.

The previously RAM-deferred square case also passed the public exporter:
480x480 -> **3840x3840 working** -> 480x480 delivered, three frames24fps,
6000 stereo samples, full decode and parent validation, source untouched.
Together with earlier 16:9, 9:16 and 4:5 examples, 1:1 completes common
shape CPU source-preservation delivery checks. Solid-color inputs correctly
skip AI. A proposed shared-grayscale CPU optimization was REVERTED after
matched moving-footage timing showed no gain. See V11 in the existing
Final Delivery documentation and task56 proof root. Native positive-detail
4K Topaz A/B, true soft-matting and physical Mac remain open.

## Current Smart Finish v10 — measured detection and 4K working geometry

The existing Final Delivery worker now uses 16 regional tiles from
aspect-preserving thumbnails of frames it already decodes for protection.
Each real detected shot records its local detail/protection evidence, useful
regions, and a source-bound representative frame. Eligible optional
Proteus/Artemis model comparisons use that evidence instead of always sampling
the highest-motion frame. The original full-resolution masks alone control
which photographic pixels can be changed. This is NOT a semantic face detector,
new VLM, new media decoder, or perceptual-quality truth.
These sixteen tiles are computed transiently. At most four per-shot
spatial samples persist, each with compact scalars and its single top
region, rather than sixteen copied tile records. Full-resolution masks
and timestamps remain unchanged; larger montages use less plan storage.

Predominantly soft/protected shots receive less aggressive Proteus detail
correction and a greater source contribution; unknown evidence leaves the
prior settings unchanged. A model audition requires complete shot coverage
and does not start a second high-memory model trial above 1920x1080 working
size without a qualifying cost/quality test. Ordinary 4K Topaz enhancement
is NOT capped by the optional trial restriction.

Exact uniform-color RGB cards skip full-frame Gaussian/Sobel/contour/glyph
analysis. All three channels must be proven constant; a single changed
pixel forces the normal detector. A small 8x8 convolution reproduces the old
float32 metrics without full-size intermediate arrays. Reference-matched
flat 4K CPU masks and metrics pass, with roughly 0.38-0.51s old versus
0.02-0.05s new detail-plus-caption CPU analysis in separate benchmark
trials. This is a component-level measurement, not a whole-video speedup.

Three actual standard-delivery source-preservation outputs verified the
4K working pipeline at 3840x2160 (16:9), 2160x3840 (9:16), and 3072x3840
(4:5), all delivered at the intended original-size aspect ratio, with
three exact frames and 6,000 valid audio samples, full decode and parent
validation. Those solid-color inputs correctly bypassed AI; they do not
prove native 4K enhancement appearance. Current GPU contention prevented
safe heavy model tests. See the V10 checkpoint in FINAL_DELIVERY_PIPELINE.md.
The separate square 3840x3840 working export was deferred without
rendering because free RAM fell below the existing 6-GiB admission
threshold. Its shape and crop tests pass, but not its complete export.
The latest source-hash-stable validation passed 361 workspace tests
+15 subtests, 111 root tests +12 subtests, scoped Ruff and diff checks.
See spatial_v10_guarded_current.json in the existing task56 proof root;
it is not evidence of a completed 4K Topaz visual-quality or Mac render.

## Measured run reports and bounded subject refinement

Normal Final Delivery jobs now write `delivery_report.html` in their existing
job directory. The result and CLI include `performance_report_path`; the report
shows total job time, source duration, seconds per video second, stage costs,
startup (already included), source/output geometry, per-shot decisions and
recorded resource snapshots. Missing measurements are unknown, not zero. The
comparison gallery consumes the same projection, so reporting code is not copied
into a separate workflow. Old comparison media retains its true render version.

Adaptive natural enhancement now has `subject_refinement: "auto"` by default
(or explicit `"off"`; CLI `--subject-refinement auto|off`). In admitted bounded
shots it calls the existing source-bound person localizer and semantic-roto
owner, rather than implementing a second detector. Accepted subject masks narrow
enhancement inside the subject while source edges, captions and motion blur stay
protected. Binary segmentation with inward feathering is not soft-alpha matting.
Current budgets are two fresh complete shot windows, the existing four-second /
120-frame limit, and resource/time admission. Unavailable, busy, ambiguous or
unqualified cases retain natural processing and record why. The retained V7
eight-frame test includes actual accepted SAM2 masks and a complete subject-aware
Topaz export; arbitrary long shots, soft-alpha matting and general appearance/
cost acceptance are not established by that small native proof.

The report and CPU natural processing are portable. On Mac, `TOPAZ_VIDEO_DIR`
may point to the installed `.app` bundle or its `Contents/MacOS` folder and the
existing explicit model-directory override remains supported. Discovery was
platform-contract tested on Windows, not physically exercised on a Mac. The
current canonical semantic specialist is CUDA-only and defers on Mac; no MPS
or Metal substitute is silently enabled. Use a genuinely installed compatible
local Topaz runtime; this change installs nothing and uses no paid API.

Short fractional-rate exports use an exact shared movie-header clock to preserve
valid audio sample endpoints in AAC/MP4. This repairs real 23.976/29.97/59.94
tests without padding audio or weakening the original sample validation.

## Automatic Smart Finish — everyday entry

The V8 CPU analysis path retains every source frame but reuses raw protection
analysis on exactly identical consecutive RGB frames within a shot. Caption
changes, even a one-channel pixel difference, and shot cuts invalidate that
reuse. Temporal protection still runs at every timestamp. In-place float32
scratch processing reduces temporary allocations without changing the mask
arithmetic. Exact real-frame comparisons now measure 37.8–44.9% less CPU time in
the detail-mask function and about 61.5% fewer peak tracemalloc-tracked allocations.
These are function-level measurements, not whole-export or process-memory claims.
Recovery of the interrupted pass also fixed a missing cv2 import in prepare().
See the V9 checkpoint in FINAL_DELIVERY_PIPELINE.md for the native tests and the
precise last-test-versus-final-guard verification boundary.

## Bounded enhancement-model selection

Adaptive natural mode now supports `model_search: "auto"` (default) or `"off"`;
the existing CLI forwards `--enhancement-model-search auto|off` through the same
Final Delivery request. Manual enhancement recipes are not silently replaced.

The existing local Proteus `prob-4` is the reference. Installed Artemis HQ
`ahq-12` can be compared on short moving witnesses using the SAME source,
protection masks, actual executed crop/recipe and working size. A descriptor
without local weights is not readiness. Production trials use `download=0`.
No new model weights, VLM/contact-sheet service, queue or persistent process is
required. Existing source/shot measurements nominate the candidate; the actual
moving outputs must pass local detail/texture, protected-source and temporal
agreement checks before timing is considered.

Initial admission scope: at most two photographic shots, every such shot
represented by eight frames, both model passes charged against a 20% active-frame
budget, source-size-or-smaller aspect-preserving delivery, safe current resources,
and bounded total/native deadlines. Damaged footage, many-shot montages, enlarged
final masters, unsupported platforms and short clips retain Proteus. A candidate
must be at least 15% faster in the measured pilot and clear a conservative
estimated payback margin. That estimate is NOT a measured whole-export speedup.
Mac model auditions remain unqualified and use the reference, not a guessed
Metal/MPS implementation.

Accepted decisions are bound to source, analyzed masks, model assets, executable,
code identity and delivery geometry in the existing job cache. Candidate controls
do not inherit Proteus sliders. The chosen model and its asset identity reach
native rendering, checkpoints, quality validation and diagnostic receipts. Failed
optional trials keep the reference; cancellation and integrity drift still fail.
The existing full-output damage/audio/timing gates are not replaced by a pilot.

Native eight-frame comparison: Proteus 22.303 s; Artemis 17.333 s, source/local/
temporal checks passed. The selector correctly refused to charge that comparison
to an eight-frame production job because its cost would not be repaid. Separate
four-second public CLI export passed in 29.064 s; rising GPU load deferred optional
trials and the ordinary Proteus route completed. Neither is a general 4K speed
claim. The final two source/cache-integrity guards have pending Windows rerun
coverage after the connector terminal queue filled; see the exact checkpoint.

Use **Smart Finish — Automatic Upscale** (`media.smart_finish`) in the existing
Command Palette with one or more finished videos selected. It uses the existing
selection, progress, cancellation and Final Delivery workers; there is no new
application or resident renderer. The command defaults to an aspect-safe 4K
working enhancement, 60 FPS, original display-size delivery and watermark off.
The automatic route writes collision-safe `__smart_finish_master.mp4` files under
`_SpiffyDelivery` next to each source. Original media is unchanged. Compiled
registration and worker execution do not by themselves prove that an older live
palette has reloaded or certify a physical hotkey interaction.

Equivalent existing command-line entry from the SS Scripts root:

```powershell
python "Spiffy Workspace/tools/run_spiffy_final_delivery.py" --input "C:/renders/finished.mp4" --delivery-set smart_finish --adaptive-enhancement --output-fps 60 --working-long-edge 3840 --execute
```

For AutoExport, use `final_delivery.delivery_set: "smart_finish"` and
`final_delivery.adaptive_enhancement: {"mode":"natural"}` in the existing
request. `frame_rate` and typed working/final `geometry` remain independent.
The automatic engine currently admits bounded zero-origin, exact-CFR, SDR,
square-pixel, unrotated finished footage. Unsupported inputs fail with a stated
reason rather than being silently retimed or tone-mapped. Separate legacy,
typed-geometry and source-timing paths retain their existing contracts.

Natural enhancement measures actual frames, reuses existing scene detection,
chooses restrained Proteus controls, and preserves a conventionally resized source
base. It skips unhelpful enhancement and admits stable contextual crops through
padding, area and pixel-phase guards. Integer and aligned rational scales are
supported. Smaller model input does not guarantee faster total rendering on a
short clip; startup and compositing can dominate. Feathered contribution masks preserve blur,
flat backgrounds and hard edges. These are measured heuristics, not perfect
semantic rotoscoping or a claim that every shot contains recoverable detail.

Visible aligned glyph groups receive an additional source-bound mask through
frame interpolation. Original lettering, outline/shadow context and authored
sample timing are preserved; no OCR, font substitution or caption rewriting
occurs. Photographic motion outside protected regions is genuinely interpolated.
Known cut boundaries use short source holds. Intentional repeated frames are not
treated as missing frames. Post-enhancement and post-interpolation damage checks
run before final publication.

Interpolation runs at final necessary pixel dimensions by default, after the
high-resolution enhancement, instead of synthesizing unused 4K frames that will
immediately be discarded. `interpolation_resolution: "working"` retains the
working-size alternative. Measured sample wins do not establish indistinguishable
results for arbitrary effects or universal rendering-speed gains.

Eligible clips at least four seconds long (20–30 FPS source, up to 60 FPS
delivery, at most 1920×1080 pixel area) may compare installed Apollo Fast against
incumbent Apollo on at most two eight-source-frame motion samples. Both pilot
renders together must fit within 20% of the clip's source-frame workload. Local,
temporal and whole-image agreement plus a measured speed win are all required.
A short job, missing candidate weights, failed comparison, higher output rate or
larger delivery keeps Apollo. Decisions are source/model-bound and cached within
the job. Normal renders never download models or use cloud/API rendering. The
full selected output must still pass caption/cut/timing/audio gates; a successful
pilot alone cannot certify the export.

The current four-second portrait test completed in 183.184 seconds versus the
retained 238.484-second run, with a byte-identical delivered MP4 after avoiding
unnecessary calibration. This is one measured example, not a universal speedup.
The same revision's native fractional crop passed image/timing checks but was
slightly slower than full-frame processing on a one-second controlled fixture;
that negative timing result is retained in the component checkpoint.

Read the native automatic-route checkpoint in
`Spiffy Workspace/docs/FINAL_DELIVERY_PIPELINE.md` for actual outputs, elapsed
times and limitations. Historical sections below describe older direct-worker
presets, not a universal default for Smart Finish.

Called by the AutoHotkey command palette with one or more selected video files.

The worker always uses Topaz Video's bundled `ffmpeg.exe` for `tvai_up` / `tvai_fi` steps and standard FFmpeg for final H.264 MP4 encoding. It writes ProRes 422 HQ intermediates into the job `temp` folder, verifies them with ffprobe, and remuxes before final compression if needed.

Outputs are written to `_UpscaledVideo` next to each source file. Source files are never modified or deleted.

The shared post-export owner at
`Spiffy Workspace\workflow\final_delivery_pipeline.py` can instead pass a
`spiffy_final_delivery_plan_v2` file through `--delivery-plan`. In that mode,
the worker still performs one Topaz enhancement pass (plus interpolation when
the selected timing policy requests CFR), then fans
the verified ProRes intermediate into only the requested physical geometries.
Multiple platform profiles with the same frame share one file. The supported
layouts preserve source aspect: native, black-fit, or explicit center crop;
stretching is not supported.

The default `SS 4K 60FPS` route uses an aspect-safe 4K Topaz working
intermediate, performs real Topaz interpolation when the source is below 60
FPS, and retains that exact frame through VideoSeal and the final delivery. For
a `1080x1920` portrait source, the Topaz stage and retained MP4 are both
`2160x3840` at 60 FPS. The worker refuses to imitate interpolation with
duplicated FFmpeg frames. The separate `SS 4K` preset keeps the same retained-4K
behavior without forcing 60 FPS interpolation. An explicit shared delivery plan
can still request source-size or platform-size derivatives.

The retained file is an H.264 High-compatible MP4 using CRF 17, the `slow`
preset, `yuv420p`, AAC 192 kbps, and fast-start metadata. ProRes intermediates
are deleted after a successful normal run; `--debug` is the explicit option to
keep them. Originals are never modified or removed.

Progress-file replacement retries transient Windows sharing violations. If a
status file remains locked, the worker records `progress_write_errors.log` and
continues the media process; notification UI contention must never abort Topaz
or FFmpeg.

## Invisible Provenance

The legacy direct-worker route embeds a private 256-bit VideoSeal payload by default.
Automatic Smart Finish explicitly uses watermark off; explicit-cadence
VideoSeal/runtime combinations remain separately guarded.
The watermark is inserted from the verified ProRes intermediate while the one
final H.264 file is encoded, so the workflow does not compress an MP4 and then
compress it again just to add provenance. Post-encode sparse detection must
recover at least 90% of the bits and the pre-encode watermark PSNR must remain
at least 35 dB. A failed gate leaves the source and any prior destination
untouched. Recovery failure gets one bounded retry at a maximum `0.3` strength;
the visibility gate still applies, and successful clips do not pay that cost.

Each protected output has a temporary signed receipt under
`_jobs/<job_id>/temp/videoseal_receipts/`. The receipt contains identities,
model/version data, quality metrics, and an HMAC signature; it is verification
evidence, not restart state, and it never stores the raw watermark bits or
private key. Receipts are retained for three days, then pruned when another
upscale starts or normal Job Cleanup removes old job temp data. The delivered
video folder therefore contains only the requested media. The key lives outside
Git under the existing external model root at
`models/_private/videoseal.key`. Keep that key backed up: it is required to
prove which private payload belongs to an export.

The payload capacity is binary, not a word limit: 256 bits equals 32 bytes.
That could hold at most 32 plain ASCII characters, but this workflow correctly
uses the whole payload as a private per-asset cryptographic identifier rather
than embedding readable words. Human-readable ownership and source details
belong in the signed receipt. A website demo limiting text to six letters is a
demo-interface policy, not this model's capacity.

Final-delivery plans may assign a brand namespace such as `spiffy.scenes`,
`spiffy.clips`, `spiffy.luna`, `spiffy.mma`, or `spiffy.chatgpt`. The namespace
changes the private payload deterministically; it does not store readable words
inside the 256 bits. Platform profile IDs remain separate receipt metadata.

For the default delivery route, the verified 4K frame is passed directly to
VideoSeal embedding and the one H.264 encode. Explicit delivery-plan variants
are resized to their requested frame before embedding, still avoiding an extra
compression pass.

VideoSeal source and weights live under the existing external model-root
junction. No training packages, watcher, or watermark database are installed.

```powershell
Set-Location "$env:USERPROFILE\Documents\AutoHotkey\SS Scripts\Spiffy Workspace"
python -m workflow.video_provenance_watermark doctor --load-model
python -m workflow.video_provenance_watermark install-assets
python -m workflow.video_provenance_watermark verify --input "C:\path\clip.mp4" --receipt "C:\path\_jobs\job_id\temp\videoseal_receipts\clip.mp4.hash.videoseal.json"
```

Use `--watermark off` on `video_upscale.py` only when an unmarked export is
intentional. Strength defaults to the upstream-recommended `0.2`; increasing it
improves recovery at the cost of visibility. HDR exports fail closed because
this RGB inference route is not yet color-managed for PQ/HLG. CUDA is the
production path; CPU and Apple MPS are supported by the loader but will be
slower and require their own real-media qualification.

On the RTX 3080 proof machine, one real 4K60 second took `20.97s` from a cold
process and `17.58s` with the model already loaded. This is not real-time; the
worker keeps one model instance across a multi-file job and uses memory-bounded
adaptive chunks. A forced 16-frame 4K chunk was slower, so it is not enabled.

The explicit source-size derivative contract was proven through the complete worker
on a real Snowfall S2E5 derivative. A `1080x1350/30` input ran through a
`3072x3840` Topaz stage and Topaz 60 FPS interpolation, then produced a
`1080x1350/60` H.264 High MP4. Full decode passed, VideoSeal recovered `100%`
of its bits, raw watermark PSNR was `45.79 dB`, and the temporary ProRes files
were removed after success. This historical proof no longer describes the
default Auto output size; it remains evidence for an explicit source-size
delivery-plan variant. Constraining Proteus's rounded 4x result to the exact
working frame reduced this 0.5-second proof from `98.7s` to `70.8s` without
weakening any output gate.

VideoSeal is a resilient forensic marker, not DRM and not a public ownership
signature. A signed C2PA delivery manifest can complement it later; it should
not replace this pixel-level proof.

Visible launcher presets are currently intentionally limited to:

- `SS 4K 60FPS`: points to `C:\ProgramData\Topaz Labs LLC\Topaz Video\presets\ss 4k 60fps.json`.
- `SS 4K`: points to `C:\ProgramData\Topaz Labs LLC\Topaz Video\presets\ss 4k.json`.

The JSON files under the repository's `presets\video` folder are launcher
metadata and pipeline mappings. The Topaz GUI preset files under ProgramData are
the actual presets saved by Topaz.

The worker sets `TVAI_MODEL_DIR` and `TVAI_MODEL_DATA_DIR` before launching
Topaz's bundled FFmpeg. Both values come only from the existing model directory
returned by shared Topaz discovery, and the data-directory value uses the
active host's path separator. If discovery cannot find that directory, the job
fails preflight with an actionable error; it never substitutes the Windows
ProgramData path on macOS. Topaz's GUI sets these variables internally;
without them, direct `tvai_up` / `tvai_fi` calls can report `Model not found`
even when model files exist.

The worker also runs a bounded one-frame model smoke test while holding the
shared GPU lock. Smoke probes disable Topaz model downloads so preflight cannot
silently mutate the model asset set. Legacy enhancement smoke gets 30 seconds;
interpolation gets 90 seconds. Adaptive actual renders use bounded first-frame
and stall monitoring rather than a redundant cold model smoke before the real
stage. A timeout or any
nonzero Topaz result fails preflight before a production render is launched.
This keeps a broken or incompatible model lane recoverable through the normal
job retry path instead of allowing the same native filter failure to occur
during an expensive source render. The worker does not silently substitute a
different enhancement model, because that would change the selected preset's
image character and invalidate its qualification.

Topaz 1.6.0 can also terminate with Windows fail-fast `0xC0000409` immediately
after creating a first-use TensorRT engine for a model block. Production Topaz
stages retry that exact command once for only that native exit code. This does
not change the model, parameters, target, device, or checkpoint identity; a
persistent second crash still fails closed through the normal job retry path.

One additional recovery is intentionally narrow. On the qualified Windows
machine, persistent multi-frame TensorRT failures were reproduced for a 256x256
working frame while the same Proteus v4 and Apollo v8 model/parameters completed
on CPU within the resource ceiling. For working frames no larger than 65,536
pixels, a persistent native GPU fail-fast can therefore retry once on Topaz's
CPU backend with model downloads disabled. Larger frames still fail closed. A
CPU-recovered stage is never retained as the default-GPU Topaz checkpoint, so
future reuse cannot silently treat the two backends as byte-equivalent.

## Safe Retry Checkpoints

The normal Retry action creates a new job but passes the failed job as explicit
lineage. The worker will reuse an old ProRes stage only when the checkpoint
matches the source or prior-stage first/middle/last identity, Topaz executable,
selected model descriptor and downloaded model assets (never mutable JIT
cache), preset
parameters, target frame/FPS, and the unchanged output identity. A hardlink is
used when possible; otherwise a sparse-identity-verified copy is created in the
new job. The failed job's evidence is never modified, and unrelated jobs
cannot supply checkpoints.

Each Topaz stage recomputes its complete source/runtime/parameters/model-assets
contract after rendering. The before and after contracts must be identical;
even a first-run model download requires one clean rerun after the assets
settle. A final output becomes reusable only when its working checkpoint was
actually retained and immediately revalidated. Stable but undiscoverable model
assets can still produce a reviewed output, but no checkpoint or final-output
reuse receipt is recorded.

Batch retry selection keeps every failed-job source. The worker carries a
completed source without rendering only when the exact scheduled source,
source-completion schema, every unique delivery receipt/output identity, and
completion-contract variant still match. It rebuilds profile routing from the
validated contracts and records the carried sources, mappings, completion rows,
and outputs in the new manifest. Changed, missing, ambiguous, legacy, or
incomplete evidence stays pending. A corrupt prior delivery is preserved and
rerendered to a collision-safe filename; unrelated existing files still fail
closed. Retry prefers the failed normalized/render-final source when a Premiere
temp vanished and snapshots the embedded delivery plan inside the new job so
later edits cannot change retry semantics. The original global watermark
default is replayed separately from the aggregate "any variant needs
provenance" preflight. A partially completed multi-profile source reuses a final
file only when its retained working checkpoint, delivery settings, path, and
sampled output identity match the failed job's atomic completion row.

Audio is fail-closed for every track. The first Topaz/ProRes pass maps all
source audio directly to lossless 24-bit PCM, avoiding a second expensive
Topaz render after a container-copy failure. Probe, remux, Topaz,
interpolation, and final-delivery gates require the stream count to remain
unchanged.

Retained-4K final/source completion receipts use schema v2. Older source-size
completion rows are intentionally not carried into a retry after the default
policy change. A matching Topaz checkpoint from an explicitly failed or
cancelled job can still be reused because it already represents the same
verified 4K working frame; the new final file is then encoded, watermarked, and
dimension-validated under the v2 contract. Existing outputs are never
overwritten: a fresh run allocates a collision-safe filename.

## Per-job enhancement recipes (Smart Finish B1)

Use the existing Final Delivery runner with `--enhancement-preset` or a request
whose `final_delivery.enhancement` is, for example,
`{"preset_id":"4k_60fps","overrides":{"noise":-0.1,"blend":0.3}}`.
The runner validates and saves the recipe in `--delivery-plan`; do not inject a
raw filter or edit Python/global presets. The worker revalidates/freeze-resolves
that recipe and uses the same canonical parameters for preflight, actual
enhancement argv, checkpoints and receipts. Both stages and parent/child reuse
are recipe-bound. The existing retry tool forwards the saved recipe snapshot.
Legacy jobs without the object retain their previous behavior/fingerprints.

Only Proteus `prob-4` accepts new numeric overrides: details, compression, blur,
noise, halo and preblur in `[-1,1]`, blend in `[0,1]`. All existing presets remain
selectable; other models reject nonempty overrides. Unknown keys/presets,
bool/string/null controls, NaN/Inf and malformed objects fail before execution.
No downloads or global parameter changes are required.

See `Spiffy Workspace/docs/FINAL_DELIVERY_PIPELINE.md`, section **Smart Finish
B1**, for executable request/CLI examples, exact test commands, failure-first
and final results, proof limits and remaining SF requirements. This is Windows
source-tested parameter plumbing, including real tiny standard-FFmpeg
encode/decode. A later independent one-frame native Topaz recipe diagnostic
exited 0 in 7.610 s after the existing eight-second preflight had timed out;
the production deadline was not changed. See the B1 checkpoint for both
retained outcomes and the open startup-reliability limit. This is not
real-media visual or complete-export qualification. Automatic tuning, custom
working/final geometry and 90/120 FPS remain backlog.

## Explicit Smart Finish source-size master

The existing Final Delivery CLI exposes a separate `smart_finish` delivery set (`smart_finish_master` profile). It runs the existing `4k_60fps` working preset and encodes a distinct final master at the probed source display dimensions. Select it with `--delivery-set smart_finish` or `final_delivery.delivery_set: "smart_finish"` in the existing request. Legacy `native_master` remains unchanged and continues to retain enhanced working dimensions.

The source-display resolver accepts positive rational SAR and orthogonal rotation metadata (0/90/180/270 degrees). It computes the square-pixel display dimensions after rotation, sizes the shared Topaz working stage from that display geometry, and records the original SAR/rotation plus exact even-canvas padding in the completion proof. FFmpeg autorotation is baked into the working intermediate; worker and parent validation require the working/final streams to be SAR 1:1 with zero residual rotation. Invalid/non-positive SAR and non-orthogonal rotation fail closed before Topaz inference. When non-square SAR or 90/270-degree rotation changes the shared working dimensions, `smart_finish_master` must be requested alone; mixed legacy output plans fail closed. Square-pixel sources with 0/180-degree rotation can retain the existing mixed-profile path. CPU FFmpeg geometry regressions do not establish native Topaz quality. The default cadence remains 60 FPS; the explicit cadence contract below is separate from that historical default. Custom working/final dimensions remain open.

## Explicit rational cadence and cancellable model startup

The same Final Delivery CLI accepts `--output-fps 90`, `--output-fps 120`, or
`--output-fps 60000/1001`. Existing JSON requests can set
`final_delivery.frame_rate` to `{"mode":"cfr","fps":"60000/1001"}`.
Absent this field, historical presets, defaults and 60-FPS limits remain intact.
Explicit constant rates are validated between 1 and 120 FPS and serialized as
exact rationals through preflight, actual interpolation, stage/checkpoint
identity, delivery variants and completion/retry bindings. No global preset
mutation, silent 90/120-to-60 clamp or plain-FFmpeg upsampling is used.

The two local master profiles can retain the requested rate through 120 FPS.
Social derivatives retain their existing profile ceilings and explicitly record
`profile_ceiling` when reduced. Requests whose every selected output would lose
the requested rate fail with instructions to include a master. This is a local
master capability, not a claim that every social platform supports 120 FPS.
`--delivery-set smart_finish --output-fps 120` combines the new cadence with the
existing source-size final canvas; `canonical` still retains enhanced geometry.

The explicit-rate route supports CFR input and watermark-off export. Native
Topaz output has passed 90/1, 120/1 and 60000/1001 on a synthetic 64×48 source;
the CPU regression suite covers 30/60/90/120 and fractional rates. Exact output
rates appear per profile in the worker progress receipt. Explicit CFR validates
each audio track's sample rate, channels, start and valid sample extent through
upscale, interpolation and final publication. See
`Spiffy Workspace/docs/FINAL_DELIVERY_PIPELINE.md` for job IDs, output hashes,
frame counts and duration evidence.

Known VFR sources remain unsupported for explicit CFR interpolation unless
their decoded PTS qualify as uniform; the separate source-PTS-preserve mode below
does not interpolate them. Interlaced sources and explicit cadence with VideoSeal
still fail closed. The fractional native fixture's video duration is 0.300300 s
for 18 frames while its source audio remains exactly 0.300000 s; this is the
bounded whole-frame rounding at 60000/1001, not a claim of identical video and
audio duration. Distinct-frame counts and successful decode are cadence/output
evidence, not visual-quality scores.

## Exact source-PTS preservation (Final Delivery B2/SF04)

Use the existing Final Delivery runner with `--preserve-source-timing` or set
`final_delivery.frame_rate` to `{"mode":"preserve_source"}` in the request.
This is not a CFR rate and does not synthesize an FPS. It is currently supported
only by the declared `native_master` and `smart_finish_master` profiles; select
one explicitly with `--profiles native_master` or
`--profiles smart_finish_master`. Social derivatives do not claim VFR support
and are rejected rather than silently resampled. The source's maximum observed
instantaneous cadence must fit the profile's 120 FPS capability.

The worker preserves exact decoded presentation timestamps through the existing
Topaz enhancement and final H.264 stages using `-copyts`, `-fps_mode passthrough`
and a MOV/MP4 track timescale derived from the exact source time base. Topaz FI is
skipped. Audio sample count, rate and channels remain independently validated.
Interlaced inputs, VideoSeal/watermarking and any FPS cap are unsupported for
this policy. Parent completion validation compares the final PTS digest and
extent against the child's saved source probe.

CPU-only verification covers a VFR source with a nonzero PTS origin, final H.264
PTS/sample preservation and full decode, as well as request/plan binding and
parent receipt validation. Native Topaz source-preserve execution is not yet
qualified: the latest GPU admission observation showed 20% utilization and
unattributable compute processes, so no GPU render was started. Use the shared
GPU lock and wait for an attributable idle admission window before native
verification. This timing evidence is not visual interpolation-quality proof.

The existing process runner now checks Cancel/deadlines even during silent
native work, reaps only its own child, and bounds the in-memory diagnostic tail.
Model startup allowances are 30 seconds for enhancement and 90 for interpolation;
the latter probe uses three synthetic input frames/two output frames with the
actual slowmo/duplicate settings. Missing models, nonzero exits and deadline
expiry still fail closed; the existing exact-once native fail-fast retry remains.
No production media, software update, model download, cloud call or new resident
process is required. See `FINAL_DELIVERY_PIPELINE.md` for current proof scope.
