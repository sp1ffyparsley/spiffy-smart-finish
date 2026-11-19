# Research protocol — high bar for recommendations

## Evidence required

An external researcher must **read the actual source** and identify which implementation and integration owners already exist before recommending new code. Every significant claim needs an exact file path, a relevant function/class and evidence from source, an upstream paper/repository, or a reproducible benchmark. Include the date of upstream model releases and last meaningful maintenance, not just GitHub stars.

The target machine is a Windows PC with **NVIDIA RTX 3080 / 10 GiB VRAM**. Proposed temporal VSR, face restoration, diffusion and encoding capabilities must have credible local-memory/precision/runtime evidence. Where direct evidence is missing, mark the idea as **unverified**, not compatible by assertion.

Measure separately: model and ComfyUI startup, preflight/decode, actual GPU inference, disk I/O and image transfers, lossless assembly, final encode, final QA, and total wall clock. Compare the **same source, output dimensions, GPU, driver, model setup and quality constraints**. Never claim an end-to-end gain based on a single microbenchmark.

Quality checks should include temporal flicker, warped/repeated details, skin and identity drift, captions and graphics, low light, grain/film texture, chroma/exposure, motion blur, scene transitions, output length, original source frame clock, and audio sample fidelity. Consider per-region quality; global VMAF is insufficient for a tiny face.

## Deliverables for each idea

- **Existing implementation**: what file/function already covers it.
- **Proposed improvement**: what changes and why the duplicate code is unnecessary.
- **Evidence**: repo/paper/weights link and benchmark or reproducible experiment; explicitly label assumptions.
- **Hardware feasibility**: RTX 3080 10 GiB, VRAM/RAM/disk needs, 1080p/4K/6K and FPS constraints.
- **Benefits**: expected *measurable* quality, time, reliability or user-interaction effect, with uncertainty.
- **Risks**: licensing, identity/temporal stability, privacy, corruption, source/publish invariants, security, support cost.
- **Integration**: exact boundary, versioned contract, model lifecycle and fallback; do not duplicate the final-delivery owner.
- **Acceptance**: exact repeatable tests, comparison matrix, stop/rollback criteria and how to identify a false improvement.

Prioritize fewer proven improvements over giant wish lists. Flag unsupported claims. Do not generate new video, change user files, download large model weights, or assume access to private assets during research.
