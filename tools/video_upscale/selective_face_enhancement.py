"""CPU-only crop/feather/paste for MotionCoach-guided local face enhancement.

The neural model is always the same pinned ComfyUI frame-GAN model as the
rest of Smart Finish; this module only handles pixel geometry and blending.
No new detector, identity recognizer, face swap or generative model is used.
The full original frame stays the geometry/photometry reference. The requested
region is AI-enhanced while the remaining pixels are Lanczos-resized source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


REGION_SCHEMA = "spiffy_selective_face_composite_v1"


def prepare_face_crop(
    source_frame: Path, bbox_xyxy: tuple[int, int, int, int],
    *, source_size: tuple[int, int], output_size: tuple[int, int],
    crop_path: Path,
) -> tuple[int, int, int, int]:
    """Write one bounded source crop for the existing local GAN and map ROI."""
    from PIL import Image

    sw, sh = source_size
    ow, oh = output_size
    if not all(type(v) is int and v >= 64 for v in (sw, sh, ow, oh)):
        raise ValueError("Face ROI requires positive source/target raster dimensions.")
    if not isinstance(bbox_xyxy, (tuple, list)) or len(bbox_xyxy) != 4:
        raise ValueError("Expected a published MotionCoach xyxy face crop.")
    x0, y0, x1, y1 = bbox_xyxy
    if not all(type(v) is int for v in (x0, y0, x1, y1)) or not (
        0 <= x0 < x1 <= sw and 0 <= y0 < y1 <= sh):
        raise ValueError("MotionCoach face crop is outside the original frame.")
    if min(x1 - x0, y1 - y0) < 32:
        raise ValueError("MotionCoach face crop is too small for safe neural enhancement.")
    dx0 = round(x0 * ow / sw)
    dy0 = round(y0 * oh / sh)
    dx1 = round(x1 * ow / sw)
    dy1 = round(y1 * oh / sh)
    dx1 -= (dx1 - dx0) % 2
    dy1 -= (dy1 - dy0) % 2
    if (dx1 > ow or dy1 > oh or min(dx1 - dx0, dy1 - dy0) < 64
            or (dx1 - dx0) * (dy1 - dy0) > 3840 * 2160):
        raise ValueError("Face ROI cannot be upscaled into a supported target region.")
    if not source_frame.is_file() or source_frame.is_symlink() or crop_path.exists():
        raise ValueError("Source face frame is missing, linked, or crop target is already occupied.")
    with Image.open(source_frame) as img:
        if img.size != (sw, sh):
            raise ValueError("MotionCoach face crop source frame has incorrect coded geometry.")
        fragment = img.convert("RGB").crop((x0, y0, x1, y1))
        fragment.save(crop_path, format="PNG")
    return (dx0, dy0, dx1, dy1)


def compose_face_roi(
    source_frame: Path, enhanced_crop: Path, destination: Path,
    rectangle: tuple[int, int, int, int],
    *, output_size: tuple[int, int], source_preservation: float = 0.0,
) -> dict[str, Any]:
    """Paste an actual cropped neural result using a soft source-preserving edge."""
    import math
    from PIL import Image, ImageDraw, ImageFilter, ImageStat

    ow, oh = output_size
    x0, y0, x1, y1 = rectangle
    if (type(ow) is not int or type(oh) is not int
            or not all(type(v) is int for v in rectangle)
            or not (0 <= x0 < x1 <= ow and 0 <= y0 < y1 <= oh)):
        raise ValueError("ROI composite rectangle has invalid geometry.")
    if (type(source_preservation) not in (int, float)
            or not math.isfinite(source_preservation) or not 0 <= source_preservation <= 0.25):
        raise ValueError("Face source-preservation factor is out of range.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Selective frame output already exists.")
    with Image.open(source_frame) as src, Image.open(enhanced_crop) as enhanced:
        background = src.convert("RGB").resize((ow, oh), Image.Resampling.LANCZOS)
        ai = enhanced.convert("RGB")
        if ai.size != (x1 - x0, y1 - y0):
            raise ValueError("Selective face GAN returned an incorrect patch geometry.")
        untouched = background.crop(rectangle)
        # A small facial region can undergo a severe color/exposure shift
        # without affecting the *whole frame* average enough to trigger the
        # incumbent global QA. Keep a second, local source-relative check.
        small_ai = ai.resize((64, 64), Image.Resampling.BILINEAR)
        small_source = untouched.resize((64, 64), Image.Resampling.BILINEAR)
        ai_means = ImageStat.Stat(small_ai).mean
        source_means = ImageStat.Stat(small_source).mean
        if any(abs(a - b) > 40 for a, b in zip(ai_means, source_means)):
            raise RuntimeError("Face-only AI crop changed the source's local color or exposure excessively.")
        if source_preservation:
            ai = Image.blend(ai, untouched, float(source_preservation))
        feather = max(3, min(20, round(min(ai.size) * 0.09)))
        mask = Image.new("L", ai.size, 0)
        drawer = ImageDraw.Draw(mask)
        drawer.rectangle(
            (feather, feather, ai.width - feather - 1, ai.height - feather - 1),
            fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, feather // 2)))
        background.paste(ai, (x0, y0), mask)
        background.save(destination, "PNG")
    return {
        "schema": REGION_SCHEMA, "face_roi": list(rectangle),
        "paste_feather_pixels": feather,
        "source_preservation": source_preservation,
        "background": "source_lanczos_no_neural_reconstruction",
        "foreground": "pinned_comfyui_neural_face_crop",
    }
