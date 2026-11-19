"""Social platform export profiles for Spiffy delivery checks.

Future-AI note:
This is a delivery contract registry, not a renderer or uploader. Keep platform
rules centralized here and feed existing probe/QA/readiness modules. Do not add
upload APIs, background workers, dependency installs, or a second render preset
system here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

try:
    from .production_contracts_common import issue, status_from_issues
except Exception:  # pragma: no cover - direct script fallback
    from production_contracts_common import issue, status_from_issues  # type: ignore


SCHEMA = "spiffy_social_export_profile_registry_v1"
PROFILE_VALIDATION_SCHEMA = "spiffy_social_export_profile_validation_v1"
POLICY = "delivery_contract_only_no_render_no_upload_no_dependency_install"

PROFILE_SOURCE_REFS: tuple[dict[str, Any], ...] = (
    {
        "name": "YouTube Help: Understand three-minute YouTube Shorts",
        "url": "https://support.google.com/youtube/answer/15424877",
        "checked_at": "2026-06-28",
        "used_for": "Shorts duration and square/vertical classification constraints.",
        "copied_code": False,
    },
    {
        "name": "TikTok Content Posting API: Upload video restrictions",
        "url": "https://developers.tiktok.com/doc/content-posting-api-reference-upload-video",
        "checked_at": "2026-06-28",
        "used_for": "TikTok media format, codec, frame-rate, picture-size, duration, and size limits.",
        "copied_code": False,
    },
    {
        "name": "Instagram Help Center: Reel size and aspect ratios",
        "url": "https://help.instagram.com/1038071743007909",
        "checked_at": "2026-06-28",
        "used_for": "Instagram reel aspect-ratio, frame-rate, and minimum-resolution guidance.",
        "copied_code": False,
        "notes": "The page can redirect to login; keep this source reviewed manually when platform rules matter.",
    },
    {
        "name": "slhck/ffmpeg-quality-metrics",
        "url": "https://github.com/slhck/ffmpeg-quality-metrics",
        "license": "MIT",
        "used_for": "Reference-only idea: keep output quality gates explicit and manifest-backed.",
        "copied_code": False,
    },
    {
        "name": "mifi/editly",
        "url": "https://github.com/mifi/editly",
        "license": "MIT",
        "used_for": "Reference-only idea: declarative social-video target profiles.",
        "copied_code": False,
    },
    {
        "name": "Snapchat Support: Post a Snap to My Story from the Web",
        "url": "https://help.snapchat.com/hc/en-us/articles/7012310003348-How-to-Post-a-Snap-to-My-Story-from-the-Web",
        "checked_at": "2026-07-13",
        "used_for": "Snapchat 9:16 story canvas guidance.",
        "copied_code": False,
    },
    {
        "name": "Meta for Business: Reels ads",
        "url": "https://www.facebook.com/business/ads/facebook-instagram-reels-ads",
        "checked_at": "2026-07-13",
        "used_for": "Facebook and Instagram Reels 9:16 creative guidance.",
        "copied_code": False,
    },
    {
        "name": "X Help: Media Studio Library",
        "url": "https://help.x.com/en/using-x/media-studio-faqs",
        "checked_at": "2026-07-13",
        "used_for": "X H.264/AAC MP4/MOV and up-to-60-FPS delivery guidance.",
        "copied_code": False,
    },
)


def build_social_export_profile_registry() -> dict[str, dict[str, Any]]:
    """Return a copy of the built-in export profile registry."""

    return {key: deepcopy(value) for key, value in _PROFILE_REGISTRY.items()}


def list_social_export_profiles() -> list[dict[str, Any]]:
    """Return built-in profiles sorted by profile id."""

    return [deepcopy(_PROFILE_REGISTRY[key]) for key in sorted(_PROFILE_REGISTRY)]


def get_social_export_profile(profile_id: str) -> dict[str, Any]:
    """Return one export profile or raise KeyError for a bad profile id."""

    key = str(profile_id or "").strip()
    if not key:
        return {}
    if key not in _PROFILE_REGISTRY:
        raise KeyError(f"Unknown social export profile: {key}")
    return deepcopy(_PROFILE_REGISTRY[key])


def output_qa_kwargs_for_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return kwargs that can be passed into package-level output QA."""

    if not profile:
        return {}
    qa = _dict(profile.get("output_qa"))
    return {
        "require_audio": bool(profile.get("require_audio")),
        "expected_width": int(profile.get("expected_width") or 0),
        "expected_height": int(profile.get("expected_height") or 0),
        "expected_fps": float(qa.get("expected_fps") or 0.0),
        "target_lufs": float(qa.get("target_lufs") or -14.0),
        "loudness_tolerance_lufs": float(qa.get("loudness_tolerance_lufs") or 2.0),
        "true_peak_ceiling_db": float(qa.get("true_peak_ceiling_db") or -1.0),
    }


def readiness_kwargs_for_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return readiness defaults derived from a profile."""

    if not profile:
        return {}
    return {
        "require_audio": bool(profile.get("require_audio")),
        "expected_width": int(profile.get("expected_width") or 0),
        "expected_height": int(profile.get("expected_height") or 0),
    }


def validate_probe_against_profile(
    probe: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    strict_preferred_codecs: bool = False,
) -> dict[str, Any]:
    """Validate an ffprobe_media payload against one social export profile."""

    if not profile:
        return {
            "schema": PROFILE_VALIDATION_SCHEMA,
            "status": "SKIP",
            "profile_id": "",
            "checks": {},
            "issues": [],
        }
    issues: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    profile_id = str(profile.get("profile_id") or "")
    if probe.get("status") != "PASS":
        issues.append(issue("social_export_profile_probe_unavailable", "ffprobe data is not available for profile validation.", severity="error", status="FAIL", profile_id=profile_id, probe_status=probe.get("status")))
        return _validation_result(profile, probe, checks, issues)

    checks["has_video"] = bool(probe.get("has_video"))
    if not checks["has_video"]:
        issues.append(issue("social_export_profile_video_missing", "Profile requires a video stream.", severity="error", status="FAIL", profile_id=profile_id))

    expected_width = int(profile.get("expected_width") or 0)
    expected_height = int(profile.get("expected_height") or 0)
    observed_width = int(probe.get("width") or 0)
    observed_height = int(probe.get("height") or 0)
    checks["width_matches"] = True if not expected_width else observed_width == expected_width
    checks["height_matches"] = True if not expected_height else observed_height == expected_height
    if not checks["width_matches"]:
        issues.append(issue("social_export_profile_width_mismatch", "Output width does not match the export profile.", severity="error", status="FAIL", profile_id=profile_id, observed=observed_width, expected=expected_width))
    if not checks["height_matches"]:
        issues.append(issue("social_export_profile_height_mismatch", "Output height does not match the export profile.", severity="error", status="FAIL", profile_id=profile_id, observed=observed_height, expected=expected_height))

    aspect = float(profile.get("aspect_ratio") or 0.0)
    tolerance = float(profile.get("aspect_tolerance") or 0.02)
    observed_aspect = observed_width / observed_height if observed_width and observed_height else 0.0
    checks["aspect_matches"] = True if not aspect else abs(observed_aspect - aspect) <= tolerance
    if not checks["aspect_matches"]:
        issues.append(issue("social_export_profile_aspect_mismatch", "Output aspect ratio does not match the export profile.", severity="error", status="FAIL", profile_id=profile_id, observed=round(observed_aspect, 5), expected=aspect, tolerance=tolerance))

    duration_s = float(probe.get("duration_s") or 0.0)
    min_duration_s = float(profile.get("min_duration_s") or 0.0)
    max_duration_s = float(profile.get("max_duration_s") or 0.0)
    checks["duration_min_ok"] = True if not min_duration_s else duration_s >= min_duration_s
    checks["duration_max_ok"] = True if not max_duration_s else duration_s <= max_duration_s
    if not checks["duration_min_ok"]:
        issues.append(issue("social_export_profile_duration_too_short", "Output duration is shorter than the export profile minimum.", severity="error", status="FAIL", profile_id=profile_id, observed_s=duration_s, minimum_s=min_duration_s))
    if not checks["duration_max_ok"]:
        issues.append(issue("social_export_profile_duration_too_long", "Output duration is longer than the export profile maximum.", severity="error", status="FAIL", profile_id=profile_id, observed_s=duration_s, maximum_s=max_duration_s))

    fps = float(probe.get("fps") or 0.0)
    min_fps = float(profile.get("min_fps") or 0.0)
    max_fps = float(profile.get("max_fps") or 0.0)
    checks["fps_min_ok"] = True if not min_fps else fps >= min_fps - 0.05
    checks["fps_max_ok"] = True if not max_fps else fps <= max_fps + 0.05
    if not checks["fps_min_ok"]:
        issues.append(issue("social_export_profile_fps_too_low", "Output FPS is lower than the export profile minimum.", severity="error", status="FAIL", profile_id=profile_id, observed=fps, minimum=min_fps))
    if not checks["fps_max_ok"]:
        issues.append(issue("social_export_profile_fps_too_high", "Output FPS is higher than the export profile maximum.", severity="error", status="FAIL", profile_id=profile_id, observed=fps, maximum=max_fps))

    checks["audio_present"] = True if not profile.get("require_audio") else bool(probe.get("has_audio"))
    if not checks["audio_present"]:
        issues.append(issue("social_export_profile_audio_missing", "Profile requires an audio stream.", severity="error", status="FAIL", profile_id=profile_id))

    container_family = str(profile.get("container_family") or "").strip().lower()
    format_name = str(probe.get("format_name") or "").lower()
    checks["container_matches"] = True if not container_family else container_family in format_name
    if not checks["container_matches"]:
        issues.append(issue("social_export_profile_container_mismatch", "Output container does not match the export profile.", severity="error", status="FAIL", profile_id=profile_id, observed=format_name, expected=container_family))

    video_codec_ok = _codec_matches(str(probe.get("video_codec") or ""), _list(profile.get("preferred_video_codec_families")))
    audio_codec_ok = True if not probe.get("has_audio") else _codec_matches(str(probe.get("audio_codec") or ""), _list(profile.get("preferred_audio_codec_families")))
    checks["preferred_video_codec"] = video_codec_ok
    checks["preferred_audio_codec"] = audio_codec_ok
    preferred_severity = "error" if strict_preferred_codecs else "warning"
    preferred_status = "FAIL" if strict_preferred_codecs else "WARN"
    if not video_codec_ok:
        issues.append(issue("social_export_profile_video_codec_not_preferred", "Output video codec is not one of the profile-preferred codecs.", severity=preferred_severity, status=preferred_status, profile_id=profile_id, observed=probe.get("video_codec"), preferred=profile.get("preferred_video_codec_families")))
    if not audio_codec_ok:
        issues.append(issue("social_export_profile_audio_codec_not_preferred", "Output audio codec is not one of the profile-preferred codecs.", severity=preferred_severity, status=preferred_status, profile_id=profile_id, observed=probe.get("audio_codec"), preferred=profile.get("preferred_audio_codec_families")))

    max_size_bytes = int(profile.get("max_size_bytes") or 0)
    observed_size = int(probe.get("size") or 0)
    checks["size_max_ok"] = True if not max_size_bytes else observed_size <= max_size_bytes
    if not checks["size_max_ok"]:
        issues.append(issue("social_export_profile_file_too_large", "Output file size is larger than the export profile maximum.", severity="error", status="FAIL", profile_id=profile_id, observed=observed_size, maximum=max_size_bytes))
    return _validation_result(profile, probe, checks, issues)


def _validation_result(profile: Mapping[str, Any], probe: Mapping[str, Any], checks: Mapping[str, bool], issues: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": PROFILE_VALIDATION_SCHEMA,
        "status": status_from_issues(issues),
        "profile_id": str(profile.get("profile_id") or ""),
        "platform": str(profile.get("platform") or ""),
        "label": str(profile.get("label") or ""),
        "probe_path": str(probe.get("path") or ""),
        "checks": dict(checks),
        "issue_count": len(issues),
        "issues": issues,
    }


def _codec_matches(observed: str, families: list[Any]) -> bool:
    allowed: set[str] = set()
    for family in families:
        allowed.update(_CODEC_FAMILIES.get(str(family).strip().lower(), {str(family).strip().lower()}))
    return str(observed or "").strip().lower() in allowed if allowed else True


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


_CODEC_FAMILIES: dict[str, set[str]] = {
    "h264": {"h264", "avc1"},
    "h265": {"hevc", "h265"},
    "aac": {"aac"},
    "vp8": {"vp8"},
    "vp9": {"vp9"},
}

_BASE_VERTICAL_PROFILE: dict[str, Any] = {
    "format_id": "vertical_1080x1920",
    "expected_width": 1080,
    "expected_height": 1920,
    "aspect_ratio": 9 / 16,
    "aspect_label": "9:16",
    "aspect_tolerance": 0.02,
    "container_family": "mp4",
    "preferred_video_codec_families": ["h264"],
    "preferred_audio_codec_families": ["aac"],
    "require_audio": True,
    "min_duration_s": 0.3,
    "max_duration_s": 180.0,
    "min_fps": 0.0,
    "max_fps": 60.0,
    "supports_source_timing": False,
    "max_size_bytes": 0,
    "output_qa": {
        "target_lufs": -14.0,
        "loudness_tolerance_lufs": 2.0,
        "true_peak_ceiling_db": -1.0,
        "expected_fps": 0.0,
    },
    "delivery_render": {
        "layout": "fit_into_frame",
        "background": "black",
        "video_codec": "libx264",
        "audio_codec": "aac",
        "crf": 17,
        "preset": "slow",
        "enhance_preset": "4k_60fps",
        "fps_policy": "topaz_up_to_60",
    },
    "source_refs": list(PROFILE_SOURCE_REFS),
    "rules": {
        "renders_or_muxes": False,
        "publishes_or_uploads": False,
        "starts_background_worker": False,
        "adds_external_dependencies": False,
        "copied_external_code": False,
    },
}


def _profile(**overrides: Any) -> dict[str, Any]:
    payload = deepcopy(_BASE_VERTICAL_PROFILE)
    payload.update(overrides)
    payload["schema"] = SCHEMA
    payload["policy"] = POLICY
    return payload


_PROFILE_REGISTRY: dict[str, dict[str, Any]] = {
    "native_master": _profile(
        profile_id="native_master",
        platform="archive",
        label="Native aspect enhanced master",
        format_id="native_source",
        expected_width=0,
        expected_height=0,
        aspect_ratio=0.0,
        aspect_label="native",
        min_duration_s=0.0,
        max_duration_s=0.0,
        min_fps=0.0,
        max_fps=120.0,
        supports_source_timing=True,
        source_refs=[],
        delivery_render={
            "layout": "keep_source_aspect",
            "background": "black",
            "video_codec": "libx264",
            "audio_codec": "aac",
            "crf": 17,
            "preset": "slow",
            "enhance_preset": "4k_60fps",
            "fps_policy": "topaz_up_to_60",
        },
        notes="Canonical reusable master. It preserves the edit aspect ratio; platform canvases are derived later.",
    ),
    "smart_finish_master": _profile(
        profile_id="smart_finish_master",
        platform="archive",
        label="Smart Finish source-size master",
        format_id="smart_finish_source",
        expected_width=0,
        expected_height=0,
        aspect_ratio=0.0,
        aspect_label="source display",
        min_duration_s=0.0,
        max_duration_s=0.0,
        min_fps=0.0,
        max_fps=120.0,
        supports_source_timing=True,
        source_refs=[],
        delivery_render={
            "layout": "fit_into_frame",
            "background": "black",
            "video_codec": "libx264",
            "audio_codec": "aac",
            "crf": 17,
            "preset": "slow",
            "enhance_preset": "4k_60fps",
            "fps_policy": "topaz_up_to_60",
            "geometry_policy": "source_display",
            "physical_name": "smart_finish_master",
        },
        notes=(
            "Default 4K/60 working enhancement with explicit rational rates up to 120 FPS; final delivery restores the source canvas. "
            "Positive rational SAR and orthogonal 0/90/180/270-degree rotation are normalized to square-pixel display geometry; invalid SAR/non-orthogonal rotation fails closed. "
            "Non-square SAR or 90/270-degree rotation of a non-square frame requires this profile alone; unchanged working geometry retains the mixed-profile path. "
            "The legacy native_master profile retains its enhanced-working geometry."
        ),
    ),
    "spiffy_4x5": _profile(
        profile_id="spiffy_4x5",
        platform="spiffy",
        label="Spiffy 4:5 portrait",
        format_id="portrait_1080x1350",
        expected_width=1080,
        expected_height=1350,
        aspect_ratio=4 / 5,
        aspect_label="4:5",
        max_duration_s=0.0,
        source_refs=[],
        notes="Primary Spiffy Scenes/Clips portrait delivery without stretching.",
    ),
    "generic_vertical_short": _profile(
        profile_id="generic_vertical_short",
        platform="generic",
        label="Generic vertical short",
        notes="Best default for Spiffy manual review packages before a specific platform is chosen.",
    ),
    "youtube_shorts": _profile(
        profile_id="youtube_shorts",
        platform="youtube",
        label="YouTube Shorts vertical",
        max_duration_s=180.0,
        min_fps=0.0,
        source_refs=[PROFILE_SOURCE_REFS[0]],
        notes="Checks the current square/vertical up-to-three-minute Shorts envelope; rights/music claims still require human review.",
    ),
    "instagram_reels": _profile(
        profile_id="instagram_reels",
        platform="instagram",
        label="Instagram Reels vertical",
        max_duration_s=900.0,
        min_fps=30.0,
        source_refs=[PROFILE_SOURCE_REFS[2]],
        notes="Uses public help-center guidance plus Spiffy's internal 1080x1920/AAC/H.264 delivery target.",
    ),
    "facebook_reels": _profile(
        profile_id="facebook_reels",
        platform="facebook",
        label="Facebook Reels vertical",
        max_duration_s=0.0,
        source_refs=[PROFILE_SOURCE_REFS[6]],
        notes="Uses the 9:16 Reels creative canvas; duration is intentionally not hard-coded because Facebook video routing changes independently of rendering.",
    ),
    "snapchat_story": _profile(
        profile_id="snapchat_story",
        platform="snapchat",
        label="Snapchat Story vertical",
        max_duration_s=60.0,
        source_refs=[PROFILE_SOURCE_REFS[5]],
        notes="Uses a 1080x1920 9:16 derivative so Snapchat does not have to make an uncontrolled crop.",
    ),
    "tiktok_vertical": _profile(
        profile_id="tiktok_vertical",
        platform="tiktok",
        label="TikTok vertical upload",
        max_duration_s=600.0,
        min_fps=23.0,
        preferred_video_codec_families=["h264", "h265"],
        max_size_bytes=4 * 1024 * 1024 * 1024,
        source_refs=[PROFILE_SOURCE_REFS[1]],
        notes="Uses TikTok API upload constraints while keeping Spiffy's vertical MP4 delivery target.",
    ),
    "x_portrait_4x5": _profile(
        profile_id="x_portrait_4x5",
        platform="x",
        label="X portrait 4:5",
        format_id="portrait_1080x1350",
        expected_width=1080,
        expected_height=1350,
        aspect_ratio=4 / 5,
        aspect_label="4:5",
        max_duration_s=0.0,
        source_refs=[PROFILE_SOURCE_REFS[7]],
        notes="Spiffy portrait delivery using X Media Studio's H.264/AAC and 60-FPS-compatible envelope.",
    ),
    "youtube_horizontal": _profile(
        profile_id="youtube_horizontal",
        platform="youtube",
        label="YouTube horizontal 1080p",
        format_id="landscape_1920x1080",
        expected_width=1920,
        expected_height=1080,
        aspect_ratio=16 / 9,
        aspect_label="16:9",
        max_duration_s=0.0,
        source_refs=[],
        notes="Native 16:9 delivery. Non-16:9 sources are fitted without stretching unless the operator explicitly chooses crop.",
    ),
    "dailymotion_horizontal": _profile(
        profile_id="dailymotion_horizontal",
        platform="dailymotion",
        label="Dailymotion horizontal 1080p",
        format_id="landscape_1920x1080",
        expected_width=1920,
        expected_height=1080,
        aspect_ratio=16 / 9,
        aspect_label="16:9",
        max_duration_s=0.0,
        source_refs=[],
        notes="Conservative H.264/AAC 1080p delivery contract; upload-account limits remain an uploader concern.",
    ),
}


__all__ = [
    "POLICY",
    "PROFILE_SOURCE_REFS",
    "PROFILE_VALIDATION_SCHEMA",
    "SCHEMA",
    "build_social_export_profile_registry",
    "get_social_export_profile",
    "list_social_export_profiles",
    "output_qa_kwargs_for_profile",
    "readiness_kwargs_for_profile",
    "validate_probe_against_profile",
]
