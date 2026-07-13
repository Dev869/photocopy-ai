"""Approximate LR-slider rendering.

Not color-exact — a directional render of the predicted edit (exposure, WB,
contrast, highlights/shadows, vibrance/saturation) applied to an image. Skips the
Look, tone curve, HSL, camera profile, and masks the sidecar carries, so it's a
proof, not a delivery-grade match to Lightroom.

render()          -> small thumbnail (menu-bar previews)
render_raw_jpeg() -> full-resolution proof JPEG for export (quality configurable)
"""
import numpy as np
from PIL import Image


def f(settings, key, scale=1.0):
    try:
        return float(settings.get(key, 0)) * scale
    except (TypeError, ValueError):
        return 0.0


def apply_edit_linear(lin, settings):
    """lin: float32 scene-linear RGB in [0,1+]. Exposure/WB/tone in linear (like
    LR), soft highlight shoulder instead of a hard clip, then sRGB-encode and
    finish with contrast/saturation in gamma space. Returns gamma image [0,1]."""
    lin = lin * (2.0 ** f(settings, "Exposure2012"))

    temp = f(settings, "Temperature")
    if temp:
        d = np.clip((temp - 5500.0) / 5500.0, -1, 1) * 0.35
        lin[..., 0] *= 1 + d
        lin[..., 2] *= 1 - d
    tint = f(settings, "Tint", 1 / 150.0)
    if tint:
        lin[..., 1] *= 1 - np.clip(tint, -1, 1) * 0.25

    lum = lin.mean(axis=-1, keepdims=True)
    sh = f(settings, "Shadows2012", 1 / 100.0)
    if sh:
        lin = lin + sh * 0.35 * np.clip(1 - lum * 3, 0, 1) * (lum + 0.02)
    hi = f(settings, "Highlights2012", 1 / 100.0)
    if hi:
        lin = lin * (1 + hi * 0.4 * np.clip(lum * 2 - 1, 0, 1))
    bl = f(settings, "Blacks2012", 1 / 100.0)
    if bl:
        lin = lin + bl * 0.08

    # LR-ish highlight shoulder: linear through the mids, rolls off above knee
    # instead of clipping (a hard clip blows skies/dresses at wedding-size lifts)
    knee = 0.8
    over = lin > knee
    lin = np.where(over, knee + (1 - knee) * (1 - np.exp(-(lin - knee) / (1 - knee))), lin)

    x = np.clip(lin, 0, 1) ** (1 / 2.2)
    c = f(settings, "Contrast2012", 1 / 100.0)
    if c:
        x = np.clip(0.5 + (x - 0.5) * (1 + 0.6 * c) + 0.4 * c * (x - 0.5) ** 3, 0, 1)

    sat = 1 + f(settings, "Saturation", 1 / 100.0) + 0.6 * f(settings, "Vibrance", 1 / 100.0)
    if sat != 1:
        mean = x.mean(axis=-1, keepdims=True)
        x = np.clip(mean + (x - mean) * sat, 0, 1)
    return x


def apply_edit(x, settings):
    """x: float32 sRGB gamma RGB in [0,1] (8-bit proxies). Degamma first; the
    16-bit linear path (render_raw_jpeg) should be preferred for deep lifts —
    8-bit shadows are quantization-crushed before the exposure gain applies."""
    return apply_edit_linear(x ** 2.2, settings)


def render(proxy_path, settings, out_path, long_side=420):
    img = Image.open(proxy_path).convert("RGB")
    img.thumbnail((long_side, long_side))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = apply_edit(x, settings)
    Image.fromarray((x * 255).astype(np.uint8)).save(out_path, "JPEG", quality=88)


def render_raw_jpeg(src_path, settings, out_path, quality=90, long_side=None):
    """Proof JPEG from the 16-bit linear path. src is a raw (rawpy-decoded,
    camera WB) or a JPEG (used directly). quality is JPEG 1-100; long_side
    optionally downscales the output (e.g. 1024 for VLM audit renders)."""
    if src_path.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(src_path).convert("RGB")
        x = apply_edit(np.asarray(img, dtype=np.float32) / 255.0, settings)
    else:
        import rawpy
        # 16-bit LINEAR decode: exposure must be applied before gamma/quantization
        # or deep underexposures (Devin shoots dark, lifts +1.5..3.5 EV) come out
        # crushed — 8-bit gamma shadows carry only a handful of code values.
        with rawpy.imread(src_path) as r:
            rgb = r.postprocess(use_camera_wb=True, gamma=(1, 1), output_bps=16,
                                no_auto_bright=True)
        # ponytail: fixed baseline exposure (LR applies the camera's DNG
        # BaselineExposure, ~+0.35..0.5 EV on Nikon bodies); read per-camera
        # from EXIF if renders still trend dark across bodies
        lin = rgb.astype(np.float32) / 65535.0 * (2.0 ** 0.45)
        x = apply_edit_linear(lin, settings)
    img = Image.fromarray((x * 255).astype(np.uint8))
    if long_side:
        img.thumbnail((long_side, long_side))
    img.save(out_path, "JPEG", quality=int(quality), subsampling=0 if quality >= 90 else 2)
