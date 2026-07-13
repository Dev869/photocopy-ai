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


def apply_edit(x, settings):
    """x: float32 sRGB RGB array in [0,1]. Returns the adjusted array in [0,1]."""
    lin = x ** 2.2
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

    x = np.clip(lin, 0, 1) ** (1 / 2.2)
    c = f(settings, "Contrast2012", 1 / 100.0)
    if c:
        x = np.clip(0.5 + (x - 0.5) * (1 + 0.6 * c) + 0.4 * c * (x - 0.5) ** 3, 0, 1)

    sat = 1 + f(settings, "Saturation", 1 / 100.0) + 0.6 * f(settings, "Vibrance", 1 / 100.0)
    if sat != 1:
        mean = x.mean(axis=-1, keepdims=True)
        x = np.clip(mean + (x - mean) * sat, 0, 1)
    return x


def render(proxy_path, settings, out_path, long_side=420):
    img = Image.open(proxy_path).convert("RGB")
    img.thumbnail((long_side, long_side))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = apply_edit(x, settings)
    Image.fromarray((x * 255).astype(np.uint8)).save(out_path, "JPEG", quality=88)


def render_raw_jpeg(src_path, settings, out_path, quality=90):
    """Full-resolution proof JPEG. src is a raw (rawpy-decoded, camera WB) or a
    JPEG (used directly). quality is JPEG 1-100 (higher = larger, less compressed)."""
    if src_path.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(src_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    else:
        import rawpy
        with rawpy.imread(src_path) as r:
            rgb = r.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=True)
        arr = rgb.astype(np.float32) / 255.0
    x = apply_edit(arr, settings)
    Image.fromarray((x * 255).astype(np.uint8)).save(
        out_path, "JPEG", quality=int(quality), subsampling=0 if quality >= 90 else 2)
