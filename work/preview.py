"""Approximate LR-slider rendering for preview thumbnails.

Not color-exact — a directional preview of the predicted edit (exposure, WB,
contrast, highlights/shadows, vibrance/saturation) applied to the 640px proxy.
"""
import numpy as np
from PIL import Image


def f(settings, key, scale=1.0):
    try:
        return float(settings.get(key, 0)) * scale
    except (TypeError, ValueError):
        return 0.0


def render(proxy_path, settings, out_path, long_side=420):
    img = Image.open(proxy_path).convert("RGB")
    img.thumbnail((long_side, long_side))
    x = np.asarray(img, dtype=np.float32) / 255.0
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

    Image.fromarray((x * 255).astype(np.uint8)).save(out_path, "JPEG", quality=88)
