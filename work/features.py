"""Condition features for retrieval: proxy luminance percentiles, EXIF EV, capture day.

Weights tuned on holdout 2025-01-04, validated on 2024-12-30 (see exp_features.py).
"""
import datetime, io, json, os, re

import numpy as np
from PIL import Image

EXPORT = os.path.expanduser("~/Pictures/agent-export")
W_LUM, W_EV, W_T = 2.0, 0.2, 0.6
EPOCH = datetime.date(2020, 1, 1).toordinal()


def frac(s):
    if not s:
        return None
    m = re.match(r"(-?\d+)/(\d+)$", s)
    return float(m.group(1)) / float(m.group(2)) if m else float(s)


def ev100(exposure_time, fnumber, iso):
    if not (exposure_time and fnumber and iso):
        return None
    return float(np.log2(fnumber * fnumber / exposure_time) - np.log2(iso / 100.0))


def ev_from_xmp_text(t):
    et = frac((re.search(r'exif:ExposureTime="([^"]*)"', t) or [None, None])[1])
    fn = frac((re.search(r'exif:FNumber="([^"]*)"', t) or [None, None])[1])
    iso_m = re.search(r"ISOSpeedRatings>\s*<rdf:Seq>\s*<rdf:li>(\d+)", t)
    return ev100(et, fn, float(iso_m.group(1)) if iso_m else None)


def ev_from_raw(path):
    """EV from the raw's embedded thumbnail EXIF."""
    import rawpy
    try:
        with rawpy.imread(path) as r:
            th = r.extract_thumb()
        ex = Image.open(io.BytesIO(th.data)).getexif().get_ifd(0x8769)
        return ev100(float(ex.get(0x829A, 0)), float(ex.get(0x829D, 0)),
                     float(ex.get(0x8827, 0)))
    except Exception:
        return None


def lum(proxy_path):
    g = np.asarray(Image.open(proxy_path).convert("L"), dtype=np.float32) / 255.0
    return np.percentile(g, [10, 50, 90]).astype(np.float32)


def day_number(date_str):
    try:
        return float(datetime.date.fromisoformat((date_str or "")[:10]).toordinal() - EPOCH)
    except ValueError:
        return None


def build_index_features(ids, recs, cache="index/features.npz"):
    if os.path.exists(cache):
        z = np.load(cache)  # own cache; ids stored as fixed-width unicode, no pickle needed
        if list(z["ids"]) == ids:
            return z["lums"], z["evs"], z["days"]
    lums = np.stack([lum(f"proxies/{x}.jpg") for x in ids])
    evs, days = [], []
    for x in ids:
        t = ""
        p = f"{EXPORT}/{x}.xmp"
        if os.path.exists(p):
            t = open(p, errors="ignore").read()
        evs.append(ev_from_xmp_text(t) if t else None)
        days.append(day_number(recs[x].get("capture_date")))
    evs = np.array([e if e is not None else 10.0 for e in evs], dtype=np.float32)
    days = np.array([d if d is not None else np.median([v for v in days if v]) for d in days],
                    dtype=np.float32)
    np.savez(cache, ids=np.array(ids), lums=lums, evs=evs, days=days)
    return lums, evs, days


def score(q_emb, q_lum, q_ev, q_day, embs, lums, evs, days):
    s = embs @ q_emb - W_LUM * ((lums - q_lum) ** 2).sum(1)
    if q_ev is not None:
        s = s - W_EV * np.abs(evs - q_ev)
    if q_day is not None:
        s = s - W_T * np.abs(days - q_day) / 365.0
    return s
