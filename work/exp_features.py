"""Experiment: condition-aware retrieval.

Score = cosine(SigLIP) - w_lum*||lum_q - lum_i||^2 - w_ev*|ev_q - ev_i| - w_t*|days|/365
Sweep weights on holdout A (2025-01-04), validate the winner on B (2024-12-30).
"""
import datetime, json, os, re, statistics, sys

import numpy as np
from PIL import Image

KEYS = ("Exposure2012 Temperature Contrast2012 Highlights2012 Shadows2012 "
        "Whites2012 Blacks2012 Vibrance Saturation").split()
K = 8
EXPORT = os.path.expanduser("~/Pictures/agent-export")


def frac(s):
    if not s: return None
    m = re.match(r"(-?\d+)/(\d+)$", s)
    return float(m.group(1)) / float(m.group(2)) if m else float(s)


def exif_features(stem):
    """EV100 from the exported xmp/embedded text; None if unavailable."""
    for cand in (f"{EXPORT}/{stem}.xmp",):
        if not os.path.exists(cand):
            return None
        t = open(cand, errors="ignore").read()
        et = frac((re.search(r'exif:ExposureTime="([^"]*)"', t) or [None, None])[1])
        fn = frac((re.search(r'exif:FNumber="([^"]*)"', t) or [None, None])[1])
        iso_m = re.search(r"ISOSpeedRatings>\s*<rdf:Seq>\s*<rdf:li>(\d+)", t)
        iso = float(iso_m.group(1)) if iso_m else None
        if et and fn and iso:
            return float(np.log2(fn * fn / et) - np.log2(iso / 100.0))
    return None


def lum_features(proxy):
    g = np.asarray(Image.open(proxy).convert("L"), dtype=np.float32) / 255.0
    return np.percentile(g, [10, 50, 90]).astype(np.float32)


def build():
    embs = np.load("index/embeddings.npy")
    ids = json.load(open("index/ids.json"))
    recs = {r["asset_id"]: r for r in map(json.loads, open("dataset/index.jsonl"))
            if r.get("is_raw", True)}
    keep = [i for i, x in enumerate(ids) if x in recs]
    ids = [ids[i] for i in keep]
    embs = embs[keep]
    lums = np.stack([lum_features(f"proxies/{x}.jpg") for x in ids])
    evs = np.array([exif_features(x) or 10.0 for x in ids], dtype=np.float32)
    epoch = datetime.date(2020, 1, 1).toordinal()
    days = np.array([(datetime.date.fromisoformat((recs[x]["capture_date"] or "2024-01-01")[:10]).toordinal() - epoch)
                     for x in ids], dtype=np.float32)
    return embs, ids, recs, lums, evs, days


def evaluate(date, w_lum, w_ev, w_t, data, quiet=True):
    embs, ids, recs, lums, evs, days = data
    hold = [i for i, x in enumerate(ids) if (recs[x].get("capture_date") or "").startswith(date)]
    train = np.array([i for i in range(len(ids)) if i not in set(hold)])
    errs = {k: [] for k in KEYS}
    for i in hold:
        s = (embs[train] @ embs[i]
             - w_lum * ((lums[train] - lums[i]) ** 2).sum(1)
             - w_ev * np.abs(evs[train] - evs[i])
             - w_t * np.abs(days[train] - days[i]) / 365.0)
        nbrs = [recs[ids[train[j]]] for j in np.argsort(-s)[:K]]
        truth = recs[ids[i]]["settings"]
        for k in KEYS:
            if k not in truth:
                continue
            preds = [float(r["settings"][k]) for r in nbrs if k in r["settings"]]
            if preds:
                errs[k].append(abs(float(truth[k]) - statistics.median(preds)))
    maes = {k: statistics.mean(v) for k, v in errs.items() if v}
    if not quiet:
        for k, v in maes.items():
            print(f"  {k:<16}{v:.3f}")
    return maes


if __name__ == "__main__":
    data = build()
    print("sweep on 2025-01-04 (exposure / temperature / saturation MAE):")
    grid = [(0, 0, 0), (2, 0, 0), (5, 0, 0), (2, 0.1, 0), (2, 0.1, 0.3),
            (5, 0.1, 0.3), (2, 0.2, 0.6), (5, 0.2, 0.6), (10, 0.2, 0.6)]
    best, best_score = None, 1e9
    for w in grid:
        m = evaluate("2025-01-04", *w, data)
        score = m["Exposure2012"] / 0.97 + m["Temperature"] / 441 + m["Saturation"] / 7.8
        print(f"  w={w}: exp {m['Exposure2012']:.3f}  temp {m['Temperature']:.0f}  "
              f"sat {m['Saturation']:.2f}  contrast {m['Contrast2012']:.1f}  score {score:.2f}")
        if score < best_score:
            best, best_score = w, score
    print(f"\nbest {best} — validate on 2024-12-30:")
    evaluate("2024-12-30", *best, data, quiet=False)
    print("\nbaseline-free reference (w=0) on 2024-12-30:")
    evaluate("2024-12-30", 0, 0, 0, data, quiet=False)
