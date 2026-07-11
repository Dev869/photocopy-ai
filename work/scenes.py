"""Scene-consistent prediction: cluster a wedding into scenes, one edit per scene.

Greedy temporal clustering: images sorted by capture time; a new scene starts
on a >6-min gap or when embedding cosine to the running centroid drops < 0.5.
Per scene, every slider gets the median of all members' per-image kNN medians.

Validation: python3 scenes.py 2025-01-04
Reports per-image vs scene-smoothed MAE and intra-scene inconsistency (the
thing Devin actually complained about).
"""
import datetime, json, statistics, sys

import numpy as np

from predict import K, knn, load_index

KEYS = ("Exposure2012 Temperature Contrast2012 Highlights2012 Shadows2012 "
        "Whites2012 Blacks2012 Vibrance Saturation").split()
GAP_MIN = 6
MIN_COS = 0.5


def capture_ts(rec):
    try:
        return datetime.datetime.fromisoformat(rec["capture_date"][:19]).timestamp()
    except (TypeError, ValueError):
        return 0.0


def cluster_scenes(items):
    """items: [(ts, emb, key)] time-sorted -> list of lists of keys."""
    scenes, cur, centroid = [], [], None
    last_ts = None
    for ts, emb, key in items:
        if cur and (ts - last_ts > GAP_MIN * 60 or float(emb @ centroid) < MIN_COS):
            scenes.append(cur)
            cur, centroid = [], None
        cur.append(key)
        centroid = emb if centroid is None else centroid + emb
        centroid = centroid / np.linalg.norm(centroid)
        last_ts = ts
    if cur:
        scenes.append(cur)
    return scenes


def predict_scene_smoothed(idx, image_idxs, exclude=()):
    """Per-image kNN medians, then per-scene median. Returns {img_idx: {key: val}}, scenes."""
    ids, recs = idx["ids"], idx["recs"]
    per_image = {}
    for i in image_idxs:
        nbrs = knn(idx, idx["embs"][i], idx["lums"][i], idx["evs"][i],
                   idx["days"][i], k=K, exclude=exclude)
        per_image[i] = {k: statistics.median(vs) for k in KEYS
                        if (vs := [float(r["settings"][k]) for r, _ in nbrs
                                   if k in r["settings"]])}
    items = sorted(((capture_ts(recs[ids[i]]), idx["embs"][i], i) for i in image_idxs),
                   key=lambda t: t[0])
    scenes = cluster_scenes(items)
    smoothed = {}
    for scene in scenes:
        for k in KEYS:
            vals = [per_image[i][k] for i in scene if k in per_image[i]]
            if not vals:
                continue
            m = statistics.median(vals)
            for i in scene:
                smoothed.setdefault(i, {})[k] = m
    return per_image, smoothed, scenes


def validate(date):
    idx = load_index()
    ids, recs = idx["ids"], idx["recs"]
    hold = [i for i, x in enumerate(ids) if (recs[x].get("capture_date") or "").startswith(date)]
    hold_ids = {ids[i] for i in hold}
    per_image, smoothed, scenes = predict_scene_smoothed(idx, hold, exclude=hold_ids)
    print(f"{date}: {len(hold)} images -> {len(scenes)} scenes "
          f"(sizes: median {statistics.median(map(len, scenes)):.0f}, max {max(map(len, scenes))})")
    print(f"{'slider':<16}{'per-img MAE':>12}{'scene MAE':>11}{'incons/img':>11}{'incons/scn':>11}")
    for k in KEYS:
        e_img, e_scn = [], []
        for i in hold:
            t = recs[ids[i]]["settings"].get(k)
            if t is None:
                continue
            if k in per_image[i]:
                e_img.append(abs(float(t) - per_image[i][k]))
            if k in smoothed.get(i, {}):
                e_scn.append(abs(float(t) - smoothed[i][k]))
        # inconsistency: mean std of predictions within TRUE scenes (devin's real
        # per-scene groups approximated by our clustering of truth timestamps)
        inc_i, inc_s = [], []
        for scene in scenes:
            pi = [per_image[i][k] for i in scene if k in per_image[i]]
            ps = [smoothed[i][k] for i in scene if k in smoothed.get(i, {})]
            if len(pi) > 1:
                inc_i.append(statistics.stdev(pi))
            if len(ps) > 1:
                inc_s.append(statistics.stdev(ps))
        print(f"{k:<16}{statistics.mean(e_img):>12.3f}{statistics.mean(e_scn):>11.3f}"
              f"{statistics.mean(inc_i) if inc_i else 0:>11.3f}"
              f"{statistics.mean(inc_s) if inc_s else 0:>11.3f}")


if __name__ == "__main__":
    validate(sys.argv[1] if len(sys.argv) > 1 else "2025-01-04")
