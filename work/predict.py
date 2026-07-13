"""Engine A: retrieval-based auto-edit. Input images -> XMP sidecars.

Usage: python3 predict.py photo1.NEF [photo2.NEF ...]
       python3 predict.py --eval          (leave-one-out on the index itself)

Blend: nearest neighbor's full settings XMP as the base (keeps Look, curves,
profile internally consistent), masks stripped, Basic-panel + HSL sliders
overridden with the k-neighbor median.
"""
import json, os, re, statistics, sys

import numpy as np

LR = os.path.expanduser(
    "~/Pictures/Lightroom Library.lrlibrary/8d15b67bae9846e39beea23257a7e871")
K = 8
BLEND_INT = (
    "Temperature Tint Contrast2012 Highlights2012 Shadows2012 Whites2012 Blacks2012 "
    "Texture Clarity2012 Dehaze Vibrance Saturation "
    + " ".join(f"{g}Adjustment{c}" for g in ("Hue", "Saturation", "Luminance")
               for c in "Red Orange Yellow Green Aqua Blue Purple Magenta".split())
).split()
BLEND_FLOAT = ["Exposure2012"]


def load_index():
    from features import build_index_features
    embs = np.load("index/embeddings.npy")
    ids = json.load(open("index/ids.json"))
    recs = {r["asset_id"]: r for r in map(json.loads, open("dataset/index.jsonl"))
            if r.get("is_raw", True)}  # jpeg edits use incremental WB — unblendable with raw
    keep = [i for i, x in enumerate(ids) if x in recs]
    ids = [ids[i] for i in keep]
    lums, evs, days = build_index_features(ids, recs)
    return {"embs": embs[keep], "ids": ids, "recs": recs,
            "lums": lums, "evs": evs, "days": days}


def blend_sidecar(neighbor_recs, overrides=None, jpeg=False):
    """neighbor_recs: ordered nearest-first. overrides: {key: value} wins over medians.
    jpeg=True: target is a rendered file — Kelvin WB doesn't apply; keep tone/color only."""
    base = open(neighbor_recs[0]["settings_path"]).read()
    # strip image-specific masks
    base = re.sub(r"\s*<crs:MaskGroupBasedCorrections>.*?</crs:MaskGroupBasedCorrections>\n",
                  "\n", base, flags=re.S)
    def median(key):
        if overrides and key in overrides:
            return overrides[key]
        vals = [float(r["settings"][key]) for r in neighbor_recs if key in r["settings"]]
        return statistics.median(vals) if vals else None
    # white balance: enum by neighbor majority; only blend Temp/Tint for Custom
    wb_votes = [r["settings"].get("WhiteBalance") for r in neighbor_recs]
    wb = statistics.mode([w for w in wb_votes if w] or ["As Shot"])
    base = re.sub(r'crs:WhiteBalance="[^"]*"', f'crs:WhiteBalance="{wb}"', base)
    skip = () if wb == "Custom" else ("Temperature", "Tint")
    if skip:  # drop absolute temp/tint so LR honors the WB preset
        base = re.sub(r'\s*crs:(Temperature|Tint)="[^"]*"\n?', "\n   ", base)
    for key in BLEND_INT + BLEND_FLOAT:
        if key in skip:
            continue
        m = median(key)
        if m is None:
            continue
        txt = f"{m:+.2f}" if key in BLEND_FLOAT else f"{round(m):+d}"
        txt = txt.replace("+0", "0", 1) if txt in ("+0", "+0.00") else txt
        base, n = re.subn(f'crs:{key}="[^"]*"', f'crs:{key}="{txt}"', base)
        if n == 0:  # attr absent in base file: inject after ProcessVersion
            base = base.replace('crs:ProcessVersion=',
                                f'crs:{key}="{txt}"\n   crs:ProcessVersion=', 1)
    if jpeg:  # LR ignores/misreads Kelvin WB on rendered files
        base = re.sub(r'\s*crs:(Temperature|Tint)="[^"]*"\n?', "\n   ", base)
        base = re.sub(r'crs:WhiteBalance="[^"]*"', 'crs:WhiteBalance="As Shot"', base)
    return ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            + base + '\n<?xpacket end="w"?>')


def subset_by_look(idx, look):
    keep = [i for i, x in enumerate(idx["ids"]) if idx["recs"][x].get("look") == look]
    assert keep, f"no history with look {look!r}"
    return {"embs": idx["embs"][keep], "ids": [idx["ids"][i] for i in keep],
            "recs": idx["recs"], "lums": idx["lums"][keep],
            "evs": idx["evs"][keep], "days": idx["days"][keep]}


def knn(idx, query_emb, q_lum, q_ev=None, q_day=None, k=K, exclude=()):
    from features import score
    s = score(query_emb, q_lum, q_ev, q_day,
              idx["embs"], idx["lums"], idx["evs"], idx["days"])
    out = []
    for i in np.argsort(-s):
        if idx["ids"][i] in exclude:
            continue
        out.append((idx["recs"][idx["ids"][i]], float(s[i])))
        if len(out) == k:
            break
    return out


SMOOTH_KEYS = ("Exposure2012 Temperature Tint Contrast2012 Highlights2012 Shadows2012 "
               "Whites2012 Blacks2012 Vibrance Saturation").split()


def main(paths, look=None, engine="a", progress=None, on_written=None):
    import datetime, statistics as st
    from embed import load_model, embed_paths
    from features import EPOCH, ev_from_raw, lum
    from proxies import make_proxy
    from scenes import cluster_scenes
    idx = load_index()
    if look:
        try:
            idx = subset_by_look(idx, look)
        except AssertionError:
            print(f"look {look!r} has no retrieval history — using Engine B style token")
            engine = "b"
    model, processor, device = load_model()
    os.makedirs("tmp_proxies", exist_ok=True)
    today = float(datetime.date.today().toordinal() - EPOCH)
    per = []  # (path, ts, emb, nbrs, phot)
    for p in paths:
        if p.lower().endswith((".jpg", ".jpeg", ".png")):
            proxy, ev = p, None
        else:
            proxy = f"tmp_proxies/{os.path.basename(p)}.jpg"
            make_proxy(p, proxy)
            ev = ev_from_raw(p)
        q = embed_paths([proxy], model, processor, device)[0]
        lp = lum(proxy)
        nbrs = knn(idx, q, lp, ev, today)
        per.append((p, os.path.getmtime(p), q, nbrs,
                    np.concatenate([lp, [ev if ev is not None else 10.0]]).astype(np.float32)))
        if progress:
            progress(len(per), len(paths))
    scenes = cluster_scenes(sorted(((ts, q, n) for n, (_p, ts, q, _nb, _f) in enumerate(per)),
                                   key=lambda t: t[0]))
    if engine == "b" and not os.path.exists("index/engineB.pt"):
        print("engine B requested but index/engineB.pt missing — run train_head.py; using A")
        engine = "a"
    if engine == "b":
        # Engine B: trained head supplies all sliders; kNN XMP still gives structure.
        from train_head import load_head, infer
        from presets import look2idx, encode
        head, meta = load_head()
        # slot map from the checkpoint's own names, so it can't drift if the
        # dataset (and thus a fresh look2idx) changes after training
        li = ({n: i for i, n in enumerate(meta["names"])}.get(look or "Unknown", 0)
              if meta.get("names") else encode(look, look2idx()[0]))
        preds = infer(head, meta, np.stack([r[2] for r in per]),
                      np.stack([r[4] for r in per]), [li] * len(per))
        overrides = {n: dict(d) for n, d in enumerate(preds)}
        for scene in scenes:  # smooth the model's own predictions within a scene
            for k in SMOOTH_KEYS:
                vals = [preds[m][k] for m in scene if k in preds[m]]
                if vals:
                    med = st.median(vals)
                    for m in scene:
                        overrides[m][k] = med
    else:
        # Engine A: scene-median of neighbor settings for exposure/WB/tone keys
        overrides = {}
        for scene in scenes:
            med = {}
            for k in SMOOTH_KEYS:
                vals = [float(r["settings"][k]) for m in scene
                        for r, _ in per[m][3] if k in r["settings"]]
                if vals:
                    med[k] = st.median(vals)
            for m in scene:
                overrides[m] = med
    for n, (p, _ts, _q, nbrs, _f) in enumerate(per):
        sidecar = os.path.splitext(p)[0] + ".xmp"
        is_jpg = p.lower().endswith((".jpg", ".jpeg"))
        open(sidecar, "w").write(
            blend_sidecar([r for r, _ in nbrs], overrides=overrides.get(n), jpeg=is_jpg))
        if on_written:
            on_written(p)
        print(f"{os.path.basename(p)} -> {os.path.basename(sidecar)} | "
              f"neighbors: {[(r['file_name'], round(s, 3)) for r, s in nbrs[:3]]}")


def loo_eval():
    idx = load_index()
    ids, recs = idx["ids"], idx["recs"]
    errs = {k: [] for k in ("Exposure2012", "Temperature", "Contrast2012")}
    for i, aid in enumerate(ids):
        nbrs = knn(idx, idx["embs"][i], idx["lums"][i], idx["evs"][i],
                   idx["days"][i], exclude={aid})
        for key in errs:
            truth = recs[aid]["settings"].get(key)
            preds = [float(r["settings"][key]) for r, _ in nbrs if key in r["settings"]]
            if truth is not None and preds:
                errs[key].append(abs(float(truth) - statistics.median(preds)))
    for k, v in errs.items():
        print(f"LOO MAE {k}: {statistics.mean(v):.3f}  (n={len(v)})")


if __name__ == "__main__":
    if sys.argv[1:] == ["--eval"]:
        loo_eval()
    else:
        args = sys.argv[1:]
        look = engine = None
        if "--look" in args:
            j = args.index("--look")
            look = args[j + 1]
            args = args[:j] + args[j + 2:]
        if "--engine" in args:
            j = args.index("--engine")
            engine = args[j + 1]
            args = args[:j] + args[j + 2:]
        main(args, look, engine=engine or "a")
