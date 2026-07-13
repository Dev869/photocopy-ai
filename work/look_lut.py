"""Fit a per-Look color transform for the proof renderer from LR's own renders.

Looks (Adobe creative profiles) are Lightroom LUTs the proof renderer can't
apply — so proofs carried correct sliders but not the style. The catalog's
preview JPEGs ARE Lightroom renders of Devin's real edits; the preview mapping
db is truncated, so pairs are recovered by SigLIP content-matching previews to
the asset proxies already in the index. For each matched pair we render OUR
version (sliders, no Look) and pool pixels; the per-channel quantile transfer
ours->LR *is* the average Look transform (256-entry LUT per channel).

    python3 look_lut.py "Modern 08" [n_preview_sample]

Writes/updates index/look_luts.npz {look: (3,256) uint8}.
"""
import json, os, random, sys

import numpy as np
from PIL import Image

LR_PREVIEWS = os.path.expanduser(
    "~/Pictures/Lightroom Library.lrlibrary/8d15b67bae9846e39beea23257a7e871/previews")
LUTS = "index/look_luts.npz"
COS_MIN = 0.88
MIN_DIM = 300


def look_assets(look):
    out = []
    for l in open("dataset/index.jsonl"):
        r = json.loads(l)
        if (r.get("look") == look and r.get("is_raw", True)
                and os.path.exists(r.get("original", ""))):
            out.append(r)
    return out


def sample_previews(n, seed=0):
    files = []
    for root, _dirs, names in os.walk(LR_PREVIEWS):
        files += [os.path.join(root, f) for f in names]
    random.Random(seed).shuffle(files)
    keep = []
    for f in files:
        if len(keep) >= n:
            break
        try:
            with Image.open(f) as im:
                if min(im.size) >= MIN_DIM:
                    keep.append(f)
        except Exception:
            continue
    return keep


def fit(look, n_sample=9000):
    from embed import load_model, embed_paths
    from mine import parse_settings
    from preview import render_raw_jpeg

    recs = look_assets(look)
    ids = json.load(open("index/ids.json"))
    embs = np.load("index/embeddings.npy")
    row = {a: i for i, a in enumerate(ids)}
    recs = [r for r in recs if r["asset_id"] in row]
    A = np.stack([embs[row[r["asset_id"]]] for r in recs])
    print(f"{look}: {len(recs)} assets with embeddings + local originals")

    previews = sample_previews(n_sample)
    print(f"embedding {len(previews)} sampled previews…")
    model, processor, device = load_model()
    P = embed_paths(previews, model, processor, device)

    sims = A @ P.T                       # both are L2-normalized by embed.py
    best = sims.argmax(1)
    pairs = [(recs[i], previews[best[i]], float(sims[i, best[i]]))
             for i in range(len(recs)) if sims[i, best[i]] >= COS_MIN]
    print(f"{len(pairs)} confident asset<->preview matches (cos >= {COS_MIN})")
    if len(pairs) < 15:
        raise SystemExit("too few matches — raise n_sample")

    ours_px, lr_px = [], []
    os.makedirs("tmp_proxies", exist_ok=True)
    for k, (r, prev, _c) in enumerate(pairs):
        out = f"tmp_proxies/_lut_{r['asset_id']}.jpg"
        try:
            render_raw_jpeg(r["original"], parse_settings(r["settings_path"]),
                            out, quality=95, long_side=640)
            o = np.asarray(Image.open(out).convert("RGB"), np.uint8).reshape(-1, 3)
            t = np.asarray(Image.open(prev).convert("RGB"), np.uint8).reshape(-1, 3)
        except Exception as e:
            print(f"  skip {r['asset_id']}: {type(e).__name__}")
            continue
        finally:
            if os.path.exists(out):
                os.remove(out)
        # subsample pixels so no single frame dominates the pool
        rng = np.random.default_rng(k)
        ours_px.append(o[rng.choice(len(o), min(20000, len(o)), replace=False)])
        lr_px.append(t[rng.choice(len(t), min(20000, len(t)), replace=False)])
        if (k + 1) % 20 == 0:
            print(f"  rendered {k + 1}/{len(pairs)}")
    ours = np.concatenate(ours_px).astype(np.float32)
    lr = np.concatenate(lr_px).astype(np.float32)

    q = np.linspace(0, 100, 256)
    lut = np.zeros((3, 256), np.uint8)
    for c in range(3):
        src_q = np.percentile(ours[:, c], q)      # value at each quantile, ours
        dst_q = np.percentile(lr[:, c], q)        # value at each quantile, LR
        # LUT[v] = LR value at the quantile where v sits in our distribution
        lut[c] = np.clip(np.interp(np.arange(256), src_q, dst_q), 0, 255).astype(np.uint8)

    store = dict(np.load(LUTS)) if os.path.exists(LUTS) else {}
    store[look] = lut
    np.savez(LUTS, **store)
    print(f"saved LUT for {look!r} ({len(ours_px)} pairs) -> {LUTS}")
    # quick fidelity readout: mean abs channel shift the Look applies
    print("mean shift per channel:", [round(float(np.mean(lut[c] - np.arange(256))), 1)
                                      for c in range(3)])


def _demo():
    # identity distributions must give an identity-ish LUT
    x = np.tile(np.arange(256, dtype=np.float32), 100)
    q = np.linspace(0, 100, 256)
    sq, dq = np.percentile(x, q), np.percentile(x, q)
    lut = np.interp(np.arange(256), sq, dq)
    assert np.abs(lut - np.arange(256)).max() < 1.5
    print("look_lut quantile-mapping identity OK")


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != "work":
        os.chdir(os.path.dirname(__file__))
    if len(sys.argv) < 2:
        _demo()
    else:
        fit(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 9000)
