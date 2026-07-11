"""Train the personal culling model: pick vs reject on SigLIP embeddings.

Usage: python3 cull_train.py ~/Pictures/agent-export-rejects
Positives: existing pick proxies/embeddings (index/). Negatives: exported
reject JPEGs. Split is grouped by capture date (whole weddings held out) to
avoid same-wedding leakage. Logistic head in numpy; weights -> index/cull.npz.

Caveat noted in NOTES: picks are camera-thumb JPEGs, rejects are LR-rendered
JPEGs. If val AUC looks too good, check the render-source confound.
"""
import datetime, glob, json, os, sys

import numpy as np
from PIL import Image

from embed import load_model, embed_paths


def catalog_dates():
    """fileName stem -> capture day from the catalog snapshot; collisions -> unknown."""
    import msgpack, sqlite3
    db = sqlite3.connect("catalog.mcat"); db.text_factory = bytes
    out = {}
    for (blob,) in db.execute("SELECT r.content FROM docs d JOIN revs r ON r.sequence=d.winningRevSequence "
                              "WHERE d.type='asset' AND d.subtype='image' AND d.deleted=0"):
        a = msgpack.unpackb(blob, raw=False)
        fn = a.get("importSource", {}).get("fileName")
        day = (a.get("captureDate") or "")[:10]
        if fn and day:
            stem = os.path.splitext(fn)[0]
            out[stem] = "unknown" if out.get(stem, day) != day else day
    return out


def capture_day(path, cat):
    try:
        ex = Image.open(path).getexif()
        dt = ex.get_ifd(0x8769).get(0x9003)  # DateTimeOriginal; 0x0132 is the export time
        if dt:
            return str(dt)[:10].replace(":", "-")
    except Exception:
        pass
    return cat.get(os.path.splitext(os.path.basename(path))[0], "unknown")


def logistic_fit(X, y, l2=1e-3, iters=300, lr_=1.0):
    w = np.zeros(X.shape[1] + 1, dtype=np.float64)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * w
        w -= lr_ * g
    return w


def auc(scores, y):
    order = np.argsort(scores)
    ranks = np.empty(len(y)); ranks[order] = np.arange(len(y))
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (pos.sum() * (~pos).sum())


def main(reject_dir):
    # positives: already-embedded pick proxies
    pick_embs = np.load("index/embeddings.npy")
    pick_ids = json.load(open("index/ids.json"))
    recs = {r["asset_id"]: r for r in map(json.loads, open("dataset/index.jsonl"))}
    pick_days = np.array([(recs.get(x, {}).get("capture_date") or "unknown")[:10] for x in pick_ids])

    rej_paths = sorted(glob.glob(os.path.join(reject_dir, "*.jpg"))
                       + glob.glob(os.path.join(reject_dir, "*.jpeg")))
    assert rej_paths, f"no jpegs in {reject_dir}"
    print(f"picks={len(pick_ids)} rejects={len(rej_paths)}")
    cache = "index/rejects_emb.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        rej_embs = z["embs"]
        assert len(rej_embs) == len(rej_paths), "rejects changed; delete cache"
    else:
        model, processor, device = load_model()
        rej_embs = embed_paths(rej_paths, model, processor, device)
        np.savez(cache, embs=rej_embs)
    cat = catalog_dates()
    rej_days = np.array([capture_day(p, cat) for p in rej_paths])
    known = (rej_days != "unknown").mean()
    print(f"reject dates resolved: {known:.0%}")

    X = np.vstack([pick_embs, rej_embs]).astype(np.float64)
    y = np.concatenate([np.ones(len(pick_embs)), np.zeros(len(rej_embs))])
    days = np.concatenate([pick_days, rej_days])

    rng = np.random.default_rng(0)
    uniq = rng.permutation(sorted(set(days)))
    val_days = set(uniq[:max(2, len(uniq) // 5)])
    val = np.array([d in val_days for d in days])
    w = logistic_fit(X[~val], y[~val])
    s = np.hstack([X[val], np.ones((val.sum(), 1))]) @ w
    pred = s > 0
    print(f"val weddings={len(val_days)} n={val.sum()}  AUC={auc(s, y[val]):.3f}  "
          f"acc={(pred == y[val]).mean():.3f}  "
          f"keep-rate@0={(pred).mean():.2f} (true pick rate {y[val].mean():.2f})")
    w_full = logistic_fit(X, y)
    np.savez("index/cull.npz", w=w_full)
    print("saved index/cull.npz")


if __name__ == "__main__":
    main(os.path.expanduser(sys.argv[1]))
