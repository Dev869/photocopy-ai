"""PPR10K (MMArt HF mirror) -> Engine B external dataset.

Unlike FiveK, this mirror's config.xmp files are PV2012 (Exposure2012, HSL,
Kelvin WB) — the same output space Engine B predicts, so all 37 targets map.
One preset token "PPR10K" for the whole expert set (single expert config per
image in this mirror). Rows are style anchors: train-only, never in eval.

    python3 ppr10k_ingest.py            # data/ppr10k/global/* -> data/ppr10k/eng/
"""
import glob, json, os

import numpy as np

OUT = "data/ppr10k/eng"
LOOK = "PPR10K"


def main():
    from embed import load_model, embed_paths
    from features import lum
    from mine import parse_settings
    groups = sorted(glob.glob("data/ppr10k/global/*"))
    items = []
    for g in groups:
        b, x = os.path.join(g, "before.jpg"), os.path.join(g, "config.xmp")
        if os.path.exists(b) and os.path.exists(x):
            items.append((os.path.basename(g), b, x))
    print(f"{len(items)} PPR10K images with before+config")

    model, processor, device = load_model()
    embs = embed_paths([b for _n, b, _x in items], model, processor, device)
    ids, recs, phot = [], [], []
    for (name, b, x), e in zip(items, embs):
        try:
            s = parse_settings(x)
        except Exception:
            continue
        aid = f"ppr_{name}"
        ids.append(aid)
        phot.append(np.concatenate([lum(b), [10.0]]).astype(np.float32))  # no EXIF EV
        recs.append({"asset_id": aid, "look": LOOK,
                     "settings": {k: str(v) for k, v in s.items()}})
    keep = [i for i, a in enumerate(ids)]
    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/embeddings.npy", embs[:len(ids)])
    np.save(f"{OUT}/phot.npy", np.stack(phot))
    json.dump(ids, open(f"{OUT}/ids.json", "w"))
    with open(f"{OUT}/index.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(ids)} rows -> {OUT}")


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != "work":
        os.chdir(os.path.dirname(__file__))
    main()
