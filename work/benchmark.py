"""Held-out wedding benchmark for Engine A.

Usage: python3 benchmark.py 2025-01-04 [--import-dir ../benchmark-import --sample 40]
Excludes the wedding (by capture-date prefix) from the neighbor pool, predicts
every raw in it, reports MAE vs Devin's real edits, and compares against a
global-median baseline. Optionally builds a padded-raw import folder for
render judgment in Lightroom.
"""
import json, os, shutil, statistics, sys

import numpy as np

from predict import K, blend_sidecar, knn, load_index

KEYS = ("Exposure2012 Temperature Contrast2012 Highlights2012 Shadows2012 "
        "Whites2012 Blacks2012 Vibrance Saturation").split()


def run(date, import_dir=None, sample=40):
    idx = load_index()
    embs, ids, recs = idx["embs"], idx["ids"], idx["recs"]
    hold = [i for i, x in enumerate(ids) if (recs[x].get("capture_date") or "").startswith(date)]
    hold_ids = {ids[i] for i in hold}
    train_ids = {x for x in ids if x not in hold_ids}
    assert hold, f"no records for {date}"
    print(f"holdout {date}: {len(hold)} images | index: {len(train_ids)} images")

    train_recs = [recs[x] for x in train_ids]
    baseline = {k: statistics.median(float(r["settings"][k]) for r in train_recs
                                     if k in r["settings"]) for k in KEYS}
    errs = {k: [] for k in KEYS}
    base_errs = {k: [] for k in KEYS}
    sims = []
    os.makedirs(f"benchmark/{date}", exist_ok=True)
    for n, i in enumerate(hold):
        aid = ids[i]
        nbrs = knn(idx, embs[i], idx["lums"][i], idx["evs"][i], idx["days"][i],
                   k=K, exclude=hold_ids)
        sims.append(nbrs[0][1])
        nbr_recs = [r for r, _ in nbrs]
        open(f"benchmark/{date}/{aid}.xmp", "w").write(blend_sidecar(nbr_recs))
        truth = recs[aid]["settings"]
        for k in KEYS:
            if k not in truth:
                continue
            preds = [float(r["settings"][k]) for r in nbr_recs if k in r["settings"]]
            if preds:
                errs[k].append(abs(float(truth[k]) - statistics.median(preds)))
            base_errs[k].append(abs(float(truth[k]) - baseline[k]))

    print(f"nearest-neighbor cosine: mean {statistics.mean(sims):.3f}, "
          f"min {min(sims):.3f}")
    print(f"{'slider':<16}{'kNN MAE':>10}{'baseline':>10}{'n':>6}")
    for k in KEYS:
        if errs[k]:
            print(f"{k:<16}{statistics.mean(errs[k]):>10.3f}"
                  f"{statistics.mean(base_errs[k]):>10.3f}{len(errs[k]):>6}")

    if import_dir:
        os.makedirs(import_dir, exist_ok=True)
        picked = [ids[i] for i in hold][::max(1, len(hold) // sample)][:sample]
        for aid in picked:
            rec = recs[aid]
            stem = os.path.splitext(os.path.basename(rec["original"]))[0]
            raw_ext = os.path.splitext(rec["original"])[1]
            dst = os.path.join(import_dir, f"{stem}_bench{raw_ext}")
            shutil.copy(rec["original"], dst)
            with open(dst, "ab") as f:  # new hash so LR won't dedupe-skip
                f.write(b"\0" * 7)
            shutil.copy(f"benchmark/{date}/{aid}.xmp",
                        os.path.join(import_dir, f"{stem}_bench.xmp"))
        print(f"import folder: {import_dir} ({len(picked)} raws + sidecars)")


if __name__ == "__main__":
    date = sys.argv[1]
    imp = sys.argv[sys.argv.index("--import-dir") + 1] if "--import-dir" in sys.argv else None
    run(date, imp)
