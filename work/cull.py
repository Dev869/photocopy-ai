"""Score raws for culling: pick-probability per image.

Usage: python3 cull.py folder_or_files...   -> cull_report.tsv (path, p_pick)
Threshold is the menu-bar knob; 0.5 ≈ Devin's own keep rate, lower = safer.
"""
import glob, os, sys

import numpy as np

from embed import load_model, embed_paths
from proxies import make_proxy


def score_paths(paths):
    w = np.load("index/cull.npz")["w"]
    model, processor, device = load_model()
    os.makedirs("tmp_proxies", exist_ok=True)
    proxies = []
    for p in paths:
        if p.lower().endswith((".jpg", ".jpeg")):
            proxies.append(p)
        else:
            q = f"tmp_proxies/{os.path.basename(p)}.jpg"
            if not os.path.exists(q):
                make_proxy(p, q)
            proxies.append(q)
    embs = embed_paths(proxies, model, processor, device).astype(np.float64)
    return 1 / (1 + np.exp(-(embs @ w[:-1] + w[-1])))


if __name__ == "__main__":
    paths = []
    for a in sys.argv[1:]:
        paths.extend(sorted(glob.glob(os.path.join(a, "*"))) if os.path.isdir(a) else [a])
    paths = [p for p in paths if p.lower().endswith((".nef", ".arw", ".dng", ".jpg", ".jpeg"))]
    probs = score_paths(paths)
    with open("cull_report.tsv", "w") as f:
        for p, pr in sorted(zip(paths, probs), key=lambda t: -t[1]):
            f.write(f"{pr:.3f}\t{p}\n")
    print(f"scored {len(paths)} images -> cull_report.tsv | "
          f"keep@0.5: {(probs > 0.5).mean():.0%}")
