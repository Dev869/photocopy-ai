"""Engine B: a small regression head on frozen SigLIP2 embeddings + photometric
features + a learned preset token. Predicts the same slider vector Engine A
blends, so it drops straight into predict.py's `overrides`.

    emb(768) + photometric(4) + preset_token  ->  MLP  ->  37 sliders

Photometric features (luminance percentiles + EXIF EV) are the point: they let
the head learn exposure-from-image, the one thing content-only kNN retrieval
can't see and loses to a global median on across weddings (see NOTES.md).

Usage:
    python3 train_head.py            # train, wedding-grouped val, save index/engineB.pt
    python3 train_head.py --compare  # + Engine A kNN and baseline on the same val
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn

from features import build_index_features
from presets import look2idx, encode

KEYS = (["Exposure2012", "Temperature", "Tint", "Contrast2012", "Highlights2012",
         "Shadows2012", "Whites2012", "Blacks2012", "Texture", "Clarity2012",
         "Dehaze", "Vibrance", "Saturation"]
        + [f"{g}Adjustment{c}" for g in ("Hue", "Saturation", "Luminance")
           for c in "Red Orange Yellow Green Aqua Blue Purple Magenta".split()])
# most-visible sliders get up-weighted in the loss (plan §3)
WEIGHT = {k: (2.0 if k in ("Exposure2012", "Temperature", "Tint") else 1.0) for k in KEYS}
REPORT = ["Exposure2012", "Temperature", "Contrast2012"]  # comparable to Engine A LOO
CKPT = "index/engineB.pt"
EPOCHS = 100  # sweet spot: fewer underfits, more overfits exposure (swept 4-fold)


def _rows_to_Y(ids, recs):
    Y = np.full((len(ids), len(KEYS)), np.nan, np.float32)
    for r, x in enumerate(ids):
        s = recs[x]["settings"]
        for c, k in enumerate(KEYS):
            if k in s:
                Y[r, c] = float(s[k])
    return Y


def load_data(external=()):
    """Devin's index + optional external expert sources (FiveK, PPR10K, ...).

    Each external source is a dir with: embeddings.npy, ids.json, phot.npy (n,4)
    lums+EV, index.jsonl (Devin-schema records with settings + look). Its rows are
    style anchors — always trained on, never in the held-out eval. Register the
    external looks (presets.register_external) BEFORE calling so they get slots.
    """
    idx, names = look2idx()
    # Devin's own data (photometrics built from proxies + EXPORT xmps)
    embs = np.load("index/embeddings.npy")
    ids = json.load(open("index/ids.json"))
    recs = {r["asset_id"]: r for r in map(json.loads, open("dataset/index.jsonl"))
            if r.get("is_raw", True)}
    keep = [i for i, x in enumerate(ids) if x in recs]
    ids = [ids[i] for i in keep]
    embs = embs[keep].astype(np.float32)
    lums, evs, _ = build_index_features(ids, recs)
    phot = np.concatenate([lums, evs[:, None]], axis=1).astype(np.float32)
    looks = np.array([encode(recs[x].get("look"), idx) for x in ids], np.int64)
    weddings = np.array([(recs[x].get("capture_date") or "")[:10] for x in ids])
    Y = _rows_to_Y(ids, recs)
    external_mask = np.zeros(len(ids), bool)

    for d in external:  # each external dir carries precomputed embs + phot
        e_embs = np.load(f"{d}/embeddings.npy").astype(np.float32)
        e_ids = json.load(open(f"{d}/ids.json"))
        e_phot = np.load(f"{d}/phot.npy").astype(np.float32)
        e_recs = {r["asset_id"]: r for r in map(json.loads, open(f"{d}/index.jsonl"))}
        e_ids = [x for x in e_ids if x in e_recs]  # keep order, drop missing
        sel = [i for i, x in enumerate(json.load(open(f"{d}/ids.json"))) if x in e_recs]
        embs = np.concatenate([embs, e_embs[sel]])
        phot = np.concatenate([phot, e_phot[sel]])
        looks = np.concatenate([looks, [encode(e_recs[x].get("look"), idx) for x in e_ids]])
        weddings = np.concatenate([weddings, [f"ext:{e_recs[x].get('look')}" for x in e_ids]])
        Y = np.concatenate([Y, _rows_to_Y(e_ids, e_recs)])
        external_mask = np.concatenate([external_mask, np.ones(len(e_ids), bool)])
        recs.update(e_recs); ids = ids + e_ids

    return dict(ids=ids, embs=embs, phot=phot, looks=looks, weddings=weddings,
                Y=Y, recs=recs, n_presets=len(names), names=names,
                external=external_mask)


def grouped_split(weddings, val_frac=0.18, seed=0):
    """Hold out whole weddings — the honest cross-wedding regime."""
    uniq = sorted(set(weddings))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    # pick weddings until ~val_frac of images are held out, from the shuffled order
    val_w, n_tot, target = set(), len(weddings), val_frac * len(weddings)
    got = 0
    for w in uniq:
        if got >= target:
            break
        val_w.add(w); got += int((weddings == w).sum())
    tr = np.array([w not in val_w for w in weddings])
    return tr, ~tr, sorted(val_w)


class Head(nn.Module):
    def __init__(self, n_presets, emb=768, phot=4, look_dim=8, hidden=384, n_out=len(KEYS)):
        super().__init__()
        self.look = nn.Embedding(n_presets, look_dim)
        self.net = nn.Sequential(
            nn.Linear(emb + phot + look_dim, hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, n_out))

    def forward(self, e, p, l):
        return self.net(torch.cat([e, p, self.look(l)], -1))


def standardize(a, mean, std):
    return (a - mean) / std


def train(compare=False, seed=0, external=()):
    torch.manual_seed(seed); np.random.seed(seed)  # reproducible weights + split
    d = load_data(external=external)
    tr, va, val_w = grouped_split(d["weddings"], seed=seed)
    tr = tr | d["external"]; va = va & ~d["external"]  # experts train-only; eval on Devin
    n_ext = int(d["external"].sum())
    print(f"train {tr.sum()} imgs ({n_ext} external) / val {va.sum()} imgs "
          f"over {len(val_w)} held-out weddings")

    # normalization from TRAIN only
    ym = np.nanmean(d["Y"][tr], 0); ys = np.nanstd(d["Y"][tr], 0) + 1e-6
    pm = d["phot"][tr].mean(0); ps = d["phot"][tr].std(0) + 1e-6
    w = np.array([WEIGHT[k] for k in KEYS], np.float32)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    def T(a): return torch.tensor(a, device=dev)
    E, P = T(d["embs"]), T(standardize(d["phot"], pm, ps).astype(np.float32))
    L = T(d["looks"])
    Yz = standardize(d["Y"], ym, ys).astype(np.float32)
    mask = T((~np.isnan(d["Y"])).astype(np.float32))
    Yz = T(np.nan_to_num(Yz)); W = T(w)
    itr, iva = T(np.where(tr)[0].astype(np.int64)), T(np.where(va)[0].astype(np.int64))

    huber = nn.HuberLoss(reduction="none", delta=1.0)

    def fit(idxs, Ptbl, Ytbl):
        m = Head(d["n_presets"]).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        for _ in range(EPOCHS):
            m.train(); opt.zero_grad()
            per = huber(m(E[idxs], Ptbl[idxs], L[idxs]), Ytbl[idxs]) * mask[idxs] * W
            (per.sum() / (mask[idxs] * W).sum()).backward(); opt.step()
        m.eval(); return m

    # eval model: train split only, measure on held-out weddings
    model = fit(itr, P, Yz)
    with torch.no_grad():
        pv = model(E[iva], P[iva], L[iva]).cpu().numpy() * ys + ym
    Yv = d["Y"][va]
    print(f"\nEngine B — val MAE (real units, {EPOCHS} epochs, held-out weddings):")
    b_mae = {}
    for c, k in enumerate(KEYS):
        m = ~np.isnan(Yv[:, c])
        if k in REPORT and m.any():
            e = np.abs(pv[m, c] - Yv[m, c]).mean(); b_mae[k] = e
            print(f"  {k:16s} {e:8.3f}  (n={m.sum()})")

    # deployed model: refit on ALL data (norm stats recomputed on all rows)
    dym = np.nanmean(d["Y"], 0); dys = np.nanstd(d["Y"], 0) + 1e-6
    dpm = d["phot"].mean(0); dps = d["phot"].std(0) + 1e-6
    Pall = T(standardize(d["phot"], dpm, dps).astype(np.float32))
    Yzall = T(np.nan_to_num(standardize(d["Y"], dym, dys).astype(np.float32)))
    iall = T(np.arange(len(d["ids"]), dtype=np.int64))
    dep = fit(iall, Pall, Yzall)
    torch.save(dict(state=dep.state_dict(), n_presets=d["n_presets"], keys=KEYS,
                    names=d["names"], ym=dym, ys=dys, pm=dpm, ps=dps), CKPT)
    print(f"saved {CKPT} (all-data refit, {EPOCHS} epochs)")

    if compare:
        compare_engines(d, tr, va, b_mae)
    return b_mae


def compare_engines(d, tr, va, b_mae):
    """Engine A (kNN over TRAIN only) and global-median baseline, same val set."""
    from features import score
    tri = np.where(tr)[0]
    embs_t, lums_t = d["embs"][tri], d["phot"][tri][:, :3]
    evs_t = d["phot"][tri][:, 3]
    a_mae, base_mae = {k: [] for k in REPORT}, {k: [] for k in REPORT}
    med = {k: np.nanmedian(d["Y"][tr][:, KEYS.index(k)]) for k in REPORT}
    for r in np.where(va)[0]:
        s = score(d["embs"][r], d["phot"][r][:3], d["phot"][r][3], None,
                  embs_t, lums_t, evs_t, None)
        nn8 = tri[np.argsort(-s)[:8]]
        for k in REPORT:
            c = KEYS.index(k)
            preds = d["Y"][nn8, c]; preds = preds[~np.isnan(preds)]
            truth = d["Y"][r, c]
            if not np.isnan(truth):
                if len(preds):
                    a_mae[k].append(abs(np.median(preds) - truth))
                base_mae[k].append(abs(med[k] - truth))
    print("\n           Engine B   Engine A(kNN)   global-median")
    for k in REPORT:
        a = np.mean(a_mae[k]); b = b_mae.get(k, float('nan')); g = np.mean(base_mae[k])
        print(f"  {k:16s} {b:8.3f}   {a:8.3f}      {g:8.3f}")


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != "work":
        os.chdir(os.path.dirname(__file__))
    train(compare="--compare" in sys.argv)


def load_head(ckpt=CKPT, device=None):
    """Load a trained head for inference. Returns (model, meta)."""
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    m = torch.load(ckpt, map_location=dev, weights_only=False)
    model = Head(m["n_presets"]).to(dev)
    model.load_state_dict(m["state"]); model.eval()
    return model, m


def infer(model, meta, embs, phot, look_idx, device=None):
    """embs (n,768), phot (n,4), look_idx (n,) -> {key: value} per row (real units)."""
    dev = next(model.parameters()).device
    P = (np.asarray(phot, np.float32) - meta["pm"]) / meta["ps"]
    with torch.no_grad():
        z = model(torch.tensor(np.asarray(embs, np.float32), device=dev),
                  torch.tensor(P.astype(np.float32), device=dev),
                  torch.tensor(np.asarray(look_idx, np.int64), device=dev)).cpu().numpy()
    vals = z * meta["ys"] + meta["ym"]
    keys = meta["keys"]
    return [{k: float(vals[r, c]) for c, k in enumerate(keys)} for r in range(len(vals))]
