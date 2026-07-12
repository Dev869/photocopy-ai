"""Build a FiveK external dataset for Engine B, restricted to what actually
transfers.

FiveK's catalog is 2011-era (Process Version <2012): no Exposure2012, no HSL,
no Clarity/Texture/Dehaze. The ONLY targets that map cleanly onto Devin's PV2012
output space are Temperature and Tint — absolute Kelvin, process-version-
independent. So FiveK enters only as a white-balance prior, tokened per expert
copy; every other slider is NaN and masked out by the head's loss.

Each of the 5000 inputs was edited by multiple expert copies (Copy 1..N). One
row per (input, copy): the input's SigLIP embedding + photometrics, target =
{Temperature, Tint}, look = "FiveK/<copy>".

    python3 fivek_ingest.py <dng_dir> <out_dir>
"""
import json, os, re, sqlite3, sys
import numpy as np

LRCAT = "data/fivek/fivek_dataset/raw_photos/fivek.lrcat"


def parse_wb(text):
    """Pull Temperature/Tint (Kelvin) from the Lua-text develop blob."""
    out = {}
    for key in ("Temperature", "Tint"):
        m = re.search(rf"\b{key} = (-?\d+(?:\.\d+)?)", text)
        if m:
            out[key] = float(m.group(1))
    return out


def catalog_wb(stems):
    """stem (filename w/o ext) -> [(copyName, {Temperature,Tint}), ...]."""
    con = sqlite3.connect(f"file:{LRCAT}?mode=ro", uri=True)
    rows = con.execute("""
        SELECT f.idx_filename, i.copyName, ds.text
        FROM Adobe_images i
        JOIN AgLibraryFile f ON f.id_local = i.rootFile
        JOIN Adobe_imageDevelopSettings ds ON ds.image = i.id_local
        WHERE ds.text LIKE '%Temperature = %'""").fetchall()
    con.close()
    want = set(stems)
    by_stem = {}
    for fname, copy, text in rows:
        stem = os.path.splitext(fname)[0]
        if stem not in want:
            continue
        wb = parse_wb(text)
        if "Temperature" in wb:
            by_stem.setdefault(stem, []).append((copy or "orig", wb))
    return by_stem


def main(dng_dir, out_dir):
    from embed import load_model, embed_paths
    from features import lum, ev_from_raw
    from proxies import make_proxy
    dngs = sorted(f for f in os.listdir(dng_dir) if f.lower().endswith(".dng"))
    stems = [os.path.splitext(f)[0] for f in dngs]
    wb = catalog_wb(stems)
    dngs = [f for f in dngs if os.path.splitext(f)[0] in wb]  # keep only labeled
    print(f"{len(dngs)} inputs with WB labels")

    os.makedirs("tmp_proxies", exist_ok=True)
    model, processor, device = load_model()
    emb_rows, phot_rows, ids, recs = [], [], [], []
    for i, f in enumerate(dngs):
        stem = os.path.splitext(f)[0]
        path = os.path.join(dng_dir, f)
        proxy = f"tmp_proxies/fivek_{stem}.jpg"
        try:
            make_proxy(path, proxy)
        except Exception:
            continue
        e = embed_paths([proxy], model, processor, device)[0]
        lp = lum(proxy); ev = ev_from_raw(path)
        ph = np.concatenate([lp, [ev if ev is not None else 10.0]]).astype(np.float32)
        for copy, s in wb[stem]:
            ids.append(f"fivek_{stem}#{copy}")
            emb_rows.append(e); phot_rows.append(ph)
            recs.append({"asset_id": f"fivek_{stem}#{copy}", "look": f"FiveK/{copy}",
                         "settings": {k: str(v) for k, v in s.items()}})
        os.remove(proxy)
        if (i + 1) % 100 == 0:
            print(f"  embedded {i + 1}/{len(dngs)} inputs -> {len(ids)} rows")

    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/embeddings.npy", np.stack(emb_rows))
    np.save(f"{out_dir}/phot.npy", np.stack(phot_rows))
    json.dump(ids, open(f"{out_dir}/ids.json", "w"))
    with open(f"{out_dir}/index.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    looks = sorted({r["look"] for r in recs})
    print(f"wrote {len(ids)} rows, {len(looks)} expert presets: {looks}")


def _demo():
    d = parse_wb('s = { Temperature = 4700, Tint = 12, Exposure = 0.15 }')
    assert d == {"Temperature": 4700.0, "Tint": 12.0}, d
    assert parse_wb("no wb here") == {}
    print("fivek_ingest.parse_wb OK")


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != "work":
        os.chdir(os.path.dirname(__file__))
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        _demo()
