"""Phase 0: mine Lightroom (cloud) Managed Catalog into a (metadata, develop-settings) dataset.

Usage: python3 mine.py [--all]   (default: picked images only)
Output: dataset/index.jsonl — one record per image with inline crs settings.
"""
import json, os, sqlite3, sys

import msgpack

try:
    import defusedxml.ElementTree as ET
except ImportError:  # local Adobe-written files, not untrusted input
    import xml.etree.ElementTree as ET

LR = os.path.expanduser(
    "~/Pictures/Lightroom Library.lrlibrary/8d15b67bae9846e39beea23257a7e871")
CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def parse_settings(path):
    """crs XMP -> flat dict. Attributes + tone-curve seqs; nested mask/look data noted, not parsed."""
    root = ET.parse(path).getroot()
    desc = root.find(f".//{{{RDF}}}Description")
    out = {}
    for k, v in desc.attrib.items():
        if k.startswith("{" + CRS + "}"):
            out[k.split("}")[1]] = v
    for child in desc:
        if not child.tag.startswith("{" + CRS + "}"):
            continue
        name = child.tag.split("}")[1]
        seq = child.find(f"{{{RDF}}}Seq")
        if name.startswith("ToneCurve") and seq is not None:
            out[name] = [li.text for li in seq.findall(f"{{{RDF}}}li")]
        else:
            out.setdefault("_nested", []).append(name)
    return out


def originals_index():
    """(capture-date-dir, fileName) and fileName -> original path on disk."""
    idx = {}
    for root, _dirs, files in os.walk(f"{LR}/originals"):
        day = os.path.basename(root)
        for f in files:
            idx[(day, f)] = os.path.join(root, f)
            idx.setdefault(f, os.path.join(root, f))
    return idx


def mine(picks_only=True):
    originals = originals_index()
    db = sqlite3.connect(os.path.join(os.path.dirname(__file__), "catalog.mcat"))
    db.text_factory = bytes
    rows = db.execute(
        "SELECT d.fullDocId, r.content FROM docs d JOIN revs r ON r.sequence=d.winningRevSequence "
        "WHERE d.type='asset' AND d.subtype='image' AND d.deleted=0")
    kept = skipped_flag = no_settings = 0
    os.makedirs("dataset", exist_ok=True)
    with open("dataset/index.jsonl", "w") as f:
        for fid, content in rows:
            a = msgpack.unpackb(content, raw=False)
            flags = [r.get("flag") for r in a.get("reviews", {}).values()]
            if picks_only and "pick" not in flags:
                skipped_flag += 1
                continue
            xcr = (a.get("develop") or {}).get("xmpCameraRaw")
            sha = xcr.get("sha256") if isinstance(xcr, dict) else xcr
            spath = sha and f"{LR}/settings/{sha}"
            if not sha or not os.path.exists(spath):
                no_settings += 1
                continue
            imp = a.get("importSource", {})
            xmp_tiff = a.get("xmp", {}).get("tiff", {})
            rec = {
                "asset_id": fid.decode(),
                "file_name": imp.get("fileName"),
                "capture_date": a.get("captureDate"),
                "camera": f'{xmp_tiff.get("Make", "?")} {xmp_tiff.get("Model", "?")}'.strip(),
                "flag": "pick" if "pick" in flags else (flags[0] if flags else None),
                "settings_sha": sha,
                "settings_path": spath,
                "original": originals.get(((a.get("captureDate") or "")[:10], imp.get("fileName")))
                            or originals.get(imp.get("fileName")),
                "settings": parse_settings(spath),
            }
            f.write(json.dumps(rec) + "\n")
            kept += 1
    print(f"kept={kept} no_pick={skipped_flag} no_settings={no_settings}")


def demo():
    """Self-check: parse the known test image's settings and assert key sliders."""
    s = parse_settings(
        f"{LR}/settings/237684c9b2d31861f5be2b874a3f46a6db6ca12bf670d7b53b1509cdb37923b1")
    assert s["Exposure2012"] == "+2.04", s["Exposure2012"]
    assert s["Temperature"] == "4172"
    assert s["ProcessVersion"] == "15.4"
    assert "ToneCurvePV2012" not in s or isinstance(s.get("ToneCurvePV2012"), list)
    print("demo ok:", {k: s[k] for k in ("Exposure2012", "Temperature", "Contrast2012")})


if __name__ == "__main__":
    demo()
    mine(picks_only="--all" not in sys.argv)
