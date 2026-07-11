"""Build dataset from a Lightroom 'Export Original & Settings' folder.

Raws carry a .xmp sidecar; JPEG/DNG originals embed the XMP packet in-file.
Normalizes each crs Description into settings_norm/ (library-blob shape) so
predict.py treats every data source identically, and so a neighbor's exif
never leaks into a generated sidecar.

Usage: python3 load_export.py ~/Pictures/agent-export
"""
import glob, json, os, re, sys

from mine import parse_settings, CRS, RDF

try:
    import defusedxml.ElementTree as DET
    from xml.etree import ElementTree as ET  # register_namespace/tostring
except ImportError:
    import xml.etree.ElementTree as ET
    DET = ET

X = "adobe:ns:meta/"
RAW_EXTS = {".nef", ".nrw", ".cr2", ".cr3", ".arw", ".raf", ".rw2", ".orf"}
EMBED_EXTS = {".jpg", ".jpeg", ".dng"}

ET.register_namespace("x", X)
ET.register_namespace("rdf", RDF)
ET.register_namespace("crs", CRS)


def xmp_text(path):
    """Settings XMP text for an original: sidecar if present, else embedded packet."""
    sidecar = os.path.splitext(path)[0] + ".xmp"
    if os.path.exists(sidecar):
        return open(sidecar, errors="ignore").read()
    data = open(path, "rb").read()
    start = data.find(b"<x:xmpmeta")
    end = data.find(b"</x:xmpmeta>")
    if start == -1 or end == -1:
        return None
    return data[start:end + len(b"</x:xmpmeta>")].decode("utf-8", errors="ignore")


def normalize(text, out_path):
    root = DET.fromstring(text)
    for desc in root.iter(f"{{{RDF}}}Description"):
        if any(k.startswith("{" + CRS + "}") for k in desc.attrib):
            for k in [k for k in desc.attrib
                      if not k.startswith("{" + CRS + "}") and not k.startswith("{" + RDF + "}")]:
                del desc.attrib[k]
            body = ET.tostring(desc, encoding="unicode")
            open(out_path, "w").write(
                f'<x:xmpmeta xmlns:x="{X}">\n <rdf:RDF xmlns:rdf="{RDF}">\n'
                f"  {body}\n </rdf:RDF>\n</x:xmpmeta>\n")
            return True
    return False


def attr(text, name):
    m = re.search(f'{name}="([^"]*)"', text)
    return m.group(1) if m else None


def load(folder):
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("settings_norm", exist_ok=True)
    stems_with_raw = {os.path.splitext(p)[0] for p in glob.glob(os.path.join(folder, "*"))
                      if os.path.splitext(p)[1].lower() in RAW_EXTS}
    kept = skipped_dup = no_xmp = no_crs = 0
    with open("dataset/index.jsonl", "w") as f:
        for path in sorted(glob.glob(os.path.join(folder, "*"))):
            stem, ext = os.path.splitext(path)
            ext = ext.lower()
            if ext not in RAW_EXTS | EMBED_EXTS:
                continue
            if ext in EMBED_EXTS and stem in stems_with_raw:
                skipped_dup += 1  # RAW+JPEG shot: raw wins
                continue
            text = xmp_text(path)
            if not text:
                no_xmp += 1
                continue
            norm = os.path.abspath(f"settings_norm/{os.path.basename(stem)}.xmp")
            if not normalize(text, norm):
                no_crs += 1
                continue
            look_m = re.search(r'<crs:Look>.*?crs:Name="([^"]*)"', text, re.S)
            rec = {
                "asset_id": os.path.basename(stem),
                "look": look_m.group(1) if look_m else None,
                "file_name": os.path.basename(path),
                "capture_date": attr(text, "exif:DateTimeOriginal") or attr(text, "xmp:CreateDate"),
                "camera": f'{attr(text, "tiff:Make") or ""} {attr(text, "tiff:Model") or ""}'.strip(),
                "flag": "pick",
                "is_raw": ext in RAW_EXTS,
                "settings_path": norm,
                "original": os.path.abspath(path),
                "settings": parse_settings(norm),
            }
            f.write(json.dumps(rec) + "\n")
            kept += 1
    print(f"kept={kept} raw_jpeg_dups_skipped={skipped_dup} no_xmp={no_xmp} no_crs={no_crs}")


if __name__ == "__main__":
    load(os.path.expanduser(sys.argv[1]))
