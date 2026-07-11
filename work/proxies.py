"""Extract 640px JPEG proxies from raw originals (embedded thumb, no raw decode).

Usage: python3 proxies.py [dataset/index.jsonl]
Output: proxies/<asset_id>.jpg
"""
import io, json, os, sys

import rawpy
from PIL import Image, ImageOps


def make_proxy(raw_path, out_path, long_side=640):
    if raw_path.lower().endswith((".jpg", ".jpeg")):
        img = ImageOps.exif_transpose(Image.open(raw_path))
        img.thumbnail((long_side, long_side))
        img.convert("RGB").save(out_path, "JPEG", quality=90)
        return
    with rawpy.imread(raw_path) as r:
        th = r.extract_thumb()
    if th.format == rawpy.ThumbFormat.JPEG:
        img = Image.open(io.BytesIO(th.data))
    else:
        img = Image.fromarray(th.data)
    img = ImageOps.exif_transpose(img)
    img.thumbnail((long_side, long_side))
    img.convert("RGB").save(out_path, "JPEG", quality=90)


if __name__ == "__main__":
    index = sys.argv[1] if len(sys.argv) > 1 else "dataset/index.jsonl"
    os.makedirs("proxies", exist_ok=True)
    done = missing = 0
    for line in open(index):
        rec = json.loads(line)
        if not rec.get("original") or not os.path.exists(rec["original"]):
            missing += 1
            continue
        out = f"proxies/{rec['asset_id']}.jpg"
        if not os.path.exists(out):
            make_proxy(rec["original"], out)
        done += 1
    print(f"proxies={done} missing_original={missing}")
