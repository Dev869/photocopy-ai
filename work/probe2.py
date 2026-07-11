import json, sqlite3, msgpack, os

db = sqlite3.connect("catalog.mcat")
db.text_factory = bytes
LR = os.path.expanduser(
    "~/Pictures/Lightroom Library.lrlibrary/8d15b67bae9846e39beea23257a7e871")

def get(full_doc_id):
    row = db.execute(
        "SELECT r.content FROM docs d JOIN revs r ON r.sequence=d.winningRevSequence "
        "WHERE d.fullDocId=?", (full_doc_id,)).fetchone()
    return msgpack.unpackb(row[0], raw=False) if row else None

asset = get("7b14a8473e784fcd97b8da5f2e69949f")
slim = {k: v for k, v in asset.items() if k != "aux"}
print(json.dumps(slim, indent=1, default=str)[:3500])

sha = "237684c9b2d31861f5be2b874a3f46a6db6ca12bf670d7b53b1509cdb37923b1"
p = f"{LR}/settings/{sha}"
print("\nsettings file exists:", os.path.exists(p), "size:", os.path.getsize(p) if os.path.exists(p) else 0)
