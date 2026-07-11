import json, sqlite3, msgpack

db = sqlite3.connect("catalog.mcat")
db.text_factory = bytes

def doc(type_, subtype, limit=1):
    rows = db.execute(
        "SELECT d.fullDocId, r.content FROM docs d JOIN revs r ON r.sequence=d.winningRevSequence "
        "WHERE d.type=? AND d.subtype=? AND d.deleted=0 LIMIT ?", (type_, subtype, limit)).fetchall()
    return [(fid, msgpack.unpackb(c, raw=False)) for fid, c in rows]

fid, asset = doc("asset", "image")[0]
print("ASSET fullDocId:", fid)
print(json.dumps(asset, indent=1, default=str)[:2500])
print("\n" + "=" * 60)
fid, edit = doc("version", "edit")[0]
print("EDIT fullDocId:", fid)
print(json.dumps(edit, indent=1, default=str)[:2500])
