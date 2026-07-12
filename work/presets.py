"""Preset registry: the discrete looks the engine can condition on.

A "preset" = a named look. Devin's history is already discrete via crs:Look
(Artistic 05, Modern 08, ...). Engine B learns one style token per preset, so
adding a preset is just another token — no retrain of the encoder.

External experts (MIT-Adobe FiveK's 5, PPR10K's 3) register here as more presets
via register_external(); their (embedding, settings, preset) rows train the same
head under their own tokens. That is how "load in more presets" scales past
Devin's own looks. See load_export.py for the ingestion side.
"""
import json, os
from collections import Counter

DATASET = "dataset/index.jsonl"
UNKNOWN = "Unknown"          # slot 0: records with no crs:Look
_EXTERNAL = []               # [(name, min_count)] registered public-data presets


def register_external(name):
    """Reserve a preset slot for a public-data expert (e.g. 'FiveK/C')."""
    if name not in _EXTERNAL:
        _EXTERNAL.append(name)


def look_counts(path=DATASET, raw_only=True):
    c = Counter()
    for line in open(path):
        r = json.loads(line)
        if raw_only and not r.get("is_raw", True):
            continue
        c[r.get("look") or UNKNOWN] += 1
    return c


def registry(path=DATASET, min_count=3):
    """Ordered preset list. Slot 0 is always UNKNOWN so unseen looks map safely.
    Looks below min_count fold into UNKNOWN (too few to learn a token)."""
    counts = look_counts(path)
    keep = sorted((n for n, k in counts.items() if k >= min_count and n != UNKNOWN),
                  key=lambda n: -counts[n])
    names = [UNKNOWN] + keep + list(_EXTERNAL)
    return names, counts


def look2idx(path=DATASET, min_count=3):
    names, _ = registry(path, min_count)
    idx = {n: i for i, n in enumerate(names)}
    return idx, names


def encode(look, idx):
    """Map a look string (or None) to its preset slot, folding unseen -> UNKNOWN."""
    return idx.get(look or UNKNOWN, 0)


def _demo():
    idx, names = look2idx()
    assert names[0] == UNKNOWN
    assert encode(None, idx) == 0 and encode("no-such-look", idx) == 0
    assert len(set(idx.values())) == len(names)  # unique slots
    c = look_counts()
    for n in names[1:]:
        assert c.get(n, 0) >= 3 or n in _EXTERNAL
    print(f"{len(names)} presets:",
          ", ".join(f"{n}({c.get(n,0)})" for n, c in [(x, c) for x in names]))


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != "work":
        os.chdir(os.path.join(os.path.dirname(__file__)))
    _demo()
