"""Synthesize a .acr sidecar (mask raster companion) from the library's auxiliary store.

Format (reverse-engineered from LR cloud 'Export Original & Settings'):
  'ACR\0' u32=1  <ext>\0 u32=1  u32=0  digest[16]  u32 size  u32=0  u32 offset=52  u32=0
  <auxiliary blob verbatim>  u16=0
ponytail: single-mask format proven byte-identical; multi-mask layout unknown —
export a multi-AI-mask image the same way and diff when needed.
"""
import os, re, struct, sys

LR = os.path.expanduser(
    "~/Pictures/Lightroom Library.lrlibrary/8d15b67bae9846e39beea23257a7e871")


def make_acr(settings_xmp_text, ext=b"NEF"):
    digests = re.findall(r'crs:MaskDigest="([0-9A-F]{32})"', settings_xmp_text)
    if not digests:
        return None
    assert len(digests) == 1, f"multi-mask acr layout unverified: {digests}"
    d = digests[0]
    aux = open(f"{LR}/auxiliary/{d[:3]}/{d[3:]}", "rb").read()
    hdr = (b"ACR\0" + struct.pack("<I", 1) + ext + b"\0" + struct.pack("<II", 1, 0)
           + bytes.fromhex(d) + struct.pack("<IIII", len(aux), 0, 52, 0))
    assert len(hdr) == 52
    return hdr + aux + b"\0\0"


if __name__ == "__main__":
    sha = "237684c9b2d31861f5be2b874a3f46a6db6ca12bf670d7b53b1509cdb37923b1"
    xmp = open(f"{LR}/settings/{sha}").read()
    blob = make_acr(xmp)
    ref = open(os.path.join(os.path.dirname(__file__), "..", "WIL_6299.acr"), "rb").read()
    print("synthesized == Adobe export:", blob == ref, f"({len(blob)} bytes)")
    out = os.path.join(os.path.dirname(__file__), "..", "test-import")
    open(f"{out}/WIL_6299_test4.acr", "wb").write(blob)
