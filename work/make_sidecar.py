"""Build test sidecars from a settings XMP.

test2: parametric masks kept, Mask/Image (AI) corrections stripped.
"""
import os, re

LR = os.path.expanduser(
    "~/Pictures/Lightroom Library.lrlibrary/8d15b67bae9846e39beea23257a7e871")
SHA = "237684c9b2d31861f5be2b874a3f46a6db6ca12bf670d7b53b1509cdb37923b1"

xml = open(f"{LR}/settings/{SHA}").read()

# Drop every whole <rdf:li>…</rdf:li> correction that contains a Mask/Image mask.
# ponytail: regex over rdf:li blocks, fine while corrections aren't nested; swap to
# an XML rewrite if Adobe ever nests li inside correction li.
lis = re.findall(r"     <rdf:li>.*?     </rdf:li>\n", xml, flags=re.S)
assert lis, "no correction blocks matched"
for li in lis:
    if 'crs:What="Mask/Image"' in li:
        xml = xml.replace(li, "")
assert 'Mask/Image' not in xml

out = ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
       + xml + '\n<?xpacket end="w"?>')
dst = os.path.join(os.path.dirname(__file__), "..", "test-import")
open(f"{dst}/WIL_6299_test2.xmp", "w").write(out)
print("wrote WIL_6299_test2.xmp; masks kept:",
      re.findall(r'crs:What="(Mask/[A-Za-z]+)"', xml))
