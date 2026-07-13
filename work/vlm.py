"""Optional VLM sanity pass (PLAN.md §Optional VLM pass), local on Apple Silicon.

Qwen3-VL-8B-Instruct-4bit via mlx-vlm looks at a proxy and returns exposure/WB
nudges — a cold-start / sanity check that rides on top of Engine A or B, useful
when retrieval neighbors are far or the trained head is uncertain. ~5-6 GB, runs
in a few hundred ms/image on an M-series Mac; nothing leaves the machine.

    python3 vlm.py photo.NEF        # assess one image
"""
import json, os, re, sys

MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
_CACHE = {}
PROMPT = (
    "You are a wedding photo editor's assistant reviewing an edited frame for "
    "objective mistakes only. The photographer's stylistic grading (fading, "
    "desaturation, warmth, film looks) is deliberate — do NOT correct style. "
    "Flag only clear errors: a subject too dark/bright to deliver, or an "
    "unmistakable white-balance mistake (green/magenta skin, wrong scene color). "
    "When in doubt, return zeros. Reply with ONLY a JSON object, no prose:\n"
    '{"exposure_ev": <float stops, + brighten / - darken, range -2..2>, '
    '"temp_shift": <int Kelvin nudge, + warmer / - cooler, range -1500..1500>, '
    '"tint_shift": <int, + magenta / - green, range -20..20>, '
    '"reason": "<8 words max>"}'
)


def _load():
    if "m" not in _CACHE:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        model, processor = load(MODEL)
        _CACHE.update(m=model, p=processor, cfg=load_config(MODEL))
    return _CACHE["m"], _CACHE["p"], _CACHE["cfg"]


def _parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k, lo, hi, cast in (("exposure_ev", -2, 2, float), ("temp_shift", -1500, 1500, int),
                            ("tint_shift", -20, 20, int)):
        if k in d:
            try:
                out[k] = max(lo, min(hi, cast(d[k])))
            except (ValueError, TypeError):
                pass
    out["reason"] = str(d.get("reason", ""))[:60]
    return out


def assess(proxy_path, refs=()):
    """Image -> {exposure_ev, temp_shift, tint_shift, reason} (clamped) or None.
    refs: optional expert-edited exemplar images shown BEFORE the candidate —
    grounds 'well edited' in real references instead of the model's generic taste."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, cfg = _load()
    prompt = PROMPT
    if refs:
        prompt = (f"The first {len(refs)} image(s) are expert-edited reference "
                  "photos — the quality bar for exposure and color. The LAST "
                  "image is the photo to review.\n" + PROMPT)
    # multi-image prompts choke above ~640px (vision-token budget: 3x1024 imgs
    # returned empty output) — downscale working copies before generation
    from PIL import Image
    os.makedirs("tmp_proxies", exist_ok=True)
    imgs = []
    for i, p in enumerate(list(refs) + [proxy_path]):
        im = Image.open(p)
        if max(im.size) > 640:
            im.thumbnail((640, 640))
            sp = f"tmp_proxies/_vlm{i}.jpg"
            im.convert("RGB").save(sp, quality=88)
            imgs.append(sp)
        else:
            imgs.append(p)
    formatted = apply_chat_template(processor, cfg, prompt, num_images=len(imgs))
    # greedy (temp 0) collapses into "exexex..."; sampling fixes it but can still
    # occasionally degenerate — retry a couple times and take the first valid JSON.
    for _ in range(3):
        out = generate(model, processor, formatted, imgs, max_tokens=120,
                       temperature=0.3, repetition_penalty=1.15, verbose=False)
        d = _parse(out.text if hasattr(out, "text") else out)
        if d is not None:
            return d
    return None


# second pass: corrections below these thresholds are noise, not fixes
DEADBAND = {"exposure_ev": 0.3, "temp_shift": 300, "tint_shift": 5}


def _set_attr(text, key, value, fmt):
    txt = fmt(value)
    text, n = re.subn(f'crs:{key}="[^"]*"', f'crs:{key}="{txt}"', text)
    if n == 0:
        text = text.replace('crs:ProcessVersion=',
                            f'crs:{key}="{txt}"\n   crs:ProcessVersion=', 1)
    return text


def audit_sidecar(xmp_path, render_path, refs=()):
    """Second pass: assess a rendered edit, fold significant corrections back
    into the sidecar (in place — exported hardlinks share the inode). Temp/Tint
    only when WB is Custom (else LR honors the preset, not Kelvin). Returns the
    applied changes ({} = nothing significant)."""
    d = assess(render_path, refs=refs)
    if not d:
        return {}
    text = open(xmp_path).read()
    fixes = {}
    ev = d.get("exposure_ev", 0.0)
    if abs(ev) >= DEADBAND["exposure_ev"]:
        m = re.search(r'crs:Exposure2012="([^"]*)"', text)
        new = max(-5.0, min(5.0, (float(m.group(1)) if m else 0.0) + ev))
        text = _set_attr(text, "Exposure2012", new, lambda v: f"{v:+.2f}")
        fixes["Exposure2012"] = round(new, 2)
    if 'crs:WhiteBalance="Custom"' in text:
        ts = d.get("temp_shift", 0)
        m = re.search(r'crs:Temperature="([^"]*)"', text)
        if m and abs(ts) >= DEADBAND["temp_shift"]:
            new = int(max(2000, min(50000, float(m.group(1)) + ts)))
            text = _set_attr(text, "Temperature", new, lambda v: f"{v}")
            fixes["Temperature"] = new
        tn = d.get("tint_shift", 0)
        m = re.search(r'crs:Tint="([^"]*)"', text)
        if m and abs(tn) >= DEADBAND["tint_shift"]:
            new = int(max(-150, min(150, float(m.group(1)) + tn)))
            text = _set_attr(text, "Tint", new, lambda v: f"{v:+d}")
            fixes["Tint"] = new
    if fixes:
        open(xmp_path, "w").write(text)
        fixes["reason"] = d.get("reason", "")
    return fixes


def _demo():
    """Round-trips the JSON contract + audit rewrite without loading the model."""
    ex = '{"exposure_ev": 3.5, "temp_shift": -9000, "tint_shift": 2, "reason": "slightly dark, cool cast"}'
    d = _parse("noise before " + ex + " noise after")
    assert d["exposure_ev"] == 2.0 and d["temp_shift"] == -1500  # clamped
    assert d["tint_shift"] == 2 and d["reason"].startswith("slightly")
    assert _parse("no json here") is None

    import tempfile
    global assess
    real = assess
    xmp = tempfile.NamedTemporaryFile("w", suffix=".xmp", delete=False)
    xmp.write('<x crs:WhiteBalance="Custom"\n   crs:Exposure2012="+1.50"\n'
              '   crs:Temperature="5000"\n   crs:Tint="+10"\n   crs:ProcessVersion="15.4"/>')
    xmp.close()
    try:
        assess = lambda p, **kw: {"exposure_ev": 0.5, "temp_shift": 400, "tint_shift": 2,
                            "reason": "test"}
        fixes = audit_sidecar(xmp.name, "unused.jpg")
        assert fixes["Exposure2012"] == 2.0 and fixes["Temperature"] == 5400, fixes
        assert "Tint" not in fixes  # +2 below deadband
        t = open(xmp.name).read()
        assert 'crs:Exposure2012="+2.00"' in t and 'crs:Temperature="5400"' in t
        assess = lambda p, **kw: {"exposure_ev": 0.1, "temp_shift": 50, "tint_shift": 0,
                            "reason": ""}
        assert audit_sidecar(xmp.name, "unused.jpg") == {}  # all under deadband
    finally:
        assess = real
        os.unlink(xmp.name)
    print("vlm._parse + audit_sidecar contracts OK")


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != "work":
        os.chdir(os.path.dirname(__file__))
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args or "--demo" in sys.argv:
        _demo()
    else:
        from proxies import make_proxy
        p = args[0]
        proxy = p
        if not p.lower().endswith((".jpg", ".jpeg", ".png")):
            os.makedirs("tmp_proxies", exist_ok=True)
            proxy = f"tmp_proxies/{os.path.basename(p)}.jpg"
            make_proxy(p, proxy)
        print(json.dumps(assess(proxy), indent=2))
