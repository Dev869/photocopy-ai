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
    "You are a wedding photo editor judging a straight-out-of-camera frame. "
    "Assess only global exposure and white balance for a natural, flattering edit. "
    "Reply with ONLY a JSON object, no prose:\n"
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


def assess(proxy_path):
    """Proxy image -> {exposure_ev, temp_shift, tint_shift, reason} (clamped) or None."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, cfg = _load()
    formatted = apply_chat_template(processor, cfg, PROMPT, num_images=1)
    # greedy (temp 0) collapses into "exexex..."; sampling fixes it but can still
    # occasionally degenerate — retry a couple times and take the first valid JSON.
    for _ in range(3):
        out = generate(model, processor, formatted, proxy_path, max_tokens=120,
                       temperature=0.3, repetition_penalty=1.15, verbose=False)
        d = _parse(out.text if hasattr(out, "text") else out)
        if d is not None:
            return d
    return None


def _demo():
    """Round-trips the JSON contract without loading the model."""
    ex = '{"exposure_ev": 3.5, "temp_shift": -9000, "tint_shift": 2, "reason": "slightly dark, cool cast"}'
    d = _parse("noise before " + ex + " noise after")
    assert d["exposure_ev"] == 2.0 and d["temp_shift"] == -1500  # clamped
    assert d["tint_shift"] == 2 and d["reason"].startswith("slightly")
    assert _parse("no json here") is None
    print("vlm._parse contract OK")


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
