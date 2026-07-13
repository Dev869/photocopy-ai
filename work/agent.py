"""Watch-folder daemon: new raws in -> XMP sidecars (+ cull report) out.

IPC with the menu-bar shell is two JSON files in this directory:
  config.json (shell writes): {"watch_dir", "look", "engine", "cull_threshold", "paused"}
                              engine: "a" (retrieval) | "b" (trained head)
  state.json  (daemon writes): {"status", "done", "total", "last_file",
                                "watch_dir", "looks", "updated"}
Run: .venv/bin/python agent.py
"""
import glob, json, os, threading, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config.json")
STATE = os.path.join(HERE, "state.json")
LOG = os.path.join(HERE, "agent.log")
RAW_EXTS = (".nef", ".nrw", ".cr2", ".cr3", ".arw", ".raf", ".rw2", ".orf", ".dng",
            ".pef", ".srw", ".3fr", ".fff", ".iiq")
JPEG_EXTS = (".jpg", ".jpeg")
SCAN_EXTS = RAW_EXTS + JPEG_EXTS
# ponytail: 2s polling, not FSEvents — one card dump per wedding, latency is irrelevant

EVENTS = os.path.join(HERE, "events.jsonl")


def event(kind, text):
    """User-facing activity feed (not the debug log)."""
    with open(EVENTS, "a") as f:
        f.write(json.dumps({"ts": time.time(), "kind": kind, "text": text}) + "\n")
    if os.path.getsize(EVENTS) > 500_000:  # keep the tail, drop ancient history
        lines = open(EVENTS).readlines()[-500:]
        open(EVENTS, "w").writelines(lines)


def notify(message):
    import subprocess
    safe = message.replace('"', "'")
    subprocess.run(["osascript", "-e",
                    f'display notification "{safe}" with title "Photocopy"'], check=False)


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def read_config():
    try:
        return json.load(open(CONFIG))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


_state_lock = threading.Lock()
_last_state = {}


def _write_state_locked(kw):
    payload = dict(kw, updated=time.time())
    json.dump(payload, open(STATE + ".tmp", "w"))
    os.replace(STATE + ".tmp", STATE)


def write_state(**kw):
    with _state_lock:
        _last_state.clear()
        _last_state.update(kw)
        _write_state_locked(kw)


def heartbeat():
    """Refresh state.json's timestamp every 2s so the shell's liveness check
    (state fresh within 8s) can't mistake a busy daemon — model load, a slow
    file — for a dead one, which hid the progress bar and spawned duplicates."""
    while True:
        time.sleep(2)
        with _state_lock:
            if _last_state:
                _write_state_locked(_last_state)


def available_looks():
    from collections import Counter
    c = Counter(r.get("look") for r in map(json.loads, open(os.path.join(HERE, "dataset/index.jsonl")))
                if r.get("is_raw", True) and r.get("look"))
    looks = [name for name, n in c.most_common() if n >= 25]
    if os.path.exists(os.path.join(HERE, "data/ppr10k/eng/index.jsonl")):
        looks.append("PPR10K")  # external expert style: Engine B token, no retrieval history
    return looks


def pending_raws(watch_dir, exclude=None):
    """Recursive: card dumps and wedding folders nest. Skips hidden dirs and the
    export tree. Returns (pending, n_edited): raws lacking a sidecar, and the
    count that already have one — so the shell can say "already edited" instead
    of looking dead when there's nothing to do."""
    out, edited = [], 0
    ex = os.path.realpath(os.path.expanduser(exclude)) if exclude else None
    for root, dirs, files in os.walk(watch_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if ex and os.path.realpath(root).startswith(ex):
            dirs[:] = []
            continue
        for f in sorted(files):
            p = os.path.join(root, f)
            if f.lower().endswith(SCAN_EXTS):
                if os.path.exists(os.path.splitext(p)[0] + ".xmp"):
                    edited += 1
                else:
                    out.append(p)
    return sorted(out), edited


def place(src, dst_dir):
    """hardlink (same volume) or copy a file into dst_dir."""
    import shutil
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def embed_xmp_jpeg(src, xmp_text, dst_dir):
    """Copy a JPEG with the develop XMP embedded as an APP1 segment.

    LR only reads sidecars for raw files; rendered files need embedded XMP.
    Replaces any existing XMP APP1. Original file is never touched.
    """
    data = open(src, "rb").read()
    assert data[:2] == b"\xff\xd8", "not a JPEG"
    # walk segments to find insert point (after EXIF APP1/APP0) and strip old XMP
    i, insert_at, out = 2, 2, [data[:2]]
    while i + 4 <= len(data) and data[i] == 0xFF and data[i + 1] not in (0xDA, 0xD9):
        marker, seglen = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        seg = data[i:i + 2 + seglen]
        if marker == 0xE1 and seg[4:4 + len(XMP_HEADER)] == XMP_HEADER:
            pass  # drop stale XMP segment
        else:
            out.append(seg)
        i += 2 + seglen
    out.append(data[i:])
    payload = XMP_HEADER + xmp_text.encode()
    seg = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    dst = os.path.join(os.path.expanduser(dst_dir), os.path.basename(src))
    # insert after the last APP segment we kept (before entropy data)
    head = b"".join(out[:-1])
    open(dst, "wb").write(head + seg + out[-1])


THUMBS = os.path.join(HERE, "thumbs")


def render_thumbs(paths):
    import shutil
    from mine import parse_settings
    from preview import look_from_xmp, render
    shutil.rmtree(THUMBS, ignore_errors=True)
    os.makedirs(THUMBS, exist_ok=True)
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        proxy = os.path.join(HERE, "tmp_proxies", os.path.basename(p) + ".jpg")
        if not os.path.exists(proxy) and p.lower().endswith(JPEG_EXTS):
            proxy = p  # jpgs are their own proxy
        xmp = os.path.splitext(p)[0] + ".xmp"
        if not (os.path.exists(proxy) and os.path.exists(xmp)):
            continue
        try:
            render(proxy, parse_settings(xmp), os.path.join(THUMBS, stem + ".jpg"),
                   look=look_from_xmp(xmp))
        except Exception:
            pass


def default_export_dir(watch_dir):
    return watch_dir.rstrip("/") + "-edited"


def process_wave(paths, cfg, looks):
    import cull, predict

    def phase(text, done=0):
        write_state(status="processing", done=done, total=len(paths), last_file=text,
                    watch_dir=cfg.get("watch_dir", ""), looks=looks)

    def progress(i, total):
        phase(os.path.basename(paths[min(i, len(paths) - 1)]), done=i)

    def written(p):
        event("photo", f"Edited {os.path.basename(p)}")
        phase("writing edits…", done=len(paths))

    event("start", f"Started editing {len(paths)} photos")
    phase("warming up…")
    do_edit = cfg.get("edit", True)
    if do_edit:
        predict.main(paths, look=cfg.get("look"), engine=cfg.get("engine", "a"),
                     progress=progress, on_written=written)

    thr, target = cfg.get("cull_threshold"), cfg.get("cull_target")
    if not cfg.get("cull", thr is not None or bool(target)):
        thr = target = None
    keep = set(paths)
    if thr is not None or target:
        phase("culling…", done=len(paths))
        probs = cull.score_paths(paths)
        ranked = sorted(zip(paths, probs), key=lambda t: -t[1])
        if target:  # best-N per import; target wins over threshold
            keep = {p for p, _ in ranked[:int(target)]}
        else:
            keep = {p for p, pr in ranked if pr >= thr}
        out_dir = os.path.expanduser(cfg.get("export_dir") or default_export_dir(cfg["watch_dir"]))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "cull_report.tsv"), "w") as f:
            for p, pr in ranked:
                f.write(f"{pr:.3f}\t{'keep' if p in keep else 'skip'}\t{os.path.basename(p)}\n")
        log(f"cull: keeping {len(keep)}/{len(paths)}"
            + (f" (best {target})" if target else f" @ {thr}"))
        event("cull", f"Culled: kept the best {len(keep)} of {len(paths)}")

    if do_edit:
        phase("rendering previews…", done=len(paths))
        render_thumbs(sorted(keep))

    if do_edit and cfg.get("audit"):
        import random
        import vlm
        from mine import parse_settings
        from preview import look_from_xmp, render_raw_jpeg
        os.makedirs(THUMBS, exist_ok=True)
        # expert-edited exemplars (PPR10K, research wedding/portrait set) ground
        # the auditor's idea of "well edited"; without them it judges zero-shot
        ref_pool = sorted(glob.glob(os.path.join(
            HERE, "data/ppr10k/global/*/processed.jpg")))[:400]
        event("start", f"Audit pass: reviewing {len(keep)} edits"
              + (f" against {min(2, len(ref_pool))} expert references" if ref_pool else ""))
        n_fixed = 0
        for i, p in enumerate(sorted(keep)):
            xmp = os.path.splitext(p)[0] + ".xmp"
            if not os.path.exists(xmp):
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            phase(f"auditing {stem}…", done=i)
            arender = os.path.join(THUMBS, stem + ".jpg")
            try:
                # accurate linear-path render doubles as the preview thumb
                render_raw_jpeg(p, parse_settings(xmp), arender, quality=85,
                                long_side=1024, look=look_from_xmp(xmp))
                refs = (random.Random(stem).sample(ref_pool, 2)
                        if len(ref_pool) >= 2 else ())
                fixes = vlm.audit_sidecar(xmp, arender, refs=refs)
                if fixes:
                    reason = fixes.pop("reason", "")
                    render_raw_jpeg(p, parse_settings(xmp), arender, quality=85,
                                    long_side=1024)  # re-render with the fix
                    n_fixed += 1
                    event("audit", f"Audit fixed {stem}: "
                          + ", ".join(f"{k} {v}" for k, v in fixes.items())
                          + (f" — {reason}" if reason else ""))
            except Exception as e:
                log(f"audit skip {stem}: {type(e).__name__}: {e}")
        event("done", f"Audit done: {n_fixed} of {len(keep)} edits adjusted")

    phase("exporting…", done=len(paths))
    out_dir = os.path.expanduser(cfg.get("export_dir") or default_export_dir(cfg["watch_dir"]))
    os.makedirs(out_dir, exist_ok=True)
    fmt = cfg.get("export_format", "raw")   # "raw" (raw+sidecar, for LR) | "jpeg" (proof render)
    quality = int(cfg.get("jpeg_quality", 90))
    if fmt == "jpeg":
        from preview import look_from_xmp, render_raw_jpeg
        from mine import parse_settings
    for p in sorted(keep):
        xmp = os.path.splitext(p)[0] + ".xmp"
        if fmt == "jpeg" and os.path.exists(xmp):
            out = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".jpg")
            render_raw_jpeg(p, parse_settings(xmp), out, quality,
                            look=look_from_xmp(xmp))
        elif p.lower().endswith(JPEG_EXTS) and os.path.exists(xmp):
            embed_xmp_jpeg(p, open(xmp).read(), out_dir)
        else:
            place(p, out_dir)
            if os.path.exists(xmp):
                place(xmp, out_dir)
    log(f"exported {len(keep)} {'proof JPEGs' if fmt == 'jpeg' else 'raws + sidecars'}"
        f" -> {out_dir}")
    event("done", f"Finished — {len(keep)} of {len(paths)} photos exported to "
          + os.path.basename(out_dir))
    notify(f"Done: {len(keep)} of {len(paths)} photos edited and exported"
           + (" — opening in Lightroom" if cfg.get("send_to_lightroom") else ""))

    if cfg.get("send_to_lightroom"):
        import subprocess
        files = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                       if f.lower().endswith(SCAN_EXTS))
        if files:
            subprocess.run(["open", "-a", "Adobe Lightroom"] + files, check=False)
            log(f"sent {len(files)} raws to Lightroom")


def run():
    os.chdir(HERE)
    looks = available_looks()
    threading.Thread(target=heartbeat, daemon=True).start()
    log("daemon up")
    prev_pending = []
    while True:
        cfg = read_config()
        wd = os.path.expanduser(cfg.get("watch_dir") or "")
        if cfg.get("paused") or not wd or not os.path.isdir(wd):
            write_state(status="paused" if cfg.get("paused") else "no-folder",
                        done=0, total=0, last_file="", watch_dir=wd, looks=looks)
            prev_pending = []
            time.sleep(2)
            continue
        pending, edited = pending_raws(wd, exclude=cfg.get("export_dir") or default_export_dir(wd))
        if pending and pending == prev_pending:  # stable across one poll = copy finished
            log(f"wave: {len(pending)} raws in {wd}")
            try:
                process_wave(pending, {**cfg, "watch_dir": wd}, looks)
                log(f"wave done: {len(pending)} sidecars")
            except Exception:
                log("wave FAILED:\n" + traceback.format_exc())
                write_state(status="error", done=0, total=0,
                            last_file="see agent.log", watch_dir=wd, looks=looks)
                time.sleep(10)
            prev_pending = []
        else:
            write_state(status="watching", done=0, total=len(pending), edited=edited,
                        last_file="", watch_dir=wd, looks=looks)
            prev_pending = pending
        time.sleep(2)


if __name__ == "__main__":
    run()
