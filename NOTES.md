# Session 1 findings (2026-07-11)

## Big deviation from the master plan: no `.lrcat`, and that's good news
Devin uses **Lightroom (cloud)**, not Classic. Catalog lives at
`~/Pictures/Lightroom Library.lrlibrary/8d15b67bae9846e39beea23257a7e871/`.

Consequence: **the Lua-blob parser (plan's #1 risk) is unnecessary.** Develop
settings are stored on disk as plain **crs: XMP files** — the exact format we
write back out as sidecars.

## Data layout
- `Managed Catalog.mcat` — SQLite doc store. `docs` (type/subtype) joined to
  `revs.content` (**MessagePack** blobs). `docs.fullDocId` = asset id.
- Asset docs (`type=asset, subtype=image`) carry everything:
  `importSource.fileName`, `captureDate`, `xmp.tiff.Make/Model`,
  `reviews.*.flag` (pick/reject), and `develop.xmpCameraRaw.sha256`.
- `settings/<sha256>` — full crs XMP per edit state (flat attributes:
  Exposure2012, Temperature, HSL, tone curves as rdf:Seq; masks referenced by
  digest, stored separately under `aux`).
- `xmp/` — per-image metadata XMPs (not develop settings).
- Previews: `previews.db` `RenditionPath(localAssetId, longSide, path)` →
  JPEGs in `previews/` (640px long side, 10.4K of them).
  Join key: `Managed Catalog.wfindex` table `assets(docId, assetId, filename,
  rating, flag)` — `assetId` = cloud doc id, `docId` = previews' localAssetId.
- Originals: `originals/YYYY/YYYY-MM-DD/<file>` on disk.

## What ran
- `work/mine.py` — Phase-0 miner. Picks-only → `work/dataset/index.jsonl`
  (**544 pairs**); `--all` → `index_all.jsonl` (**2,210 edited images**).
  Has an assert-based `demo()` self-check. Deps: `msgpack`, `defusedxml` (pip).
- `work/catalog.mcat`, `work/wfindex.db` — working copies of the DBs.
- `test-import/` — WIL_6299 NEF + generated `.xmp` sidecar (settings blob
  wrapped in xpacket). NEF has 1 trailing byte appended so Lightroom's
  duplicate detection (sha256) doesn't skip the re-import.

## Pending human step (gates everything — plan §Phase 1)
Import `test-import/` into Lightroom and check the sliders populate:
Exposure **+2.04**, Temp **4172**, Tint **+8**, Contrast **+23**,
Highlights **+27**, Blacks **-12**. Masks may not transfer (stored as
separate aux files) — sliders are the pass criterion.

## Masks / .acr sidecar (session 1 continued)
- First import test: global sliders applied, masks did NOT (neither AI nor
  parametric gradients).
- Cause: modern LR uses **dual sidecars** — `.xmp` (settings text) + `.acr`
  (binary mask rasters). LR cloud "Export Original & Settings" emits all three.
- `.acr` format (reverse-engineered, single mask): 52-byte header
  (`ACR\0` u32=1, `NEF\0` u32=1, u32=0, MaskDigest[16], u32 size, u32=0,
  u32 offset=52, u32=0) + `auxiliary/<d[:3]>/<d[3:]>` blob verbatim + 2 zero
  bytes. `work/make_acr.py` synthesizes it **byte-identical** to Adobe's
  export. Multi-mask layout unverified.
- Aux blobs: mask bitmaps keyed by MaskDigest (embedded TIFF at +0x10).
- Pending: import test3 (Adobe's verbatim triplet) + test4 (synthesized
  triplet). test3 fails ⇒ LR cloud can't restore masks from sidecars at all.

## Phase 1 PASSED (2026-07-11)
Both import tests worked — full round trip including masks via synthesized
`.xmp` + `.acr` pair. The writer can reproduce a complete edit.

## Phase 2 — retrieval engine (built, needs full data)
- `work/.venv` (uv, py3.12): torch/MPS, transformers, rawpy.
- `work/proxies.py` — 640px proxies from NEF embedded thumbs (~1ms/image).
  IMPORTANT: proxies come from originals, NOT LR previews — previews are
  rendered with edits applied (target leakage + they're volatile; LR truncated
  previews.db mid-session).
- `work/embed.py` — SigLIP2-base-384 on MPS → `index/embeddings.npy` + ids.
- `work/predict.py` — NEF → proxy → embed → cosine kNN (k=8) → sidecar.
  Blend: NN's whole settings XMP as base (Look/curves/profile consistent),
  masks stripped, Basic+HSL overridden with k-median. `--eval` = LOO MAE.
- Validated end-to-end on 16 local images (`demo/`); neighbors sensible
  (adjacent frames at 0.98 cosine).

## Export ingestion (ready, waiting on Devin's bulk export)
- `work/load_export.py <folder>` — builds dataset from Export Original &
  Settings pairs; normalizes each XMP's crs Description into
  `settings_norm/` (library-blob shape) so predict.py treats both data
  sources identically. Records carry `settings_path` (also added to mine.py).
- WB fix in blend: WhiteBalance = neighbor mode; Temp/Tint blended only when
  mode is Custom, stripped otherwise so LR honors the preset.
- Full-index run once export lands:
  `.venv/bin/python load_export.py <folder> && .venv/bin/python proxies.py && .venv/bin/python embed.py && .venv/bin/python predict.py --eval`
- Side effect observed: the bulk export makes LR materialize settings blobs
  into the library store (catalog dataset grew 544→685 mid-export).

## Data gap (blocks a real index)
Only 20 originals + 544 settings files are local; 2,256 picked+edited images
have BOTH raw and settings cloud-only. Fix: in Lightroom select all picked
photos → Export → **Original & Settings** (~50GB) → gives NEF+XMP(+acr) pairs;
then a loader builds the dataset from that folder directly. TODO: exported-
folder loader in mine.py; WhiteBalance="As Shot" records have no Temperature
attr — skip Temp/Tint override when WB mode is As Shot.

## Full index built (2026-07-11, export ingestion)
- Export composition: 1,022 NEF + 12 ARW + 3 DNG (sidecar XMPs) + 1,763 JPG
  (settings embedded in-file); 156 RAW+JPEG dup shots → raw wins. 2,644 records.
- JPEG-original edits are excluded from the retrieval index: JPEGs use
  IncrementalTemperature (relative) not Kelvin — unblendable with raw records.
- LOO MAE over 1,034 raw picks: Exposure 0.514 stops, Temperature 279 K,
  Contrast 9.6. Optimistic (same-wedding neighbors); real benchmark = held-out
  wedding judged on renders.
- Roadmap tracked in TODO.html (project root).

## Held-out wedding benchmark (2026-07-11)
- Holdouts: 2025-01-04 (105 raws) tune, 2024-12-30 (343) validate.
- Finding: editing has strong PER-WEDDING regimes (median exposure +0.85 to
  +2.39 across weddings; one desaturated-look wedding, Sat median -34).
  Content-only SigLIP retrieval can't see capture conditions -> lost to the
  global-median baseline on exposure.
- Fix shipped: condition-aware scoring (features.py) = cosine
  - 2.0*||lum_p10/50/90 diff||^2 - 0.2*|EV100 diff| - 0.6*|days diff|/365.
  Validated on holdout B: exposure 0.99->0.58, highlights 44->12, sat 19->10.
- Still open: exposure remains at/above baseline cross-wedding. Paths:
  (1) hybrid shrinkage of kNN medians toward era prior, (2) Engine B with
  photometric features (learn exposure-from-luminance properly),
  (3) anchored mode: Devin edits ~5 anchors of a new wedding first, agent
  derives the wedding-level offset.
- benchmark-import/ has 40 raws+sidecars from holdout A for render judgment.

## Consistency + presets + culling groundwork (2026-07-11)
- Devin's render verdict on benchmark A: decent but inconsistent (exposure/tint
  jitter). Fix: scene smoothing (scenes.py) — greedy time+embedding clustering,
  per-scene median for exposure/WB/tone keys. Validated 2024-12-30: 343 imgs ->
  5 scenes, MAE flat, within-scene stdev -> 0. Default in predict.py batch mode.
- 2025-01-04 "wedding" is actually a 48-min session (one scene is correct).
- Presets: history is already discrete via crs:Look (Artistic 05 x366,
  Modern 08 x263, Modern 10 x148, Modern 09 x65...). Loader stores "look";
  predict.py --look "<name>" restricts retrieval. Menu-bar preset picker maps
  to this.
- Culling data: 2,800 picks / 3,065 rejects flagged in catalog; Adobe
  aesthetics scores on all 10,497 assets. Rejects' originals are cloud-only —
  classifier needs their previews or a second export.
- Public data: PPR10K = closest wedding dataset (11K raws, 3 experts,
  group-level consistency); FiveK for 5 more styles. Phase 4 as planned.

## Shrinkage experiment (2026-07-11, exp_shrink.py)
Era-prior shrinkage of scene medians: helps exposure on holdout A, hurts on B.
No alpha wins both -> not shipped. Cross-wedding exposure variance is
irreducible without photometric learning (Engine B) or anchors.

## Culling model v1 (2026-07-11)
- Data: 2,644 pick proxies (already embedded) vs 3,065 reject JPEGs (small-JPEG
  export at ~/Pictures/agent-export-rejected-only; embeddings cached
  index/rejects_emb.npz).
- Gotcha: exported JPEGs' EXIF 0x0132 is the EXPORT time — capture date is
  sub-IFD 0x9003 (fall back to catalog stem match). First two training runs
  had an all-picks validation split because of this.
- Wedding-grouped val (13 weddings, 945 imgs, 49% picks): AUC 0.963, acc 0.867.
- work/cull.py scores folders -> cull_report.tsv (p_pick ranked).
- Source confound (picks=camera thumbs, rejects=LR renders): at inference all
  inputs are camera thumbs, so any confound biases toward "keep" — the safe
  direction. Revisit if keep-rates look inflated in practice.

## Phase 3 v1 shipped (2026-07-11)
- work/agent.py: watch-folder daemon. Polls every 2s; a wave = pending raws
  stable across two polls; runs predict (scene-smoothed, optional look) then
  cull report if threshold set. IPC = config.json (shell->daemon) +
  state.json (daemon->shell). Log: agent.log.
- menubar/: SwiftUI MenuBarExtra SPM app (macOS 26). @Observable Agent model,
  spawns the daemon at launch (NOT in view .task — window-style MenuBarExtra
  content doesn't appear until first click), polls state.json every 2s.
  Controls: watch-folder chooser, pause, look picker, cull toggle+slider,
  log, quit. Run: menubar/.build/debug/PhotocopyMenuBar
- Deviation from plan: no Rust layer in v1 (macOS-only, single user). The
  daemon boundary is the future Rust extraction seam for Windows.
- E2E verified: dropped NEF -> sidecar + cull report while app running.

## Menu bar v2 — Dropbox-style panel (2026-07-11)
- Layout per Devin's reference: left icon rail (Home / Activity / Settings +
  power button), main pane, bottom status strip with live progress.
- Home: start/stop daemon (tinted), pause, look picker, keep-best-N field.
- Activity: agent.log feed (newest first) + open-full-log.
- Settings pane: import (watch) + export folder choosers, cull by count OR
  threshold (count wins), look default.
- Daemon: cull_target (best-N per wave) + export_dir — keepers' raws+sidecars
  hardlinked (copy fallback) into export dir with cull_report.tsv.
- App supervises the daemon: respawns after 3 stale polls (fixed race where a
  just-killed daemon's fresh state.json blocked the initial spawn).
- E2E: 3 raws -> best-2 cull -> 2 raw+xmp pairs in export dir. Verified.

## Notes for later phases
- 640px previews are plenty for SigLIP (384px input) — no proxy export needed.
- 544 picks is thinner than the plan's "thousands"; `--all` (2,210) or
  rating-based filtering can widen the training set.
- Catalog copy is a snapshot; re-run `mine.py` after each delivered wedding.

## Engine B — trained head shipped (2026-07-11, branch worktree-engine-b-training)
Built the plan's Engine B on Devin's own data — no public download needed yet.
- `presets.py` — preset registry. Each crs:Look is a preset token (15 slots incl.
  UNKNOWN). Adding a preset = another token; `register_external()` reserves slots
  for FiveK/PPR10K experts so "more presets" scales past Devin's looks.
- `train_head.py` — SigLIP2 emb(768) + photometric(lum p10/50/90 + EXIF EV) +
  learned preset token -> MLP(384) -> 37 sliders (Basic + WB + 24 HSL). Huber on
  z-scored targets, exposure/WB up-weighted. Photometry is the point: lets the head
  learn exposure-from-image, which content-only kNN can't see.
  - Fixed 100 epochs (swept: fewer underfit, more overfit exposure). torch seeded.
  - Eval on held-out weddings; deployed checkpoint refits on ALL 1,034 raw picks.
- Wired into `predict.py --engine b`: trained head supplies sliders, kNN neighbor's
  XMP still gives structure (Look/curves/profile); model preds scene-smoothed.
  `--look` conditions both retrieval and the preset token. Falls back to A if no
  checkpoint. `agent.py` reads engine from config.json (menu-bar A/B toggle ready).
- Honest 6-fold grouped (cross-wedding) MAE, B vs A(kNN) vs global-median baseline:
    Exposure   0.67 ± 0.15  |  0.88  |  1.00   -> B wins (A used to LOSE to baseline)
    Contrast   22.1 ± 2.4   |  29.2  |  27.0   -> B wins
    Temperature 1044 ± 343  |  949   |  857    -> baseline wins (Temp/Tint stripped
                 at inference unless WB=Custom, so this rarely bites end-to-end)
- Engine B closes the cross-wedding exposure gap NOTES flagged as irreducible from
  retrieval alone. Checkpoint (index/engineB.pt) is gitignored — regenerate with
  `.venv/bin/python train_head.py`.
- Not done (needs Devin): download FiveK (~50GB) + PPR10K to add the 5+3 expert
  presets and the multi-style pretrain the plan describes.

## Local VLM sanity pass + Phase 4 kickoff (2026-07-11, branch engine-b-phase4)
Everything already runs local (SigLIP2 encoder + Engine A/B head on MPS, no
cloud). Added the plan's optional VLM pass as a genuinely local open-weight model
tuned for Apple Silicon.
- `work/vlm.py` — Qwen3-VL-8B-Instruct-4bit via **mlx-vlm** (0.6.4). Proxy in ->
  {exposure_ev, temp_shift, tint_shift, reason} JSON out. A cold-start / sanity
  check to ride on top of Engine A/B (far neighbors, uncertain head).
  - M5 Max 64 GB: 108 tok/s, 6.8 GB peak — trivial headroom. Model ~5.5 GB.
  - Gotcha: greedy decoding (temp 0) collapses to "exexex…"; needs
    repetition_penalty≈1.15 + temperature≈0.3, and a 3x retry for the rare
    residual collapse. Image grounding verified (captions match the frame).
  - Deps added to .venv: `mlx-vlm` (pulls mlx, mlx-lm); bumped transformers to
    5.12.1 — SigLIP2 still loads clean (checked), embeddings unchanged.
- Model choice: Qwen3-VL-8B-4bit over Qwen2.5-VL — newer, strong fine visual
  judgment, 4-bit fits 64 GB with room. 2B/4B variants exist if speed matters.

## FiveK / PPR10K reality (2026-07-11) — before downloading blindly
- **Engine B predicts Lightroom slider values.** Public datasets don't natively
  carry those:
  - FiveK ships input DNGs + expert **rendered** TIFFs (A–E) publicly; the actual
    per-expert slider values live only inside its bundled Lightroom **catalog**
    (Adobe_imageDevelopSettings Lua blobs) → needs the deferred Lua parser +
    downloading the 47 GB `fivek_dataset.tar` (the catalog isn't offered alone).
  - PPR10K labels are retouched images + 3D LUTs — **no slider values at all**.
    Wrong output space for Engine B; would suit a pixel/LUT model, a different
    engine. Not downloading it.
- Action: downloading FiveK (`work/data/fivek/`, gitignored, ~47 GB, resumable
  `curl -C -`). Once down: extract the .lrcat, write the Lua parser against the
  REAL blob (validate vs a known image, per the mining playbook), map experts
  A–E to preset tokens via presets.register_external(), multi-expert train, then
  fine-tune back onto Devin's 1,034. Gated on the download completing.

## FiveK fine-tune — investigated, measured, NOT shipped (2026-07-11)
Downloaded FiveK (47 GB), extracted the catalog, built the ingest, and let the
benchmark decide (per the plan). Verdict: FiveK does not help this engine.
- Catalog is `fivek_dataset/raw_photos/fivek.lrcat` (1.7 GB SQLite, LR3 2011).
  Develop settings are readable Lua-text blobs (`s = { Key = value, ... }`) — the
  feared binary-blob parser was unnecessary; `fivek_ingest.parse_wb` handles it.
- Process-version mismatch is fatal for slider regression: **0** of 96,458
  develop rows use PV2012, **0** have HSL. All are PV<2012 (Brightness,
  HighlightRecovery, old Exposure). Devin's engine predicts PV2012 (Exposure2012,
  24 HSL, Clarity/Texture/Dehaze) — almost none of which exist in FiveK. Only
  Temperature/Tint (absolute Kelvin) transfer cleanly.
- Empirical test (700 imgs, 8,400 expert WB rows, WB-only external, masked loss,
  3-fold): FiveK WB prior **hurts** Devin's held-out temperature 960 -> 1036 K
  (+76), and drags exposure/contrast slightly too. FiveK is stock photography;
  its WB decisions don't transfer to wedding lighting.
- Conclusion: keep Engine B **Devin-only** (shipped checkpoint is clean, 15
  presets). `work/fivek_ingest.py` kept as the validated parser + the template
  for ingesting any *properly PV2012-aligned* expert data later. PPR10K already
  ruled out (no slider labels). The multi-expert machinery stands ready; it just
  needs data in the right parameter space, which neither public set provides.
- `work/data/fivek/fivek_dataset.tar` (47 GB) can be deleted — re-downloadable;
  only useful later for an image-to-image engine, a different design.

## Fix: progress bar vanished / "not processing" — daemon heartbeat (2026-07-12)
Symptom: loading bar not showing for some waves; NEFs seemed not to process.
Root cause (not a processing bug — a liveness bug): the shell's `daemonAlive`
is "state.json written in the last 8s". The daemon only wrote state on per-file
progress, so its silent warm-up window — `import predict` + `load_model`
(SigLIP2) measured ~7s and network-variable via the HF hub check — routinely
neared/passed 8s. A busy daemon was judged dead, which (a) hid the progress bar
(gated on daemonAlive, App.swift:415) and (b) let the supervisor treat it as
stale and spawn a duplicate that raced on the same folder (the 12+ "daemon up"
restarts in agent.log). Per-file work is fast (~0.25s/NEF), so the bar showed
mid-wave but not during warm-up.
Fix: `agent.py` heartbeat thread refreshes state.json's timestamp every 2s
regardless of phase, so "fresh" reliably means "alive". Verified end-to-end:
warm-up staleness 7s -> 1.3s max; bar stays visible; no duplicate spawns.
Daemon-only change — no app rebuild (shell reads agent.py from disk).

## Export format + JPEG quality + faithful-export follow-up (2026-07-12)
- Export now configurable: raw+sidecar (default, for Lightroom) OR proof JPEG.
  agent.py export honors config export_format + jpeg_quality; preview.render_raw_jpeg
  renders a full-res JPEG (rawpy decode → shared apply_edit slider math → quality).
- Menu bar: segmented Export picker + quality slider (App.swift exportRow), shown
  when JPEG. DaemonConfig got a tolerant init(from:) so old config.json files don't
  reset every setting when a new field appears.
- The "Open in Lightroom when done" toggle already existed (send_to_lightroom) —
  it just never fired because the export step kept dying pre-heartbeat-fix.
- Honest scope: proof JPEG is an APPROXIMATE render (exposure/WB/tone/color only;
  no Look/curves/HSL/profile/masks). TODO (user asked, deferred): Lightroom-faithful
  JPEG export by driving LR's own export/auto-import — matches delivery quality.

## Fix: blank Activity + "nothing happens" on already-edited folders (2026-07-12)
Two separate causes behind "bar still missing, activity not updating":
1. Activity was blank because work/events.jsonl got DELETED from disk: PR #4
   untracked it with `git rm --cached`, and pulling that commit into main applied
   the recorded deletion to the working tree. Restored (history lost, seeded with
   a note); it's gitignored now so git can never touch it again.
2. No bar because there was genuinely nothing to process: every NEF in the watch
   folder already had a sidecar, and the daemon skips edited raws by design. The
   UI just never said so — "watching, 0 pending" looked identical to broken.
   Fix: pending_raws now also returns the already-edited count, daemon reports it
   in state (edited), and the status strip says "All N photos already edited —
   drop in new ones". DaemonState.edited is optional so old state.json decodes.
Verified live: wave on a fresh NEF held status=processing through warm-up (bar
visible), events feed appended start/photo/done, cory-test idle state reads
edited: 40. To re-edit an already-edited folder, delete its .xmp sidecars.

## Menu-bar overflow escape hatch (2026-07-12)
macOS silently hides status items when the menu bar is full (notch MacBooks
especially) — and a hidden icon made this menu-bar-only app unreachable.
Can't force the icon visible (OS behavior). Fix: re-opening Photocopy.app
(Spotlight/Finder/Dock) now shows the panel as a regular 460x520 window —
AppDelegate.applicationShouldHandleReopen -> NSHostingController(PanelView).
Agent became a shared singleton so the window and the menu panel drive the SAME
agent (two instances would each supervise/spawn daemons). Verified: first launch
0 windows, reopen 1 window, still exactly 1 daemon.

## Fix: "keeps starting even though I tell it not to" (2026-07-12)
Stop never persisted intent — it only killed the process. Three compounding bugs:
(1) config.paused was never written by the app, so nothing on disk recorded the
user's "no"; any revived daemon resumed editing. (2) After Stop, poll() re-read
state.json whose timestamp the dead daemon had just refreshed (heartbeat), so
the UI flipped back to "running" for up to 8s — Stop looked ignored. (3) The
Start/Stop button toggled on that flickering liveness bit, so a click during the
flicker could land as Start and spawn a real daemon (observed: app-spawned
daemon at 19:47:45). Some phantom starts were also detached test daemons the
app had no Process handle for.
Fix: config.paused is the source of truth. Stop = paused:true + kill; Start =
paused:false + spawn. UI "running" = daemonAlive && !paused (config is local,
no flicker). App quit kills all daemons (terminate + pkill, catching strays).
Verified: paused daemon idles with pending photos; unpause processes; quit
leaves nothing running. Deployed stopped-by-default per the user's intent.

## Re-edit an already-edited batch (2026-07-12)
Start (or picking/dropping a folder) now scans the folder app-side; if every
photo already has a sidecar, an alert says "All N photos already edited" with
Re-edit N photos / Watch for new photos only / Cancel. Re-edit deletes our
.xmp/.acr sidecars (exported copies are hardlinks — untouched) and starts the
daemon, whose normal pending logic re-processes the batch. Verified with the
real daemon: fresh sidecar written (hash changed) after the delete+unpause
sequence the button performs. Idle strip now says "press Start to re-edit".
