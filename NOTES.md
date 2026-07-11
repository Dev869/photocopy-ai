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
