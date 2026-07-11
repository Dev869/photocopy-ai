# The Bridal Camera — Auto-Edit Agent: Master Plan
Local, private Lightroom companion for wedding photography. Watches an import folder, analyzes each raw on-device, writes XMP sidecars that match Devin's editing style. No cloud, no subscriptions, photos never leave the machine.

**Target:** macOS ARM (M5 Max, 128GB) now · cross-platform-ready core for Windows later
**User:** just Devin (The Bridal Camera) — no licensing/packaging polish needed for v1

---

## 1. Product definition

**Workflow it slots into:**
```
card dump ──► [AGENT: analyze + write XMP sidecars] ──► import to Lightroom ──► final tweaks ──► deliver
```
Sidecars are written **before** LR import. This avoids catalog write-conflicts entirely and means LR picks up the edits natively on import — no plugin required.

**What "matching my style" means technically:** predicting develop-slider values (exposure, WB, tone, HSL, curves, profile) the way Devin would have set them, per image, informed by his entire editing history.

**Experience:** menu-bar agent (SwiftUI MenuBarExtra). Status (142/2,038), pause/resume, look-era picker, log access, completion notification. No preview UI in v1 — Lightroom *is* the review surface.

---

## 2. Architecture

```
┌───────────────────────────────────────────────┐
│ Swift MenuBarExtra shell (macOS ARM)          │  thin UI only
├───────────────────────────────────────────────┤
│ Rust core (portable, cross-platform)          │
│  • lrcat miner (rusqlite + Lua-blob parser)   │
│  • embedding index + kNN retrieval            │
│  • settings blender / model output mapper     │
│  • XMP sidecar writer (quick-xml, crs: ns)    │
│  • watch-folder daemon (notify crate)         │
├───────────────────────────────────────────────┤
│ Inference trait (swappable backend)           │
│  macOS: MLX — SigLIP encoder, MLP head, VLM   │
│  future Windows: llama.cpp / ONNX Runtime     │
└───────────────────────────────────────────────┘
```

Cross-platform strategy: everything below the shell is Rust and portable (SQLite, XML, file watching all cross-platform). Inference sits behind a trait. Only the menu-bar shell is rewritten per OS, and it's deliberately thin.

**Stack:** Rust (core) · Swift/SwiftUI (shell) · MLX + mlx-vlm (inference; Python sidecar acceptable for v1, mlx-swift later) · SigLIP embeddings · PyTorch MPS or MLX for training · no vector DB (flat index, brute-force kNN — thousands of vectors is nothing).

**Prior art to mine:** LrGeniusAI (AGPL-3.0, open source) — read their develop-settings parsing and XMP handling for reference. Don't vendor AGPL code; learn from it. Architecture stays ours (background agent, not LR plugin).

---

## 3. The intelligence layer — two engines, one benchmark

### Engine A — Retrieval (personal taste, zero training)
The `.lrcat` catalog is SQLite containing every develop setting Devin has ever applied — a ready-made dataset of his taste.
- Mine catalog → (preview, settings) pairs, filtered to delivered/starred images only
- Embed all previews with SigLIP → flat index
- New photo → embed → k nearest neighbors (start k=8) → blend settings:
  - numeric sliders: median · enums: mode · tone curve: nearest neighbor's whole curve
- Debuggable by construction: log which historical photos drove each edit
- Improves automatically every time a wedding is delivered and re-mined

### Engine B — Trained model (competence from public data + personal fine-tune)
Small regression head on a frozen encoder. Not an LLM; doesn't touch pixels.
```
raw proxy ─► SigLIP (frozen) ─► embedding ⊕ style-embedding ─► MLP head (~2-5M params) ─► slider vector
```
**Datasets:**
| Dataset | Size | Key property |
|---|---|---|
| MIT-Adobe FiveK | 5,000 raws × 5 experts | Ships as a **Lightroom catalog** — labels are literal slider values; same lrcat parser mines it (~25K pairs) |
| PPR10K | ~11K portraits × 3 experts | Portrait-focused — closest public data to weddings |
| Devin's catalog | thousands of pairs | The style layer; fine-tune target |

**Style conditioning:** expert ID is a learned embedding token — FiveK trains as 5 distinct styles, not their average. "Devin" becomes the 6th slot via fine-tune (freeze encoder + lower MLP, low LR, aggressive early stopping).

**Output space (v1):** Basic panel (11 floats), WB (2, temp log-scaled), tone curve (8-point param or curve-snap), HSL (24), profile/WB-preset enums. Skip masks/local adjustments — global sliders carry 90% of a look.

**Training:** Huber loss per group + CE for enums, WB/exposure weighted up (most visible errors). Batch 64, AdamW, 20–40 epochs → 1–3 hrs per run on the M5 Max. Hyperparam sweeps overnight.

**License note:** FiveK/PPR10K are research-use. Fine for personal tooling; trained weights inherit the restriction if this ever productizes. The catalog-only path (Engine A, or B fine-tuned purely on own data) stays the clean commercial asset.

### The benchmark (decides the inference path)
Hold out one complete real wedding. Run A, B, and hybrid (B predicts; A overrides when neighbor distance is small). Expected: A wins bread-and-butter shots, B wins novel venues/light, hybrid wins overall — but decide from renders, not theory.

### Optional VLM pass
Qwen2.5-VL via mlx-vlm for per-image exposure/WB sanity correction on top of either engine. M5 Max's Neural Accelerators (~4× prefill vs M4) make a 32B-class VLM viable at batch speeds.

---

## 4. Phased roadmap

**Phase 0 — Catalog mining (weekend 1)**
- Copy `.lrcat`, work on the copy. Tables: `Adobe_imageDevelopSettings` (Lua-style text blob → ~100-line parser), join `Adobe_images` + `AgLibraryFile`
- Filter to delivered/starred images; export 1024px proxies (or pull smart previews from `.lrdata`)
- Output: `dataset/` of (proxy.jpg, settings.json) pairs

**Phase 1 — XMP round-trip (1–2 days) ← critical de-risk**
- settings JSON → `.xmp` sidecar (`crs:` namespace) → place beside raw → import to LR → confirm sliders populate
- If this round-trips, the project is real. Everything after is quality iteration.

**Phase 2 — Retrieval engine (week 2)**
- Embed dataset, flat kNN, blending rules, first end-to-end auto-edited folder

**Phase 3 — Daemon + menu bar (week 3)**
- Rust watcher → queue → batch embed → blend → write sidecars
- MenuBarExtra: progress, pause, log, look-era filter (style drift across years → filter index by date range/collection), completion notification

**Phase 4 — Trained model track (weeks 3–5, parallel-friendly)**
- Download FiveK (~50GB) + PPR10K → mine with Phase 0 parser
- Baseline: single-expert head (FiveK expert C) as sanity check
- Full multi-expert + PPR10K training → personal fine-tune → **the A/B/hybrid benchmark**

**Phase 5 — Quality passes (ongoing)**
- Scene clustering within a wedding (ceremony/reception/golden hour) so blends stay coherent per lighting condition
- Confidence gating: far neighbors + high model uncertainty → skip and flag for manual edit instead of guessing
- Re-mine catalog after each delivered wedding (zero-effort improvement loop)

---

## 5. Risk register

| Risk | Mitigation |
|---|---|
| Develop-settings blob parsing (Lua-ish) | Small parser; validate against known images; LrGeniusAI as reference |
| Process Version mismatches across catalog eras | Normalize to current PV; drop odd old-PV images from dataset |
| Camera profile / body differences | Include camera model as retrieval filter or model feature |
| LR ignores sidecars post-import | Enforced by workflow: process before import |
| Tone curves blend/average poorly | Nearest-neighbor curve transfer; curve-snap in model head |
| Personal fine-tune overfits | Freeze most layers, low LR, early stopping, held-out wedding eval |
| Raw decode throughput | rawpy/embedded-JPEG previews for proxies; decode is the bottleneck, not inference |

## 6. Compute reality (M5 Max, 128GB)
- Embedding 40K images: one afternoon, batched
- Head training: 1–3 hrs/run
- Production inference: tens of ms per image (encoder + MLP); a 2,000-image wedding processes while culling, not overnight
- VLM pass (if used): 32B-class model fits comfortably in unified memory

## 7. First Claude Code session (do this first)
1. Copy `.lrcat`, explore schema, dump one image's develop settings
2. Write the Lua-blob parser
3. Hand-map one settings dict → XMP → import test in Lightroom
Round-trip success = green light for everything else.
