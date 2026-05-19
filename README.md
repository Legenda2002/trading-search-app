# Trading Chart Fragment Search

Desktop application for finding similar trading chart images by a small
fragment. Combines OpenCV ORB+RANSAC for pixel-accurate fragment matching
with DINOv2 embeddings for style-agnostic visual similarity.

Stack: Python, PySide6, OpenCV, PyTorch (CPU), HuggingFace transformers,
SQLite, file storage on disk.

## Features

- Import a folder of chart images (recursive).
- Store metadata in SQLite; originals, thumbnails, ORB descriptors and
  DINOv2 embeddings live on disk under `data/`.
- Three search modes selectable from the toolbar:
  - **Smart** (default) — DINOv2 retrieves top-200 visual neighbours, then
    ORB+RANSAC verifies and localises. Returns calibrated similarity %
    and bounding polygon when an exact fragment is found.
  - **Exact only (ORB)** — the original Phase 1 pipeline. Best for
    pixel-accurate fragment crops from the same source.
  - **Similar only (AI)** — embedding-only, fastest. Surfaces visually
    close charts even when the rendering style differs (white vs. dark
    background, different drawing tool, etc.).
- Mouse-drag region selection inside the image viewer and `Ctrl+V` paste
  from clipboard.
- Localised region is highlighted with a polygon overlay when ORB
  homography succeeds.

Tunable thresholds live in [app/core/config.py](app/core/config.py):

- `MIN_MATCH_THRESHOLD` — minimum good ORB matches before attempting
  localisation.
- `RANSAC_REPROJ_THRESHOLD` — RANSAC reprojection threshold in pixels.
- `MIN_INLIERS` — minimum RANSAC inliers to accept a homography.
- `HYBRID_TOP_CANDIDATES` — how many embedding nearest-neighbours to
  shortlist before the ORB verification stage (default 200).
- `EXACT_MATCH_INLIER_THRESHOLD` — inlier count above which a candidate
  is labelled an `exact` match in the UI (default 50).

## Project layout

```
app/
  main.py                  # Entry point
  core/                    # Config, dataclasses
  storage/                 # SQLite + image/descriptor files
  vision/                  # OpenCV preprocessing, features, matching
  indexing/                # Folder import + descriptor build
  search/                  # Fragment-to-image search engine
  workers/                 # Qt background workers
  ui/                      # PySide6 widgets and main window
data/                      # Runtime data (originals, thumbnails, descriptors, app.db)
tests/                     # Reserved for future tests
```

## Setup

> On Debian / Ubuntu 24.04+ system Python refuses to install packages
> directly (PEP 668 `externally-managed-environment`). Always use the
> project venv — either activate it (`source venv/bin/activate`) or call
> pip via the venv binary (`venv/bin/pip ...`).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Torch CPU build (avoids the multi-GB CUDA wheels pulled from PyPI by default).
# Make sure the venv is active — running `pip install ...` outside it will
# fail with "externally-managed-environment" on modern Ubuntu/Debian.
pip install --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple \
            torch torchvision transformers safetensors huggingface_hub
```

The first run downloads `facebook/dinov2-small` (~85 MB) from HuggingFace
into `data/hf_cache/`. Subsequent runs work offline.

## Run

```bash
source venv/bin/activate     # activate the venv first
python -m app.main           # normal run
python -m app.main --debug   # verbose DEBUG logs
```

## Phase 2A: hybrid AI search

After upgrading from a previous (ORB-only) install, generate embeddings
for the existing library once:

```bash
python -m scripts.build_embeddings
```

On CPU this processes ~10 images per second, so a 17k library takes
~25-30 minutes. The script saves progress every ~500 images, so you can
interrupt and resume safely. Subsequent `Import folder` actions also
populate embeddings automatically.

The UI Mode dropdown lets you switch between Smart / Exact / Similar at
any time without restarting. Search results show a single 0-100 %
similarity column:

| Symbol | Meaning                                         | Range    |
| ------ | ----------------------------------------------- | -------- |
| `[E]`  | ORB found a geometrically consistent fragment   | 70-100 % |
| `[S]`  | Visually similar (embedding match, no fragment) | 25-89 %  |
| `[ ]`  | Below the similarity threshold                  | < 25 %   |

### Choosing a mode

- **Full-frame screenshot** from MetaTrader / TradingView / IDE viewer:
  Smart mode finds the exact frame in ~3 seconds, top-1 with the green
  bounding box and 95-100 % similarity. This is the common case.
- **Different visual style** (white background, different colour palette,
  redrawn chart): Smart mode falls back to embedding-only candidates
  in the 50-80 % range, no green box, but the closest visual matches are
  surfaced.
- **Tight centre crop of a chart you already have**: switch to
  `Exact only (ORB)`. DINOv2 ranks visually-similar charts above the
  exact source on homogeneous candlestick libraries when the crop
  removes axis labels and scale markers; ORB-only sidesteps that.

## Typical workflow

1. Click "Import folder" and select a directory with chart images.
2. Wait while images are imported and ORB descriptors are built.
3. Click "Search by fragment" and select a small fragment image.
4. Pick a result from the right panel to preview it on the left.

The status bar shows query keypoint count, target keypoint count, score,
and number of good matches for the selected result.

## End-to-end CLI test workflow

For a UI-less smoke test of the pipeline:

```bash
# 1. Generate ~10 synthetic candlestick charts and a query fragment
python -m scripts.generate_sample_images --count 10

# 2. Run the full workflow against the generated samples
python -m scripts.test_workflow --reset --debug

# Optional: also open the best match in your system image viewer
python -m scripts.test_workflow --reset --open-best
```

The CLI prints a table with rank, score, match count, target keypoint count
and filename for the top-10 results. With `--debug` you also get per-image
match counts, descriptor extraction details, and import progress.

## Convert MetaTrader-5 CSV exports to charts

The MT5 platform exports OHLC history as tab-separated CSV with the header
`<DATE> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>` (intraday
timeframes also include a time column). The `csv_to_charts` script slices the
history with a sliding window and renders each window as a candlestick PNG.

```bash
# Drop the CSV anywhere, e.g.
ls data/csv/XAUUSD_Daily.csv

# Render 60-candle windows with a 30-candle step
python -m scripts.csv_to_charts \
    --csv data/csv/XAUUSD_Daily.csv \
    --output samples/library/xauusd_daily \
    --window 60 --step 30

# Then in the desktop app: Import folder -> samples/library/xauusd_daily
```

Useful flags:

- `--window N`   — number of candles in one chart image (default 60).
- `--step N`     — sliding step in candles (default 30, so windows overlap).
- `--width / --height` — output PNG size (default 1024x560, TradingView-like).
- `--max-charts N` — render only the first N windows (handy for tests).
- `--prefix STR` — prefix added to filenames (default = CSV stem).

A 20-year Daily CSV produces roughly 100-200 chart images per window size of
60/30, which is enough material for an ORB-based search demo. For minute
timeframes you'll want larger windows and bigger steps, otherwise the
generated library can run into the hundreds of thousands of PNGs.

## Diagnostic benchmark

`scripts/debug_benchmark.py` runs both an in-library suite (centre crops
of known images) and an optional external-screenshots suite, then reports
per-query rejection reasons so you can tell ORB-pipeline bugs apart from
style-mismatch limitations:

```bash
# Plain ORB benchmark
python -m scripts.debug_benchmark \
    --in-library 5 \
    --external-dir data/csv/external_queries \
    --top 10 \
    --out reports/orb.txt

# Same queries through the hybrid engine
python -m scripts.debug_benchmark \
    --engine hybrid --mode smart \
    --in-library 5 \
    --external-dir data/csv/external_queries \
    --top 10 \
    --out reports/hybrid_smart.txt
```

Useful flags:

- `--engine orb|hybrid`   — which retrieval pipeline to use (default `orb`).
- `--mode smart|exact|similar` — hybrid mode (only with `--engine hybrid`).
- `--in-library N`        — how many random library images to crop and
  search.
- `--external-dir DIR`    — folder of additional query screenshots
  (skipped if not provided).
- `--top N`               — rows to print per query.

Each row shows the calibrated similarity %, match type, raw embedding
cosine, inlier count, and the RANSAC rejection reason for the candidate.
