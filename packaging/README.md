# Packaging TradingSearch for Windows

This folder contains everything needed to produce a standalone Windows
distribution that a non-developer can run by double-clicking a `.exe`.

## Two ways to build

### Option A — GitHub Actions (recommended, no Windows needed)

1. Push this repository to GitHub.
2. Open the **Actions** tab → **build-windows** workflow → **Run workflow**.
3. Wait 6–10 minutes.
4. Download the `TradingSearch-windows` artifact from the run summary.
5. Send the ZIP to the client.

To publish a tagged release that the client can download as a permanent link:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The same workflow then publishes the ZIP as a GitHub Release asset.

### Option B — Local build on a Windows machine

Requires Python 3.11 64-bit installed from [python.org](https://www.python.org/downloads/windows/).

```cmd
git clone <your-repo>
cd trading-search-app
packaging\build_windows.bat
```

Result: `dist\TradingSearch\` — zip it up and send to the client.

## What goes into the bundle

| Item                         | Size      | Notes                                   |
|------------------------------|-----------|-----------------------------------------|
| Python 3.11 runtime          | ~30 MB    | embedded                                |
| PySide6 (Qt6)                | ~120 MB   | UI framework                            |
| torch CPU build              | ~200 MB   | DINOv2 inference                        |
| opencv-python-headless       | ~70 MB    | ORB features and homography             |
| transformers                 | ~30 MB    | DINOv2 model loader                     |
| DINOv2-small weights         | ~85 MB    | bundled in `resources/hf_cache/`        |
| App code (`app/`)            | <1 MB     |                                         |
| **Total ZIP**                | **~600 MB** | **~750 MB unzipped**                  |

## Where the client's data goes at runtime

The app writes everything (database, descriptors, embeddings, originals,
thumbnails) into a per-user directory so the client does not need admin
rights and reinstalling the app does not wipe their library.

- Windows: `%LOCALAPPDATA%\TradingSearch\data\`
- macOS: `~/Library/Application Support/TradingSearch/data/`
- Linux: `~/.local/share/TradingSearch/data/`

## What to send the client

Just the ZIP. Inside it they get:

```
TradingSearch/
├── TradingSearch.exe          <- double-click this
├── HOW_TO_RUN_RUSSIAN.txt     <- Russian quick-start guide
└── _internal/                 <- DLLs, model weights, Python runtime
    └── resources/hf_cache/    <- DINOv2 weights for offline first run
```

## Troubleshooting build issues

- **`ModuleNotFoundError: transformers.models.dinov2`** — already listed in
  `hiddenimports` of the spec; if a future transformers version reorganises
  modules, add the new module path there.

- **Bundle bigger than ~900 MB** — usually a UPX flag accidentally turned
  off, or torch pulled in CUDA libraries. Make sure pip used the CPU index
  (`--index-url https://download.pytorch.org/whl/cpu`).

- **Antivirus flags `TradingSearch.exe`** — common with PyInstaller bundles
  (false positive on `pyi-bootloader`). The workflow uses
  `--onedir` rather than `--onefile`, which significantly reduces this.
  If it still happens, codesign the binary or submit to Microsoft for
  whitelisting.
