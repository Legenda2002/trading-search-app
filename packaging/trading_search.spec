# PyInstaller spec for the TradingSearch desktop app.
#
# Build (run from the repository root, on Windows):
#   pip install -r requirements.txt pyinstaller==6.10
#   pyinstaller --noconfirm packaging\trading_search.spec
#
# Output: dist\TradingSearch\TradingSearch.exe  (folder you ZIP and ship)
#
# We use --onedir (the default when you list scripts here). It launches faster
# than --onefile and is friendlier to antivirus heuristics that get jumpy with
# self-extracting executables.

from pathlib import Path

block_cipher = None

SPEC_DIR = Path(SPECPATH).resolve()
PROJECT_ROOT = SPEC_DIR.parent

# Bundle the DINOv2 weights so the client doesn't need internet on first run.
HF_CACHE_SRC = PROJECT_ROOT / "data" / "hf_cache"
extra_datas = []
if HF_CACHE_SRC.is_dir():
    extra_datas.append((str(HF_CACHE_SRC), "resources/hf_cache"))


a = Analysis(
    [str(PROJECT_ROOT / "app" / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=extra_datas,
    hiddenimports=[
        "PIL._tkinter_finder",
        "transformers.models.dinov2",
        "transformers.models.dinov2.modeling_dinov2",
        "transformers.image_processing_utils_fast",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Strip large, unused chunks of torch/transformers to keep the bundle
    # closer to ~700 MB instead of 1.2 GB. If something breaks at runtime,
    # remove items here first.
    excludes=[
        "tkinter",
        "tk",
        "tcl",
        "torch.distributed",
        "torch.testing",
        "torch.fx",
        "torchaudio",
        "torchvision.datasets",
        "transformers.models.bert",
        "transformers.models.gpt2",
        "transformers.models.t5",
        "transformers.models.llama",
        "transformers.models.whisper",
        "pandas",
        "scipy",
        "matplotlib",
        "notebook",
        "IPython",
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TradingSearch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "packaging" / "app.ico") if (PROJECT_ROOT / "packaging" / "app.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TradingSearch",
)
