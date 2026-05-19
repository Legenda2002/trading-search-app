@echo off
REM ============================================================
REM  TradingSearch - Windows build script
REM ============================================================
REM
REM Run from the repository root in a fresh Command Prompt:
REM
REM    packaging\build_windows.bat
REM
REM Requirements:
REM    - Python 3.11 or 3.12 64-bit (https://www.python.org/downloads/windows/)
REM    - ~5 GB free disk for the build
REM    - Internet on first run (downloads torch + DINOv2 weights)
REM
REM Output:
REM    dist\TradingSearch\TradingSearch.exe   <- the ZIP-able folder for the client
REM ============================================================

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0\.."

echo.
echo [1/6] Creating virtual environment in .\build-venv
if not exist build-venv (
    python -m venv build-venv || goto :error
)

echo.
echo [2/6] Activating virtual environment
call build-venv\Scripts\activate.bat || goto :error

echo.
echo [3/6] Installing project requirements (torch CPU build)
python -m pip install --upgrade pip wheel || goto :error
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements.txt || goto :error
pip install pyinstaller==6.10 || goto :error

echo.
echo [4/6] Pre-downloading DINOv2 weights into data\hf_cache (so they ship in the bundle)
python -c "from app.vision.embedding_extractor import EmbeddingExtractor; EmbeddingExtractor().warmup(); print('ok')" || goto :error

echo.
echo [5/6] Running PyInstaller (this takes 3-10 minutes)
pyinstaller --noconfirm --clean packaging\trading_search.spec || goto :error

echo.
echo [6/6] Copying user-facing docs into the dist folder
copy /Y packaging\HOW_TO_RUN_RUSSIAN.txt dist\TradingSearch\ >nul 2>&1

echo.
echo ============================================================
echo  Build done.
echo.
echo  Result folder:  dist\TradingSearch\
echo  Launcher:       dist\TradingSearch\TradingSearch.exe
echo.
echo  Zip the whole "dist\TradingSearch" folder and send it to the
echo  client. They unzip anywhere and double-click TradingSearch.exe.
echo ============================================================
goto :eof

:error
echo.
echo BUILD FAILED. See the messages above.
exit /b 1
