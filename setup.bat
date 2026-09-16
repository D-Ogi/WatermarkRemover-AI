@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title WatermarkRemover-AI Setup

echo.
echo   =============================================
echo      WatermarkRemover-AI Setup (Windows)
echo   =============================================
echo.

set PYTHON_VERSION=3.12.7
set PYTHON_DIR=python
set PYTHON_EXE=%PYTHON_DIR%\python.exe

:: China mirror configuration
set CHINA_MODE=0
set PIP_MIRROR=
set PIP_TRUSTED_HOST=
set HF_ENDPOINT=

:: Check if user is in China (for mirror selection)
echo   [?] Are you in China? (y/n)
echo       This will use faster mirrors for downloads
set /p CHINA_CHOICE="      "
if /i "%CHINA_CHOICE%"=="y" (
    set CHINA_MODE=1
    set PIP_MIRROR=-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    set HF_ENDPOINT=https://hf-mirror.com
    echo   [OK] Using China mirrors (Tsinghua PyPI + HF-Mirror)
) else (
    echo   [OK] Using default mirrors
)
echo.

:: Check if embedded Python exists
if not exist "%PYTHON_EXE%" (
    echo   [*] Downloading Python %PYTHON_VERSION%...

    :: Determine architecture
    if "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
        set ARCH=amd64
    ) else (
        set ARCH=win32
    )

    set PYTHON_ZIP=python-%PYTHON_VERSION%-embed-!ARCH!.zip
    set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/!PYTHON_ZIP!

    :: Download Python using PowerShell (available on all modern Windows)
    powershell -Command "Invoke-WebRequest -Uri '!PYTHON_URL!' -OutFile '!PYTHON_ZIP!' -UseBasicParsing"
    if errorlevel 1 (
        echo   [X] Failed to download Python
        pause
        exit /b 1
    )
    echo   [OK] Downloaded Python

    :: Extract using PowerShell
    echo   [*] Extracting...
    powershell -Command "Expand-Archive -Path '!PYTHON_ZIP!' -DestinationPath '%PYTHON_DIR%' -Force"
    del "!PYTHON_ZIP!"

    :: Enable pip by modifying python312._pth
    set PTH_FILE=%PYTHON_DIR%\python312._pth
    if exist "!PTH_FILE!" (
        powershell -Command "(Get-Content '!PTH_FILE!' -Raw) -replace '#import site', 'import site' | Set-Content '!PTH_FILE!' -NoNewline"
        echo Lib\site-packages>> "!PTH_FILE!"
    )

    :: Create Lib\site-packages directory
    if not exist "%PYTHON_DIR%\Lib\site-packages" mkdir "%PYTHON_DIR%\Lib\site-packages"

    :: Download and install pip
    echo   [*] Installing pip...
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py' -UseBasicParsing"
    "%PYTHON_EXE%" get-pip.py --no-warn-script-location >nul 2>&1
    del get-pip.py

    echo   [OK] Python %PYTHON_VERSION% ready
) else (
    echo   [OK] Python found
)

:: Include the application root for embedded Python package imports.
set PTH_FILE=%PYTHON_DIR%\python312._pth
if exist "%PTH_FILE%" (
    findstr /l /x /c:".." "%PTH_FILE%" >nul
    if errorlevel 1 echo ..>> "%PTH_FILE%"
)

echo.
echo   [*] Installing dependencies...
echo       This takes 5-10 minutes. Please wait...
echo.

:: Upgrade pip and install build tools
if "%CHINA_MODE%"=="1" (
    "%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel %PIP_MIRROR% >nul 2>&1
) else (
    "%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel >nul 2>&1
)

:: Install base dependencies
echo   [*] Installing base packages...
if "%CHINA_MODE%"=="1" (
    "%PYTHON_EXE%" -m pip install --upgrade -r requirements.txt --no-cache-dir %PIP_MIRROR%
) else (
    "%PYTHON_EXE%" -m pip install --upgrade -r requirements.txt --no-cache-dir
)
if errorlevel 1 (
    echo   [X] Failed to install base dependencies
    pause
    exit /b 1
)

:: Verify the full application environment, not only selected imports.
"%PYTHON_EXE%" -m pip check
if errorlevel 1 (
    echo   [X] Dependency verification failed. Use a fresh application environment.
    pause
    exit /b 1
)
"%PYTHON_EXE%" -c "import remwm; import webview; import yaml; import psutil"
if errorlevel 1 (
    echo   [X] Failed to import application packages
    pause
    exit /b 1
)

:: Shared checksum-verified, atomic model download.
echo   [*] Preparing LaMA model (196MB)...
if "%CHINA_MODE%"=="1" echo       If GitHub is blocked, preseed the verified cache: docs/lama-runtime.md#restricted-networks
"%PYTHON_EXE%" -m lama_inpaint download
if errorlevel 1 (
    echo   [X] Could not prepare verified LaMA weights. Fix the error above and retry.
    pause
    exit /b 1
)

:: Download Florence-2 model
echo.
echo   [*] Downloading Florence-2 model (~1.5GB)...
if "%CHINA_MODE%"=="1" (
    echo       Using HF-Mirror for faster download in China
    "%PYTHON_EXE%" -c "import os; os.environ['HF_ENDPOINT']='%HF_ENDPOINT%'; from huggingface_hub import snapshot_download; from model_assets import FLORENCE_REPO, FLORENCE_REVISION, MANIFEST, ensure_florence; snapshot_download(FLORENCE_REPO, revision=FLORENCE_REVISION, allow_patterns=[f['name'] for f in MANIFEST['files']]); ensure_florence(download=False)"
) else (
    "%PYTHON_EXE%" -c "from huggingface_hub import snapshot_download; from model_assets import FLORENCE_REPO, FLORENCE_REVISION, MANIFEST, ensure_florence; snapshot_download(FLORENCE_REPO, revision=FLORENCE_REVISION, allow_patterns=[f['name'] for f in MANIFEST['files']]); ensure_florence(download=False)"
)
if errorlevel 1 (
    echo   [!] Warning: Could not download Florence-2 model
    echo       Open Models in the application and choose Download / Retry
) else (
    echo   [OK] Florence-2 model ready
)

echo.
echo   =============================================
echo      Setup complete! Ready to go!
echo   =============================================
echo.
echo   To run the app: Double-click run.bat
echo.

set /p LAUNCH="  Launch now? (y/n): "
if /i "%LAUNCH%"=="y" (
    echo.
    echo   Starting WatermarkRemover-AI...
    start "" "%PYTHON_EXE%" remwmgui.py
)

echo.
echo   Have fun yeeting watermarks!
echo.
pause
