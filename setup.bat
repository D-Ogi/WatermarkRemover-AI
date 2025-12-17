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
    "%PYTHON_EXE%" -m pip install --upgrade -r requirements.txt --no-cache-dir --use-deprecated=legacy-resolver %PIP_MIRROR%
) else (
    "%PYTHON_EXE%" -m pip install --upgrade -r requirements.txt --no-cache-dir --use-deprecated=legacy-resolver
)
if errorlevel 1 (
    echo   [X] Failed to install base dependencies
    pause
    exit /b 1
)

:: Verify key packages
"%PYTHON_EXE%" -c "import torch; import transformers; import webview; import cv2; print('OK')" >nul 2>&1
if errorlevel 1 (
    echo   [X] Failed to verify base packages
    pause
    exit /b 1
)
echo   [OK] Base packages installed

:: Install iopaint without dependencies
echo   [*] Installing iopaint...
if "%CHINA_MODE%"=="1" (
    "%PYTHON_EXE%" -m pip install --upgrade iopaint --no-deps --no-cache-dir %PIP_MIRROR%
) else (
    "%PYTHON_EXE%" -m pip install --upgrade iopaint --no-deps --no-cache-dir
)
if errorlevel 1 (
    echo   [X] Failed to install iopaint
    pause
    exit /b 1
)
echo   [OK] iopaint installed

:: Install iopaint dependencies manually
echo   [*] Installing iopaint dependencies...
if "%CHINA_MODE%"=="1" (
    "%PYTHON_EXE%" -m pip install pydantic typer einops omegaconf easydict yacs --no-cache-dir %PIP_MIRROR%
) else (
    "%PYTHON_EXE%" -m pip install pydantic typer einops omegaconf easydict yacs --no-cache-dir
)
if errorlevel 1 (
    echo   [X] Failed to install iopaint dependencies
    pause
    exit /b 1
)

:: Verify iopaint dependencies
"%PYTHON_EXE%" -c "import pydantic; import typer; import einops; import omegaconf; import easydict; import yacs; print('OK')" >nul 2>&1
if errorlevel 1 (
    echo   [!] iopaint dependencies verification failed, attempting reinstall...
    if "%CHINA_MODE%"=="1" (
        "%PYTHON_EXE%" -m pip install pydantic --no-cache-dir --force-reinstall %PIP_MIRROR%
        "%PYTHON_EXE%" -m pip install typer --no-cache-dir --force-reinstall %PIP_MIRROR%
        "%PYTHON_EXE%" -m pip install einops --no-cache-dir --force-reinstall %PIP_MIRROR%
        "%PYTHON_EXE%" -m pip install omegaconf --no-cache-dir --force-reinstall %PIP_MIRROR%
        "%PYTHON_EXE%" -m pip install easydict --no-cache-dir --force-reinstall %PIP_MIRROR%
        "%PYTHON_EXE%" -m pip install yacs --no-cache-dir --force-reinstall %PIP_MIRROR%
    ) else (
        "%PYTHON_EXE%" -m pip install pydantic --no-cache-dir --force-reinstall
        "%PYTHON_EXE%" -m pip install typer --no-cache-dir --force-reinstall
        "%PYTHON_EXE%" -m pip install einops --no-cache-dir --force-reinstall
        "%PYTHON_EXE%" -m pip install omegaconf --no-cache-dir --force-reinstall
        "%PYTHON_EXE%" -m pip install easydict --no-cache-dir --force-reinstall
        "%PYTHON_EXE%" -m pip install yacs --no-cache-dir --force-reinstall
    )

    "%PYTHON_EXE%" -c "import pydantic; import typer; import einops; import omegaconf; import easydict; import yacs; print('OK')" >nul 2>&1
    if errorlevel 1 (
        echo   [X] Could not install iopaint dependencies
        echo       Please try running: pip install pydantic typer einops omegaconf easydict yacs
        pause
        exit /b 1
    )
)
echo   [OK] iopaint dependencies installed and verified

:: Download LaMA model
echo.
echo   [*] Downloading LaMA model (196MB)...
set LAMA_DIR=%USERPROFILE%\.cache\torch\hub\checkpoints
set LAMA_FILE=%LAMA_DIR%\big-lama.pt

if not exist "%LAMA_FILE%" (
    if not exist "%LAMA_DIR%" mkdir "%LAMA_DIR%"
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt' -OutFile '%LAMA_FILE%' -UseBasicParsing"
    if exist "%LAMA_FILE%" (
        echo   [OK] LaMA model ready
    ) else (
        echo   [!] Warning: Could not download LaMA model
        echo       It will be downloaded on first use
    )
) else (
    echo   [OK] LaMA model already exists
)

:: Download Florence-2 model
echo.
echo   [*] Downloading Florence-2 model (~1.5GB)...
if "%CHINA_MODE%"=="1" (
    echo       Using HF-Mirror for faster download in China
    "%PYTHON_EXE%" -c "import os; os.environ['HF_ENDPOINT']='%HF_ENDPOINT%'; from huggingface_hub import snapshot_download; snapshot_download('florence-community/Florence-2-large', local_dir_use_symlinks=False)"
) else (
    "%PYTHON_EXE%" -c "from huggingface_hub import snapshot_download; snapshot_download('florence-community/Florence-2-large', local_dir_use_symlinks=False)"
)
if errorlevel 1 (
    echo   [!] Warning: Could not download Florence-2 model
    echo       It will be downloaded on first use
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
