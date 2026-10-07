# -*- mode: python ; coding: utf-8 -*-
# OpenWipe PyInstaller spec — portable folder build.
# Produces dist/OpenWipe/ folder with OpenWipe.exe + all dependencies.
# Zip the folder and distribute. Users unzip and double-click to run.
# No Python installation required.

block_cipher = None

a = Analysis(
    ['desktop_main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('ui', 'ui'),
        ('ui-openwipe', 'ui-openwipe'),
        ('models', 'models'),
        ('licenses', 'licenses'),
    ],
    hiddenimports=[
        'remwm',
        'remwmgui',
        'utils',
        'model_assets',
        'desktop_runtime',
        'desktop_models',
        'quality_pipeline',
        'lama_inpaint',
        'lama_inpaint.runtime',
        'lama_inpaint.processing',
        'lama_inpaint.model_store',
        'torch',
        'torchvision',
        'transformers',
        'cv2',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFilter',
        'numpy',
        'click',
        'loguru',
        'tqdm',
        'webview',
        'psutil',
        'yaml',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'pytest',
        'IPython',
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
    name='OpenWipe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OpenWipe',
)
