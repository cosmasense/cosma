# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the cosma CLI binary.

Bundles the cosma CLI plus the entire cosma-backend runtime so the
binary can be dropped onto a clean macOS machine and run `cosma serve`
without any pip / uv install.
"""
import os
import sys
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


def merge(*tuples):
    """Sum (datas, binaries, hiddenimports) triples from collect_all."""
    datas, binaries, hidden = [], [], []
    for d, b, h in tuples:
        datas += d
        binaries += b
        hidden += h
    return datas, binaries, hidden


# Heavy deps that ship data files, native libs, and/or use dynamic
# imports PyInstaller can't statically resolve. collect_all is the
# blunt-but-reliable hammer for these.
collected = merge(
    collect_all("cosma"),
    collect_all("cosma_backend"),
    collect_all("cosma_client"),
    collect_all("cosma_tui"),
    collect_all("litellm"),
    collect_all("markitdown"),
    # markitdown delegates content-type sniffing to magika, which loads a
    # model file from its own package data dir at runtime.
    collect_all("magika"),
    collect_all("tiktoken"),
    collect_all("tiktoken_ext"),
    collect_all("sqlite_vec"),
    collect_all("llama_cpp"),
    collect_all("pywhispercpp"),
    collect_all("sentence_transformers"),
    collect_all("transformers"),
    collect_all("huggingface_hub"),
    collect_all("torch"),
    collect_all("pillow_heif"),
    collect_all("imageio_ffmpeg"),
    collect_all("mutagen"),
    collect_all("pypdf"),
    collect_all("quart"),
    collect_all("quart_schema"),
    collect_all("structlog"),
    collect_all("watchdog"),
    collect_all("platformdirs"),
    collect_all("click"),
    collect_all("click_help_colors"),
    collect_all("rich"),
    collect_all("ollama"),
    collect_all("asqlite"),
)
datas, binaries, hiddenimports = collected

# `cosma --version` reads versions via importlib.metadata, which needs the
# dist-info dirs to be present in the bundle. collect_all doesn't copy
# those; copy_metadata does.
for pkg in ("cosma", "cosma-backend", "cosma-client", "cosma-tui"):
    datas += copy_metadata(pkg)

# pyobjc Vision / Quartz frameworks — Darwin-only OCR.
if sys.platform == "darwin":
    for pkg in ("Quartz", "Vision", "Foundation", "AppKit", "CoreFoundation",
                "objc"):
        try:
            datas += collect_data_files(pkg)
            binaries += collect_dynamic_libs(pkg)
            hiddenimports += collect_submodules(pkg)
        except Exception:
            pass

# Explicit hidden imports for tokenizers + dynamically-imported plugins.
hiddenimports += [
    "tiktoken_ext.openai_public",
    "litellm.llms.tokenizers",
    "encodings.idna",
    "encodings.utf_8",
    # markitdown converters are loaded by name.
    "markitdown.converters",
]


a = Analysis(
    ["main.py"],
    pathex=[
        os.path.abspath("src"),
        os.path.abspath("packages/cosma-backend/src"),
        os.path.abspath("packages/cosma-client/src"),
        os.path.abspath("packages/cosma-tui/src"),
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Test frameworks pull in heavy junk we never run.
        "pytest",
        "pytest_asyncio",
        "pytest_mock",
        "pytest_cov",
        # Notebook stack — neither cosma nor its deps need it at runtime.
        "IPython",
        "jupyter",
        "notebook",
        # Optional torch components cosma doesn't use.
        "torch.distributed",
        "torch.testing",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cosma",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX corrupts signed Mach-O binaries on macOS; never enable it here.
    upx=False,
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
    upx=False,
    upx_exclude=[],
    name="cosma",
)
