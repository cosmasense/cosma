# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Get the absolute path to schema.sql
schema_path = os.path.abspath('schema.sql')

# Collect all data files from litellm
litellm_datas = collect_data_files('litellm')

# Collect magika model files
magika_datas = collect_data_files('magika')

# Collect sqlite_vec binaries
import sqlite_vec
sqlite_vec_path = os.path.dirname(sqlite_vec.__file__)
vec0_dylib = os.path.join(sqlite_vec_path, 'vec0.dylib')
sqlite_vec_binaries = collect_dynamic_libs('sqlite_vec')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[
        (vec0_dylib, 'sqlite_vec'),
    ] + sqlite_vec_binaries,
    datas=[
        (schema_path, '.'),
    ] + litellm_datas + magika_datas,  # Add magika data
    hiddenimports=[
        'tiktoken', 
        'tiktoken.registry', 
        'tiktoken_ext.openai_public', 
        'tiktoken_ext', 
        'litellm.llms.tokenizers',
        'sqlite_vec',
        'magika',  # Add this
        'markitdown',  # And this
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='backend',
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
    name='backend',
)