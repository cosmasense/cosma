#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2025/01/10 14:30:00
@Author  :   Ethan Pan 
@Version :   1.0
@Contact :   epan@cs.wisc.edu
@License :   (C)Copyright 2025, Ethan Pan
@Desc    :   Parser module for multi-format file parsing
'''

from .parser import (
    FileParser,
    # parse_file,
    # parse_directory,
    get_supported_extensions,
    is_supported_file,
)

__all__ = [
    "FileParser",
    # "parse_file", 
    # "parse_directory",
    "get_supported_extensions",
    "is_supported_file",
]
