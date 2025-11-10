#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2025/07/06 10:43:12
@Author  :   Ethan Pan 
@Version :   1.0
@Contact :   epan@cs.wisc.edu
@License :   (C)Copyright 2025, Ethan Pan
@Desc    :   Summarizer module for AI-powered file summarization
'''

from .summarizer import (
    AutoSummarizer,
    OllamaSummarizer,
    OnlineSummarizer,
    BaseSummarizer,
    SummarizerError,
    AIProviderError,
    summarize_file,
    get_available_providers,
    is_summarizer_available,
)

__all__ = [
    "AutoSummarizer",
    "OllamaSummarizer",
    "OnlineSummarizer",
    "BaseSummarizer",
    "SummarizerError",
    "AIProviderError",
    "summarize_file",
    "get_available_providers",
    "is_summarizer_available",
]

# Put summarizing code in here.
# Maybe expose a Summarizer class that contains all initialization logic
# and has a method for summarizing a file?
#
# Something like:
# summarizer = Summarizer(...)
# summarizer.summarize_file(file)

