import json
import logging
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID
import numpy as np


class Encoder(json.JSONEncoder):
    def default(self, o):
        # Handle sets
        if isinstance(o, set):
            return tuple(o)
        
        # Handle datetime types
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, time):
            return o.isoformat()
        if isinstance(o, timedelta):
            return o.total_seconds()
        
        # Handle numeric types
        if isinstance(o, Decimal):
            return float(o)
        
        # Handle UUID (common in databases)
        if isinstance(o, UUID):
            return str(o)
        
        # Handle Path objects
        if isinstance(o, Path):
            return str(o)
        
        # Handle bytes
        if isinstance(o, bytes):
            return o.decode('utf-8', errors='replace')
        
        # Handle numpy arrays
        if isinstance(o, np.ndarray):
            return f"<ndarray shape={o.shape} dtype={o.dtype}>"
        
        # Handle File model from cosma_backend.models
        if hasattr(o, '__class__') and o.__class__.__name__ == 'File':
            return {
                'id': getattr(o, 'id', None),
                'filename': getattr(o, 'filename', None),
                'file_path': getattr(o, 'file_path', None),
                'status': getattr(o, 'status', None).name if hasattr(getattr(o, 'status', None), 'name') else str(getattr(o, 'status', None)),
                'content_hash': getattr(o, 'content_hash', None)
            }
        
        # return super().default(o)
        if hasattr(o, '__str__'):
            return str(o)
        
        return repr(o)


class StructuredMessage:
    def __init__(self, message, /, **kwargs):
        self.message = message
        self.kwargs = kwargs

    def __str__(self):
        s = Encoder().encode(self.kwargs)
        return f'{self.message} {s}'


sm = StructuredMessage   # optional, to improve readability
