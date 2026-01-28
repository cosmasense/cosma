import structlog
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID
import numpy as np


def _serialize_value(value):
    """Serialize special types to JSON-friendly representations."""
    if isinstance(value, set):
        return list(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        return f"<ndarray shape={value.shape} dtype={value.dtype}>"
    # Handle File model from cosma_backend.models
    if hasattr(value, '__class__') and value.__class__.__name__ == 'File':
        return {
            'id': getattr(value, 'id', None),
            'filename': getattr(value, 'filename', None),
            'file_path': getattr(value, 'file_path', None),
            'status': getattr(value, 'status', None).name if hasattr(getattr(value, 'status', None), 'name') else str(getattr(value, 'status', None)),
            'content_hash': getattr(value, 'content_hash', None)
        }
    return value


def _serialize_event_dict(logger, method_name, event_dict):
    """Processor to serialize special types in event dict values."""
    for key, value in event_dict.items():
        event_dict[key] = _serialize_value(value)
    return event_dict


def configure_logging():
    """Configure structlog with default pretty console output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            _serialize_event_dict,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(structlog.stdlib.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger instance."""
    return structlog.get_logger(name)
