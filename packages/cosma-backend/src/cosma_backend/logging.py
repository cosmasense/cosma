"""
Structured Logging Configuration

Provides structlog-based logging with automatic serialization of complex types.
All log output is structured JSON-friendly for easy parsing and analysis.

Usage:
    from cosma_backend.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Processing file", filename="test.pdf", size=1024)
"""

import logging
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import structlog


def _serialize_value(value: Any) -> Any:
    """
    Serialize special types to JSON-friendly representations.

    Handles types that aren't natively JSON-serializable:
    - datetime/date/time → ISO format strings
    - Path → string path
    - numpy arrays → shape/dtype description (not full data)
    - File model → dict with key fields only
    """
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
        # Don't serialize full array data (could be huge)
        return f"<ndarray shape={value.shape} dtype={value.dtype}>"
    # Handle File model - serialize only key fields for logging
    if hasattr(value, '__class__') and value.__class__.__name__ == 'File':
        status = getattr(value, 'status', None)
        return {
            'id': getattr(value, 'id', None),
            'filename': getattr(value, 'filename', None),
            'file_path': getattr(value, 'file_path', None),
            'status': status.name if hasattr(status, 'name') else str(status),
            'content_hash': getattr(value, 'content_hash', None)
        }
    return value


def _serialize_event_dict(logger, method_name, event_dict):
    """Processor to serialize special types in event dict values."""
    for key, value in event_dict.items():
        event_dict[key] = _serialize_value(value)
    return event_dict


def configure_logging(log_path: Path | None = None) -> None:
    """Configure structlog with console output and optional file logging."""
    pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        _serialize_event_dict,
    ]

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(),
        foreign_pre_chain=pre_chain,
    )

    handlers: list[logging.Handler] = []
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=pre_chain,
        )
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    for handler in handlers:
        root_logger.addHandler(handler)

    structlog.configure(
        processors=pre_chain + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.typing.FilteringBoundLogger:
    """Get a structlog logger instance."""
    return structlog.get_logger(name)
