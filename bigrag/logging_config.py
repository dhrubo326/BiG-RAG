"""
BiG-RAG Centralized Logging Configuration

Provides:
- Structured logging with JSON format option
- Log rotation (daily or size-based)
- Multiple handlers (file, console, error-only)
- Contextual logging
- Performance optimization
"""

import os
import sys
import json
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict, Any


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra context if provided
        if hasattr(record, 'context'):
            log_data['context'] = record.context

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


def setup_logger(
    name: str,
    log_dir: str,
    log_file: str = "app.log",
    level: str = "INFO",
    json_format: bool = False,
    rotation: str = "size",  # "size", "time", or "none"
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console_output: bool = True,
    error_separate: bool = True,
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name (e.g., "bigrag.api")
        log_dir: Directory for log files
        log_file: Log file name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (default: False)
        rotation: Rotation strategy ("size", "time", "none")
        max_bytes: Max file size for rotation (default: 10MB)
        backup_count: Number of backup files to keep
        console_output: Print logs to console (default: True)
        error_separate: Create separate error.log file (default: True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Choose formatter
    if json_format:
        formatter = JSONFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    # File handler with rotation
    if rotation == "size":
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
    elif rotation == "time":
        file_handler = TimedRotatingFileHandler(
            log_path,
            when='midnight',
            interval=1,
            backupCount=backup_count,
            encoding='utf-8'
        )
    else:
        file_handler = logging.FileHandler(log_path, encoding='utf-8')

    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(file_handler)

    # Separate error-only handler
    if error_separate:
        error_log_path = os.path.join(log_dir, "error.log")
        if rotation == "size":
            error_handler = RotatingFileHandler(
                error_log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        elif rotation == "time":
            error_handler = TimedRotatingFileHandler(
                error_log_path,
                when='midnight',
                interval=1,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            error_handler = logging.FileHandler(error_log_path, encoding='utf-8')

        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        logger.addHandler(error_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        # Use simpler format for console
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger by name"""
    return logging.getLogger(name)


def reset_logger(name: str):
    """Remove all handlers from logger (useful for testing)"""
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter for adding context to log messages"""

    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs["extra"] = {}
        if 'context' not in kwargs['extra']:
            kwargs['extra']['context'] = {}
        kwargs['extra']['context'].update(self.extra)
        return msg, kwargs


def add_context(logger: logging.Logger, **context: Any) -> ContextAdapter:
    """
    Add context to logger for structured logging

    Usage:
        ctx_logger = add_context(logger, request_id="abc123", user_id="user456")
        ctx_logger.info("Processing request")
    """
    return ContextAdapter(logger, context)


# Backward compatibility with old logging system
def set_logger(log_file: str, level: str = "INFO"):
    """
    Legacy function for backward compatibility

    This maintains compatibility with old code using:
        from bigrag.utils import set_logger
        set_logger(log_file)
    """
    log_dir = os.path.dirname(log_file)
    log_filename = os.path.basename(log_file)

    return setup_logger(
        name="bigrag",
        log_dir=log_dir if log_dir else ".",
        log_file=log_filename,
        level=level,
        rotation="size",
        max_bytes=10 * 1024 * 1024,
        backup_count=5,
        console_output=False  # Old behavior was file-only
    )
