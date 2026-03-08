"""
Centralised logging setup.
Call get_logger(__name__) in every module.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from config.settings import settings


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring root handlers only once."""
    root = logging.getLogger()

    if not root.handlers:
        _configure_root_logger()

    return logging.getLogger(name)


def _configure_root_logger() -> None:
    cfg = settings.logging
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # rotating file handler
    if cfg.log_to_file:
        Path(cfg.log_filename).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            cfg.log_filename,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
