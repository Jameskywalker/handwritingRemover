"""Internal logging helpers."""

from __future__ import annotations

import logging
import os

_LOG_NAME = "inkstrip"


def get_logger(name: str | None = None) -> logging.Logger:
    base = logging.getLogger(_LOG_NAME)
    if not base.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        base.addHandler(handler)
        level = os.environ.get("INKSTRIP_LOG", "WARNING").upper()
        base.setLevel(level)
    return base if name is None else base.getChild(name)
