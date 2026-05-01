"""inkstrip — remove handwriting from documents."""

from inkstrip.api import remove_handwriting
from inkstrip.config import InkstripConfig
from inkstrip.types import BBox, ProgressEvent, RemoveResult

__version__ = "0.1.0"

__all__ = [
    "remove_handwriting",
    "InkstripConfig",
    "BBox",
    "ProgressEvent",
    "RemoveResult",
    "__version__",
]
