"""Pipeline ABC."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from inkstrip.config import InkstripConfig
from inkstrip.types import OutputLike, ProgressCallback, RemoveResult


@runtime_checkable
class Pipeline(Protocol):
    def run(
        self,
        source,
        output: OutputLike,
        cfg: InkstripConfig,
        *,
        progress: ProgressCallback | None = None,
    ) -> RemoveResult: ...
