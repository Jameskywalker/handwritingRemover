"""Progress callback helpers."""

from __future__ import annotations

from typing import Callable

from inkstrip.types import ProgressCallback, ProgressEvent, Stage


def emit(
    cb: ProgressCallback | None,
    stage: Stage,
    *,
    page_idx: int = 0,
    total_pages: int = 1,
    message: str = "",
) -> None:
    if cb is None:
        return
    cb(ProgressEvent(stage=stage, page_idx=page_idx, total_pages=total_pages, message=message))


def chain(*callbacks: ProgressCallback | None) -> ProgressCallback | None:
    """Compose multiple callbacks into one. None entries are dropped."""
    real = [c for c in callbacks if c is not None]
    if not real:
        return None
    if len(real) == 1:
        return real[0]

    def combined(ev: ProgressEvent) -> None:
        for c in real:
            c(ev)

    return combined


def make_rich_callback() -> tuple[ProgressCallback, Callable[[], None]]:
    """Return (callback, finalize). Callback prints a single rich-formatted line per event.

    Kept tiny — heavier rich.Progress is wired only inside the CLI entrypoint to avoid
    surprising library users with imported global console state.
    """
    from rich.console import Console

    console = Console()

    def cb(ev: ProgressEvent) -> None:
        prefix = f"[{ev.page_idx + 1}/{ev.total_pages}]" if ev.total_pages > 1 else ""
        console.log(f"{prefix} {ev.stage}: {ev.message}".strip())

    def finalize() -> None:
        pass

    return cb, finalize
