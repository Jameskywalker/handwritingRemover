"""Command-line interface for inkstrip."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from inkstrip import __version__
from inkstrip.api import remove_handwriting
from inkstrip.config import InkstripConfig
from inkstrip.utils.progress import make_rich_callback

app = typer.Typer(help="Remove handwriting from documents.", no_args_is_help=True)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"inkstrip {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    pass


@app.command()
def run(
    source: Path = typer.Argument(..., exists=True, readable=True, help="Input image or PDF."),
    output: Path = typer.Argument(..., help="Output path (PNG/JPG for images, PDF for PDFs)."),
    photo: bool = typer.Option(False, "--photo", help="Apply photo preprocessing (M2)."),
    dpi: int = typer.Option(300, "--dpi", help="Render DPI for scanned PDFs (M2)."),
    device: str = typer.Option("auto", "--device", help="auto / cuda / cpu / mps."),
    inpainter: str = typer.Option("lama_torch", "--inpainter", help="lama_torch or lama_onnx."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Remove handwriting from a single document."""
    cfg = InkstripConfig(
        photo_mode=photo,
        render_dpi=dpi,
        device=device,  # type: ignore[arg-type]
        inpainter=inpainter,  # type: ignore[arg-type]
        verbose=not quiet,
    )

    cb = None if quiet else make_rich_callback()[0]
    result = remove_handwriting(source, output, config=cfg, progress=cb)

    page = result.pages[0] if result.pages else None
    if page is not None:
        console.print(
            f"[bold green]done[/]: {result.output_path} "
            f"({page.bbox_count} bbox, mask cov {page.mask_coverage * 100:.2f}%, "
            f"{page.elapsed_ms:.0f} ms)"
        )
    for w in result.warnings:
        console.print(f"[yellow]warning[/]: {w}")


@app.command("download-weights")
def download_weights(
    name: Optional[str] = typer.Option(None, "--name", help="Specific model to fetch; default = all."),
) -> None:
    """Pre-download model weights for offline use."""
    from inkstrip.models.weights import prefetch_all

    names = [name] if name else None
    paths = prefetch_all(names=names)
    for n, p in paths.items():
        console.print(f"[green]{n}[/] → {p}")


if __name__ == "__main__":
    app()
