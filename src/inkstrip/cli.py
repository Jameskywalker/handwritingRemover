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
    page_crop: Optional[bool] = typer.Option(
        None,
        "--page-crop/--no-page-crop",
        help="Auto-detect and warp page quadrilateral. Default inherits from --photo.",
    ),
    strategy: str = typer.Option(
        "color_red",
        "--strategy",
        help="Mask strategy: color_red | color_blue | color_any | yolo_morph | ocr_inverse.",
    ),
    dpi: int = typer.Option(300, "--dpi", help="Render DPI for scanned PDFs (M2)."),
    device: str = typer.Option("auto", "--device", help="auto / cuda / cpu / mps."),
    inpainter: str = typer.Option("lama_onnx", "--inpainter", help="Inpainting backend."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Remove handwriting from a single document."""
    cfg = InkstripConfig(
        photo_mode=photo,
        page_crop=page_crop,
        mask_strategy=strategy,  # type: ignore[arg-type]
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


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio link."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't auto-open browser."),
) -> None:
    """Launch the Gradio demo UI."""
    try:
        from inkstrip.web.app import serve as _serve
    except ImportError as e:
        raise typer.BadParameter(
            "Gradio not installed. Re-install with: pip install -e '.[ui]'"
        ) from e
    _serve(host=host, port=port, share=share, open_browser=not no_browser)


if __name__ == "__main__":
    app()
