"""Classical paper-color fill inpainter.

Replaces masked pixels with a locally estimated paper colour (a weighted
box-filter of nearby unmasked pixels). Targets handwriting-on-paper
specifically:

- ~100x faster than LaMa (≈50 ms vs ≈5000 ms for 1500-px page).
- No "ghost" residue from edge anti-aliasing — the output pixel is the
  paper colour, not LaMa's reconstruction influenced by the surrounding
  greyscale gradient.

Trade-off: LaMa can re-synthesise printed-glyph texture if the mask
accidentally covers a printed character. Paper-fill cannot — it just
paints paper colour. To avoid wiping out printed text, the inpainter
narrows the mask to a "confident handwriting" zone before filling:

    confident = mask AND (hw_box_dilated ∪ color_layer)

Anything outside that zone (printed-region mask noise) is left untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from inkstrip.config import InkstripConfig
    from inkstrip.detect.hw_classifier import HwClassifier


class PaperFillInpainter:
    """Paper-colour fill, optionally narrowed to handwriting+color zones."""

    def __init__(
        self,
        cfg: "InkstripConfig",
        *,
        hw_classifier: "HwClassifier | None" = None,
        hw_box_dilate_px: int = 21,
        paper_threshold_offset: int = 25,
        sample_ksize: int = 51,
        min_samples: int = 30,
    ) -> None:
        self.cfg = cfg
        self.hw_classifier = hw_classifier
        self.hw_box_dilate_px = int(hw_box_dilate_px)
        self.paper_threshold_offset = int(paper_threshold_offset)
        self.sample_ksize = int(sample_ksize)
        self.min_samples = int(min_samples)
        self.last_effective_mask: np.ndarray | None = None

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            raise ValueError("image must be uint8")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be HxWx3 RGB")

        confident = self._narrow_mask(image, mask)
        self.last_effective_mask = confident
        return _paper_fill(
            image,
            confident,
            paper_threshold_offset=self.paper_threshold_offset,
            sample_ksize=self.sample_ksize,
            min_samples=self.min_samples,
        )

    def _narrow_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from inkstrip.mask.color import detect_color_mask

        h, w = image.shape[:2]
        # 1. coloured ink — always confident
        color_layer = detect_color_mask(
            image,
            profile="any_colored",
            dilate_px=5,
            closing_px=3,
            min_component_area=12,
            protect_print=True,
        )
        # 2. handwriting bboxes — confident
        hw_layer = np.zeros((h, w), dtype=np.uint8)
        if self.hw_classifier is not None:
            for box in self.hw_classifier.detect(image):
                x, y, bw, bh = box.bbox
                hw_layer[y : y + bh, x : x + bw] = 255
            if self.hw_box_dilate_px > 0:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (self.hw_box_dilate_px, self.hw_box_dilate_px),
                )
                hw_layer = cv2.dilate(hw_layer, k, iterations=1)

        zone = cv2.bitwise_or(hw_layer, color_layer)
        narrowed = cv2.bitwise_and(mask, zone)
        # always include color_layer (even if it slipped past the input mask)
        return cv2.bitwise_or(narrowed, color_layer)


def _paper_fill(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    paper_threshold_offset: int,
    sample_ksize: int,
    min_samples: int,
) -> np.ndarray:
    """Replace mask>0 pixels with the local mean of nearby paper pixels.

    The "paper" set = non-mask pixels whose grayscale falls within
    ``[paper_brightness - paper_threshold_offset, paper_brightness + 20]``.
    The lower bound excludes printed glyphs / shadows; the upper bound
    excludes specular highlights that would tint the fill toward pure white.

    Per-pixel fill colour = average RGB of all paper pixels inside an
    ``sample_ksize × sample_ksize`` neighbourhood, computed via box filter
    (so the cost is O(H·W) regardless of kernel size). Where fewer than
    ``min_samples`` paper pixels are available locally, fall back to the
    global paper colour.

    This avoids the nearest-neighbour "stroke ghost" artefact: each fill
    pixel sees an ensemble of paper samples instead of a single one, so
    the filled region reads as paper texture rather than projected stroke
    outlines.
    """
    from inkstrip.mask.color import estimate_paper_color

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Use the median (50th percentile) for the basis — the previously used
    # 70th percentile sat above the true paper colour, so the sampling band
    # leaned highlight-ward and the box-filter mean came out too white.
    paper_color = estimate_paper_color(rgb, percentile=50)
    paper_brightness = int(
        0.299 * paper_color[0] + 0.587 * paper_color[1] + 0.114 * paper_color[2]
    )
    lower = max(120, paper_brightness - paper_threshold_offset)
    upper = paper_brightness + 10

    # Exclude a small ring around the mask from the paper sample pool. The
    # pixels immediately adjacent to handwriting strokes are biased bright
    # (anti-aliasing halo, JPEG compensation, scanner local contrast) — if
    # we let them in, the box-filter mean tints the fill toward white the
    # closer we get to a stroke.
    halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask_with_halo = cv2.dilate(mask, halo_kernel, iterations=1)
    is_paper = (
        (mask_with_halo == 0) & (gray >= lower) & (gray <= upper)
    ).astype(np.float32)
    fill = mask > 0

    if not is_paper.any():
        out = rgb.copy()
        out[fill] = np.array(paper_color, dtype=np.uint8)
        return out

    rgb_f = rgb.astype(np.float32)
    ksz = (sample_ksize, sample_ksize)
    num = np.stack(
        [
            cv2.boxFilter(rgb_f[..., c] * is_paper, -1, ksz, normalize=False)
            for c in range(3)
        ],
        axis=-1,
    )
    den = cv2.boxFilter(is_paper, -1, ksz, normalize=False)
    sufficient = den >= min_samples
    safe_den = np.where(sufficient, den, 1.0)
    local = num / safe_den[..., None]
    global_paper = np.array(paper_color, dtype=np.float32)
    paper = np.where(sufficient[..., None], local, global_paper[None, None, :])

    out = rgb.copy()
    out[fill] = paper[fill].astype(np.uint8)
    return out
