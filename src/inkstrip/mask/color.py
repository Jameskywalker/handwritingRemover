"""Color-based handwriting detection.

When handwriting is in colored ink on a printed page, color thresholding
beats a learned detector — it's free, deterministic, language-agnostic, and
nails strokes pixel-perfect.

Why we use RGB channel-difference rather than HSV:
HSV thresholds fail catastrophically on aged / off-white paper. A
yellowed scan can land its paper hue in the [0..14] red arc and trip the
detector across the whole page (verified empirically: paper HSV mean
H=7, S=166 on the test fixture, indistinguishable from red ink under any
reasonable HSV cutoff). Channel-difference (R - G > τ AND R - B > τ)
keys on *relative* color rather than absolute hue, which separates
red ink (R≫G,B) from warm paper (R≈G≈B with R slightly higher).

Profiles below cover the common cases (red, blue, green); custom rules
go through `ChannelRule` directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ChannelRule:
    """Pixel test: dominant channel must beat each rival by `delta` and be at
    least `min_brightness`. Pure black/white/gray fall out automatically because
    R≈G≈B → all deltas ≈ 0.
    """
    dominant: str  # "R" | "G" | "B"
    rivals: tuple[str, ...]  # which channels to compare against
    delta: int = 35  # how many counts the dominant must exceed each rival
    min_brightness: int = 60  # filter out near-black


PROFILES: dict[str, ChannelRule] = {
    # `delta` calibrated on a yellowed-paper fixture where paper itself has
    # R-G ≈ 16, R-B ≈ 22, while red ink sits at R-G ≈ 68, R-B ≈ 76. delta=50
    # cleanly separates them and leaves margin for stroke edges. `min_brightness`
    # filters out shadowed regions that satisfy the channel rule but are too
    # dark to actually be ink — without this we mark print drop-shadows as red.
    "red": ChannelRule(dominant="R", rivals=("G", "B"), delta=50, min_brightness=120),
    "blue": ChannelRule(dominant="B", rivals=("R", "G"), delta=40, min_brightness=80),
    "green": ChannelRule(dominant="G", rivals=("R", "B"), delta=40, min_brightness=80),
}

# Special profile name that ORs together all colored profiles. Use this when
# the page has mixed-color annotations (red + blue + green pens).
ANY_COLORED = "any_colored"

_CHANNEL_IDX = {"R": 0, "G": 1, "B": 2}


def estimate_paper_color(image: np.ndarray, percentile: int = 70) -> tuple[int, int, int]:
    """Pick a representative paper RGB from the image's bright pixels.

    Uses the median of pixels above the `percentile`-th gray-value cutoff so
    we ignore printed text (dark) but include light shadows on paper.
    """
    gray = image.mean(axis=-1)
    cutoff = np.percentile(gray, percentile)
    bright = image[gray > cutoff]
    if bright.size == 0:
        return (255, 255, 255)
    med = np.median(bright, axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def detect_adaptive_mask(
    image: np.ndarray,
    *,
    profile: str = "red",
    threshold: int = 50,
    edge_threshold: int | None = None,
    dilate_px: int = 7,
    closing_px: int = 3,
    min_component_area: int = 12,
    protect_print: bool = True,
    print_threshold: int = 90,
) -> np.ndarray:
    """Adaptive color-ink detection that auto-calibrates to the page's paper.

    Computes a "color signal" per pixel relative to the estimated paper RGB:
        signal = (dom - paper_dom) + max(paper_rival - rival for rival in rivals)

    Pure paper has signal ≈ 0; colored ink has high signal regardless of
    whether the paper is bright white, off-white, or aged red. `threshold`
    therefore generalizes across photos with different lighting / paper
    color where a fixed-RGB rule wouldn't.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown color profile {profile!r}")
    rule = PROFILES[profile]
    if edge_threshold is None:
        edge_threshold = max(15, threshold // 2)

    paper_r, paper_g, paper_b = estimate_paper_color(image)
    paper_rgb = {"R": paper_r, "G": paper_g, "B": paper_b}

    img32 = image.astype(np.int16)
    dom = img32[..., _CHANNEL_IDX[rule.dominant]]
    rivals_arr = [img32[..., _CHANNEL_IDX[r]] for r in rule.rivals]
    paper_rivals = [paper_rgb[r] for r in rule.rivals]

    dom_excess = dom - paper_rgb[rule.dominant]
    rival_deficit = np.maximum.reduce([
        paper_rivals[i] - rivals_arr[i] for i in range(len(rivals_arr))
    ])
    signal = dom_excess + rival_deficit

    seed = signal >= threshold
    candidate = signal >= edge_threshold

    if edge_threshold < threshold:
        seed_u8 = seed.astype(np.uint8) * 255
        cand_u8 = candidate.astype(np.uint8) * 255
        prev = np.zeros_like(seed_u8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cur = seed_u8
        for _ in range(20):
            cur = cv2.bitwise_and(cv2.dilate(cur, kernel, iterations=1), cand_u8)
            if np.array_equal(cur, prev):
                break
            prev = cur
        mask = cur
    else:
        mask = seed.astype(np.uint8) * 255

    return _post_process(
        mask,
        dilate_px=dilate_px,
        closing_px=closing_px,
        min_component_area=min_component_area,
        image=image if protect_print else None,
        print_threshold=print_threshold,
    )


def detect_color_mask(
    image: np.ndarray,
    *,
    profile: str = "red",
    dilate_px: int = 5,
    closing_px: int = 3,
    min_component_area: int = 12,
    delta: int | None = None,
    min_brightness: int | None = None,
    edge_delta: int | None = None,
    protect_print: bool = True,
    print_threshold: int = 90,
) -> np.ndarray:
    """Build a binary mask covering pixels matching the color profile.

    Two-tier hysteresis to capture stroke edges without bleeding into paper:
      1. *Seeds*: pixels passing the strict `delta` rule — confidently red ink.
      2. *Candidates*: pixels passing a permissive `edge_delta` rule —
         possibly red but ambiguous (stroke anti-alias, ink bleed).
      3. Candidates are kept only if they're connected to a seed (4-conn).
         This keeps stroke borders and drops scattered paper noise.
      4. Closing → dilation → small-component pruning, same as before.

    `edge_delta` defaults to ~half of `delta` (max 30). Set both equal to
    disable hysteresis.
    """
    if profile == ANY_COLORED:
        # OR together every single-color profile. Each profile is run with its
        # own calibrated thresholds — that's the whole point of having profiles.
        masks = [
            detect_color_mask(
                image,
                profile=p,
                # Per-color masks: skip post-processing, we'll do it after the OR
                # so the closing/dilation kernels see the merged stroke set.
                dilate_px=0,
                closing_px=0,
                min_component_area=0,
                protect_print=False,  # apply once after merge
            )
            for p in PROFILES
        ]
        merged = np.zeros_like(masks[0])
        for m in masks:
            merged |= m
        return _post_process(
            merged,
            dilate_px=dilate_px,
            closing_px=closing_px,
            min_component_area=min_component_area,
            image=image if protect_print else None,
            print_threshold=print_threshold,
        )

    if profile not in PROFILES:
        raise ValueError(f"unknown color profile {profile!r}; have {sorted(PROFILES)}")
    rule = PROFILES[profile]
    eff_delta = rule.delta if delta is None else delta
    eff_brightness = rule.min_brightness if min_brightness is None else min_brightness
    eff_edge_delta = edge_delta if edge_delta is not None else max(15, min(30, eff_delta // 2))

    img32 = image.astype(np.int16)
    dom = img32[..., _CHANNEL_IDX[rule.dominant]]
    rivals_arr = [img32[..., _CHANNEL_IDX[r]] for r in rule.rivals]

    # Seeds need full brightness; candidates only need to be above near-black
    # so we pick up shadowed / aged-ink portions of the same stroke. The
    # connectivity step ensures dark-but-colored regions only count when they
    # touch a confident seed pixel — pure shadows on neutral paper don't pass.
    seed_bright = dom >= eff_brightness
    cand_bright = dom >= max(40, eff_brightness // 3)

    seed = seed_bright.copy()
    candidate = cand_bright.copy()
    for rv in rivals_arr:
        seed &= dom - rv >= eff_delta
        candidate &= dom - rv >= eff_edge_delta

    if eff_edge_delta < eff_delta:
        # Reconstruction by dilation: keep candidate pixels reachable from a
        # seed via 4-connected neighbours of also-candidate pixels.
        seed_u8 = seed.astype(np.uint8) * 255
        cand_u8 = candidate.astype(np.uint8) * 255
        prev = np.zeros_like(seed_u8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cur = seed_u8
        # iterate dilation∩candidate until stable (bounded by O(diameter))
        for _ in range(20):
            cur = cv2.bitwise_and(cv2.dilate(cur, kernel, iterations=1), cand_u8)
            if np.array_equal(cur, prev):
                break
            prev = cur
        mask = cur
    else:
        mask = seed.astype(np.uint8) * 255

    return _post_process(
        mask,
        dilate_px=dilate_px,
        closing_px=closing_px,
        min_component_area=min_component_area,
        image=image if protect_print else None,
        print_threshold=print_threshold,
    )


def _post_process(
    mask: np.ndarray,
    *,
    dilate_px: int,
    closing_px: int,
    min_component_area: int,
    image: np.ndarray | None = None,
    print_threshold: int = 90,
) -> np.ndarray:
    if closing_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_px, closing_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k, iterations=1)

    if image is not None:
        # Subtract printed-text pixels from the mask. Anything whose grayscale
        # value is below `print_threshold` AND that has neutral chroma
        # (R≈G≈B) is treated as black ink — keep it untouched. Without this,
        # dilation around colored strokes pulls adjacent printed glyphs into
        # the mask and LaMa wipes them out.
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rgb_max = image.max(axis=-1).astype(int)
        rgb_min = image.min(axis=-1).astype(int)
        chroma_low = (rgb_max - rgb_min) <= 25
        printed = (gray < print_threshold) & chroma_low
        mask = np.where(printed, 0, mask).astype(np.uint8)

    if min_component_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
                keep[labels == i] = 255
        mask = keep

    return mask


class ColorInkMaskBuilder:
    """Adapter so this can plug into ImagePipeline as a `mask_builder`.

    The pipeline normally calls `build(image, boxes)`; for color we ignore
    boxes (the detector won't even be queried — see ImagePipeline branch).
    """

    def __init__(
        self,
        profile: str = "red",
        dilate_px: int = 5,
        closing_px: int = 3,
        min_component_area: int = 12,
    ) -> None:
        self.profile = profile
        self.dilate_px = dilate_px
        self.closing_px = closing_px
        self.min_component_area = min_component_area

    def build(self, image: np.ndarray, boxes=None) -> np.ndarray:  # noqa: ARG002
        return detect_color_mask(
            image,
            profile=self.profile,
            dilate_px=self.dilate_px,
            closing_px=self.closing_px,
            min_component_area=self.min_component_area,
        )
