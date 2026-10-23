"""Turning a radar frame into a map tile (a small palette PNG)."""

from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from weather_radar import geo
from weather_radar.frames import Grid, RadarFrame
from weather_radar.palettes import Palette

TILE_SIZES = (256, 512)


def _bilinear(grid: Grid, col: NDArray[np.float64], row: NDArray[np.float64]) -> Grid:
    """Smoothly sample [grid] at fractional cell coordinates; NaN outside coverage.

    Missing neighbours are left out and the remaining weights renormalised, so rain at the edge
    of the coverage area is not dimmed. Positions that are mostly outside the data become NaN.
    """
    rows, cols = grid.shape
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    fc = col - c0
    fr = row - r0
    weighted = np.zeros(col.shape, dtype=np.float64)
    weights = np.zeros(col.shape, dtype=np.float64)
    for dr, wr in ((0, 1.0 - fr), (1, fr)):
        for dc, wc in ((0, 1.0 - fc), (1, fc)):
            rr = r0 + dr
            cc = c0 + dc
            inside = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
            values = np.full(col.shape, np.nan, dtype=np.float64)
            values[inside] = grid[rr[inside], cc[inside]]
            valid = ~np.isnan(values)
            w = wr * wc
            weighted += np.where(valid, values * w, 0.0)
            weights += np.where(valid, w, 0.0)
    sampled = np.where(weights >= 0.5, weighted / np.maximum(weights, 1e-12), np.nan)
    return sampled.astype(np.float32)


def sample_tile(frame: RadarFrame, z: int, x: int, y: int, size: int) -> Grid:
    """Rain rate in mm/h at every pixel of a tile (NaN where there is no data)."""
    level = geo.detail_level(z, size)
    col, row = geo.tile_grid_coords(z, x, y, size, level)
    return _bilinear(frame.level(level), col, row)


@lru_cache(maxsize=len(TILE_SIZES))
def empty_tile(size: int) -> bytes:
    """A fully transparent tile; the same bytes are reused for every tile without rain."""
    image = Image.frombytes("P", (size, size), bytes(size * size))
    image.putpalette(bytes(3))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, transparency=bytes([0]), bits=1)
    return buffer.getvalue()


def render_tile(
    frame: RadarFrame, palette: Palette, z: int, x: int, y: int, size: int = 256
) -> bytes:
    """PNG bytes of the tile; no rain, or outside the coverage, gives the shared empty tile."""
    if size not in TILE_SIZES:
        raise ValueError(f"tile size must be one of {TILE_SIZES}")
    if not geo.tile_overlaps_coverage(z, x, y):
        return empty_tile(size)
    rain = sample_tile(frame, z, x, y, size)
    thresholds = np.asarray(palette.thresholds, dtype=np.float32)
    # Band 0 means "below the first threshold or no data"; band i+1 starts at thresholds[i].
    bands = np.searchsorted(thresholds, np.nan_to_num(rain, nan=-1.0), side="right")
    if not bands.any():
        return empty_tile(size)
    rgb, alpha = palette.png_palette()
    image = Image.frombytes("P", (size, size), bands.astype(np.uint8).tobytes())
    image.putpalette(rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, transparency=alpha, bits=4)
    return buffer.getvalue()
