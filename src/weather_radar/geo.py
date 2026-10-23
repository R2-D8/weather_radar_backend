"""Geometry: where a map tile's pixels fall on the radar grid.

The DPC product is a 1200 x 1400 grid of 1 km cells in a Transverse Mercator projection centred
on 42 N 12.5 E (WGS84), with the top-left corner at (-600000, 650000) m. These numbers come from
the GeoTIFF tags; `frames.decode_geotiff` checks every file against them, so a changed product
is rejected instead of drawn in the wrong place.

Map tiles use Web Mercator (EPSG:3857, the "slippy map" scheme: zoom z has 2**z x 2**z tiles).
"""

from __future__ import annotations

import math
import threading

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer

GRID_COLS = 1200
GRID_ROWS = 1400
GRID_PIXEL_M = 1000.0
#: Left edge of column 0 and top edge of row 0, in radar-projection metres.
GRID_ORIGIN_X = -600_000.0
GRID_ORIGIN_Y = 650_000.0
RADAR_PROJ = "+proj=tmerc +lat_0=42 +lon_0=12.5 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
#: Latitude used to turn Web Mercator metres into ground metres when choosing a detail level.
CENTRE_LATITUDE = 42.0

#: Half the width of the Web Mercator world in metres.
HALF_WORLD = 20_037_508.342789244

_local = threading.local()


def _to_radar() -> Transformer:
    # pyproj transformers are not documented as thread-safe, and tiles render in worker threads.
    t: Transformer | None = getattr(_local, "to_radar", None)
    if t is None:
        t = Transformer.from_crs("EPSG:3857", RADAR_PROJ, always_xy=True)
        _local.to_radar = t
    return t


def _from_lonlat() -> Transformer:
    t: Transformer | None = getattr(_local, "from_lonlat", None)
    if t is None:
        t = Transformer.from_crs("EPSG:4326", RADAR_PROJ, always_xy=True)
        _local.from_lonlat = t
    return t


def lonlat_to_cell(lon: float, lat: float) -> tuple[int, int] | None:
    """(row, column) of the 1 km grid cell containing a point, or None if outside the grid."""
    x, y = _from_lonlat().transform(lon, lat)
    col = math.floor((x - GRID_ORIGIN_X) / GRID_PIXEL_M)
    row = math.floor((GRID_ORIGIN_Y - y) / GRID_PIXEL_M)
    if 0 <= col < GRID_COLS and 0 <= row < GRID_ROWS:
        return row, col
    return None


def _coverage_bounds() -> tuple[float, float, float, float]:
    """Bounding box (min_x, min_y, max_x, max_y) of the radar grid in Web Mercator metres."""
    back = Transformer.from_crs(RADAR_PROJ, "EPSG:3857", always_xy=True)
    xs = np.linspace(GRID_ORIGIN_X, GRID_ORIGIN_X + GRID_COLS * GRID_PIXEL_M, 80)
    ys = np.linspace(GRID_ORIGIN_Y - GRID_ROWS * GRID_PIXEL_M, GRID_ORIGIN_Y, 80)
    edge_x = np.concatenate([xs, xs, np.full_like(ys, xs[0]), np.full_like(ys, xs[-1])])
    edge_y = np.concatenate([np.full_like(xs, ys[0]), np.full_like(xs, ys[-1]), ys, ys])
    wx, wy = back.transform(edge_x, edge_y)
    return float(np.min(wx)), float(np.min(wy)), float(np.max(wx)), float(np.max(wy))


COVERAGE_BOUNDS = _coverage_bounds()


def metres_per_pixel(z: int, size: int) -> float:
    """Web Mercator metres per tile pixel at zoom [z] for tiles of [size] pixels."""
    return 2 * HALF_WORLD / (size * (1 << z))


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) of a tile in Web Mercator metres; y counts down from the top."""
    span = 2 * HALF_WORLD / (1 << z)
    min_x = -HALF_WORLD + x * span
    max_y = HALF_WORLD - y * span
    return min_x, max_y - span, min_x + span, max_y


def tile_overlaps_coverage(z: int, x: int, y: int) -> bool:
    """Cheap test used to skip rendering tiles that cannot contain radar data."""
    min_x, min_y, max_x, max_y = tile_bounds(z, x, y)
    c_min_x, c_min_y, c_max_x, c_max_y = COVERAGE_BOUNDS
    return not (max_x < c_min_x or min_x > c_max_x or max_y < c_min_y or min_y > c_max_y)


MAX_LEVEL = 3


def detail_level(z: int, size: int) -> int:
    """Which pre-averaged grid to sample: 0 is the full 1 km grid, each level halves the detail.

    Sampling the full grid at low zoom would skip most cells and give a noisy picture, so zoomed
    out tiles read an averaged copy whose cells are about as big as the tile's pixels.
    """
    ground = metres_per_pixel(z, size) * math.cos(math.radians(CENTRE_LATITUDE))
    level = math.floor(math.log2(max(ground / GRID_PIXEL_M, 1.0)))
    return max(0, min(MAX_LEVEL, level))


def tile_grid_coords(
    z: int, x: int, y: int, size: int, level: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fractional (column, row) in the level-[level] grid for every pixel centre of a tile.

    Cell centres sit at index + 0.5 cell sizes, so an integer coordinate is the middle of a cell.
    """
    res = metres_per_pixel(z, size)
    px = x * size + np.arange(size) + 0.5
    py = y * size + np.arange(size) + 0.5
    xx, yy = np.meshgrid(-HALF_WORLD + px * res, HALF_WORLD - py * res)
    rx, ry = _to_radar().transform(xx, yy)
    scale = GRID_PIXEL_M * 2**level
    col = (np.asarray(rx) - GRID_ORIGIN_X) / scale - 0.5
    row = (GRID_ORIGIN_Y - np.asarray(ry)) / scale - 0.5
    return col, row
