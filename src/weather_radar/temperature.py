"""Temperature maps from ICON-D2 and ICON-EU, corrected for the real ground height.

The 2 m temperature of each hour comes from ICON-D2 (2 km) where it has data, faded into ICON-EU
(7 km) towards the edge of its area, and from ICON-EU alone elsewhere and beyond D2's +48 h (see
`modeldata`). Grids are stored in degrees C as float16.

**Height correction.** On a 7 km grid the model's ground is smoothed: an Alpine valley floor may
sit 900 m lower than the model thinks, and air there is several degrees warmer. Each pixel is
corrected with a standard lapse rate: T = T_model + 6.5 K/km * (model ground - real ground). This is
what most point forecasts do too; it can be wrong under winter inversions (cold air pooled in
valleys), where the valley is colder than the correction says. ICON-D2's finer ground makes the
correction smaller, and it resolves some of the cold pools itself.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from weather_radar.icon import ICON_D2, ICON_EU, Model
from weather_radar.modeldata import Grid, blend, sample
from weather_radar.terrain import TILE_SIZE, Heights

LAPSE_RATE_K_PER_M = 0.0065
HOUR = 3600


@dataclass(frozen=True)
class TemperaturePalette:
    """Fixed colour bands: band i covers [low + i * step, low + (i + 1) * step).

    Like the rain palettes, an existing palette must never change its colours (phones keep tiles
    for good): make a new one with a new name instead.
    """

    name: str
    low: float
    high: float
    step: float
    #: (degrees C, RGB) colour stops; band colours are interpolated at the band centre.
    stops: tuple[tuple[float, tuple[int, int, int]], ...]

    @property
    def band_count(self) -> int:
        return round((self.high - self.low) / self.step)

    def band_start(self, i: int) -> float:
        return self.low + i * self.step

    def colour(self, t: float) -> tuple[int, int, int]:
        stops = self.stops
        if t <= stops[0][0]:
            return stops[0][1]
        for (a, ca), (b, cb) in zip(stops, stops[1:], strict=False):
            if t <= b:
                f = (t - a) / (b - a)
                r, g, bl = (round(ca[k] + (cb[k] - ca[k]) * f) for k in range(3))
                return r, g, bl
        return stops[-1][1]

    def band_colours(self) -> list[tuple[int, int, int]]:
        return [self.colour(self.band_start(i) + self.step / 2) for i in range(self.band_count)]

    def bands(self, t: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Palette index per pixel: 0 for no data, 1 + band otherwise (clamped to the ends)."""
        index = np.floor((np.nan_to_num(t, nan=0.0) - self.low) / self.step)
        index = np.clip(index, 0, self.band_count - 1) + 1
        return np.where(np.isnan(t), 0, index).astype(np.uint8)

    def png_palette(self) -> tuple[bytes, bytes]:
        rgb = bytes([0, 0, 0] + [c for colour in self.band_colours() for c in colour])
        alpha = bytes([0] + [255] * self.band_count)
        return rgb, alpha


TEMP_2C = TemperaturePalette(
    name="temp2c",
    low=-30.0,
    high=46.0,
    step=2.0,
    stops=(
        (-30.0, (60, 0, 90)),
        (-20.0, (110, 40, 170)),
        (-10.0, (40, 80, 200)),
        (0.0, (120, 190, 240)),
        (5.0, (90, 200, 170)),
        (10.0, (80, 180, 80)),
        (15.0, (180, 210, 60)),
        (20.0, (250, 220, 60)),
        (25.0, (250, 150, 40)),
        (30.0, (230, 60, 40)),
        (40.0, (140, 0, 40)),
        (46.0, (90, 0, 30)),
    ),
)
TEMPERATURE_PALETTES = {TEMP_2C.name: TEMP_2C}
DEFAULT_TEMPERATURE_PALETTE = TEMP_2C.name


# --- sampling ---------------------------------------------------------------------------------


def tile_lonlat(
    z: int, x: int, y: int, size: int = TILE_SIZE
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(lat, lon) of every pixel centre of a Web Mercator tile, each shaped (size, size)."""
    i = (np.arange(size, dtype=np.float64) + 0.5) / size
    lon = (x + i) / 2**z * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * (y + i) / 2**z))))
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return lat_grid, lon_grid


def corrected(
    t_model: NDArray[np.float32], model_ground: NDArray[np.float32], ground: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Temperature moved from the model's ground height to the real one (sea counts as 0 m)."""
    real = np.maximum(ground, 0.0)
    result: NDArray[np.float32] = (t_model + LAPSE_RATE_K_PER_M * (model_ground - real)).astype(
        np.float32
    )
    return result


@dataclass(frozen=True)
class ModelHour:
    """One hour's grids: ICON-EU always, ICON-D2 where it has the hour (with its fade weight)."""

    eu: NDArray[np.floating]
    eu_ground: NDArray[np.floating]
    d2: NDArray[np.floating] | None = None
    d2_ground: NDArray[np.floating] | None = None
    d2_weight: NDArray[np.floating] | None = None


#: Height matching on the ICON-D2 grid: cells within this many cells count...
MATCH_RADIUS = 3
#: ...weighted by distance (Gaussian, in cells)...
MATCH_SIGMA = 1.5
#: ...and by how close their ground is to the real one (falls to 1/e per this many metres).
MATCH_HEIGHT_M = 100.0


def height_matched(
    grid: NDArray[np.floating],
    model_ground: NDArray[np.floating],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    ground: Heights,
    model: Model = ICON_D2,
) -> Grid:
    """Temperature at points from the nearby model cells whose ground is closest to the real one.

    Each cell's value is first moved to the real height with the standard lapse rate, then the
    cells are averaged with weights that favour nearby cells and, strongly, cells at about the
    same height. In a 2 km grid an Alpine valley floor usually has a cell of its own, and on a calm
    night that cell holds the cold air the model pooled there; plain interpolation would mix in
    the slopes above and then warm the result by 6.5 K/km, which is what made valleys too warm.
    Checked against Open-Meteo's ICON-D2 values for seven Alpine and Po valley points on
    2026-11-02 20 UTC: mean error 0.3 C, against 0.5 C for interpolation (Sondrio 8.9 C instead
    of 10.2 C, reference 8.3 C).
    """
    real = np.maximum(ground, 0.0).astype(np.float64)
    row = (lat - model.lat0) / model.step
    col = (lon - model.lon0) / model.step
    r0 = np.floor(row).astype(np.int64)
    c0 = np.floor(col).astype(np.int64)
    total = np.zeros(lat.shape, dtype=np.float64)
    weights = np.zeros(lat.shape, dtype=np.float64)
    for dr in range(-MATCH_RADIUS + 1, MATCH_RADIUS + 1):
        rr = r0 + dr
        for dc in range(-MATCH_RADIUS + 1, MATCH_RADIUS + 1):
            cc = c0 + dc
            inside = (rr >= 0) & (rr < model.rows) & (cc >= 0) & (cc < model.cols)
            ri = np.clip(rr, 0, model.rows - 1)
            ci = np.clip(cc, 0, model.cols - 1)
            t = grid[ri, ci].astype(np.float64)
            z = model_ground[ri, ci].astype(np.float64)
            ok = inside & ~np.isnan(t) & ~np.isnan(z)
            distance2 = (rr - row) ** 2 + (cc - col) ** 2
            w = np.exp(-distance2 / (2 * MATCH_SIGMA**2) - np.abs(z - real) / MATCH_HEIGHT_M)
            w = np.where(ok, w, 0.0)
            total += w * np.where(ok, t + LAPSE_RATE_K_PER_M * (z - real), 0.0)
            weights += w
    # Far from every cell's height (deep below a ridge) the weights underflow: plain lapse rate.
    fallback = corrected(
        sample(grid, lat, lon, model), sample(model_ground, lat, lon, model), ground
    )
    matched = total / np.maximum(weights, 1e-300)
    return np.where(weights > 1e-12, matched, fallback).astype(np.float32)


def temperature_at(
    hour: ModelHour, lat: NDArray[np.float64], lon: NDArray[np.float64], ground: Heights
) -> Grid:
    """Height-corrected temperature at points; NaN outside ICON-EU's area."""
    t = corrected(sample(hour.eu, lat, lon), sample(hour.eu_ground, lat, lon), ground)
    if hour.d2 is None or hour.d2_ground is None or hour.d2_weight is None:
        return t
    t_d2 = height_matched(hour.d2, hour.d2_ground, lat, lon, ground)
    return blend(t, t_d2, sample(hour.d2_weight, lat, lon, ICON_D2))


def tile_overlaps_model(z: int, x: int, y: int) -> bool:
    """Whether tile z/x/y touches ICON-EU's area (ICON-D2's lies inside it)."""
    n = 2**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    m = ICON_EU
    return not (east < m.lon0 or west > m.lon1 or north < m.lat0 or south > m.lat1)


@lru_cache(maxsize=1)
def empty_tile() -> bytes:
    image = Image.frombytes("P", (TILE_SIZE, TILE_SIZE), bytes(TILE_SIZE * TILE_SIZE))
    image.putpalette(bytes(3))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, transparency=bytes([0]), bits=1)
    return buffer.getvalue()


def render_tile(
    hour: ModelHour, ground: Heights, palette: TemperaturePalette, z: int, x: int, y: int
) -> bytes:
    """An 8-bit palette PNG of the height-corrected temperature over tile z/x/y."""
    lat, lon = tile_lonlat(z, x, y)
    index = palette.bands(temperature_at(hour, lat, lon, ground))
    if not index.any():
        return empty_tile()
    rgb, alpha = palette.png_palette()
    image = Image.frombytes("P", (TILE_SIZE, TILE_SIZE), index.tobytes())
    image.putpalette(rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, transparency=alpha)
    return buffer.getvalue()


def point_value(hour: ModelHour, lat: float, lon: float, ground: float) -> float | None:
    """Height-corrected temperature at a point, or None outside the model area."""
    t = temperature_at(hour, np.array([lat]), np.array([lon]), np.array([ground], np.float32))
    value = float(t[0])
    return None if math.isnan(value) else value
