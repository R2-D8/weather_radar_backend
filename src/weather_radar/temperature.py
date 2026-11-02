"""Temperature maps from ICON-EU, corrected for the real ground height.

The store keeps one grid per hour (the 2 m temperature in degrees C as float16, about 1.8 MB) from
the newest model run that covers that hour, from [past] hours ago to [future] hours ahead. Past
hours therefore come from the first hours of recent runs, which are close to the observed weather.

**Height correction.** On a 7 km grid the model's ground is smoothed: an Alpine valley floor may
sit 900 m lower than the model thinks, and air there is several degrees warmer. Each pixel is
corrected with a standard lapse rate: T = T_model + 6.5 K/km * (model ground - real ground). This is
what most point forecasts do too; it can be wrong under winter inversions (cold air pooled in
valleys), where the valley is colder than the correction says.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from weather_radar import icon
from weather_radar.icon import IconClient, IconError
from weather_radar.terrain import TILE_SIZE, Heights

log = logging.getLogger(__name__)

LAPSE_RATE_K_PER_M = 0.0065
HOUR = 3600

_FILE = re.compile(r"^(\d+)_(\d+)\.npy$")


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


def sample(
    grid: NDArray[np.floating], lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> NDArray[np.float32]:
    """Bilinear sample of an ICON-EU grid at points; NaN outside the model area."""
    row = (lat - icon.GRID_LAT0) / icon.GRID_STEP
    col = (lon - icon.GRID_LON0) / icon.GRID_STEP
    inside = (row >= 0) & (row <= icon.GRID_ROWS - 1) & (col >= 0) & (col <= icon.GRID_COLS - 1)
    r0 = np.clip(np.floor(row).astype(np.int64), 0, icon.GRID_ROWS - 2)
    c0 = np.clip(np.floor(col).astype(np.int64), 0, icon.GRID_COLS - 2)
    fr = np.clip(row - r0, 0.0, 1.0)
    fc = np.clip(col - c0, 0.0, 1.0)
    v00 = grid[r0, c0].astype(np.float64)
    v01 = grid[r0, c0 + 1].astype(np.float64)
    v10 = grid[r0 + 1, c0].astype(np.float64)
    v11 = grid[r0 + 1, c0 + 1].astype(np.float64)
    value = (v00 * (1 - fc) + v01 * fc) * (1 - fr) + (v10 * (1 - fc) + v11 * fc) * fr
    return np.where(inside, value, np.nan).astype(np.float32)


def corrected(
    t_model: NDArray[np.float32], model_ground: NDArray[np.float32], ground: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Temperature moved from the model's ground height to the real one (sea counts as 0 m)."""
    real = np.maximum(ground, 0.0)
    result: NDArray[np.float32] = (t_model + LAPSE_RATE_K_PER_M * (model_ground - real)).astype(
        np.float32
    )
    return result


def tile_overlaps_model(z: int, x: int, y: int) -> bool:
    n = 2**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat1 = icon.GRID_LAT0 + (icon.GRID_ROWS - 1) * icon.GRID_STEP
    lon1 = icon.GRID_LON0 + (icon.GRID_COLS - 1) * icon.GRID_STEP
    return not (east < icon.GRID_LON0 or west > lon1 or north < icon.GRID_LAT0 or south > lat1)


@lru_cache(maxsize=1)
def empty_tile() -> bytes:
    image = Image.frombytes("P", (TILE_SIZE, TILE_SIZE), bytes(TILE_SIZE * TILE_SIZE))
    image.putpalette(bytes(3))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, transparency=bytes([0]), bits=1)
    return buffer.getvalue()


def render_tile(
    grid: NDArray[np.floating],
    model_ground: NDArray[np.floating],
    ground: Heights,
    palette: TemperaturePalette,
    z: int,
    x: int,
    y: int,
) -> bytes:
    """An 8-bit palette PNG of the height-corrected temperature over tile z/x/y."""
    lat, lon = tile_lonlat(z, x, y)
    t = corrected(sample(grid, lat, lon), sample(model_ground, lat, lon), ground)
    index = palette.bands(t)
    if not index.any():
        return empty_tile()
    rgb, alpha = palette.png_palette()
    image = Image.frombytes("P", (TILE_SIZE, TILE_SIZE), index.tobytes())
    image.putpalette(rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, transparency=alpha)
    return buffer.getvalue()


def point_value(
    grid: NDArray[np.floating],
    model_ground: NDArray[np.floating],
    lat: float,
    lon: float,
    ground: float,
) -> float | None:
    """Height-corrected temperature at a point, or None outside the model area."""
    la = np.array([lat])
    lo = np.array([lon])
    here = np.array([ground], np.float32)
    t = corrected(sample(grid, la, lo), sample(model_ground, la, lo), here)
    value = float(t[0])
    return None if math.isnan(value) else value


# --- storage ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hour:
    #: Valid time and model run start, both epoch seconds.
    valid: int
    run: int


class TemperatureStore:
    """Files `{valid}_{run}.npy` (float16 degrees C) and `model_ground.npy` (float32 metres)."""

    def __init__(self, directory: Path, open_grids: int = 6) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._open: OrderedDict[Hour, NDArray[np.float16]] = OrderedDict()
        self._open_max = open_grids
        self._ground: NDArray[np.float32] | None = None

    @property
    def ground_path(self) -> Path:
        return self._dir / "model_ground.npy"

    def hours(self) -> list[Hour]:
        """Stored hours, oldest first (one per valid time: the newest run)."""
        best: dict[int, int] = {}
        for path in self._dir.iterdir():
            m = _FILE.match(path.name)
            if m:
                valid, run = int(m.group(1)), int(m.group(2))
                if run > best.get(valid, -1):
                    best[valid] = run
        return [Hour(v, best[v]) for v in sorted(best)]

    def has_ground(self) -> bool:
        return self.ground_path.exists()

    def save_ground(self, heights: NDArray[np.float32]) -> None:
        _save_atomically(self.ground_path, heights.astype(np.float32))
        self._ground = None

    def model_ground(self) -> NDArray[np.float32]:
        if self._ground is None:
            self._ground = cast(NDArray[np.float32], np.load(self.ground_path, mmap_mode="r"))
        return self._ground

    def save(self, hour: Hour, celsius: NDArray[np.float32]) -> None:
        _save_atomically(self._dir / f"{hour.valid}_{hour.run}.npy", celsius.astype(np.float16))
        # Older runs for the same hour are no longer needed.
        for path in self._dir.glob(f"{hour.valid}_*.npy"):
            m = _FILE.match(path.name)
            if m and int(m.group(2)) < hour.run:
                path.unlink(missing_ok=True)

    def load(self, hour: Hour) -> NDArray[np.float16]:
        grid = self._open.get(hour)
        if grid is not None:
            self._open.move_to_end(hour)
            return grid
        path = self._dir / f"{hour.valid}_{hour.run}.npy"
        try:
            # Memory-mapped: a tile only reads the few pages of the grid it covers.
            grid = cast(NDArray[np.float16], np.load(path, mmap_mode="r"))
        except FileNotFoundError:
            raise KeyError(hour) from None
        self._open[hour] = grid
        while len(self._open) > self._open_max:
            self._open.popitem(last=False)
        return grid

    def prune(self, oldest_valid: int) -> int:
        removed = 0
        for path in self._dir.iterdir():
            m = _FILE.match(path.name)
            if m and int(m.group(1)) < oldest_valid:
                path.unlink(missing_ok=True)
                self._open = OrderedDict(
                    (h, g) for h, g in self._open.items() if h.valid >= oldest_valid
                )
                removed += 1
        return removed


def _save_atomically(path: Path, array: NDArray[np.generic]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    tmp.replace(path)


# --- ingest -----------------------------------------------------------------------------------


@dataclass
class TemperatureStatus:
    last_success: float | None = None
    last_error: str | None = None
    downloaded_total: int = 0
    newest_run: int | None = None


#: Runs looked at: the newest four 6-hourly runs cover the past day, which is all DWD still has.
RUNS_CONSIDERED = 4
#: At most this many files per cycle, with a short pause between them, to be polite to DWD.
MAX_DOWNLOADS_PER_CYCLE = 60
DOWNLOAD_PAUSE_SECONDS = 0.5
#: A run is used until it is up to this old (the next one appears about 8.5 h after its
#: predecessor started), so each run is read this much further ahead than the window needs.
RUN_AGE_MARGIN = timedelta(hours=12)
#: ICON-EU is hourly up to +78 h (3-hourly after that, which this code does not handle).
MAX_HOURLY_STEP = 78


def max_step(future: timedelta) -> int:
    """The furthest forecast hour read from a run, so the window always reaches [future]."""
    return min(int((future + RUN_AGE_MARGIN).total_seconds()) // HOUR, MAX_HOURLY_STEP)


def plan_downloads(
    now: float,
    available_runs: list[int],
    stored: dict[int, int],
    past: timedelta,
    future: timedelta,
) -> list[Hour]:
    """Which (valid hour, run) files to fetch so every hour in the window has its newest run.

    The window runs from [past] before to [future] after the current hour. [available_runs] are
    complete runs on DWD's server; [stored] maps valid hour -> stored run. An hour uses the
    newest run that started at or before it and reaches it (see [max_step]). Newest hours first,
    so the forecast is complete soonest after a new run.
    """
    if not available_runs:
        return []
    this_hour = int(now) // HOUR * HOUR
    first = this_hour - int(past.total_seconds())
    last = this_hour + int(future.total_seconds())
    reach = max_step(future) * HOUR
    wanted: list[Hour] = []
    for valid in range(last, first - 1, -HOUR):
        candidates = [r for r in available_runs if r <= valid <= r + reach]
        if not candidates:
            continue
        best = max(candidates)
        if stored.get(valid, -1) < best:
            wanted.append(Hour(valid, best))
    return wanted


class TemperatureIngestor:
    """Keeps the store filled from DWD: checks for new runs every [poll_seconds]."""

    def __init__(
        self,
        client: IconClient,
        store: TemperatureStore,
        past: timedelta,
        future: timedelta,
        poll_seconds: float = 600.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._store = store
        self._past = past
        self._future = future
        self._poll = poll_seconds
        self._clock = clock
        self._sleep = sleep
        #: Runs known to be complete (they never change once complete).
        self._complete: set[int] = set()
        self.status = TemperatureStatus()

    async def run(self) -> None:
        while True:
            try:
                await self.cycle()
                self.status.last_success = self._clock()
                self.status.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as e:  # keep the loop alive whatever happens
                log.warning("temperature update failed: %s", e)
                self.status.last_error = str(e)
            await self._sleep(self._poll)

    async def available_runs(self) -> list[int]:
        """Complete runs among the newest few: their last needed hour is on the server."""
        last_step = max_step(self._future)
        runs = []
        for run in icon.latest_run_ids(self._clock(), RUNS_CONSIDERED + 1):
            if run in self._complete or await self._client.exists(
                self._client.t2m_url(run, last_step)
            ):
                self._complete.add(run)
                runs.append(run)
        self._complete &= set(runs)
        return sorted(runs, reverse=True)[:RUNS_CONSIDERED]

    async def cycle(self) -> int:
        """One update; returns the number of hours downloaded."""
        runs = await self.available_runs()
        if not runs:
            return 0
        self.status.newest_run = runs[0]
        if not self._store.has_ground():
            data = await self._client.download(self._client.hsurf_url(runs[0]))
            if data is None:
                raise IconError("model ground height file missing")
            self._store.save_ground(await asyncio.to_thread(icon.decode, data))
        stored = {h.valid: h.run for h in self._store.hours()}
        todo = plan_downloads(self._clock(), runs, stored, self._past, self._future)
        done = 0
        for hour in todo[:MAX_DOWNLOADS_PER_CYCLE]:
            step = (hour.valid - hour.run) // HOUR
            data = await self._client.download(self._client.t2m_url(hour.run, step))
            if data is None:
                log.info("run %d step %d not on the server, skipping", hour.run, step)
                continue
            kelvin = await asyncio.to_thread(icon.decode, data)
            await asyncio.to_thread(self._store.save, hour, kelvin - np.float32(273.15))
            done += 1
            self.status.downloaded_total += 1
            await self._sleep(DOWNLOAD_PAUSE_SECONDS)
        oldest = int(self._clock()) // HOUR * HOUR - int(self._past.total_seconds())
        self._store.prune(oldest)
        if done:
            log.info("temperature: %d hours updated (newest run %s)", done, icon.run_time(runs[0]))
        return done
