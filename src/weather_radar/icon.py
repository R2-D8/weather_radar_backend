"""Client for DWD's ICON model output on its open data server (opendata.dwd.de).

Two models are used, both on regular latitude/longitude grids, rows south to north:

- **ICON-EU**: 0.0625 degrees (about 7 km), 657 x 1377 points from 29.5 N 23.5 W to 70.5 N 62.5 E,
  so it covers Italy and Finland. The 00, 06, 12 and 18 UTC runs are hourly to +78 h.
- **ICON-D2**: 0.02 degrees (about 2 km), 746 x 1215 points from 43.18 N 3.94 W to 58.08 N 20.34 E
  (Germany, the Alps and northern Italy down to about Florence). Runs every 3 hours, each hourly to
  +48 h; we use the same four runs a day as for ICON-EU, which halves the downloads. Points outside
  its (rotated) model area are missing in the files and become NaN here.

Runs appear on the server one to three hours after their start time, and the server keeps only
the newest run of each start hour, so anything older than about a day has to be kept by us.

Each field of each forecast hour is one bzip2-compressed GRIB2 file of 0.4 to 4 MB, for example
`.../icon-eu/grib/06/t_2m/icon-eu_europe_regular-lat-lon_single-level_2026110206_012_T_2M.grib2.bz2`
(2 m temperature of the 06 UTC run at +12 h). Licence: CC BY 4.0, source Deutscher Wetterdienst.
"""

from __future__ import annotations

import bz2
from dataclasses import dataclass
from datetime import UTC, datetime

import aiohttp
import eccodes
import numpy as np
from numpy.typing import NDArray

DWD_BASE = "https://opendata.dwd.de/weather/nwp"

#: The runs we use (the 03, 09, 15 and 21 UTC ICON-EU runs stop at +30 h).
RUN_HOURS = (0, 6, 12, 18)
RUN_STEP_SECONDS = 6 * 3600

#: A file is at most a few MB; anything much bigger is not what we asked for.
MAX_FILE_BYTES = 20 * 1024 * 1024

Field = NDArray[np.float32]


class IconError(Exception):
    """DWD could not be reached, or a file is not the grid we expect."""


def run_time(run_id: int) -> datetime:
    return datetime.fromtimestamp(run_id, tz=UTC)


def _stamp(run_id: int) -> str:
    return run_time(run_id).strftime("%Y%m%d%H")


@dataclass(frozen=True)
class Model:
    """One model's grid and file naming."""

    name: str
    rows: int
    cols: int
    #: Centre of row 0 / column 0 (the south-west corner) and the spacing, in degrees.
    lat0: float
    lon0: float
    step: float
    #: Furthest hourly forecast step of the runs we use.
    max_hourly_step: int

    @property
    def lat1(self) -> float:
        return self.lat0 + (self.rows - 1) * self.step

    @property
    def lon1(self) -> float:
        return self.lon0 + (self.cols - 1) * self.step

    def field_url(self, field: str, run_id: int, step: int, base: str = DWD_BASE) -> str:
        """Address of [field] (DWD's lower-case name, e.g. "t_2m") of a run at +[step] hours."""
        hh = run_time(run_id).strftime("%H")
        stamp = _stamp(run_id)
        if self.name == "icon-d2":
            return (
                f"{base}/icon-d2/grib/{hh}/{field}/icon-d2_germany_regular-lat-lon_single-level_"
                f"{stamp}_{step:03d}_2d_{field}.grib2.bz2"
            )
        return (
            f"{base}/icon-eu/grib/{hh}/{field}/icon-eu_europe_regular-lat-lon_single-level_"
            f"{stamp}_{step:03d}_{field.upper()}.grib2.bz2"
        )

    def hsurf_url(self, run_id: int, base: str = DWD_BASE) -> str:
        """Address of the model's ground height (the same for every run, published with each)."""
        hh = run_time(run_id).strftime("%H")
        stamp = _stamp(run_id)
        if self.name == "icon-d2":
            return (
                f"{base}/icon-d2/grib/{hh}/hsurf/icon-d2_germany_regular-lat-lon_time-invariant_"
                f"{stamp}_000_0_hsurf.grib2.bz2"
            )
        return (
            f"{base}/icon-eu/grib/{hh}/hsurf/icon-eu_europe_regular-lat-lon_time-invariant_"
            f"{stamp}_HSURF.grib2.bz2"
        )


ICON_EU = Model("icon-eu", 657, 1377, 29.5, -23.5, 0.0625, max_hourly_step=78)
ICON_D2 = Model("icon-d2", 746, 1215, 43.18, -3.94, 0.02, max_hourly_step=48)

# ICON-EU's grid under its old names.
GRID_ROWS = ICON_EU.rows
GRID_COLS = ICON_EU.cols
GRID_LAT0 = ICON_EU.lat0
GRID_LON0 = ICON_EU.lon0
GRID_STEP = ICON_EU.step


def t2m_url(run_id: int, step: int, base: str = DWD_BASE) -> str:
    """ICON-EU 2 m temperature of run [run_id] (epoch seconds) at +[step] hours."""
    return ICON_EU.field_url("t_2m", run_id, step, base)


def hsurf_url(run_id: int, base: str = DWD_BASE) -> str:
    return ICON_EU.hsurf_url(run_id, base)


def latest_run_ids(now: float, count: int) -> list[int]:
    """The [count] newest runs that have started by [now], newest first (they may not be ready)."""
    newest = int(now) // RUN_STEP_SECONDS * RUN_STEP_SECONDS
    return [newest - k * RUN_STEP_SECONDS for k in range(count)]


def decode(compressed: bytes, model: Model = ICON_EU) -> Field:
    """The values of a bzip2-compressed single-field GRIB2 file, rows south to north.

    The grid is checked against [model]'s layout so a changed product is rejected instead of
    drawn in the wrong place. Missing points (outside ICON-D2's area) become NaN.
    """
    try:
        message = bz2.decompress(compressed)
    except (OSError, ValueError) as e:
        raise IconError(f"not a bzip2 file: {e}") from e
    try:
        handle = eccodes.codes_new_from_message(message)
    except Exception as e:  # eccodes raises its own error classes
        raise IconError(f"not a GRIB message: {e}") from e
    try:
        expected: dict[str, object] = {
            "gridType": "regular_ll",
            "Ni": model.cols,
            "Nj": model.rows,
            "jScansPositively": 1,
        }
        for key, want in expected.items():
            got = eccodes.codes_get(handle, key)
            if got != want:
                raise IconError(f"unexpected grid: {key} is {got!r}, expected {want!r}")
        lat0 = eccodes.codes_get(handle, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(handle, "longitudeOfFirstGridPointInDegrees")
        step = eccodes.codes_get(handle, "iDirectionIncrementInDegrees")
        lon0 = lon0 - 360 if lon0 > 180 else lon0
        if (
            abs(lat0 - model.lat0) > 1e-6
            or abs(lon0 - model.lon0) > 1e-6
            or abs(step - model.step) > 1e-6
        ):
            raise IconError(f"unexpected grid origin or spacing: {lat0}, {lon0}, {step}")
        values = np.asarray(eccodes.codes_get_values(handle), dtype=np.float64)
        if eccodes.codes_get(handle, "bitmapPresent"):
            values[values == eccodes.codes_get(handle, "missingValue")] = np.nan
    finally:
        eccodes.codes_release(handle)
    return values.astype(np.float32).reshape(model.rows, model.cols)


class IconClient:
    def __init__(self, session: aiohttp.ClientSession, base_url: str = DWD_BASE) -> None:
        self._session = session
        self._base = base_url.rstrip("/")

    def field_url(self, model: Model, field: str, run_id: int, step: int) -> str:
        return model.field_url(field, run_id, step, self._base)

    def hsurf_url(self, model: Model, run_id: int) -> str:
        return model.hsurf_url(run_id, self._base)

    async def exists(self, url: str) -> bool:
        try:
            async with self._session.head(url) as response:
                if response.status == 404:
                    return False
                if response.status != 200:
                    raise IconError(f"HEAD {url} answered HTTP {response.status}")
                return True
        except (aiohttp.ClientError, TimeoutError) as e:
            raise IconError(f"HEAD {url} failed: {e}") from e

    async def download(self, url: str) -> bytes | None:
        """The file's bytes, or None if it is not there (yet)."""
        try:
            async with self._session.get(url) as response:
                if response.status == 404:
                    return None
                if response.status != 200:
                    raise IconError(f"GET {url} answered HTTP {response.status}")
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.content.iter_chunked(64 * 1024):
                    total += len(chunk)
                    if total > MAX_FILE_BYTES:
                        raise IconError(f"{url} is larger than {MAX_FILE_BYTES} bytes")
                    chunks.append(chunk)
        except (aiohttp.ClientError, TimeoutError) as e:
            raise IconError(f"GET {url} failed: {e}") from e
        return b"".join(chunks)
