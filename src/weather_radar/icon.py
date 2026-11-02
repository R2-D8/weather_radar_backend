"""Client for DWD's ICON-EU model output on its open data server.

ICON-EU is DWD's European weather model: a regular latitude/longitude grid of 0.0625 degrees (about
7 km), 657 rows from 29.5 N to 70.5 N and 1377 columns from 23.5 W to 62.5 E, so it covers Italy
and Finland. A new run starts every 3 hours; the 00, 06, 12 and 18 UTC runs go to +120 h (hourly
to +78 h) and appear on the server about 2.5 hours after their start time. The server keeps only
the newest run of each start hour, so anything older than about a day has to be kept by us.

Each field of each forecast hour is one bzip2-compressed GRIB2 file of about 1 MB, for example
`.../grib/06/t_2m/icon-eu_europe_regular-lat-lon_single-level_2026110206_012_T_2M.grib2.bz2`
(2 m temperature of the 06 UTC run at +12 h). Licence: CC BY 4.0, source Deutscher Wetterdienst.
"""

from __future__ import annotations

import bz2
from datetime import UTC, datetime

import aiohttp
import eccodes
import numpy as np
from numpy.typing import NDArray

BASE_URL = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"

#: Runs that go far enough ahead for us (the 03, 09, 15 and 21 UTC runs stop at +30 h).
RUN_HOURS = (0, 6, 12, 18)
RUN_STEP_SECONDS = 6 * 3600

GRID_ROWS = 657
GRID_COLS = 1377
#: Centre of row 0 / column 0 (the south-west corner) and the spacing, in degrees.
GRID_LAT0 = 29.5
GRID_LON0 = -23.5
GRID_STEP = 0.0625

#: A file is about 1 MB; anything much bigger is not what we asked for.
MAX_FILE_BYTES = 20 * 1024 * 1024

Field = NDArray[np.float32]


class IconError(Exception):
    """DWD could not be reached, or a file is not the grid we expect."""


def run_time(run_id: int) -> datetime:
    return datetime.fromtimestamp(run_id, tz=UTC)


def _stamp(run_id: int) -> str:
    return run_time(run_id).strftime("%Y%m%d%H")


def t2m_url(run_id: int, step: int, base: str = BASE_URL) -> str:
    """Address of the 2 m temperature of run [run_id] (epoch seconds) at +[step] hours."""
    hh = run_time(run_id).strftime("%H")
    return (
        f"{base}/{hh}/t_2m/icon-eu_europe_regular-lat-lon_single-level_"
        f"{_stamp(run_id)}_{step:03d}_T_2M.grib2.bz2"
    )


def hsurf_url(run_id: int, base: str = BASE_URL) -> str:
    """Address of the model's ground height (the same for every run, published with each)."""
    hh = run_time(run_id).strftime("%H")
    return (
        f"{base}/{hh}/hsurf/icon-eu_europe_regular-lat-lon_time-invariant_"
        f"{_stamp(run_id)}_HSURF.grib2.bz2"
    )


def latest_run_ids(now: float, count: int) -> list[int]:
    """The [count] newest runs that have started by [now], newest first (they may not be ready)."""
    newest = int(now) // RUN_STEP_SECONDS * RUN_STEP_SECONDS
    return [newest - k * RUN_STEP_SECONDS for k in range(count)]


def decode(compressed: bytes) -> Field:
    """The values of a bzip2-compressed single-field GRIB2 file, rows south to north.

    The grid is checked against the expected ICON-EU layout so a changed product is rejected
    instead of drawn in the wrong place.
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
            "Ni": GRID_COLS,
            "Nj": GRID_ROWS,
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
            abs(lat0 - GRID_LAT0) > 1e-6
            or abs(lon0 - GRID_LON0) > 1e-6
            or abs(step - GRID_STEP) > 1e-6
        ):
            raise IconError(f"unexpected grid origin or spacing: {lat0}, {lon0}, {step}")
        values = eccodes.codes_get_values(handle)
    finally:
        eccodes.codes_release(handle)
    return np.asarray(values, dtype=np.float32).reshape(GRID_ROWS, GRID_COLS)


class IconClient:
    def __init__(self, session: aiohttp.ClientSession, base_url: str = BASE_URL) -> None:
        self._session = session
        self._base = base_url.rstrip("/")

    def t2m_url(self, run_id: int, step: int) -> str:
        return t2m_url(run_id, step, self._base)

    def hsurf_url(self, run_id: int) -> str:
        return hsurf_url(run_id, self._base)

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
