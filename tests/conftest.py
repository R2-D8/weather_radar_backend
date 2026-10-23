from __future__ import annotations

import math
from pathlib import Path

import pytest
from pyproj import Transformer

from weather_radar import geo
from weather_radar.frames import Grid, RadarFrame, decode_geotiff

SAMPLE = Path(__file__).parent / "data" / "sri_sample.tif"

_radar_to_lonlat = Transformer.from_crs(geo.RADAR_PROJ, "EPSG:4326", always_xy=True)


@pytest.fixture(scope="session")
def sample_bytes() -> bytes:
    """A real DPC SRI frame (23 Oct 2026, 21:45 UTC) with rain over the north-east."""
    return SAMPLE.read_bytes()


@pytest.fixture(scope="session")
def sample_grid(sample_bytes: bytes) -> Grid:
    return decode_geotiff(sample_bytes)


@pytest.fixture
def sample_frame(sample_grid: Grid) -> RadarFrame:
    return RadarFrame(1, sample_grid)


def radar_point_to_tile(x: float, y: float, z: int) -> tuple[int, int, int, int]:
    """Tile (x, y) and pixel (col, row) of a 256 px tile that contain a radar-projection point."""
    lon, lat = _radar_to_lonlat.transform(x, y)
    n = 2**z
    fx = (lon + 180) / 360 * n
    fy = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return int(fx), int(fy), int((fx % 1) * 256), int((fy % 1) * 256)
