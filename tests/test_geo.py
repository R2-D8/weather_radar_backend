from __future__ import annotations

import numpy as np
from conftest import radar_point_to_tile
from pyproj import Transformer

from weather_radar import geo


def test_projection_origin_is_the_stated_centre() -> None:
    """lat_0/lon_0 of the DPC projection (42 N, 12.5 E) must be the radar coordinate (0, 0)."""
    to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    mx, my = to_mercator.transform(12.5, 42.0)
    rx, ry = Transformer.from_crs("EPSG:3857", geo.RADAR_PROJ, always_xy=True).transform(mx, my)
    assert abs(rx) < 1.0 and abs(ry) < 1.0


def test_origin_lands_on_the_right_grid_cell() -> None:
    """Radar (0, 0) is 600 km right of and 650 km below the top-left corner: cell (599.5, 649.5)."""
    z = 12
    tx, ty, px, py = radar_point_to_tile(0.0, 0.0, z)
    col, row = geo.tile_grid_coords(z, tx, ty, 256, level=0)
    assert abs(col[py, px] - 599.5) < 0.1
    assert abs(row[py, px] - 649.5) < 0.1


def test_north_is_up_and_east_is_right() -> None:
    """Moving north in a tile decreases the grid row; moving east increases the column."""
    col, row = geo.tile_grid_coords(8, 137, 91, 256, level=0)
    assert row[0, 128] < row[255, 128]  # top of the tile is further north
    assert col[128, 0] < col[128, 255]  # left of the tile is further west


def test_coarser_levels_halve_the_coordinates() -> None:
    col0, row0 = geo.tile_grid_coords(6, 34, 22, 256, level=0)
    col1, row1 = geo.tile_grid_coords(6, 34, 22, 256, level=1)
    assert np.allclose((col0 + 0.5) / 2 - 0.5, col1)
    assert np.allclose((row0 + 0.5) / 2 - 0.5, row1)


def test_detail_level_follows_zoom() -> None:
    assert geo.detail_level(8, 256) == 0
    assert geo.detail_level(7, 256) == 0
    assert geo.detail_level(5, 256) == 1
    assert geo.detail_level(4, 256) == 2
    assert geo.detail_level(0, 256) == geo.MAX_LEVEL
    # 512 px tiles are twice as detailed at the same zoom
    assert geo.detail_level(6, 512) == 0


def test_tile_bounds_cover_the_world_at_zoom_0() -> None:
    assert geo.tile_bounds(0, 0, 0) == (
        -geo.HALF_WORLD,
        -geo.HALF_WORLD,
        geo.HALF_WORLD,
        geo.HALF_WORLD,
    )


def test_coverage_check() -> None:
    assert geo.tile_overlaps_coverage(0, 0, 0)
    assert geo.tile_overlaps_coverage(8, 137, 91)  # north-east Italy
    assert not geo.tile_overlaps_coverage(8, 150, 50)  # Finland
    assert not geo.tile_overlaps_coverage(8, 60, 100)  # Atlantic


def test_lonlat_to_cell() -> None:
    # The projection centre is 600 km from the left edge and 650 km below the top edge.
    assert geo.lonlat_to_cell(12.5, 42.0) == (650, 600)
    assert geo.lonlat_to_cell(24.94, 60.17) is None  # Helsinki, far outside the grid
    assert geo.lonlat_to_cell(-20.0, 20.0) is None
    row, col = geo.lonlat_to_cell(12.48, 41.89)  # Rome, a little south-west of the centre
    assert 650 < row < 670 and 590 < col < 600
