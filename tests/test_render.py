from __future__ import annotations

import io
import time

import numpy as np
import pytest
from conftest import radar_point_to_tile
from PIL import Image

from weather_radar import geo
from weather_radar.frames import RadarFrame
from weather_radar.palettes import BLUES, CLASSIC, Palette
from weather_radar.render import _bilinear, empty_tile, render_tile, sample_tile


def uniform_frame(value: float) -> RadarFrame:
    return RadarFrame(1, np.full((geo.GRID_ROWS, geo.GRID_COLS), value, dtype=np.float32))


def decode(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def test_rain_is_painted_with_the_palette_colour() -> None:
    tx, ty, px, py = radar_point_to_tile(0.0, 0.0, 8)
    pixels = decode(render_tile(uniform_frame(3.0), CLASSIC, 8, tx, ty))
    # 3 mm/h is in the band that starts at 2 mm/h
    assert tuple(pixels[py, px]) == CLASSIC.colours[CLASSIC.thresholds.index(2.0)]


@pytest.mark.parametrize(
    ("rate", "band_start"), [(0.1, 0.1), (0.49, 0.1), (0.5, 0.5), (99.0, 50.0), (500.0, 100.0)]
)
def test_band_edges(rate: float, band_start: float) -> None:
    tx, ty, px, py = radar_point_to_tile(0.0, 0.0, 8)
    pixels = decode(render_tile(uniform_frame(rate), CLASSIC, 8, tx, ty))
    assert tuple(pixels[py, px]) == CLASSIC.colours[CLASSIC.thresholds.index(band_start)]


def test_no_rain_gives_the_shared_empty_tile() -> None:
    tx, ty, _, _ = radar_point_to_tile(0.0, 0.0, 8)
    assert render_tile(uniform_frame(0.0), CLASSIC, 8, tx, ty) == empty_tile(256)
    assert render_tile(uniform_frame(0.05), CLASSIC, 8, tx, ty) == empty_tile(256)  # below 0.1


def test_missing_data_is_transparent() -> None:
    tx, ty, _, _ = radar_point_to_tile(0.0, 0.0, 8)
    assert render_tile(uniform_frame(np.nan), CLASSIC, 8, tx, ty) == empty_tile(256)


def test_tiles_outside_the_coverage_are_empty_without_any_work() -> None:
    class Untouchable(RadarFrame):
        def level(self, level: int):  # type: ignore[no-untyped-def]
            raise AssertionError("must not sample a tile outside the coverage")

    frame = Untouchable(1, np.zeros((geo.GRID_ROWS, geo.GRID_COLS), dtype=np.float32))
    assert render_tile(frame, CLASSIC, 8, 150, 50) == empty_tile(256)  # Finland


def test_empty_tile_is_tiny_and_fully_transparent() -> None:
    png = empty_tile(256)
    assert len(png) < 300
    pixels = decode(png)
    assert pixels.shape == (256, 256, 4)
    assert pixels[..., 3].max() == 0
    assert empty_tile(256) is empty_tile(256)


def test_north_is_up_in_the_rendered_picture() -> None:
    """Rain only in the north-west corner of the grid must not appear in the south-east."""
    grid = np.zeros((geo.GRID_ROWS, geo.GRID_COLS), dtype=np.float32)
    grid[:100, :100] = 5.0  # top-left 100 km x 100 km
    frame = RadarFrame(1, grid)
    nw = radar_point_to_tile(-550_000.0, 600_000.0, 8)
    se = radar_point_to_tile(550_000.0, -700_000.0, 8)
    assert decode(render_tile(frame, CLASSIC, 8, nw[0], nw[1]))[nw[3], nw[2], 3] > 0
    assert decode(render_tile(frame, CLASSIC, 8, se[0], se[1]))[se[3], se[2], 3] == 0


def test_tile_is_a_small_palette_png() -> None:
    tx, ty, _, _ = radar_point_to_tile(0.0, 0.0, 8)
    png = render_tile(uniform_frame(8.0), CLASSIC, 8, tx, ty)
    image = Image.open(io.BytesIO(png))
    assert image.size == (256, 256)
    assert image.mode == "P"
    assert "transparency" in image.info
    assert len(png) < 1000  # one flat colour compresses to almost nothing


def test_512_pixel_tiles() -> None:
    tx, ty, _, _ = radar_point_to_tile(0.0, 0.0, 7)
    image = Image.open(io.BytesIO(render_tile(uniform_frame(3.0), CLASSIC, 7, tx, ty, size=512)))
    assert image.size == (512, 512)
    assert empty_tile(512) != empty_tile(256)


def test_rejects_unsupported_tile_size() -> None:
    with pytest.raises(ValueError, match="tile size"):
        render_tile(uniform_frame(1.0), CLASSIC, 8, 137, 91, size=100)


def test_other_palette_gives_other_colours() -> None:
    tx, ty, px, py = radar_point_to_tile(0.0, 0.0, 8)
    classic = decode(render_tile(uniform_frame(3.0), CLASSIC, 8, tx, ty))[py, px]
    blues = decode(render_tile(uniform_frame(3.0), BLUES, 8, tx, ty))[py, px]
    assert tuple(classic) != tuple(blues)


def test_bilinear_sampling_and_edges() -> None:
    grid = np.array([[0, 10], [20, 30]], dtype=np.float32)
    col = np.array([[0.0, 0.5, 1.0, -0.4, -0.6]])
    row = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
    out = _bilinear(grid, col, row)
    assert out[0, 0] == pytest.approx(0)  # exactly on a cell centre
    assert out[0, 1] == pytest.approx(5)  # halfway between 0 and 10
    assert out[0, 2] == pytest.approx(30)
    assert out[0, 3] == pytest.approx(0)  # still inside the first cell, which spans -0.5 to 0.5
    assert np.isnan(out[0, 4])  # outside the grid


def test_bilinear_does_not_dim_rain_next_to_missing_data() -> None:
    nan = np.nan
    grid = np.array([[8.0, nan], [8.0, nan]], dtype=np.float32)
    out = _bilinear(grid, np.array([[0.3]]), np.array([[0.5]]))
    assert out[0, 0] == pytest.approx(8.0)  # missing neighbours are left out, not counted as 0


def test_real_frame_renders_quickly_and_compactly(sample_frame: RadarFrame) -> None:
    # The strongest storm of the sample, near Trieste, at zoom 8.
    tx, ty = 137, 91
    sample_tile(sample_frame, 8, tx, ty, 256)  # warm up (builds nothing for level 0, loads pyproj)
    start = time.perf_counter()
    png = render_tile(sample_frame, CLASSIC, 8, tx, ty)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert png != empty_tile(256), "the sample has rain here"
    assert len(png) < 20_000
    assert elapsed_ms < 100, f"rendering took {elapsed_ms:.0f} ms"


def test_palette_validation() -> None:
    with pytest.raises(ValueError, match="one colour"):
        Palette("x", "X", (1.0, 2.0), ((0, 0, 0, 255),))
    with pytest.raises(ValueError, match="ascending"):
        Palette("x", "X", (2.0, 1.0), ((0, 0, 0, 255), (1, 1, 1, 255)))
    with pytest.raises(ValueError, match="15"):
        Palette("x", "X", tuple(float(i + 1) for i in range(16)), ((0, 0, 0, 255),) * 16)
