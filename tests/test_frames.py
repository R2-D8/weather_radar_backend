from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from weather_radar import geo
from weather_radar.frames import (
    FrameFormatError,
    FrameStore,
    Grid,
    RadarFrame,
    _halve,
    decode_geotiff,
)


def test_decodes_the_real_sample(sample_grid: Grid) -> None:
    assert sample_grid.shape == (geo.GRID_ROWS, geo.GRID_COLS)
    assert sample_grid.dtype == np.float32
    missing = float(np.isnan(sample_grid).mean())
    assert 0.3 < missing < 0.6  # no radar coverage outside the national mosaic
    valid = sample_grid[~np.isnan(sample_grid)]
    assert valid.min() >= 0
    assert valid.max() > 10  # there is a real storm in the sample


def _tiff(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="TIFF")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "data",
    [
        b"not an image at all",
        b"",
        _tiff(Image.new("F", (10, 10))),  # wrong size
        _tiff(Image.new("RGB", (geo.GRID_COLS, geo.GRID_ROWS))),  # wrong mode
        _tiff(Image.new("F", (geo.GRID_COLS, geo.GRID_ROWS))),  # right shape, no georeferencing
    ],
)
def test_rejects_things_that_are_not_a_dpc_frame(data: bytes) -> None:
    with pytest.raises(FrameFormatError):
        decode_geotiff(data)


def test_rejects_a_png() -> None:
    buffer = io.BytesIO()
    Image.new("L", (4, 4)).save(buffer, format="PNG")
    with pytest.raises(FrameFormatError):
        decode_geotiff(buffer.getvalue())


def test_halve_averages_valid_cells_only() -> None:
    nan = np.nan
    grid = np.array([[1, 3, nan, nan], [nan, 5, nan, nan]], dtype=np.float32)
    halved = _halve(grid)
    assert halved.shape == (1, 2)
    assert halved[0, 0] == pytest.approx(3.0)  # (1 + 3 + 5) / 3
    assert np.isnan(halved[0, 1])  # a block with no data stays missing


def test_levels_are_built_lazily_and_halve_each_time(sample_grid: Grid) -> None:
    frame = RadarFrame(1, sample_grid)
    assert frame.level(0).shape == (1400, 1200)
    assert frame.level(2).shape == (350, 300)
    assert frame.level(3).shape == (175, 150)


def test_store_round_trip_and_ordering(tmp_path: Path, sample_bytes: bytes) -> None:
    store = FrameStore(tmp_path)
    assert store.ids() == []
    for frame_id in (300, 100, 200):
        store.save(frame_id, sample_bytes)
    assert store.ids() == [100, 200, 300]
    assert store.has(200) and not store.has(250)
    assert store.load(200).id == 200
    assert not list(tmp_path.glob(".*.tmp")), "temporary files must not be left behind"


def test_store_ignores_unrelated_files(tmp_path: Path, sample_bytes: bytes) -> None:
    store = FrameStore(tmp_path)
    store.save(100, sample_bytes)
    (tmp_path / "notes.tif").write_bytes(b"x")
    (tmp_path / "readme.txt").write_text("hi")
    assert store.ids() == [100]


def test_load_of_a_missing_frame_is_a_key_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        FrameStore(tmp_path).load(1)


def test_corrupt_frame_is_deleted_so_it_is_fetched_again(tmp_path: Path) -> None:
    store = FrameStore(tmp_path)
    store.save(100, b"garbage")
    with pytest.raises(KeyError):
        store.load(100)
    assert not store.has(100)


def test_decoded_frames_are_cached_with_a_limit(tmp_path: Path, sample_bytes: bytes) -> None:
    store = FrameStore(tmp_path, max_decoded=2)
    for frame_id in (1, 2, 3):
        store.save(frame_id, sample_bytes)
    first = store.load(1)
    assert store.load(1) is first
    store.load(2)
    store.load(3)  # evicts the least recently used one (frame 1)
    assert store.load(1) is not first


def test_prune_removes_only_older_frames(tmp_path: Path, sample_bytes: bytes) -> None:
    store = FrameStore(tmp_path)
    for frame_id in (100, 200, 300):
        store.save(frame_id, sample_bytes)
    store.load(100)
    assert store.prune(200) == [100]
    assert store.ids() == [200, 300]
    with pytest.raises(KeyError):
        store.load(100)
