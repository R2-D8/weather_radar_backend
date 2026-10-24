from __future__ import annotations

import cv2
import numpy as np
import pytest

from weather_radar import geo, nowcast
from weather_radar.frames import Grid

ROWS, COLS = geo.GRID_ROWS, geo.GRID_COLS


def blob(row: float, col: float, peak: float = 20.0, radius: float = 15.0) -> Grid:
    """A round shower of [peak] mm/h centred on (row, col)."""
    r, c = np.mgrid[0:ROWS, 0:COLS]
    grid = peak * np.exp(-(((r - row) ** 2 + (c - col) ** 2) / (2 * radius**2)))
    grid[grid < 0.05] = 0.0
    return grid.astype(np.float32)


def centre(grid: Grid) -> tuple[float, float]:
    weights = np.nan_to_num(grid, nan=0.0)
    r, c = np.mgrid[0:ROWS, 0:COLS]
    return float((r * weights).sum() / weights.sum()), float((c * weights).sum() / weights.sum())


def textured_shower(row: int, col: int) -> Grid:
    """A shower with internal structure (real rain has texture the motion tracker can follow),
    surrounded by truly dry cells."""
    rng = np.random.default_rng(1)
    noise = cv2.GaussianBlur(rng.random((ROWS, COLS)).astype(np.float32), (0, 0), 3)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    r, c = np.mgrid[0:ROWS, 0:COLS]
    grid = 20 * np.exp(-(((r - row) ** 2 + (c - col) ** 2) / (2 * 25.0**2))) * noise
    grid[grid < 0.2] = 0.0
    return grid.astype(np.float32)


def test_a_moving_shower_keeps_moving_the_same_way() -> None:
    # The same shower moved 6 cells down and 8 cells right in 10 minutes (about 60 km/h).
    earlier = textured_shower(600, 500)
    newest = np.roll(np.roll(earlier, 6, axis=0), 8, axis=1)
    start_row, start_col = centre(newest)

    forecast = nowcast.compute(earlier, newest, newest_id=1_000_000)

    assert sorted(forecast.frames) == list(nowcast.LEAD_MINUTES)
    for lead, frame in forecast.frames.items():
        row, col = centre(frame.level(0))
        steps = lead / nowcast.MOTION_MINUTES
        # The direction must be right. Optical flow under-measures motion in the faint parts of a
        # shower (about 70 % of the true speed there, 90 % in the cores), so the shower as a whole
        # covers somewhat more than half the distance; a known limitation noted in the plan.
        assert 0.5 * 6 * steps <= row - start_row <= 1.2 * 6 * steps, (lead, row - start_row)
        assert 0.5 * 8 * steps <= col - start_col <= 1.2 * 8 * steps, (lead, col - start_col)
        assert frame.id == 1_000_000 + lead * 60


def test_a_third_frame_averages_the_motion_of_both_steps() -> None:
    # Still for the first 10 minutes, then 6 down and 8 right: the averaged motion is about half.
    oldest = textured_shower(600, 500)
    earlier = oldest.copy()
    newest = np.roll(np.roll(earlier, 6, axis=0), 8, axis=1)
    start_row, start_col = centre(newest)

    one_step = nowcast.compute(earlier, newest, newest_id=1).frames[30].level(0)
    two_steps = nowcast.compute(earlier, newest, newest_id=1, oldest=oldest).frames[30].level(0)

    moved_one = np.subtract(centre(one_step), (start_row, start_col))
    moved_two = np.subtract(centre(two_steps), (start_row, start_col))
    assert np.all(moved_two > 0), moved_two
    assert np.allclose(moved_two, moved_one / 2, rtol=0.25), (moved_one, moved_two)


def test_a_still_shower_stays_put() -> None:
    newest = blob(700, 600)
    forecast = nowcast.compute(newest.copy(), newest, newest_id=1)
    row, col = centre(forecast.frames[30].level(0))
    assert row == pytest.approx(700, abs=1.0)
    assert col == pytest.approx(600, abs=1.0)


def test_cells_without_radar_stay_missing() -> None:
    newest = blob(600, 500)
    newest[:, :100] = np.nan  # no radar coverage on the left
    forecast = nowcast.compute(newest.copy(), newest, newest_id=1)
    assert np.isnan(forecast.frames[10].level(0)[:, :100]).all()
    assert not np.isnan(forecast.frames[10].level(0)[:, 200:]).any()


def test_motion_is_capped() -> None:
    # An absurd jump (a different storm, not motion) must not fling rain across the country.
    motion = nowcast.measure_motion(blob(200, 200), blob(1200, 1000))
    assert np.hypot(motion[..., 0], motion[..., 1]).max() <= nowcast.MAX_SHIFT_CELLS + 1e-3


def test_dry_sky_gives_dry_forecast() -> None:
    dry = np.zeros((ROWS, COLS), dtype=np.float32)
    forecast = nowcast.compute(dry, dry, newest_id=1)
    assert all(float(np.nanmax(f.level(0))) == 0.0 for f in forecast.frames.values())


def test_real_frame_forecast_is_fast_enough(sample_grid: Grid) -> None:
    import time

    start = time.perf_counter()
    nowcast.compute(sample_grid, sample_grid, newest_id=1)
    assert time.perf_counter() - start < 10.0
