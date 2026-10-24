"""A short rain forecast ("nowcast") made by moving the newest frame along its recent motion.

The motion is measured between the newest frame and the one 10 minutes earlier (dense optical
flow), smoothed and capped, and averaged with the motion of the 10 minutes before that when that
frame is there; the newest rain field is then shifted along it. This cannot
predict storms that have not formed yet and loses skill quickly: a backtest on DPC data (see the
plan) showed a clear gain over "the rain stays where it is" up to about 30 minutes, so that is
as far as it goes.

Averaging two motion estimates was the only change that helped in a second backtest (8 rain events,
114 start times, October 2026): +0.005 to +0.008 on every score and lead time, better in about
70 % of cases. Open-Meteo model data (700/850 hPa winds as motion, model rain trend as growth and
decay, blending in model rain) did not help and was left out; see the app's docs/plan.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from weather_radar.frames import Grid, RadarFrame

#: Minutes ahead that are forecast.
LEAD_MINUTES = (10, 20, 30)
#: The motion is measured over this many minutes (between the newest frame and this much earlier).
MOTION_MINUTES = 10
#: Upper limit for the measured motion, in grid cells (1 km) per MOTION_MINUTES: 210 km/h.
MAX_SHIFT_CELLS = 35.0
#: The flow field is smoothed with a Gaussian of this many cells, so noisy patches do not tear.
FLOW_SMOOTHING_CELLS = 12.0


@dataclass(frozen=True)
class Nowcast:
    """Forecast frames for one base (the newest observed frame)."""

    base_id: int
    #: Lead time in minutes to the forecast frame for that time.
    frames: dict[int, RadarFrame]


def _as_image(rain: Grid) -> np.ndarray:
    """Rain rate as an 8-bit picture for optical flow; log scale gives light rain texture too."""
    return np.asarray(
        np.clip(np.log1p(np.nan_to_num(rain, nan=0.0)) * 60.0, 0, 255), dtype=np.uint8
    )


def measure_motion(earlier: Grid, newest: Grid) -> np.ndarray:
    """Displacement per MOTION_MINUTES in cells, shape (rows, cols, 2) as (dx, dy)."""
    first = _as_image(earlier)
    second = _as_image(newest)
    # With flags=0 the initial flow is ignored; OpenCV's type stubs just do not accept None.
    initial = np.zeros((*first.shape, 2), dtype=np.float32)
    flow = cv2.calcOpticalFlowFarneback(first, second, initial, 0.5, 4, 31, 3, 7, 1.5, 0)
    # Only rainy cells say anything about motion. Smooth the flow as a weighted average over them,
    # so a small shower is not slowed down by the zero motion of the dry area around it, and
    # nearby dry cells inherit the motion of the rain next to them.
    weight = np.asarray(np.maximum(first, second) > 0, dtype=np.float32)
    blurred_weight = cv2.GaussianBlur(weight, (0, 0), FLOW_SMOOTHING_CELLS)
    smoothed = np.empty_like(flow)
    for axis in range(2):
        blurred = cv2.GaussianBlur(flow[..., axis] * weight, (0, 0), FLOW_SMOOTHING_CELLS)
        smoothed[..., axis] = np.where(
            blurred_weight > 1e-3, blurred / np.maximum(blurred_weight, 1e-6), 0.0
        )
    speed = np.hypot(smoothed[..., 0], smoothed[..., 1])
    scale = np.minimum(1.0, MAX_SHIFT_CELLS / np.maximum(speed, 1e-6))
    return np.asarray(smoothed * scale[..., None], dtype=np.float32)


def advect(newest: Grid, motion: np.ndarray, minutes: int) -> Grid:
    """[newest] moved forward by [minutes] along [motion]; cells without radar stay missing."""
    rows, cols = newest.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32)
    )
    shift = motion * (minutes / MOTION_MINUTES)
    # Semi-Lagrangian: each cell takes the value from where its rain is coming from.
    moved = cv2.remap(
        np.nan_to_num(newest, nan=0.0).astype(np.float32),
        grid_x - shift[..., 0],
        grid_y - shift[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    result: Grid = np.asarray(moved, dtype=np.float32)
    result[np.isnan(newest)] = np.nan
    return result


def compute(earlier: Grid, newest: Grid, newest_id: int, oldest: Grid | None = None) -> Nowcast:
    """Forecast frames LEAD_MINUTES after [newest_id] (epoch seconds).

    [earlier] is MOTION_MINUTES before [newest]; [oldest], if given, MOTION_MINUTES before
    [earlier]. With it, the motion is the average of the two steps, which is steadier.
    """
    motion = measure_motion(earlier, newest)
    if oldest is not None:
        motion = (motion + measure_motion(oldest, earlier)) / 2
    frames = {
        lead: RadarFrame(newest_id + lead * 60, advect(newest, motion, lead))
        for lead in LEAD_MINUTES
    }
    return Nowcast(newest_id, frames)
