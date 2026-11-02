"""Rain or snow per pixel, and rain fields from the models (frames after the radar nowcast).

**Rain or snow.** DPC's radar says how much falls, not in which form. ICON-EU's snow line
(`snowlmt`: the height above sea level where the falling precipitation turns from snow to rain;
DWD defines it by the wet-bulb temperature, so it accounts for dry air) is compared with the real
ground height of each pixel (the same terrain tiles as the temperature map):

- ground more than [SNOW_MARGIN_M] above the snow line: snow;
- within [SNOW_MARGIN_M] of it: mixed (sleet, wet snow);
- further below: rain.

Where the whole air column down to the model's ground is cold, DWD leaves the snow line out; it
is then taken to be [2 * SNOW_MARGIN_M] below the model ground, so the model's own ground gets
snow and valleys well below it get mixed or rain. The snow line is a smooth field, so its 7 km
grid is enough: the detail comes from the terrain.

**Model rain.** ICON's precipitation in the hour before each time, ICON-D2 where it has data
(faded into ICON-EU at the edge of its area) and ICON-EU elsewhere. An hourly amount in mm is the
average rate in mm/h over that hour, so it is drawn with the radar palettes; it is smoother than
a radar picture, which shows one instant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from weather_radar.icon import ICON_D2
from weather_radar.modeldata import Grid, blend, sample
from weather_radar.palettes import MIXED_PHASE, RAIN, SNOW_PHASE

#: Half the depth of the rain/snow transition, in metres.
SNOW_MARGIN_M = 150.0


def fill_snowline(values: Grid, model_ground: NDArray[np.float32] | None) -> Grid:
    """Snow line with the gaps (all snow down to the model ground) filled in, see the module doc."""
    if model_ground is None:
        raise ValueError("the snow line needs the model ground height")
    filled = np.where(np.isnan(values), model_ground - 2 * SNOW_MARGIN_M, values)
    return filled.astype(np.float32)


def phase(snowline: Grid, ground: NDArray[np.float32]) -> NDArray[np.uint8]:
    """0 rain, 1 mixed, 2 snow per pixel; sea counts as 0 m, a missing snow line as rain."""
    above = np.maximum(ground, 0.0) - snowline
    result = np.full(snowline.shape, RAIN, dtype=np.uint8)
    result[above >= -SNOW_MARGIN_M] = MIXED_PHASE
    result[above > SNOW_MARGIN_M] = SNOW_PHASE
    result[np.isnan(snowline)] = RAIN
    return result


def phase_at(
    snowline: NDArray[np.floating],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    ground: NDArray[np.float32],
) -> NDArray[np.uint8]:
    """Phase at points, from an ICON-EU snow-line grid and the real ground there."""
    return phase(sample(snowline, lat, lon), ground)


@dataclass(frozen=True)
class RainHour:
    """One hour's precipitation (mm in the hour): ICON-EU, and ICON-D2 if it has the hour."""

    eu: NDArray[np.floating]
    d2: NDArray[np.floating] | None = None
    d2_weight: NDArray[np.floating] | None = None


def rain_at(hour: RainHour, lat: NDArray[np.float64], lon: NDArray[np.float64]) -> Grid:
    """Precipitation in mm/h at points; NaN outside ICON-EU's area."""
    rain = sample(hour.eu, lat, lon)
    if hour.d2 is None or hour.d2_weight is None:
        return rain
    d2 = sample(hour.d2, lat, lon, ICON_D2)
    return blend(rain, d2, sample(hour.d2_weight, lat, lon, ICON_D2))
