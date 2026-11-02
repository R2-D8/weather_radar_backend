"""ICON-D2 next to ICON-EU, hourly rain from the models, and rain or snow on the radar."""

from __future__ import annotations

import bz2
import io
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import eccodes
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image
from test_temperature import FakeIcon, field, terrarium_png, tile_of

from weather_radar import icon, modeldata, precip
from weather_radar.api import IMMUTABLE, SHORT, ModelFeature, create_app, version_of
from weather_radar.config import Settings
from weather_radar.frames import FrameStore
from weather_radar.icon import ICON_D2, ICON_EU, Model
from weather_radar.modeldata import (
    BLEND_CELLS,
    FieldJob,
    Hour,
    HourlyStore,
    ModelIngestor,
    blend,
    edge_weight,
)
from weather_radar.palettes import CLASSIC, MIXED, SNOW, phased, resolve
from weather_radar.render import empty_tile, encode
from weather_radar.terrain import Terrain

H = 3600
RUN = 1_793_577_600  # 2026-11-02 00:00 UTC


def grib_for(model: Model, values: np.ndarray) -> bytes:
    """A bzip2 GRIB2 message on [model]'s grid; NaN values are written as missing (bitmap)."""
    handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        eccodes.codes_set(handle, "Ni", model.cols)
        eccodes.codes_set(handle, "Nj", model.rows)
        eccodes.codes_set(handle, "jScansPositively", 1)
        eccodes.codes_set(handle, "latitudeOfFirstGridPointInDegrees", model.lat0)
        eccodes.codes_set(handle, "longitudeOfFirstGridPointInDegrees", model.lon0 % 360)
        eccodes.codes_set(handle, "latitudeOfLastGridPointInDegrees", model.lat1)
        eccodes.codes_set(handle, "longitudeOfLastGridPointInDegrees", model.lon1 % 360)
        eccodes.codes_set(handle, "iDirectionIncrementInDegrees", model.step)
        eccodes.codes_set(handle, "jDirectionIncrementInDegrees", model.step)
        if np.isnan(values).any():
            eccodes.codes_set(handle, "bitmapPresent", 1)
            eccodes.codes_set(handle, "missingValue", 9999)
            values = np.where(np.isnan(values), 9999.0, values)
        eccodes.codes_set_values(handle, values.astype(np.float64).ravel())
        return bz2.compress(eccodes.codes_get_message(handle))
    finally:
        eccodes.codes_release(handle)


def d2_field(value: float) -> np.ndarray:
    """An ICON-D2 grid with its rotated area imitated: the outer 40 cells are missing."""
    grid = np.full((ICON_D2.rows, ICON_D2.cols), value, dtype=np.float32)
    grid[:40, :] = grid[-40:, :] = grid[:, :40] = grid[:, -40:] = np.nan
    return grid


# --- models -----------------------------------------------------------------------------------


def test_d2_decode_turns_missing_points_into_nan() -> None:
    grid = icon.decode(grib_for(ICON_D2, d2_field(280.0)), ICON_D2)
    assert grid.shape == (746, 1215)
    assert np.isnan(grid[0, 0]) and grid[373, 600] == pytest.approx(280.0, abs=0.01)
    # An ICON-D2 file is not accepted as ICON-EU.
    with pytest.raises(icon.IconError):
        icon.decode(grib_for(ICON_D2, d2_field(280.0)), ICON_EU)


def test_urls_of_both_models_follow_dwd_naming() -> None:
    assert ICON_D2.field_url("tot_prec", RUN + 6 * H, 5) == (
        "https://opendata.dwd.de/weather/nwp/icon-d2/grib/06/tot_prec/"
        "icon-d2_germany_regular-lat-lon_single-level_2026110206_005_2d_tot_prec.grib2.bz2"
    )
    assert ICON_D2.hsurf_url(RUN).endswith(
        "/icon-d2/grib/00/hsurf/"
        "icon-d2_germany_regular-lat-lon_time-invariant_2026110200_000_0_hsurf.grib2.bz2"
    )
    assert ICON_EU.field_url("snowlmt", RUN, 1).endswith(
        "/icon-eu/grib/00/snowlmt/"
        "icon-eu_europe_regular-lat-lon_single-level_2026110200_001_SNOWLMT.grib2.bz2"
    )


def test_d2_fades_into_eu_at_the_edge_of_its_area() -> None:
    weight = edge_weight(d2_field(0.0))
    assert weight[0, 0] == 0.0 and weight[373, 600] == 1.0
    # Rises over BLEND_CELLS from the first cell with data.
    assert 0.0 < weight[373, 40] < weight[373, 45] < 1.0
    assert weight[373, 40 + int(BLEND_CELLS) + 1] == 1.0
    eu = np.array([10.0, 10.0, 10.0], np.float32)
    d2 = np.array([20.0, 20.0, np.nan], np.float32)
    w = np.array([1.0, 0.25, 1.0], np.float32)
    assert blend(eu, d2, w).tolist() == [20.0, 12.5, 10.0]
    assert blend(eu, None, None) is eu


# --- ingest -----------------------------------------------------------------------------------


async def _no_sleep(_: float) -> None:
    return None


async def test_rain_is_stored_per_hour_from_the_running_sum(tmp_path: Path) -> None:
    fake = FakeIcon({RUN})
    store = HourlyStore(tmp_path)
    job = FieldJob(
        ICON_EU, "tot_prec", store, timedelta(hours=1), timedelta(hours=3), accumulated=True
    )
    ingestor = ModelIngestor(fake, [job], clock=lambda: RUN + 2 * H, sleep=_no_sleep)  # type: ignore[arg-type]
    # Window RUN + 1 h .. RUN + 5 h; RUN + 1 h is the first step, which needs no step 0.
    assert await ingestor.cycle() == 5
    hours = store.hours()
    assert [h.valid for h in hours] == [RUN + k * H for k in range(1, 6)]
    for hour in hours:
        assert float(store.load(hour)[100, 100]) == pytest.approx(1.5)
    downloaded = [u for u in fake.downloads if "/tot_prec/" in u]
    # Steps 1 to 5 once each (step 4 and 3 are reused while walking back), never step 0.
    assert sorted(downloaded) == [f"icon-eu/tot_prec/{RUN}/{s}" for s in range(1, 6)]


# --- rain or snow -----------------------------------------------------------------------------


def test_phase_compares_the_ground_with_the_snow_line() -> None:
    snowline = np.full(6, 1000.0, np.float32)
    snowline[5] = np.nan
    ground = np.array([500.0, 849.0, 900.0, 1150.0, 1151.0, 3000.0], np.float32)
    assert precip.phase(snowline, ground).tolist() == [0, 0, 1, 1, 2, 0]
    # Missing snow line (cold down to the model ground): 300 m below the model ground.
    filled = precip.fill_snowline(
        np.array([np.nan, 1200.0], np.float32), np.array([800.0, 800.0], np.float32)
    )
    assert filled.tolist() == [500.0, 1200.0]


def test_phased_tiles_use_the_snow_and_mixed_scales() -> None:
    rain = np.full((256, 256), 2.5, np.float32)
    phase = np.zeros((256, 256), np.uint8)
    phase[:, 100:200] = 1
    phase[:, 200:] = 2
    png = encode(rain, CLASSIC, phase)
    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    assert tuple(pixels[0, 0]) == (0, 170, 100, 230)  # rain 2 to 4 mm/h
    assert tuple(pixels[0, 150]) == MIXED.colours[1]  # mixed 1 to 4
    assert tuple(pixels[0, 250]) == SNOW.colours[3]  # snow 1.5 to 3
    rgb, alpha, scales = phased(CLASSIC)
    assert len(alpha) == 1 + 12 + 4 + 6 and len(scales) == 3
    assert resolve("classic-snow") == (CLASSIC, True)
    assert resolve("classic") == (CLASSIC, False) and resolve("neon-snow") is None
    # Nothing falls: the shared empty tile, whatever the phase.
    assert encode(np.zeros((256, 256), np.float32), CLASSIC, phase) == empty_tile(256)


# --- API --------------------------------------------------------------------------------------

ALPS = (46.5, 11.3)  # inside both models
RAINY_FRAME_TILE = (137, 91)  # where the sample DPC frame has rain, zoom 8


def write_terrain(directory: Path, z: int, x: int, y: int, height: float) -> None:
    path = directory / str(z) / str(x) / f"{y}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(terrarium_png(height))


@pytest.fixture
async def models_api(tmp_path: Path, sample_bytes: bytes) -> AsyncIterator[TestClient]:
    frames = FrameStore(tmp_path / "frames")
    newest = RUN + 2 * H
    frames.save(newest, sample_bytes)
    t2m_eu = HourlyStore(tmp_path / "t2m_eu")
    t2m_eu.save_ground(field(1500.0))
    t2m_eu.save(Hour(RUN + 3 * H, RUN), field(0.0))
    t2m_d2 = HourlyStore(tmp_path / "t2m_d2")
    t2m_d2.save_ground(d2_field(1000.0))
    t2m_d2.save(Hour(RUN + 3 * H, RUN), d2_field(5.0))
    rain_eu = HourlyStore(tmp_path / "rain_eu")
    rain_d2 = HourlyStore(tmp_path / "rain_d2")
    for k in range(1, 16):
        rain_eu.save(Hour(RUN + k * H, RUN), field(0.5))
    rain_d2.save(Hour(RUN + 3 * H, RUN), d2_field(3.0))
    snowline = HourlyStore(tmp_path / "snowline")
    snowline.save(Hour(RUN + 2 * H, RUN), field(1000.0))
    snowline.save(Hour(RUN + 3 * H, RUN), field(1000.0))
    terrain_dir = tmp_path / "terrain"
    # High ground (snow) under the rainy radar tile and the Alps tile; the point lookup too.
    write_terrain(terrain_dir, 8, *RAINY_FRAME_TILE, 2000.0)
    write_terrain(terrain_dir, 8, *tile_of(*ALPS, 8), 2000.0)
    write_terrain(terrain_dir, 11, *tile_of(*ALPS, 11), 1000.0)
    feature = ModelFeature(
        terrain=Terrain(None, terrain_dir),
        t2m_eu=t2m_eu,
        t2m_d2=t2m_d2,
        rain_eu=rain_eu,
        rain_d2=rain_d2,
        snowline=snowline,
    )
    app = create_app(
        Settings(data_dir=tmp_path), frames, None, clock=lambda: newest + 600, models=feature
    )
    async with TestServer(app) as server, TestClient(server) as client:
        yield client


async def test_palettes_offer_snow_when_the_snow_line_is_there(models_api: TestClient) -> None:
    body = await (await models_api.get("/palettes")).json()
    assert body["snow_suffix"] == "-snow"
    assert len(body["snow"]) == 6 and len(body["mixed"]) == 4
    assert body["snow"][0] == {"from": 0.1, "color": "#f0f0ff", "alpha": 190}


async def test_radar_tiles_show_snow_on_high_ground(models_api: TestClient) -> None:
    x, y = RAINY_FRAME_TILE
    frame = RUN + 2 * H
    plain = await models_api.get(f"/tiles/{frame}/classic/8/{x}/{y}.png")
    snowy = await models_api.get(f"/tiles/{frame}/classic-snow/8/{x}/{y}.png")
    assert plain.status == snowy.status == 200
    rain_px = np.asarray(Image.open(io.BytesIO(await plain.read())).convert("RGBA"))
    snow_px = np.asarray(Image.open(io.BytesIO(await snowy.read())).convert("RGBA"))
    wet = rain_px[..., 3] > 0
    assert wet.any() and (snow_px[..., 3] > 0).sum() == wet.sum()
    # 2000 m ground, snow line 1000 m: every wet pixel is drawn in a snow colour.
    snow_rgb = {c[:3] for c in SNOW.colours}
    assert {tuple(p[:3]) for p in snow_px[wet]} <= snow_rgb


async def test_without_terrain_snow_tiles_fall_back_to_rain_briefly(
    models_api: TestClient, sample_bytes: bytes
) -> None:
    # The neighbouring tile has rain too in the sample frame but no terrain on disk.
    x, y = RAINY_FRAME_TILE
    response = await models_api.get(f"/tiles/{RUN + 2 * H}/classic-snow/8/{x}/{y + 1}.png")
    assert response.status == 200
    assert await response.read() != empty_tile(256)
    assert response.headers["Cache-Control"] == SHORT


async def test_model_lists_the_coming_hours(models_api: TestClient) -> None:
    body = await (await models_api.get("/model")).json()
    ids = [h["id"] for h in body["hours"]]
    # Now is RUN + 2 h 10 min: the hours after it, 12 of them; stored up to RUN + 15 h.
    assert ids == [RUN + k * H for k in range(3, 15)]
    assert body["hours"][0]["version"] == version_of(RUN, RUN)
    assert body["hours"][1]["version"] == str(RUN)
    assert "ICON-D2" in body["attribution"] and body["step_seconds"] == 3600


async def test_model_tiles_blend_d2_and_show_snow(models_api: TestClient) -> None:
    x, y = tile_of(*ALPS, 8)
    hour = RUN + 3 * H
    response = await models_api.get(f"/model/{RUN}-{RUN}/{hour}/classic/8/{x}/{y}.png")
    assert response.status == 200 and response.headers["Cache-Control"] == IMMUTABLE
    pixels = np.asarray(Image.open(io.BytesIO(await response.read())).convert("RGBA"))
    # ICON-D2 has 3 mm inside its area: classic band 2 to 4 mm/h.
    assert tuple(pixels[128, 128]) == (0, 170, 100, 230)
    snowy = await models_api.get(f"/model/{RUN}-{RUN}/{hour}/classic-snow/8/{x}/{y}.png")
    pixels = np.asarray(Image.open(io.BytesIO(await snowy.read())).convert("RGBA"))
    assert tuple(pixels[128, 128]) == SNOW.colours[4]  # 3 mm as snow: band 3 to 6
    # ICON-EU only for an hour ICON-D2 does not have: 0.5 mm.
    eu_only = await models_api.get(f"/model/{RUN}/{RUN + 4 * H}/classic/8/{x}/{y}.png")
    pixels = np.asarray(Image.open(io.BytesIO(await eu_only.read())).convert("RGBA"))
    assert tuple(pixels[128, 128]) == CLASSIC.colours[1]
    # A replaced run or a bad version.
    assert (
        await models_api.get(f"/model/{RUN - 6 * H}/{hour}/classic/8/{x}/{y}.png")
    ).status == 404
    assert (await models_api.get(f"/model/x-y/{hour}/classic/8/{x}/{y}.png")).status == 404


async def test_temperature_uses_d2_inside_its_area(models_api: TestClient) -> None:
    body = await (await models_api.get("/temperature")).json()
    assert body["hours"][0]["version"] == f"{RUN}-{RUN}"
    point = await models_api.get(
        f"/temperature/point?lat={ALPS[0]}&lon={ALPS[1]}&valid={RUN + 3 * H}"
    )
    # ICON-D2: 5 C at its ground of 1000 m, real ground 1000 m. ICON-EU would say 3.25 C.
    assert (await point.json())["temperature"] == pytest.approx(5.0, abs=0.05)
    x, y = tile_of(*ALPS, 8)
    tile = await models_api.get(f"/temperature/{RUN}-{RUN}/{RUN + 3 * H}/temp2c/8/{x}/{y}.png")
    assert tile.status == 200
    # The older URL form (ICON-EU only) still works.
    old = await models_api.get(f"/temperature/{RUN}/{RUN + 3 * H}/temp2c/8/{x}/{y}.png")
    assert old.status == 200 and await old.read() != await tile.read()


async def test_health_reports_each_model_field(models_api: TestClient) -> None:
    body = await (await models_api.get("/health")).json()
    assert body["models"]["icon-d2 tot_prec"]["hours"] == 1
    assert body["models"]["icon-eu snowlmt"]["last"] == "2026-11-02T03:00:00Z"


def test_height_matching_prefers_cells_at_the_real_height() -> None:
    from weather_radar.temperature import corrected, height_matched

    # A valley cell at 300 m holding cold air (4 C) next to slopes at 1300 m (2 C).
    t = np.full((ICON_D2.rows, ICON_D2.cols), 2.0, np.float32)
    z = np.full((ICON_D2.rows, ICON_D2.cols), 1300.0, np.float32)
    r, c = 300, 600
    t[r, c] = 4.0
    z[r, c] = 300.0
    lat = np.array([ICON_D2.lat0 + (r + 0.4) * ICON_D2.step])
    lon = np.array([ICON_D2.lon0 + (c + 0.4) * ICON_D2.step])
    ground = np.array([300.0], np.float32)
    matched = float(height_matched(t, z, lat, lon, ground)[0])
    # The valley cell decides, instead of slopes warmed by 6.5 K/km (8.5 C).
    assert matched == pytest.approx(4.0, abs=0.1)
    plain = corrected(
        modeldata.sample(t, lat, lon, ICON_D2), modeldata.sample(z, lat, lon, ICON_D2), ground
    )
    assert float(plain[0]) > 6.0
