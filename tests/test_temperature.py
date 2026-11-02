from __future__ import annotations

import bz2
import io
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import eccodes
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from weather_radar import icon, modeldata
from weather_radar import temperature as temp
from weather_radar.__main__ import to_celsius
from weather_radar.api import MODELS, ModelFeature, create_app
from weather_radar.config import Settings
from weather_radar.frames import FrameStore
from weather_radar.icon import ICON_EU
from weather_radar.modeldata import FieldJob, Hour, HourlyStore, ModelIngestor, plan_downloads
from weather_radar.temperature import TEMP_2C, ModelHour
from weather_radar.terrain import Terrain, TerrainError, point_pixel

H = 3600
RUN = 1_793_577_600  # 2026-11-02 00:00 UTC
NOW = RUN + 9 * H


def grib_bytes(values: np.ndarray, lon0: float = icon.GRID_LON0) -> bytes:
    """A bzip2-compressed GRIB2 message laid out like ICON-EU's regular grid."""
    handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        nj, ni = values.shape
        lat1 = icon.GRID_LAT0 + (nj - 1) * icon.GRID_STEP
        lon1 = lon0 + (ni - 1) * icon.GRID_STEP
        eccodes.codes_set(handle, "Ni", ni)
        eccodes.codes_set(handle, "Nj", nj)
        eccodes.codes_set(handle, "jScansPositively", 1)
        eccodes.codes_set(handle, "latitudeOfFirstGridPointInDegrees", icon.GRID_LAT0)
        eccodes.codes_set(handle, "longitudeOfFirstGridPointInDegrees", lon0 % 360)
        eccodes.codes_set(handle, "latitudeOfLastGridPointInDegrees", lat1)
        eccodes.codes_set(handle, "longitudeOfLastGridPointInDegrees", lon1 % 360)
        eccodes.codes_set(handle, "iDirectionIncrementInDegrees", icon.GRID_STEP)
        eccodes.codes_set(handle, "jDirectionIncrementInDegrees", icon.GRID_STEP)
        eccodes.codes_set_values(handle, values.astype(np.float64).ravel())
        return bz2.compress(eccodes.codes_get_message(handle))
    finally:
        eccodes.codes_release(handle)


def field(value: float) -> np.ndarray:
    return np.full((icon.GRID_ROWS, icon.GRID_COLS), value, dtype=np.float32)


def terrarium_png(height_m: float) -> bytes:
    v = height_m + 32768.0
    r, g = divmod(int(v), 256)
    b = round((v - int(v)) * 256)
    image = Image.new("RGB", (256, 256), (r, g, b))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def tile_of(lat: float, lon: float, z: int) -> tuple[int, int]:
    tx, ty, _, _ = point_pixel(lat, lon, z)
    return tx, ty


# --- decoding ---------------------------------------------------------------------------------


def test_decode_reads_rows_south_to_north() -> None:
    values = field(280.0)
    values[0, 0] = 250.0  # south-west corner
    values[-1, -1] = 300.0  # north-east corner
    grid = icon.decode(grib_bytes(values))
    assert grid.shape == (icon.GRID_ROWS, icon.GRID_COLS)
    assert grid[0, 0] == pytest.approx(250.0, abs=0.01)
    assert grid[-1, -1] == pytest.approx(300.0, abs=0.01)


def test_decode_rejects_a_shifted_grid() -> None:
    with pytest.raises(icon.IconError, match="origin"):
        icon.decode(grib_bytes(field(280.0), lon0=-20.0))


def test_decode_rejects_garbage() -> None:
    with pytest.raises(icon.IconError):
        icon.decode(b"not bzip2")


def test_urls_follow_dwd_naming() -> None:
    assert icon.t2m_url(RUN + 6 * H, 12) == (
        "https://opendata.dwd.de/weather/nwp/icon-eu/grib/06/t_2m/"
        "icon-eu_europe_regular-lat-lon_single-level_2026110206_012_T_2M.grib2.bz2"
    )
    assert icon.hsurf_url(RUN).endswith(
        "/00/hsurf/icon-eu_europe_regular-lat-lon_time-invariant_2026110200_HSURF.grib2.bz2"
    )
    assert icon.latest_run_ids(RUN + 5 * H, 2) == [RUN, RUN - 6 * H]


# --- planning ---------------------------------------------------------------------------------


def test_plan_fills_the_whole_window_from_the_newest_covering_run() -> None:
    runs = [RUN, RUN - 6 * H, RUN - 12 * H, RUN - 18 * H]
    plan = plan_downloads(NOW, runs, {}, timedelta(hours=24), timedelta(hours=48))
    by_valid = {h.valid: h.run for h in plan}
    # Future hours, 48 h from the current hour (RUN + 9 h), come from the newest run.
    assert max(by_valid) == RUN + 57 * H
    assert by_valid[RUN + 57 * H] == RUN
    assert by_valid[RUN] == RUN
    # Earlier hours from the newest run that had started by then.
    assert by_valid[RUN - 1 * H] == RUN - 6 * H
    assert by_valid[RUN - 13 * H] == RUN - 18 * H
    # The window starts 24 h before the current hour (RUN + 9 h).
    assert min(by_valid) == RUN - 15 * H
    assert plan[0].valid == RUN + 57 * H  # newest first
    assert modeldata.max_step(timedelta(hours=48)) == 60
    assert modeldata.max_step(timedelta(hours=90)) == 78


def test_plan_skips_hours_already_stored_from_the_same_or_a_newer_run() -> None:
    runs = [RUN, RUN - 6 * H]
    stored = {RUN + h * H: RUN for h in range(61)}
    stored[RUN - 3 * H] = RUN - 6 * H
    plan = plan_downloads(RUN, runs, stored, timedelta(hours=6), timedelta(hours=48))
    assert [h.valid for h in plan] == [
        RUN - 1 * H,
        RUN - 2 * H,
        RUN - 4 * H,
        RUN - 5 * H,
        RUN - 6 * H,
    ]


def test_plan_replaces_hours_when_a_newer_run_arrives() -> None:
    stored = {RUN + 10 * H: RUN - 6 * H}
    plan = plan_downloads(NOW, [RUN, RUN - 6 * H], stored, timedelta(0), timedelta(hours=48))
    assert Hour(RUN + 10 * H, RUN) in plan


# --- maps -------------------------------------------------------------------------------------


def test_palette_bands_clamp_and_mark_missing() -> None:
    t = np.array([np.nan, -50.0, -30.0, -29.0, 0.0, 13.9, 45.9, 60.0], dtype=np.float32)
    assert TEMP_2C.bands(t).tolist() == [0, 1, 1, 1, 16, 22, 38, 38]
    assert TEMP_2C.band_count == 38
    rgb, alpha = TEMP_2C.png_palette()
    assert len(rgb) == 3 * 39 and alpha[0] == 0 and set(alpha[1:]) == {255}


def test_height_correction_warms_valleys_below_the_model_ground() -> None:
    t = temp.corrected(
        np.array([0.0, 0.0, 10.0], np.float32),
        np.array([1000.0, 0.0, 5.0], np.float32),
        np.array([300.0, 1000.0, -40.0], np.float32),  # valley, peak, sea floor
    )
    assert t.tolist() == pytest.approx([4.55, -6.5, 10.0325])


def test_sample_is_nan_outside_the_model_area() -> None:
    grid = field(5.0)
    lat = np.array([45.0, 20.0, 45.0])
    lon = np.array([10.0, 10.0, 70.0])
    assert np.isnan(modeldata.sample(grid, lat, lon)).tolist() == [False, True, True]


def test_render_tile_uses_the_ground_height() -> None:
    z, (x, y) = 8, tile_of(46.2, 9.9, 8)
    flat = np.full((256, 256), 650.0, np.float32)
    png = temp.render_tile(ModelHour(field(10.0), field(0.0) + 650.0), flat, TEMP_2C, z, x, y)
    image = Image.open(io.BytesIO(png))
    assert image.mode == "P"
    # Model and real ground agree: 10 C, band 20 -> palette index 21.
    assert set(np.asarray(image).ravel().tolist()) == {21}
    valley = np.full((256, 256), 0.0, np.float32)
    png = temp.render_tile(ModelHour(field(10.0), field(0.0) + 650.0), valley, TEMP_2C, z, x, y)
    # 650 m lower: 14.2 C, band 22 -> index 23.
    assert set(np.asarray(Image.open(io.BytesIO(png))).ravel().tolist()) == {23}


def test_tiles_outside_the_model_area() -> None:
    assert temp.tile_overlaps_model(5, 17, 11)  # Italy
    assert not temp.tile_overlaps_model(5, 5, 11)  # mid-Atlantic, west of the grid
    assert Image.open(io.BytesIO(temp.empty_tile())).size == (256, 256)


# --- storage ----------------------------------------------------------------------------------


def test_store_keeps_only_the_newest_run_per_hour(tmp_path: Path) -> None:
    store = HourlyStore(tmp_path)
    store.save(Hour(RUN + H, RUN - 6 * H), field(1.0))
    store.save(Hour(RUN + H, RUN), field(2.0))
    store.save(Hour(RUN + 2 * H, RUN), field(3.0))
    assert store.hours() == [Hour(RUN + H, RUN), Hour(RUN + 2 * H, RUN)]
    assert len(list(tmp_path.glob("*.npy"))) == 2
    assert float(store.load(Hour(RUN + H, RUN))[10, 10]) == 2.0
    with pytest.raises(KeyError):
        store.load(Hour(RUN + H, RUN - 6 * H))
    assert store.prune(RUN + 2 * H) == 1
    assert store.hours() == [Hour(RUN + 2 * H, RUN)]


# --- terrain ----------------------------------------------------------------------------------


async def test_terrain_fetches_once_and_then_reads_from_disk(tmp_path: Path) -> None:
    calls: list[str] = []

    async def serve(request: web.Request) -> web.Response:
        calls.append(request.path)
        return web.Response(body=terrarium_png(812.5), content_type="image/png")

    app = web.Application()
    app.router.add_get("/terrarium/{z}/{x}/{y}.png", serve)
    async with TestServer(app) as server, TestClient(server) as client:
        base = str(server.make_url("/terrarium"))
        terrain = Terrain(client.session, tmp_path, base_url=base)
        heights = await terrain.heights(8, 136, 91)
        assert heights.shape == (256, 256) and float(heights[0, 0]) == pytest.approx(812.5)
        await terrain.heights(8, 136, 91)
        assert calls == ["/terrarium/8/136/91.png"]
        assert (tmp_path / "8" / "136" / "91.png").exists()

        # A new instance (after a restart) reads the disk copy without asking again.
        again = Terrain(None, tmp_path)
        assert float((await again.heights(8, 136, 91))[5, 5]) == pytest.approx(812.5)
        assert calls == ["/terrarium/8/136/91.png"]


async def test_terrain_errors_are_not_cached(tmp_path: Path) -> None:
    async def fail(request: web.Request) -> web.Response:
        return web.Response(status=503)

    app = web.Application()
    app.router.add_get("/terrarium/{z}/{x}/{y}.png", fail)
    async with TestServer(app) as server, TestClient(server) as client:
        terrain = Terrain(client.session, tmp_path, base_url=str(server.make_url("/terrarium")))
        with pytest.raises(TerrainError):
            await terrain.heights(3, 4, 2)
        assert not any(tmp_path.rglob("*.png"))


# --- ingest -----------------------------------------------------------------------------------


class FakeIcon:
    """Serves synthetic files for the runs in [runs]; counts downloads."""

    def __init__(self, runs: set[int]) -> None:
        self.runs = runs
        self.downloads: list[str] = []
        self.kelvin = grib_bytes(field(283.15))
        self.ground = grib_bytes(field(100.0))

    def field_url(self, model: icon.Model, name: str, run_id: int, step: int) -> str:
        return f"{model.name}/{name}/{run_id}/{step}"

    def hsurf_url(self, model: icon.Model, run_id: int) -> str:
        return f"{model.name}/hsurf/{run_id}/0"

    async def exists(self, url: str) -> bool:
        return int(url.split("/")[2]) in self.runs

    async def download(self, url: str) -> bytes | None:
        self.downloads.append(url)
        _, name, run, step = url.split("/")
        if name == "hsurf":
            return self.ground
        if int(run) not in self.runs:
            return None
        if name == "tot_prec":
            # 1.5 mm in every hour: the sum from the run start grows by 1.5 a step.
            return grib_bytes(field(1.5 * int(step)))
        return self.kelvin


async def _no_sleep(_: float) -> None:
    return None


async def test_ingest_fills_the_store_and_converts_to_celsius(tmp_path: Path) -> None:
    fake = FakeIcon({RUN, RUN - 6 * H})
    store = HourlyStore(tmp_path)
    job = FieldJob(
        ICON_EU,
        "t_2m",
        store,
        past=timedelta(hours=5),
        future=timedelta(hours=4),
        needs_ground=True,
        convert=to_celsius,
    )
    ingestor = ModelIngestor(
        fake,  # type: ignore[arg-type]
        [job],
        clock=lambda: RUN + 3 * H,
        sleep=_no_sleep,
    )
    # Window RUN - 2 h .. RUN + 7 h. Runs are read to + 16 h (4 h + 12 h margin), so all of it is
    # covered: RUN - 2 h and RUN - 1 h from the previous run, the rest from the newest.
    assert await ingestor.cycle() == 10
    hours = store.hours()
    assert [h.valid for h in hours] == [RUN + k * H for k in range(-2, 8)]
    assert [h.run for h in hours[:3]] == [RUN - 6 * H, RUN - 6 * H, RUN]
    assert float(store.load(hours[-1])[0, 0]) == pytest.approx(10.0, abs=0.05)
    assert store.has_ground()
    # A second cycle has nothing left to do.
    assert await ingestor.cycle() == 0
    assert ingestor.status[job.name].newest_run == RUN


# --- API --------------------------------------------------------------------------------------

ROME = (41.9, 12.5)


@pytest.fixture
async def api(tmp_path: Path) -> AsyncIterator[TestClient[web.Request, web.Application]]:
    store = HourlyStore(tmp_path / "temperature")
    store.save_ground(field(500.0))
    store.save(Hour(RUN + H, RUN), field(12.0))
    # Terrain on disk only (no network): 175 m everywhere around Rome.
    terrain_dir = tmp_path / "terrain"
    for z in (8, 11):
        tx, ty = tile_of(*ROME, z)
        path = terrain_dir / str(z) / str(tx) / f"{ty}.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(terrarium_png(175.0))
    feature = ModelFeature(terrain=Terrain(None, terrain_dir), t2m_eu=store)
    settings = Settings(data_dir=tmp_path)
    app = create_app(settings, FrameStore(tmp_path / "frames"), None, models=feature)
    assert app[MODELS] is feature
    async with TestServer(app) as server, TestClient(server) as client:
        yield client


async def test_temperature_hours_lists_runs_and_legend(
    api: TestClient[web.Request, web.Application],
) -> None:
    response = await api.get("/temperature")
    assert response.status == 200
    body = await response.json()
    assert body["hours"] == [
        {"id": RUN + H, "time": "2026-11-02T01:00:00Z", "run": RUN, "version": str(RUN)}
    ]
    assert body["palette"] == "temp2c" and len(body["bands"]) == 38
    assert body["bands"][0]["from"] == -30.0 and body["tile_size"] == 256
    assert "Deutscher Wetterdienst" in body["attribution"]


async def test_temperature_tile_and_point(api: TestClient[web.Request, web.Application]) -> None:
    tx, ty = tile_of(*ROME, 8)
    response = await api.get(f"/temperature/{RUN}/{RUN + H}/temp2c/8/{tx}/{ty}.png")
    assert response.status == 200
    assert response.headers["Cache-Control"].endswith("immutable")
    pixels = np.asarray(Image.open(io.BytesIO(await response.read())))
    # 12 C at a model ground of 500 m, real ground 175 m: 14.1 C, band 22 -> index 23.
    assert set(pixels.ravel().tolist()) == {23}

    response = await api.get(f"/temperature/point?lat={ROME[0]}&lon={ROME[1]}&valid={RUN + H}")
    assert response.status == 200
    body = await response.json()
    assert body["temperature"] == pytest.approx(14.1, abs=0.05)
    assert body["ground_m"] == 175


async def test_temperature_errors(api: TestClient[web.Request, web.Application]) -> None:
    # A replaced run, an unknown hour, a tile without terrain available, bad parameters.
    assert (
        await api.get(f"/temperature/{RUN - 6 * H}/{RUN + H}/temp2c/8/136/95.png")
    ).status == 404
    assert (await api.get(f"/temperature/point?lat=41.9&lon=12.5&valid={RUN}")).status == 404
    # Outside the model area: the shared empty tile, no terrain needed.
    outside = await api.get(f"/temperature/{RUN}/{RUN + H}/temp2c/8/10/10.png")
    assert outside.status == 200 and await outside.read() == temp.empty_tile()
    # Inside, but the terrain tile cannot be fetched: 503, so nobody keeps an uncorrected tile.
    tx, ty = tile_of(*ROME, 8)
    assert (await api.get(f"/temperature/{RUN}/{RUN + H}/temp2c/8/{tx + 1}/{ty}.png")).status == 503
    assert (await api.get(f"/temperature/{RUN}/{RUN + H}/nope/8/136/95.png")).status == 404
    assert (await api.get("/temperature/point?lat=x&lon=1&valid=1")).status == 400


async def test_temperature_is_absent_when_switched_off(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path), FrameStore(tmp_path / "frames"), None)
    async with TestServer(app) as server, TestClient(server) as client:
        assert (await client.get("/temperature")).status == 404
