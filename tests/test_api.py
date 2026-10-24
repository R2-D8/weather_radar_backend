from __future__ import annotations

import io
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from weather_radar.api import (
    IMMUTABLE,
    SHORT,
    STALE_AFTER_SECONDS,
    STATE,
    create_app,
)
from weather_radar.config import Settings
from weather_radar.frames import FrameStore
from weather_radar.ingest import Ingestor
from weather_radar.render import empty_tile

NEWEST = 1_792_791_900
FRAMES = [NEWEST - 600, NEWEST - 300, NEWEST]
# North-east Italy at zoom 8, where the sample frame has its strongest rain.
RAINY = "8/137/91"


class Now:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


@pytest.fixture
def now() -> Now:
    return Now(NEWEST + 120)


@pytest.fixture
async def client(tmp_path: Path, sample_bytes: bytes, now: Now) -> AsyncIterator[TestClient]:
    store = FrameStore(tmp_path)
    for frame_id in FRAMES:
        store.save(frame_id, sample_bytes)
    settings = Settings(data_dir=tmp_path)
    async with TestClient(TestServer(create_app(settings, store, None, clock=now))) as c:
        yield c


async def test_frames_are_listed_oldest_first(client: TestClient) -> None:
    response = await client.get("/frames")
    body = await response.json()
    assert response.status == 200
    assert [f["id"] for f in body["frames"]] == FRAMES
    assert body["frames"][-1]["time"] == "2026-10-23T21:45:00Z"
    assert body["step_seconds"] == 300


async def test_health_ok_when_frames_are_fresh(client: TestClient) -> None:
    response = await client.get("/health")
    body = await response.json()
    assert response.status == 200
    assert body["status"] == "ok"
    assert body["frames"] == 3
    assert body["newest_frame_age_seconds"] == 120


async def test_health_reports_stale_frames(client: TestClient, now: Now) -> None:
    now.value = NEWEST + STALE_AFTER_SECONDS + 1
    response = await client.get("/health")
    assert response.status == 503
    assert (await response.json())["status"] == "stale"


async def test_health_reports_starting_without_frames(tmp_path: Path, now: Now) -> None:
    app = create_app(Settings(data_dir=tmp_path), FrameStore(tmp_path / "empty"), None, clock=now)
    async with TestClient(TestServer(app)) as c:
        response = await c.get("/health")
        assert response.status == 503
        assert (await response.json())["status"] == "starting"


async def test_health_includes_ingest_status(tmp_path: Path, sample_bytes: bytes, now: Now) -> None:
    class Dummy:
        pass

    store = FrameStore(tmp_path)
    ingestor = Ingestor(Dummy(), store, timedelta(hours=1), 60, clock=now)  # type: ignore[arg-type]
    ingestor.status.last_error = "DPC down"
    app = create_app(Settings(data_dir=tmp_path), store, None, clock=now)
    async with TestClient(TestServer(app)) as c:
        # attached after startup so the app does not start polling with the dummy client
        app[STATE].ingestor = ingestor
        body = await (await c.get("/health")).json()
    assert body["ingest"]["last_error"] == "DPC down"


async def test_tile_is_a_png_with_long_cache_for_past_frames(client: TestClient) -> None:
    response = await client.get(f"/tiles/{FRAMES[0]}/classic/{RAINY}.png")
    assert response.status == 200
    assert response.headers["Content-Type"] == "image/png"
    assert response.headers["Cache-Control"] == IMMUTABLE
    body = await response.read()
    assert Image.open(io.BytesIO(body)).size == (256, 256)
    assert body != empty_tile(256)


async def test_newest_frame_gets_a_short_cache(client: TestClient) -> None:
    response = await client.get(f"/tiles/{NEWEST}/classic/{RAINY}.png")
    assert response.status == 200
    assert response.headers["Cache-Control"] == SHORT


async def test_second_request_comes_from_the_cache(client: TestClient) -> None:
    url = f"/tiles/{FRAMES[0]}/classic/{RAINY}.png"
    first = await (await client.get(url)).read()
    cache = client.server.app[STATE].cache
    assert len(cache) == 1
    assert await (await client.get(url)).read() == first
    assert len(cache) == 1


async def test_tile_size_option(client: TestClient) -> None:
    response = await client.get(f"/tiles/{NEWEST}/classic/7/68/45.png?size=512")
    assert response.status == 200
    assert Image.open(io.BytesIO(await response.read())).size == (512, 512)
    assert (await client.get(f"/tiles/{NEWEST}/classic/{RAINY}.png?size=100")).status == 400


@pytest.mark.parametrize(
    "path",
    [
        "/tiles/1/classic/8/137/91.png",  # no such frame
        f"/tiles/{NEWEST}/neon/8/137/91.png",  # no such palette
        f"/tiles/{NEWEST}/classic/9/274/182.png",  # beyond the highest served zoom
        f"/tiles/{NEWEST}/classic/8/256/91.png",  # x out of range
        f"/tiles/{NEWEST}/classic/8/137/-1.png",  # y out of range
        "/tiles/abc/classic/8/137/91.png",  # not a number
        f"/tiles/{NEWEST}/classic/8/1x/91.png",
    ],
)
async def test_bad_tile_requests_are_404(client: TestClient, path: str) -> None:
    assert (await client.get(path)).status == 404


async def test_empty_tiles_are_tiny(client: TestClient) -> None:
    response = await client.get(f"/tiles/{NEWEST}/classic/8/150/50.png")  # Finland
    assert response.status == 200
    assert await response.read() == empty_tile(256)


async def test_palettes_describe_the_legend(client: TestClient) -> None:
    body = await (await client.get("/palettes")).json()
    assert body["default"] == "classic"
    classic = next(p for p in body["palettes"] if p["name"] == "classic")
    assert classic["unit"] == "mm/h"
    assert classic["bands"][0] == {"from": 0.1, "color": "#aae1ff", "alpha": 170}
    assert [b["from"] for b in classic["bands"]] == sorted(b["from"] for b in classic["bands"])


async def test_attribution(client: TestClient) -> None:
    body = await (await client.get("/attribution")).json()
    assert "Radar-DPC" in body["attribution"]


async def test_corrupt_frame_is_404_not_a_crash(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / f"{FRAMES[0]}.tif").write_bytes(b"corrupt")
    store = client.server.app[STATE].store
    store._decoded.clear()
    assert (await client.get(f"/tiles/{FRAMES[0]}/classic/{RAINY}.png")).status == 404
    assert not store.has(FRAMES[0])


async def test_frames_say_how_to_use_the_tiles(client: TestClient) -> None:
    body = await (await client.get("/frames")).json()
    assert body["max_zoom"] == 8
    assert body["tile_size"] == 256


@pytest.mark.parametrize(
    ("lat", "lon", "covered"),
    [
        (41.9, 12.5, True),  # Rome
        (45.65, 13.78, True),  # Trieste
        (46.2, 6.14, False),  # Geneva: inside the grid, but DPC has no data there
        (60.45, 22.27, False),  # Turku: outside the grid
        (0.0, 0.0, False),
    ],
)
async def test_coverage_at_a_point(
    client: TestClient, lat: float, lon: float, covered: bool
) -> None:
    response = await client.get(f"/coverage?lat={lat}&lon={lon}")
    assert response.status == 200
    body = await response.json()
    assert body["covered"] is covered
    assert body["frame"] == NEWEST


@pytest.mark.parametrize(
    "query", ["", "?lat=45", "?lat=abc&lon=12", "?lat=95&lon=12", "?lat=45&lon=200"]
)
async def test_coverage_rejects_bad_requests(client: TestClient, query: str) -> None:
    assert (await client.get(f"/coverage{query}")).status == 400


async def test_coverage_without_frames_is_503(tmp_path: Path, now: Now) -> None:
    app = create_app(Settings(data_dir=tmp_path), FrameStore(tmp_path / "empty"), None, clock=now)
    async with TestClient(TestServer(app)) as c:
        assert (await c.get("/coverage?lat=45&lon=12")).status == 503


async def test_frames_list_the_short_forecast(client: TestClient) -> None:
    body = await (await client.get("/frames")).json()
    # FRAMES has the newest frame and the one 10 minutes earlier, so a forecast can be made.
    assert body["forecast_base"] == NEWEST
    assert [f["lead_minutes"] for f in body["forecast"]] == [10, 20, 30]
    assert [f["id"] for f in body["forecast"]] == [NEWEST + 600, NEWEST + 1200, NEWEST + 1800]


async def test_forecast_tiles(client: TestClient) -> None:
    await client.get("/frames")
    response = await client.get(f"/forecast/{NEWEST}/20/classic/{RAINY}.png")
    assert response.status == 200
    assert response.headers["Cache-Control"] == IMMUTABLE
    assert Image.open(io.BytesIO(await response.read())).size == (256, 256)


@pytest.mark.parametrize(
    "path",
    [
        f"/forecast/{NEWEST - 300}/20/classic/8/137/91.png",  # an older base: replaced
        f"/forecast/{NEWEST}/25/classic/8/137/91.png",  # not a forecast lead time
        f"/forecast/{NEWEST}/20/neon/8/137/91.png",
        f"/forecast/{NEWEST}/20/classic/9/1/1.png",
    ],
)
async def test_bad_forecast_requests_are_404(client: TestClient, path: str) -> None:
    assert (await client.get(path)).status == 404


async def test_no_forecast_without_a_frame_ten_minutes_earlier(
    tmp_path: Path, sample_bytes: bytes, now: Now
) -> None:
    store = FrameStore(tmp_path / "one")
    store.save(NEWEST, sample_bytes)
    app = create_app(Settings(data_dir=tmp_path), store, None, clock=now)
    async with TestClient(TestServer(app)) as c:
        body = await (await c.get("/frames")).json()
    assert body["forecast"] == []
    assert body["forecast_base"] is None
