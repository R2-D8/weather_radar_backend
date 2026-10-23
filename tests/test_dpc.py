from __future__ import annotations

from collections.abc import AsyncIterator

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from weather_radar.dpc import DpcClient, DpcError
from weather_radar.frames import decode_geotiff


class FakeDpc:
    """A stand-in for the DPC API that serves the sample frame."""

    def __init__(self, frame_bytes: bytes) -> None:
        self.frame_bytes = frame_bytes
        self.requested: list[dict[str, object]] = []
        self.latest_body: object = {"lastProducts": [{"productType": "SRI", "time": 1792791900000}]}
        self.latest_status = 200
        self.download_status = 200
        self.file_status = 200
        self.link: object = "FILE"

    def app(self) -> web.Application:
        app = web.Application()
        app.add_routes(
            [
                web.get("/findLastProductByType", self.find_last),
                web.post("/downloadProduct", self.download_product),
                web.get("/files/frame.tif", self.file),
            ]
        )
        return app

    async def find_last(self, request: web.Request) -> web.Response:
        assert request.query["type"] == "SRI"
        return web.json_response(self.latest_body, status=self.latest_status)

    async def download_product(self, request: web.Request) -> web.Response:
        self.requested.append(await request.json())
        link = str(request.url.with_path("/files/frame.tif")) if self.link == "FILE" else self.link
        return web.json_response({"url": link}, status=self.download_status)

    async def file(self, request: web.Request) -> web.Response:
        return web.Response(body=self.frame_bytes, status=self.file_status)


@pytest.fixture
async def dpc(sample_bytes: bytes) -> AsyncIterator[tuple[DpcClient, FakeDpc]]:
    fake = FakeDpc(sample_bytes)
    async with TestServer(fake.app()) as server, aiohttp.ClientSession() as session:
        yield DpcClient(session, base_url=str(server.make_url(""))), fake


async def test_latest_returns_epoch_seconds(dpc: tuple[DpcClient, FakeDpc]) -> None:
    client, _ = dpc
    assert await client.latest() == 1_792_791_900


@pytest.mark.parametrize(
    "body",
    [{}, {"lastProducts": []}, {"lastProducts": [{"time": "soon"}]}, ["not", "a", "dict"]],
)
async def test_latest_rejects_unexpected_answers(
    dpc: tuple[DpcClient, FakeDpc], body: object
) -> None:
    client, fake = dpc
    fake.latest_body = body
    with pytest.raises(DpcError):
        await client.latest()


async def test_latest_reports_http_errors(dpc: tuple[DpcClient, FakeDpc]) -> None:
    client, fake = dpc
    fake.latest_status = 503
    with pytest.raises(DpcError, match="503"):
        await client.latest()


async def test_download_sends_the_frame_time_in_milliseconds(
    dpc: tuple[DpcClient, FakeDpc],
) -> None:
    client, fake = dpc
    data = await client.download(1_792_791_900)
    assert fake.requested == [{"productType": "SRI", "productDate": 1_792_791_900_000}]
    assert decode_geotiff(data).shape == (1400, 1200)


async def test_download_fails_cleanly(dpc: tuple[DpcClient, FakeDpc]) -> None:
    client, fake = dpc
    fake.download_status = 404
    with pytest.raises(DpcError, match="404"):
        await client.download(1)
    fake.download_status = 200
    fake.link = None
    with pytest.raises(DpcError, match="no download link"):
        await client.download(1)
    fake.link = "FILE"
    fake.file_status = 403
    with pytest.raises(DpcError, match="403"):
        await client.download(1)


async def test_unreachable_server_is_a_dpc_error() -> None:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
        with pytest.raises(DpcError):
            await DpcClient(session, base_url="http://127.0.0.1:9").latest()


async def test_non_https_download_links_are_refused_for_the_real_api() -> None:
    class Plain:
        pass

    async with aiohttp.ClientSession() as session:
        client = DpcClient(session)  # base URL is the real https API
        client._json = _fake_json  # type: ignore[method-assign]
        with pytest.raises(DpcError, match="non-https"):
            await client.download(1)


async def _fake_json(*_: object, **__: object) -> dict[str, str]:
    return {"url": "http://example.invalid/frame.tif"}
