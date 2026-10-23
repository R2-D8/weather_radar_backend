"""The HTTP API the phone talks to.

GET /frames                                  frame ids (epoch seconds) and times, oldest first
GET /tiles/{frame}/{palette}/{z}/{x}/{y}.png a map tile (?size=256 or 512)
GET /palettes                                colour palettes with their legend bands
GET /health                                  200 when frames are fresh, 503 otherwise
GET /attribution                             data credit to show in the app
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from aiohttp import web

from weather_radar.config import Settings
from weather_radar.dpc import FRAME_STEP_SECONDS
from weather_radar.frames import FrameStore
from weather_radar.ingest import Ingestor
from weather_radar.palettes import DEFAULT_PALETTE, PALETTES, Palette
from weather_radar.render import TILE_SIZES, render_tile
from weather_radar.tilecache import TileCache

ATTRIBUTION = (
    "Radar data: Radar-DPC, Presidenza del Consiglio dei Ministri - "
    "Dipartimento della Protezione Civile (CC BY-SA 4.0)"
)
#: Frames older than this make /health report "stale".
STALE_AFTER_SECONDS = 30 * 60
#: Tiles rendering at the same time; rendering is CPU work done in worker threads.
MAX_PARALLEL_RENDERS = 4
#: Past frames never change, so phones may keep their tiles for good.
IMMUTABLE = "public, max-age=31536000, immutable"
#: The newest frame is replaced soon; a short cache still saves repeated requests while panning.
SHORT = "public, max-age=60"


@dataclass
class AppState:
    settings: Settings
    store: FrameStore
    cache: TileCache
    ingestor: Ingestor | None
    renders: asyncio.Semaphore
    started: float
    clock: Callable[[], float] = time.time


STATE = web.AppKey("state", AppState)
INGEST_TASK = web.AppKey("ingest_task", asyncio.Task[None])


def _iso(frame_id: int) -> str:
    return datetime.fromtimestamp(frame_id, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hex(rgb_alpha: tuple[int, int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb_alpha[:3])


async def frames(request: web.Request) -> web.Response:
    state = request.app[STATE]
    ids = state.store.ids()
    return web.json_response(
        {
            "frames": [{"id": i, "time": _iso(i)} for i in ids],
            "step_seconds": FRAME_STEP_SECONDS,
        },
        headers={"Cache-Control": "public, max-age=30"},
    )


async def palettes(request: web.Request) -> web.Response:
    def describe(p: Palette) -> dict[str, Any]:
        return {
            "name": p.name,
            "label": p.label,
            "unit": "mm/h",
            "bands": [
                {"from": t, "color": _hex(c), "alpha": c[3]}
                for t, c in zip(p.thresholds, p.colours, strict=True)
            ],
        }

    return web.json_response(
        {"default": DEFAULT_PALETTE, "palettes": [describe(p) for p in PALETTES.values()]},
        headers={"Cache-Control": "public, max-age=3600"},
    )


async def attribution(request: web.Request) -> web.Response:
    return web.json_response({"attribution": ATTRIBUTION})


async def health(request: web.Request) -> web.Response:
    state = request.app[STATE]
    ids = state.store.ids()
    now = state.clock()
    newest_age = None if not ids else max(0.0, now - ids[-1])
    if not ids:
        status = "starting"
    elif newest_age is not None and newest_age > STALE_AFTER_SECONDS:
        status = "stale"
    else:
        status = "ok"
    body: dict[str, Any] = {
        "status": status,
        "frames": len(ids),
        "newest_frame": None if not ids else _iso(ids[-1]),
        "newest_frame_age_seconds": None if newest_age is None else int(newest_age),
        "uptime_seconds": int(now - state.started),
        "tile_cache": {"tiles": len(state.cache), "bytes": state.cache.size_bytes},
    }
    if state.ingestor is not None:
        s = state.ingestor.status
        body["ingest"] = {
            "last_success_age_seconds": None
            if s.last_success is None
            else int(now - s.last_success),
            "last_error": s.last_error,
            "downloaded_total": s.downloaded_total,
        }
    return web.json_response(body, status=200 if status == "ok" else 503)


def _int(request: web.Request, name: str) -> int:
    try:
        return int(request.match_info[name])
    except ValueError:
        raise web.HTTPNotFound from None


async def tile(request: web.Request) -> web.Response:
    state = request.app[STATE]
    frame_id = _int(request, "frame")
    z, x, y = _int(request, "z"), _int(request, "x"), _int(request, "y")
    palette = PALETTES.get(request.match_info["palette"])
    if palette is None:
        raise web.HTTPNotFound(text="unknown palette")
    if not 0 <= z <= state.settings.max_zoom or not (0 <= x < 2**z and 0 <= y < 2**z):
        raise web.HTTPNotFound(text="tile outside the served range")
    size_text = request.query.get("size", "256")
    if size_text not in {str(s) for s in TILE_SIZES}:
        raise web.HTTPBadRequest(text=f"size must be one of {TILE_SIZES}")
    size = int(size_text)

    key = (frame_id, palette.name, size, z, x, y)
    png = state.cache.get(key)
    if png is None:
        try:
            frame = await asyncio.to_thread(state.store.load, frame_id)
        except KeyError:
            raise web.HTTPNotFound(text="no such frame") from None
        async with state.renders:
            png = await asyncio.to_thread(render_tile, frame, palette, z, x, y, size)
        state.cache.put(key, png)

    ids = state.store.ids()
    is_newest = bool(ids) and frame_id == ids[-1]
    return web.Response(
        body=png,
        content_type="image/png",
        headers={"Cache-Control": SHORT if is_newest else IMMUTABLE},
    )


async def _ingest_lifecycle(app: web.Application) -> AsyncIterator[None]:
    ingestor = app[STATE].ingestor
    task = asyncio.create_task(ingestor.run()) if ingestor is not None else None
    yield
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def create_app(
    settings: Settings,
    store: FrameStore,
    ingestor: Ingestor | None,
    clock: Callable[[], float] = time.time,
) -> web.Application:
    app = web.Application()
    app[STATE] = AppState(
        settings=settings,
        store=store,
        cache=TileCache(settings.tile_cache_mib * 1024 * 1024),
        ingestor=ingestor,
        renders=asyncio.Semaphore(MAX_PARALLEL_RENDERS),
        started=clock(),
        clock=clock,
    )
    app.add_routes(
        [
            web.get("/frames", frames),
            web.get("/palettes", palettes),
            web.get("/attribution", attribution),
            web.get("/health", health),
            web.get(r"/tiles/{frame}/{palette}/{z}/{x}/{y}.png", tile),
        ]
    )
    app.cleanup_ctx.append(_ingest_lifecycle)
    return app
