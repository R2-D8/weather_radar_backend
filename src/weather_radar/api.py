"""The HTTP API the phone talks to.

GET /frames                                  frame ids (epoch seconds) and times, oldest first
GET /tiles/{frame}/{palette}/{z}/{x}/{y}.png a map tile (?size=256 or 512)
GET /forecast/{base}/{lead}/{palette}/{z}/{x}/{y}.png
                                             a tile of the short forecast made from frame {base},
                                             {lead} minutes ahead (listed in /frames as "forecast")
GET /coverage?lat=..&lon=..                  whether the newest frame has data at a point
GET /palettes                                colour palettes with their legend bands
GET /health                                  200 when frames are fresh, 503 otherwise
GET /attribution                             data credit to show in the app
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from aiohttp import web

from weather_radar import geo
from weather_radar import nowcast as nowcasting
from weather_radar.config import Settings
from weather_radar.dpc import FRAME_STEP_SECONDS
from weather_radar.frames import FrameStore
from weather_radar.ingest import Ingestor
from weather_radar.nowcast import MOTION_MINUTES, Nowcast
from weather_radar.palettes import DEFAULT_PALETTE, PALETTES, Palette
from weather_radar.render import TILE_SIZES, render_tile
from weather_radar.tilecache import TileCache

log = logging.getLogger(__name__)

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
    #: The forecast for the newest frame, made on first use after each new frame.
    nowcast: Nowcast | None = None
    nowcast_lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)


STATE = web.AppKey("state", AppState)
INGEST_TASK = web.AppKey("ingest_task", asyncio.Task[None])


def _iso(frame_id: int) -> str:
    return datetime.fromtimestamp(frame_id, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hex(rgb_alpha: tuple[int, int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb_alpha[:3])


async def current_nowcast(state: AppState) -> Nowcast | None:
    """The forecast for the newest frame, computing it if needed; None without enough frames."""
    ids = state.store.ids()
    if not ids:
        return None
    newest = ids[-1]
    earlier = newest - MOTION_MINUTES * 60
    if earlier not in ids:
        return None
    if state.nowcast is not None and state.nowcast.base_id == newest:
        return state.nowcast
    async with state.nowcast_lock:
        if state.nowcast is not None and state.nowcast.base_id == newest:
            return state.nowcast
        try:
            before = await asyncio.to_thread(state.store.load, earlier)
            latest = await asyncio.to_thread(state.store.load, newest)
            state.nowcast = await asyncio.to_thread(
                nowcasting.compute, before.level(0), latest.level(0), newest
            )
        except Exception:
            log.exception("could not compute the forecast for frame %d", newest)
            return None
        return state.nowcast


async def frames(request: web.Request) -> web.Response:
    state = request.app[STATE]
    ids = state.store.ids()
    forecast = await current_nowcast(state)
    return web.json_response(
        {
            "frames": [{"id": i, "time": _iso(i)} for i in ids],
            "step_seconds": FRAME_STEP_SECONDS,
            # What a client needs to set up a raster layer for these tiles.
            "max_zoom": state.settings.max_zoom,
            "tile_size": 256,
            # Short forecast made from the newest frame; tiles under /forecast/{base}/{lead}/.
            "forecast_base": None if forecast is None else forecast.base_id,
            "forecast": []
            if forecast is None
            else [
                {"lead_minutes": lead, "id": frame.id, "time": _iso(frame.id)}
                for lead, frame in sorted(forecast.frames.items())
            ],
        },
        headers={"Cache-Control": "public, max-age=30"},
    )


async def coverage(request: web.Request) -> web.Response:
    """Whether the newest frame has radar data at a point (dry areas count as covered)."""
    state = request.app[STATE]
    try:
        lat = float(request.query["lat"])
        lon = float(request.query["lon"])
    except (KeyError, ValueError):
        raise web.HTTPBadRequest(text="lat and lon are required numbers") from None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise web.HTTPBadRequest(text="lat or lon out of range")
    ids = state.store.ids()
    if not ids:
        raise web.HTTPServiceUnavailable(text="no frames yet")
    try:
        frame = await asyncio.to_thread(state.store.load, ids[-1])
    except KeyError:
        raise web.HTTPServiceUnavailable(text="no frames yet") from None
    cell = geo.lonlat_to_cell(lon, lat)
    covered = False
    if cell is not None:
        row, col = cell
        grid = frame.level(0)
        # A 3 x 3 cell neighbourhood, so a point on the very edge of the coverage is not missed.
        block = grid[max(row - 1, 0) : row + 2, max(col - 1, 0) : col + 2]
        covered = bool((~np.isnan(block)).any())
    return web.json_response(
        {"covered": covered, "frame": ids[-1]},
        headers={"Cache-Control": "public, max-age=300"},
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
        "forecast_base": None if state.nowcast is None else _iso(state.nowcast.base_id),
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


def _tile_request(request: web.Request) -> tuple[Palette, int, int, int, int]:
    """Validated (palette, z, x, y, size) of a tile request."""
    state = request.app[STATE]
    z, x, y = _int(request, "z"), _int(request, "x"), _int(request, "y")
    palette = PALETTES.get(request.match_info["palette"])
    if palette is None:
        raise web.HTTPNotFound(text="unknown palette")
    if not 0 <= z <= state.settings.max_zoom or not (0 <= x < 2**z and 0 <= y < 2**z):
        raise web.HTTPNotFound(text="tile outside the served range")
    size_text = request.query.get("size", "256")
    if size_text not in {str(s) for s in TILE_SIZES}:
        raise web.HTTPBadRequest(text=f"size must be one of {TILE_SIZES}")
    return palette, z, x, y, int(size_text)


async def forecast_tile(request: web.Request) -> web.Response:
    state = request.app[STATE]
    base = _int(request, "base")
    lead = _int(request, "lead")
    palette, z, x, y, size = _tile_request(request)
    key = ("forecast", base, lead, palette.name, size, z, x, y)
    png = state.cache.get(key)
    if png is None:
        forecast = await current_nowcast(state)
        # An older base means the phone has an old frame list; it will ask for the new one.
        if forecast is None or forecast.base_id != base or lead not in forecast.frames:
            raise web.HTTPNotFound(text="no such forecast (it may have been replaced)")
        async with state.renders:
            png = await asyncio.to_thread(
                render_tile, forecast.frames[lead], palette, z, x, y, size
            )
        state.cache.put(key, png)
    # A forecast for a given base and lead never changes; a newer forecast has a different base.
    return web.Response(body=png, content_type="image/png", headers={"Cache-Control": IMMUTABLE})


async def tile(request: web.Request) -> web.Response:
    state = request.app[STATE]
    frame_id = _int(request, "frame")
    palette, z, x, y, size = _tile_request(request)

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
            web.get("/coverage", coverage),
            web.get("/palettes", palettes),
            web.get("/attribution", attribution),
            web.get("/health", health),
            web.get(r"/tiles/{frame}/{palette}/{z}/{x}/{y}.png", tile),
            web.get(r"/forecast/{base}/{lead}/{palette}/{z}/{x}/{y}.png", forecast_tile),
        ]
    )
    app.cleanup_ctx.append(_ingest_lifecycle)
    return app
