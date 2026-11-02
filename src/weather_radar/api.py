"""The HTTP API the phone talks to.

GET /frames                                  frame ids (epoch seconds) and times, oldest first
GET /tiles/{frame}/{palette}/{z}/{x}/{y}.png a map tile (?size=256 or 512)
GET /forecast/{base}/{lead}/{palette}/{z}/{x}/{y}.png
                                             a tile of the short forecast made from frame {base},
                                             {lead} minutes ahead (listed in /frames as "forecast")
GET /coverage?lat=..&lon=..                  whether the newest frame has data at a point
GET /model                                   model rain frames (ICON-D2/ICON-EU): the next hours
GET /model/{version}/{valid}/{palette}/{z}/{x}/{y}.png
                                             a tile of the rain in the hour ending at {valid}
GET /temperature                             hourly temperature maps (ICON-D2/ICON-EU,
                                             height-corrected): hours, palette and legend
GET /temperature/{version}/{valid}/{palette}/{z}/{x}/{y}.png
                                             a temperature tile (256 px)
GET /temperature/point?lat=..&lon=..&valid=..
                                             the temperature at a point for one hour
GET /palettes                                colour palettes with their legend bands; a palette
                                             name ending in "-snow" draws snow and mixed
                                             precipitation in their own colours (if the server
                                             has the model snow line)
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

from weather_radar import geo, precip
from weather_radar import nowcast as nowcasting
from weather_radar import temperature as temp
from weather_radar.config import Settings
from weather_radar.dpc import FRAME_STEP_SECONDS
from weather_radar.frames import FrameStore
from weather_radar.ingest import Ingestor
from weather_radar.modeldata import Grid, Hour, HourlyStore, ModelIngestor, edge_weight
from weather_radar.nowcast import MOTION_MINUTES, Nowcast
from weather_radar.palettes import (
    DEFAULT_PALETTE,
    MIXED,
    PALETTES,
    PHASE_SUFFIX,
    SNOW,
    Palette,
    Scale,
    resolve,
)
from weather_radar.render import TILE_SIZES, empty_tile, encode, render_tile
from weather_radar.temperature import (
    DEFAULT_TEMPERATURE_PALETTE,
    TEMPERATURE_PALETTES,
    ModelHour,
    tile_lonlat,
)
from weather_radar.terrain import ATTRIBUTION as TERRAIN_ATTRIBUTION
from weather_radar.terrain import TILE_SIZE as TERRAIN_TILE_SIZE
from weather_radar.terrain import Terrain, TerrainError
from weather_radar.tilecache import TileCache

log = logging.getLogger(__name__)

ATTRIBUTION = (
    "Radar data: Radar-DPC, Presidenza del Consiglio dei Ministri - "
    "Dipartimento della Protezione Civile (CC BY-SA 4.0)"
)
TEMPERATURE_ATTRIBUTION = (
    "Temperature: Deutscher Wetterdienst, ICON-D2 and ICON-EU models (CC BY 4.0), corrected for "
    "ground height. " + TERRAIN_ATTRIBUTION
)
MODEL_ATTRIBUTION = "Model rain: Deutscher Wetterdienst, ICON-D2 and ICON-EU models (CC BY 4.0)"
#: A radar frame uses the snow line of the closest model hour, if one is at most this far away.
SNOWLINE_REACH_SECONDS = 90 * 60
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


@dataclass
class ModelFeature:
    """Data from DWD's models; each part is None when switched off, the whole when all are."""

    terrain: Terrain
    t2m_eu: HourlyStore | None = None
    t2m_d2: HourlyStore | None = None
    rain_eu: HourlyStore | None = None
    rain_d2: HourlyStore | None = None
    snowline: HourlyStore | None = None
    ingestor: ModelIngestor | None = None
    #: How many hours of model rain /model lists.
    rain_hours: int = 12
    _d2_weight: Grid | None = dataclasses.field(default=None, repr=False)

    def d2_weight(self, d2_grid: Any) -> Grid:
        """ICON-D2's fade weight; its area is the same in every field, so it is computed once."""
        if self._d2_weight is None:
            self._d2_weight = edge_weight(d2_grid)
        return self._d2_weight

    def snowline_near(self, when: int) -> Any | None:
        """The snow-line grid of the stored hour closest to [when], if close enough."""
        if self.snowline is None:
            return None
        hours = self.snowline.hours()
        if not hours:
            return None
        best = min(hours, key=lambda h: abs(h.valid - when))
        if abs(best.valid - when) > SNOWLINE_REACH_SECONDS:
            return None
        try:
            return self.snowline.load(best)
        except KeyError:
            return None


def version_of(eu_run: int, d2_run: int | None) -> str:
    """The model runs an hour's tiles are made from, as written in their URLs."""
    return str(eu_run) if d2_run is None else f"{eu_run}-{d2_run}"


def parse_version(text: str) -> tuple[int, int | None]:
    eu, _, d2 = text.partition("-")
    try:
        return int(eu), (int(d2) if d2 else None)
    except ValueError:
        raise web.HTTPNotFound(text="bad model version") from None


def combined_hours(eu: HourlyStore, d2: HourlyStore | None) -> list[tuple[Hour, int | None]]:
    """ICON-EU's hours, each with the ICON-D2 run stored for the same hour (if any)."""
    d2_runs = {} if d2 is None else {h.valid: h.run for h in d2.hours()}
    return [(h, d2_runs.get(h.valid)) for h in eu.hours()]


STATE = web.AppKey("state", AppState)
MODELS = web.AppKey("models", ModelFeature)
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
            # A third frame makes the motion steadier; the forecast works without it too.
            oldest_id = earlier - MOTION_MINUTES * 60
            oldest = (
                (await asyncio.to_thread(state.store.load, oldest_id)).level(0)
                if oldest_id in ids
                else None
            )
            state.nowcast = await asyncio.to_thread(
                nowcasting.compute, before.level(0), latest.level(0), newest, oldest
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


def _bands(thresholds: tuple[float, ...], colours: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {"from": t, "color": _hex(c), "alpha": c[3]}
        for t, c in zip(thresholds, colours, strict=True)
    ]


async def palettes(request: web.Request) -> web.Response:
    def describe(p: Palette) -> dict[str, Any]:
        return {
            "name": p.name,
            "label": p.label,
            "unit": "mm/h",
            "bands": _bands(p.thresholds, p.colours),
        }

    def scale(s: Scale) -> list[dict[str, Any]]:
        return _bands(s.thresholds, s.colours)

    models = request.app.get(MODELS)
    snow = models is not None and models.snowline is not None
    return web.json_response(
        {
            "default": DEFAULT_PALETTE,
            "palettes": [describe(p) for p in PALETTES.values()],
            # Only offered when the server can tell snow from rain.
            "snow_suffix": PHASE_SUFFIX if snow else None,
            "mixed": scale(MIXED) if snow else [],
            "snow": scale(SNOW) if snow else [],
        },
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
    models = request.app.get(MODELS)
    if models is not None:
        # Reported only: the radar's health does not depend on the model data.
        m: dict[str, Any] = {"terrain_tiles_fetched": models.terrain.fetched_total}
        jobs = {} if models.ingestor is None else models.ingestor.status
        stores = {
            "icon-eu t_2m": models.t2m_eu,
            "icon-d2 t_2m": models.t2m_d2,
            "icon-eu tot_prec": models.rain_eu,
            "icon-d2 tot_prec": models.rain_d2,
            "icon-eu snowlmt": models.snowline,
        }
        for name, store in stores.items():
            if store is None:
                continue
            hours = store.hours()
            entry: dict[str, Any] = {
                "hours": len(hours),
                "first": None if not hours else _iso(hours[0].valid),
                "last": None if not hours else _iso(hours[-1].valid),
            }
            job = jobs.get(name)
            if job is not None:
                entry["newest_run"] = None if job.newest_run is None else _iso(job.newest_run)
                entry["last_error"] = job.last_error
                entry["downloaded_total"] = job.downloaded_total
            m[name] = entry
        body["models"] = m
    return web.json_response(body, status=200 if status == "ok" else 503)


def _int(request: web.Request, name: str) -> int:
    try:
        return int(request.match_info[name])
    except ValueError:
        raise web.HTTPNotFound from None


@dataclass(frozen=True)
class TileRequest:
    palette: Palette
    #: Snow and mixed precipitation in their own colours ("-snow" palette names).
    phased: bool
    #: The palette name as asked for, part of cache keys.
    name: str
    z: int
    x: int
    y: int
    size: int


def _tile_request(request: web.Request) -> TileRequest:
    """A validated tile request."""
    state = request.app[STATE]
    z, x, y = _int(request, "z"), _int(request, "x"), _int(request, "y")
    name = request.match_info["palette"]
    resolved = resolve(name)
    if resolved is None:
        raise web.HTTPNotFound(text="unknown palette")
    if not 0 <= z <= state.settings.max_zoom or not (0 <= x < 2**z and 0 <= y < 2**z):
        raise web.HTTPNotFound(text="tile outside the served range")
    size_text = request.query.get("size", "256")
    if size_text not in {str(s) for s in TILE_SIZES}:
        raise web.HTTPBadRequest(text=f"size must be one of {TILE_SIZES}")
    return TileRequest(resolved[0], resolved[1], name, z, x, y, int(size_text))


async def _ground(models: ModelFeature, t: TileRequest) -> Any:
    """Real ground height of every pixel of the tile (terrain tiles are 256 px)."""
    heights = await models.terrain.heights(t.z, t.x, t.y)
    if t.size != TERRAIN_TILE_SIZE:
        factor = t.size // TERRAIN_TILE_SIZE
        heights = np.repeat(np.repeat(heights, factor, axis=0), factor, axis=1)
    return heights


async def _phase(request: web.Request, t: TileRequest, when: int) -> Any | None:
    """Rain/mixed/snow per pixel of a tile at time [when], or None if that cannot be told now."""
    models = request.app.get(MODELS)
    if models is None:
        return None
    snowline = models.snowline_near(when)
    if snowline is None:
        return None
    try:
        ground = await _ground(models, t)
    except TerrainError as e:
        log.warning("terrain unavailable, drawing rain only: %s", e)
        return None
    lat, lon = tile_lonlat(t.z, t.x, t.y, t.size)
    return await asyncio.to_thread(precip.phase_at, snowline, lat, lon, ground)


async def _radar_png(request: web.Request, frame: Any, t: TileRequest) -> tuple[bytes, bool]:
    """(PNG, whether it is final). Not final: snow was asked for but could not be told."""
    state = request.app[STATE]
    async with state.renders:
        png = await asyncio.to_thread(render_tile, frame, t.palette, t.z, t.x, t.y, t.size)
    if not t.phased or png == empty_tile(t.size):
        return png, True
    # Only tiles with rain need the snow line and the terrain.
    phase = await _phase(request, t, frame.id)
    if phase is None:
        return png, False
    async with state.renders:
        png = await asyncio.to_thread(render_tile, frame, t.palette, t.z, t.x, t.y, t.size, phase)
    return png, True


async def forecast_tile(request: web.Request) -> web.Response:
    state = request.app[STATE]
    base = _int(request, "base")
    lead = _int(request, "lead")
    t = _tile_request(request)
    key = ("forecast", base, lead, t.name, t.size, t.z, t.x, t.y)
    png = state.cache.get(key)
    final = True
    if png is None:
        forecast = await current_nowcast(state)
        # An older base means the phone has an old frame list; it will ask for the new one.
        if forecast is None or forecast.base_id != base or lead not in forecast.frames:
            raise web.HTTPNotFound(text="no such forecast (it may have been replaced)")
        png, final = await _radar_png(request, forecast.frames[lead], t)
        if final:
            state.cache.put(key, png)
    # A forecast for a given base and lead never changes; a newer forecast has a different base.
    return web.Response(
        body=png, content_type="image/png", headers={"Cache-Control": IMMUTABLE if final else SHORT}
    )


async def tile(request: web.Request) -> web.Response:
    state = request.app[STATE]
    frame_id = _int(request, "frame")
    t = _tile_request(request)

    key = (frame_id, t.name, t.size, t.z, t.x, t.y)
    png = state.cache.get(key)
    final = True
    if png is None:
        try:
            frame = await asyncio.to_thread(state.store.load, frame_id)
        except KeyError:
            raise web.HTTPNotFound(text="no such frame") from None
        png, final = await _radar_png(request, frame, t)
        if final:
            state.cache.put(key, png)

    ids = state.store.ids()
    is_newest = bool(ids) and frame_id == ids[-1]
    return web.Response(
        body=png,
        content_type="image/png",
        headers={"Cache-Control": SHORT if is_newest or not final else IMMUTABLE},
    )


def _models(request: web.Request) -> ModelFeature:
    models = request.app.get(MODELS)
    if models is None:
        raise web.HTTPNotFound(text="model data is switched off on this server")
    return models


def _temperature_stores(request: web.Request) -> tuple[ModelFeature, HourlyStore]:
    models = _models(request)
    if models.t2m_eu is None:
        raise web.HTTPNotFound(text="temperature maps are switched off on this server")
    return models, models.t2m_eu


def _model_hour(
    models: ModelFeature, eu: HourlyStore, valid: int, d2_run: int | None, eu_run: int
) -> ModelHour:
    """The grids of one temperature hour; KeyError if a run was replaced."""
    grid = eu.load(Hour(valid, eu_run))
    if d2_run is None or models.t2m_d2 is None:
        return ModelHour(grid, eu.model_ground())
    d2 = models.t2m_d2.load(Hour(valid, d2_run))
    return ModelHour(
        grid, eu.model_ground(), d2, models.t2m_d2.model_ground(), models.d2_weight(d2)
    )


async def temperature_hours(request: web.Request) -> web.Response:
    models, eu = _temperature_stores(request)
    state = request.app[STATE]
    hours = combined_hours(eu, models.t2m_d2) if eu.has_ground() else []
    if models.t2m_d2 is not None and not models.t2m_d2.has_ground():
        hours = [(h, None) for h, _ in hours]
    palette = TEMPERATURE_PALETTES[DEFAULT_TEMPERATURE_PALETTE]
    return web.json_response(
        {
            "hours": [
                {
                    "id": h.valid,
                    "time": _iso(h.valid),
                    "run": h.run,
                    "version": version_of(h.run, d2_run),
                }
                for h, d2_run in hours
            ],
            "now": int(state.clock()) // temp.HOUR * temp.HOUR,
            "max_zoom": state.settings.max_zoom,
            "tile_size": TERRAIN_TILE_SIZE,
            "palette": palette.name,
            "unit": "°C",
            "bands": [
                {"from": palette.band_start(i), "color": _hex((*c, 255))}
                for i, c in enumerate(palette.band_colours())
            ],
            "attribution": TEMPERATURE_ATTRIBUTION,
        },
        headers={"Cache-Control": "public, max-age=60"},
    )


def _map_tile(request: web.Request) -> tuple[int, int, int]:
    state = request.app[STATE]
    z, x, y = _int(request, "z"), _int(request, "x"), _int(request, "y")
    if not 0 <= z <= state.settings.max_zoom or not (0 <= x < 2**z and 0 <= y < 2**z):
        raise web.HTTPNotFound(text="tile outside the served range")
    return z, x, y


async def temperature_tile(request: web.Request) -> web.Response:
    models, eu = _temperature_stores(request)
    state = request.app[STATE]
    valid = _int(request, "valid")
    eu_run, d2_run = parse_version(request.match_info["version"])
    palette = TEMPERATURE_PALETTES.get(request.match_info["palette"])
    if palette is None:
        raise web.HTTPNotFound(text="unknown palette")
    z, x, y = _map_tile(request)
    key = ("temperature", valid, eu_run, d2_run, palette.name, z, x, y)
    png = state.cache.get(key)
    if png is None:
        try:
            hour = await asyncio.to_thread(_model_hour, models, eu, valid, d2_run, eu_run)
        except KeyError:
            # Usually a newer model run replaced this hour: the phone will ask for the new list.
            raise web.HTTPNotFound(text="no such hour (it may have been replaced)") from None
        if not temp.tile_overlaps_model(z, x, y):
            png = temp.empty_tile()
        else:
            try:
                ground = await models.terrain.heights(z, x, y)
            except TerrainError as e:
                # Not cached by anyone: a tile without the height correction would be kept for good.
                log.warning("terrain unavailable: %s", e)
                raise web.HTTPServiceUnavailable(text="terrain unavailable, retry later") from e
            async with state.renders:
                png = await asyncio.to_thread(temp.render_tile, hour, ground, palette, z, x, y)
        state.cache.put(key, png)
    # One run's map of one hour never changes; a newer run has a different address.
    return web.Response(body=png, content_type="image/png", headers={"Cache-Control": IMMUTABLE})


async def temperature_point(request: web.Request) -> web.Response:
    models, eu = _temperature_stores(request)
    try:
        lat = float(request.query["lat"])
        lon = float(request.query["lon"])
        valid = int(request.query["valid"])
    except (KeyError, ValueError):
        raise web.HTTPBadRequest(text="lat, lon and valid are required numbers") from None
    if not (-85 <= lat <= 85 and -180 <= lon <= 180):
        raise web.HTTPBadRequest(text="lat or lon out of range")
    found = next(
        (pair for pair in combined_hours(eu, models.t2m_d2) if pair[0].valid == valid), None
    )
    if found is None:
        raise web.HTTPNotFound(text="no such hour")
    hour, d2_run = found
    if models.t2m_d2 is not None and not models.t2m_d2.has_ground():
        d2_run = None
    try:
        grids = await asyncio.to_thread(_model_hour, models, eu, valid, d2_run, hour.run)
        ground = await models.terrain.height_at(lat, lon)
    except KeyError:
        raise web.HTTPNotFound(text="no such hour") from None
    except TerrainError as e:
        raise web.HTTPServiceUnavailable(text="terrain data unavailable, retry later") from e
    value = temp.point_value(grids, lat, lon, ground)
    return web.json_response(
        {
            "temperature": None if value is None else round(value, 1),
            "valid": hour.valid,
            "run": hour.run,
            "ground_m": round(max(ground, 0.0)),
        },
        headers={"Cache-Control": "public, max-age=300"},
    )


def _rain_stores(request: web.Request) -> tuple[ModelFeature, HourlyStore]:
    models = _models(request)
    if models.rain_eu is None:
        raise web.HTTPNotFound(text="model rain is switched off on this server")
    return models, models.rain_eu


async def model_hours(request: web.Request) -> web.Response:
    """The model rain frames of the next [rain_hours] hours (each: the hour ending at its time)."""
    models, eu = _rain_stores(request)
    state = request.app[STATE]
    now = int(state.clock())
    last = now + models.rain_hours * temp.HOUR
    hours = [
        (h, d2_run) for h, d2_run in combined_hours(eu, models.rain_d2) if now < h.valid <= last
    ]
    return web.json_response(
        {
            "hours": [
                {"id": h.valid, "time": _iso(h.valid), "version": version_of(h.run, d2_run)}
                for h, d2_run in hours
            ],
            "step_seconds": temp.HOUR,
            "max_zoom": state.settings.max_zoom,
            "tile_size": TERRAIN_TILE_SIZE,
            "attribution": MODEL_ATTRIBUTION,
        },
        headers={"Cache-Control": "public, max-age=60"},
    )


def _rain_hour(
    models: ModelFeature, eu: HourlyStore, valid: int, eu_run: int, d2_run: int | None
) -> precip.RainHour:
    grid = eu.load(Hour(valid, eu_run))
    if d2_run is None or models.rain_d2 is None:
        return precip.RainHour(grid)
    d2 = models.rain_d2.load(Hour(valid, d2_run))
    return precip.RainHour(grid, d2, models.d2_weight(d2))


async def model_tile(request: web.Request) -> web.Response:
    models, eu = _rain_stores(request)
    state = request.app[STATE]
    valid = _int(request, "valid")
    eu_run, d2_run = parse_version(request.match_info["version"])
    resolved = resolve(request.match_info["palette"])
    if resolved is None:
        raise web.HTTPNotFound(text="unknown palette")
    palette, phased = resolved
    z, x, y = _map_tile(request)
    key = ("model", valid, eu_run, d2_run, request.match_info["palette"], z, x, y)
    png = state.cache.get(key)
    final = True
    if png is None:
        try:
            hour = await asyncio.to_thread(_rain_hour, models, eu, valid, eu_run, d2_run)
        except KeyError:
            raise web.HTTPNotFound(text="no such hour (it may have been replaced)") from None
        size = TERRAIN_TILE_SIZE
        if not temp.tile_overlaps_model(z, x, y):
            png = empty_tile(size)
        else:
            lat, lon = tile_lonlat(z, x, y, size)
            async with state.renders:
                rain = await asyncio.to_thread(precip.rain_at, hour, lat, lon)
                png = await asyncio.to_thread(encode, rain, palette)
            if phased and png != empty_tile(size):
                # The amount fell during the hour before [valid]: use the snow line of its middle.
                phase = await _phase(
                    request,
                    TileRequest(palette, True, request.match_info["palette"], z, x, y, size),
                    valid - temp.HOUR // 2,
                )
                if phase is None:
                    final = False
                else:
                    async with state.renders:
                        png = await asyncio.to_thread(encode, rain, palette, phase)
        if final:
            state.cache.put(key, png)
    return web.Response(
        body=png, content_type="image/png", headers={"Cache-Control": IMMUTABLE if final else SHORT}
    )


async def _ingest_lifecycle(app: web.Application) -> AsyncIterator[None]:
    ingestor = app[STATE].ingestor
    models = app.get(MODELS)
    tasks = [asyncio.create_task(ingestor.run())] if ingestor is not None else []
    if models is not None and models.ingestor is not None:
        tasks.append(asyncio.create_task(models.ingestor.run()))
    yield
    for task in tasks:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def create_app(
    settings: Settings,
    store: FrameStore,
    ingestor: Ingestor | None,
    clock: Callable[[], float] = time.time,
    models: ModelFeature | None = None,
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
            web.get("/model", model_hours),
            web.get(r"/model/{version}/{valid}/{palette}/{z}/{x}/{y}.png", model_tile),
            web.get("/temperature", temperature_hours),
            web.get("/temperature/point", temperature_point),
            web.get(r"/temperature/{version}/{valid}/{palette}/{z}/{x}/{y}.png", temperature_tile),
        ]
    )
    if models is not None:
        app[MODELS] = models
    app.cleanup_ctx.append(_ingest_lifecycle)
    return app
