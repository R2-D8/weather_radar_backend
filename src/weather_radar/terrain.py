"""Ground height for map tiles, from the open "Terrain Tiles" dataset on AWS (Mapzen/Tilezen).

The tiles use the same Web Mercator scheme as our map tiles, so the terrain tile z/x/y covers
exactly the same pixels as the temperature tile z/x/y. They are fetched only when a tile that
needs them is first drawn, then kept on disk for good (the ground does not change), so only the
areas someone actually looks at are ever downloaded (about 50 to 100 KB per tile).

Format "terrarium": 256 px RGB PNG, height in metres = R * 256 + G + B / 256 - 32768. Sea floor is
negative. No key needed. Sources include SRTM, GMTED2010, ETOPO1 and EU-DEM (see ATTRIBUTION).
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
from collections import OrderedDict
from pathlib import Path

import aiohttp
import numpy as np
from numpy.typing import NDArray
from PIL import Image

log = logging.getLogger(__name__)

BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"
TILE_SIZE = 256
#: The dataset goes to zoom 15; the point lookup uses this zoom (about 100 m per pixel in Italy).
POINT_ZOOM = 11
MAX_TILE_BYTES = 2 * 1024 * 1024
ATTRIBUTION = (
    "Terrain: Mapzen Terrain Tiles (SRTM and GMTED2010 courtesy of the U.S. Geological Survey, "
    "ETOPO1 by NOAA, EU-DEM produced using Copernicus data and information funded by the "
    "European Union, and others)"
)

Heights = NDArray[np.float32]


class TerrainError(Exception):
    """A terrain tile could not be fetched or read."""


def decode(png: bytes) -> Heights:
    """Height in metres of every pixel of a terrarium tile."""
    try:
        with Image.open(io.BytesIO(png)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    except (OSError, ValueError) as e:
        raise TerrainError(f"not a readable PNG: {e}") from e
    if rgb.shape != (TILE_SIZE, TILE_SIZE, 3):
        raise TerrainError(f"unexpected terrain tile shape {rgb.shape}")
    heights = rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0
    return heights.astype(np.float32)


def point_pixel(lat: float, lon: float, z: int) -> tuple[int, int, int, int]:
    """(tile x, tile y, pixel column, pixel row) of a point at zoom [z]."""
    n = 2**z
    fx = (lon + 180.0) / 360.0 * n
    fy = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    fx = min(max(fx, 0.0), n - 1e-9)
    fy = min(max(fy, 0.0), n - 1e-9)
    tx, ty = int(fx), int(fy)
    return tx, ty, int((fx - tx) * TILE_SIZE), int((fy - ty) * TILE_SIZE)


class Terrain:
    """Terrain tiles from disk, or from the internet the first time; recent ones kept in memory."""

    def __init__(
        self,
        session: aiohttp.ClientSession | None,
        directory: Path,
        base_url: str = BASE_URL,
        memory_tiles: int = 64,
    ) -> None:
        self._session = session
        self._dir = directory
        self._base = base_url.rstrip("/")
        self._memory: OrderedDict[tuple[int, int, int], Heights] = OrderedDict()
        self._memory_tiles = memory_tiles
        self._pending: dict[tuple[int, int, int], asyncio.Future[Heights]] = {}
        self.fetched_total = 0

    def _path(self, z: int, x: int, y: int) -> Path:
        return self._dir / str(z) / str(x) / f"{y}.png"

    async def heights(self, z: int, x: int, y: int) -> Heights:
        """Ground height of every pixel of tile z/x/y (sea floor negative)."""
        key = (z, x, y)
        cached = self._memory.get(key)
        if cached is not None:
            self._memory.move_to_end(key)
            return cached
        # Several map tiles rendered at once may need the same terrain tile: fetch it only once.
        pending = self._pending.get(key)
        if pending is not None:
            return await asyncio.shield(pending)
        future: asyncio.Future[Heights] = asyncio.get_running_loop().create_future()
        self._pending[key] = future
        try:
            heights = await self._load(z, x, y)
        except BaseException as e:
            future.set_exception(e)
            # Nobody else may be waiting; mark the exception as seen so asyncio does not log it.
            future.exception()
            raise
        finally:
            self._pending.pop(key, None)
        future.set_result(heights)
        self._memory[key] = heights
        while len(self._memory) > self._memory_tiles:
            self._memory.popitem(last=False)
        return heights

    async def height_at(self, lat: float, lon: float) -> float:
        tx, ty, col, row = point_pixel(lat, lon, POINT_ZOOM)
        return float((await self.heights(POINT_ZOOM, tx, ty))[row, col])

    async def _load(self, z: int, x: int, y: int) -> Heights:
        path = self._path(z, x, y)
        try:
            png = await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError:
            png = await self._fetch(z, x, y)
            await asyncio.to_thread(_write_atomically, path, png)
        return await asyncio.to_thread(decode, png)

    async def _fetch(self, z: int, x: int, y: int) -> bytes:
        if self._session is None:
            raise TerrainError("no network session for terrain tiles")
        url = f"{self._base}/{z}/{x}/{y}.png"
        try:
            async with self._session.get(url) as response:
                if response.status != 200:
                    raise TerrainError(f"terrain tile {z}/{x}/{y}: HTTP {response.status}")
                # read() returns only what has arrived so far, so read chunk by chunk to the end.
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.content.iter_chunked(64 * 1024):
                    total += len(chunk)
                    if total > MAX_TILE_BYTES:
                        raise TerrainError(f"terrain tile {z}/{x}/{y} is too large")
                    chunks.append(chunk)
        except (aiohttp.ClientError, TimeoutError) as e:
            raise TerrainError(f"terrain tile {z}/{x}/{y}: {e}") from e
        body = b"".join(chunks)
        decode(body)  # reject anything that is not a terrain tile before keeping it
        self.fetched_total += 1
        return body


def _write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
