"""Client for the Italian Civil Protection (DPC) radar API.

Two calls are used: `findLastProductByType` says which frame is newest, and `downloadProduct`
returns a short-lived pre-signed link to the GeoTIFF of one frame.
"""

from __future__ import annotations

from typing import Any

import aiohttp

BASE_URL = "https://radar-api.protezionecivile.it"
#: SRI is the surface rain rate in mm/h, one frame every 5 minutes.
PRODUCT = "SRI"
FRAME_STEP_SECONDS = 300
#: A frame is about 0.5 MB; anything far bigger is not what we asked for.
MAX_FRAME_BYTES = 20 * 1024 * 1024


class DpcError(Exception):
    """DPC could not be reached or answered with something unusable."""


class DpcClient:
    def __init__(self, session: aiohttp.ClientSession, base_url: str = BASE_URL) -> None:
        self._session = session
        self._base = base_url.rstrip("/")

    async def latest(self) -> int:
        """Id (epoch seconds) of the newest frame."""
        data = await self._json("GET", "/findLastProductByType", params={"type": PRODUCT})
        try:
            return int(data["lastProducts"][0]["time"]) // 1000
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise DpcError(f"unexpected answer from findLastProductByType: {data!r}") from e

    async def download(self, frame_id: int) -> bytes:
        """The GeoTIFF bytes of one frame (not validated here)."""
        data = await self._json(
            "POST",
            "/downloadProduct",
            json={"productType": PRODUCT, "productDate": frame_id * 1000},
        )
        url = data.get("url") if isinstance(data, dict) else None
        if not isinstance(url, str):
            raise DpcError(f"no download link for frame {frame_id}: {data!r}")
        # The link must be https (tests use a plain-http local server and a plain-http base URL).
        if not url.startswith("https://") and not self._base.startswith("http://"):
            raise DpcError("refusing to download from a non-https link")
        try:
            async with self._session.get(url) as response:
                if response.status != 200:
                    raise DpcError(f"download of frame {frame_id} failed: HTTP {response.status}")
                # read() returns only what has arrived so far, so read chunk by chunk to the end.
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.content.iter_chunked(64 * 1024):
                    total += len(chunk)
                    if total > MAX_FRAME_BYTES:
                        raise DpcError(f"frame {frame_id} is larger than {MAX_FRAME_BYTES} bytes")
                    chunks.append(chunk)
        except (aiohttp.ClientError, TimeoutError) as e:
            raise DpcError(f"download of frame {frame_id} failed: {e}") from e
        return b"".join(chunks)

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            async with self._session.request(method, self._base + path, **kwargs) as response:
                if response.status != 200:
                    raise DpcError(f"{method} {path} answered HTTP {response.status}")
                return await response.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError, ValueError) as e:
            raise DpcError(f"{method} {path} failed: {e}") from e


def make_session() -> aiohttp.ClientSession:
    """A session with sensible time limits for talking to DPC."""
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60, connect=15),
        headers={"User-Agent": "weather-radar (personal, non-commercial project)"},
    )
