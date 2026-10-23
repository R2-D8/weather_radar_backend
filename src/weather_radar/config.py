"""Settings, read from environment variables so Docker Compose can set them."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    #: Where downloaded frames are stored (a volume in Docker).
    data_dir: Path
    #: How long frames are kept. The app shows less (2 h by default); the rest allows extending it.
    history: timedelta = timedelta(hours=6)
    host: str = "127.0.0.1"
    port: int = 8080
    #: How often to re-check while a new frame is overdue. Normally the next poll is timed for
    #: when DPC should have published the next frame (every 5 minutes, a few minutes late).
    poll_seconds: float = 30.0
    #: Memory for rendered tiles (PNG bytes), in MiB.
    tile_cache_mib: int = 32
    #: Highest zoom served. The DPC grid is 1 km, about zoom 8; the app overzooms beyond.
    max_zoom: int = 8

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        return cls(
            data_dir=Path(e.get("RADAR_DATA_DIR", "data")),
            history=timedelta(hours=float(e.get("RADAR_HISTORY_HOURS", "6"))),
            host=e.get("RADAR_HOST", "127.0.0.1"),
            port=int(e.get("RADAR_PORT", "8080")),
            poll_seconds=float(e.get("RADAR_POLL_SECONDS", "30")),
            tile_cache_mib=int(e.get("RADAR_TILE_CACHE_MIB", "32")),
            max_zoom=int(e.get("RADAR_MAX_ZOOM", "8")),
        )
