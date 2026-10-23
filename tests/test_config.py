from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from weather_radar.config import Settings


def test_defaults() -> None:
    settings = Settings.from_env({})
    assert settings.data_dir == Path("data")
    assert settings.history == timedelta(hours=6)
    assert settings.host == "127.0.0.1"  # safe default: not reachable from outside
    assert settings.port == 8080
    assert settings.poll_seconds == 30
    assert settings.max_zoom == 8


def test_environment_overrides() -> None:
    settings = Settings.from_env(
        {
            "RADAR_DATA_DIR": "/data",
            "RADAR_HISTORY_HOURS": "3",
            "RADAR_HOST": "0.0.0.0",
            "RADAR_PORT": "9000",
            "RADAR_POLL_SECONDS": "30",
            "RADAR_TILE_CACHE_MIB": "8",
            "RADAR_MAX_ZOOM": "7",
        }
    )
    assert settings.data_dir == Path("/data")
    assert settings.history == timedelta(hours=3)
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.poll_seconds == 30
    assert settings.tile_cache_mib == 8
    assert settings.max_zoom == 7
