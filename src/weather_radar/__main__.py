"""`python -m weather_radar`: run the ingest loop and the HTTP server together."""

from __future__ import annotations

import logging

from aiohttp import web

from weather_radar.api import TemperatureFeature, create_app
from weather_radar.config import Settings
from weather_radar.dpc import DpcClient, make_session
from weather_radar.frames import FrameStore
from weather_radar.icon import IconClient
from weather_radar.ingest import Ingestor
from weather_radar.temperature import TemperatureIngestor, TemperatureStore
from weather_radar.terrain import Terrain


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    store = FrameStore(settings.data_dir / "frames")

    async def build() -> web.Application:
        # The session must be created inside the running event loop.
        session = make_session()
        ingestor = Ingestor(DpcClient(session), store, settings.history, settings.poll_seconds)
        temperature = None
        if settings.temperature:
            temperature_store = TemperatureStore(settings.data_dir / "temperature")
            temperature = TemperatureFeature(
                store=temperature_store,
                terrain=Terrain(session, settings.data_dir / "terrain"),
                ingestor=TemperatureIngestor(
                    IconClient(session),
                    temperature_store,
                    settings.temperature_past,
                    settings.temperature_future,
                    settings.temperature_poll_seconds,
                ),
            )
        app = create_app(settings, store, ingestor, temperature=temperature)

        async def close_session(_: web.Application) -> None:
            await session.close()

        app.on_cleanup.append(close_session)
        return app

    logging.getLogger(__name__).info(
        "starting on %s:%d, keeping %s of frames in %s",
        settings.host,
        settings.port,
        settings.history,
        settings.data_dir,
    )
    web.run_app(build(), host=settings.host, port=settings.port, access_log=None)


if __name__ == "__main__":
    main()
