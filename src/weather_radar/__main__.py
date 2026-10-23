"""`python -m weather_radar`: run the ingest loop and the HTTP server together."""

from __future__ import annotations

import logging

from aiohttp import web

from weather_radar.api import create_app
from weather_radar.config import Settings
from weather_radar.dpc import DpcClient, make_session
from weather_radar.frames import FrameStore
from weather_radar.ingest import Ingestor


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
        app = create_app(settings, store, ingestor)

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
