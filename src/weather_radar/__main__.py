"""`python -m weather_radar`: run the ingest loops and the HTTP server together."""

from __future__ import annotations

import logging
from datetime import timedelta

from aiohttp import web

from weather_radar import precip
from weather_radar.api import ModelFeature, create_app
from weather_radar.config import Settings
from weather_radar.dpc import DpcClient, make_session
from weather_radar.frames import FrameStore
from weather_radar.icon import ICON_D2, ICON_EU, IconClient
from weather_radar.ingest import Ingestor
from weather_radar.modeldata import FieldJob, Grid, HourlyStore, ModelIngestor
from weather_radar.terrain import Terrain

KELVIN = 273.15


def to_celsius(values: Grid, _: object) -> Grid:
    return values - KELVIN


def model_feature(settings: Settings, client: IconClient, terrain: Terrain) -> ModelFeature | None:
    """The model data that is switched on, with one download job per model and field."""
    root = settings.data_dir
    feature = ModelFeature(terrain=terrain, rain_hours=settings.model_rain_hours)
    jobs: list[FieldJob] = []
    models = [ICON_EU, ICON_D2] if settings.icon_d2 else [ICON_EU]
    if settings.temperature:
        for model in models:
            # ICON-EU's temperature keeps its old directory, so an update does not fetch it again.
            directory = root / "temperature" if model is ICON_EU else root / model.name / "t_2m"
            store = HourlyStore(directory)
            jobs.append(
                FieldJob(
                    model,
                    "t_2m",
                    store,
                    settings.temperature_past,
                    settings.temperature_future,
                    needs_ground=True,
                    convert=to_celsius,
                )
            )
            if model is ICON_EU:
                feature.t2m_eu = store
            else:
                feature.t2m_d2 = store
    if settings.model_rain_hours > 0:
        future = timedelta(hours=settings.model_rain_hours + 1)
        for model in models:
            store = HourlyStore(root / model.name / "tot_prec")
            jobs.append(
                FieldJob(model, "tot_prec", store, timedelta(hours=1), future, accumulated=True)
            )
            if model is ICON_EU:
                feature.rain_eu = store
            else:
                feature.rain_d2 = store
    if settings.snow:
        # Radar history and nowcast behind, model rain ahead.
        past = settings.history + timedelta(hours=1)
        future = timedelta(hours=max(settings.model_rain_hours, 1) + 1)
        feature.snowline = HourlyStore(root / ICON_EU.name / "snowlmt")
        jobs.append(
            FieldJob(
                ICON_EU,
                "snowlmt",
                feature.snowline,
                past,
                future,
                needs_ground=True,
                convert=precip.fill_snowline,
            )
        )
    if not jobs:
        return None
    feature.ingestor = ModelIngestor(client, jobs, settings.temperature_poll_seconds)
    return feature


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
        models = model_feature(
            settings, IconClient(session), Terrain(session, settings.data_dir / "terrain")
        )
        app = create_app(settings, store, ingestor, models=models)

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
