"""Hourly fields of DWD's ICON models: downloading them, keeping them on disk and sampling them.

Each (model, field) pair has its own [HourlyStore] directory with one grid per valid hour, taken
from the newest model run that covers that hour, from [past] hours ago to [future] hours ahead.
Past hours therefore come from the first hours of recent runs, which are close to what happened.

Fields used: 2 m temperature (`t_2m`), precipitation (`tot_prec`, which DWD sums from the run
start; we keep the amount of each hour) and the snow line (`snowlmt`). See [ModelIngestor].
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from weather_radar import icon
from weather_radar.icon import ICON_EU, IconClient, IconError, Model

log = logging.getLogger(__name__)

HOUR = 3600
Grid = NDArray[np.float32]

_FILE = re.compile(r"^(\d+)_(\d+)\.npy$")


# --- sampling ---------------------------------------------------------------------------------


def sample(
    grid: NDArray[np.floating],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    model: Model = ICON_EU,
) -> Grid:
    """Bilinear sample of a [model] grid at points; NaN outside the model area or next to NaN."""
    row = (lat - model.lat0) / model.step
    col = (lon - model.lon0) / model.step
    inside = (row >= 0) & (row <= model.rows - 1) & (col >= 0) & (col <= model.cols - 1)
    r0 = np.clip(np.floor(row).astype(np.int64), 0, model.rows - 2)
    c0 = np.clip(np.floor(col).astype(np.int64), 0, model.cols - 2)
    fr = np.clip(row - r0, 0.0, 1.0)
    fc = np.clip(col - c0, 0.0, 1.0)
    v00 = grid[r0, c0].astype(np.float64)
    v01 = grid[r0, c0 + 1].astype(np.float64)
    v10 = grid[r0 + 1, c0].astype(np.float64)
    v11 = grid[r0 + 1, c0 + 1].astype(np.float64)
    value = (v00 * (1 - fc) + v01 * fc) * (1 - fr) + (v10 * (1 - fc) + v11 * fc) * fr
    return np.where(inside, value, np.nan).astype(np.float32)


#: ICON-D2 fades into ICON-EU over this many of its cells (30 km) inside the edge of its area, so
#: there is no visible seam where one model ends.
BLEND_CELLS = 15.0


def edge_weight(d2_grid: NDArray[np.floating]) -> Grid:
    """Per ICON-D2 cell: 0 outside its area, rising to 1 at [BLEND_CELLS] inside the edge."""
    valid = np.zeros((d2_grid.shape[0] + 2, d2_grid.shape[1] + 2), dtype=np.uint8)
    # Pad with "outside", so the rectangular border of the grid counts as an edge too.
    valid[1:-1, 1:-1] = ~np.isnan(np.asarray(d2_grid, dtype=np.float32))
    distance = np.asarray(cv2.distanceTransform(valid, cv2.DIST_L2, 5), dtype=np.float32)
    weight: Grid = np.clip(distance[1:-1, 1:-1] / BLEND_CELLS, 0.0, 1.0).astype(np.float32)
    return weight


def blend(eu: Grid, d2: Grid | None, weight: Grid | None) -> Grid:
    """ICON-D2 where it has data, faded into ICON-EU towards the edge of its area."""
    if d2 is None or weight is None:
        return eu
    w = np.where(np.isnan(d2) | np.isnan(weight), 0.0, weight)
    mixed = np.nan_to_num(d2) * w + eu * (1.0 - w)
    return np.where(w > 0, mixed, eu).astype(np.float32)


# --- storage ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hour:
    #: Valid time and model run start, both epoch seconds.
    valid: int
    run: int


class HourlyStore:
    """Files `{valid}_{run}.npy` (float16) and, if needed, `model_ground.npy` (float32 metres)."""

    def __init__(self, directory: Path, open_grids: int = 6) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._open: OrderedDict[Hour, NDArray[np.float16]] = OrderedDict()
        self._open_max = open_grids
        self._ground: NDArray[np.float32] | None = None

    @property
    def ground_path(self) -> Path:
        return self._dir / "model_ground.npy"

    def hours(self) -> list[Hour]:
        """Stored hours, oldest first (one per valid time: the newest run)."""
        best: dict[int, int] = {}
        for path in self._dir.iterdir():
            m = _FILE.match(path.name)
            if m:
                valid, run = int(m.group(1)), int(m.group(2))
                if run > best.get(valid, -1):
                    best[valid] = run
        return [Hour(v, best[v]) for v in sorted(best)]

    def run_for(self, valid: int) -> int | None:
        """The run stored for hour [valid], if any."""
        return next((h.run for h in self.hours() if h.valid == valid), None)

    def has_ground(self) -> bool:
        return self.ground_path.exists()

    def save_ground(self, heights: NDArray[np.float32]) -> None:
        _save_atomically(self.ground_path, heights.astype(np.float32))
        self._ground = None

    def model_ground(self) -> NDArray[np.float32]:
        if self._ground is None:
            self._ground = cast(NDArray[np.float32], np.load(self.ground_path, mmap_mode="r"))
        return self._ground

    def save(self, hour: Hour, values: NDArray[np.float32]) -> None:
        _save_atomically(self._dir / f"{hour.valid}_{hour.run}.npy", values.astype(np.float16))
        # Older runs for the same hour are no longer needed.
        for path in self._dir.glob(f"{hour.valid}_*.npy"):
            m = _FILE.match(path.name)
            if m and int(m.group(2)) < hour.run:
                path.unlink(missing_ok=True)

    def load(self, hour: Hour) -> NDArray[np.float16]:
        grid = self._open.get(hour)
        if grid is not None:
            self._open.move_to_end(hour)
            return grid
        path = self._dir / f"{hour.valid}_{hour.run}.npy"
        try:
            # Memory-mapped: a tile only reads the few pages of the grid it covers.
            grid = cast(NDArray[np.float16], np.load(path, mmap_mode="r"))
        except FileNotFoundError:
            raise KeyError(hour) from None
        self._open[hour] = grid
        while len(self._open) > self._open_max:
            self._open.popitem(last=False)
        return grid

    def prune(self, oldest_valid: int) -> int:
        removed = 0
        for path in self._dir.iterdir():
            m = _FILE.match(path.name)
            if m and int(m.group(1)) < oldest_valid:
                path.unlink(missing_ok=True)
                removed += 1
        if removed:
            self._open = OrderedDict(
                (h, g) for h, g in self._open.items() if h.valid >= oldest_valid
            )
        return removed


def _save_atomically(path: Path, array: NDArray[np.generic]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    tmp.replace(path)


# --- planning ---------------------------------------------------------------------------------

#: Runs looked at: the newest four 6-hourly runs cover the past day, which is all DWD still has.
RUNS_CONSIDERED = 4
#: At most this many files per field and cycle, with a short pause between them, to be polite.
MAX_DOWNLOADS_PER_CYCLE = 60
DOWNLOAD_PAUSE_SECONDS = 0.5
#: A run is used until it is up to this old (the next one appears a few hours after its
#: predecessor started), so each run is read this much further ahead than the window needs.
RUN_AGE_MARGIN = timedelta(hours=12)


def max_step(future: timedelta, model: Model = ICON_EU) -> int:
    """The furthest forecast hour read from a run, so the window always reaches [future]."""
    return min(int((future + RUN_AGE_MARGIN).total_seconds()) // HOUR, model.max_hourly_step)


def plan_downloads(
    now: float,
    available_runs: list[int],
    stored: dict[int, int],
    past: timedelta,
    future: timedelta,
    model: Model = ICON_EU,
) -> list[Hour]:
    """Which (valid hour, run) files to fetch so every hour in the window has its newest run.

    The window runs from [past] before to [future] after the current hour. [available_runs] are
    complete runs on DWD's server; [stored] maps valid hour -> stored run. An hour uses the
    newest run that started at or before it and reaches it (see [max_step]). Newest hours first,
    so the forecast is complete soonest after a new run. Hours no run reaches are left out (for
    ICON-D2 beyond +48 h: ICON-EU covers them).
    """
    if not available_runs:
        return []
    this_hour = int(now) // HOUR * HOUR
    first = this_hour - int(past.total_seconds())
    last = this_hour + int(future.total_seconds())
    reach = max_step(future, model) * HOUR
    wanted: list[Hour] = []
    for valid in range(last, first - 1, -HOUR):
        candidates = [r for r in available_runs if r <= valid <= r + reach]
        if not candidates:
            continue
        best = max(candidates)
        if stored.get(valid, -1) < best:
            wanted.append(Hour(valid, best))
    return wanted


# --- ingest -----------------------------------------------------------------------------------

#: (values as decoded, the model's ground height or None) -> values to store.
Convert = Callable[[Grid, "NDArray[np.float32] | None"], Grid]


@dataclass(frozen=True)
class FieldJob:
    """One field of one model to keep in a store."""

    model: Model
    #: DWD's directory name of the field, e.g. "t_2m".
    field: str
    store: HourlyStore
    past: timedelta
    future: timedelta
    #: DWD sums this field from the run start (precipitation); the store gets each hour's amount.
    accumulated: bool = False
    #: Download the model's ground height into the store first (needed by [convert] or later).
    needs_ground: bool = False
    convert: Convert | None = None

    @property
    def name(self) -> str:
        return f"{self.model.name} {self.field}"


@dataclass
class JobStatus:
    last_success: float | None = None
    last_error: str | None = None
    downloaded_total: int = 0
    newest_run: int | None = None


@dataclass
class _Runs:
    #: Runs known to be complete (they never change once complete).
    complete: set[int] = field(default_factory=set)


class ModelIngestor:
    """Keeps every job's store filled from DWD: checks for new runs every [poll_seconds]."""

    def __init__(
        self,
        client: IconClient,
        jobs: list[FieldJob],
        poll_seconds: float = 600.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self.jobs = jobs
        self._poll = poll_seconds
        self._clock = clock
        self._sleep = sleep
        self._runs = {job.name: _Runs() for job in jobs}
        self.status = {job.name: JobStatus() for job in jobs}

    async def run(self) -> None:
        while True:
            await self.cycle()
            await self._sleep(self._poll)

    async def cycle(self) -> int:
        """One update of every job (a failing job does not stop the others); files stored."""
        total = 0
        for job in self.jobs:
            status = self.status[job.name]
            try:
                total += await self.cycle_job(job)
                status.last_success = self._clock()
                status.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as e:  # keep the loop alive whatever happens
                log.warning("%s update failed: %s", job.name, e)
                status.last_error = str(e)
        return total

    async def available_runs(self, job: FieldJob) -> list[int]:
        """Complete runs among the newest few: their last needed hour is on the server."""
        last_step = max_step(job.future, job.model)
        known = self._runs[job.name].complete
        runs = []
        for run in icon.latest_run_ids(self._clock(), RUNS_CONSIDERED + 1):
            if run in known or await self._client.exists(
                self._client.field_url(job.model, job.field, run, last_step)
            ):
                known.add(run)
                runs.append(run)
        known &= set(runs)
        return sorted(runs, reverse=True)[:RUNS_CONSIDERED]

    async def _fetch(self, job: FieldJob, run: int, step: int) -> Grid | None:
        data = await self._client.download(self._client.field_url(job.model, job.field, run, step))
        if data is None:
            return None
        return await asyncio.to_thread(icon.decode, data, job.model)

    async def cycle_job(self, job: FieldJob) -> int:
        """One update of one field; returns the number of hours stored."""
        runs = await self.available_runs(job)
        if not runs:
            return 0
        self.status[job.name].newest_run = runs[0]
        store = job.store
        if job.needs_ground and not store.has_ground():
            data = await self._client.download(self._client.hsurf_url(job.model, runs[0]))
            if data is None:
                raise IconError(f"{job.model.name} ground height file missing")
            store.save_ground(await asyncio.to_thread(icon.decode, data, job.model))
        ground = store.model_ground() if job.needs_ground else None
        stored = {h.valid: h.run for h in store.hours()}
        todo = plan_downloads(self._clock(), runs, stored, job.past, job.future, job.model)
        # Sums from the run start, by (run, step), kept while walking back through the steps.
        sums: dict[tuple[int, int], Grid] = {}
        done = 0
        for hour in todo[:MAX_DOWNLOADS_PER_CYCLE]:
            step = (hour.valid - hour.run) // HOUR
            if job.accumulated:
                values = await self._hourly_amount(job, hour.run, step, sums)
            else:
                values = await self._fetch(job, hour.run, step)
            if values is None:
                log.info("%s: run %d step %d not on the server, skipping", job.name, hour.run, step)
                continue
            if job.convert is not None:
                values = job.convert(values, ground)
            await asyncio.to_thread(store.save, hour, values)
            done += 1
            self.status[job.name].downloaded_total += 1
            await self._sleep(DOWNLOAD_PAUSE_SECONDS)
        oldest = int(self._clock()) // HOUR * HOUR - int(job.past.total_seconds())
        store.prune(oldest)
        if done:
            log.info("%s: %d hours updated (newest run %s)", job.name, done, icon.run_time(runs[0]))
        return done

    async def _hourly_amount(
        self, job: FieldJob, run: int, step: int, sums: dict[tuple[int, int], Grid]
    ) -> Grid | None:
        """The amount that fell in the hour ending at +[step]: sum(step) - sum(step - 1)."""

        async def total(s: int) -> Grid | None:
            if s == 0:
                return None  # nothing has fallen at the start of a run
            if (run, s) not in sums:
                fetched = await self._fetch(job, run, s)
                if fetched is None:
                    raise KeyError(s)
                sums[(run, s)] = fetched
            return sums[(run, s)]

        try:
            now = await total(step)
            before = await total(step - 1)
        except KeyError:
            return None
        assert now is not None
        # Only the newest two are ever needed again (todo walks back one step at a time).
        for key in [k for k in sums if k[0] == run and k[1] > step]:
            del sums[key]
        amount = now if before is None else now - before
        # Packing noise can make a sum shrink by a hair; rain is never negative.
        return np.maximum(amount, 0.0).astype(np.float32)
