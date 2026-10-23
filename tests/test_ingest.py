from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from weather_radar.dpc import FRAME_STEP_SECONDS, DpcError
from weather_radar.frames import FrameStore
from weather_radar.ingest import (
    BACKFILL_PAUSE_SECONDS,
    DEFAULT_PUBLISH_LAG_SECONDS,
    DOWNLOAD_PAUSE_SECONDS,
    MAX_BACKOFF_SECONDS,
    MAX_DOWNLOADS_PER_POLL,
    MAX_IDLE_SECONDS,
    MAX_PUBLISH_LAG_SECONDS,
    RETRY_NEWEST_SECONDS,
    RETRY_OLDER_SECONDS,
    Ingestor,
    wanted_frames,
)

LATEST = 1_792_791_900  # a multiple of 300, like real DPC frame ids


class FakeClient:
    def __init__(self, frame_bytes: bytes, latest: int = LATEST) -> None:
        self.frame_bytes = frame_bytes
        self.newest = latest
        self.downloads: list[int] = []
        self.failing: set[int] = set()
        self.bad_payload: set[int] = set()
        self.latest_error: Exception | None = None

    async def latest(self) -> int:
        if self.latest_error is not None:
            raise self.latest_error
        return self.newest

    async def download(self, frame_id: int) -> bytes:
        self.downloads.append(frame_id)
        if frame_id in self.failing:
            raise DpcError("HTTP 404")
        return b"not a tiff" if frame_id in self.bad_payload else self.frame_bytes


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


async def no_sleep(_: float) -> None:
    return None


def make(
    tmp_path: Path, sample_bytes: bytes, history_minutes: int = 30, clock: Clock | None = None
) -> tuple[Ingestor, FakeClient, FrameStore]:
    client = FakeClient(sample_bytes)
    store = FrameStore(tmp_path)
    ingestor = Ingestor(
        client,  # type: ignore[arg-type]
        store,
        timedelta(minutes=history_minutes),
        poll_seconds=10,
        clock=clock or Clock(),
        sleep=no_sleep,
    )
    return ingestor, client, store


def test_wanted_frames_are_five_minutes_apart_oldest_first() -> None:
    ids = wanted_frames(LATEST, timedelta(minutes=30))
    assert len(ids) == 7
    assert ids[-1] == LATEST and ids[0] == LATEST - 30 * 60
    assert all(b - a == FRAME_STEP_SECONDS for a, b in zip(ids, ids[1:], strict=False))


async def test_first_sync_fetches_the_whole_window_newest_first(
    tmp_path: Path, sample_bytes: bytes
) -> None:
    ingestor, client, store = make(tmp_path, sample_bytes)

    assert await ingestor.sync_once() == 7

    assert store.ids() == wanted_frames(LATEST, timedelta(minutes=30))
    assert client.downloads == sorted(client.downloads, reverse=True), "newest frame comes first"


async def test_nothing_is_downloaded_twice(tmp_path: Path, sample_bytes: bytes) -> None:
    ingestor, client, _ = make(tmp_path, sample_bytes)
    await ingestor.sync_once()
    client.downloads.clear()

    assert await ingestor.sync_once() == 0
    assert client.downloads == []


async def test_a_new_frame_is_fetched_and_the_oldest_pruned(
    tmp_path: Path, sample_bytes: bytes
) -> None:
    ingestor, client, store = make(tmp_path, sample_bytes)
    await ingestor.sync_once()
    client.downloads.clear()
    client.newest = LATEST + FRAME_STEP_SECONDS

    assert await ingestor.sync_once() == 1

    assert client.downloads == [LATEST + FRAME_STEP_SECONDS]
    assert store.ids() == wanted_frames(LATEST + FRAME_STEP_SECONDS, timedelta(minutes=30))


async def test_big_backfill_is_spread_over_several_polls(
    tmp_path: Path, sample_bytes: bytes
) -> None:
    ingestor, _, store = make(tmp_path, sample_bytes, history_minutes=6 * 60)

    assert await ingestor.sync_once() == MAX_DOWNLOADS_PER_POLL
    assert store.ids()[-1] == LATEST, "the newest frames are the ones fetched first"
    assert await ingestor.sync_once() == MAX_DOWNLOADS_PER_POLL


async def test_failed_frame_is_retried_after_a_pause(tmp_path: Path, sample_bytes: bytes) -> None:
    clock = Clock()
    ingestor, client, store = make(tmp_path, sample_bytes, clock=clock)
    older = LATEST - 600
    client.failing.add(older)

    await ingestor.sync_once()
    assert not store.has(older)
    assert store.has(LATEST)
    client.downloads.clear()

    await ingestor.sync_once()
    assert older not in client.downloads, "not retried straight away"

    client.failing.clear()
    clock.now += RETRY_OLDER_SECONDS + 1
    await ingestor.sync_once()
    assert store.has(older)


async def test_newest_frame_is_retried_quickly(tmp_path: Path, sample_bytes: bytes) -> None:
    clock = Clock()
    ingestor, client, store = make(tmp_path, sample_bytes, clock=clock)
    client.failing.add(LATEST)  # DPC announced it but has not published the file yet
    await ingestor.sync_once()
    assert not store.has(LATEST)

    client.failing.clear()
    clock.now += RETRY_NEWEST_SECONDS + 1
    await ingestor.sync_once()
    assert store.has(LATEST)


async def test_garbage_download_is_not_stored(tmp_path: Path, sample_bytes: bytes) -> None:
    ingestor, client, store = make(tmp_path, sample_bytes)
    client.bad_payload.add(LATEST)

    await ingestor.sync_once()

    assert not store.has(LATEST)
    assert len(store.ids()) == 6


async def test_status_is_updated(tmp_path: Path, sample_bytes: bytes) -> None:
    clock = Clock()
    ingestor, _, _ = make(tmp_path, sample_bytes, clock=clock)
    assert ingestor.status.last_success is None
    await ingestor.sync_once()
    assert ingestor.status.last_success == clock.now
    assert ingestor.status.downloaded_total == 7


async def test_run_survives_errors_and_backs_off(tmp_path: Path, sample_bytes: bytes) -> None:
    ingestor, client, _ = make(tmp_path, sample_bytes)
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)
        if len(delays) >= 9:
            raise asyncio.CancelledError

    ingestor._sleep = record_sleep  # type: ignore[assignment]
    client.latest_error = DpcError("down")

    with pytest.raises(asyncio.CancelledError):
        await ingestor.run()

    assert delays[:3] == [20, 40, 80]
    assert max(delays) == MAX_BACKOFF_SECONDS
    assert ingestor.status.last_error == "down"


async def test_run_recovers_and_resets_the_pause(tmp_path: Path, sample_bytes: bytes) -> None:
    clock = Clock()
    clock.now = LATEST + 1000  # the next frame is overdue, so polls come every poll interval
    ingestor, client, store = make(tmp_path, sample_bytes, clock=clock)
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        if seconds == DOWNLOAD_PAUSE_SECONDS:
            return  # the short pause between two downloads, not a poll delay
        delays.append(seconds)
        if len(delays) == 2:
            client.latest_error = None  # DPC is back
        if len(delays) >= 4:
            raise asyncio.CancelledError

    ingestor._sleep = record_sleep  # type: ignore[assignment]
    client.latest_error = DpcError("down")

    with pytest.raises(asyncio.CancelledError):
        await ingestor.run()

    assert delays == [20, 40, 10, 10]
    assert store.has(LATEST)
    assert ingestor.status.last_error is None


async def test_run_survives_an_unexpected_exception(tmp_path: Path, sample_bytes: bytes) -> None:
    ingestor, client, _ = make(tmp_path, sample_bytes)
    calls = 0

    async def record_sleep(_: float) -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise asyncio.CancelledError

    ingestor._sleep = record_sleep  # type: ignore[assignment]
    client.latest_error = RuntimeError("boom")

    with pytest.raises(asyncio.CancelledError):
        await ingestor.run()

    assert calls == 2, "the loop kept going after the first failure"
    assert ingestor.status.last_error == "RuntimeError('boom')"


# --- timing of polls ---------------------------------------------------------------------------


async def synced(
    tmp_path: Path, sample_bytes: bytes, now: float
) -> tuple[Ingestor, FakeClient, Clock]:
    clock = Clock()
    clock.now = now
    ingestor, client, _ = make(tmp_path, sample_bytes, clock=clock)
    await ingestor.sync_once()
    return ingestor, client, clock


async def test_next_poll_waits_until_the_next_frame_is_due(
    tmp_path: Path, sample_bytes: bytes
) -> None:
    # The newest frame was seen 600 s after its time. The next one is due 300 s later plus the
    # assumed publication delay (540 s), minus a 15 s margin: LATEST + 825.
    ingestor, _, _ = await synced(tmp_path, sample_bytes, LATEST + 600)
    assert ingestor._next_delay() == pytest.approx(225)


async def test_next_poll_is_capped_when_the_frame_is_far_away(
    tmp_path: Path, sample_bytes: bytes
) -> None:
    ingestor, _, _ = await synced(tmp_path, sample_bytes, LATEST + 10)
    assert ingestor._next_delay() == MAX_IDLE_SECONDS


async def test_overdue_frame_is_checked_every_poll_interval(
    tmp_path: Path, sample_bytes: bytes
) -> None:
    ingestor, _, _ = await synced(tmp_path, sample_bytes, LATEST + 900)
    assert ingestor._next_delay() == 10  # poll_seconds in these tests


async def test_backfill_polls_soon(tmp_path: Path, sample_bytes: bytes) -> None:
    clock = Clock()
    clock.now = LATEST + 600
    ingestor, _, _ = make(tmp_path, sample_bytes, history_minutes=6 * 60, clock=clock)
    await ingestor.sync_once()
    assert ingestor._next_delay() == BACKFILL_PAUSE_SECONDS


async def test_publication_delay_is_learned_from_new_frames(
    tmp_path: Path, sample_bytes: bytes
) -> None:
    ingestor, client, clock = await synced(tmp_path, sample_bytes, LATEST + 600)
    assert (
        ingestor._publish_lag == DEFAULT_PUBLISH_LAG_SECONDS
    )  # nothing learned from the first poll

    client.newest = LATEST + FRAME_STEP_SECONDS
    clock.now = LATEST + FRAME_STEP_SECONDS + 420  # this one appeared 420 s after its time
    await ingestor.sync_once()

    assert ingestor._publish_lag == pytest.approx((DEFAULT_PUBLISH_LAG_SECONDS + 420) / 2)
    # an unchanged newest frame teaches nothing
    clock.now += 60
    before = ingestor._publish_lag
    await ingestor.sync_once()
    assert ingestor._publish_lag == before


async def test_learned_delay_stays_within_bounds(tmp_path: Path, sample_bytes: bytes) -> None:
    ingestor, client, clock = await synced(tmp_path, sample_bytes, LATEST + 600)
    client.newest = LATEST + FRAME_STEP_SECONDS
    clock.now = LATEST + FRAME_STEP_SECONDS + 100_000  # a bizarre outage-sized delay
    await ingestor.sync_once()
    assert ingestor._publish_lag <= MAX_PUBLISH_LAG_SECONDS


async def test_run_sleeps_until_the_next_frame_is_due(tmp_path: Path, sample_bytes: bytes) -> None:
    clock = Clock()
    clock.now = LATEST + 600
    ingestor, _, _ = make(tmp_path, sample_bytes, clock=clock)
    slept: list[float] = []

    async def record_sleep(seconds: float) -> None:
        if seconds == DOWNLOAD_PAUSE_SECONDS:
            return
        slept.append(seconds)
        raise asyncio.CancelledError

    ingestor._sleep = record_sleep  # type: ignore[assignment]
    with pytest.raises(asyncio.CancelledError):
        await ingestor.run()
    assert slept == [pytest.approx(225)]
