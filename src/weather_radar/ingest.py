"""Keeps the frame store up to date: asks DPC for the newest frame and fetches what is missing."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta

from weather_radar.dpc import FRAME_STEP_SECONDS, DpcClient, DpcError
from weather_radar.frames import FrameFormatError, FrameStore, decode_geotiff

log = logging.getLogger(__name__)

#: At most this many frames are downloaded per poll, so a start with an empty store (72 frames
#: for 6 hours) is spread over a few minutes instead of hammering DPC.
MAX_DOWNLOADS_PER_POLL = 12
#: Pause between two downloads.
DOWNLOAD_PAUSE_SECONDS = 0.3
#: A frame that failed is not asked for again before this many seconds. The newest frame is retried
#: soon because DPC may simply not have published it yet.
RETRY_NEWEST_SECONDS = 60.0
RETRY_OLDER_SECONDS = 3600.0
#: Longest wait between polls while DPC keeps failing.
MAX_BACKOFF_SECONDS = 600.0

# DPC publishes one frame every 5 minutes, but each appears only some minutes after its time
# (about 9 to 10 minutes in practice). Instead of asking every minute, the next poll is timed for
# when the next frame should be there, and only then re-checked every `poll_seconds`.
#: Publication delay assumed until one has been observed.
DEFAULT_PUBLISH_LAG_SECONDS = 540.0
#: Bounds for the learned delay.
MIN_PUBLISH_LAG_SECONDS = 60.0
MAX_PUBLISH_LAG_SECONDS = 1200.0
#: Poll this long before the frame is expected, to absorb a bit of jitter.
EARLY_MARGIN_SECONDS = 15.0
#: Never sleep longer than this, whatever the estimate says.
MAX_IDLE_SECONDS = 300.0
#: Pause between polls while older frames are still being fetched (first start).
BACKFILL_PAUSE_SECONDS = 20.0


def wanted_frames(latest: int, history: timedelta) -> list[int]:
    """Ids of every frame in the window ending at [latest], oldest first."""
    steps = int(history.total_seconds() // FRAME_STEP_SECONDS)
    return [latest - k * FRAME_STEP_SECONDS for k in range(steps, -1, -1)]


@dataclass
class IngestStatus:
    """What /health reports about the ingest loop."""

    last_success: float | None = None
    last_error: str | None = None
    downloaded_total: int = 0


class Ingestor:
    def __init__(
        self,
        client: DpcClient,
        store: FrameStore,
        history: timedelta,
        poll_seconds: float,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._store = store
        self._history = history
        self._poll = poll_seconds
        self._clock = clock
        self._sleep = sleep
        self._failed_at: dict[int, float] = {}
        self._newest_seen: int | None = None
        self._publish_lag = DEFAULT_PUBLISH_LAG_SECONDS
        self._backlog = 0
        self.status = IngestStatus()

    async def sync_once(self) -> int:
        """One poll: prune old frames, download missing ones (newest first). Returns how many."""
        latest = await self._client.latest()
        self._learn_publish_lag(latest)
        wanted = wanted_frames(latest, self._history)
        self._store.prune(wanted[0])
        missing = [
            frame_id
            for frame_id in reversed(wanted)
            if not self._store.has(frame_id) and not self._recently_failed(frame_id, latest)
        ]
        self._backlog = max(0, len(missing) - MAX_DOWNLOADS_PER_POLL)
        downloaded = 0
        for frame_id in missing[:MAX_DOWNLOADS_PER_POLL]:
            if await self._fetch(frame_id):
                downloaded += 1
            await self._sleep(DOWNLOAD_PAUSE_SECONDS)
        self.status.downloaded_total += downloaded
        self.status.last_success = self._clock()
        self.status.last_error = None
        if downloaded:
            log.info(
                "stored %d frame(s); newest is %d; %d on disk",
                downloaded,
                latest,
                len(self._store.ids()),
            )
        return downloaded

    def _learn_publish_lag(self, latest: int) -> None:
        """When a newer frame than before shows up, note how long after its time that was."""
        if self._newest_seen is not None and latest > self._newest_seen:
            observed = self._clock() - latest
            blended = (self._publish_lag + observed) / 2
            self._publish_lag = min(MAX_PUBLISH_LAG_SECONDS, max(MIN_PUBLISH_LAG_SECONDS, blended))
        if self._newest_seen is None or latest > self._newest_seen:
            self._newest_seen = latest

    def _next_delay(self) -> float:
        """Seconds until the next poll: soon while backfilling, else when a new frame is due."""
        if self._backlog > 0:
            return BACKFILL_PAUSE_SECONDS
        if self._newest_seen is None:
            return self._poll
        due = self._newest_seen + FRAME_STEP_SECONDS + self._publish_lag - EARLY_MARGIN_SECONDS
        wait = due - self._clock()
        if wait <= 0:
            return self._poll  # the frame is overdue: check every poll interval until it appears
        return min(wait, MAX_IDLE_SECONDS)

    def _recently_failed(self, frame_id: int, latest: int) -> bool:
        failed = self._failed_at.get(frame_id)
        if failed is None:
            return False
        wait = RETRY_NEWEST_SECONDS if frame_id == latest else RETRY_OLDER_SECONDS
        return self._clock() - failed < wait

    async def _fetch(self, frame_id: int) -> bool:
        try:
            data = await self._client.download(frame_id)
            # Reject anything we cannot place on the map before it reaches the store.
            await asyncio.to_thread(decode_geotiff, data)
        except (DpcError, FrameFormatError) as e:
            log.warning("frame %d not stored: %s", frame_id, e)
            self._failed_at[frame_id] = self._clock()
            return False
        self._store.save(frame_id, data)
        self._failed_at.pop(frame_id, None)
        return True

    async def run(self) -> None:
        """Poll forever, timed to when frames are due; failures are retried with a growing pause."""
        backoff = self._poll
        while True:
            delay = backoff
            try:
                await self.sync_once()
                backoff = self._poll
                delay = self._next_delay()
            except asyncio.CancelledError:
                raise
            except DpcError as e:
                self.status.last_error = str(e)
                log.warning("DPC poll failed: %s", e)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                delay = backoff
            except Exception as e:  # keep the service alive whatever goes wrong in one poll
                self.status.last_error = repr(e)
                log.exception("unexpected error in poll")
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                delay = backoff
            await self._sleep(delay)
