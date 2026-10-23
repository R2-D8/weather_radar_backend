"""Radar frames: reading DPC GeoTIFFs and keeping a rolling window of them on disk."""

from __future__ import annotations

import io
import logging
import threading
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from weather_radar import geo

log = logging.getLogger(__name__)

# DPC's GeoTIFFs carry a malformed EXIF block that makes Pillow warn on every file; the pixel
# data and the georeferencing tags we use are fine.
warnings.filterwarnings("ignore", message="Corrupt EXIF data", category=UserWarning)

Grid = NDArray[np.float32]

_TAG_PIXEL_SCALE = 33550
_TAG_TIEPOINT = 33922
_NODATA = -9999.0


class FrameFormatError(ValueError):
    """The bytes are not a DPC rain-rate frame we know how to place on the map."""


def decode_geotiff(data: bytes) -> Grid:
    """Rain rate in mm/h per 1 km cell, shape (rows, cols); cells without radar coverage are NaN."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (OSError, ValueError, SyntaxError) as e:
        raise FrameFormatError(f"not a readable image: {e}") from e
    if image.mode != "F" or image.size != (geo.GRID_COLS, geo.GRID_ROWS):
        raise FrameFormatError(f"unexpected image: mode {image.mode}, size {image.size}")
    tags = getattr(image, "tag_v2", None)  # only TIFF images have tags
    if tags is None:
        raise FrameFormatError("not a TIFF image")
    scale = tags.get(_TAG_PIXEL_SCALE)
    tiepoint = tags.get(_TAG_TIEPOINT)
    if scale is None or tiepoint is None:
        raise FrameFormatError("missing georeferencing tags")
    if tuple(scale[:2]) != (geo.GRID_PIXEL_M, geo.GRID_PIXEL_M) or tuple(tiepoint[3:5]) != (
        geo.GRID_ORIGIN_X,
        geo.GRID_ORIGIN_Y,
    ):
        raise FrameFormatError(f"grid changed: pixel scale {scale[:2]}, tiepoint {tiepoint}")
    grid: Grid = np.array(image, dtype=np.float32)
    # -9999 marks "no data"; a rain rate can never be negative, so treat any negative as missing.
    grid[grid < 0] = np.nan
    return grid


def _halve(grid: Grid) -> Grid:
    """Average 2 x 2 blocks, ignoring missing cells; a block with no data stays missing."""
    rows, cols = grid.shape
    blocks = grid.reshape(rows // 2, 2, cols // 2, 2)
    valid = ~np.isnan(blocks)
    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
    count = valid.sum(axis=(1, 3))
    mean = total / np.maximum(count, 1)
    return np.where(count > 0, mean, np.nan).astype(np.float32)


class RadarFrame:
    """One decoded frame plus lazily built, progressively coarser copies for zoomed-out tiles."""

    def __init__(self, frame_id: int, grid: Grid) -> None:
        self.id = frame_id
        self._levels: list[Grid] = [grid]
        self._lock = threading.Lock()

    def level(self, level: int) -> Grid:
        with self._lock:
            while len(self._levels) <= level:
                self._levels.append(_halve(self._levels[-1]))
            return self._levels[level]


class FrameStore:
    """Frames kept as the original GeoTIFFs (`<epoch seconds>.tif`) plus a decoded-frame cache."""

    def __init__(self, directory: Path, max_decoded: int = 6) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_decoded = max_decoded
        self._decoded: OrderedDict[int, RadarFrame] = OrderedDict()
        self._lock = threading.Lock()

    def _path(self, frame_id: int) -> Path:
        return self._dir / f"{frame_id}.tif"

    def ids(self) -> list[int]:
        """All stored frame ids, oldest first."""
        found = []
        for p in self._dir.glob("*.tif"):
            try:
                found.append(int(p.stem))
            except ValueError:
                continue
        return sorted(found)

    def has(self, frame_id: int) -> bool:
        return self._path(frame_id).exists()

    def save(self, frame_id: int, data: bytes) -> None:
        """Store the bytes (already validated by the caller) atomically."""
        tmp = self._dir / f".{frame_id}.tmp"
        tmp.write_bytes(data)
        tmp.replace(self._path(frame_id))

    def load(self, frame_id: int) -> RadarFrame:
        """The decoded frame, or KeyError if absent. A corrupt file is deleted to be re-fetched."""
        with self._lock:
            cached = self._decoded.get(frame_id)
            if cached is not None:
                self._decoded.move_to_end(frame_id)
                return cached
        path = self._path(frame_id)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise KeyError(frame_id) from None
        try:
            frame = RadarFrame(frame_id, decode_geotiff(data))
        except FrameFormatError:
            log.exception("deleting unreadable frame %s", frame_id)
            path.unlink(missing_ok=True)
            raise KeyError(frame_id) from None
        with self._lock:
            self._decoded[frame_id] = frame
            while len(self._decoded) > self._max_decoded:
                self._decoded.popitem(last=False)
        return frame

    def prune(self, oldest_to_keep: int) -> list[int]:
        """Delete frames older than [oldest_to_keep] (epoch seconds); returns the deleted ids."""
        removed = [i for i in self.ids() if i < oldest_to_keep]
        for frame_id in removed:
            self._path(frame_id).unlink(missing_ok=True)
        with self._lock:
            for frame_id in removed:
                self._decoded.pop(frame_id, None)
        return removed
