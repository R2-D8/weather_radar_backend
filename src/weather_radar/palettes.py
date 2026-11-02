"""Colour palettes: rain rate (mm/h) to colour.

A palette is a list of ascending lower bounds with one colour each. Rates below the first bound
are fully transparent (no rain). Snow and mixed rain and snow have their own scales, shared by all
palettes and used only by the "-snow" variants (see [phased]): the rate there is the melted
water, so the snow bands start lower. Tiles are cached by the phone, so **changing the colours of an
existing palette would leave stale tiles on phones: give the new colours a new name instead.**
"""

from __future__ import annotations

from dataclasses import dataclass

RGBA = tuple[int, int, int, int]


@dataclass(frozen=True)
class Palette:
    name: str
    label: str
    #: Ascending lower bounds in mm/h; band i starts at thresholds[i].
    thresholds: tuple[float, ...]
    colours: tuple[RGBA, ...]

    def __post_init__(self) -> None:
        if len(self.thresholds) != len(self.colours):
            raise ValueError("one colour per threshold")
        if list(self.thresholds) != sorted(set(self.thresholds)):
            raise ValueError("thresholds must be ascending and unique")
        if len(self.colours) > 15:
            raise ValueError("at most 15 colours so tiles can use a 4-bit PNG palette")

    def png_palette(self) -> tuple[bytes, bytes]:
        """(RGB palette bytes, alpha bytes) with index 0 reserved for 'transparent'."""
        entries: list[RGBA] = [(0, 0, 0, 0), *self.colours]
        rgb = bytes(c for entry in entries for c in entry[:3])
        alpha = bytes(entry[3] for entry in entries)
        return rgb, alpha


CLASSIC = Palette(
    name="classic",
    label="Classic",
    thresholds=(0.1, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0),
    colours=(
        (170, 225, 255, 170),  # drizzle: pale blue, see-through
        (100, 190, 255, 200),
        (50, 140, 255, 220),
        (0, 170, 100, 230),  # light to moderate: green
        (90, 200, 40, 235),
        (230, 230, 0, 240),  # moderate to heavy: yellow, orange
        (255, 180, 0, 245),
        (255, 120, 0, 250),
        (240, 50, 30, 255),  # heavy: red
        (190, 0, 60, 255),
        (200, 0, 170, 255),  # extreme: magenta, white
        (255, 255, 255, 255),
    ),
)

BLUES = Palette(
    name="blues",
    label="Blues",
    thresholds=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
    colours=(
        (198, 219, 239, 150),
        (158, 202, 225, 185),
        (107, 174, 214, 210),
        (66, 146, 198, 230),
        (33, 113, 181, 240),
        (8, 81, 156, 250),
        (8, 48, 107, 255),
        (3, 19, 43, 255),
    ),
)

PALETTES: dict[str, Palette] = {p.name: p for p in (CLASSIC, BLUES)}
DEFAULT_PALETTE = CLASSIC.name


@dataclass(frozen=True)
class Scale:
    """Ascending lower bounds (mm/h of water) with one colour each."""

    thresholds: tuple[float, ...]
    colours: tuple[RGBA, ...]


#: Mixed rain and snow (sleet): pinks.
MIXED = Scale(
    thresholds=(0.1, 1.0, 4.0, 10.0),
    colours=(
        (255, 200, 230, 190),
        (250, 150, 210, 215),
        (230, 100, 180, 235),
        (200, 50, 150, 250),
    ),
)
#: Snow: white to purple.
SNOW = Scale(
    thresholds=(0.1, 0.3, 0.7, 1.5, 3.0, 6.0),
    colours=(
        (240, 240, 255, 190),
        (210, 205, 255, 210),
        (180, 165, 250, 225),
        (150, 115, 240, 235),
        (120, 65, 210, 245),
        (90, 20, 170, 255),
    ),
)

#: Appended to a palette name for tiles that show snow and mixed precipitation in their own
#: colours ("classic-snow"). Plain names stay rain-only, so tiles phones already keep stay valid.
PHASE_SUFFIX = "-snow"

#: Precipitation phase per pixel, see `precip.phase`.
RAIN, MIXED_PHASE, SNOW_PHASE = 0, 1, 2


def resolve(name: str) -> tuple[Palette, bool] | None:
    """(palette, whether snow is shown) for a palette name from a URL, or None if unknown."""
    phased = name.endswith(PHASE_SUFFIX)
    palette = PALETTES.get(name.removesuffix(PHASE_SUFFIX) if phased else name)
    return None if palette is None else (palette, phased)


def phased(palette: Palette) -> tuple[bytes, bytes, list[tuple[float, ...]]]:
    """(RGB palette, alpha, thresholds per phase) for an 8-bit tile with rain, mixed and snow.

    Index 0 is transparent, then the rain colours, then mixed, then snow.
    """
    entries: list[RGBA] = [(0, 0, 0, 0), *palette.colours, *MIXED.colours, *SNOW.colours]
    rgb = bytes(c for entry in entries for c in entry[:3])
    alpha = bytes(entry[3] for entry in entries)
    return rgb, alpha, [palette.thresholds, MIXED.thresholds, SNOW.thresholds]
