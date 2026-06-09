from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


MIN_LEVEL = 0
MAX_LEVEL = 1023


@dataclass(frozen=True)
class PhasePattern:
    x_start: int
    x_end: int
    data: np.ndarray


@dataclass(frozen=True)
class XSegment:
    """A vertical band [x_start, x_end) filled with a constant phase level."""

    x_start: int
    x_end: int
    level: int


def make_vertical_window(
    width: int,
    height: int,
    x_start: int,
    level: int,
    window_px: int = 5,
) -> np.ndarray:
    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    x_start = _bounded_int(x_start, "x_start", 0, width - 1)
    window_px = _positive_int(window_px, "window_px")
    level = _bounded_int(level, "level", MIN_LEVEL, MAX_LEVEL)

    data = np.zeros((height, width), dtype=np.uint16)
    x_end = min(width, x_start + window_px)
    data[:, x_start:x_end] = level
    return data


def make_x_segments(
    width: int,
    height: int,
    segments: Sequence[XSegment | tuple[int, int, int]],
    *,
    background_level: int = 0,
) -> np.ndarray:
    """Build a pattern from explicit vertical bands along the x axis.

    Each segment fills columns [x_start, x_end) with its level for the full
    height. Segments must not overlap; uncovered columns get background_level.
    """
    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    background_level = _bounded_int(background_level, "background_level", MIN_LEVEL, MAX_LEVEL)
    if not segments:
        raise ValueError("segments must not be empty")

    normalized: list[XSegment] = []
    for index, segment in enumerate(segments):
        if isinstance(segment, XSegment):
            x_start, x_end, level = segment.x_start, segment.x_end, segment.level
        else:
            x_start, x_end, level = segment
        name = f"segments[{index}]"
        x_start = _bounded_int(x_start, f"{name}.x_start", 0, width - 1)
        x_end = _bounded_int(x_end, f"{name}.x_end", 1, width)
        level = _bounded_int(level, f"{name}.level", MIN_LEVEL, MAX_LEVEL)
        if x_end <= x_start:
            raise ValueError(f"{name}: x_end must be greater than x_start")
        normalized.append(XSegment(x_start=x_start, x_end=x_end, level=level))

    ordered = sorted(normalized, key=lambda segment: segment.x_start)
    for previous, current in zip(ordered, ordered[1:]):
        if current.x_start < previous.x_end:
            raise ValueError(
                f"segments overlap: [{previous.x_start}, {previous.x_end}) and "
                f"[{current.x_start}, {current.x_end})"
            )

    data = np.full((height, width), background_level, dtype=np.uint16)
    for segment in ordered:
        data[:, segment.x_start:segment.x_end] = segment.level
    return data


def make_equal_x_segments(
    width: int,
    height: int,
    levels: Sequence[int],
) -> np.ndarray:
    """Divide the x axis into len(levels) equal parts with one level each.

    Boundaries are rounded so the parts cover the full width exactly even
    when width is not divisible by the number of parts.
    """
    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    if not levels:
        raise ValueError("levels must not be empty")
    count = len(levels)
    if count > width:
        raise ValueError("number of parts cannot exceed width")

    edges = equal_x_segment_edges(width, count)
    segments = [
        XSegment(x_start=edges[index], x_end=edges[index + 1], level=int(level))
        for index, level in enumerate(levels)
    ]
    return make_x_segments(width, height, segments)


def equal_x_segment_edges(width: int, count: int) -> list[int]:
    """Return count+1 boundary positions dividing [0, width) into equal parts."""
    width = _positive_int(width, "width")
    count = _positive_int(count, "count")
    if count > width:
        raise ValueError("number of parts cannot exceed width")
    return [round(index * width / count) for index in range(count + 1)]


def iter_center_scan_positions(
    width: int,
    *,
    window_px: int = 5,
    step_px: int = 5,
    start_x: int = 0,
    end_x: int | None = None,
) -> Iterator[int]:
    width = _positive_int(width, "width")
    window_px = _positive_int(window_px, "window_px")
    step_px = _positive_int(step_px, "step_px")
    start_x = _bounded_int(start_x, "start_x", 0, width - 1)
    if end_x is None:
        end_x = width - 1
    end_x = _bounded_int(end_x, "end_x", 0, width - 1)
    if end_x < start_x:
        raise ValueError("end_x must be greater than or equal to start_x")

    position = start_x
    while position <= end_x:
        yield position
        position += step_px


def generate_center_scan(
    width: int,
    height: int,
    level: int,
    *,
    window_px: int = 5,
    step_px: int = 5,
    start_x: int = 0,
    end_x: int | None = None,
) -> Iterator[PhasePattern]:
    for x_start in iter_center_scan_positions(
        width,
        window_px=window_px,
        step_px=step_px,
        start_x=start_x,
        end_x=end_x,
    ):
        data = make_vertical_window(width, height, x_start, level, window_px)
        yield PhasePattern(x_start=x_start, x_end=min(width, x_start + window_px), data=data)


def write_santec_csv(data: np.ndarray, csv_path: str | Path) -> Path:
    data_uint16 = _validate_mask_array(data)
    path = Path(csv_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = data_uint16.shape
    # plain ASCII without BOM: the DLL's CSV reader may not skip a UTF-8 BOM
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["y/x", *range(width)])
        for y in range(height):
            writer.writerow([y, *data_uint16[y].tolist()])

    return path


def export_center_scan_sequence(
    output_dir: str | Path,
    width: int,
    height: int,
    level: int,
    *,
    window_px: int = 5,
    step_px: int = 5,
    start_x: int = 0,
    end_x: int | None = None,
    prefix: str = "center_scan",
) -> list[Path]:
    output_path = Path(output_dir).resolve()
    paths: list[Path] = []
    for pattern in generate_center_scan(
        width,
        height,
        level,
        window_px=window_px,
        step_px=step_px,
        start_x=start_x,
        end_x=end_x,
    ):
        csv_path = output_path / f"{prefix}_x{pattern.x_start:04d}.csv"
        paths.append(write_santec_csv(pattern.data, csv_path))
    return paths


def _validate_mask_array(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError("SLM mask data must be a 2D array")
    if array.size == 0:
        raise ValueError("SLM mask data cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("SLM mask data must be finite")
    if np.any(array < MIN_LEVEL) or np.any(array > MAX_LEVEL):
        raise ValueError("SLM mask data must be in 0..1023")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise ValueError("SLM mask data must contain integer levels")
    return rounded.astype(np.uint16, copy=False)


def _positive_int(value: int, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _bounded_int(value: int, name: str, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return result
