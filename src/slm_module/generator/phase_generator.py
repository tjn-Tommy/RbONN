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
class Segment:
    """A band [start, end) along the partition axis, filled with a constant level."""

    start: int
    end: int
    level: int


def make_vertical_window(
    width: int,
    height: int,
    x_start: int,
    level: int,
    window_px: int = 5,
    background_level: int = 0,
) -> np.ndarray:
    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    x_start = _bounded_int(x_start, "x_start", 0, width - 1)
    window_px = _positive_int(window_px, "window_px")
    level = _bounded_int(level, "level", MIN_LEVEL, MAX_LEVEL)
    background_level = _bounded_int(
        background_level, "background_level", MIN_LEVEL, MAX_LEVEL
    )

    data = np.full((height, width), background_level, dtype=np.uint16)
    x_end = min(width, x_start + window_px)
    data[:, x_start:x_end] = level
    return data


def make_segments(
    width: int,
    height: int,
    segments: Sequence[Segment | tuple[int, int, int]],
    *,
    axis: str = "x",
    background_level: int = 0,
) -> np.ndarray:
    """Build a pattern from explicit bands along the chosen axis.

    With ``axis="x"`` each segment fills columns [start, end) with its level
    for the full height (vertical bands); with ``axis="y"`` it fills rows
    [start, end) for the full width (horizontal bands). Segments must not
    overlap; uncovered lines get background_level.
    """
    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    size = _axis_size(axis, width, height)
    background_level = _bounded_int(background_level, "background_level", MIN_LEVEL, MAX_LEVEL)
    if not segments:
        raise ValueError("segments must not be empty")

    normalized: list[Segment] = []
    for index, segment in enumerate(segments):
        if isinstance(segment, Segment):
            start, end, level = segment.start, segment.end, segment.level
        else:
            start, end, level = segment
        name = f"segments[{index}]"
        start = _bounded_int(start, f"{name}.start", 0, size - 1)
        end = _bounded_int(end, f"{name}.end", 1, size)
        level = _bounded_int(level, f"{name}.level", MIN_LEVEL, MAX_LEVEL)
        if end <= start:
            raise ValueError(f"{name}: end must be greater than start")
        normalized.append(Segment(start=start, end=end, level=level))

    ordered = sorted(normalized, key=lambda segment: segment.start)
    for previous, current in zip(ordered, ordered[1:]):
        if current.start < previous.end:
            raise ValueError(
                f"segments overlap: [{previous.start}, {previous.end}) and "
                f"[{current.start}, {current.end})"
            )

    data = np.full((height, width), background_level, dtype=np.uint16)
    for segment in ordered:
        if axis == "x":
            data[:, segment.start:segment.end] = segment.level
        else:
            data[segment.start:segment.end, :] = segment.level
    return data


def make_equal_segments(
    width: int,
    height: int,
    levels: Sequence[int],
    *,
    axis: str = "x",
) -> np.ndarray:
    """Divide the chosen axis into len(levels) equal parts with one level each.

    Boundaries are rounded so the parts cover the full axis exactly even
    when its size is not divisible by the number of parts.
    """
    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    size = _axis_size(axis, width, height)
    if not levels:
        raise ValueError("levels must not be empty")
    count = len(levels)
    if count > size:
        raise ValueError("number of parts cannot exceed the axis size")

    edges = equal_segment_edges(size, count)
    segments = [
        Segment(start=edges[index], end=edges[index + 1], level=int(level))
        for index, level in enumerate(levels)
    ]
    return make_segments(width, height, segments, axis=axis)


def equal_segment_edges(size: int, count: int) -> list[int]:
    """Return count+1 boundary positions dividing [0, size) into equal parts."""
    size = _positive_int(size, "size")
    count = _positive_int(count, "count")
    if count > size:
        raise ValueError("number of parts cannot exceed the axis size")
    return [round(index * size / count) for index in range(count + 1)]


def _axis_size(axis: str, width: int, height: int) -> int:
    if axis == "x":
        return width
    if axis == "y":
        return height
    raise ValueError('axis must be "x" or "y"')


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
    background_level: int = 0,
) -> Iterator[PhasePattern]:
    for x_start in iter_center_scan_positions(
        width,
        window_px=window_px,
        step_px=step_px,
        start_x=start_x,
        end_x=end_x,
    ):
        data = make_vertical_window(
            width, height, x_start, level, window_px, background_level
        )
        yield PhasePattern(x_start=x_start, x_end=min(width, x_start + window_px), data=data)


def read_santec_csv(csv_path: str | Path) -> np.ndarray:
    """Read a Santec SLM CSV back into a 2D uint16 array.

    Inverse of write_santec_csv: drops the ``y/x`` header row and the leading
    y-index column, returning the (height, width) grayscale grid that was
    written. Used to recover exactly what was sent to the SLM for monitoring.
    """
    path = Path(csv_path)
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    # tolerate a trailing empty field left by a line-ending comma
    rows = [row[:-1] if row and row[-1] == "" else row for row in rows]
    if len(rows) < 2:
        raise ValueError(f"CSV has no data rows: {path}")
    try:
        data = [[int(cell) for cell in row[1:]] for row in rows[1:]]
    except ValueError as exc:
        raise ValueError(f"CSV contains a non-integer grayscale: {path}") from exc
    array = np.asarray(data, dtype=np.int64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"CSV did not parse to a 2D grid: {path}")
    if np.any(array < MIN_LEVEL) or np.any(array > MAX_LEVEL):
        raise ValueError(f"CSV grayscale out of range 0..{MAX_LEVEL}: {path}")
    return array.astype(np.uint16, copy=False)


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
    background_level: int = 0,
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
        background_level=background_level,
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
