from __future__ import annotations

import csv
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .detector import (
    CenterResult,
    Detector,
    ScanSample,
    compute_beam_center,
    write_samples_csv,
)
from .driver import MODE_DVI, MODE_MEMORY, SLM_DVI_Driver as SLMDriver
from .generator import make_vertical_window, write_santec_csv


def validate_slm_csv(
    csv_path: str | Path,
    expected_width: int | None = None,
    expected_height: int | None = None,
    check_header_index: bool = True,
) -> tuple[int, int]:
    """
    Validate Santec SLM CSV format.

    Expected format:
        A1: y/x or similar label
        row 1, columns B...: x indices, 0, 1, 2, ...
        column A, rows 2...: y indices, 0, 1, 2, ...
        data area: integer grayscale values in [0, 1023]

    Returns:
        (width, height)
    """
    csv_path = Path(csv_path).resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV file is not valid UTF-8/UTF-8-BOM: {csv_path}") from exc

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    rows = [row[:-1] if row and row[-1] == "" else row for row in rows]
    header = rows[0]
    x_labels = header[1:]
    width = len(x_labels)
    height = len(rows) - 1

    if expected_width is not None and width != expected_width:
        raise ValueError(f"CSV width mismatch: got {width}, expected {expected_width}")

    if expected_height is not None and height != expected_height:
        raise ValueError(f"CSV height mismatch: got {height}, expected {expected_height}")

    expected_cols = width + 1
    for row_idx, row in enumerate(rows):
        if len(row) != expected_cols:
            raise ValueError(
                f"CSV row {row_idx + 1} has {len(row)} columns, "
                f"expected {expected_cols}"
            )

    if check_header_index:
        for x, label in enumerate(x_labels):
            try:
                value = int(label)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid x header at row 1, column {x + 2}: {label!r}"
                ) from exc

            if value != x:
                raise ValueError(
                    f"Wrong x index at row 1, column {x + 2}: "
                    f"got {value}, expected {x}"
                )

    for y, row in enumerate(rows[1:]):
        y_label = row[0]

        if check_header_index:
            try:
                y_value = int(y_label)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid y header at row {y + 2}, column 1: {y_label!r}"
                ) from exc

            if y_value != y:
                raise ValueError(
                    f"Wrong y index at row {y + 2}, column 1: "
                    f"got {y_value}, expected {y}"
                )

        for x, cell in enumerate(row[1:]):
            try:
                value = int(cell)
            except ValueError as exc:
                raise ValueError(f"Invalid grayscale at x={x}, y={y}: {cell!r}") from exc

            if not (0 <= value <= 1023):
                raise ValueError(
                    f"Grayscale out of range at x={x}, y={y}: "
                    f"{value}, expected 0~1023"
                )
    return width, height


@dataclass(frozen=True)
class ScanSettings:
    """Immutable snapshot of the adjustable scan parameters."""

    level: int
    window_px: int
    step_px: int
    dwell_seconds: float


class ScanParams:
    """Mutable, thread-safe scan parameters.

    The GUI thread calls update() while the scan worker calls snapshot()
    once per frame, so changes take effect on the next frame.
    """

    def __init__(
        self,
        level: int,
        window_px: int = 5,
        step_px: int = 5,
        dwell_seconds: float = 0.2,
    ):
        self._lock = threading.Lock()
        self._level = _validate_level(level)
        self._window_px = _validate_positive_int(window_px, "window_px")
        self._step_px = _validate_positive_int(step_px, "step_px")
        self._dwell_seconds = _validate_positive_float(dwell_seconds, "dwell_seconds")

    def update(
        self,
        *,
        level: int | None = None,
        window_px: int | None = None,
        step_px: int | None = None,
        dwell_seconds: float | None = None,
    ) -> None:
        with self._lock:
            if level is not None:
                self._level = _validate_level(level)
            if window_px is not None:
                self._window_px = _validate_positive_int(window_px, "window_px")
            if step_px is not None:
                self._step_px = _validate_positive_int(step_px, "step_px")
            if dwell_seconds is not None:
                self._dwell_seconds = _validate_positive_float(
                    dwell_seconds, "dwell_seconds"
                )

    def snapshot(self) -> ScanSettings:
        with self._lock:
            return ScanSettings(
                level=self._level,
                window_px=self._window_px,
                step_px=self._step_px,
                dwell_seconds=self._dwell_seconds,
            )


@dataclass
class ScanResult:
    frames: list[Path] = field(default_factory=list)
    samples: list[ScanSample] = field(default_factory=list)
    center: CenterResult | None = None
    samples_path: Path | None = None


def _wait_while_paused(
    pause_event: threading.Event | None,
    stop_event: threading.Event | None,
) -> bool:
    """Block while paused; return False if stop fires (also while paused)."""
    while pause_event is not None and pause_event.is_set():
        if stop_event is not None and stop_event.is_set():
            return False
        time.sleep(0.05)
    return not (stop_event is not None and stop_event.is_set())


def _interruptible_dwell(
    seconds: float,
    stop_event: threading.Event | None,
    pause_event: threading.Event | None,
) -> bool:
    """Sleep in small slices; freeze the countdown while paused.

    Returns False if stop fires before the dwell completes.
    """
    remaining = float(seconds)
    while remaining > 0:
        if stop_event is not None and stop_event.is_set():
            return False
        if pause_event is not None and pause_event.is_set():
            if not _wait_while_paused(pause_event, stop_event):
                return False
            continue
        slice_seconds = min(0.05, remaining)
        time.sleep(slice_seconds)
        remaining -= slice_seconds
    return True


class SLMController:
    def __init__(
        self,
        display_no: int = 1,
        driver: Any | None = None,
        rate120: bool = False,
    ):
        self.display_no = int(display_no)
        if driver is not None:
            self.driver = driver
        else:
            self.driver = SLMDriver(display_no, rate120=rate120)
        self._opened = False
        self._io_lock = threading.RLock()
        # last pattern sent over DVI: ("gray", level) or ("csv", path)
        self._last_display: tuple[str, Any] | None = None

    @property
    def is_open(self) -> bool:
        return self._opened

    def get_slm_info(self) -> tuple[int, int]:
        with self._io_lock:
            return self.driver.slm_info()

    def detect_displays(self) -> list[tuple[int, int, int, str]]:
        """Probe display numbers and return (no, width, height, name) tuples.

        The SLM reports a name starting with "LCOS-SLM" (Guide 2.4.2).
        """
        if not hasattr(self.driver, "search_displays"):
            raise RuntimeError("driver does not support display search")
        with self._io_lock:
            return self.driver.search_displays()

    def open_slm(self) -> None:
        with self._io_lock:
            self.driver.open_slm()
            self._opened = True

    def close_slm(self) -> None:
        with self._io_lock:
            try:
                self.driver.close_slm()
            finally:
                self._opened = False

    def _ensure_open(self) -> None:
        # Guide 1.3.2: SLM_Disp_Open must precede the display functions.
        if not self._opened:
            self.open_slm()

    def set_dvi_mode(self, slm_number: int = 1) -> None:
        with self._io_lock:
            self.driver.set_video_mode(MODE_DVI, slm_number=slm_number)

    def set_memory_mode(self, slm_number: int = 1) -> None:
        with self._io_lock:
            self.driver.set_video_mode(MODE_MEMORY, slm_number=slm_number)

    def display_grayscale(self, grayscale_value: int, interval: float = 0.2) -> None:
        grayscale_value = _validate_level(grayscale_value)
        with self._io_lock:
            self._ensure_open()
            self.driver.load_grayscale(grayscale_value, interval)
            self._last_display = ("gray", grayscale_value)

    def display_csv(self, csv_path: str | Path, interval: float = 0.2) -> None:
        slm_width, slm_height = self.get_slm_info()
        validate_slm_csv(csv_path, expected_width=slm_width, expected_height=slm_height)
        resolved = str(Path(csv_path).resolve())
        with self._io_lock:
            self._ensure_open()
            self.driver.load_csv(resolved, interval)
            self._last_display = ("csv", resolved)

    def display_mask_csv(
        self,
        data,
        csv_path: str | Path | None = None,
        interval: float = 0.2,
    ) -> Path:
        slm_width, slm_height = self.get_slm_info()
        if csv_path is None:
            csv_path = _temporary_csv_path("slm_mask_")
        csv_path = write_santec_csv(data, csv_path)
        validate_slm_csv(csv_path, expected_width=slm_width, expected_height=slm_height)
        with self._io_lock:
            self._ensure_open()
            self.driver.load_csv(str(csv_path), interval)
            self._last_display = ("csv", str(csv_path))
        return csv_path

    def display_vertical_window(
        self,
        x_start: int,
        level: int,
        window_px: int = 5,
        csv_path: str | Path | None = None,
        interval: float = 0.2,
    ) -> Path:
        slm_width, slm_height = self.get_slm_info()
        data = make_vertical_window(slm_width, slm_height, x_start, level, window_px)
        return self.display_mask_csv(data, csv_path=csv_path, interval=interval)

    def ping(self, slm_number: int = 1, verify_dvi: bool = False) -> None:
        """USB heartbeat (SLM_Ctrl_ReadSU); optionally verify DVI is active.

        Only useful when the USB control channel is connected; the DVI
        keep-alive path is refresh_display().
        """
        with self._io_lock:
            mode = self.driver.ping(slm_number, verify_video_mode=verify_dvi)
        if verify_dvi and mode != MODE_DVI:
            raise RuntimeError(
                f"SLM video interface is no longer DVI (mode={mode})"
            )

    def refresh_display(self) -> bool:
        """Re-send the last displayed pattern over DVI (keep-alive).

        Keeps the DVI link active by repeating the same data; returns False
        when the SLM is busy, not open, or nothing has been displayed yet.
        """
        if not self._io_lock.acquire(blocking=False):
            return False
        try:
            last = self._last_display
            if not self._opened or last is None:
                return False
            kind, value = last
            if kind == "gray":
                self.driver.load_grayscale(value, 0.0)
            else:
                self.driver.load_csv(value, 0.0)
            return True
        finally:
            self._io_lock.release()

    def run_center_scan(
        self,
        params: ScanParams,
        *,
        start_x: int = 0,
        end_x: int | None = None,
        output_dir: str | Path | None = None,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        detector: Detector | None = None,
        progress_callback: Callable[[int, int, Path], None] | None = None,
        sample_callback: Callable[[float, float], None] | None = None,
    ) -> ScanResult:
        """Run a center scan with live-adjustable parameters.

        Parameters are re-read from `params` before every frame, so updates
        from another thread apply on the next frame. `pause_event` suspends
        the scan (set = paused); `stop_event` aborts it, also while paused.
        """
        slm_width, slm_height = self.get_slm_info()
        if output_dir is None:
            output_path = Path(tempfile.mkdtemp(prefix="santec_center_scan_"))
        else:
            output_path = Path(output_dir).resolve()
            output_path.mkdir(parents=True, exist_ok=True)

        start_x = max(0, min(int(start_x), slm_width - 1))
        if end_x is None:
            end_x = slm_width - 1
        end_x = max(0, min(int(end_x), slm_width - 1))
        if end_x < start_x:
            raise ValueError("end_x must be greater than or equal to start_x")

        self._ensure_open()
        result = ScanResult()
        position = start_x
        index = 0
        while position <= end_x:
            if not _wait_while_paused(pause_event, stop_event):
                break
            settings = params.snapshot()

            data = make_vertical_window(
                slm_width, slm_height, position, settings.level, settings.window_px
            )
            csv_path = output_path / f"center_scan_x{position:04d}.csv"
            write_santec_csv(data, csv_path)
            validate_slm_csv(csv_path, expected_width=slm_width, expected_height=slm_height)
            # dwell is handled here in interruptible slices, not in the driver
            with self._io_lock:
                self.driver.load_csv(str(csv_path), 0.0)
                self._last_display = ("csv", str(csv_path))
            result.frames.append(csv_path)

            dwell_completed = _interruptible_dwell(
                settings.dwell_seconds, stop_event, pause_event
            )

            if detector is not None:
                x_center = position + settings.window_px / 2.0
                detector.on_frame(x_center)
                signal = float(detector.read())
                result.samples.append(ScanSample(x_center=x_center, signal=signal))
                if sample_callback is not None:
                    sample_callback(x_center, signal)

            if progress_callback is not None:
                progress_callback(index, position, csv_path)

            if not dwell_completed:
                break
            # re-read the step so a change during this frame moves the very
            # next position
            position += params.snapshot().step_px
            index += 1

        if len(result.samples) >= 2:
            result.center = compute_beam_center(result.samples)
        if result.samples:
            result.samples_path = write_samples_csv(
                result.samples, output_path / "scan_samples.csv"
            )
        return result

    def display_center_scan(
        self,
        level: int,
        *,
        window_px: int = 5,
        step_px: int = 5,
        start_x: int = 0,
        end_x: int | None = None,
        dwell_seconds: float = 0.2,
        output_dir: str | Path | None = None,
        stop_event: threading.Event | None = None,
        progress_callback: Callable[[int, Path], None] | None = None,
    ) -> list[Path]:
        """Fixed-parameter center scan (wrapper around run_center_scan)."""
        params = ScanParams(
            level,
            window_px=window_px,
            step_px=step_px,
            dwell_seconds=dwell_seconds,
        )
        adapted = (
            (lambda index, _x, path: progress_callback(index, path))
            if progress_callback is not None
            else None
        )
        result = self.run_center_scan(
            params,
            start_x=start_x,
            end_x=end_x,
            output_dir=output_dir,
            stop_event=stop_event,
            progress_callback=adapted,
        )
        return result.frames


def _validate_level(level: int) -> int:
    try:
        value = int(level)
    except (TypeError, ValueError) as exc:
        raise ValueError("grayscale level must be an integer") from exc
    if not 0 <= value <= 1023:
        raise ValueError("grayscale level must be in 0..1023")
    return value


def _validate_positive_int(value: int, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_positive_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _temporary_csv_path(prefix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=prefix, delete=False
    )
    handle.close()
    return Path(handle.name)
