from .driver import SLM_DVI_Driver as SLMDriver
import csv
import os
from pathlib import Path

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
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except UnicodeDecodeError as e:
        raise ValueError(f"CSV file is not valid UTF-8/UTF-8-BOM: {csv_path}") from e


    # Remove possible trailing empty cells caused by final commas.
    rows = [row[:-1] if row and row[-1] == "" else row for row in rows]
    header = rows[0]
    x_labels = header[1:]
    width = len(x_labels)
    height = len(rows) - 1

    if expected_width is not None and width != expected_width:
        raise ValueError(
            f"CSV width mismatch: got {width}, expected {expected_width}"
        )

    if expected_height is not None and height != expected_height:
        raise ValueError(
            f"CSV height mismatch: got {height}, expected {expected_height}"
        )

    # Check every row has the same length.
    expected_cols = width + 1
    for row_idx, row in enumerate(rows):
        if len(row) != expected_cols:
            raise ValueError(
                f"CSV row {row_idx + 1} has {len(row)} columns, "
                f"expected {expected_cols}"
            )

    if check_header_index:
        # Check x labels: 0, 1, 2, ...
        for x, label in enumerate(x_labels):
            try:
                value = int(label)
            except ValueError as e:
                raise ValueError(
                    f"Invalid x header at row 1, column {x + 2}: {label!r}"
                ) from e

            if value != x:
                raise ValueError(
                    f"Wrong x index at row 1, column {x + 2}: "
                    f"got {value}, expected {x}"
                )

    # Check y labels and grayscale data.
    for y, row in enumerate(rows[1:]):
        y_label = row[0]

        if check_header_index:
            try:
                y_value = int(y_label)
            except ValueError as e:
                raise ValueError(
                    f"Invalid y header at row {y + 2}, column 1: {y_label!r}"
                ) from e

            if y_value != y:
                raise ValueError(
                    f"Wrong y index at row {y + 2}, column 1: "
                    f"got {y_value}, expected {y}"
                )

        for x, cell in enumerate(row[1:]):
            try:
                value = int(cell)
            except ValueError as e:
                raise ValueError(
                    f"Invalid grayscale at x={x}, y={y}: {cell!r}"
                ) from e

            if not (0 <= value <= 1023):
                raise ValueError(
                    f"Grayscale out of range at x={x}, y={y}: "
                    f"{value}, expected 0~1023"
                )
    return width, height


class SLMController:
    def __init__(self, display_no: int = 1):
        self.driver = SLMDriver(display_no)

    def get_slm_info(self):
        return self.driver.slm_info()

    def open_slm(self):
        self.driver.open_slm()

    def close_slm(self):
        self.driver.close_slm()

    def display_grayscale(self, grayscale_value: int):
        self.driver.load_grayscale(grayscale_value)
    
    def display_csv(self, csv_path: str, interval: float = 0.2):
        slm_width, slm_height = self.get_slm_info()
        validate_slm_csv(csv_path, expected_width=slm_width, expected_height=slm_height)
        self.driver.load_csv(csv_path, interval)