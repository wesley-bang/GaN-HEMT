import argparse
import csv
import math
from pathlib import Path


ENCODINGS = ["utf-8-sig", "utf-8", "cp936", "gbk", "big5"]


def read_csv_rows(path: Path):
    last_error = None
    for encoding in ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                rows = list(csv.reader(handle))
            return rows
        except Exception as exc:
            last_error = exc
    raise last_error


def choose_default_x(path: Path, header: list[str]) -> int:
    name = path.name.lower()
    if "vd_id" in name or "idvd" in name:
        return 2 if len(header) > 2 else 0
    return 0


def choose_default_y(header: list[str]) -> int:
    return 3 if len(header) > 3 else 1


def resolve_column(selector: str | None, header: list[str], default_index: int) -> int:
    if selector is None:
        return default_index

    try:
        index = int(selector)
    except ValueError:
        if selector not in header:
            raise ValueError(f"Column '{selector}' not found in header: {header}")
        return header.index(selector)

    if index < 0 or index >= len(header):
        raise ValueError(f"Column index {index} is out of range for {len(header)} columns")
    return index


def collect_points(rows: list[list[str]], x_index: int, y_index: int):
    points = []
    for row_number, row in enumerate(rows[1:], start=2):
        if max(x_index, y_index) >= len(row):
            continue
        x_text = row[x_index].strip()
        y_text = row[y_index].strip()
        if not x_text or not y_text:
            continue
        x_value = float(x_text)
        y_value = float(y_text)
        points.append((row_number, x_value, y_value))
    return points


def transform_points_for_sqrt_current(points):
    transformed = []
    for row_number, x_value, y_value in points:
        if y_value < 0:
            raise ValueError(
                f"Cannot take sqrt of negative current at row {row_number}: {y_value}"
            )
        transformed.append((row_number, x_value, math.sqrt(y_value)))
    return transformed


def find_max_slope(points):
    best = None
    for left, right in zip(points, points[1:]):
        left_row, x1, y1 = left
        right_row, x2, y2 = right
        dx = x2 - x1
        if dx <= 0:
            continue
        slope = (y2 - y1) / dx
        if best is None or slope > best["slope"]:
            best = {
                "slope": slope,
                "left_row": left_row,
                "right_row": right_row,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
    return best


def safe_text(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


def main():
    parser = argparse.ArgumentParser(
        description="Find the maximum slope from adjacent points in a CSV file."
    )
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument(
        "--x",
        help="X column name or zero-based index. Default: auto-detect from file name.",
    )
    parser.add_argument(
        "--y",
        help="Y column name or zero-based index. Default: column 3 if available, otherwise 1.",
    )
    parser.add_argument(
        "--sqrt-current",
        action="store_true",
        help="Take sqrt of the y/current values before computing adjacent-point slopes.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    rows = read_csv_rows(csv_path)
    if len(rows) < 2:
        raise ValueError("CSV must contain a header and at least one data row")

    header = rows[0]
    x_index = resolve_column(args.x, header, choose_default_x(csv_path, header))
    y_index = resolve_column(args.y, header, choose_default_y(header))

    points = collect_points(rows, x_index, y_index)
    if len(points) < 2:
        raise ValueError("Not enough numeric data points to compute slopes")
    if args.sqrt_current:
        points = transform_points_for_sqrt_current(points)

    result = find_max_slope(points)
    if result is None:
        raise ValueError("No valid adjacent pairs found with increasing x values")

    print(f"file: {csv_path}")
    print(f"x_column: {safe_text(header[x_index])} (index {x_index})")
    print(f"y_column: {safe_text(header[y_index])} (index {y_index})")
    print(f"sqrt_current: {args.sqrt_current}")
    print(f"max_slope: {result['slope']:.12g}")
    print(f"rows: {result['left_row']} -> {result['right_row']}")
    print(f"point1: ({result['x1']}, {result['y1']})")
    print(f"point2: ({result['x2']}, {result['y2']})")


if __name__ == "__main__":
    main()
