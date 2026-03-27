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


def safe_text(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


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


def find_max_positive_slope_pair(points):
    best = None
    for i, (left, right) in enumerate(zip(points, points[1:])):
        left_row, x1, y1 = left
        right_row, x2, y2 = right
        dx = x2 - x1
        if dx <= 0:
            continue
        slope = (y2 - y1) / dx
        if slope <= 0:
            continue
        if best is None or slope > best["slope"]:
            best = {
                "index": i,
                "slope": slope,
                "left_row": left_row,
                "right_row": right_row,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
    return best


def fit_line(points):
    xs = [point[1] for point in points]
    ys = [point[2] for point in points]
    n = len(points)

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0:
        raise ValueError("Cannot fit line when all x values are identical")

    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

    return slope, intercept, r_squared


def pick_best_window(points, pair_index: int, window_size: int):
    if len(points) < 2:
        raise ValueError("Need at least two points to estimate threshold voltage")

    window_size = max(2, min(window_size, len(points)))
    min_start = max(0, pair_index - window_size + 2)
    max_start = min(pair_index, len(points) - window_size)

    candidates = []
    for start in range(min_start, max_start + 1):
        window = points[start : start + window_size]
        slope, intercept, r_squared = fit_line(window)
        if slope <= 0:
            continue
        candidates.append(
            {
                "start": start,
                "stop": start + window_size,
                "points": window,
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
            }
        )

    if not candidates:
        raise ValueError("No valid positive-slope fitting window found")

    candidates.sort(key=lambda item: (item["r_squared"], item["slope"]), reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate threshold voltage Vt from an Id-Vg CSV measured in the linear region "
            "by fitting a line around the steepest current-growth segment."
        )
    )
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument(
        "--x",
        help="X column name or zero-based index. Default: column 0 (gate voltage).",
    )
    parser.add_argument(
        "--y",
        help="Y column name or zero-based index. Default: column 3 (drain current).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Number of adjacent points used for the local linear fit. Default: 7.",
    )
    parser.add_argument(
        "--sqrt-current",
        action="store_true",
        help="Take sqrt of the y/current values before slope search and linear fitting.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    rows = read_csv_rows(csv_path)
    if len(rows) < 2:
        raise ValueError("CSV must contain a header and at least one data row")

    header = rows[0]
    x_index = resolve_column(args.x, header, 0)
    y_index = resolve_column(args.y, header, 3 if len(header) > 3 else 1)
    points = collect_points(rows, x_index, y_index)
    if len(points) < 2:
        raise ValueError("Not enough numeric data points to estimate threshold voltage")
    if args.sqrt_current:
        points = transform_points_for_sqrt_current(points)

    max_pair = find_max_positive_slope_pair(points)
    if max_pair is None:
        raise ValueError("No adjacent pair with positive slope was found")

    best_window = pick_best_window(points, max_pair["index"], args.window)
    slope = best_window["slope"]
    intercept = best_window["intercept"]
    vt = -intercept / slope

    first_row = best_window["points"][0][0]
    last_row = best_window["points"][-1][0]
    first_x = best_window["points"][0][1]
    last_x = best_window["points"][-1][1]

    print(f"file: {csv_path}")
    print(f"x_column: {safe_text(header[x_index])} (index {x_index})")
    print(f"y_column: {safe_text(header[y_index])} (index {y_index})")
    print(f"sqrt_current: {args.sqrt_current}")
    print(f"max_adjacent_slope: {max_pair['slope']:.12g}")
    print(f"max_slope_rows: {max_pair['left_row']} -> {max_pair['right_row']}")
    print(f"fit_window_size: {len(best_window['points'])}")
    print(f"fit_rows: {first_row} -> {last_row}")
    print(f"fit_x_range: {first_x} -> {last_x}")
    print(f"fit_slope: {slope:.12g}")
    print(f"fit_intercept: {intercept:.12g}")
    print(f"fit_r_squared: {best_window['r_squared']:.12g}")
    print(f"estimated_Vt: {vt:.12g} V")


if __name__ == "__main__":
    main()
