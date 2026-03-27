import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
        points.append((row_number, float(x_text), float(y_text)))
    return points


def merge_duplicate_x(points):
    grouped = {}
    order = []
    for _, x_value, y_value in points:
        if x_value not in grouped:
            grouped[x_value] = []
            order.append(x_value)
        grouped[x_value].append(y_value)

    x_values = np.array(sorted(order), dtype=float)
    y_values = np.array([sum(grouped[x]) / len(grouped[x]) for x in x_values], dtype=float)
    return x_values, y_values


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot Id-Vgs and gm on dual y-axes, draw the tangent at gm_max, "
            "and estimate Vt from the tangent x-intercept."
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
        "--output",
        help="Output image path. Default: <csv_stem>_gm_vt.png next to the CSV.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    rows = read_csv_rows(csv_path)
    if len(rows) < 3:
        raise ValueError("CSV must contain a header and at least two data rows")

    header = rows[0]
    x_index = resolve_column(args.x, header, 0)
    y_index = resolve_column(args.y, header, 3 if len(header) > 3 else 1)
    points = collect_points(rows, x_index, y_index)
    if len(points) < 2:
        raise ValueError("Not enough numeric data points to compute gm")

    x_values, y_values = merge_duplicate_x(points)
    if len(x_values) < 2:
        raise ValueError("Need at least two unique V_GS points to compute gm")

    gm_values = np.gradient(y_values, x_values)
    gm_max_index = int(np.argmax(gm_values))
    gm_max = float(gm_values[gm_max_index])
    vgs_at_gm_max = float(x_values[gm_max_index])
    ids_at_gm_max = float(y_values[gm_max_index])

    if gm_max == 0:
        raise ValueError("gm_max is zero, cannot build tangent or estimate Vt")

    vt = vgs_at_gm_max - ids_at_gm_max / gm_max

    out_path = (
        Path(args.output)
        if args.output
        else csv_path.with_name(f"{csv_path.stem}_gm_vt.png")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_line = np.linspace(float(x_values.min()), float(x_values.max()), 200)
    tangent_y = gm_max * (x_line - vgs_at_gm_max) + ids_at_gm_max

    fig, ax1 = plt.subplots(figsize=(7.0, 5.0))
    ax2 = ax1.twinx()

    line_ids, = ax1.plot(
        x_values,
        y_values,
        color="tab:blue",
        marker="o",
        markersize=3,
        linewidth=1.3,
        label="I_ds",
    )
    line_gm, = ax2.plot(
        x_values,
        gm_values,
        color="tab:orange",
        marker="s",
        markersize=3,
        linewidth=1.1,
        label="g_m",
    )
    line_tangent, = ax1.plot(
        x_line,
        tangent_y,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label="Tangent @ g_m,max",
    )
    line_gm_vgs = ax1.axvline(
        vgs_at_gm_max,
        color="tab:purple",
        linestyle="--",
        linewidth=1.1,
        label="V_GS @ g_m,max",
    )
    # line_vt = ax1.axvline(
    #     vt,
    #     color="tab:green",
    #     linestyle=":",
    #     linewidth=1.1,
    #     label="V_t",
    # )

    ax1.scatter([vgs_at_gm_max], [ids_at_gm_max], color="tab:red", s=28, zorder=4)
    ax2.scatter([vgs_at_gm_max], [gm_max], color="tab:purple", s=30, zorder=5)

    ax1.set_xlabel("V_GS (V)")
    ax1.set_ylabel("I_DS (A)")
    ax2.set_ylabel("g_m (S)")
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    handles = [line_ids, line_gm, line_tangent, line_gm_vgs]
    labels = [handle.get_label() for handle in handles]
    ax1.legend(handles, labels, loc="upper right", frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"file: {csv_path}")
    print(f"x_column: {safe_text(header[x_index])} (index {x_index})")
    print(f"y_column: {safe_text(header[y_index])} (index {y_index})")
    print(f"output_image: {out_path}")
    print(f"max_gm: {gm_max:.12g} S")
    print(f"vgs_at_max_gm: {vgs_at_gm_max:.12g} V")
    print(f"vt: {vt:.12g} V")


if __name__ == "__main__":
    main()
