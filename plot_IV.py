from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUR_DATA_DIR = BASE_DIR / "our_data"
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "image"
ENCODINGS = ["utf-8-sig", "utf-8", "cp936", "gbk", "big5"]


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise last_error


def find_group_starts(drain_series: pd.Series) -> list[int]:
    starts = [0]
    for i in range(1, len(drain_series)):
        if float(drain_series.iloc[i]) < float(drain_series.iloc[i - 1]):
            starts.append(i)
    return starts


def gate_labels_for_groups(group_gate_values: list[float]) -> list[float]:
    rounded = [round(float(v), 1) for v in group_gate_values]
    unique_count = len(set(rounded))
    monotonic = all(rounded[i + 1] >= rounded[i] for i in range(len(rounded) - 1))
    unit_steps = (
        all(abs((rounded[i + 1] - rounded[i]) - 1.0) < 1e-6 for i in range(len(rounded) - 1))
        if len(rounded) > 1
        else True
    )

    if unique_count == len(rounded) and monotonic and unit_steps:
        return rounded
    if len(rounded) == 7:
        return [float(v) for v in range(-10, -3)]
    if len(rounded) == 13:
        return [float(v) for v in range(-10, 3)]
    return rounded


def plot_idvd(df: pd.DataFrame, title: str, out_path: Path) -> None:
    drain_v_col = df.columns[2]
    drain_i_col = df.columns[3]

    starts = find_group_starts(df[drain_v_col])
    stops = starts[1:] + [len(df)]
    group_gate_values = [df.iloc[start, 0] for start in starts]
    labels = gate_labels_for_groups(group_gate_values)

    plt.figure(figsize=(7.5, 5.2))
    for start, stop, gate_label in zip(starts, stops, labels):
        segment = df.iloc[start:stop, [2, 3]].dropna().sort_values(by=drain_v_col)
        plt.plot(
            segment.iloc[:, 0],
            segment.iloc[:, 1],
            marker="o",
            markersize=2.2,
            linewidth=1.1,
            label=f"Vg={gate_label:.1f}V",
        )

    plt.xlabel("V_ds (V)")
    plt.ylabel("I_ds (A)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_idvg(df: pd.DataFrame, title: str, out_path: Path) -> None:
    gate_v_col = df.columns[0]

    segment = df.iloc[:, [0, 3]].dropna().sort_values(by=gate_v_col)

    plt.figure(figsize=(6.2, 4.5))
    plt.plot(segment.iloc[:, 0], segment.iloc[:, 1], marker="o", markersize=2.2, linewidth=1.1)
    plt.xlabel("V_gs (V)")
    plt.ylabel("I_ds (A)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_idvg_sqrt_ids(df: pd.DataFrame, title: str, out_path: Path) -> None:
    gate_v_col = df.columns[0]

    segment = df.iloc[:, [0, 3]].dropna().sort_values(by=gate_v_col)
    sqrt_ids = np.sqrt(segment.iloc[:, 1].clip(lower=0))

    plt.figure(figsize=(6.2, 4.5))
    plt.plot(segment.iloc[:, 0], sqrt_ids, marker="o", markersize=2.2, linewidth=1.1)
    plt.xlabel("V_gs (V)")
    plt.ylabel("sqrt(I_DS)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def should_make_sqrt_ids_plot(csv_path: Path, df: pd.DataFrame) -> bool:
    lower_name = csv_path.name.lower()
    if "vd_id" in lower_name or "idvd" in lower_name:
        return False

    try:
        drain_values = sorted({round(float(value), 6) for value in df.iloc[:, 2].dropna()})
    except Exception:
        return False

    if len(drain_values) != 1 or drain_values[0] not in {5.0, 8.0}:
        return False

    if csv_path.is_relative_to(OUR_DATA_DIR):
        return True

    return csv_path.is_relative_to(DATA_DIR / "wee4_exp_data1")


def plot_csv(csv_path: Path, out_path: Path) -> None:
    df = read_csv_with_fallback(csv_path)
    lower_name = csv_path.name.lower()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "vd_id" in lower_name or "idvd" in lower_name:
        plot_idvd(df, csv_path.name, out_path)
    else:
        plot_idvg(df, csv_path.name, out_path)

    if should_make_sqrt_ids_plot(csv_path, df):
        sqrt_out_path = out_path.with_name(f"{out_path.stem}_sqrt_ids{out_path.suffix}")
        plot_idvg_sqrt_ids(df, csv_path.name, sqrt_out_path)


def main() -> None:
    IMAGE_DIR.mkdir(exist_ok=True)

    for csv_path in sorted(OUR_DATA_DIR.glob("*.csv")):
        out_path = IMAGE_DIR / f"{csv_path.stem}.png"
        plot_csv(csv_path, out_path)

    for csv_path in sorted(DATA_DIR.rglob("*.csv")):
        relative_parent = csv_path.parent.relative_to(DATA_DIR)
        out_path = IMAGE_DIR / relative_parent / f"{csv_path.stem}.png"
        plot_csv(csv_path, out_path)


if __name__ == "__main__":
    main()
