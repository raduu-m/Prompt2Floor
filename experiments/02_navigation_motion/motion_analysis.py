"""Compare generated worker paths and THOR-MAGNI motion in eight-second windows."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d


def window_metrics(
    time,
    xy,
    sample_rate=20,
    window_s=8,
    smoothing_s=0.3,
    min_speed=0.05,
    reversal_deg=120,
    max_gap_s=0.25,
):
    time = np.asarray(time, dtype=float)
    xy = np.asarray(xy, dtype=float)
    valid = np.isfinite(time) & np.all(np.isfinite(xy), axis=1)
    time = time[valid]
    xy = xy[valid]
    if len(time) < 3:
        return []
    order = np.argsort(time)
    time = time[order]
    xy = xy[order]
    unique = np.r_[True, np.diff(time) > 0]
    time = time[unique]
    xy = xy[unique]
    breaks = np.r_[0, np.flatnonzero(np.diff(time) > max_gap_s) + 1, len(time)]
    size = round(window_s * sample_rate)
    if size < 3:
        raise ValueError("Windows need at least three samples")
    rows = []
    for first, last in zip(breaks[:-1], breaks[1:]):
        t = time[first:last]
        points = xy[first:last]
        if len(t) < 3 or t[-1] - t[0] < window_s:
            continue
        clock = np.arange(t[0], t[-1] + 1e-9, 1 / sample_rate)
        positions = np.column_stack(
            [np.interp(clock, t, points[:, axis]) for axis in range(2)]
        )
        if smoothing_s > 0:
            positions = gaussian_filter1d(
                positions, smoothing_s * sample_rate, axis=0, mode="nearest"
            )
        velocity = np.gradient(positions, 1 / sample_rate, axis=0)
        acceleration = np.gradient(velocity, 1 / sample_rate, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        # Detect true direction reversals on either side of the local smoothing interval.
        lag = max(1, round(max(smoothing_s, 0.1) * sample_rate))
        cosine = np.ones(len(clock))
        a, b = velocity[: -2 * lag], velocity[2 * lag :]
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cosine[lag:-lag] = np.divide(
            np.sum(a * b, axis=1),
            denominator,
            out=np.ones_like(denominator),
            where=denominator > min_speed**2,
        )
        reversals = binary_dilation(
            cosine < math.cos(math.radians(reversal_deg)), iterations=lag
        )
        curvature = (
            np.abs(
                velocity[:, 0] * acceleration[:, 1]
                - velocity[:, 1] * acceleration[:, 0]
            )
            / np.maximum(speed, 1e-12) ** 3
        )
        for start in range(0, len(clock) - size, size):
            stop = start + size
            v = speed[start:stop]
            mean = float(np.mean(v))
            if mean < min_speed:
                continue
            mask = (speed[start:stop] >= min_speed) & ~reversals[start:stop]
            k = curvature[start:stop][mask]
            rows.append(
                dict(
                    start_s=float(clock[start]),
                    end_s=float(clock[stop]),
                    mean_speed=mean,
                    normalized_speed_sd=float(np.std(v) / mean),
                    curvature_p95=float(np.quantile(k, 0.95)) if len(k) else None,
                    curvature_samples=int(len(k)),
                )
            )
    return rows


def synthetic_tracks(root, duration_s=120, sample_rate=20, worldc_root=None):
    if worldc_root is not None:
        sys.path.insert(0, str(Path(worldc_root).resolve() / "src"))
    from worldc.runtime import PathFollower

    root = Path(root)
    clock = np.arange(0, duration_s + 1 / sample_rate / 2, 1 / sample_rate)
    graphs = [root] if root.is_file() else sorted(root.rglob("scene_graph.json"))
    for path in graphs:
        record = path.parent.parent / "record.json"
        if record.exists() and not json.loads(record.read_text(encoding="utf-8")).get(
            "accepted", False
        ):
            continue
        graph = json.loads(path.read_text(encoding="utf-8"))
        group = (
            path.relative_to(root).parts[0]
            if root.is_dir() and len(path.relative_to(root).parts) > 3
            else "synthetic"
        )
        for index, hazard in enumerate(graph.get("hazards", [])):
            if hazard.get("kind") != "human":
                continue
            follower = PathFollower(
                hazard["waypoints"],
                hazard["speed_mps"],
                hazard.get("path_mode", "loop"),
                hazard.get("z", 0),
                hazard.get("path_phase_m", 0),
                dwell_s=hazard.get("dwell_s", 0),
                ease_s=hazard.get("ease_s", 0),
                turn_s=hazard.get("turn_s", 0),
            )
            points = np.asarray([follower.pose(float(t))[:2] for t in clock])
            yield group, str(path), hazard.get("path", str(index)), clock, points


def magni_tracks(path):
    """Read helmet centroids from the native multimodal CSV, converting mm to m."""
    with Path(path).open(encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream)
        for header in reader:
            if header and header[0].strip() == "Frame":
                break
        else:
            raise ValueError(f"No frame header in {path}")
        columns = {name.strip(): i for i, name in enumerate(header)}
        time_col = columns["Time"]
        actors = [
            name[: -len(" Centroid_X")]
            for name in columns
            if name.startswith("Helmet") and name.endswith(" Centroid_X")
        ]
        tracks = {name: [] for name in actors}
        times = []

        def number(row, index):
            try:
                return float(row[index])
            except (ValueError, IndexError):
                return float("nan")

        for row in reader:
            t = number(row, time_col)
            if not np.isfinite(t):
                continue
            times.append(t)
            for name in actors:
                point = [
                    number(row, columns[name + " Centroid_" + axis]) / 1000
                    for axis in ["X", "Y"]
                ]
                if point == [0, 0]:
                    point = [float("nan")] * 2
                tracks[name].append(point)
    for name, points in tracks.items():
        yield "THOR-MAGNI", str(path), name, np.asarray(times), np.asarray(points)


def save_outputs(rows, output, settings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No valid motion windows")
    with (output / "windows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = sorted({r["group"] for r in rows})
    summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for group in groups:
        subset = [r for r in rows if r["group"] == group]
        summary[group] = dict(
            windows=len(subset),
            recordings=len({r["recording"] for r in subset}),
            tracks=len({(r["recording"], r["actor"]) for r in subset}),
            mean_speed=float(np.mean([r["mean_speed"] for r in subset])),
            median_normalized_speed_sd=float(
                np.median([r["normalized_speed_sd"] for r in subset])
            ),
            median_curvature_p95=float(
                np.median(
                    [
                        r["curvature_p95"]
                        for r in subset
                        if r["curvature_p95"] is not None
                    ]
                )
            ),
        )
        for axis, key, label in zip(
            axes,
            ["normalized_speed_sd", "curvature_p95"],
            ["Normalized speed SD", "95th percentile absolute curvature (1/m)"],
        ):
            values = np.asarray([r[key] for r in subset if r[key] is not None])
            if len(values) > 1 and np.std(values) > 1e-9:
                x = np.linspace(0, max(float(np.quantile(values, 0.995)), 0.01), 400)
                axis.plot(x, gaussian_kde(values)(x), label=group)
            elif len(values):
                axis.axvline(values[0], label=group)
            axis.set_xlabel(label)
            axis.set_ylabel("Density")
            axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output / "motion_distributions.png", dpi=180)
    fig.savefig(output / "motion_distributions.pdf")
    plt.close(fig)
    (output / "summary.json").write_text(
        json.dumps(dict(settings=settings, groups=summary), indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worldc-root", type=Path, help="Path to the worldc dependency checkout"
    )
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--magni", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/motion"))
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--sample-rate", type=float, default=20)
    parser.add_argument("--window", type=float, default=8)
    parser.add_argument("--smoothing", type=float, default=0.3)
    parser.add_argument("--min-speed", type=float, default=0.05)
    parser.add_argument("--max-gap", type=float, default=0.25)
    parser.add_argument("--reversal-deg", type=float, default=120)
    args = parser.parse_args()
    if (
        min(args.duration, args.sample_rate, args.window, args.max_gap) <= 0
        or args.smoothing < 0
    ):
        parser.error("Invalid time or smoothing settings")
    rows = []

    def collect(tracks):
        for group, recording, actor, t, xy in tracks:
            for window in window_metrics(
                t,
                xy,
                args.sample_rate,
                args.window,
                args.smoothing,
                args.min_speed,
                args.reversal_deg,
                args.max_gap,
            ):
                rows.append(
                    dict(group=group, recording=recording, actor=actor, **window)
                )

    collect(
        synthetic_tracks(args.scenes, args.duration, args.sample_rate, args.worldc_root)
    )
    for file in sorted(args.magni.rglob("*.csv")):
        collect(magni_tracks(file))
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    print(json.dumps(save_outputs(rows, args.out, settings), indent=2))


if __name__ == "__main__":
    main()
