"""Analyze recorded CSV trajectories without modifying the original recordings.

Run this file normally to write geometry_summary.csv beside it.
Run with --validate to check synthetic trajectories without reading experiment data.
Positions/lengths: pixels. Time: seconds. Curvature: 1/pixel. Angles: radians.
"""

import argparse
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd


def validate_settings(num_resample_points, smoothing_window):
    # Seven or more output points support the local cubic fit and interior estimates.
    for name, value in (("num_resample_points", num_resample_points),
                        ("smoothing_window", smoothing_window)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if num_resample_points < 7:
        raise ValueError("num_resample_points must be at least 7")
    if smoothing_window < 5 or smoothing_window % 2 != 1:
        raise ValueError("smoothing_window must be odd and at least 5")
    if smoothing_window > num_resample_points - 2:
        raise ValueError("smoothing_window must leave at least three interior estimates")


def numeric_column(df, name):
    values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Column {name} contains missing, nonnumeric or infinite values")
    return values


def constant_value(df, name, fallback):
    if name not in df:
        return fallback
    valid = df[name].dropna()
    if len(valid) == 0:
        return fallback
    if valid.nunique() != 1:
        raise ValueError(f"Column {name} must contain one consistent value per file")
    return str(valid.iloc[0])


def compute_total_angle_change(points):
    """Compute total absolute angular change in radians from discrete trajectory points."""
    if len(points) < 2:
        return np.nan
    vectors = np.diff(points, axis=0)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    change = np.diff(angles)
    wrapped = np.arctan2(np.sin(change), np.cos(change))
    return float(np.abs(wrapped).sum()) if len(wrapped) else 0.0


def estimate_curvature(points, segment_lengths, count, window):
    # Resample only the geometric copy; repeated timed samples remain in the source.
    distance = np.r_[0.0, np.cumsum(segment_lengths)]
    grid = np.linspace(0.0, distance[-1], count)
    sampled = np.column_stack([
        np.interp(grid, distance, points[:, axis]) for axis in (0, 1)
    ])
    ds = grid[1] - grid[0]
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=float)
    design = np.column_stack([offsets ** degree for degree in range(4)])
    first, second = [], []
    # Fit a local cubic using NumPy least squares. Derivatives are evaluated at
    # the center, avoiding one-sided endpoint estimates and numerical differencing twice.
    # This smooths at the recorded span; it cannot reconstruct missing source detail.
    for center in range(half, count - half):
        local = sampled[center - half:center + half + 1] - sampled[center]
        coefficients = np.linalg.lstsq(design, local, rcond=None)[0]
        first.append(coefficients[1] / ds)
        second.append(2 * coefficients[2] / ds ** 2)
    first, second = np.asarray(first), np.asarray(second)
    speed = np.linalg.norm(first, axis=1)
    valid = speed > 1e-6
    curvature = np.full(len(speed), np.nan)
    # Do not hide undefined derivatives at cusps with an arbitrary epsilon denominator.
    curvature[valid] = np.abs(
        first[valid, 0] * second[valid, 1] - first[valid, 1] * second[valid, 0]
    ) / speed[valid] ** 3
    return curvature, (window - 1) * ds


def local_curvature_profile(points, segment_lengths, count=100, window=7):
    """Exclude locally unsupported fits while retaining the original source geometry.

    Coverage measures arc length between adjacent retained fit centers. It never
    bridges an excluded center, and omits the endpoint margins of the cubic fit.
    """
    validate_settings(count, window)
    curvature, span = estimate_curvature(points, segment_lengths, count, window)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    length = float(cumulative[-1])
    half_window = window // 2
    centers = np.linspace(0, length, count)[half_window:-half_window]
    radius = span / 2
    ds = length / (count - 1)
    excluded = []

    def exclude(left, right, reason):
        excluded.append({"start_s_px": float(max(0, left)),
                         "end_s_px": float(min(length, right)), "reason": reason})

    # Endpoint ranges are always omitted, including on an otherwise clean path.
    exclude(0, radius, "endpoint")
    exclude(length - radius, length, "endpoint")
    allowed = np.isfinite(curvature)
    gaps = np.flatnonzero(segment_lengths > max(80.0, 2.5 * span))
    for i in gaps:
        # Mask every fit whose full window touches the unobserved source interval.
        left, right = cumulative[i] - radius, cumulative[i + 1] + radius
        allowed &= ~((centers >= left) & (centers <= right))
        exclude(left, right, "sparse_source")

    vectors = np.diff(points, axis=0)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    change = np.diff(angles)
    wrapped = np.arctan2(np.sin(change), np.cos(change))
    reversal_ids = np.flatnonzero(np.abs(wrapped) >= np.pi - 1e-6) + 1
    for i in reversal_ids:
        left, right = cumulative[i] - radius, cumulative[i] + radius
        allowed &= ~((centers >= left) & (centers <= right))
        exclude(left, right, "direction_reversal")

    for i in np.flatnonzero(~np.isfinite(curvature)):
        exclude(centers[i] - ds / 2, centers[i] + ds / 2, "undefined_derivative")

    # Isolated centers do not support an interval and must not affect the summary.
    connected = allowed[:-1] & allowed[1:]
    used = allowed & (np.r_[False, connected] | np.r_[connected, False])
    covered_length = float(np.diff(centers)[connected].sum())
    return {
        "arc_positions": centers, "curvature": curvature, "used": used,
        "span_px": span, "coverage_pct": 100 * covered_length / length,
        "reversal_count": len(reversal_ids), "sparse_gap_count": len(gaps),
        "excluded_ranges": excluded,
        "total_angle_change": float(np.abs(wrapped).sum()) if len(wrapped) else 0.0,
    }

def analyze_single_trajectory(csv_path, num_resample_points=100, smoothing_window=7):
    """Return metrics or an explicit invalid row; never silently omit a bad file."""
    validate_settings(num_resample_points, smoothing_window)
    path = Path(csv_path)
    result = {
        "file_name": path.name, "source_file": str(path.resolve()),
        "session_id": path.parent.name if path.parent.name.startswith("session_") else "legacy",
        "participant_id": path.stem, "status": "invalid", "notes": "",
        "curvature_status": "unavailable", "timing_status": "unavailable",
        "curvature_method": "local_mask_v1",
        "curvature_coverage_pct": 0.0, "curvature_used_points": 0,
        "curvature_total_points": 0, "curvature_excluded_ranges": "[]",
        "curvature_reversal_count": 0, "curvature_sparse_gap_count": 0,
        "num_resample_points": num_resample_points, "smoothing_window": smoothing_window,
        "power_law_beta": np.nan, "power_law_r": np.nan,
    }
    notes = []
    try:
        df = pd.read_csv(path)
        if not {"x", "y"}.issubset(df.columns):
            raise ValueError("Required columns x and y are missing")
        if len(df) < 2:
            raise ValueError("At least two recorded samples are required")
        for key in ("session_id", "participant_id", "trajectory_id"):
            result[key] = constant_value(df, key, result.get(key, ""))
        # Earlier recordings in this project predate bots and are human trials.
        # Keep provenance as labels, not as movement features for a classifier.
        result["source_type"] = constant_value(df, "source_type", "human")
        if result["source_type"] not in ("human", "bot_linear", "bot_curved", "bot_noisy", "bot_smart_jerk", "bot_smart_full"):
                    raise ValueError("Unrecognized source_type")
        for key in ("bot_seed", "planned_duration_s", "control_x", "control_y",
                    "noise_amplitude_px", "generator_version"):
            result[key] = (constant_value(df, key, "")
                           if key in df and df[key].notna().any() else "")
        if "sample_index" in df:
            indices = numeric_column(df, "sample_index")
            if not np.array_equal(indices, np.arange(len(df))):
                raise ValueError("Sample indices are not consecutive from zero")
        points = np.column_stack([numeric_column(df, "x"), numeric_column(df, "y")])
        raw_steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        clean = points[np.r_[True, raw_steps > 0]]
        segments = raw_steps[raw_steps > 0]
        length = float(np.sum(segments))
        direct = float(np.linalg.norm(points[-1] - points[0]))
        result.update(sample_count=len(df), moving_position_count=len(clean),
                      path_length=length, direct_distance=direct,
                      path_ratio=length / direct if direct > 0 else np.nan)
        if direct == 0:
            notes.append("path_ratio undefined: start and end coordinates coincide")
                # Maximum perpendicular deviation from the straight chord (straightness metric)
        chord_vec = clean[-1] - clean[0]
        chord_len = np.linalg.norm(chord_vec)
        if chord_len > 1e-6:
            diff_start = clean - clean[0]
            cross = np.abs(diff_start[:, 0] * chord_vec[1] - diff_start[:, 1] * chord_vec[0])
            result["max_chord_dev_px"] = float(np.max(cross / chord_len))
        else:
            result["max_chord_dev_px"] = 0.0
        # Timing uses every original sample, including pauses, with actual intervals.
        if "time" in df:
            try:
                times = numeric_column(df, "time")
                dt = np.diff(times)
                if times[0] < 0 or np.any(dt <= 0):
                    raise ValueError("Time must be nonnegative and strictly increasing")
                duration = float(times[-1] - times[0])
                result.update(timing_status="ok", duration_s=duration,
                              mean_speed_px_s=length / duration,
                              stationary_interval_s=float(dt[raw_steps == 0].sum()),
                              median_sample_interval_ms=float(np.median(dt) * 1000),
                              max_sample_interval_ms=float(np.max(dt) * 1000))
                            # Velocity profile: ratio of peak speed to mean speed across time slices
                unique_times = np.r_[True, dt > 1e-4]
                if duration > 0 and np.sum(unique_times) >= 5:
                    time_grid = np.linspace(0, duration, 21)
                    grid_x = np.interp(time_grid, times[unique_times] - times[0], points[unique_times, 0])
                    grid_y = np.interp(time_grid, times[unique_times] - times[0], points[unique_times, 1])
                    slice_speeds = np.hypot(np.diff(grid_x), np.diff(grid_y)) / (duration / 20.0)
                    v_mean = np.mean(slice_speeds)
                    result["peak_to_mean_speed"] = float(np.max(slice_speeds) / v_mean) if v_mean > 0 else 1.0
                else:
                    result["peak_to_mean_speed"] = 1.0    
            except ValueError as error:
                result["timing_status"] = "invalid"
                notes.append(str(error))
        else:
            notes.append("No time column: temporal metrics unavailable")

        # Report the target-center distance separately from the actual endpoint chord.
        centers = ("start_x", "start_y", "target_x", "target_y")
        if all(key in df for key in centers):
            values = []
            for key in centers:
                constant_value(df, key, "")
                values.append(numeric_column(df, key)[0])
            sx, sy, tx, ty = values
            result["target_center_distance"] = float(np.hypot(tx - sx, ty - sy))

        # Compute total angular change across all moving positions
        result["total_angle_change"] = compute_total_angle_change(clean)

        result.update(mean_curvature=np.nan, max_curvature=np.nan,
                      p95_curvature=np.nan, max_source_step_px=float(max(segments, default=0)))
        if length == 0 or len(clean) < 7:
            # This numerical eligibility rule is not a claim that seven points are
            # sufficient for scientifically reliable curvature on every trajectory.
            notes.append("Curvature unavailable: fewer than seven moving positions")
        else:
            profile = local_curvature_profile(clean, segments, num_resample_points, smoothing_window)
            used = profile["used"]
            result.update(smoothing_span_px=float(profile["span_px"]),
                          curvature_coverage_pct=profile["coverage_pct"],
                          curvature_used_points=int(used.sum()),
                          curvature_total_points=len(used),
                          curvature_reversal_count=profile["reversal_count"],
                          curvature_sparse_gap_count=profile["sparse_gap_count"],
                          curvature_excluded_ranges=json.dumps(profile["excluded_ranges"]))
            if profile["sparse_gap_count"]:
                notes.append(f"Excluded windows touching {profile['sparse_gap_count']} sparse source gaps")
            if profile["reversal_count"]:
                notes.append(f"Excluded windows around {profile['reversal_count']} exact direction reversals")
            if not np.isfinite(profile["curvature"]).all():
                notes.append("Excluded undefined local derivatives")
            if np.any(used):
                values = profile["curvature"][used]
                result.update(curvature_status="estimated" if used.all() else "partial_estimate",
                              mean_curvature=float(values.mean()),
                              max_curvature=float(values.max()),
                              p95_curvature=float(np.percentile(values, 95)),
                              curvature_std=float(values.std()))                
                if not used.all():
                    notes.append(f"Curvature describes retained portions only ({profile['coverage_pct']:.1f}% of path length)")

                # Two-Thirds Power Law: v(t) ~ alpha * kappa(t)^(-beta)
                # Evaluated on the same resampled arc-length grid
                if result.get("timing_status") == "ok" and length > 0 and len(clean) >= 7:
                    try:
                        cum_s = np.r_[0.0, np.cumsum(segments)]
                        grid_s = np.linspace(0.0, length, num_resample_points)
                        clean_times = numeric_column(df, "time")[np.r_[True, raw_steps > 0]]
                        grid_t = np.interp(grid_s, cum_s, clean_times)
                        ds = grid_s[1] - grid_s[0]
                        dt = np.gradient(grid_t)
                        dt_safe = np.maximum(dt, 1e-6)
                        v_grid = ds / dt_safe

                        half_w = smoothing_window // 2
                        v_mid = v_grid[half_w:num_resample_points - half_w]
                        k_mid = profile["curvature"]

                        valid_pl = (used & np.isfinite(k_mid) & (k_mid > 5e-5) &
                                    np.isfinite(v_mid) & (v_mid > 5.0))
                        if np.sum(valid_pl) >= 8:
                            log_k = np.log(k_mid[valid_pl])
                            log_v = np.log(v_mid[valid_pl])
                            # Fit: log(v) = intercept - beta * log(kappa)
                            slope, _ = np.polyfit(log_k, log_v, 1)
                            beta = float(-slope)
                            r_mat = np.corrcoef(log_k, log_v)
                            r_val = float(r_mat[0, 1]) if r_mat.shape == (2, 2) else np.nan
                            result.update(power_law_beta=beta, power_law_r=r_val)
                    except Exception:
                        pass
            else:
                notes.append("No connected supported portion remains for curvature")
        result["status"] = "review" if notes else "ok"
        result["notes"] = "; ".join(notes)
    except (OSError, ValueError, KeyError, pd.errors.ParserError, np.linalg.LinAlgError) as error:
        # Failed files retain an identifiable row and reason in the batch summary.
        result.update(status="invalid", notes=str(error))
    return result


def discover_trajectories(data_dir):
    # Include earlier direct data/*.csv files, but never search the project root
    # where geometry_summary.csv lives. Sessions keep repeated filenames distinct.
    directory = Path(data_dir)
    files = set(directory.glob("*.csv")) | set(directory.glob("session_*/*.csv"))
    def natural_key(path):
        return [int(part) if part.isdigit() else part
                for part in re.split(r"(\d+)", str(path))]
    return sorted(files, key=natural_key)


def analyze_directory(data_dir, num_resample_points=100, smoothing_window=7):
    validate_settings(num_resample_points, smoothing_window)
    return pd.DataFrame([
        analyze_single_trajectory(path, num_resample_points, smoothing_window)
        for path in discover_trajectories(data_dir)
    ])


def validate_geometry():
    # TemporaryDirectory avoids overwriting or deleting user files during validation.
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        line = directory / "line.csv"
        pd.DataFrame({"x": np.linspace(0, 100, 50), "y": np.zeros(50)}).to_csv(line, index=False)
        result = analyze_single_trajectory(line)
        if not np.isclose(result["path_ratio"], 1) or result["max_curvature"] > 1e-9:
            raise AssertionError("Straight-line validation failed")
        theta = np.linspace(0, np.pi / 2, 200)
        arc = directory / "arc.csv"
        pd.DataFrame({"x": 100 * np.cos(theta), "y": 100 * np.sin(theta)}).to_csv(arc, index=False)
        result = analyze_single_trajectory(arc)
        if not np.isclose(result["mean_curvature"], 0.01, rtol=0.02):
            raise AssertionError("Circular-arc curvature validation failed")
        if not np.isclose(result["path_length"], 50 * np.pi, rtol=1e-4):
            raise AssertionError("Circular-arc length validation failed")
    print("Geometry validation passed: straight line and circular arc.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "geometry_summary.csv")
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate_geometry()
        return
    validate_settings(args.points, args.window)
    # Do not permit the summary to overwrite an input or become an input next time.
    output = args.output.resolve()
    if output.is_relative_to(args.data_dir.resolve()):
        parser.error("Output must be outside the raw data directory")
    results = analyze_directory(args.data_dir, args.points, args.window)
    if results.empty:
        print("No trajectory CSV files found; no summary was written.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results.reindex(columns=["session_id", "participant_id", "source_type", "status", "path_length", "curvature_status", "curvature_coverage_pct"])
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSummary saved: {output}")


if __name__ == "__main__":
    main()
