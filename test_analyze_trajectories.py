"""Regression checks for geometry, timing and session discovery. Uses temporary files only."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analyze_trajectories import (analyze_directory, analyze_single_trajectory,
                                  local_curvature_profile, validate_geometry)


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, values, name="P01.csv"):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(values).to_csv(path, index=False)
        return path

    def test_known_geometry(self):
        validate_geometry()

    def test_pauses_and_irregular_time_are_preserved(self):
        path = self.write({"x": [0, 0, 10, 10, 20], "y": [0] * 5,
                           "time": [0, 0.2, 0.3, 0.8, 1.0]})
        original = path.read_bytes()
        result = analyze_single_trajectory(path)
        self.assertAlmostEqual(result["duration_s"], 1)
        self.assertAlmostEqual(result["stationary_interval_s"], 0.7)
        self.assertAlmostEqual(result["mean_speed_px_s"], 20)
        self.assertEqual(result["moving_position_count"], 3)
        self.assertTrue(np.isnan(result["mean_curvature"]))
        self.assertEqual(original, path.read_bytes())

    def test_closed_path_ratio_is_not_one(self):
        result = analyze_single_trajectory(self.write({
            "x": [0, 10, 10, 0, 0], "y": [0, 0, 10, 10, 0]}))
        self.assertEqual(result["path_length"], 40)
        self.assertTrue(np.isnan(result["path_ratio"]))

    def test_stationary_path(self):
        result = analyze_single_trajectory(self.write({
            "x": [4] * 10, "y": [7] * 10, "time": np.linspace(0, 1, 10)}))
        self.assertEqual(result["path_length"], 0)
        self.assertAlmostEqual(result["stationary_interval_s"], 1)
        self.assertEqual(result["curvature_status"], "unavailable")

    def test_invalid_time_does_not_destroy_geometry(self):
        for times in ([0, 0, 1], [0, 1, 0.5], [0, np.nan, 1]):
            with self.subTest(times=times):
                result = analyze_single_trajectory(self.write({
                    "x": [0, 1, 2], "y": [0, 0, 0], "time": times}))
                self.assertEqual(result["timing_status"], "invalid")
                self.assertEqual(result["path_length"], 2)
                self.assertNotIn("mean_speed_px_s", result)

    def test_bad_files_are_reported(self):
        cases = [{"x": [0, np.nan], "y": [0, 1]},
                 {"x": [0, np.inf], "y": [0, 1]}, {"x": [0]},
                 {"x": [0, 1], "y": [0, 1], "participant_id": ["P01", "P02"]},
                 {"x": [0, 1], "y": [0, 1], "sample_index": [0, 3]}]
        for values in cases:
            with self.subTest(values=values):
                result = analyze_single_trajectory(self.write(values))
                self.assertEqual(result["status"], "invalid")
                self.assertTrue(result["notes"])

    def test_sessions_remain_distinguishable_and_summary_is_not_input(self):
        for session in ("session_01", "session_02"):
            self.write({"x": [0, 1], "y": [0, 0]}, f"data/{session}/P01.csv")
        self.write({"summary": [123]}, "geometry_summary.csv")
        result = analyze_directory(self.directory / "data")
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["session_id"]), {"session_01", "session_02"})
        self.assertEqual(result["source_file"].nunique(), 2)

    def test_wraparound_angle(self):
        # Small changes across the -pi/pi boundary are not nearly full rotations.
        result = analyze_single_trajectory(self.write({
            "x": [0, -10, -20], "y": [0, 0.1, 0]}))
        self.assertLess(result["total_angle_change"], 0.03)

    def test_sparse_arc_is_flagged(self):
        theta = np.linspace(0, np.pi / 2, 10)
        result = analyze_single_trajectory(self.write({
            "x": 100 * np.cos(theta), "y": 100 * np.sin(theta)}))
        self.assertEqual(result["curvature_status"], "unavailable")
        self.assertEqual(result["curvature_coverage_pct"], 0)
        self.assertGreater(result["curvature_sparse_gap_count"], 0)
        self.assertTrue(np.isnan(result["mean_curvature"]))

    def test_exact_reversal_is_not_reported_as_zero_curvature(self):
        x = [0, 10, 20, 30, 20, 10, 0]
        result = analyze_single_trajectory(self.write({"x": x, "y": [0] * len(x)}))
        self.assertEqual(result["curvature_status"], "unavailable")
        self.assertEqual(result["curvature_reversal_count"], 1)
        self.assertTrue(np.isnan(result["mean_curvature"]))

    def test_dense_reversal_preserves_both_straight_arms(self):
        x = np.r_[np.arange(0, 101), np.arange(99, -1, -1)]
        result = analyze_single_trajectory(self.write({"x": x, "y": np.zeros(len(x))}))
        self.assertEqual(result["curvature_status"], "partial_estimate")
        self.assertEqual(result["curvature_reversal_count"], 1)
        self.assertGreater(result["curvature_coverage_pct"], 80)
        self.assertLess(result["curvature_coverage_pct"], 94)
        self.assertLess(result["max_curvature"], 1e-9)
        self.assertIn("retained portions only", result["notes"])

    def test_gap_excludes_every_overlapping_fit_not_only_its_center(self):
        x = np.r_[np.arange(0, 31), np.arange(70, 101)]
        points = np.column_stack([x, np.zeros(len(x))])
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        profile = local_curvature_profile(points, lengths)
        centers = profile["arc_positions"]
        touching = (centers >= 30 - profile["span_px"] / 2) & (centers <= 70 + profile["span_px"] / 2)
        self.assertFalse(np.any(profile["used"][touching]))
        self.assertTrue(np.any(profile["used"][centers < 25]))
        self.assertTrue(np.any(profile["used"][centers > 75]))
        # A 40 px gap plus a 6.06 px fitting margin and endpoint exclusions
        # leaves under 48% covered; no bridge across the gap is counted.
        self.assertLess(profile["coverage_pct"], 48)
        self.assertGreater(profile["coverage_pct"], 40)

    def test_coverage_accounts_for_endpoints_and_no_gap_double_counting(self):
        points = np.column_stack([np.arange(101), np.zeros(101)])
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        profile = local_curvature_profile(points, lengths)
        self.assertAlmostEqual(profile["coverage_pct"], 100 * 93 / 99)
        self.assertTrue(profile["used"].all())
        self.assertEqual(profile["sparse_gap_count"], 0)

    def test_translated_geometry_keeps_local_masks(self):
        x = np.r_[np.arange(0, 31), np.arange(70, 101)]
        points = np.column_stack([x, np.zeros(len(x))])
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        a = local_curvature_profile(points, lengths)
        b = local_curvature_profile(points + [1234, -987], lengths)
        np.testing.assert_array_equal(a["used"], b["used"])
        self.assertAlmostEqual(a["coverage_pct"], b["coverage_pct"])
        np.testing.assert_allclose(a["curvature"], b["curvature"], atol=1e-10)

    def test_sampling_resolution_and_scale(self):
        theta = np.linspace(0, np.pi / 2, 200)
        path = self.write({"x": 100 * np.cos(theta), "y": 100 * np.sin(theta)})
        a = analyze_single_trajectory(path, 100, 7)
        b = analyze_single_trajectory(path, 200, 13)
        self.assertAlmostEqual(a["mean_curvature"], b["mean_curvature"], delta=0.0001)
        scaled = self.write({"x": 200 * np.cos(theta), "y": 200 * np.sin(theta)}, "scaled.csv")
        c = analyze_single_trajectory(scaled)
        self.assertAlmostEqual(c["mean_curvature"], a["mean_curvature"] / 2, delta=1e-8)
        self.assertAlmostEqual(c["path_ratio"], a["path_ratio"])

    def test_invalid_configuration(self):
        for count, window in [(1, 7), (100, 4), (7, 7), (100.5, 7)]:
            with self.subTest(count=count, window=window):
                with self.assertRaises(ValueError):
                    analyze_single_trajectory(self.directory / "missing.csv", count, window)


if __name__ == "__main__":
    unittest.main()
