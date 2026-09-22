"""
test_live_demo.py

Unit tests for in-memory feature extraction and local classification modules in live_demo.py.
Verifies that geometric metrics, kinematics, and decision trees operate reliably in real-time.
"""

import math
import unittest
from pathlib import Path
import numpy as np

from live_demo import TrajectoryClassifier, extract_live_metrics, BASE_DIR, SUMMARY_CSV


class TestLiveDemoComponents(unittest.TestCase):
    def setUp(self):
        # Synthetic straight line trajectory
        self.straight_traj = [
            {"x": float(i), "y": 100.0, "time": i * 0.01}
            for i in range(50)
        ]
        # Synthetic circular arc trajectory
        self.arc_traj = [
            {
                "x": 200.0 + 100.0 * math.cos(t),
                "y": 200.0 + 100.0 * math.sin(t),
                "time": i * 0.01
            }
            for i, t in enumerate(np.linspace(0, math.pi / 2, 50))
        ]

    def test_extract_live_metrics_straight(self):
        metrics = extract_live_metrics(self.straight_traj)
        self.assertIn("path_length", metrics)
        self.assertIn("max_chord_dev_px", metrics)
        self.assertIn("peak_to_mean_speed", metrics)
        self.assertIn("power_law_beta", metrics)
        self.assertIn("power_law_r", metrics)
        self.assertIn("log_dimensionless_jerk", metrics)
        self.assertIn("affine_velocity_cv", metrics)
        self.assertIn("bezier_residual_px", metrics)
        self.assertIn("time_to_peak_ratio", metrics)

        # An ideal straight line should have near-zero chord deviation and path ratio == 1.0
        self.assertLess(metrics["max_chord_dev_px"], 1e-3)
        self.assertLess(metrics["bezier_residual_px"], 1e-3)
        self.assertAlmostEqual(metrics["path_ratio"], 1.0, places=2)

    def test_extract_live_metrics_arc(self):
        metrics = extract_live_metrics(self.arc_traj)
        self.assertGreater(metrics["max_chord_dev_px"], 10.0)
        self.assertGreater(metrics["total_angle_change"], 1.0)
        self.assertFalse(np.isnan(metrics["power_law_beta"]))
        self.assertFalse(np.isnan(metrics["log_dimensionless_jerk"]))

    def test_extract_live_metrics_empty_or_short(self):
        metrics = extract_live_metrics([])
        self.assertEqual(metrics, {})
        metrics_short = extract_live_metrics([{"x": 0, "y": 0, "time": 0}])
        self.assertEqual(metrics_short, {})

    def test_classifier_training_and_prediction(self):
        classifier = TrajectoryClassifier(SUMMARY_CSV)
        if SUMMARY_CSV.exists():
            self.assertTrue(classifier.is_trained)
            
            # Predict on test trajectory
            metrics = extract_live_metrics(self.arc_traj)
            bin_pred, conf_b, multi_pred, conf_m = classifier.predict(metrics)
            self.assertIn(bin_pred, ["human", "bot"])
            self.assertGreaterEqual(conf_b, 0.0)
            self.assertLessEqual(conf_b, 1.0)
            self.assertIsInstance(multi_pred, str)
            self.assertGreater(len(multi_pred), 0)
            self.assertGreaterEqual(conf_m, 0.0)
            self.assertLessEqual(conf_m, 1.0)

    def test_classifier_fallback_heuristic(self):
        # Untrained model fallback (non-existent path)
        dummy_classifier = TrajectoryClassifier(Path("/nonexistent/file.csv"))
        self.assertFalse(dummy_classifier.is_trained)

        # Verify fallback heuristic logic
        bin_pred, conf_b, multi_pred, conf_m = dummy_classifier.predict({"peak_to_mean_speed": 2.5, "log_dimensionless_jerk": 8.0})
        self.assertEqual(bin_pred, "human")
        self.assertGreater(conf_b, 0.5)

        bin_pred_bot, conf_b_bot, multi_pred_bot, conf_m_bot = dummy_classifier.predict({"peak_to_mean_speed": 1.0, "log_dimensionless_jerk": 6.0})
        self.assertEqual(bin_pred_bot, "bot")
        self.assertGreater(conf_b_bot, 0.5)


if __name__ == "__main__":
    unittest.main()
