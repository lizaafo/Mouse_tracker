"""
test_live_demo.py

בדיקות יחידה למודול הסיווג המקומי וחילוץ המדדים בזיכרון של live_demo.py.
מוודא שזיהוי התנועות והמדדים הקינמטיים עובדים בצורה מדויקת וללא חריגות בזמן אמת.
"""

import math
import unittest
from pathlib import Path
import numpy as np

from live_demo import TrajectoryClassifier, extract_live_metrics, BASE_DIR, SUMMARY_CSV


class TestLiveDemoComponents(unittest.TestCase):
    def setUp(self):
        # מסלול ישר סינתטי
        self.straight_traj = [
            {"x": float(i), "y": 100.0, "time": i * 0.01}
            for i in range(50)
        ]
        # מסלול קשתי סינתטי
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

        # מסלול ישר לחלוטין צריך סטיית מיתר אפסית
        self.assertLess(metrics["max_chord_dev_px"], 1e-3)
        self.assertAlmostEqual(metrics["path_ratio"], 1.0, places=2)

    def test_extract_live_metrics_arc(self):
        metrics = extract_live_metrics(self.arc_traj)
        self.assertGreater(metrics["max_chord_dev_px"], 10.0)
        self.assertGreater(metrics["total_angle_change"], 1.0)
        self.assertFalse(np.isnan(metrics["power_law_beta"]))

    def test_extract_live_metrics_empty_or_short(self):
        metrics = extract_live_metrics([])
        self.assertEqual(metrics, {})
        metrics_short = extract_live_metrics([{"x": 0, "y": 0, "time": 0}])
        self.assertEqual(metrics_short, {})

    def test_classifier_training_and_prediction(self):
        classifier = TrajectoryClassifier(SUMMARY_CSV)
        if SUMMARY_CSV.exists():
            self.assertTrue(classifier.is_trained)
            
            # בדיקת חיזוי על מסלול
            metrics = extract_live_metrics(self.arc_traj)
            bin_pred, multi_pred = classifier.predict(metrics)
            self.assertIn(bin_pred, ["human", "bot"])
            self.assertIsInstance(multi_pred, str)
            self.assertGreater(len(multi_pred), 0)

    def test_classifier_fallback_heuristic(self):
        # מודל שלא אומן (נתיב לא קיים)
        dummy_classifier = TrajectoryClassifier(Path("/nonexistent/file.csv"))
        self.assertFalse(dummy_classifier.is_trained)

        # בדיקת היוריסטיקה
        bin_pred, multi_pred = dummy_classifier.predict({"peak_to_mean_speed": 2.5, "power_law_beta": 0.4})
        self.assertEqual(bin_pred, "human")

        bin_pred_bot, multi_pred_bot = dummy_classifier.predict({"peak_to_mean_speed": 1.0, "power_law_beta": 0.0})
        self.assertEqual(bin_pred_bot, "bot")


if __name__ == "__main__":
    unittest.main()
