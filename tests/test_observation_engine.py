from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.observation_engine import DetectorConfig, generate_observations, read_summary_metrics


class ObservationEngineTests(unittest.TestCase):
    def test_detects_plateau_gap_and_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.csv"
            path.write_text(
                "epoch,train_loss,val_loss,train_auc,val_auc\n"
                "1,0.5,0.5,0.50,0.40\n"
                "2,0.4,0.4,0.60,0.50\n"
                "3,0.3,0.3,0.70,0.60\n"
                "4,0.2,0.2,0.80,0.70\n"
                "5,0.1,0.1,0.90,0.79\n",
                encoding="utf-8",
            )
            observations = generate_observations(
                "exp_1",
                history_path=path,
                config=DetectorConfig(plateau_window=3, plateau_min_delta=0.02, generalisation_gap_threshold=0.1),
            )
            types = {item["type"] for item in observations}
            self.assertIn("generalisation_gap", types)

    def test_summary_metrics_supports_target_metric_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary_metrics.csv"
            path.write_text(
                "target,metric,value\ncompound,macro_f1,0.25\naction,accuracy,0.40\n",
                encoding="utf-8",
            )
            metrics = read_summary_metrics(path)
            self.assertEqual(metrics["compound.macro_f1"], 0.25)
            self.assertEqual(metrics["action.accuracy"], 0.40)


if __name__ == "__main__":
    unittest.main()
