from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from src.training.reporting import (
    plot_confusion_matrices,
    plot_embedding_projection,
    plot_grouped_independent_axis_history,
    plot_training_history,
)
from src.training.pretraining import _save_pretraining_loss_pdf


class PlotTrainingHistoryTests(unittest.TestCase):
    def test_plot_training_history_supports_loess_overlay(self) -> None:
        history = pd.DataFrame(
            {
                "epoch": [1, 2, 3, 4, 5],
                "train_loss": [1.0, 0.8, 0.7, 0.65, 0.6],
                "val_loss": [1.1, 0.9, 0.75, 0.7, 0.68],
            }
        )

        fig, axes = plot_training_history(history, loess_frac=0.6)
        ax = axes[0]

        self.assertEqual(len(fig.axes), 2)
        train_ax, val_ax = fig.axes
        self.assertEqual(len(train_ax.lines), 2)
        self.assertEqual(len(val_ax.lines), 2)
        self.assertEqual([line.get_label() for line in train_ax.lines], ["train_loss", "_nolegend_"])
        self.assertEqual([line.get_label() for line in val_ax.lines], ["val_loss", "_nolegend_"])
        self.assertEqual(train_ax.lines[0].get_color(), train_ax.lines[1].get_color())
        self.assertEqual(val_ax.lines[0].get_color(), val_ax.lines[1].get_color())
        self.assertAlmostEqual(train_ax.lines[0].get_alpha(), 0.25)
        self.assertAlmostEqual(train_ax.lines[1].get_alpha(), 0.95)
        self.assertEqual(train_ax.lines[0].get_marker(), "None")
        self.assertEqual(val_ax.lines[0].get_marker(), "None")
        fig.clf()

    def test_grouped_history_limits_panels_to_four_independent_axes(self) -> None:
        history = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "a": [1.0, 0.8, 0.7],
                "b": [10.0, 8.0, 7.0],
                "c": [100.0, 80.0, 70.0],
                "d": [1000.0, 800.0, 700.0],
                "e": [0.1, 0.08, 0.07],
            }
        )

        fig, axes = plot_grouped_independent_axis_history(
            history,
            [("Dense Group", ["a", "b", "c", "d", "e"])],
            title="Grouped",
            smoothing_window=2,
        )

        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].get_title(), "Dense Group 1")
        self.assertEqual(axes[1].get_title(), "Dense Group 2")
        self.assertEqual(len(fig.axes), 5)
        self.assertLessEqual(sum(1 for axis in fig.axes if axis.get_shared_x_axes().joined(axis, axes[0])), 4)
        fig.clf()

    def test_plot_training_history_can_filter_excluded_losses(self) -> None:
        history = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "train_loss": [1.0, 0.8, 0.7],
                "train_compound_loss": [4.0, 4.0, 4.0],
            }
        )

        fig, axes = plot_training_history(history, excluded_loss_names=["compound_loss"])

        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0].get_title(), "Total loss")
        fig.clf()

    def test_pretraining_plot_omits_inactive_cross_losses_and_pairs_train_val(self) -> None:
        class FakeEstimator:
            lambda_cross = 0.0
            prototype_alignment_weight = 0.0
            latent_alignment_weight = 0.0
            lambda_align = 0.0

        history_rows = [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "val_loss": 1.1,
                "train_self_probe_loss": 0.9,
                "val_self_probe_loss": 1.0,
                "train_cross_probe_loss": 10.0,
                "val_cross_probe_loss": 11.0,
                "train_self_probe_local_loss": 0.5,
                "val_self_probe_local_loss": 0.6,
                "train_cross_probe_local_loss": 20.0,
                "val_cross_probe_local_loss": 21.0,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("src.training.pretraining.plot_grouped_independent_axis_history") as plot_mock:
                fig, ax = plt.subplots()
                plot_mock.return_value = (fig, [ax])
                _save_pretraining_loss_pdf(
                    history_rows,
                    Path(tmpdir) / "loss.pdf",
                    smoothing_window=2,
                    estimator=FakeEstimator(),
                )

        groups = plot_mock.call_args.args[1]
        flattened_columns = [column for _, columns in groups for column in columns]
        self.assertIn(("Local Self-Probe Loss", ["train_self_probe_local_loss", "val_self_probe_local_loss"]), groups)
        self.assertNotIn("train_cross_probe_loss", flattened_columns)
        self.assertNotIn("val_cross_probe_loss", flattened_columns)
        self.assertNotIn("train_cross_probe_local_loss", flattened_columns)
        self.assertNotIn("val_cross_probe_local_loss", flattened_columns)

    def test_plot_embedding_projection_places_legend_outside_axes(self) -> None:
        embeddings = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.1, 0.1],
                [1.0, 1.0, 1.0],
                [1.2, 0.9, 1.1],
            ]
        )
        labels = [0, 0, 1, 1]
        plot_embedding_projection(embeddings, labels, {0: "Water", 1: "Drug"}, title="Test projection")
        fig = plt.gcf()
        ax = fig.axes[0]
        legend = ax.get_legend()

        self.assertIsNotNone(legend)
        self.assertEqual(legend._loc, 6)
        anchor_box = legend.get_bbox_to_anchor()._bbox
        self.assertGreater(anchor_box.x0, 1.0)
        fig.clf()

    def test_plot_confusion_matrices_suppresses_zero_cell_text(self) -> None:
        fig, axes, _, _ = plot_confusion_matrices(
            [0, 0, 1],
            [0, 1, 1],
            class_labels=[0, 1],
            label_map={0: "A", 1: "B"},
        )

        count_texts = [text.get_text() for text in axes[0].texts]
        fraction_texts = [text.get_text() for text in axes[1].texts]
        self.assertEqual(count_texts, ["1", "1", "1"])
        self.assertEqual(fraction_texts, ["0.50", "0.50", "1.00"])
        self.assertEqual(axes[1].texts[-1].get_color(), "white")
        fig.clf()


if __name__ == "__main__":
    unittest.main()
