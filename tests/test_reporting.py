from __future__ import annotations

import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from src.training.reporting import plot_embedding_projection, plot_training_history


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

        self.assertEqual(len(ax.lines), 4)
        self.assertEqual([line.get_label() for line in ax.lines], ["Train", "_nolegend_", "Val", "_nolegend_"])
        self.assertEqual(ax.lines[0].get_color(), ax.lines[1].get_color())
        self.assertEqual(ax.lines[2].get_color(), ax.lines[3].get_color())
        self.assertAlmostEqual(ax.lines[0].get_alpha(), 0.3)
        self.assertAlmostEqual(ax.lines[1].get_alpha(), 0.9)
        self.assertAlmostEqual(ax.lines[1].get_linewidth(), 3.0)
        self.assertEqual(ax.lines[0].get_marker(), "None")
        self.assertEqual(ax.lines[2].get_marker(), "None")
        fig.clf()

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


if __name__ == "__main__":
    unittest.main()
