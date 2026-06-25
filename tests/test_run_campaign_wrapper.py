from __future__ import annotations

import io
import unittest
from unittest import mock

import run_campaign


class RunCampaignWrapperTests(unittest.TestCase):
    def test_terminate_defaults_to_most_recent_live_campaign(self) -> None:
        live = [
            (
                "cnn",
                {"config": "configs/experiment_campaigns/cnn_campaign.yaml", "description": "cnn"},
                {"campaign_id": "cnn_pretrain_finetune", "running": True, "pid": 111, "state_mtime": 10.0},
            ),
            (
                "transformer",
                {"config": "configs/experiment_campaigns/transformer_campaign.yaml", "description": "transformer"},
                {"campaign_id": "transformer_pretrain_finetune", "running": True, "pid": 222, "state_mtime": 20.0},
            ),
        ]
        with mock.patch("run_campaign._known_live_campaigns", return_value=live), mock.patch(
            "run_campaign.campaign_main", return_value=0
        ) as campaign_main, mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_campaign.terminate_command([])
        self.assertEqual(code, 0)
        campaign_main.assert_called_once_with(
            [
                "terminate",
                "--campaign",
                "configs/experiment_campaigns/transformer_campaign.yaml",
                "--reason",
                "terminated by run_campaign CLI",
            ]
        )
        self.assertIn("selected live campaign transformer", stdout.getvalue())

    def test_terminate_without_live_campaign_is_noop(self) -> None:
        with mock.patch("run_campaign._known_live_campaigns", return_value=[]), mock.patch(
            "run_campaign.campaign_main"
        ) as campaign_main, mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_campaign.terminate_command([])
        self.assertEqual(code, 0)
        campaign_main.assert_not_called()
        self.assertIn("no running campaign found", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
