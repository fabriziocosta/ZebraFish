from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.campaign_watchdog import _process_identity, inspect_campaign


class CampaignWatchdogTests(unittest.TestCase):
    def test_live_proc_argv_hash_uses_launch_representation(self) -> None:
        proc_cmdline = Path(f"/proc/{os.getpid()}/cmdline").read_bytes()
        expected = hashlib.sha256(b"\x00".join(part for part in proc_cmdline.split(b"\x00") if part)).hexdigest()
        self.assertEqual(_process_identity(os.getpid())["command_hash"], expected)

    def test_matching_process_identity_is_not_a_mismatch(self) -> None:
        command = ["/venv/bin/python", "runner.py", "--config", "config.yaml"]
        command_hash = hashlib.sha256("\x00".join(command).encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "campaign_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "active_launch_state": {
                            "pid": 123,
                            "command_hash": command_hash,
                            "process_start_ticks": "456",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch("src.campaign_watchdog._running", return_value=True), patch(
                "src.campaign_watchdog._process_identity",
                return_value={"pid": 123, "running": True, "command_hash": command_hash, "process_start_ticks": "456"},
            ):
                snapshot = inspect_campaign({}, campaign_state_path=state_path)
        self.assertFalse(snapshot["process_identity_mismatch"])
        self.assertEqual(snapshot["triggers"], [])


if __name__ == "__main__":
    unittest.main()
