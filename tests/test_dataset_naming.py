from __future__ import annotations

import unittest

from src.dataset_naming import build_mechanism_filename_prefix


class DatasetNamingTests(unittest.TestCase):
    def test_build_mechanism_filename_prefix(self) -> None:
        selected_mechanisms = [
            "GABAAR_Antagonist",
            "Kv_Blocker_NonSelective",
            "NMDAR_Activation",
            "NMDAR_Antagonist",
        ]

        prefix = build_mechanism_filename_prefix(selected_mechanisms)

        self.assertEqual(prefix, "GA_An_Kv_Bl_NM_Ac_NM_An")


if __name__ == "__main__":
    unittest.main()
