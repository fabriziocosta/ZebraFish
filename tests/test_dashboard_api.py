from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.dashboard_api import app


class DashboardApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_is_read_only_service_metadata(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_investigation_returns_normalized_live_payload(self) -> None:
        response = self.client.get("/api/investigation/cnn?level=2&relation_depth=1")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["schema_version"], 1)
        self.assertIn("active_hypothesis", body)
        self.assertIn("current_experiment", body)
        self.assertIn("metric_display", body["current_experiment"])
        self.assertIn("compute", body["current_experiment"])
        self.assertIn("registration_status", body["expected_outcomes"])
        self.assertIn("unclassified", body["evidence"])
        self.assertIn("health", body)
        self.assertIn("graph", body)

    def test_unknown_campaign_and_entity_are_errors(self) -> None:
        self.assertEqual(self.client.get("/api/investigation/not-a-campaign").status_code, 404)
        self.assertEqual(self.client.get("/api/investigation/cnn/entities/not-an-entity").status_code, 404)

    def test_graph_endpoint_is_read_only_and_focused(self) -> None:
        response = self.client.get("/api/investigation/cnn/graph?level=2&relation_depth=0")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("nodes", body)
        self.assertIn("edges", body)
        self.assertIn("svg", body)


if __name__ == "__main__":
    unittest.main()
