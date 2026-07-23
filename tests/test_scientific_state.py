from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.scientific_state import (
    ImmutableEntityError,
    OperationConflictError,
    apply_operations,
    empty_state,
    load_state,
    record_entity,
    save_state,
)


class ScientificStateTests(unittest.TestCase):
    def test_atomic_round_trip_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.yaml"
            state = empty_state()
            state = record_entity(
                state,
                "hypotheses",
                "hyp_1",
                {"statement": "bounded change helps", "status": "active"},
            )
            save_state(path, state)
            loaded = load_state(path)
            self.assertEqual(loaded["entities"]["hypotheses"]["hyp_1"]["statement"], "bounded change helps")
            self.assertTrue(loaded["audit_log"])

    def test_update_requires_current_expected_value(self) -> None:
        state = record_entity(empty_state(), "hypotheses", "hyp_1", {"status": "active"})
        with self.assertRaises(OperationConflictError):
            apply_operations(
                state,
                [{"operation": "update", "path": "entities.hypotheses.hyp_1.status", "expected_old": "closed", "value": "active"}],
            )

    def test_terminal_experiment_is_immutable(self) -> None:
        state = record_entity(
            empty_state(),
            "experiments",
            "exp_1",
            {"status": "completed", "execution": {"provenance": {"source": "test"}}},
        )
        with self.assertRaises(ImmutableEntityError):
            apply_operations(
                state,
                [{"operation": "update", "path": "entities.experiments.exp_1.status", "value": "failed"}],
            )

    def test_relation_vocabulary_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            apply_operations(
                empty_state(),
                [{"operation": "relation", "value": {"type": "invented", "source": "a", "target": "b"}}],
            )


if __name__ == "__main__":
    unittest.main()
