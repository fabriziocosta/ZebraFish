from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.domain_guidance import (
    DomainGuidanceError,
    aligned_evaluation_metadata,
    aggregate_domain_evaluations,
    build_live_domain_diagnostic,
    calibrate_domain_baseline,
    domain_evaluation_observations,
    evaluate_domain_guidance,
    load_domain_contract,
    persist_domain_evaluation,
    persist_live_domain_diagnostic,
    save_domain_calibration,
    validate_domain_contract,
)


CONTRACT_PATH = Path("configs/domain_guidance/cnn_action_domain_v1.yaml")


def synthetic_domain_data(*, collapsed: bool = False, sparse: bool = False):
    labels = [
        "GABAAR_Antagonist",
        "NMDAR_Activation",
        "AChE_Inhibitor_Reversible",
        "mAChR_Agonist_NonSelective",
    ]
    centers = {
        labels[0]: np.asarray([0.0, 0.0, 0.0]),
        labels[1]: np.asarray([0.5, 0.0, 0.0]),
        labels[2]: np.asarray([5.0, 0.0, 0.0]),
        labels[3]: np.asarray([5.5, 0.0, 0.0]),
    }
    rows = []
    features = []
    truth = []
    predictions = []
    rng = np.random.default_rng(44)
    for label_index, label in enumerate(labels):
        compound_count = 1 if sparse and label == labels[-1] else 2
        for compound_index in range(compound_count):
            compound = f"compound_{label_index}_{compound_index}"
            for run_index in range(2):
                run = f"{compound}_run_{run_index}"
                for frame_index in range(3):
                    rows.append(
                        {
                            "compound": compound,
                            "experimental_run_id": run,
                            "image_condition_dir": f"/data/{run}/condition_{frame_index}",
                            "concentration_band": "high",
                            "is_control": False,
                        }
                    )
                    features.append(centers[label] + rng.normal(0.0, 0.04, size=3))
                    truth.append(label)
                    predictions.append(labels[2] if collapsed else label)
    return np.asarray(features), pd.DataFrame(rows), truth, predictions


class DomainGuidanceTests(unittest.TestCase):
    def contract(self):
        contract = load_domain_contract(CONTRACT_PATH)
        contract["evaluation"]["bootstrap_iterations"] = 0
        return contract

    def evaluate(
        self,
        experiment_id: str,
        *,
        collapsed: bool = False,
        sparse: bool = False,
        protocol: str = "three_seed_replicate_lockbox_v1",
        calibration=None,
        training_seed: int | None = None,
    ):
        features, metadata, truth, predictions = synthetic_domain_data(collapsed=collapsed, sparse=sparse)
        return evaluate_domain_guidance(
            experiment_id=experiment_id,
            latent_features=features,
            metadata=metadata,
            y_true=truth,
            y_pred=predictions,
            contract=self.contract(),
            split_hash="split-fixed",
            evaluation_protocol=protocol,
            calibration_profile=calibration,
            training_seed=training_seed,
            created_at="2026-01-01T00:00:00+00:00",
        )

    def test_contract_validation_and_hashing(self) -> None:
        contract = self.contract()
        self.assertEqual(contract["id"], "cnn_action_domain_v1")
        self.assertEqual(len(contract["_hash"]), 64)
        invalid = deepcopy(contract)
        invalid["constraints"][0]["role"] = "optional_opinion"
        with self.assertRaises(DomainGuidanceError):
            validate_domain_contract(invalid)

    def test_metadata_alignment_adds_run_identity_and_probabilities(self) -> None:
        metadata = pd.DataFrame({"compound": ["a"], "image_condition_dir": ["/root/run-1/high/F1"]})
        aligned = aligned_evaluation_metadata(
            metadata,
            y_true=["A"],
            y_pred=["A"],
            probabilities=np.asarray([[0.75, 0.25]]),
            probability_labels=["A", "B"],
        )
        self.assertEqual(aligned.loc[0, "experimental_run_id"], "high")
        self.assertEqual(aligned.loc[0, "proba_A"], 0.75)
        with self.assertRaises(DomainGuidanceError):
            aligned_evaluation_metadata(metadata, y_true=["A", "B"], y_pred=["A"])

    def test_original_latent_geometry_and_classification_are_measured(self) -> None:
        report = self.evaluate("exp-1")
        constraints = {item["id"]: item for item in report["constraints"]}
        ache = constraints["ache_machr_separability"]
        self.assertEqual(ache["status"], "calibrating")
        values = {item["metric"]: item["value"] for item in ache["checks"]}
        self.assertAlmostEqual(values["pairwise_balanced_accuracy"], 1.0)
        self.assertIsNotNone(values["leave_one_compound_out_balanced_accuracy"])
        geometry = constraints["gaba_nmda_related_geometry"]
        geometry_values = {item["metric"]: item["value"] for item in geometry["checks"]}
        self.assertIsNotNone(geometry_values["normalized_centroid_distance"])
        self.assertGreaterEqual(geometry_values["related_distance_rank_score"], 0.0)
        self.assertGreaterEqual(geometry_values["neighbourhood_purity"], 0.0)
        self.assertFalse(report["umap_used_for_decision"])

    def test_bootstrap_is_deterministic(self) -> None:
        contract = self.contract()
        contract["evaluation"]["bootstrap_iterations"] = 20
        features, metadata, truth, predictions = synthetic_domain_data()
        arguments = {
            "experiment_id": "same",
            "latent_features": features,
            "metadata": metadata,
            "y_true": truth,
            "y_pred": predictions,
            "contract": contract,
            "split_hash": "split-fixed",
            "evaluation_protocol": "three_seed_replicate_lockbox_v1",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        left = evaluate_domain_guidance(**arguments)
        right = evaluate_domain_guidance(**arguments)
        self.assertEqual(left["metrics"], right["metrics"])

    def test_sparse_class_support_is_unresolved(self) -> None:
        report = self.evaluate("sparse", sparse=True)
        constraint = next(item for item in report["constraints"] if item["id"] == "ache_machr_separability")
        loco = next(item for item in constraint["checks"] if item["metric"] == "leave_one_compound_out_balanced_accuracy")
        self.assertIsNone(loco["value"])
        self.assertEqual(constraint["status"], "unresolved")
        self.assertFalse(constraint["support"]["sufficient"])
        self.assertEqual(
            constraint["support"]["by_class"]["mAChR_Agonist_NonSelective"]["compounds"],
            1,
        )

    def test_three_seed_baseline_freezes_and_enables_relative_decisions(self) -> None:
        reports = [
            self.evaluate(f"baseline-{seed}", training_seed=seed)
            for seed in range(3)
        ]
        calibration = calibrate_domain_baseline(
            reports,
            contract=self.contract(),
            candidate_family_id="baseline-family",
            created_at="2026-01-02T00:00:00+00:00",
        )
        self.assertEqual(calibration["status"], "frozen")
        self.assertEqual(calibration["replicate_count"], 3)
        candidate = self.evaluate("candidate", calibration=calibration, training_seed=0)
        self.assertEqual(candidate["objective_eligibility"], "eligible")
        self.assertTrue(all(item["status"] == "pass" for item in candidate["constraints"]))
        aggregate = aggregate_domain_evaluations(
            [
                self.evaluate(
                    f"candidate-{seed}",
                    calibration=calibration,
                    training_seed=seed,
                )
                for seed in range(3)
            ],
            calibration=calibration,
            contract=self.contract(),
        )
        self.assertEqual(aggregate["status"], "pass")
        self.assertTrue(aggregate["hard_guardrails_pass"])

    def test_prediction_collapse_fails_hard_guardrails(self) -> None:
        reports = [
            self.evaluate(f"baseline-{seed}", training_seed=seed)
            for seed in range(3)
        ]
        calibration = calibrate_domain_baseline(
            reports,
            contract=self.contract(),
            candidate_family_id="baseline-family",
        )
        collapsed = self.evaluate(
            "collapsed",
            collapsed=True,
            calibration=calibration,
            training_seed=0,
        )
        self.assertEqual(collapsed["objective_eligibility"], "hard_guardrail_failed")
        failed = [item["id"] for item in collapsed["constraints"] if item["status"] == "fail"]
        self.assertIn("action_identifiability", failed)
        self.assertIn("ache_machr_separability", failed)
        aggregate = aggregate_domain_evaluations(
            [
                self.evaluate(
                    f"collapsed-{seed}",
                    collapsed=True,
                    calibration=calibration,
                    training_seed=seed,
                )
                for seed in range(3)
            ],
            calibration=calibration,
            contract=self.contract(),
        )
        self.assertEqual(aggregate["status"], "fail")
        self.assertFalse(aggregate["hard_guardrails_pass"])

    def test_calibration_and_candidate_require_the_same_distinct_training_seeds(self) -> None:
        duplicate_seed_reports = [
            self.evaluate(f"baseline-{index}", training_seed=0)
            for index in range(3)
        ]
        with self.assertRaises(DomainGuidanceError):
            calibrate_domain_baseline(
                duplicate_seed_reports,
                contract=self.contract(),
                candidate_family_id="baseline-family",
            )
        baseline = [
            self.evaluate(f"baseline-{seed}", training_seed=seed)
            for seed in range(3)
        ]
        calibration = calibrate_domain_baseline(
            baseline,
            contract=self.contract(),
            candidate_family_id="baseline-family",
        )
        mismatched = [
            self.evaluate(
                f"candidate-{seed}",
                calibration=calibration,
                training_seed=seed,
            )
            for seed in (0, 1, 7)
        ]
        aggregate = aggregate_domain_evaluations(
            mismatched,
            calibration=calibration,
            contract=self.contract(),
        )
        self.assertEqual(aggregate["status"], "unresolved")
        self.assertFalse(aggregate["hard_guardrails_pass"])

    def test_legacy_results_remain_descriptive(self) -> None:
        report = self.evaluate("legacy", protocol="legacy_single_seed")
        self.assertEqual(report["objective_eligibility"], "legacy_descriptive")

    def test_observations_are_classified_without_using_projection(self) -> None:
        report = self.evaluate("exp")
        observations = domain_evaluation_observations(report)
        self.assertTrue(observations)
        self.assertTrue(all(item["direction"] == "inconclusive" for item in observations))
        self.assertTrue(all(item["detection"]["method"] == "deterministic_domain_contract" for item in observations))

    def test_live_validation_detects_collapse_without_becoming_termination_evidence(self) -> None:
        features, metadata, truth, predictions = synthetic_domain_data(collapsed=True)
        report = build_live_domain_diagnostic(
            experiment_id="running",
            epoch=20,
            latent_features=features,
            metadata=metadata,
            y_true=truth,
            y_pred=predictions,
            contract=self.contract(),
            split_hash="split-fixed",
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertTrue(report["live_triggers"])
        self.assertFalse(report["termination_eligible"])
        self.assertFalse(report["umap_used_for_decision"])
        with tempfile.TemporaryDirectory() as directory:
            paths = persist_live_domain_diagnostic(run_dir=directory, report=report)
            self.assertTrue(Path(paths["latest_path"]).exists())

    def test_artifacts_and_calibration_are_persisted_with_hashes(self) -> None:
        features, metadata, truth, predictions = synthetic_domain_data()
        aligned = aligned_evaluation_metadata(metadata, y_true=truth, y_pred=predictions)
        report = self.evaluate("persisted")
        with tempfile.TemporaryDirectory() as directory:
            diagnostic = Path(directory) / "action_confusion_counts.csv"
            diagnostic.write_text("true,pred,count\nA,A,1\n", encoding="utf-8")
            persisted = persist_domain_evaluation(
                run_dir=directory,
                report=report,
                latent_features=features,
                aligned_metadata=aligned,
                diagnostic_paths=[diagnostic],
            )
            self.assertTrue(Path(persisted["report_path"]).exists())
            loaded = json.loads(Path(persisted["report_path"]).read_text(encoding="utf-8"))
            self.assertTrue(loaded["artifacts"]["latent_embeddings"]["hash"])
            self.assertEqual(
                loaded["artifacts"]["diagnostics"][0]["decision_role"],
                "deterministic_evidence",
            )
            calibration = calibrate_domain_baseline(
                [
                    self.evaluate(f"baseline-{seed}", training_seed=seed)
                    for seed in range(3)
                ],
                contract=self.contract(),
                candidate_family_id="baseline-family",
            )
            output = save_domain_calibration(Path(directory) / "calibration.json", calibration)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
