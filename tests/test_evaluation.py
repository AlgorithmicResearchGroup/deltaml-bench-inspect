from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from deltamlbench_inspect.evaluation import (
    EvaluationPolicyError,
    evaluation_policy,
    format_evaluation_contract,
    load_evaluation_policies,
    score_evaluator_result,
)
from deltamlbench_inspect.runtime import (
    discover_pwc_specs,
    report_command,
    review_command,
    sample_files_for_task,
    score_command,
    setup_script_for_task,
)
from deltamlbench_inspect.scorer_driver import run_scorer
from deltamlbench_inspect.report_collector import collect_report


class EvaluationPolicyTests(unittest.TestCase):
    def test_every_active_task_has_a_policy(self) -> None:
        active = {spec.name for spec in discover_pwc_specs()}
        policies = set(load_evaluation_policies())
        self.assertEqual(active, policies)
        self.assertEqual(35, len(active))

    def test_revised_contract_supersedes_legacy_scoring(self) -> None:
        policy = evaluation_policy("pwc_fashion_mnist_energize")
        assert policy is not None
        contract = format_evaluation_contract(
            "pwc_fashion_mnist_energize", policy
        )
        self.assertIn("supersedes earlier scoring", contract)
        self.assertIn("power_consumption", contract)
        self.assertIn("`solution.evaluate()` must compute and return", contract)
        self.assertIn("submission/evidence.json", contract)

    def test_weighted_error_improvements(self) -> None:
        policy = evaluation_policy("pwc_electricity_192_cyclenet")
        assert policy is not None
        result = score_evaluator_result(
            policy,
            {"mse": 0.1296, "mae": 0.2133},
        )
        self.assertAlmostEqual(10.0, result["score"])
        self.assertEqual("scored", result["status"])

    def test_constraint_is_fail_closed(self) -> None:
        policy = evaluation_policy("pwc_fashion_mnist_energize")
        assert policy is not None
        missing = score_evaluator_result(policy, {"accuracy": 0.92})
        self.assertEqual(0.0, missing["score"])
        self.assertEqual("constraint_failed", missing["status"])

        passing = score_evaluator_result(
            policy,
            {"accuracy": 0.92},
            {"power_consumption": 70.0},
        )
        self.assertGreater(passing["score"], 0.0)

    def test_all_metrics_must_improve(self) -> None:
        policy = evaluation_policy("pwc_kvasir_seg_yolo_sam_2")
        assert policy is not None
        result = score_evaluator_result(
            policy,
            {"f1_score": 0.91, "mean_dice": 0.87, "miou": 0.764},
        )
        self.assertEqual(0.0, result["score"])

    def test_out_of_bounds_metric_is_rejected(self) -> None:
        policy = evaluation_policy("pwc_cifar_100_pro_dsc")
        assert policy is not None
        with self.assertRaises(EvaluationPolicyError):
            score_evaluator_result(policy, {"accuracy": 1.1, "nmi": 0.9})

    def test_policy_status_does_not_replace_integrity_gate(self) -> None:
        policy = {
            "status": "calibration_required",
            "aggregation": "relative_improvement",
            "metrics": {"accuracy": {"baseline": 80.0, "direction": "maximize", "weight": 1.0, "bounds": [0, 100]}},
            "required_outputs": ["predictions"],
        }
        result = score_evaluator_result(policy, {"accuracy": 84.0})
        self.assertEqual("scored", result["status"])
        self.assertEqual(5.0, result["score"])

    def test_report_runs_unprivileged_before_root_owned_scorer(self) -> None:
        self.assertIn("/usr/local/bin/deltamlbench_report.py", report_command())
        command = score_command(visible_score=False, scorer_owned=True)
        self.assertIn("/root/deltamlbench/scorer_driver.py", command)
        self.assertIn("--strip-score-information", command)
        self.assertNotIn("/home/agent/score.py", command)

    def test_setup_removes_protected_staging_files(self) -> None:
        spec = next(
            spec
            for spec in discover_pwc_specs()
            if spec.name == "pwc_california_housing_binary_diffusion"
        )
        setup = setup_script_for_task(spec)
        self.assertIn("rm -rf /tmp/deltamlbench_assets/shared", setup)
        self.assertIn("/usr/local/bin/deltamlbench_report.py", setup)
        self.assertIn("chmod -R go-rwx /root/deltamlbench", setup)
        self.assertIn("/root/deltamlbench/review_bundle.py", setup)
        self.assertIn("baseline_hashes.json", setup)
        self.assertNotIn("cp -R /tmp/deltamlbench_assets/anti_cheat_validation", setup)
        self.assertIn("--task-name pwc_cnn", review_command("pwc_cnn"))
        staged = sample_files_for_task(spec)
        self.assertFalse(any("anti_cheat_validation" in path for path in staged))

    def test_california_housing_disables_wandb_authentication(self) -> None:
        spec = next(
            spec
            for spec in discover_pwc_specs()
            if spec.name == "pwc_california_housing_binary_diffusion"
        )
        setup = setup_script_for_task(spec)
        self.assertIn("/home/agent/.config/wandb/settings", setup)
        self.assertIn("mode = disabled", setup)

    def test_active_tasks_use_canonical_hugging_face_bucket(self) -> None:
        specs = discover_pwc_specs()
        commands = [
            command
            for spec in specs
            for step in spec.build_steps
            for command in step.get("commands", [])
        ]
        self.assertFalse(any("benchmarking_tasks_repos" in command for command in commands))
        downloads = [command for command in commands if "hf buckets cp" in command]
        self.assertEqual(3 * len(specs), len(downloads))
        self.assertTrue(
            all("hf://buckets/AlgorithmicResearchGroup/dmlb/" in command for command in downloads)
        )


class ScorerDriverTests(unittest.TestCase):
    def _workspace(self, root: Path, metric: float = 83.0, *, literal: bool = False):
        policy = evaluation_policy("pwc_chameleon_coed")
        assert policy is not None
        workspace = root / "workspace"
        solution_dir = workspace / "solution"
        submission = workspace / "submission"
        solution_dir.mkdir(parents=True)
        submission.mkdir()
        source = (
            f"def evaluate():\n    return {{'accuracy': {metric}}}\n"
            if literal
            else f"def evaluate():\n    measured = 80.0 + {metric - 80.0}\n    return {{'accuracy': measured}}\n"
        )
        solution = solution_dir / "solution.py"
        solution.write_text(source, encoding="utf-8")
        training_log = solution_dir / "training_log.json"
        training_log.write_text(json.dumps({"final_metrics": {"accuracy": metric}}), encoding="utf-8")
        evidence = submission / "evidence.json"
        evidence.write_text(
            json.dumps(
                {
                    "version": 1,
                    "final_metrics": {"accuracy": metric},
                    "artifacts": ["solution/solution.py", "solution/training_log.json"],
                }
            ),
            encoding="utf-8",
        )
        report = submission / "report.json"
        report.write_text(
            json.dumps(
                {
                    "version": 1,
                    "source": "solution.evaluate",
                    "solution_sha256": hashlib.sha256(solution.read_bytes()).hexdigest(),
                    "metrics": {"accuracy": metric},
                    "measurements": {},
                }
            ),
            encoding="utf-8",
        )
        policy_path = root / "policy.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        return policy_path, report, evidence, workspace, solution

    def test_missing_report_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy_path, _, evidence, workspace, solution = self._workspace(root)
            result = run_scorer(policy_path, root / "missing.json", evidence, workspace, solution)
        self.assertEqual("report_missing", result["status"])
        self.assertEqual(0.0, result["score"])

    def test_integrity_validated_report_is_compared_with_paper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._workspace(root)
            result = run_scorer(*args)
        self.assertEqual("scored", result["status"])
        self.assertGreater(result["score"], 0.0)
        self.assertEqual("passed", result["integrity"]["status"])
        self.assertEqual({"accuracy": 83.0}, result["reported_metrics"])

    def test_literal_metric_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_scorer(*self._workspace(root, literal=True))
        self.assertEqual("integrity_failed", result["status"])
        self.assertTrue(any("literal metric" in issue for issue in result["integrity"]["issues"]))

    def test_evidence_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = list(self._workspace(root))
            evidence = args[2]
            data = json.loads(evidence.read_text(encoding="utf-8"))
            data["artifacts"] = ["../outside.json"]
            evidence.write_text(json.dumps(data), encoding="utf-8")
            result = run_scorer(*args)
        self.assertEqual("integrity_failed", result["status"])
        self.assertTrue(any("escapes workspace" in issue for issue in result["integrity"]["issues"]))

    def test_evidence_cannot_traverse_internal_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = list(self._workspace(root))
            workspace = args[3]
            (workspace / "linked.json").symlink_to(workspace / "solution" / "training_log.json")
            evidence = args[2]
            data = json.loads(evidence.read_text(encoding="utf-8"))
            data["artifacts"] = ["solution/solution.py", "linked.json"]
            evidence.write_text(json.dumps(data), encoding="utf-8")
            result = run_scorer(*args)
        self.assertEqual("integrity_failed", result["status"])
        self.assertTrue(any("symbolic link" in issue for issue in result["integrity"]["issues"]))

    def test_collector_runs_solution_and_records_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solution = root / "solution.py"
            solution.write_text("def evaluate():\n    value = 40 + 2\n    return {'accuracy': value}\n", encoding="utf-8")
            expected_hash = hashlib.sha256(solution.read_bytes()).hexdigest()
            report = collect_report(solution, root / "submission" / "report.json")
        self.assertEqual({"accuracy": 42.0}, report["metrics"])
        self.assertEqual(expected_hash, report["solution_sha256"])


if __name__ == "__main__":
    unittest.main()
