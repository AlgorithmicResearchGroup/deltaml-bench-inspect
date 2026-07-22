from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deltamlbench_inspect.integrity_judge import (
    apply_judge_verdict,
    build_judge_prompt,
    parse_judge_response,
    run_integrity_judge,
)
from deltamlbench_inspect.review_bundle import build_review_bundle, snapshot_tree


class IntegrityJudgeTests(unittest.IsolatedAsyncioTestCase):
    def test_parses_structured_fenced_verdict(self) -> None:
        result = parse_judge_response(
            '```json\n{"verdict":"pass","confidence":0.9,"summary":"supported","violations":[]}\n```'
        )
        self.assertEqual("pass", result["verdict"])
        self.assertEqual(0.9, result["confidence"])

    def test_parser_tolerates_trailing_text_but_escalates_weak_pass(self) -> None:
        result = parse_judge_response(
            '{"verdict":"pass","confidence":0.0,"summary":"maybe","violations":[]}} trailing'
        )
        self.assertEqual("escalate", result["verdict"])

    def test_nonpassing_verdict_fails_closed(self) -> None:
        result = apply_judge_verdict(
            {"score": 12.5, "status": "scored"},
            {"verdict": "escalate", "summary": "ambiguous"},
        )
        self.assertEqual(0.0, result["score"])
        self.assertEqual(12.5, result["unjudged_score"])
        self.assertEqual("integrity_review_required", result["status"])

    def test_prompt_redacts_secret_like_values(self) -> None:
        prompt = build_judge_prompt(
            task_instructions="use API_KEY=super-secret-value",
            messages=[],
            review_bundle={"authorization": "Bearer abcdefghijklmnop"},
            deterministic_audit={},
            execution_events=[],
        )
        self.assertNotIn("super-secret-value", prompt)
        self.assertNotIn("abcdefghijklmnop", prompt)
        self.assertIn("[REDACTED]", prompt)

    def test_review_bundle_contains_source_and_declared_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            solution_dir = workspace / "solution"
            submission = workspace / "submission"
            solution_dir.mkdir()
            submission.mkdir()
            solution = solution_dir / "solution.py"
            solution.write_text("def evaluate():\n    return result\n", encoding="utf-8")
            baseline_path = workspace / "baseline.json"
            baseline_path.write_text(json.dumps(snapshot_tree(solution_dir)), encoding="utf-8")
            helper = solution_dir / "helper.py"
            helper.write_text("IMPROVED = True\n", encoding="utf-8")
            log = solution_dir / "training.json"
            log.write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")
            report = submission / "report.json"
            report.write_text(json.dumps({"version": 1, "metrics": {"accuracy": 0.9}}), encoding="utf-8")
            evidence = submission / "evidence.json"
            evidence.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "final_metrics": {"accuracy": 0.9},
                        "artifacts": ["solution/solution.py", "solution/training.json"],
                    }
                ),
                encoding="utf-8",
            )
            bundle = build_review_bundle(
                task_name="example",
                workspace=workspace,
                solution_path=solution,
                report_path=report,
                evidence_path=evidence,
                baseline_manifest_path=baseline_path,
            )
        self.assertIn("def evaluate", bundle["solution"]["content"])
        self.assertEqual(2, len(bundle["artifacts"]))
        self.assertEqual("solution/helper.py", bundle["changed_solution_files"][0]["path"])

    async def test_model_judge_is_invoked_and_parsed(self) -> None:
        class FakeJudge:
            async def generate(self, prompt, config):
                self.prompt = prompt
                return SimpleNamespace(
                    completion=json.dumps(
                        {
                            "verdict": "pass",
                            "confidence": 0.8,
                            "summary": "supported",
                            "violations": [],
                        }
                    )
                )

        fake = FakeJudge()
        with patch.dict(os.environ, {"DELTAML_JUDGE_MODEL": "mock/judge"}, clear=True):
            with patch("deltamlbench_inspect.integrity_judge.get_model", return_value=fake):
                result = await run_integrity_judge(
                    task_instructions="task",
                    messages=[],
                    review_bundle={},
                    deterministic_audit={"status": "passed"},
                    execution_events=[],
                )
        self.assertEqual("pass", result["verdict"])
        self.assertEqual("mock/judge", result["model"])


if __name__ == "__main__":
    unittest.main()
