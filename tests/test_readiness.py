from __future__ import annotations

import contextlib
import io
import unittest

from deltamlbench_inspect.cli import main
from deltamlbench_inspect.readiness import build_readiness_report, registered_task


class ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_readiness_report(check_runtime=False)

    def test_suite_contract_is_structurally_valid(self) -> None:
        self.assertTrue(self.report["platform_ready"])
        self.assertEqual(35, self.report["suite"]["families"])
        self.assertEqual(70, self.report["suite"]["variants"])
        self.assertEqual(35, self.report["suite"]["policies"])

    def test_release_gate_uses_report_and_integrity_contracts(self) -> None:
        self.assertTrue(self.report["production_ready"])
        self.assertEqual(35, self.report["suite"]["reported_metric_contracts"])
        self.assertEqual(35, self.report["suite"]["judge_review_policies"])
        self.assertEqual(35, self.report["suite"]["release_ready_families"])

    def test_registered_task_parser_handles_suffixes(self) -> None:
        match = registered_task(self.report, "pwc_cnn_hidden_score")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual("pwc_cnn", match[0]["name"])
        self.assertEqual("hidden_score", match[1])

    def test_ready_task_is_accepted(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = main(["check-task", "pwc_cnn_main"])
        self.assertEqual(0, result)
        self.assertIn("Task accepted", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
