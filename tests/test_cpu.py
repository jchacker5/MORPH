"""CPU regressions; no MLX import, model training, or network required."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from audit.fresh import ROOT, compare, table
from joint.infer import WEIGHTS, predict
from joint.replay_cpu import dense_reference, examples, replay
from lm.generate import generate, validate_inputs


class ReplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.original = Path(self.temp.name) / "original"
        self.fresh = Path(self.temp.name) / "fresh"
        self.spec = [("sample.csv", ["seed", "cell"], ["accuracy"])]
        self.write(self.original, "seed,cell,accuracy\n1,GM,0.5\n2,GM,0.9\n")

    def write(self, folder, text):
        (folder / "results").mkdir(parents=True, exist_ok=True)
        (folder / "results/sample.csv").write_text(text)

    def test_order_does_not_change_comparison(self):
        self.write(self.fresh, "seed,cell,accuracy\n2,GM,0.9000001\n1,GM,0.5\n")
        result = compare(self.original, self.fresh, self.spec)["sample.csv"]
        self.assertEqual(result["rows"], 2)
        self.assertLess(result["max_abs_difference"]["accuracy"], 1e-6)

    def test_nonfinite_and_invalid_metrics_are_rejected_on_either_side(self):
        for bad in ("nan", "inf", "-inf", "", "not-a-number"):
            for side in (self.original, self.fresh):
                with self.subTest(value=bad, side=side.name):
                    for folder in (self.original, self.fresh):
                        self.write(folder, "seed,cell,accuracy\n1,GM,0.5\n")
                    self.write(side, f"seed,cell,accuracy\n1,GM,{bad}\n")
                    with self.assertRaises(ValueError):
                        compare(self.original, self.fresh, self.spec)

    def test_duplicate_empty_and_malformed_tables_are_rejected(self):
        cases = [
            "seed,cell,accuracy\n1,GM,0.5\n1,GM,0.9\n",
            "seed,cell,accuracy\n", "", "seed,accuracy\n1,0.5\n",
            "seed,cell,accuracy,accuracy\n1,GM,0.5,0.9\n",
            "seed,cell,accuracy\n1,GM\n", "seed,cell,accuracy\n1,GM,0.5,extra\n",
            "seed,cell,accuracy\n,GM,0.5\n",
        ]
        for case in cases:
            with self.subTest(csv=case):
                self.write(self.fresh, case)
                with self.assertRaises(ValueError):
                    table(self.fresh, "sample.csv", ["seed", "cell"])

    def test_missing_extra_and_changed_records_fail(self):
        for content in ("1,GM,0.5\n", "1,GM,0.5\n2,GM,0.9\n3,GM,0.5\n", "1,GM,0.5\n2,GM,0.91\n"):
            with self.subTest(content=content):
                self.write(self.fresh, "seed,cell,accuracy\n" + content)
                with self.assertRaises(ValueError):
                    compare(self.original, self.fresh, self.spec)

    def test_missing_metric_fails_with_context(self):
        self.write(self.fresh, "seed,cell\n1,GM\n2,GM\n")
        with self.assertRaisesRegex(ValueError, "sample.csv.*accuracy"):
            compare(self.original, self.fresh, self.spec)

    def test_published_tables_are_well_formed(self):
        result = compare(ROOT, ROOT)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(all(v == 0 for v in r["max_abs_difference"].values()) for r in result.values()))


class GenerationTests(unittest.TestCase):
    def test_invalid_generation_inputs_fail_before_loading_model(self):
        cases = [("", 1, 1), (None, 1, 1), ("Hi", -1, 1), ("Hi", 1.5, 1),
                 ("Hi", float("nan"), 1), ("Hi", float("inf"), 1), ("Hi", True, 1),
                 ("Hi", 1, -1), ("Hi", 1, float("nan")), ("Hi", 1, float("inf")),
                 ("Hi", 1, float("-inf")), ("Hi", 1, "1"), ("Hi", 1, True)]
        for prompt, length, temperature in cases:
            with self.subTest(prompt=prompt, length=length, temperature=temperature):
                with self.assertRaises(ValueError):
                    generate(Path("nonexistent-checkpoint"), prompt, length, temperature, 73)

    def test_zero_length_returns_utf8_prompt_without_loading_model(self):
        self.assertEqual(generate(None, "café", 0, 0, 73), "café".encode())

    def test_greedy_and_positive_temperature_are_allowed(self):
        for temperature in (0, .8, 1):
            validate_inputs("Hi", 10, temperature)

    def test_cli_reports_invalid_inputs_without_mlx(self):
        for args in (["--temperature", "nan"], ["--temperature", "inf"], ["--bytes", "-1"], ["--prompt", ""]):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, str(ROOT / "lm/generate.py"), *args], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(WEIGHTS, allow_pickle=False) as archive:
            cls.weights = dict(archive)

    def test_documented_example(self):
        self.assertEqual(predict(self.weights, [2, 0, 0, 0], [0, 1, 2, 3, 2, 0, 3, 1], 1),
                         {"class": 1, "retained_bank": 1, "pair_layout": 1, "expert": 1, "recurrent_steps": 2})

    def test_selected_paths_match_independent_dense_reference(self):
        header, tokens, query, _ = examples(90210, 512)
        results = [predict(self.weights, h, t, q) for h, t, q in zip(header, tokens, query)]
        np.testing.assert_array_equal([r["class"] for r in results], dense_reference(self.weights, header, tokens, query))
        self.assertEqual({(r["pair_layout"], r["retained_bank"], r["expert"]) for r in results},
                         {(g, m, q) for g in (0, 1) for m in (0, 1) for q in (0, 1)})

    def test_invalid_classifier_inputs(self):
        good_header, good_tokens = [2, 0, 0, 0], [0, 1, 2, 3, 2, 0, 3, 1]
        for header, tokens, query in ([[], good_tokens, 1], [[np.nan, 0, 0, 0], good_tokens, 1],
                                      [[np.inf, 0, 0, 0], good_tokens, 1], [good_header, [], 1],
                                      [good_header, [0.0] * 8, 1], [good_header, [-1] * 8, 1],
                                      [good_header, [4] * 8, 1], [good_header, good_tokens, 2],
                                      [good_header, good_tokens, .5]):
            with self.subTest(header=header, tokens=tokens, query=query):
                with self.assertRaises(ValueError):
                    predict(self.weights, header, tokens, query)

    def test_frozen_ten_thousand_example_replay(self):
        result = replay()
        self.assertEqual((result["correct"], result["examples"], result["dense_reference_matches"]), (9999, 10000, 128))


if __name__ == "__main__":
    unittest.main()
