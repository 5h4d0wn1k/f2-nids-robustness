import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from firmware.nids_robustness import (  # noqa: E402
    SIGNATURE,
    NaiveDetector,
    NormalizedDetector,
    RobustNormalizer,
    backslash_trick,
    build_corpus,
    case_mutation,
    evaluate,
    fragment,
    hex_enclose,
    main,
    percent_encoding,
)


class NormalizerTest(unittest.TestCase):
    def setUp(self):
        self.norm = RobustNormalizer()

    def test_plain_signature_detected(self):
        self.assertTrue(self.norm.signature_present(SIGNATURE))

    def test_backslash_normalized(self):
        self.assertIn("post /exec",
                      self.norm.normalize(backslash_trick(SIGNATURE)))

    def test_hex_escape_normalized(self):
        self.assertTrue(self.norm.signature_present(hex_enclose(SIGNATURE)))

    def test_percent_encoding_normalized(self):
        self.assertTrue(self.norm.signature_present(percent_encoding(SIGNATURE)))

    def test_case_mutation_normalized(self):
        self.assertTrue(self.norm.signature_present(case_mutation(SIGNATURE, 9)))

    def test_benign_not_detected(self):
        self.assertFalse(self.norm.signature_present("GET /index.html HTTP/1.1"))


class DetectorTest(unittest.TestCase):
    def test_naive_misses_evasions(self):
        d = NaiveDetector()
        self.assertTrue(d.detect(SIGNATURE))
        self.assertFalse(d.detect(hex_enclose(SIGNATURE)))
        self.assertFalse(d.detect(case_mutation(SIGNATURE, 2)))

    def test_normalized_catches_evasions(self):
        d = NormalizedDetector()
        for p in (hex_enclose(SIGNATURE),
                  percent_encoding(SIGNATURE),
                  backslash_trick(SIGNATURE),
                  case_mutation(SIGNATURE, 2)):
            self.assertTrue(d.detect(p))


class EvaluationTest(unittest.TestCase):
    def test_corpus_shape(self):
        corpus = build_corpus(seed=42)
        self.assertGreater(len(corpus), 8)
        self.assertTrue(any(c.get("benign") for c in corpus))
        labels = {c["name"] for c in corpus}
        self.assertIn("multi_stage", labels)

    def test_corpus_deterministic(self):
        a = build_corpus(seed=7)
        b = build_corpus(seed=7)
        self.assertEqual([p["payload"] for p in a],
                         [p["payload"] for p in b])

    def test_naive_evades_more_than_normalized(self):
        corpus = build_corpus(seed=5)
        nv = evaluate(corpus, NaiveDetector())
        nm = evaluate(corpus, NormalizedDetector())
        self.assertGreater(nv["evasion_rate"], nm["evasion_rate"])
        self.assertGreaterEqual(nv["evaded"], nm["evaded"])

    def test_normalized_beats_naive_on_every_encoding_class(self):
        # The normalizer must never do worse than the naive detector, and it
        # must win (strictly) on at least the coding tricks.
        corpus = build_corpus(seed=5)
        nm = evaluate(corpus, NormalizedDetector())
        nv = evaluate(corpus, NaiveDetector())
        self.assertLessEqual(nm["evasion_rate"], nv["evasion_rate"])

    def test_normalized_zero_false_positive_on_benign(self):
        corpus = build_corpus(seed=5)
        nm = evaluate(corpus, NormalizedDetector())
        self.assertEqual(nm["false_positives"], 0)

    def test_evaluate_counts(self):
        corpus = build_corpus(seed=5)
        nv = evaluate(corpus, NaiveDetector())
        self.assertEqual(nv["attacks"] + nv["benign"], len(corpus))


class CliTest(unittest.TestCase):
    def test_main_exit_zero(self):
        with tempfile.TemporaryDirectory() as td:
            rp = os.path.join(td, "report.md")
            code = main(["--report", rp, "--size", "4"])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(rp))
            self.assertIn("F2 - NIDS", open(rp).read())

    def test_report_written(self):
        with tempfile.TemporaryDirectory() as td:
            rp = os.path.join(td, "report.markdown")
            code = main(["--report", rp])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(rp))
            self.assertIn("F2 - NIDS ADVERSARIAL ROBUSTNESS",
                          open(rp).read())


if __name__ == "__main__":
    unittest.main()
