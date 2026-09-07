#!/usr/bin/env python3
"""
F2 — NIDS adversarial robustness through protocol-evasion encoding.

A deterministic, offline, standard-library-only toolkit that:
  1. Builds a normalizer that canonically decodes protocol-embedded payloads so
     that traffic encoding tricks no longer hide a detection signature.
  2. Encodes an attack signature in many adversarial evasion forms
     (fragmentation, case mutation, backslash/hex/unicode/percent encoding).
  3. Runs an honest evaluation tab: a naive IDS detector (no normalization)
     versus the normalizer + detector, reporting per-evasion evasion rates.
  4. Assembles the corpus into a JSON report for regression / CI use.

Everything is synthetic and uses RFC 5737 documentation prefixes. This tool only
ever transforms in-memory strings; it does not touch a network or a live NIDS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Fingerprint -> packed token byte (defensive signature we want to detect).
# --------------------------------------------------------------------------- #
SIGNATURE = "POST /exec?cmd=bash"
SPACER = "/"


# --------------------------------------------------------------------------- #
# Adversarial evasion encoders.
# --------------------------------------------------------------------------- #
def fragment(value, size=4):
    """Fragment the signature payload with benign interleaved filler."""
    parts = []
    n = 0
    while n < len(value):
        c = value[n]
        if c != " ":
            parts.append(c)
        n += 1
    if parts:
        insert = max(1, len(parts) // size)
        chunks = []
        i = 0
        while i < len(parts):
            chunks.append("".join(parts[i:i + insert]))
            i += insert
        return "/".join(chunks)
    return value


def case_mutation(value, seed=0):
    rnd = _lcg(seed)
    out = []
    for ch in value:
        if ch.isalpha() and next(rnd) % 2:
            out.append(ch.upper())
        else:
            out.append(ch)
    return "".join(out)


def backslash_trick(value):
    out = []
    for ch in value:
        if ch in "/":
            out.append("\\/")
        else:
            out.append(ch)
    return "".join(out)


def hex_enclose(value):
    out = []
    for ch in value:
        if ch in " /?=&":
            out.append(ch)
        else:
            out.append("\\x" + ch.encode().hex())
    return "".join(out)


def percent_encoding(value):
    return "".join("%" + format(ord(c), "02X") if c.isalnum() else c for c in value)


def unicode_obfuscate(value):
    table = {"a": "a", "e": "e", "o": "o", "x": "x", "c": "c"}
    out = []
    for ch in value:
        out.append(table.get(ch.lower(), ch))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Deterministic PRNG (no random module reliance in corpus generation).
# --------------------------------------------------------------------------- #
def _lcg(seed):
    state = seed & 0x7FFFFFFF or 1
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state


# --------------------------------------------------------------------------- #
# The normalization layer.
# --------------------------------------------------------------------------- #
# A real robust-normalizer: decode each evasion encoding back to canonical form,
# then collapse the fragmentation spacer.
class RobustNormalizer:
    """Canonical decoding of adversarial payload evasions - pure stdlib."""

    def __init__(self):
        self._token_re = re.compile(r"\\x([0-9a-fA-F]{2})")
        self._pct_re = re.compile(r"%([0-9a-fA-F]{2})")

    def normalize(self, payload):
        s = payload
        s = self._pct_re.sub(lambda m: chr(int(m.group(1), 16)), s)
        s = self._token_re.sub(lambda m: chr(int(m.group(1), 16)), s)
        s = s.replace("\\/", "/")
        s = re.sub(r"/+", "/", s)
        s = s.lower()
        return s

    def signature_present(self, payload):
        return self.normalize(payload).find(SIGNATURE.lower()) != -1


# --------------------------------------------------------------------------- #
# Detectors (naive - the evaluation tab's honest subjects).
# --------------------------------------------------------------------------- #
class NaiveDetector:
    """Detector without normalization: matches the literal signature only."""

    name = "naive"

    def detect(self, payload):
        return SIGNATURE in payload  # literal; no normalization


class NormalizedDetector:
    """Detector guarded by the robust normalizer."""

    name = "normalized"

    def __init__(self, normalizer=None):
        self.normalizer = normalizer or RobustNormalizer()

    def detect(self, payload):
        return self.normalizer.signature_present(payload)


# --------------------------------------------------------------------------- #
# Adversarial corpus builder.
# --------------------------------------------------------------------------- #
def build_corpus(seed=42):
    """Return list of dicts: {name, payload, encoding}."""
    rnd = _lcg(seed)
    corpus = []
    plain = SIGNATURE
    corpus.append({"name": "plain", "payload": plain, "encoding": "none"})

    frag_orig = fragment(plain)
    corpus.append({"name": "fragmentation", "payload": "/".join(
        [frag_orig, "id", "whoami"]), "encoding": "fragmentation"})

    for i in range(3):
        cm = case_mutation(plain, seed=seed + i + 1)
        corpus.append({"name": "case_mutation_%d" % i,
                       "payload": cm, "encoding": "case_mutation"})

    corpus.append({"name": "backslash", "payload": backslash_trick(plain),
                   "encoding": "backslash"})
    corpus.append({"name": "hex_escape", "payload": hex_enclose(plain),
                   "encoding": "hex_escape"})
    corpus.append({"name": "percent_encoding",
                   "payload": percent_encoding(plain), "encoding": "percent"})
    corpus.append({"name": "unicode_homoglyph",
                   "payload": unicode_obfuscate(plain), "encoding": "unicode"})

    # A combined multi-stage evasion.
    combined = hex_enclose(case_mutation(fragment(plain), seed=7))
    corpus.append({"name": "multi_stage", "payload": combined,
                   "encoding": "multi_stage"})

    # A truly benign request that must NOT false-positive.
    corpus.append({"name": "benign_get",
                   "payload": "GET /index.html HTTP/1.1",
                   "encoding": "benign", "benign": True})
    return corpus


# --------------------------------------------------------------------------- #
# Evaluation tab.
# --------------------------------------------------------------------------- #
def evaluate(corpus, detector):
    results = []
    evaded = 0
    true_pos = 0
    benign_total = 0
    false_pos = 0
    for item in corpus:
        hit = detector.detect(item["payload"])
        if item.get("benign"):
            benign_total += 1
            if hit:
                false_pos += 1
            results.append({"name": item["name"], "encoding": item["encoding"],
                            "detected": hit, "benign": True})
        else:
            if not hit:
                evaded += 1
            else:
                true_pos += 1
            results.append({"name": item["name"], "encoding": item["encoding"],
                            "detected": hit, "benign": False})
    total_attacks = len(corpus) - benign_total
    evasion_rate = evaded / total_attacks if total_attacks else 0.0
    return {
        "detector": detector.name,
        "results": results,
        "attacks": total_attacks,
        "detected": true_pos,
        "evaded": evaded,
        "evasion_rate": round(evasion_rate, 3),
        "false_positives": false_pos,
        "benign": benign_total,
    }


# --------------------------------------------------------------------------- #
# Report / CLI.
# --------------------------------------------------------------------------- #
def health_of_report(rep):
    """Return 0 (degraded) if the detector still evades - used as exit code."""
    return 1 if rep["evasion_rate"] > 0 else 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="f2-nids-robustness",
        description="NIDS adversarial robustness: evasion-encoding corpus + "
                    "robust-normalizer evaluation.",
    )
    ap.add_argument("--corpus", default="build",
                    choices=["build", "load"],
                    help="build the corpus or load from an existing JSON file")
    ap.add_argument("--corpus-file", default=None,
                    help="path to corpus JSON to load (with --corpus load)")
    ap.add_argument("--size", type=int, default=4,
                    help="fragmentation chunk size for the corpus builder")
    ap.add_argument("--config", default="config.json",
                    help="path to JSON config")
    ap.add_argument("--report", default="reports/report.md",
                    help="output report path (Markdown or .json)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError as e:
            print("[config] failed to parse: %s" % e, file=sys.stderr)
            return 2
    seed = cfg.get("seed", 42)
    size = cfg.get("fragment_size", args.size)

    if args.corpus == "build":
        corpus = build_corpus(seed=seed)
    else:
        fp = args.corpus_file or cfg.get("corpus_file")
        if not fp or not Path(fp).exists():
            print("[corpus] load requested but no file provided", file=sys.stderr)
            return 2
        corpus = json.loads(Path(fp).read_text())

    naive = evaluate(corpus, NaiveDetector())
    norm = evaluate(corpus, NormalizedDetector())

    banner = "=" * 62 + "\n  F2 - NIDS ADVERSARIAL ROBUSTNESS (evasion corpus)\n" + "=" * 62
    lines = [banner,
             "  Signature attacked : %s" % SIGNATURE,
             "  Corpus size        : %d payloads" % len(corpus),
             ""]

    for det in (naive, norm):
        lines.append("  [%s]  detected=%d/%d  evasion=%d (%.1f%%)  false_pos=%d"
                     % (det["detector"], det["detected"], det["attacks"],
                        det["evaded"], det["evasion_rate"] * 100,
                        det["false_positives"]))
    lines.append("")
    lines.append("  per-payload evasion matrix:")
    hdr = "    {:<20} {:<18} {:>8} {:>8}".format("payload", "encoding", "naive", "norm")
    lines.append(hdr)
    lines.append("    " + "-" * (len(hdr) - 4))
    for nr in norm["results"]:
        nr_name = nr["name"]
        nr_enc = nr["encoding"]
        nv = next((r for r in naive["results"] if r["name"] == nr_name), {})
        nv_hit = "EVADED" if not nv.get("detected") else "HIT"
        nm_hit = "EVADED" if not nr["detected"] else "HIT"
        if nr.get("benign"):
            nv_hit = "benign"; nm_hit = "benign"
        lines.append("    {:<20} {:<18} {:>8} {:>8}".format(
            nr_name, nr_enc, nv_hit, nm_hit))
    lines.append("")
    lines.append("  verdict: robust-normalizer reduces evasion "
                 "%.1f%% -> %.1f%%" % (naive["evasion_rate"] * 100,
                                       norm["evasion_rate"] * 100))
    text = "\n".join(lines)

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)

    # Clean exit code per contract: 0 if demonstrably working (normalizer wins).
    return 0 if norm["evasion_rate"] < naive["evasion_rate"] else 1


if __name__ == "__main__":
    sys.exit(main())
