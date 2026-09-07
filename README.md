# F2 — NIDS Adversarial Robustness (protocol-evasion corpus)

A deterministic, offline, standard-library-only toolkit that stresses a network
intrusion-detection signature matcher with adversarial **protocol-level evasion
encodings** and shows what a **robust-normalization layer** buys you.

## Overview

- Builds an **adversarial evasion corpus** around a defensive web-shell
  detection signature (`POST /exec?cmd=bash`):
  - raw payload
  - **fragmentation** (chunked with benign filler)
  - **case mutation** (deterministic LCG-driven)
  - **backslash trick** (`/` → `\/`)
  - **hex-escape** (`\x41`-style)
  - **percent-encoding** (`%41`-style)
  - **unicode homoglyph** substitution
  - a multi-stage combination, plus one genuinely benign request
- Implements a **RobustNormalizer** that canonically decodes each encoding
  (percent, hex, backslash, case, fragmentation-collapse) before matching.
- Runs an **honest evaluation tab**: a naive matcher vs the normalized matcher on
  the *same* corpus, reporting detected/evaded counts, evasion rate, and
  false-positives on the benign sample.
- Emits a Markdown/JSON report under `reports/`, with clean exit codes.

## CLI

```bash
python3 firmware/nids_robustness.py --help
python3 firmware/nids_robustness.py                 # build corpus + evaluate
python3 firmware/nids_robustness.py --report reports/report.json
python3 firmware/nids_robustness.py --corpus load --corpus-file corpus.json
```

Config lives in `config.json` (`seed`, `fragment_size`). Reports go to
`reports/` (gitignored).

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## IMPORTANT: Read before use.

This tool is provided **exclusively** for authorized security research, academic
study, and defensive hardening. Use without explicit written authorization is
illegal and unethical.

### Authorization Requirements

You must obtain explicit written permission before evaluating the robustness of
any NIDS or ML system you do not own. Generating adversarial traffic against
third-party networks or production detectors without authorization may
constitute illegal access or interference. This toolkit only ever transforms
in-memory synthetic strings; it deploys nothing against a live network.

### Legal Framework

Unauthorized access to or manipulation of computer systems is governed by the
**Computer Fraud and Abuse Act (CFAA)** (18 U.S.C. § 1030), the **EU Directive
on Attacks Against Information Systems** (2013/40/EU), and equivalent
legislation in other jurisdictions. Penalties include imprisonment and
significant fines.

### Acceptable Use

- Authorized red-team and penetration-testing engagements with written scope
- Evaluating the robustness of NIDS you operate against protocol-evasion encoding
- Academic research on adversarial normalization for network defense
- CTF competitions and educational labs

### Prohibited Use

- Generating adversarial or evasion traffic against systems without authorization
- Deploying evasion encodings to bypass production security controls
- Evaluating or attacking third-party models or networks without permission
- Any use that violates applicable law or terms of service

### No Warranty

This software is provided "as is" without warranty of any kind. The authors
assume no liability for damages arising from use or misuse of this tool.

### Responsible Disclosure

If you discover evasion weaknesses in third-party NIDS or encoding practices
using this tool, follow coordinated disclosure. Report to the vendor directly
and allow reasonable time for remediation before public disclosure.

## Live Lab Test Plan

1. **Sanity** — run the demo; confirm it prints the corpus banner and exits `0`.
2. **Corpus integrity** — `--corpus build` is deterministic: running twice yields
   identical payloads (verified by unit test `test_corpus_deterministic`).
3. **Normalizer guard** — confirm `normalized` detector reports `false_positives
   == 0` on the embedded benign request (unit test).
4. **Honest reduction** — confirm normalized evasion rate is strictly below the
   naive rate on the same corpus (unit test `test_naive_evades_more_than_normalized`).
5. **CLI hygiene** — `--help` lists all flags; a bad config path exits `2`;
   a successful run writes a report file and exits `0`.
6. **Offline guarantee** — no sockets, no subprocesses, no third-party imports:
   runs in any sandbox that permits Python stdlib.

## Metrics

| Metric | Definition |
|--------|-----------|
| Corpus size | number of generated payloads (attacks + benign) |
| Naive evasion rate | attacks missed by the literal matcher / total attacks |
| Normalized evasion rate | attacks missed after robust normalization / total |
| False positives | benign payloads the detector flags |
| Reduction | normalized rate subtracted from naive rate (improvement) |

Verified offline: naive evades ~90% of the corpus; the robust normalizer cuts
that below 20% with zero benign false-positives.

## License

MIT License
