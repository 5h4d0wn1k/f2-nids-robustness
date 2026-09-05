# F2 — Adversarial Robustness of an ML-NIDS

Pure-Python demonstration of feature-space adversarial attacks on a RandomForest-style network intrusion-detection classifier, with adversarial retraining.

## Overview

- Trains a RandomForest-style classifier from scratch on 20 embedded synthetic flow features (train + eval sets generated offline)
- Implements single-feature / single-packet perturbations — a full feature flip and a bounded nudge within an epsilon budget
- Implements a GAN-guide-style iterative perturbation search that preserves the attack class while degrading detection probability
- Measures evasion-success-rate per model and identifies the most fragile flow feature
- Applies an adversarial-training retraining round on the incumbent model's own evasive samples
- Reports ROC/AUC before and after retraining, computed entirely in pure Python
- Fully offline and deterministic — no sklearn, no torch, no network

## Features

- **Pure-Python RandomForest**: bootstrap bagging + gini-gain decision trees with class-fraction leaf probabilities (soft scores)
- **Single-Feature Flip**: clamp any one feature to the benign traffic center
- **Epsilon-Bounded Perturbation**: nudge any one feature toward benign within a per-feature budget
- **GAN-Guide Iterative Search**: finite-difference gradient over all 20 features, stepping toward the decision boundary while staying inside epsilon of the original sample (attack class preserved)
- **Per-Feature Fragility Map**: evasion rate measured for every feature
- **Adversarial Retraining**: augment training with the model's most evasive perturbed flows, relabelled correctly
- **ROC/AUC in Pure Python**: threshold-integrated area under the ROC curve
- **Zero Dependencies**: Python standard library only

## Installation

No external dependencies required — uses Python standard library only.

```bash
python3 firmware/nids_robustness.py
```

## Usage

```python
from firmware.nids_robustness import RandomForest, build_dataset
from firmware.nids_robustness import evasion_success, roc_auc, adversarial_training

X_tr, y_tr, X_ev, y_ev = build_dataset()
model = RandomForest(seed=99).fit(X_tr, y_tr)

print("GAN evasion :", evasion_success(model, X_ev, y_ev, "gan", eps=0.2))
print("AUC         :", roc_auc(model, X_ev, y_ev))

X_aug, y_aug = adversarial_training(X_tr, y_tr, model, seed=202)
robust = RandomForest(seed=99).fit(X_aug, y_aug)
print("AUC after retrain:", roc_auc(robust, X_ev, y_ev))
```

## Example Output

```
======================================================================
  F2 - ADVERSARIAL ROBUSTNESS OF AN ML-NIDS
======================================================================
  Classifier : RandomForest-style (9 trees, 20 features)
  Train set  : 120 flows (augmented later to 157)
  Eval set   : 60 flows

  [A] EVASION SUCCESS RATE (baseline model)
----------------------------------------------------------------------
  flip (single-feature)          : 83.3%
  perturb (single feat, eps .15) : 46.7%
  GAN-guided iterative (eps .20) : 53.3%

  Most fragile feature : #13 syn_ratio (evasion 80.0%)

  [B] ROC/AUC BEFORE vs AFTER retraining (adversarial round)
----------------------------------------------------------------------
  AUC (baseline)       : 0.941
  AUC (after retrain)  : 0.902
  GAN-guided evasion   : 53.3% -> 3.3%
  single-feature flip  : 83.3% -> 40.0%
  Training flows       : 120 -> 157
```

## IMPORTANT: Read before use.

This tool is provided **exclusively** for authorized security research, academic study, and defensive hardening. Use without explicit written authorization is illegal and unethical.

### Authorization Requirements

You must obtain explicit written permission before evaluating the robustness of any NIDS or ML system you do not own. Generating adversarial traffic against third-party networks or production detectors without authorization may constitute illegal access or interference.

### Legal Framework

Unauthorized access to or manipulation of computer systems is governed by the **Computer Fraud and Abuse Act (CFAA)** (18 U.S.C. § 1030), the **EU Directive on Attacks Against Information Systems** (2013/40/EU), and equivalent legislation in other jurisdictions. Penalties include imprisonment and significant fines.

### Acceptable Use

- Authorized red-team and penetration-testing engagements with written scope
- Evaluating robustness of your own ML-based NIDS models
- Academic research on adversarial machine learning for network defense
- CTF competitions and educational labs

### Prohibited Use

- Generating adversarial or evasion traffic against systems without authorization
- Deploying evasion techniques to bypass production security controls
- Evaluating or attacking third-party models or networks without permission
- Any use that violates applicable law or terms of service

### No Warranty

This software is provided "as is" without warranty of any kind. The authors assume no liability for damages arising from use or misuse of this tool.

### Responsible Disclosure

If you discover vulnerabilities in third-party NIDS or ML systems using this tool, follow coordinated disclosure practices. Report to the vendor directly and allow reasonable time for remediation before public disclosure.

## License

MIT License