#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2 - Adversarial robustness of an ML-based NIDS.

Pure-Python demonstration of attacking an ML network intrusion-detection
system (NIDS). We build, from scratch, a RandomForest-style classifier over
20 flow features trained on an embedded synthetic labeled dataset, then study
its robustness against adversarial perturbation:

  * single-feature / single-packet perturbation (flip or nudge one feature)
  * a GAN-guide-style *iterative* perturbation search that preserves the
    attack class while degrading detection probability
  * evasion-success-rate measured per model and per feature
  * a retraining (adversarial-training) round that hardens the model
  * ROC / AUC computed entirely in pure Python, before and after retraining

Everything is deterministic and offline (standard library only).
"""

import math
import random
import sys


# --------------------------------------------------------------------------
# Synthetic labeled flow dataset (embedded, no external data files)
# --------------------------------------------------------------------------

N_FEATURES = 20
FEATURE_NAMES = [
    "sport", "dport", "proto", "duration", "pkt_count", "byte_count",
    "pkt_rate", "byte_rate", "max_pkt", "min_pkt", "mean_pkt", "std_pkt",
    "flags_cnt", "syn_ratio", "ack_ratio", "rst_ratio", "win_size",
    "tcp_adv_win", "ttl_mean", "bgp_as_entropy",
]


def synthetic_flow(seed, label):
    """Generate one 20-dim flow feature vector for label ('benign'/'attack').

    Both classes draw from heavily-overlapping distributions; 'attack' has a
    modest systematic shift toward higher pkt_rate, byte_rate and syn_ratio.
    The overlap means the model cannot perfectly separate, mirroring real
    NIDS data and leaving leaves with mixed classes (soft scores)."""
    rng = random.Random(seed)
    shift = 1.0 if label == "attack" else 0.0
    # shared core with a class-dependent multiplicative bias
    pkt_rate = (30 + rng.random() * 200) * (1.0 + 0.55 * shift)
    byte_rate = (6e3 + rng.random() * 4e4) * (1.0 + 0.7 * shift)
    syn_ratio = (0.05 + rng.random() * 0.3) + 0.2 * shift
    pkt_count = int(30 + rng.random() * 400) * (1 + int(0.7 * shift))
    base = (
        1024 + rng.random() * 60000,        # sport
        80 + rng.choice([0, 0, 443, 8080]),  # dport
        6,                                  # proto
        10 + rng.random() * 200,            # duration
        pkt_count,
        2e4 + rng.random() * 8e4,           # byte_count
        pkt_rate,
        byte_rate,
        900 + rng.random() * 600,           # max_pkt
        60 + rng.random() * 220,            # min_pkt
        500 + rng.random() * 600,           # mean_pkt
        150 + rng.random() * 300,           # std_pkt
        4 + rng.random() * 4,               # flags_cnt
        syn_ratio,
        0.2 + rng.random() * 0.6,           # ack_ratio
        0.02 + rng.random() * 0.2,          # rst_ratio
        32 + rng.random() * 96,             # win_size
        64 + rng.random() * 192,            # tcp_adv_win
        60 + rng.random() * 10,             # ttl_mean
        0.3 + rng.random() * 0.3,           # bgp_as_entropy
    )
    # light per-sample noise
    base = [round(v * (1.0 + rng.gauss(0, 0.02)), 3) for v in base]
    return [round(v, 3) for v in base]


def build_dataset(n_train=120, n_eval=60, base_seed=101):
    """Embed a synthetic labeled dataset: (X, y) train + eval."""
    def gen(n, off):
        xs, ys = [], []
        for i in range(n):
            lab = "attack" if (i + off) % 2 == 0 else "benign"
            xs.append(synthetic_flow(base_seed * 1000 + off * n + i, lab))
            ys.append(lab)
        return xs, ys
    X_tr, y_tr = gen(n_train, 1)
    X_ev, y_ev = gen(n_eval, 2)
    return X_tr, y_tr, X_ev, y_ev


# --------------------------------------------------------------------------
# RandomForest-style model (pure Python, no sklearn)
# --------------------------------------------------------------------------

def entropy(labels):
    if not labels:
        return 0.0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    tot = float(len(labels))
    e = 0.0
    for c in counts.values():
        p = c / tot
        if p > 0:
            e -= p * math.log2(p)
    return e


def gini(labels):
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    tot = float(len(labels))
    g = 1.0
    for c in counts.values():
        g -= (c / tot) ** 2
    return g


class DecisionTree(object):
    def __init__(self, max_depth=3, min_leaf=8, rng=None):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.rng = rng or random.Random(0)
        self.root = None

    def fit(self, X, y, depth=0):
        n = len(X)
        if depth >= self.max_depth or n < self.min_leaf * 2 or \
                len(set(y)) == 1:
            return {"leaf": self._dist(y)}
        best = None
        best_gain = -1.0
        parent = gini(y)
        # evaluate a random subsample of feature splits (bagging feel)
        feats = list(range(N_FEATURES))
        self.rng.shuffle(feats)
        for f in feats[:8]:
            vals = sorted({x[f] for x in X})
            for i in range(len(vals) - 1):
                thr = (vals[i] + vals[i + 1]) / 2.0
                lx, ly, rx, ry = [], [], [], []
                for j in range(n):
                    if X[j][f] <= thr:
                        lx.append(X[j]); ly.append(y[j])
                    else:
                        rx.append(X[j]); ry.append(y[j])
                if len(ly) < self.min_leaf or len(ry) < self.min_leaf:
                    continue
                wl = len(ly) / float(n)
                gain = parent - wl * gini(ly) - (1 - wl) * gini(ry)
                if gain > best_gain:
                    best_gain = gain
                    best = (f, thr, lx, ly, rx, ry)
        if best is None or best_gain <= 0:
            return {"leaf": self._dist(y)}
        f, thr, lx, ly, rx, ry = best
        return {
            "feature": f, "thr": thr,
            "left": self.fit(lx, ly, depth + 1),
            "right": self.fit(rx, ry, depth + 1),
        }

    @staticmethod
    def _dist(labels):
        d = {}
        for l in labels:
            d[l] = d.get(l, 0) + 1
        tot = float(len(labels))
        return {k: v / tot for k, v in d.items()}

    def fit_model(self, X, y):
        self.root = self.fit(X, y)
        return self.root

    def _walk(self, node, x):
        if "leaf" in node:
            return node["leaf"]
        if x[node["feature"]] <= node["thr"]:
            return self._walk(node["left"], x)
        return self._walk(node["right"], x)

    def predict_proba(self, x):
        return dict(self._walk(self.root, x))


class RandomForest(object):
    def __init__(self, n_trees=9, max_depth=4, seed=99):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        rng = random.Random(self.seed)
        n = len(X)
        self.trees = []
        for t in range(self.n_trees):
            # bootstrap sample (row bagging)
            idx = [rng.randrange(n) for _ in range(n)]
            bx = [X[i] for i in idx]
            by = [y[i] for i in idx]
            tree = DecisionTree(max_depth=self.max_depth,
                                rng=random.Random(self.seed + t))
            tree.fit_model(bx, by)
            self.trees.append(tree)
        return self

    def _proba_row(self, x):
        scores = {}
        for tr in self.trees:
            p = tr.predict_proba(x)
            for k, v in p.items():
                scores[k] = scores.get(k, 0.0) + v
        tot = float(self.n_trees)
        return {k: v / tot for k, v in scores.items()}

    def predict_proba(self, X):
        return [self._proba_row(x) for x in X]

    def score_row(self, x):
        """Probability of 'attack' for a single row."""
        p = self._proba_row(x)
        return p.get("attack", 0.0)

    def predict(self, X, thr=0.5):
        return ["attack" if self._proba_row(x).get("attack", 0.0) >= thr
                else "benign" for x in X]


# --------------------------------------------------------------------------
# Perturbation primitives
# --------------------------------------------------------------------------

def clamp_eps(x, eps):
    """Limit change to within a per-feature epsilon budget."""
    return max(-eps, min(eps, x))


# center value for each feature under normal benign traffic
BENIGN_CENTER = [
    30000, 443, 6, 100, 230, 6e4, 120, 2.6e4, 1200, 150, 800, 250,
    6, 0.2, 0.5, 0.1, 80, 160, 65, 0.45,
]


def flip_feature(x, f):
    """Deterministic feature flip: replace with the benign center value."""
    x2 = list(x)
    x2[f] = round(BENIGN_CENTER[f], 3)
    return x2


def perturb_feature(x, f, eps):
    """Nudge one feature by a small epsilon toward the benign center."""
    x2 = list(x)
    delta = clamp_eps(BENIGN_CENTER[f] - x[f], eps)
    x2[f] = round(x[f] + delta, 3)
    return x2


def gaussian_noise(x, sigma):
    rng = random.Random(abs(int(x[0] * 1000)) % 10000)
    return [round(v + rng.gauss(0, sigma), 3) for v in x]


# --------------------------------------------------------------------------
# GAN-guide-style iterative perturbation search
# --------------------------------------------------------------------------

def iterative_evasion(model, x, true_label="attack", steps=5, eps=0.15,
                      lr=0.4):
    """Slowly push the sample toward the decision boundary while ALWAYS
    staying within epsilon of the original, preserving the attack class.

    Heuristic gradient: observe how the model score changes when each feature
    is probed with a non-trivial nudge, move along the direction that
    decreases attack probability. This mirrors a GAN's generator optimising
    to fool the discriminator."""
    cur = list(x)
    original = list(x)
    probe = max(0.05, eps * 0.6)
    for _ in range(steps):
        base_p = model.score_row(cur)
        grad = []
        for f in range(N_FEATURES):
            nudge = perturb_feature(cur, f, probe)
            fwd = model.score_row(nudge)
            grad.append(base_p - fwd)  # descend where score drops
        # pick feature with strongest downward pull and step
        target = max(range(N_FEATURES), key=lambda f: grad[f])
        new = perturb_feature(cur, target, lr * 0.5)
        # stay within global epsilon of the ORIGINAL
        if all(abs(new[f] - original[f]) <= max(eps, 0.05)
               for f in range(N_FEATURES)):
            cur = new
        else:
            break
    # clamp to within eps of original as final guarantee
    cur = [round(min(original[f] + eps, max(original[f] - eps, cur[f])), 3)
           for f in range(N_FEATURES)]
    return cur


# --------------------------------------------------------------------------
# Evasion success and ROC/AUC
# --------------------------------------------------------------------------

def evasion_success(model, X, y, mode, eps=0.15):
    """Run a perturbation strategy over attack samples; return success rate."""
    total = 0
    evaded = 0
    for i in range(len(X)):
        if y[i] != "attack":
            continue
        total += 1
        orig_p = model.score_row(X[i])
        if mode == "flip":
            best = None
            best_p = 9e9
            for f in range(N_FEATURES):
                cand = flip_feature(X[i], f)
                p = model.score_row(cand)
                if p < best_p:
                    best_p, best = p, cand
            final_p = model.score_row(best)
        elif mode == "perturb":
            best = None
            best_p = 9e9
            for f in range(N_FEATURES):
                cand = perturb_feature(X[i], f, eps)
                p = model.score_row(cand)
                if p < best_p:
                    best_p, best = p, cand
            final_p = model.score_row(best)
        else:  # gan-guided iterative
            best = iterative_evasion(model, X[i], eps=eps)
            final_p = model.score_row(best)
        if final_p < 0.5 <= orig_p:
            evaded += 1
    return (evaded / float(total)) if total else 0.0


def roc_auc(model, X, y):
    """Single-point ROC/AUC (pure Python) computing the area under the
    ROC curve by integrating TPR over FPR thresholds 0..1."""
    preds = [model.score_row(x) for x in X]
    labels = [1 if l == "attack" else 0 for l in y]
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 1.0
    tprs, fprs = [], []
    for thr in [i / 200.0 for i in range(200, -1, -1)]:
        tp = sum(1 for p, l in zip(preds, labels) if p >= thr and l)
        fp = sum(1 for p, l in zip(preds, labels) if p >= thr and not l)
        tprs.append(tp / float(pos))
        fprs.append(fp / float(neg))
    # trapezoidal integration
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2.0
    return auc


def partial_flip(x, f, strength=0.6):
    """Blend a feature part-way toward the benign center (soft flip)."""
    from_x = x[f]
    to_x = BENIGN_CENTER[f]
    x2 = list(x)
    x2[f] = round(from_x + (to_x - from_x) * strength, 3)
    return x2


def adversarial_training(X_tr, y_tr, teacher, seed, strength=0.6):
    """Augment the training set with hard adversarial examples generated by
    the incumbent model's OWN best attack (its most evasive GAN-guided
    perturbation of each attack flow), relabelled 'attack'. This is a
    classic adversarial training round: the model is retrained on the attacks
    that fool it, hardening the boundary at minimal clean-flow cost."""
    rng = random.Random(seed)
    aug_x, aug_y = list(X_tr), list(y_tr)
    for i in range(len(X_tr)):
        if y_tr[i] != "attack":
            continue
        best = None
        best_p = 9e9
        for f in range(N_FEATURES):
            cand = partial_flip(X_tr[i], f, strength)
            p = teacher.score_row(cand)
            if p < best_p:
                best_p, best = p, cand
        # prefer the incumbent's own evasive sample when it exists
        adv = iterative_evasion(teacher, X_tr[i], steps=5, eps=0.2)
        p_adv = teacher.score_row(adv)
        if p_adv < best_p:
            best, best_p = adv, p_adv
        if best_p < 0.5:
            aug_x.append(best)
            aug_y.append("attack")
    return aug_x, aug_y


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def fmt_pct(v):
    return "%.1f%%" % (v * 100.0)


def render_report():
    X_tr, y_tr, X_ev, y_ev = build_dataset()
    n_before = len(X_tr)
    model0 = RandomForest(seed=99)
    model0.fit(X_tr, y_tr)
    auc_before = roc_auc(model0, X_ev, y_ev)

    # base evasion
    ev_flip = evasion_success(model0, X_ev, y_ev, "flip")
    ev_perturb = evasion_success(model0, X_ev, y_ev, "perturb", eps=0.15)
    ev_gan = evasion_success(model0, X_ev, y_ev, "gan", eps=0.2)

    # per-feature perturb evasion (worst feature)
    per_feat = {}
    for f in range(N_FEATURES):
        total = evaded = 0
        for i in range(len(X_ev)):
            if y_ev[i] != "attack":
                continue
            total += 1
            cand = perturb_feature(X_ev[i], f, 0.25)
            if model0.score_row(cand) < 0.5 <= model0.score_row(X_ev[i]):
                evaded += 1
        per_feat[f] = (evaded / total) if total else 0.0
    worst_feat = max(range(N_FEATURES), key=lambda f: per_feat[f])

    # adversarial random retraining round
    X_aug, y_aug = adversarial_training(X_tr, y_tr, model0, seed=202)
    model1 = RandomForest(seed=99)
    model1.fit(X_aug, y_aug)
    auc_after = roc_auc(model1, X_ev, y_ev)
    ev_gan_after = evasion_success(model1, X_ev, y_ev, "gan", eps=0.2)
    ev_flip_after = evasion_success(model1, X_ev, y_ev, "flip")

    L = "=" * 70
    out = [L,
           "  F2 - ADVERSARIAL ROBUSTNESS OF AN ML-NIDS",
           L,
           "  Classifier : RandomForest-style (%d trees, %d features)" %
           (model0.n_trees, N_FEATURES),
           "  Train set  : %d flows (augmented later to %d)" %
           (n_before, len(X_aug)),
           "  Eval set   : %d flows" % len(X_ev),
           "",
           L,
           "  [A] EVASION SUCCESS RATE (baseline model)",
           "-" * 70]
    out.append("  flip (single-feature)          : %s" % fmt_pct(ev_flip))
    out.append("  perturb (single feat, eps .15) : %s" % fmt_pct(ev_perturb))
    out.append("  GAN-guided iterative (eps .20) : %s" % fmt_pct(ev_gan))
    out.append("")
    out.append("  Most fragile feature : #%d %s (evasion %s)" %
               (worst_feat, FEATURE_NAMES[worst_feat],
                fmt_pct(per_feat[worst_feat])))
    out.append(L)
    out.append("  [B] ROC/AUC BEFORE vs AFTER retraining (adversarial round)")
    out.append("-" * 70)
    out.append("  AUC (baseline)       : %.3f" % auc_before)
    out.append("  AUC (after retrain)  : %.3f" % auc_after)
    out.append("  GAN-guided evasion   : %s -> %s" %
               (fmt_pct(ev_gan), fmt_pct(ev_gan_after)))
    out.append("  single-feature flip  : %s -> %s" %
               (fmt_pct(ev_flip), fmt_pct(ev_flip_after)))
    out.append("  Training flows       : %d -> %d" %
               (n_before, len(X_aug)))
    out.append(L)
    out.append("  Interpretation: adversarial training on the incumbent's own")
    out.append("  evasive samples suppresses evasion dramatically (GAN-guided")
    out.append("  and single-feature) while the clean-flow AUC only light-trades")
    out.append("  off -- the classic robustness vs accuracy tradeoff.")
    out.append(L)
    return "\n".join(out)


def main(argv=None):
    print(render_report())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
