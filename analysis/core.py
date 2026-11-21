"""Corrected analysis core: exact fitting on aggregated (pattern, n, clicks) counts.

All six predictors are categorical, so every impression is one of ~61k patterns.
Binary log-likelihoods, Gini splits and minute-block bootstraps are computed exactly
from pattern counts (two weighted rows per pattern: clicked / not clicked).
"""
from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

O = PREPARED
UF = [f'user_feature_{i}' for i in range(4)]
LEVELS = {'item_id': 80, 'position': 3, 'user_feature_0': 4, 'user_feature_1': 6,
          'user_feature_2': 11, 'user_feature_3': 10}
GROUPS = {'item': ['item_id'], 'position': ['position'], 'user': UF}
FULL = ['item_id', 'position'] + UF
ITEMPOS = ['item_id', 'position']
SEED = 20261114


def load():
    bts = np.load(O + 'bts_compact.npy')
    rnd = np.load(O + 'random_compact.npy')
    seen = rnd['user_feature_0'] != 65535
    harm = rnd.copy()
    # Label-free harmonisation: the single Random-only profile (all four codes unseen,
    # 35.4% of Random) is mapped to the single BTS-only profile (code 0 in all four
    # fields, 35.3% of BTS). Decided from feature codes only, never from Random clicks.
    for c in UF:
        harm[c][harm[c] == 65535] = 0
    idx = {k: np.load(O + f'indices_{k}.npy') for k in ('train', 'validation', 'internal_test')}
    order = np.load(O + 'train_order_minute_rep0.npy')
    return bts, rnd, harm, seen, idx, order


def keys(rows):
    k = np.zeros(len(rows), dtype=np.int64)
    for c in FULL:
        k = k * LEVELS[c] + rows[c].astype(np.int64) - (1 if c == 'position' else 0)
    return k


def decode(k):
    out = {}
    for c in reversed(FULL):
        out[c] = (k % LEVELS[c]).astype(np.int64)
        k = k // LEVELS[c]
    return out


def aggregate(rows):
    u, inv = np.unique(keys(rows), return_inverse=True)
    n = np.bincount(inv).astype(float)
    c = np.bincount(inv, weights=rows['click'].astype(float))
    return u, n, c, inv


def design(u, fields):
    d = decode(u.copy())
    blocks, off = [], 0
    rows = np.arange(len(u))
    cols, data = [], []
    for f in fields:
        cols.append(off + d[f]); off += LEVELS[f]
    cols = np.concatenate(cols)
    r = np.tile(rows, len(fields))
    return sparse.csr_matrix((np.ones(len(cols)), (r, cols)), shape=(len(u), off))


def stack(X, n, c):
    """Two weighted rows per pattern: clicked (weight c) and not clicked (weight n-c)."""
    Xs = sparse.vstack([X, X]).tocsr()
    y = np.r_[np.ones(len(n)), np.zeros(len(n))]
    w = np.r_[c, n - c]
    keep = w > 0
    return Xs[keep], y[keep], w[keep]


def logloss(n, c, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(c * np.log(p) + (n - c) * np.log1p(-p)).sum() / n.sum())


# ---------------------------------------------------------------- models
class Baseline:
    """B0 constant CTR; B1 smoothed item x position CTR (prior strength m)."""
    def __init__(self, kind, m=1000.0):
        self.kind, self.m = kind, m

    def fit(self, u, n, c):
        self.p0 = c.sum() / n.sum()
        if self.kind == 'B1':
            d = decode(u.copy()); key = d['item_id'] * 3 + d['position']
            N = np.bincount(key, weights=n, minlength=240); C = np.bincount(key, weights=c, minlength=240)
            self.rate = (C + self.m * self.p0) / (N + self.m)
        return self

    def predict(self, u):
        if self.kind == 'B0':
            return np.full(len(u), self.p0)
        d = decode(u.copy()); return self.rate[d['item_id'] * 3 + d['position']]


class Logit:
    """Penalised logistic regression (L2 ridge or L1 lasso); intercept unpenalised (saga/lbfgs)."""
    def __init__(self, fields, C, penalty='l2'):
        self.fields, self.C, self.penalty = fields, C, penalty

    def fit(self, u, n, c):
        X, y, w = stack(design(u, self.fields), n, c)
        if self.penalty == 'l2':
            m = LogisticRegression(C=self.C, solver='lbfgs', max_iter=2000, tol=1e-8)
        else:
            m = LogisticRegression(C=self.C, penalty='l1', solver='saga', max_iter=3000, tol=1e-6,
                                   random_state=SEED)
        self.m = m.fit(X, y, sample_weight=w)
        return self

    def predict(self, u):
        return self.m.predict_proba(design(u, self.fields))[:, 1]


def _leaf_rates(tree, X, n, c, p0, m):
    leaf = tree.apply(X)
    L = tree.tree_.node_count
    N = np.bincount(leaf, weights=n, minlength=L); C = np.bincount(leaf, weights=c, minlength=L)
    return (C + m * p0) / (N + m)


class CART:
    """Gini classification tree, cost-complexity pruned (rpart cp analogue).
    Leaf click rates are shrunk towards the training CTR with prior strength m, so that
    leaves without clicks never output probability 0."""
    def __init__(self, fields, ccp_alpha, min_leaf_frac=2e-5, m=100.0):
        self.fields, self.a, self.mlf, self.mm = fields, ccp_alpha, min_leaf_frac, m

    def fit(self, u, n, c, tree=None):
        X = design(u, self.fields)
        Xs, y, w = stack(X, n, c)
        self.t = DecisionTreeClassifier(criterion='gini', ccp_alpha=self.a, min_weight_fraction_leaf=self.mlf,
                                        random_state=SEED).fit(Xs, y, sample_weight=w)
        self.p0 = c.sum() / n.sum()
        self.rate = _leaf_rates(self.t, X, n, c, self.p0, self.mm)
        return self

    def leaves(self):
        return int(self.t.get_n_leaves())

    def predict(self, u):
        return self.rate[self.t.apply(design(u, self.fields))]


class Forest:
    """Random forest via Poisson bootstrap of pattern counts (exact analogue of row bootstrap),
    random feature subsets at each split, smoothed leaf rates averaged over trees."""
    def __init__(self, fields, max_features, min_leaf_frac, B=100, m=100.0, seed=SEED):
        self.fields, self.mf, self.mlf, self.B, self.mm, self.seed = fields, max_features, min_leaf_frac, B, m, seed

    def fit(self, u, n, c):
        X = design(u, self.fields)
        rng = np.random.default_rng(self.seed)
        self.trees = []
        p0 = c.sum() / n.sum()
        for b in range(self.B):
            cb = rng.poisson(c).astype(float); nb = cb + rng.poisson(n - c)
            Xs, y, w = stack(X, nb, cb)
            t = DecisionTreeClassifier(criterion='gini', max_features=self.mf, min_weight_fraction_leaf=self.mlf,
                                       random_state=int(rng.integers(1 << 31))).fit(Xs, y, sample_weight=w)
            self.trees.append((t, _leaf_rates(t, X, nb, cb, p0, self.mm)))
        return self

    def predict(self, u):
        X = design(u, self.fields)
        return np.mean([r[t.apply(X)] for t, r in self.trees], axis=0)

    def mean_leaves(self):
        return float(np.mean([t.get_n_leaves() for t, _ in self.trees]))


# ---------------------------------------------------------------- evaluation helpers
def auc(n, c, p):
    """Exact ROC-AUC from pattern counts (ties count 1/2)."""
    order = np.argsort(p)
    p, pos, neg = p[order], c[order], (n - c)[order]
    # group ties
    _, start = np.unique(p, return_index=True)
    P = np.add.reduceat(pos, start); Nn = np.add.reduceat(neg, start)
    cumneg = np.cumsum(Nn) - Nn
    return float((P * (cumneg + 0.5 * Nn)).sum() / (pos.sum() * neg.sum()))


def evaluate(model, rows, agg=None):
    u, n, c, inv = agg if agg is not None else aggregate(rows)
    p = model.predict(u)
    return {'log_loss': logloss(n, c, p), 'auc': auc(n, c, p), 'mean_pred': float((n * p).sum() / n.sum()),
            'ctr': float(c.sum() / n.sum()), 'brier': float(((c * (1 - p) ** 2) + (n - c) * p ** 2).sum() / n.sum())}, p


def minute_loss_sums(rows, inv, p_pattern):
    """Per-minute summed log loss and counts, for the paired block bootstrap."""
    y = rows['click'].astype(float); p = np.clip(p_pattern[inv], 1e-12, 1 - 1e-12)
    l = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    mins = rows['minute_id'].astype(np.int64)
    return np.bincount(mins, weights=l, minlength=10080), np.bincount(mins, minlength=10080).astype(float)


def paired_bootstrap(sumA, sumB, cnt, draws=1000, seed=1):
    """Difference A - B in mean loss; minute blocks resampled with replacement within each date."""
    rng = np.random.default_rng(seed)
    d = sumA - sumB
    mins = np.nonzero(cnt)[0]; date = mins // 1440
    est = d.sum() / cnt.sum(); out = []
    groups = [mins[date == k] for k in np.unique(date)]
    for _ in range(draws):
        pick = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        out.append(d[pick].sum() / cnt[pick].sum())
    lo, hi = np.percentile(out, [2.5, 97.5])
    return est, lo, hi
