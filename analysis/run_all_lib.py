import itertools
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from core import *

def tune(method, fields, A, va, V, B=100, quick=False):
    """Return list of grid rows and best fitted model (selected on BTS validation log loss)."""
    u, n, c = A[:3]
    rows, best = [], (np.inf, None, None)
    def consider(m, params):
        nonlocal best
        s = evaluate(m, va, V)[0]['log_loss']
        rows.append({**params, 'val_log_loss': s})
        if s < best[0]: best = (s, m, params)
    if method == 'Logit':
        for pen in ['l2']:
            for C in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]:
                consider(Logit(fields, C, pen).fit(u, n, c), {'penalty': pen, 'C': C})
    elif method == 'CART':
        X = design(u, fields); Xs, y, w = stack(X, n, c)
        for mlf in [2e-6, 2e-5]:
            path = DecisionTreeClassifier(criterion='gini', min_weight_fraction_leaf=mlf, random_state=SEED) \
                .cost_complexity_pruning_path(Xs, y, sample_weight=w).ccp_alphas
            path = path[path > 0]
            alphas = np.unique(np.quantile(path, np.linspace(0, 0.999, 12 if quick else 25)))
            for a in alphas:
                for mm in ([100.0] if quick else [10.0, 100.0, 1000.0]):
                    m = CART(fields, a, mlf, mm).fit(u, n, c)
                    consider(m, {'min_leaf_frac': mlf, 'ccp_alpha': float(a), 'm': mm, 'leaves': m.leaves()})
    elif method == 'Forest':
        grid = itertools.product([0.1, 0.3, 0.6, 1.0], [1e-5, 1e-4, 1e-3]) if not quick else \
            itertools.product([0.3, 0.6], [1e-5, 1e-4])
        for mf, mlf in grid:
            m = Forest(fields, mf, mlf, B=B).fit(u, n, c)
            consider(m, {'max_features': mf, 'min_leaf_frac': mlf, 'B': B, 'mean_leaves': m.mean_leaves()})
    return rows, best


