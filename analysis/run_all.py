import os
import sys, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from core import *
from config import OUT

os.makedirs(OUT, exist_ok=True)
bts, rnd, harm, seen, idx, order = load()
tr, va, te = bts[idx['train']], bts[idx['validation']], bts[idx['internal_test']]
V = aggregate(va)
log = []


def tune(method, fields, A, B=100, quick=False):
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


# ------------------------------------------------------------------ full-size tuning
A = aggregate(tr)
final, grids = {}, []
for method in ['Logit', 'CART', 'Forest']:
    for fname, fields in [('item+pos', ITEMPOS), ('full', FULL)]:
        t = time.time()
        rows, (s, m, params) = tune(method, fields, A)
        if method == 'Forest':  # refit selected configuration with more trees
            m = Forest(fields, params['max_features'], params['min_leaf_frac'], B=300).fit(*A[:3])
            s = evaluate(m, va, V)[0]['log_loss']; params = {**params, 'B': 300}
        for r in rows: grids.append({'method': method, 'features': fname, **r})
        final[(method, fname)] = m
        print(method, fname, params, round(s, 7), f'{time.time()-t:.0f}s', flush=True)
        log.append({'method': method, 'features': fname, 'selected': params, 'val_log_loss': s})
for b in ['B0', 'B1']:
    final[(b, 'item+pos')] = Baseline(b).fit(*A[:3])
pd.DataFrame(grids).to_csv(OUT + 'tuning_grid.csv', index=False)
json.dump(log, open(OUT + 'selected.json', 'w'), indent=1, default=float)

# ------------------------------------------------------------------ evaluation
sets = {'BTS_test': te, 'Random': harm, 'Random_seen': harm[seen], 'Random_profile': harm[~seen]}
aggs = {k: aggregate(v) for k, v in sets.items()}
metrics, sums = [], {}
for (meth, fname), m in final.items():
    for sname, rows in sets.items():
        r, p = evaluate(m, rows, aggs[sname])
        r.update(method=meth, features=fname, set=sname); metrics.append(r)
        if sname in ('BTS_test', 'Random'):
            sums[(meth, fname, sname)] = minute_loss_sums(rows, aggs[sname][3], p)
    if (meth, fname) in [(k[0], k[1]) for k in final]:
        r, _ = evaluate(m, va, V); r.update(method=meth, features=fname, set='BTS_validation'); metrics.append(r)
M = pd.DataFrame(metrics); M.to_csv(OUT + 'metrics.csv', index=False)
print(M.pivot_table(index=['method', 'features'], columns='set', values='log_loss').round(6))

# paired bootstraps
boot = []
def bs(a, b, label):
    for sname in ('BTS_test', 'Random'):
        sa, cnt = sums[(*a, sname)]; sb, _ = sums[(*b, sname)]
        est, lo, hi = paired_bootstrap(sa, sb, cnt)
        boot.append({'contrast': label, 'set': sname, 'A': '/'.join(a), 'B': '/'.join(b),
                     'diff_A_minus_B_x1e4': est * 1e4, 'lo': lo * 1e4, 'hi': hi * 1e4})
for meth in ['Logit', 'CART', 'Forest']:
    bs((meth, 'item+pos'), (meth, 'full'), f'context_gain_{meth}')
    bs(('B0', 'item+pos'), (meth, 'full'), f'gain_vs_B0_{meth}')
    bs(('B1', 'item+pos'), (meth, 'full'), f'gain_vs_B1_{meth}')
bs(('Logit', 'full'), ('CART', 'full'), 'logit_minus_cart')
bs(('Logit', 'full'), ('Forest', 'full'), 'logit_minus_forest')
bs(('CART', 'full'), ('Forest', 'full'), 'cart_minus_forest')
B = pd.DataFrame(boot); B.to_csv(OUT + 'bootstrap.csv', index=False)
print(B.round(3).to_string())

import pickle
pickle.dump({k: v for k, v in final.items()}, open(OUT + 'final_models.pkl', 'wb'))
