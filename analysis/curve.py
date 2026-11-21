import os
"""Learning curves: each method re-tuned on BTS validation at every training size (nested minute-block prefixes)."""
import sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import run_all_lib as L
from core import *
from config import OUT

bts, rnd, harm, seen, idx, order = load()
va, te = bts[idx['validation']], bts[idx['internal_test']]
V = aggregate(va); T = aggregate(te); R = aggregate(harm)
rows = []
for size in [100_000, 300_000, 1_000_000, 3_000_000, len(order)]:
    A = aggregate(bts[order[:size]])
    for method in ['Logit', 'CART', 'Forest']:
        if method == 'Forest' and size == len(order):
            # full size: reuse the final 300-tree forest from evaluate_final.py
            M = pd.read_csv(OUT + 'metrics.csv').set_index(['method', 'features', 'set'])
            rows.append({'size': size, 'method': 'Forest', 'bts_test': M.loc[('Forest', 'full', 'BTS_test')].log_loss,
                         'random': M.loc[('Forest', 'full', 'Random')].log_loss})
            continue
        grid, (s, m, params) = L.tune(method, FULL, A, va, V, quick=(method != 'Forest'))
        rt = evaluate(m, te, T)[0]; rr = evaluate(m, harm, R)[0]
        rows.append({'size': size, 'method': method, 'val': s, 'bts_test': rt['log_loss'], 'random': rr['log_loss'],
                     'bts_auc': rt['auc'], 'random_auc': rr['auc'], **{f'p_{k}': v for k, v in params.items()}})
        print(rows[-1], flush=True)
    for b in ['B0', 'B1']:
        m = Baseline(b).fit(*A[:3])
        rows.append({'size': size, 'method': b, 'bts_test': evaluate(m, te, T)[0]['log_loss'],
                     'random': evaluate(m, harm, R)[0]['log_loss']})
pd.DataFrame(rows).to_csv(OUT + 'learning_curve.csv', index=False)
