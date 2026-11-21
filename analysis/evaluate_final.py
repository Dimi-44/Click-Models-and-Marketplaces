import os
import sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from core import *
from config import OUT
bts, rnd, harm, seen, idx, order = load()
tr, va, te = bts[idx['train']], bts[idx['validation']], bts[idx['internal_test']]
V = aggregate(va); A = aggregate(tr)
final = pickle.load(open(OUT + 'final_models.pkl', 'rb'))
best = sys.argv[1:]  # max_features min_leaf_frac
final[('Forest', 'full')] = Forest(FULL, float(best[0]), float(best[1]), B=300).fit(*A[:3])
print('refit forest', evaluate(final[('Forest','full')], va, V)[0], flush=True)
sel = json.load(open(OUT + 'selected.json'))
for r in sel:
    if r['method'] == 'Forest' and r['features'] == 'full':
        r['selected'] = {'max_features': float(best[0]), 'min_leaf_frac': float(best[1]), 'B': 300,
                         'mean_leaves': final[('Forest','full')].mean_leaves()}
        r['val_log_loss'] = evaluate(final[('Forest','full')], va, V)[0]['log_loss']
json.dump(sel, open(OUT + 'selected.json', 'w'), indent=1, default=float)
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
