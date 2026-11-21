import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
"""Write numbers.tex and backup tables for the deck from analysis outputs (no hand-typed results)."""
import json, numpy as np, pandas as pd
O = OUT; D = DECK
e = json.load(open(O + 'eda.json')); M = pd.read_csv(O + 'metrics.csv'); B = pd.read_csv(O + 'bootstrap.csv')
X = pd.read_csv(O + 'bootstrap_extra.csv'); sel = {(r['method'], r['features']): r for r in json.load(open(O + 'selected.json'))}
cal = pd.read_csv(O + 'calibration.csv'); imp = pd.read_csv(O + 'importance.csv'); co = json.load(open(O + 'ridge_m1_coef.json'))
cur = pd.read_csv(O + 'learning_curve.csv'); G = pd.read_csv(O + 'tuning_grid.csv')
ll = M.set_index(['method', 'features', 'set'])
mac = {}
def m(name, val): mac[name] = val
def sg(x, d=1): return f'{x:+.{d}f}'.replace('-', '\\(-\\)')
def boot(contrast, s):
    r = B[(B.contrast == contrast) & (B.set == s)].iloc[0]; return r.diff_A_minus_B_x1e4, r.lo, r.hi
def ci(lo, hi): return f'{sg(lo)}, {sg(hi)}'

# data / EDA
m('btsN', f"{e['BTS']['n']/1e6:.2f}m"); m('rndN', f"{e['Random']['n']/1e6:.2f}m")
m('btsClicks', f"{e['BTS']['clicks']:,}".replace(',', '{,}')); m('rndClicks', f"{e['Random']['clicks']:,}".replace(',', '{,}'))
m('btsCTR', f"{e['BTS']['ctr']*100:.3f}"); m('rndCTR', f"{e['Random']['ctr']*100:.3f}")
m('btsTopTen', f"{e['BTS']['top10_share']*100:.0f}"); m('rndTopTen', f"{e['Random']['top10_share']*100:.0f}")
m('corrShareCTR', f"{e['corr_bts_share_vs_bts_ctr']:.2f}")
p = e['BTS']['ctr_by_pos']; m('posCTRa', f"{p['1']*100:.2f}"); m('posCTRb', f"{p['2']*100:.2f}"); m('posCTRc', f"{p['3']*100:.2f}")
m('profShareB', f"{e['BTS']['profile_share']*100:.1f}"); m('profShareR', f"{e['Random']['profile_share']*100:.1f}")
m('profLiftB', f"{e['BTS']['profile_ctr']/e['BTS']['other_ctr']:.2f}"); m('profLiftR', f"{e['Random']['profile_ctr']/e['Random']['other_ctr']:.2f}")
m('profCTRR', f"{e['Random']['profile_ctr']*100:.2f}")
m('btsTestCTR', f"{ll.loc[('B0','item+pos','BTS_test')].ctr*100:.3f}")
# naive encoding (refit of the original pipeline, see refit_results.json / boot.py)
naive = json.load(open(OUT + 'naive_encoding.json'))[1]
nb = {r['set']: r for r in json.load(open(OUT + 'naive_encoding_bootstrap.json'))}
naive_diff = nb['Random_naive']['m1_minus_m2_x1e4']
m('naivePred', f"{naive['pred_unk_asis']*100:.1f}"); m('naiveDiff', sg(naive_diff, 1))
ridgeR = boot('context_gain_Logit', 'Random')[0]
m('naiveRatio', f"{abs(naive_diff)/abs(ridgeR):.0f}")
m('ctxRidgeR', sg(ridgeR, 2))
seenX = X[X.set == 'Random_seen'].set_index('A')
m('ctxRidgeSeen', sg(seenX.loc['Logit/item+pos']['diff'], 2))
# tuning
m('ridgeC', f"{sel[('Logit','full')]['selected']['C']:g}")
m('cartLeaves', f"{sel[('CART','full')]['selected']['leaves']:,}".replace(',', '{,}'))
fs = sel[('Forest', 'full')]['selected']
m('forestB', str(fs['B'])); m('forestMf', f"{fs['max_features']*100:.0f}")
n_train = json.load(open(OUT + 'prepare_manifest.json'))['complete_train_pool']
m('forestLeafText', f"min.\\ leaf \\(\\approx\\){fs['min_leaf_frac']*n_train:.0f} impressions")
fg = pd.concat([G[(G.method == 'Forest') & (G.features == 'full')], pd.read_csv(OUT + 'forest_ext.csv'), pd.read_csv(OUT + 'forest_ext2.csv')])
m('forestGrid', str(len(fg.drop_duplicates(['max_features', 'min_leaf_frac']))))
# result 1
for s, k in [('BTS_test', 'BTS'), ('Random', 'RND')]:
    d, lo, hi = boot('logit_minus_forest', s); m(f'fmr{k}', sg(d)); m(f'fmr{k}ci', ci(lo, hi))
# result 2
cb = {meth: boot(f'context_gain_{meth}', 'BTS_test')[0] for meth in ['Logit', 'CART', 'Forest']}
cr = {meth: boot(f'context_gain_{meth}', 'Random')[0] for meth in ['Logit', 'CART', 'Forest']}
m('ctxCartB', sg(cb['CART'])); m('ctxForestB', sg(cb['Forest']))
m('ctxRangeR', f"{sg(min(cr.values()))} to {sg(max(cr.values()))}")
sv = [seenX.loc[f'{k}/item+pos']['diff'] for k in ['Logit', 'CART', 'Forest']]
m('ctxRangeSeen', f"{sg(min(sv))} to {sg(max(sv))}")
# final model
f = ('Forest', 'full')
b0B, b0R = ll.loc[('B0','item+pos','BTS_test')].log_loss, ll.loc[('B0','item+pos','Random')].log_loss
fB, fR = ll.loc[(*f,'BTS_test')], ll.loc[(*f,'Random')]
m('finLLB', f"{fB.log_loss:.4f}"); m('finLLR', f"{fR.log_loss:.4f}")
m('finSkillB', f"\\(-\\){(1-fB.log_loss/b0B)*100:.1f}"); m('finSkillR', f"\\(-\\){(1-fR.log_loss/b0R)*100:.1f}")
m('finAUCB', f"{fB.auc:.3f}"); m('finAUCR', f"{fR.auc:.3f}"); m('finMeanR', f"{fR.mean_pred*100:.3f}")
top = cal[(cal.model == 'Forest') & (cal.set == 'Random')].sort_values('q').iloc[-1]
m('finTopPredR', f"{top.pred*100:.2f}"); m('finTopObsR', f"{top.obs*100:.2f}")
vs = X[(X.A == 'Logit/item+pos') & (X.B == 'Forest/full')].set_index('set')
m('finVsSimpleB', sg(vs.loc['BTS_test']['diff'])); m('finVsSimpleBci', ci(vs.loc['BTS_test'].lo, vs.loc['BTS_test'].hi))
m('finVsSimpleR', sg(vs.loc['Random']['diff'])); m('finVsSimpleRci', ci(vs.loc['Random'].lo, vs.loc['Random'].hi))
orr = np.exp(np.array(co['item_coef']) - np.mean(co['item_coef'])); m('itemSpread', f"{orr.max()/orr.min():.1f}")
po = np.exp(np.array(co['pos_coef']) - np.mean(co['pos_coef'])); m('posEffect', f"{np.abs(po-1).max()*100:.1f}")
m('posORs', ', '.join(f'{x:.3f}' for x in po))
# learning curve text
c = cur.copy()
def at(meth, size, col): return c[(c.method == meth) & (c['size'] == size)][col].iloc[0]
full = c['size'].max()
gB = {k: (at(k, 1_000_000, 'bts_test') - at(k, full, 'bts_test')) * 1e4 for k in ['Logit', 'Forest']}
gR = {k: (at(k, 1_000_000, 'random') - at(k, full, 'random')) * 1e4 for k in ['Logit', 'CART', 'Forest']}
fr = [at(k, full, 'random') for k in ['Logit', 'CART', 'Forest', 'B1']]
m('curveBTStext', f"the trees keep pulling away from ridge as data grow (1m \\(\\to\\) 8.66m: forest improves by {gB['Forest']:.1f}, ridge by {gB['Logit']:.1f}).")
m('curveRNDtext', f"every model improves too (by {min(gR.values()):.1f} to {max(gR.values()):.1f}), but at 8.66m all models and the item\\(\\times\\)position table lie within {(max(fr)-min(fr))*1e4:.1f} of each other.")
with open(D + 'numbers.tex', 'w') as fh:
    for k, v in mac.items(): fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')

# ---- tables
NAMES = {'B0': 'Constant CTR', 'B1': 'Item\\(\\times\\)position CTR', 'Logit': 'Ridge logistic', 'CART': 'CART (pruned)', 'Forest': 'Random forest'}
rows = []
for meth in ['B0', 'B1', 'Logit', 'CART', 'Forest']:
    for feat in (['item+pos'] if meth.startswith('B') else ['item+pos', 'full']):
        r = [NAMES[meth], '' if meth.startswith('B') else ('+ user' if feat == 'full' else 'item+pos')]
        for s in ['BTS_test', 'Random']:
            q = ll.loc[(meth, feat, s)]
            r += [f'{q.log_loss*1e4:.2f}', f'{q.auc:.3f}', f'{q.mean_pred*100:.3f}']
        rows.append(' & '.join(r) + ' \\\\')
t = ['\\begin{tabular}{@{}llrrr@{\\hspace{.6cm}}rrr@{}}\\toprule',
     ' & & \\multicolumn{3}{c}{\\color{bts}\\textbf{BTS test}} & \\multicolumn{3}{c}{\\color{rnd}\\textbf{Random}}\\\\',
     'Model & Features & Log loss & AUC & Mean pred. & Log loss & AUC & Mean pred.\\\\',
     ' & & \\(\\times10^{4}\\) & & \\% & \\(\\times10^{4}\\) & & \\%\\\\\\midrule'] + rows + ['\\bottomrule\\end{tabular}']
open(D + 'tab_metrics.tex', 'w').write('\n'.join(t))

lab = {'logit_minus_forest': 'Ridge \\(-\\) Forest (both + user)', 'logit_minus_cart': 'Ridge \\(-\\) CART (both + user)',
       'cart_minus_forest': 'CART \\(-\\) Forest (both + user)'}
for k in ['Logit', 'CART', 'Forest']:
    lab[f'context_gain_{k}'] = f'{NAMES[k]}: item+pos \\(-\\) + user'
    lab[f'gain_vs_B0_{k}'] = f'Constant \\(-\\) {NAMES[k]} (+ user)'
order = [f'gain_vs_B0_{k}' for k in ['Logit', 'CART', 'Forest']] + ['logit_minus_forest', 'logit_minus_cart', 'cart_minus_forest'] + \
        [f'context_gain_{k}' for k in ['Logit', 'CART', 'Forest']]
rows = []
for cname in order:
    r = [lab[cname]]
    for s in ['BTS_test', 'Random']:
        d, lo, hi = boot(cname, s); r += [sg(d, 2), f'[{sg(lo,2)}, {sg(hi,2)}]']
    rows.append(' & '.join(r) + ' \\\\')
vsr = [ 'Ridge item+pos \\(-\\) Forest + user (final)', sg(vs.loc['BTS_test']['diff'],2), f"[{sg(vs.loc['BTS_test'].lo,2)}, {sg(vs.loc['BTS_test'].hi,2)}]",
        sg(vs.loc['Random']['diff'],2), f"[{sg(vs.loc['Random'].lo,2)}, {sg(vs.loc['Random'].hi,2)}]"]
rows.insert(6, '\\midrule'); rows.insert(6, ' & '.join(vsr) + ' \\\\'); rows.insert(3, '\\midrule')
t = ['\\begin{tabular}{@{}lrlrl@{}}\\toprule', 'Contrast (A \\(-\\) B) & \\multicolumn{2}{c}{\\color{bts}\\textbf{BTS test}} & \\multicolumn{2}{c}{\\color{rnd}\\textbf{Random}}\\\\\\midrule'] + rows + ['\\bottomrule\\end{tabular}']
open(D + 'tab_boot.tex', 'w').write('\n'.join(t))

rows = []
for meth in ['Logit', 'CART', 'Forest']:
    r = [NAMES[meth]]
    for s in ['Random_seen', 'Random_profile']:
        a, b = ll.loc[(meth, 'item+pos', s)].log_loss, ll.loc[(meth, 'full', s)].log_loss
        r += [f'{a*1e4:.2f}', f'{b*1e4:.2f}', sg((a-b)*1e4, 2)]
    rows.append(' & '.join(r) + ' \\\\')
n_prof = round(e['Random']['n'] * e['Random']['profile_share']); n_seen = e['Random']['n'] - n_prof
fmt = lambda x: f'{x:,}'.replace(',', '{,}')
t = ['\\begin{tabular}{@{}lrrr@{\\hspace{.6cm}}rrr@{}}\\toprule',
     f' & \\multicolumn{{3}}{{c}}{{Matched-code rows ({fmt(n_seen)})}} & \\multicolumn{{3}}{{c}}{{Relabelled segment ({fmt(n_prof)})}}\\\\',
     'Log loss \\(\\times10^{4}\\) & item+pos & + user & gain & item+pos & + user & gain\\\\\\midrule'] + rows + \
    ['\\bottomrule\\end{tabular}\\par\\vspace{.15cm}', f'{{\\footnotesize\\color{{muted}} Context-gain 95\\% CIs on matched rows: ridge [{sg(seenX.loc["Logit/item+pos"].lo,2)}, {sg(seenX.loc["Logit/item+pos"].hi,2)}], CART [{sg(seenX.loc["CART/item+pos"].lo,2)}, {sg(seenX.loc["CART/item+pos"].hi,2)}], forest [{sg(seenX.loc["Forest/item+pos"].lo,2)}, {sg(seenX.loc["Forest/item+pos"].hi,2)}].}}']
open(D + 'tab_subgroups.tex', 'w').write('\n'.join(t))

sizes = sorted(c['size'].unique())
hdr = ' & '.join(['Training rows'] + [f'{s/1e6:.2f}'.rstrip('0').rstrip('.') + 'm' for s in sizes])
rows = []
for s, nm in [('bts_test', 'BTS test'), ('random', 'Random')]:
    rows.append(f'\\multicolumn{{{len(sizes)+1}}}{{@{{}}l}}{{\\textbf{{{nm}}}}}\\\\')
    for meth in ['B0', 'B1', 'Logit', 'CART', 'Forest']:
        rows.append(' & '.join([NAMES[meth]] + [f'{at(meth, z, s)*1e4:.2f}' for z in sizes]) + ' \\\\')
t = ['\\begin{tabular}{@{}l' + 'r' * len(sizes) + '@{}}\\toprule', hdr + '\\\\\\midrule'] + rows + ['\\bottomrule\\end{tabular}']
open(D + 'tab_curve.tex', 'w').write('\n'.join(t))

P = imp[imp.model.isin(['Logit', 'Forest'])].pivot_table(index='group', columns=['model', 'set'], values='increase_x1e4')
rows = []
for g in ['item', 'position', 'user profile']:
    rows.append(' & '.join([g] + [f'{P.loc[g, (mm, s)]:.2f}' for mm in ['Logit', 'Forest'] for s in ['BTS_test', 'Random']]) + ' \\\\')
t = ['\\begin{tabular}{@{}lrrrr@{}}\\toprule', ' & \\multicolumn{2}{c}{Ridge + user} & \\multicolumn{2}{c}{Forest + user}\\\\',
     'Group & BTS & Random & BTS & Random\\\\\\midrule'] + rows + ['\\bottomrule\\end{tabular}']
open(D + 'tab_importance.tex', 'w').write('\n'.join(t))
print('\n'.join(f'{k} = {v}' for k, v in mac.items()))
