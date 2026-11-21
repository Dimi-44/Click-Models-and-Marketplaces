"""First-pass (naive UNKNOWN) encoding: refit original ridge M1/M2 and show the relabelled-profile effect."""
import sys, json, zipfile, time, numpy as np, pandas as pd
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PREPARED, RAW_ZIP, OUT, REPO
sys.path.insert(0, str(REPO / 'src'))
from marketplace_ml.full_encoding import CompactSparseEncoder, CONTEXTUAL, ITEM_POSITION
from sklearn.linear_model import LogisticRegression
O=PREPARED
bts=np.load(O+'bts_compact.npy'); rnd=np.load(O+'random_compact.npy')
tr=bts[np.load(O+'indices_train.npy')]; te=bts[np.load(O+'indices_internal_test.npy')]
print('train',len(tr),'test',len(te),'random',len(rnd),flush=True)
# BTS-only codes (hash absent from Random)
cb=json.load(open(O+'bts_storage_codebooks.json'))
cols=[f'user_feature_{i}' for i in range(4)]
remap=rnd.copy()
import os
if os.path.exists(RAW_ZIP):  # identify the BTS-only code per feature from the raw Random hashes
    z=zipfile.ZipFile(RAW_ZIP)
    with z.open('all/All random.csv') as f: r=pd.read_csv(f,usecols=cols)
    for c in cols:
        only=[code for h,code in cb[c].items() if h not in set(r[c].unique())]
        only=sorted(only,key=lambda k:-(tr[c]==k).sum()); print(c,"BTS-only codes",only,[(tr[c]==k).sum() for k in only])
        m=remap[c]==65535; remap[c][m]=only[0]
else:  # without the raw ZIP: the dominant BTS-only code is 0 in every feature (see analysis/core.py)
    for c in cols:
        m=remap[c]==65535; remap[c][m]=0
def ll(y,p):
    p=np.clip(p,1e-12,1-1e-12); return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))
y_te=te['click'].astype(float); y_r=rnd['click'].astype(float); unk=rnd['user_feature_0']==65535
res=[]
for name,fields,C in [('M1',ITEM_POSITION,None),('M2',CONTEXTUAL,10.0),('M2',CONTEXTUAL,0.1)]:
    if C is None: C=10.0  # M1 selected C at full size
    enc=CompactSparseEncoder(fields).fit(tr); X,_=enc.transform(tr)
    t=time.time(); m=LogisticRegression(C=C,solver='lbfgs',tol=1e-6,max_iter=300,random_state=20261114,fit_intercept=True).fit(X,tr['click'])
    out={'model':name,'C':C,'fit_s':round(time.time()-t),'intercept':float(m.intercept_[0])}
    coef=m.coef_[0]
    if name=='M2':
        for c in cols:
            o=enc.offsets[c]; k=len(enc.levels[c]); out[c+'_unknown_coef']=float(coef[o]); out[c+'_mean_level_coef']=float(coef[o+1:o+1+k].mean())
    out['bts_test_ll']=ll(y_te,m.predict_proba(enc.transform(te)[0])[:,1])
    pr=m.predict_proba(enc.transform(rnd)[0])[:,1]
    out['random_ll_asis']=ll(y_r,pr); out['pred_unk_asis']=float(pr[unk].mean())
    pm=m.predict_proba(enc.transform(remap)[0])[:,1]
    out['random_ll_remap']=ll(y_r,pm); out['pred_unk_remap']=float(pm[unk].mean())
    out['ll_unk_asis']=ll(y_r[unk],pr[unk]); out['ll_unk_remap']=ll(y_r[unk],pm[unk]); out['ll_seen']=ll(y_r[~unk],pr[~unk])
    print(json.dumps(out),flush=True); res.append(out)
print('observed CTR unseen-profile rows',y_r[unk].mean())
json.dump(res,open(OUT + 'naive_encoding.json','w'),indent=1)
