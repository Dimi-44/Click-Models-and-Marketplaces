"""Minute-block bootstrap of the ridge context contrast under naive vs harmonised encoding."""
import sys, json, zipfile, numpy as np, pandas as pd
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PREPARED, REPO
sys.path.insert(0, str(REPO / 'src'))
from marketplace_ml.full_encoding import CompactSparseEncoder, CONTEXTUAL, ITEM_POSITION
from sklearn.linear_model import LogisticRegression
O=PREPARED
bts=np.load(O+'bts_compact.npy'); rnd=np.load(O+'random_compact.npy')
tr=bts[np.load(O+'indices_train.npy')]; te=bts[np.load(O+'indices_internal_test.npy')]
remap=rnd.copy()
for c in [f'user_feature_{i}' for i in range(4)]: remap[c][remap[c]==65535]=0
def fit(fields,C):
    e=CompactSparseEncoder(fields).fit(tr)
    return e, LogisticRegression(C=C,solver='lbfgs',tol=1e-6,max_iter=300,random_state=20261114).fit(e.transform(tr)[0],tr['click'])
def lossvec(y,p): p=np.clip(p,1e-12,1-1e-12); return -y*np.log(p)-(1-y)*np.log1p(-p)
e1,m1=fit(ITEM_POSITION,10.0); e2,m2=fit(CONTEXTUAL,10.0)
rng=np.random.default_rng(1)
def boot(rows,label):
    y=rows['click'].astype(float)
    d=lossvec(y,m1.predict_proba(e1.transform(rows)[0])[:,1])-lossvec(y,m2.predict_proba(e2.transform(rows)[0])[:,1])
    blk=rows['minute_id'].astype(int); ub,inv=np.unique(blk,return_inverse=True)
    S=np.bincount(inv,weights=d); N=np.bincount(inv)
    draws=[]
    for _ in range(1000):
        i=rng.integers(0,len(ub),len(ub)); draws.append(S[i].sum()/N[i].sum())
    lo,hi=np.percentile(draws,[2.5,97.5])
    print(f'{label}: M1-M2 = {d.mean()*1e4:+.2f} x1e-4  95% CI [{lo*1e4:+.2f}, {hi*1e4:+.2f}]',flush=True)
unk=rnd['user_feature_0']==65535
boot(te,'BTS test')
boot(rnd,'Random as-is (your method)')
boot(remap,'Random, profile remapped')
boot(rnd[~unk],'Random, seen-code rows only')
