import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import pandas as pd
from core import *
bts,rnd,harm,seen,idx,order=load(); tr,va=bts[idx['train']],bts[idx['validation']]
A=aggregate(tr); V=aggregate(va); rows=[]
for mf,mlf in [(0.1,0.0),(0.05,0.0),(0.2,0.0),(0.2,3e-6)]:
    m=Forest(FULL,mf,mlf,B=100).fit(*A[:3]); s=evaluate(m,va,V)[0]['log_loss']
    rows.append({'method':'Forest','features':'full','max_features':mf,'min_leaf_frac':mlf,'B':100,'mean_leaves':m.mean_leaves(),'val_log_loss':s}); print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT + 'forest_ext2.csv',index=False)
