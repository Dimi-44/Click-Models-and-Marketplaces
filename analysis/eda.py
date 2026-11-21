import os
import sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import numpy as np, pandas as pd
from core import load
bts,rnd,harm,seen,idx,order=load()
out={}
for name,a in [('BTS',bts),('Random',rnd)]:
    d=pd.DataFrame({'item':a['item_id'],'pos':a['position'],'click':a['click'],'prof_new':(a['user_feature_0']==(0 if name=='BTS' else 65535)) & (a['user_feature_1']==(0 if name=='BTS' else 65535))})
    out[name]={'n':len(d),'clicks':int(d.click.sum()),'ctr':d.click.mean(),
      'ctr_by_pos':d.groupby('pos').click.mean().to_dict(),'share_by_pos':d.pos.value_counts(normalize=True).sort_index().to_dict(),
      'profile_share':d.prof_new.mean(),'profile_ctr':d[d.prof_new].click.mean(),'other_ctr':d[~d.prof_new].click.mean()}
    g=d.groupby('item').agg(n=('click','size'),ctr=('click','mean'))
    g['share']=g.n/g.n.sum(); g.to_csv(OUT + f'item_{name}.csv')
    out[name]['top10_share']=g.share.sort_values(ascending=False).head(10).sum()
    out[name]['item_ctr_range']=[g.ctr.min(),g.ctr.max()]
# exposure-weighted: CTR BTS would have under random exposure using BTS item-pos CTRs
gb=pd.read_csv(OUT + 'item_BTS.csv',index_col=0); gr=pd.read_csv(OUT + 'item_Random.csv',index_col=0)
out['bts_item_ctr_reweighted_to_random_mix']=float((gb.ctr*gr.share).sum())
# correlation between BTS exposure share and BTS item CTR
out['corr_bts_share_vs_bts_ctr']=float(np.corrcoef(gb.share,gb.ctr)[0,1])
out['corr_item_ctr_bts_vs_random']=float(np.corrcoef(gb.ctr,gr.ctr)[0,1])
json.dump(out,open(OUT + 'eda.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
