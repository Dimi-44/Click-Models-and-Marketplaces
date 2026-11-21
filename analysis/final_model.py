import os
import sys, pickle, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import numpy as np, pandas as pd
from core import *
bts,rnd,harm,seen,idx,order=load(); te=bts[idx['internal_test']]
F=pickle.load(open(OUT + 'final_models.pkl','rb'))
sets={'BTS_test':te,'Random':harm}
rng=np.random.default_rng(7)
imp=[]; cal=[]; extra=[]
for key in [('Forest','full'),('Logit','full'),('CART','full')]:
    m=F[key]
    for sname,rows in sets.items():
        u,n,c,inv=aggregate(rows); base=logloss(n,c,m.predict(u))
        for g,cols in [('item',['item_id']),('position',['position']),('user profile',UF)]:
            r=rows.copy(); perm=rng.permutation(len(r))
            for col in cols: r[col]=rows[col][perm]
            uu,nn,cc,_=aggregate(r); imp.append({'model':key[0],'set':sname,'group':g,'increase_x1e4':(logloss(nn,cc,m.predict(uu))-base)*1e4})
        if key[0]!='CART':
            p=m.predict(u)[inv]; y=rows['click']
            q=pd.qcut(p,10,labels=False,duplicates='drop')
            d=pd.DataFrame({'q':q,'p':p,'y':y}).groupby('q').agg(pred=('p','mean'),obs=('y','mean'),n=('y','size')).reset_index()
            d['model']=key[0]; d['set']=sname; cal.append(d)
pd.DataFrame(imp).to_csv(OUT + 'importance.csv',index=False); pd.concat(cal).to_csv(OUT + 'calibration.csv',index=False)
print(pd.DataFrame(imp).pivot_table(index=['model','group'],columns='set',values='increase_x1e4').round(2))
# bootstrap: selected model (forest full) vs simplest model (ridge item+pos)
for a,b in [(('Logit','item+pos'),('Forest','full')),(('Forest','item+pos'),('Forest','full'))]:
    for sname,rows in sets.items():
        u,n,c,inv=aggregate(rows)
        sa,cnt=minute_loss_sums(rows,inv,F[a].predict(u)); sb,_=minute_loss_sums(rows,inv,F[b].predict(u))
        est,lo,hi=paired_bootstrap(sa,sb,cnt); extra.append({'A':'/'.join(a),'B':'/'.join(b),'set':sname,'diff':est*1e4,'lo':lo*1e4,'hi':hi*1e4})
# context gain on seen-code rows only (robustness), all methods
rows=harm[seen]; u,n,c,inv=aggregate(rows)
for meth in ['Logit','CART','Forest']:
    sa,cnt=minute_loss_sums(rows,inv,F[(meth,'item+pos')].predict(u)); sb,_=minute_loss_sums(rows,inv,F[(meth,'full')].predict(u))
    est,lo,hi=paired_bootstrap(sa,sb,cnt); extra.append({'A':meth+'/item+pos','B':meth+'/full','set':'Random_seen','diff':est*1e4,'lo':lo*1e4,'hi':hi*1e4})
E=pd.DataFrame(extra); E.to_csv(OUT + 'bootstrap_extra.csv',index=False); print(E.round(3))
# ridge M1 coefficients: item and position effects (odds ratios relative to mean)
m=F[('Logit','item+pos')].m; co=m.coef_[0]; it=co[:80]; po=co[80:83]
json.dump({'item_coef':it.tolist(),'pos_coef':po.tolist(),'item_or_range':[float(np.exp(it.min()-it.mean())),float(np.exp(it.max()-it.mean()))],'pos_or':np.exp(po-po.mean()).tolist()},open(OUT + 'ridge_m1_coef.json','w'))
print('item OR range', np.exp(it.min()-it.mean()), np.exp(it.max()-it.mean()), 'pos OR', np.exp(po-po.mean()))
