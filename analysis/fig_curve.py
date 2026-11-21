import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import numpy as np, pandas as pd
from figstyle import *
C=pd.read_csv(OUT + 'learning_curve.csv')
fig,axs=plt.subplots(1,2,figsize=(6.5,2.2))
for ax,col,t in zip(axs,['bts_test','random'],['BTS internal test (same policy)','Random (new policy)']):
    for meth in ['B1','Logit','CART','Forest']:
        d=C[C.method==meth].sort_values('size')
        kw=dict(color=BASE,ls=(0,(3,2)),lw=1.6) if meth=='B1' else dict(color=METHOD[meth])
        ax.plot(d['size']/1e6,d[col]*1e4,marker='o' if meth!='B1' else None,ms=5,mec='white',mew=1.2,**kw,label=NAMES[meth])
    b0=C[(C.method=='B0')][col].mean()*1e4
    ax.axhline(b0,color=MUTED,lw=1,label='Constant CTR')
    ax.set_xscale('log'); ax.set_xticks([0.1,0.3,1,3,8.66]); ax.set_xticklabels(['0.1','0.3','1','3','8.66'])
    ax.set_xlabel('BTS training impressions (millions)'); ax.set_title(t)
axs[0].set_ylabel('Log loss (×10⁻⁴), lower is better')
axs[0].set_ylim(310.5,318.2); axs[1].set_ylim(228.8,236)
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=5,bbox_to_anchor=(.5,1.07),fontsize=7.5)
fig.tight_layout(w_pad=2.5); save(fig,'res_curve')
