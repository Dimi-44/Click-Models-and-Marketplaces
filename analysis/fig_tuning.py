import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import numpy as np, pandas as pd
from figstyle import *
G=pd.read_csv(OUT + 'tuning_grid.csv'); E=pd.concat([pd.read_csv(OUT + 'forest_ext.csv'),pd.read_csv(OUT + 'forest_ext2.csv')])
fig,axs=plt.subplots(1,3,figsize=(7.8,2.75),gridspec_kw={'width_ratios':[1,1,1.45]})
ax=axs[0]; g=G[G.method=='Logit']
for f,ls,lab in [('full','-','+ user context'),('item+pos','--','item + position')]:
    d=g[g.features==f].sort_values('C'); ax.plot(d.C,d.val_log_loss*1e4,color=METHOD['Logit'],ls=ls,marker='o',ms=4,label=lab)
ax.set_xscale('log'); ax.set_xlabel('C (inverse penalty)'); ax.set_ylabel('BTS validation log loss (×10⁻⁴)'); ax.set_title('Ridge logistic'); ax.legend(fontsize=7,loc='center right')
ax=axs[1]; g=G[(G.method=='CART')&(G.features=='full')]
d=g.groupby('leaves').val_log_loss.min().reset_index()
ax.plot(d.leaves,d.val_log_loss*1e4,color=METHOD['CART'],marker='o',ms=3.5)
b=d.loc[d.val_log_loss.idxmin()]; ax.scatter([b.leaves],[b.val_log_loss*1e4],s=80,color=METHOD['CART'],edgecolor=INK,zorder=4)
ax.annotate(f'selected: {int(b.leaves):,} leaves',(b.leaves,b.val_log_loss*1e4),xytext=(12,22),textcoords='offset points',fontsize=8.5,color=INK,arrowprops=dict(arrowstyle='-',color=MUTED,lw=.8))
ax.set_xscale('log'); ax.set_xlabel('Leaves after pruning'); ax.set_title('CART (cost-complexity path)')
ax=axs[2]; g=G[(G.method=='Forest')&(G.features=='full')][['max_features','min_leaf_frac','val_log_loss']]
g=pd.concat([g,E[['max_features','min_leaf_frac','val_log_loss']]]).drop_duplicates(['max_features','min_leaf_frac'],keep='last')
P=g.pivot_table(index='max_features',columns='min_leaf_frac',values='val_log_loss')*1e4
P=P.loc[[0.05,0.1,0.2],[0.0,3e-6,1e-5,3e-5,1e-4]]
xl=['grown','3e-6','1e-5','3e-5','1e-4']
for mf,col,lw in [(0.05,'#C9B3E6',1.6),(0.1,'#8E55C4',2.4),(0.2,'#4B2A73',1.6)]:
    row=P.loc[mf]; ok=~row.isna(); xi=[i for i,o in enumerate(ok) if o]
    ax.plot(xi,row[ok].values,marker='o',ms=4,lw=lw,color=col,label=f'{int(mf*100)}% of features')
bi=np.unravel_index(np.nanargmin(P.values),P.shape)
ax.scatter([bi[1]],[P.values[bi]],s=90,color='#8E55C4',edgecolor=INK,zorder=5)
ax.annotate('selected',(bi[1],P.values[bi]),xytext=(10,-2),textcoords='offset points',fontsize=8,color=INK,va='center')
ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl,fontsize=6.8)
ax.set_xlabel('Minimum leaf size (share of rows)'); ax.set_title('Random forest')
ax.legend(fontsize=7,loc='upper right')
fig.tight_layout(w_pad=1.8); save(fig,'bk_tuning')
