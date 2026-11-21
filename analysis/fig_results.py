import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import json, numpy as np, pandas as pd
from figstyle import *
B=pd.read_csv(OUT + 'bootstrap.csv'); M=pd.read_csv(OUT + 'metrics.csv')
LL=M.set_index(['method','features','set']).log_loss
meths=['Logit','CART','Forest']

def panel_dots(ax, rows, title, xlim, ref=None, reflabel=None):
    for i,(lab,col,est,lo,hi) in enumerate(rows):
        y=len(rows)-1-i
        ax.plot([lo,hi],[y,y],color=col,lw=2.6,solid_capstyle='round',zorder=2)
        ax.scatter([est],[y],s=80,color=col,edgecolor='white',linewidth=1.5,zorder=3)
        ax.text(hi+0.15 if est>=0 else hi+0.15,y,f'{est:+.1f}',va='center',color=INK,fontsize=8.5,fontweight='bold')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1],color=INK,fontsize=8.5)
    ax.axvline(0,color=MUTED,lw=1,zorder=1); ax.set_xlim(*xlim); ax.set_ylim(-.6,len(rows)-.4)
    if ref is not None:
        ax.axvline(ref,color=BASE,lw=1.5,ls=(0,(3,2)),zorder=1)
        ax.text(ref+0.12,-.5,reflabel,color=MUTED,fontsize=7,ha='left',va='bottom')
    ax.set_title(title); ax.grid(axis='y',visible=False)

# Fig: gains over constant CTR
fig,axs=plt.subplots(1,2,figsize=(6.5,2.0),sharex=True)
for ax,s,t in zip(axs,['BTS_test','Random'],['BTS internal test (same policy)','Random (new policy)']):
    rows=[]
    for m in meths:
        r=B[(B.contrast==f'gain_vs_B0_{m}')&(B.set==s)].iloc[0]; rows.append((NAMES[m],METHOD[m],r.diff_A_minus_B_x1e4,r.lo,r.hi))
    b1=(LL[('B0','item+pos',s)]-LL[('B1','item+pos',s)])*1e4
    panel_dots(ax,rows,t,(0,7.6),b1,'item×position CTR table')
for ax in axs: ax.set_xlabel('Gain over constant CTR (×10⁻⁴)')
fig.tight_layout(w_pad=2.5); save(fig,'res_gains')

# Fig: context gain
fig,axs=plt.subplots(1,2,figsize=(6.5,1.9),sharex=True)
for ax,s,t in zip(axs,['BTS_test','Random'],['BTS internal test (same policy)','Random (new policy)']):
    rows=[]
    for m in meths:
        r=B[(B.contrast==f'context_gain_{m}')&(B.set==s)].iloc[0]; rows.append((NAMES[m],METHOD[m],r.diff_A_minus_B_x1e4,r.lo,r.hi))
    panel_dots(ax,rows,t,(-1.6,5.8))
for ax in axs: ax.set_xlabel('Gain from user context (×10⁻⁴)')
fig.tight_layout(w_pad=2.5); save(fig,'res_context')

# Fig: calibration, final forest, both logs in one panel
C=pd.read_csv(OUT + 'calibration.csv')
fig,ax=plt.subplots(figsize=(3.9,2.95))
ax.plot([0,1.0],[0,1.0],color=BASE,lw=1,zorder=1,ls=(0,(3,2)),label='perfect calibration')
for s_,col,lab in [('BTS_test',BTS,'BTS test'),('Random',RANDOM,'Random')]:
    d=C[(C.model=='Forest')&(C.set==s_)]
    ax.plot(d.pred*100,d.obs*100,color=col,marker='o',ms=5,mec='white',mew=1.2,zorder=3,label=lab)
d=C[(C.model=='Forest')&(C.set=='Random')].sort_values('q').iloc[-1]
ax.annotate('most confident 10%\nbreaks under Random',(d.pred*100,d.obs*100),xytext=(0.97,0.08),textcoords='axes fraction',ha='right',va='bottom',fontsize=7.5,color=INK,arrowprops=dict(arrowstyle='-',color=MUTED,lw=.8))
ax.set_xlim(0,1.0); ax.set_ylim(0,1.0); ax.set_xlabel('Predicted CTR, decile mean (%)'); ax.set_ylabel('Observed CTR (%)')
ax.legend(loc='upper left'); ax.set_title('Random forest: calibration by decile')
save(fig,'res_calibration')

# Fig: item effects from ridge (sorted odds ratios)
co=json.load(open(OUT + 'ridge_m1_coef.json')); it=np.array(co['item_coef']); orr=np.exp(it-it.mean())
gb=pd.read_csv(OUT + 'item_BTS.csv',index_col=0)
o=np.argsort(orr)
fig,ax=plt.subplots(figsize=(3.6,2.3))
ax.bar(range(80),orr[o]-1,bottom=1,width=.7,color=METHOD['Logit'],zorder=3)
ax.axhline(1,color=MUTED,lw=1)
ax.set_yscale('log'); ax.yaxis.set_minor_formatter(plt.NullFormatter()); ax.set_yticks([0.5,0.75,1,1.5,2]); ax.set_yticklabels(['0.5×','0.75×','1×','1.5×','2×'])
ax.set_xticks([]); ax.set_xlabel('80 items, sorted'); ax.set_ylabel('Click odds vs. average item')
pos=np.exp(np.array(co['pos_coef'])-np.mean(co['pos_coef']))
ax.set_title('Item identity: 3.6× spread in click odds'); ax.grid(axis='x',visible=False)
save(fig,'res_items')
print(pos)
