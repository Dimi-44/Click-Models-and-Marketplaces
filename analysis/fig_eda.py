import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import json, numpy as np, pandas as pd
from figstyle import *
e=json.load(open(OUT + 'eda.json'))
gb=pd.read_csv(OUT + 'item_BTS.csv',index_col=0); gr=pd.read_csv(OUT + 'item_Random.csv',index_col=0)

# Fig 1: BTS exposure follows CTR (each dot an item)
fig,ax=plt.subplots(figsize=(4.3,2.75))
ax.scatter(gb.ctr*100, gb.share*100, s=30, color=BTS, edgecolor='white', linewidth=1, zorder=3, label='BTS: one dot per item')
ax.axhline(100/80, color=RANDOM, lw=2, zorder=2, label='Random: every item 1.25%')
ax.legend(loc='upper left', bbox_to_anchor=(0,0.92))
ax.set_xlabel('Item click-through rate under BTS (%)'); ax.set_ylabel('Share of impressions (%)')
ax.set_xlim(0.1, 0.65); ax.set_ylim(0, None)
ax.text(0.11, gb.share.max()*100, f'r = {e["corr_bts_share_vs_bts_ctr"]:.2f}', ha='left', va='top', color=INK, fontsize=10, fontweight='bold')
ax.set_title('Each dot is one of the 80 items')
save(fig,'eda_exposure')

# Fig 2: item CTR agrees across policies
fig,ax=plt.subplots(figsize=(3.0,2.85))
size=np.sqrt(gb.n/gb.n.max())*90+10
ax.plot([0,0.8],[0,0.8],color=BASE,lw=1,zorder=1)
ax.scatter(gb.ctr*100, gr.ctr*100, s=size, color=BTS, alpha=.75, edgecolor='white', linewidth=1, zorder=3)
ax.set_xlabel('Item CTR under BTS (%)'); ax.set_ylabel('Item CTR under Random (%)')
ax.set_xlim(0,0.8); ax.set_ylim(0,0.8); ax.set_aspect('equal')
ax.text(0.04,0.74,f'r = {e["corr_item_ctr_bts_vs_random"]:.2f}',color=INK,fontweight='bold')
ax.text(0.78,0.70,'equal CTR',color=MUTED,fontsize=7,ha='right',rotation=45)
ax.set_title('Same item, similar click rate')
save(fig,'eda_item_ctr')

# Fig 3: CTR decomposition bars
fig,ax=plt.subplots(figsize=(3.7,2.55))
vals=[e['BTS']['ctr']*100, e['bts_item_ctr_reweighted_to_random_mix']*100, e['Random']['ctr']*100]
labs=['BTS\nobserved','BTS item CTRs,\nRandom exposure','Random\nobserved']
cols=[BTS,'#A9B3E3',RANDOM]
b=ax.bar(range(3),vals,width=.55,color=cols,zorder=3)
for i,v in enumerate(vals): ax.text(i,v+0.012,f'{v:.3f}%',ha='center',color=INK,fontsize=10,fontweight='bold')
ax.set_xticks(range(3)); ax.set_xticklabels(labs,fontsize=8,color=INK); ax.set_ylim(0,0.58); ax.set_ylabel('Click-through rate (%)')
ax.grid(axis='x',visible=False); ax.set_title('Reweighting reproduces Random CTR')
save(fig,'eda_ctr_decomposition')

print('ok')
