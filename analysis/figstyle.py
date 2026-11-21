import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK = '#1C3046'; MUTED = '#617386'; GRID = '#E3E8EE'
BTS = '#5163C4'; RANDOM = '#D04A3A'
METHOD = {'Logit': '#008C9E', 'CART': '#C27C0E', 'Forest': '#8E55C4'}
NAMES = {'Logit': 'Ridge logistic', 'CART': 'Pruned tree (CART)', 'Forest': 'Random forest',
         'B0': 'Constant CTR', 'B1': 'Item×position CTR'}
BASE = '#9AA0A8'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': GRID, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9, 'axes.titlecolor': INK,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False, 'legend.fontsize': 7.5,
    'lines.linewidth': 2, 'lines.solid_capstyle': 'round', 'savefig.bbox': 'tight', 'savefig.pad_inches': .03,
    'pdf.fonttype': 42,
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True); os.makedirs(IMG, exist_ok=True)
    fig.savefig(FIG + name + '.pdf'); fig.savefig(IMG + name + '.png', dpi=160)
    plt.close(fig)
