"""Paths used by the analysis scripts. Override with environment variables if needed."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Output folder of `marketplace_ml.prepare_full` (compact arrays + split indices)
PREPARED = str(Path(os.environ.get('MML_PREPARED_DIR', REPO / 'data' / 'prepared'))) + '/'
# Original Open Bandit Dataset "All" campaign ZIP
RAW_ZIP = str(Path(os.environ.get('MML_RAW_ZIP', REPO / 'data' / 'raw' / 'ZOZOTOWN_All_original.zip')))
OUT = str(REPO / 'results') + '/'           # result tables
DECK = str(REPO / 'presentation') + '/'     # LaTeX deck (numbers.tex, tab_*.tex)
FIG = DECK + 'figures/'                     # PDF figures used by the deck
IMG = str(REPO / 'docs' / 'img') + '/'      # PNG copies for the README
