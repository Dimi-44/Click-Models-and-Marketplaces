"""Rebuild the analysis inputs in data/prepared/ from the compressed data shipped in data/processed/.

The BTS/Random compact arrays are stored compressed. The train/validation/test indices and the
learning-curve training order are regenerated exactly from the frozen minute-block split manifest
(results/minute_split_manifest.csv) with the original preparation code, and checked against
recorded SHA256 checksums.
"""
import os, sys, json, hashlib, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import REPO, PREPARED, OUT
sys.path.insert(0, str(REPO / 'src'))
from marketplace_ml.full_data import manifest_minutes, split_indices
from marketplace_ml.prepare_full import SEED, minute_nested_order

SRC = REPO / 'data' / 'processed'
DST = PREPARED
os.makedirs(DST, exist_ok=True)
expected = json.load(open(SRC / 'checksums.json'))


def save(name, arr):
    np.save(DST + name + '.npy', arr)
    digest = hashlib.sha256(open(DST + name + '.npy', 'rb').read()).hexdigest()
    status = 'ok' if digest == expected[name] else 'CHECKSUM MISMATCH'
    print(f'{name:28s} {len(arr):>12,} rows  {status}')
    if status != 'ok':
        sys.exit(1)


bts = np.load(SRC / 'bts_compact.npz')['a']; save('bts_compact', bts)
save('random_compact', np.load(SRC / 'random_compact.npz')['a'])
shutil.copy(SRC / 'bts_storage_codebooks.json', DST + 'bts_storage_codebooks.json')
split = manifest_minutes(REPO / 'results' / 'minute_split_manifest.csv')
for part in ('train', 'validation', 'internal_test'):
    save(f'indices_{part}', split_indices(bts, split, part))
train = np.load(DST + 'indices_train.npy')
save('train_order_minute_rep0', minute_nested_order(bts, train, SEED + 10_000))
print('Prepared data written to', DST)
