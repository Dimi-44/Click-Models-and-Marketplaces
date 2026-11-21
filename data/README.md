# Data

This folder contains the **Open Bandit Dataset** (ZOZOTOWN, campaign "All"), re-encoded and compressed so that it fits in the repository.

| File | Contents |
|---|---|
| `processed/bts_compact.npz` | All 12,357,200 BTS impressions |
| `processed/random_compact.npz` | All 1,374,327 Random impressions |
| `processed/bts_storage_codebooks.json` | Maps the original hashed user-feature values to the integer codes used here |
| `processed/checksums.json` | SHA256 of every array that `unpack_data.py` rebuilds |

Each row holds the fields the models use: `minute_id`, `item_id`, `position`, `click` and `user_feature_0` to `user_feature_3`. User codes are assigned on BTS only; a Random value never seen in BTS is stored as `65535`.

Run `python analysis/unpack_data.py` to write the arrays to `data/prepared/`. It also rebuilds the train/validation/test split and the learning-curve training order from [`../results/minute_split_manifest.csv`](../results/minute_split_manifest.csv), and checks every file against the recorded checksums.

## Source and license

- Saito, Aihara, Matsutani, Narita (2021). *Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.* [arXiv:2008.07146](https://arxiv.org/abs/2008.07146)
- Provided by ZOZO, Inc. under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Full dataset: https://research.zozo.com/data.html
- Changes made here: only the modelled columns are kept; hashed user features are mapped to integers; timestamps are reduced to calendar minutes. Propensity scores and user-item affinity columns are omitted.

The original archive used has SHA256 `ab082954506663f4136e15475f83164ee10cda5feea3cb36b5be5ee2104eb007`. Place it at `data/raw/ZOZOTOWN_All_original.zip` if you want to rebuild everything from the raw CSV files. The user features are anonymised hashes; no user identity can be inferred from them.
