import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import Data, Dataset

DATA_DIR = Path(__file__).parent.parent.parent / 'data'

# forecast horizons in trading days
HORIZONS = [3, 5, 10, 30]

class FXGraphDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        print(f"Loading data...")
        self.X          = np.load(DATA_DIR / 'X.npy')
        self.M          = np.load(DATA_DIR / 'nan_mask.npy')
        self.dates      = pd.read_csv(DATA_DIR / 'dates.csv')['0'].values
        self.node_names = pd.read_csv(DATA_DIR / 'node_names.csv')['0'].values
        self.indices    = self._get_indices(split)
        print(f"Ready — {split} split: {len(self.indices):,} samples")

    def _get_indices(self, split):
        dates = pd.to_datetime(self.dates)
        if split == 'train':
            idx = np.where(dates <= pd.Timestamp('2021-12-31'))[0]
        elif split == 'val':
            idx = np.where((dates >= pd.Timestamp('2022-01-01')) &
                           (dates <= pd.Timestamp('2023-12-31')))[0]
        elif split == 'test':
            idx = np.where(dates >= pd.Timestamp('2024-01-01'))[0]
        else:
            idx = np.arange(len(dates))
        # PURGE: drop samples whose longest forecast horizon would cross the
        # end of this split. Two birds:
        #   1. tail of the dataset — no future data exists, so the old code
        #      emitted fake all-zero targets there
        #   2. split boundaries — a train sample on 2021-12-30 with h=30 had
        #      its label computed from Jan 2022 (validation period) data.
        #      That is lookahead leakage across splits (de Prado purging).
        if len(idx) > 0:
            idx = idx[idx + max(HORIZONS) <= idx.max()]
        return idx

    def len(self):
        return len(self.indices)

    def get(self, idx):
        t        = self.indices[idx]
        x        = torch.tensor(self.X[t], dtype=torch.float32)
        nan_mask = torch.tensor(self.M[t], dtype=torch.bool)
        return Data(x=x, nan_mask=nan_mask)

    def get_targets(self, idx):
        t       = self.indices[idx]
        targets = []
        for h in HORIZONS:
            # FIX: the h-day move is the ACCUMULATION of daily returns over
            # the window (t, t+h], i.e. sum of pct_change rows t+1 .. t+h.
            # The old code did X[t+h,:,0] - X[t,:,0] — a difference of two
            # 1-day return snapshots, which is ~0 for any steady trend and
            # does not measure the horizon move at all.
            # (Exact if pct_change is log returns; close approximation for
            # small daily simple returns. Purged indices guarantee t+h is
            # always in-bounds and inside this split.)
            window = self.X[t + 1 : t + h + 1, :3, 0]           # [h, 3]
            targets.append(torch.tensor(window.sum(axis=0), dtype=torch.float32))
        return torch.stack(targets)                              # [4, 3]


if __name__ == '__main__':
    ds = FXGraphDataset(split='train')
    print(f"train samples: {ds.len()}")
    d  = ds.get(0)
    tg = ds.get_targets(0)
    print(f"x:        {d.x.shape}")
    print(f"nan_mask: {d.nan_mask.shape}")
    print(f"targets:  {tg.shape}")