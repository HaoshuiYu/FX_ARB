import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import Data, Dataset

DATA_DIR = Path(__file__).parent.parent.parent / 'data'

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
            return np.where(dates <= pd.Timestamp('2021-12-31'))[0]
        elif split == 'val':
            return np.where((dates >= pd.Timestamp('2022-01-01')) &
                            (dates <= pd.Timestamp('2023-12-31')))[0]
        elif split == 'test':
            return np.where(dates >= pd.Timestamp('2024-01-01'))[0]
        else:
            return np.arange(len(dates))

    def len(self):
        return len(self.indices)

    def get(self, idx):
        t        = self.indices[idx]
        x        = torch.tensor(self.X[t], dtype=torch.float32)
        nan_mask = torch.tensor(self.M[t], dtype=torch.bool)
        return Data(x=x, nan_mask=nan_mask)

    def get_targets(self, idx):
        t        = self.indices[idx]
        horizons = [3, 5, 10, 30]
        targets  = []
        for h in horizons:
            t_future = t + h
            if t_future >= len(self.X):
                targets.append(torch.zeros(3))
            else:
                future  = torch.tensor(self.X[t_future, :3, 0], dtype=torch.float32)
                current = torch.tensor(self.X[t,        :3, 0], dtype=torch.float32)
                targets.append(future - current)
        return torch.stack(targets)


if __name__ == '__main__':
    ds = FXGraphDataset(split='train')
    print(f"train samples: {ds.len()}")
    d  = ds.get(0)
    tg = ds.get_targets(0)
    print(f"x:        {d.x.shape}")
    print(f"nan_mask: {d.nan_mask.shape}")
    print(f"targets:  {tg.shape}")