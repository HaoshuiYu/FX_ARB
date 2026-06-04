"""
train_graph.py — end-to-end training for the FX regime model.

Pipeline per step:
  raw days [25, 3] -> FXEdgeTransformer (per day) -> edge states [6, 64]
  -> 20-day windows -> EdgeGRU -> 3 correlation-shift predictions
  -> Huber loss vs forward corr-shift targets + orthogonality penalty.

Anti-overfit stack: tiny dims, dropout, shared FFN/K/V, differential
weight decay, early stopping w/ best-checkpoint restore, purged targets,
train-stats-only feature standardization, grad clipping.

Run from repo root:  python -m src.training.train_graph
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from src.models.graph_transformer import FXEdgeTransformer, NUM_EDGES
from src.models.edge_gru import EdgeGRU, SEQ_LEN, PAIR_EDGES

# config
CORR_W        = 20          # realized-correlation window; target = corr(t+1..t+W) - corr(t-W+1..t)
TARGET_PAIRS  = [(0, 1), (0, 2), (1, 2)]   # node pairs: EUR-GBP, EUR-JPY, GBP-JPY
LR            = 3e-4
WD_TRANSFORMER= 1e-3        # differential weight decay: lighter on the spatial encoder
WD_GRU        = 1e-2        # heavier where memorization risk is highest
ORTHO_LAMBDA  = 0.05        # orthogonality penalty weight (first knob if attention collapses)
HUBER_PCT     = 85          # Huber delta = this percentile of |train target|, per pair
MAX_EPOCHS    = 200
MIN_EPOCHS    = 15          # floor: early stopping cannot fire before this
PATIENCE      = 12          # epochs of no val improvement before stopping
GRAD_CLIP     = 1.0
WINDOWS_PER_CHUNK = 64      # contiguous prediction dates per step (shares per-day compute)
SEED          = 0
CKPT_DIR      = Path('checkpoints')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# modelling proper
class FXRegimeModel(nn.Module):
    """Transformer (spatial, per day) + EdgeGRU (temporal) end to end."""
    def __init__(self):
        super().__init__()
        self.transformer = FXEdgeTransformer()
        self.gru = EdgeGRU()

    def encode_days(self, X_days, M_days):
        """[D, 25, 3], [D, 25] -> edge states [D, 6, 64], attn [D, 6, H, 25]"""
        states, attns = [], []
        for d in range(X_days.shape[0]):
            e, a = self.transformer(X_days[d], M_days[d])
            states.append(e); attns.append(a)
        return torch.stack(states), torch.stack(attns)

    @staticmethod
    def windows_from_states(states):
        """[D, 6, 64] -> [D-SEQ_LEN+1, SEQ_LEN, 6, 64] sliding windows."""
        D = states.shape[0]
        return torch.stack([states[i:i + SEQ_LEN] for i in range(D - SEQ_LEN + 1)])

    def forward_chunk(self, X_days, M_days):
        states, attns = self.encode_days(X_days, M_days)
        preds = self.gru(self.windows_from_states(states))   # [N, 3]
        return preds, attns


# data
def resolve_data_dir():
    for c in (Path(__file__).parent.parent.parent / 'data', Path('data')):
        if (c / 'X.npy').exists():
            return c
    raise FileNotFoundError('X.npy not found (expected in repo data/)')


def build_targets(X):
    """Forward shift in CORR_W-day realized correlation per pair. [T, 3], NaN where undefined."""
    T = X.shape[0]
    tgt = np.full((T, len(TARGET_PAIRS)), np.nan, dtype=np.float32)
    for p, (a, b) in enumerate(TARGET_PAIRS):
        ra, rb = pd.Series(X[:, a, 0]), pd.Series(X[:, b, 0])
        trail = ra.rolling(CORR_W).corr(rb).to_numpy()        # corr over (t-W+1 .. t]
        fwd = np.full(T, np.nan, dtype=np.float64)
        fwd[:T - CORR_W] = trail[CORR_W:]                      # corr over (t+1 .. t+W]
        tgt[:, p] = (fwd - trail).astype(np.float32)
    return tgt


def standardize(X, M, train_hi):
    """Per-feature standardization from TRAIN-period unmasked cells only."""
    mu, sd = np.zeros(3, np.float32), np.ones(3, np.float32)
    for f in range(3):
        cells = X[:train_hi + 1, :, f][~M[:train_hi + 1]]
        mu[f], sd[f] = cells.mean(), max(cells.std(), 1e-8)
    Xs = (X - mu) / sd
    Xs[M] = 0.0                                               # keep masked cells inert
    return Xs.astype(np.float32), mu, sd


def valid_ts(split_lo, split_hi, tgt):
    """Prediction dates t with full lookback and a leak-free forward target."""
    lo = max(split_lo, SEQ_LEN - 1, CORR_W - 1)
    hi = split_hi - CORR_W                                    # purge: t+W stays inside split
    ts = np.arange(lo, hi + 1)
    return ts[~np.isnan(tgt[ts]).any(axis=1)]


def load_real(data_dir):
    X = np.load(data_dir / 'X.npy').astype(np.float32)
    M = np.load(data_dir / 'nan_mask.npy').astype(bool)
    dates = pd.to_datetime(pd.read_csv(data_dir / 'dates.csv')['0'].values)
    train_hi = int(np.where(dates <= pd.Timestamp('2021-12-31'))[0].max())
    val_hi   = int(np.where(dates <= pd.Timestamp('2023-12-31'))[0].max())
    tgt = build_targets(X)
    Xs, mu, sd = standardize(X, M, train_hi)
    return Xs, M, tgt, valid_ts(0, train_hi, tgt), valid_ts(train_hi + 1, val_hi, tgt), mu, sd


# huber loss and joint penalties
def huber(pred, target, delta):
    err = (pred - target).abs()
    quad = 0.5 * err ** 2
    lin = delta * (err - 0.5 * delta)
    return torch.where(err <= delta, quad, lin).mean()


def ortho_penalty(attns, eps=1e-8):
    """Push the 6 edges' attention patterns apart. attns: [D, 6, H, 25]."""
    A = attns.mean(dim=2)                  
    A = A / (A.norm(dim=-1, keepdim=True) + eps)
    G = torch.bmm(A, A.transpose(1, 2)) # [D, 6, 6] cosine grid
    off = G - torch.eye(NUM_EDGES, device=G.device)
    return (off ** 2).mean()


# training
def run_split(model, Xs, M, tgt, ts, delta, train=False, opt=None, rng=None):
    """One pass over the given prediction dates, in contiguous chunks."""
    model.train() if train else model.eval()
    losses, n = [], WINDOWS_PER_CHUNK
    offset = int(rng.integers(0, n)) if train else 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for c0 in range(offset, len(ts), n):
            block = ts[c0:c0 + n]
            if len(block) == 0:
                continue
            d0, d1 = int(block[0]) - SEQ_LEN + 1, int(block[-1])
            Xd = torch.from_numpy(Xs[d0:d1 + 1]).to(DEVICE)
            Md = torch.from_numpy(M[d0:d1 + 1]).to(DEVICE)
            y  = torch.from_numpy(tgt[block]).to(DEVICE)
            preds, attns = model.forward_chunk(Xd, Md)
            # select the windows whose end-day is in block (handles gaps safely)
            preds = preds[(block - block[0])]
            loss = huber(preds, y, delta) + ORTHO_LAMBDA * ortho_penalty(attns)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def build_optimizer(model):
    groups = {('t', True): [], ('t', False): [], ('g', True): [], ('g', False): []}
    for name, p in model.named_parameters():
        part = 't' if name.startswith('transformer') else 'g'
        decay = p.ndim >= 2                                   # no decay on biases/norms/attn_bias
        groups[(part, decay)].append(p)
    return torch.optim.AdamW([
        {'params': groups[('t', True)],  'weight_decay': WD_TRANSFORMER},
        {'params': groups[('t', False)], 'weight_decay': 0.0},
        {'params': groups[('g', True)],  'weight_decay': WD_GRU},
        {'params': groups[('g', False)], 'weight_decay': 0.0},
    ], lr=LR)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    Xs, M, tgt, ts_tr, ts_va, mu, sd = load_real(resolve_data_dir())
    delta = torch.from_numpy(
        np.nanpercentile(np.abs(tgt[ts_tr]), HUBER_PCT, axis=0).astype(np.float32)).to(DEVICE)

    model = FXRegimeModel().to(DEVICE)
    opt = build_optimizer(model)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device={DEVICE} | params={n_par:,} | train={len(ts_tr):,} val={len(ts_va):,} "
          f"| huber delta per pair={delta.cpu().numpy().round(4)}")

    max_ep = MAX_EPOCHS
    best_val, best_ep, bad, CKPT = float('inf'), -1, 0, CKPT_DIR / 'best_model.pt'
    CKPT_DIR.mkdir(exist_ok=True)

    for ep in range(1, max_ep + 1):
        tr = run_split(model, Xs, M, tgt, ts_tr, delta, train=True, opt=opt, rng=rng)
        va = run_split(model, Xs, M, tgt, ts_va, delta)
        flag = ''
        if va < best_val:
            best_val, best_ep, bad = va, ep, 0
            torch.save({'model': model.state_dict(), 'mu': mu, 'sd': sd,
                        'epoch': ep, 'val': va}, CKPT)
            flag = '  <- best (saved)'
        else:
            bad += 1
        print(f"epoch {ep:3d} | train {tr:.5f} | val {va:.5f}{flag}")
        if ep >= MIN_EPOCHS and bad >= PATIENCE:
            print(f"early stop: no val improvement for {PATIENCE} epochs")
            break

    model.load_state_dict(torch.load(CKPT, weights_only=False)['model'])
    print(f"done. best val {best_val:.5f} @ epoch {best_ep} | checkpoint: {CKPT}")


if __name__ == '__main__':
    main()
