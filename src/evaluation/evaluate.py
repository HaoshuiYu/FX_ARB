"""
evaluate.py — the judge. Scores the trained FXRegimeModel on the untouched
test period (2024) against baselines, and exports attention maps for the
regime-fingerprint analysis.

Baselines:
  zero            predict no correlation shift (strong null)
  mean-reversion  predict the negative of the most recent realized shift
  plain GRU       same task, raw target-node features, NO graph (the
                  ablation that isolates what the graph buys)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from src.training.train_graph import (FXRegimeModel, build_targets, build_edge_feats,
                                      valid_ts, resolve_data_dir, CORR_W,
                                      WINDOWS_PER_CHUNK, RESIDUAL)
from src.models.edge_gru import SEQ_LEN

CKPT       = Path('checkpoints/best_model.pt')
ATTN_OUT   = Path('checkpoints/attn_test.npz')
PAIR_NAMES = ['EUR-GBP', 'EUR-JPY', 'GBP-JPY']
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# plain-GRU baseline (no graph): raw features of the 3 target nodes
BASE_HIDDEN     = 24
BASE_MAX_EPOCHS = 100
BASE_PATIENCE   = 8


# helpers
def load_everything():
    data_dir = resolve_data_dir()
    X = np.load(data_dir / 'X.npy').astype(np.float32)
    M = np.load(data_dir / 'nan_mask.npy').astype(bool)
    dates = pd.to_datetime(pd.read_csv(data_dir / 'dates.csv')['0'].values)
    names = pd.read_csv(data_dir / 'node_names.csv')['0'].values

    ck = torch.load(CKPT, weights_only=False)
    mu, sd = ck['mu'], ck['sd']                      # train-period stats, frozen
    Xs = ((X - mu) / sd).astype(np.float32)
    Xs[M] = 0.0

    tgt = build_targets(X)
    EF = build_edge_feats(X)
    train_hi = int(np.where(dates <= pd.Timestamp('2021-12-31'))[0].max())
    val_hi   = int(np.where(dates <= pd.Timestamp('2023-12-31'))[0].max())
    ts_tr = valid_ts(0, train_hi, tgt)
    ts_va = valid_ts(train_hi + 1, val_hi, tgt)
    ts_te = valid_ts(val_hi + 1, len(dates) - 1, tgt)

    model = FXRegimeModel().to(DEVICE)
    model.load_state_dict(ck['model'])
    model.eval()
    return Xs, M, EF, tgt, ts_tr, ts_va, ts_te, dates, names, model


def chunked_preds(forward_fn, ts, n=WINDOWS_PER_CHUNK):
    """Run any chunk-wise forward over prediction dates ts; collect preds (and attn)."""
    preds, attns = [], []
    with torch.no_grad():
        for c0 in range(0, len(ts), n):
            block = ts[c0:c0 + n]
            p, a = forward_fn(block)
            preds.append(p)
            if a is not None:
                attns.append(a)
    preds = torch.cat(preds).cpu().numpy()
    attns = torch.cat(attns).cpu().numpy() if attns else None
    return preds, attns


def model_forward_factory(model, Xs, M, EF):
    def f(block):
        d0, d1 = int(block[0]) - SEQ_LEN + 1, int(block[-1])
        Xd = torch.from_numpy(Xs[d0:d1 + 1]).to(DEVICE)
        Md = torch.from_numpy(M[d0:d1 + 1]).to(DEVICE)
        Ed = torch.from_numpy(EF[d0:d1 + 1]).to(DEVICE)
        p, a = model.forward_chunk(Xd, Md, Ed)
        keep = block - block[0]
        return p[keep], a[keep + SEQ_LEN - 1]        # attn of each prediction DAY
    return f


def metrics(pred, true):
    out = {}
    for i, name in enumerate(PAIR_NAMES):
        p, t = pred[:, i], true[:, i]
        err = p - t
        nz = t != 0
        out[name] = dict(
            MAE=float(np.abs(err).mean()),
            RMSE=float(np.sqrt((err ** 2).mean())),
            DirAcc=float((np.sign(p[nz]) == np.sign(t[nz])).mean()),
            IC=float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-12 else 0.0,
        )
    out['MEAN'] = {k: float(np.mean([out[n][k] for n in PAIR_NAMES]))
                   for k in ['MAE', 'RMSE', 'DirAcc', 'IC']}
    return out


def print_table(rows):
    cols = ['MAE', 'RMSE', 'DirAcc', 'IC']
    hdr = f"{'model':<16}{'pair':<10}" + ''.join(f"{c:>9}" for c in cols)
    print('\n' + hdr + '\n' + '-' * len(hdr))
    for model_name, m in rows:
        for pair in PAIR_NAMES + ['MEAN']:
            v = m[pair]
            print(f"{model_name:<16}{pair:<10}" + ''.join(f"{v[c]:>9.4f}" for c in cols))
        print('-' * len(hdr))


# baseline general
def baseline_zero(ts, tgt):
    return np.zeros((len(ts), 3), dtype=np.float32)


def baseline_mean_reversion(ts, X):
    """Predict the NEGATIVE of the last realized CORR_W-day shift."""
    pred = np.zeros((len(ts), 3), dtype=np.float32)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for p, (a, b) in enumerate(pairs):
        trail = pd.Series(X[:, a, 0]).rolling(CORR_W).corr(pd.Series(X[:, b, 0])).to_numpy()
        last_shift = trail[ts] - trail[ts - CORR_W]
        pred[:, p] = -np.nan_to_num(last_shift)
    return pred


class PlainGRU(nn.Module):
    """No-graph control: SAME information as the graph model (raw target-node
    features + the 3 pairs' [rho_trail, d_rho] = 15 inputs), no graph. Any
    edge the graph model shows over this is attributable to architecture."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(15, BASE_HIDDEN, batch_first=True)
        self.drop = nn.Dropout(0.4)
        self.head = nn.Linear(BASE_HIDDEN, 3)

    def forward(self, w):                            # w: [B, 20, 9]
        _, h = self.gru(self.drop(w))
        return self.head(self.drop(h.squeeze(0)))


def raw_windows(Xs, EF, block):
    d0, d1 = int(block[0]) - SEQ_LEN + 1, int(block[-1])
    raw = Xs[d0:d1 + 1, :3, :].reshape(d1 - d0 + 1, 9)
    ef  = EF[d0:d1 + 1, [0, 2, 4], :].reshape(d1 - d0 + 1, 6)   # one per pair
    days = torch.from_numpy(np.concatenate([raw, ef], axis=1))
    w = torch.stack([days[i:i + SEQ_LEN] for i in range(len(days) - SEQ_LEN + 1)])
    return w[(block - block[0])].to(DEVICE)


def train_plain_gru(Xs, EF, tgt, ts_tr, ts_va):
    torch.manual_seed(0)
    net = PlainGRU().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    best, bad, best_state = float('inf'), 0, None
    for ep in range(BASE_MAX_EPOCHS):
        net.train()
        for c0 in range(0, len(ts_tr), WINDOWS_PER_CHUNK):
            block = ts_tr[c0:c0 + WINDOWS_PER_CHUNK]
            y = torch.from_numpy(tgt[block]).to(DEVICE)
            loss = nn.functional.huber_loss(net(raw_windows(Xs, EF, block)), y)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = float(np.mean([
                float(nn.functional.huber_loss(
                    net(raw_windows(Xs, EF, ts_va[c0:c0 + WINDOWS_PER_CHUNK])),
                    torch.from_numpy(tgt[ts_va[c0:c0 + WINDOWS_PER_CHUNK]]).to(DEVICE)))
                for c0 in range(0, len(ts_va), WINDOWS_PER_CHUNK)]))
        if va < best:
            best, bad, best_state = va, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= BASE_PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


# attention diagnostics
def attention_report(attns, names):
    """attns: [D, 6, H, 25] over test days."""
    A = attns.mean(axis=2)                           # [D, 6, 25] head-avg
    edges = ['EUR>GBP', 'GBP>EUR', 'EUR>JPY', 'JPY>EUR', 'GBP>JPY', 'JPY>GBP']
    print("\nattention readout (test period)")
    for e in range(6):
        mean_a = A[:, e, :].mean(axis=0)
        top = np.argsort(mean_a)[::-1][:3]
        survivors = float((A[:, e, :] > 0).sum(axis=1).mean())
        tops = ', '.join(f"{names[i]} ({mean_a[i]:.2f})" for i in top)
        print(f"  {edges[e]:<8} avg survivors/day {survivors:4.1f} | top: {tops}")


# main
def main():
    Xs, M, EF, tgt, ts_tr, ts_va, ts_te, dates, names, model = load_everything()
    X_raw = np.load(resolve_data_dir() / 'X.npy').astype(np.float32)
    true = tgt[ts_te]
    print(f"test predictions: {len(ts_te)}  ({dates[ts_te[0]].date()} → {dates[ts_te[-1]].date()})")

    preds, attns = chunked_preds(model_forward_factory(model, Xs, M, EF), ts_te)
    rows = [('graph model', metrics(preds, true)),
            ('zero', metrics(baseline_zero(ts_te, tgt), true))]
    if RESIDUAL:
        print("target is MR-residual: mean-reversion baseline == zero, row omitted")
    else:
        rows.append(('mean-reversion', metrics(baseline_mean_reversion(ts_te, X_raw), true)))

    print("training plain-GRU baseline (no graph)...")
    base = train_plain_gru(Xs, EF, tgt, ts_tr, ts_va)
    base_preds, _ = chunked_preds(lambda b: (base(raw_windows(Xs, EF, b)), None), ts_te)
    rows.insert(1, ('plain GRU', metrics(base_preds, true)))

    print_table(rows)
    attention_report(attns, names)
    np.savez(ATTN_OUT, attn=attns, dates=dates[ts_te].astype(str), preds=preds, true=true)
    print(f"\nattention maps + predictions saved -> {ATTN_OUT} (for inspect_attention.ipynb)")


if __name__ == '__main__':
    main()
