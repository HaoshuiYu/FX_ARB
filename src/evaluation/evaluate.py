"""
evaluate.py — the judge. Scores the trained FXRegimeModel on the untouched
out-of-sample test period against baselines, and exports attention maps for the
regime-fingerprint analysis.

Baselines:
  zero            predict no correlation shift (strong null)
  calibrated-lin  OLS on trailing correlation features (the linear mechanics bar)
  plain GRU       same task, raw target-node features, NO graph (the
                  ablation that isolates what the graph buys)

Metrics per pair + mean: MAE, RMSE, directional accuracy, Pearson IC.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from src.training.train_graph import (FXRegimeModel, build_targets, build_edge_feats, trailing_corr,
                                      valid_ts, resolve_data_dir, CORR_W, HORIZON, WINDOWS_PER_CHUNK)
from src.models.edge_gru import SEQ_LEN
from src.evaluation.significance_test import perm_test, ic

CKPT       = Path('checkpoints/best_model.pt')
ATTN_OUT   = Path('checkpoints/attn_test.npz')
PAIR_NAMES = ['EUR-GBP', 'EUR-JPY', 'GBP-JPY']
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# plain-GRU baseline (no graph): raw features of the 3 target nodes
BASE_HIDDEN     = 24
BASE_MAX_EPOCHS = 100
BASE_PATIENCE   = 8

def load_everything():
    data_dir = resolve_data_dir()
    X = np.load(data_dir / 'X.npy').astype(np.float32)
    M = np.load(data_dir / 'nan_mask.npy').astype(bool)
    dates = pd.to_datetime(pd.read_csv(data_dir / 'dates.csv')['0'].values)
    names = pd.read_csv(data_dir / 'node_names.csv')['0'].values

    ck = torch.load(CKPT, weights_only=False)
    mu, sd = ck['mu'], ck['sd'] # train-period stats, frozen
    Xs = ((X - mu) / sd).astype(np.float32)
    Xs[M] = 0.0

    train_hi = int(np.where(dates <= pd.Timestamp('2021-12-31'))[0].max())
    val_hi   = int(np.where(dates <= pd.Timestamp('2023-12-31'))[0].max())
    tgt = build_targets(X, train_hi)
    EF = build_edge_feats(X)
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
        return p[keep], a[keep + SEQ_LEN - 1]   
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


# baseline
def baseline_zero(ts, tgt):
    return np.zeros((len(ts), 3), dtype=np.float32)


def baseline_calibrated_lin(X, tgt, ts_tr, ts_te):
    """The linear mechanics bench: an OLS regression of the
    target on [1, trailing correlation, recent change], fit on train and scored
    on test. 
    """
    trail = trailing_corr(X)
    ls = np.full_like(trail, np.nan)
    ls[CORR_W:] = trail[CORR_W:] - trail[:-CORR_W]
    pred = np.zeros((len(ts_te), tgt.shape[1]), dtype=np.float32)
    for p in range(tgt.shape[1]):
        m = (np.isfinite(tgt[ts_tr, p]) & np.isfinite(trail[ts_tr, p])
             & np.isfinite(ls[ts_tr, p]))
        A = np.column_stack([np.ones(m.sum()), trail[ts_tr, p][m], ls[ts_tr, p][m]])
        coef, *_ = np.linalg.lstsq(A, tgt[ts_tr, p][m], rcond=None)
        pred[:, p] = (coef[0] + coef[1] * np.nan_to_num(trail[ts_te, p])
                      + coef[2] * np.nan_to_num(ls[ts_te, p]))
    return pred

class PlainGRU(nn.Module):
    """No-graph: same information as the graph model (raw target-node
    features + the 3 pairs' [rho_trail, d_rho] = 15 inputs)
    """
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(15, BASE_HIDDEN, batch_first=True)
        self.drop = nn.Dropout(0.4)
        self.head = nn.Linear(BASE_HIDDEN, 3)

    def forward(self, w):                            # w: [B, SEQ_LEN, 15]
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


def significance_block(true, named_preds):
    """Final verdict: (1) block-permutation p per model: is its IC
    distinguishable from shuffled noise? (2) block-bootstrap of the IC margin
    graph-minus-calibrated-lin: does the architecture beat the 3-number
    mechanics formula? 
    """
    rng = np.random.default_rng(0)
    print("\nSIGNIFICANCE (block permutation, 2000 shuffles)")
    for name, pred in named_preds:
        ps, ics = [], []
        for p in range(true.shape[1]):
            r, pv = perm_test(pred[:, p], true[:, p], rng)
            ps.append(pv); ics.append(r)
        print(f"  {name:<15} mean IC {np.mean(ics):+.4f}   per-pair p = "
              + ", ".join(f"{v:.3f}" for v in ps))

    g = dict(named_preds)['graph model']; c = dict(named_preds)['calibrated-lin']
    T = len(true); BLK = CORR_W + HORIZON; blocks = np.arange(0, T, BLK); diffs = []
    for _ in range(2000):
        sel = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([np.arange(s, min(s + BLK, T)) for s in sel])
        d = [ic(g[idx, p], true[idx, p]) - ic(c[idx, p], true[idx, p])
             for p in range(true.shape[1])]
        diffs.append(np.mean(d))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  graph minus calibrated-lin: mean IC margin {diffs.mean():+.4f}, "
          f"95% CI [{lo:+.3f}, {hi:+.3f}], P(margin>0) = {(diffs > 0).mean():.3f}")


def quarterly_ic(dates_te, name, pred, true):
    """Per-quarter IC (mean across pairs). Exposes episode concentration:
    if one quarter carries the whole score, the 'skill' is one event."""
    q = pd.PeriodIndex(pd.DatetimeIndex(dates_te), freq='Q')
    out = [f"quarterly IC — {name}:"]
    for per in q.unique():
        m = (q == per)
        if m.sum() < 8:
            out.append(f"  {per} n={int(m.sum())} (too few)")
            continue
        ics = []
        for p in range(true.shape[1]):
            tp, pp = true[m, p], pred[m, p]
            ics.append(np.corrcoef(pp, tp)[0, 1]
                       if pp.std() > 1e-12 and tp.std() > 1e-12 else 0.0)
        out.append(f"  {per}: IC {np.mean(ics):+.3f} (n={int(m.sum())})")
    print('\n'.join(out))

# main
def main():
    Xs, M, EF, tgt, ts_tr, ts_va, ts_te, dates, names, model = load_everything()
    X_raw = np.load(resolve_data_dir() / 'X.npy').astype(np.float32)
    true = tgt[ts_te]
    print(f"test predictions: {len(ts_te)}  ({dates[ts_te[0]].date()} → {dates[ts_te[-1]].date()})")

    preds, attns = chunked_preds(model_forward_factory(model, Xs, M, EF), ts_te)
    rows = [('graph model', metrics(preds, true)),
            ('zero', metrics(baseline_zero(ts_te, tgt), true)),
            ('calibrated-lin', metrics(baseline_calibrated_lin(X_raw, tgt, ts_tr, ts_te), true))]

    print("training plain-GRU baseline (no graph)...")
    base = train_plain_gru(Xs, EF, tgt, ts_tr, ts_va)
    base_preds, _ = chunked_preds(lambda b: (base(raw_windows(Xs, EF, b)), None), ts_te)
    rows.insert(1, ('plain GRU', metrics(base_preds, true)))

    print_table(rows)
    lin_preds = baseline_calibrated_lin(X_raw, tgt, ts_tr, ts_te)
    significance_block(true, [('graph model', preds), ('plain GRU', base_preds),
                              ('calibrated-lin', lin_preds)])
    d_te = dates[ts_te]
    quarterly_ic(d_te, 'graph model', preds, true)
    quarterly_ic(d_te, 'plain GRU', base_preds, true)
    quarterly_ic(d_te, 'calibrated-lin', lin_preds, true)
    np.savez(ATTN_OUT, attn=attns, dates=dates[ts_te].astype(str), preds=preds, true=true)
    print(f"\nattention maps + predictions saved -> {ATTN_OUT} (for inspect_data.ipynb)")


if __name__ == '__main__':
    main()
