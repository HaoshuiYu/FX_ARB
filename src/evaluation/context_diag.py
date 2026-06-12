"""
context_diagnostics.py — the gate before the frame sweep.

The graph's only edge over the plain GRU is the 22 non-target currencies. This
asks whether they carry distinct, lead-lagging signal for the forward
correlation shift at all, before any wall-clock is spent training graphs.

Method: ridge on TRAIN, IC on VAL, two feature sets per pair —
  base    = the pair's own [rho_trail, d_rho]            (no context)
  +context= base plus 22 context nodes' [trailing ret, vol] at t
If +context lifts val IC by at least PULSE_MIN, there's a pulse and the sweep
is worth running. If not, the graph is unlikely to help and that's a clean,
defensible negative on its own.

Run from repo root:  python -m src.evaluation.context_diagnostics
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.training.train_graph import (build_targets, build_edge_feats, valid_ts,
                                       resolve_data_dir, CORR_W, TARGET_PAIRS)

PULSE_MIN = 0.02                      # min mean val-IC uplift to call it a pulse
ALPHAS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
PAIRS     = ['EUR-GBP', 'EUR-JPY', 'GBP-JPY']
CONTEXT   = list(range(3, 25))        # the 22 non-target nodes


def node_feats(X, t):
    """[trailing CORR_W return, vol] for every node at day t."""
    ret = X[t - CORR_W + 1:t + 1, :, 0].sum(axis=0)
    vol = X[t, :, 2]
    return np.concatenate([ret, vol])


def feature_rows(X, ts):
    """[len(ts), 50] context-node features (ret+vol for all 25 nodes)."""
    return np.nan_to_num(np.stack([node_feats(X, int(t)) for t in ts]))


def ic(pred, true):
    if pred.std() < 1e-12 or true.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, true)[0, 1])


def ridge(A, y, alpha):
    """Centered ridge; returns weights (no intercept, data is demeaned)."""
    G = A.T @ A + alpha * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ y)


def best_val_ic(Atr, ytr, Ava, yva):
    """Fit per alpha on train, keep the best val IC."""
    mu_a, sd_a = Atr.mean(0), Atr.std(0) + 1e-8
    mu_y = ytr.mean()
    Atr, Ava = (Atr - mu_a) / sd_a, (Ava - mu_a) / sd_a
    out = -1.0
    for al in ALPHAS:
        w = ridge(Atr, ytr - mu_y, al)
        out = max(out, ic(Ava @ w, yva - mu_y))
    return out


def main():
    data_dir = resolve_data_dir()
    X = np.load(data_dir / 'X.npy').astype(np.float32)
    dates = pd.to_datetime(pd.read_csv(data_dir / 'dates.csv')['0'].values)
    train_hi = int(np.where(dates <= pd.Timestamp('2021-12-31'))[0].max())
    val_hi   = int(np.where(dates <= pd.Timestamp('2023-12-31'))[0].max())

    tgt = build_targets(X)
    EF  = build_edge_feats(X)[:, [0, 2, 4], :]          # one [rho_trail, d_rho] per pair
    ts_tr = valid_ts(0, train_hi, tgt)
    ts_va = valid_ts(train_hi + 1, val_hi, tgt)

    Ftr, Fva = feature_rows(X, ts_tr), feature_rows(X, ts_va)
    ctx = np.concatenate([CONTEXT, [n + 25 for n in CONTEXT]])   # ret + vol cols
    base_tr, base_va = EF[ts_tr].reshape(len(ts_tr), -1), EF[ts_va].reshape(len(ts_va), -1)

    print(f"train obs {len(ts_tr)}  val obs {len(ts_va)}  "
          f"(~{len(ts_va) // (CORR_W)} independent val blocks)\n")
    print(f"{'pair':<10}{'base IC':>10}{'+ctx IC':>10}{'uplift':>10}")
    print('-' * 40)

    base_ics, full_ics = [], []
    for p, name in enumerate(PAIRS):
        ytr, yva = tgt[ts_tr, p], tgt[ts_va, p]
        b = best_val_ic(base_tr[:, 2 * p:2 * p + 2], ytr,
                        base_va[:, 2 * p:2 * p + 2], yva)
        full_tr = np.column_stack([base_tr[:, 2 * p:2 * p + 2], Ftr[:, ctx]])
        full_va = np.column_stack([base_va[:, 2 * p:2 * p + 2], Fva[:, ctx]])
        f = best_val_ic(full_tr, ytr, full_va, yva)
        base_ics.append(b); full_ics.append(f)
        print(f"{name:<10}{b:>10.4f}{f:>10.4f}{f - b:>+10.4f}")

    uplift = float(np.mean(full_ics) - np.mean(base_ics))
    print('-' * 40)
    print(f"{'MEAN':<10}{np.mean(base_ics):>10.4f}{np.mean(full_ics):>10.4f}{uplift:>+10.4f}")
    verdict = "PULSE — run the sweep" if uplift >= PULSE_MIN else "FLAT — graph unlikely to help"
    print(f"\nmean val-IC uplift {uplift:+.4f}  (threshold {PULSE_MIN:+.4f})  ->  {verdict}")


if __name__ == '__main__':
    main()
