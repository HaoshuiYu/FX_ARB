"""
hmm_baseline.py — industry standard baseline: scored through the same
target, splits, metric, and significance test as the graph model.

Design: a Gaussian HMM is fit on the 3 target pairs' daily returns (train only).
Each state is mapped to a forecast via the train-set mean forward shift in that
state; the test prediction is the posterior-weighted blend of those state means
(predict_proba).

NOTE: this is GaussianHMM which is the standard baseline but not necessarily the optimal variant for FX.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

from src.training.train_graph import (build_targets, valid_ts, resolve_data_dir,
                                       CORR_W, HORIZON)
from src.evaluation.significance_test import ic, perm_test

NPZ      = Path('checkpoints/attn_test.npz')
N_STATES = 2  # calm / stressed; low to avoid overfit
SEED     = 0
PAIRS    = ['EUR-GBP', 'EUR-JPY', 'GBP-JPY']


def main():
    data_dir = resolve_data_dir()
    X = np.load(data_dir / 'X.npy').astype(np.float32)
    dates = pd.to_datetime(pd.read_csv(data_dir / 'dates.csv')['0'].values)
    train_hi = int(np.where(dates <= pd.Timestamp('2021-12-31'))[0].max())
    val_hi   = int(np.where(dates <= pd.Timestamp('2023-12-31'))[0].max())

    tgt = build_targets(X, train_hi)                    
    ts_tr = valid_ts(0, train_hi, tgt)                  
    ts_te = valid_ts(val_hi + 1, len(dates) - 1, tgt)
    true = tgt[ts_te]

    obs = X[:, :3, 0] # 3 pairs' daily returns, recall the edges as articulated here is undirected weighted

    hmm = GaussianHMM(n_components=N_STATES, covariance_type='full',
                      n_iter=200, random_state=SEED)
    hmm.fit(obs[ts_tr])                     

    # per-state mean forward shift, learned on train
    s_tr = hmm.predict(obs[ts_tr])
    state_mean = np.stack([np.nanmean(tgt[ts_tr[s_tr == s]], axis=0)
                           for s in range(N_STATES)])   # [n_states, 3]

    # test forecast = posterior-weighted blend of state means (continuous)
    post = hmm.predict_proba(obs[ts_te])                # [n_test, n_states]
    pred = post @ state_mean                            # [n_test, 3]

    print(f"HMM regime baseline | {N_STATES} states | "
          f"test {dates[ts_te[0]].date()} -> {dates[ts_te[-1]].date()}  (n={len(ts_te)})")
    rng = np.random.default_rng(SEED)
    ics, ps = [], []
    for p, name in enumerate(PAIRS):
        r, pv = perm_test(pred[:, p], true[:, p], rng)
        ics.append(r); ps.append(pv)
        print(f"  {name}: IC {r:+.4f}   p = {pv:.3f}")
    print(f"  MEAN IC {np.mean(ics):+.4f}   per-pair p = "
          + ", ".join(f"{v:.3f}" for v in ps))

    if NPZ.exists():
        g = np.load(NPZ, allow_pickle=True)['preds']
        g_ic = np.mean([ic(g[:, p], true[:, p]) for p in range(3)])
        print(f"\n  graph model (from {NPZ.name}): mean IC {g_ic:+.4f}")
        print(f"  HMM minus graph: {np.mean(ics) - g_ic:+.4f}")
    else:
        print(f"\n  ({NPZ} not found — run evaluate.py first for the graph comparison)")


if __name__ == '__main__':
    main()
