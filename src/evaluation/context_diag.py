"""
context_diag.py — the gate before the frame sweep.

The graph's only edge over the plain GRU is the 22 non-target currencies. This
asks whether they carry distinct, lead-lagging signal for the forward
correlation shift, before any graph-training wall-clock is spent.

Estimation window CORR_W is held fixed (label noise constant) at 20; only the forecast
horizon varies. Ridge on TRAIN, IC on VAL. PULSE at any horizon -> sweep is
justified. FLAT at all three -> defensible negative, stop before training.

"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.training.train_graph import trailing_corr, resolve_data_dir, CORR_W

PULSE_MIN = 0.02
N_SHUFFLE = 200 # context-permutation null draws
ALPHAS    = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
HORIZONS  = [5, 10, 20]
PAIRS     = ['EUR-GBP', 'EUR-JPY', 'GBP-JPY']
CONTEXT   = list(range(3, 25))  # the 22 non-target nodes


def node_feats(X, t):
    """[trailing CORR_W return, vol] for every node at day t (horizon-free)."""
    ret = X[t - CORR_W + 1:t + 1, :, 0].sum(axis=0)
    vol = X[t, :, 2]
    return np.concatenate([ret, vol])


def feature_rows(X, ts):
    return np.nan_to_num(np.stack([node_feats(X, int(t)) for t in ts]))


def ic(pred, true):
    if pred.std() < 1e-12 or true.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, true)[0, 1])


def ridge(A, y, alpha):
    G = A.T @ A + alpha * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ y)


def best_val_ic(Atr, ytr, Ava, yva):
    """Standardize on train, fit per alpha, keep best val IC."""
    mu_a, sd_a = Atr.mean(0), Atr.std(0) + 1e-8
    mu_y = ytr.mean()
    Atr, Ava = (Atr - mu_a) / sd_a, (Ava - mu_a) / sd_a
    out = -1.0
    for al in ALPHAS:
        w = ridge(Atr, ytr - mu_y, al)
        out = max(out, ic(Ava @ w, yva - mu_y))
    return out


def sterilized_target(trail, ls, H, train_hi):
    """Forward H-day shift with the train-only [1, trail, d_rho] fit removed."""
    T = trail.shape[0]
    fwd = np.full_like(trail, np.nan)
    fwd[:T - H] = trail[H:]
    raw = fwd - trail
    tgt = raw.copy()
    rows = np.arange(T)
    for p in range(trail.shape[1]):
        m = (rows <= train_hi) & np.isfinite(raw[:, p]) & np.isfinite(trail[:, p]) & np.isfinite(ls[:, p])
        A = np.column_stack([np.ones(m.sum()), trail[m, p], ls[m, p]])
        coef, *_ = np.linalg.lstsq(A, raw[m, p], rcond=None)
        fit = coef[0] + coef[1] * trail[:, p] + coef[2] * ls[:, p]
        tgt[:, p] = raw[:, p] - fit
    return tgt


def valid_idx(lo, hi, H, tgt):
    """In-split prediction dates with valid features and a leak-free t+H target."""
    lo = max(lo, 2 * CORR_W - 1)
    ts = np.arange(lo, hi - H + 1)
    return ts[~np.isnan(tgt[ts]).any(axis=1)]


def block_perm(n, block, rng):
    """Row permutation that shuffles contiguous blocks (preserves overlap)."""
    starts = np.arange(0, n, block)
    rng.shuffle(starts)
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])


def mean_uplift(base, Ftr, Fva, ctx, tgt, ts_tr, ts_va, perm_tr=None, perm_va=None):
    """Mean over pairs of (base+context IC - base IC). perm_* permute the
    context rows only, leaving base and target aligned -> the null."""
    Ctr = Ftr[:, ctx][perm_tr] if perm_tr is not None else Ftr[:, ctx]
    Cva = Fva[:, ctx][perm_va] if perm_va is not None else Fva[:, ctx]
    b_ics, f_ics = [], []
    for p in range(len(PAIRS)):
        ytr, yva = tgt[ts_tr, p], tgt[ts_va, p]
        bt, bv = np.nan_to_num(base[ts_tr, p]), np.nan_to_num(base[ts_va, p])
        b_ics.append(best_val_ic(bt, ytr, bv, yva))
        f_ics.append(best_val_ic(np.column_stack([bt, Ctr]), ytr,
                                 np.column_stack([bv, Cva]), yva))
    return float(np.mean(f_ics) - np.mean(b_ics)), float(np.mean(b_ics)), b_ics, f_ics


def run_horizon(X, trail, ls, H, train_hi, val_hi, rng):
    tgt = sterilized_target(trail, ls, H, train_hi)
    ts_tr = valid_idx(0, train_hi, H, tgt)
    ts_va = valid_idx(train_hi + 1, val_hi, H, tgt)

    base = np.stack([trail, ls], axis=-1)                       # [T, 3, 2] per-pair feats
    Ftr, Fva = feature_rows(X, ts_tr), feature_rows(X, ts_va)
    ctx = np.concatenate([CONTEXT, [n + 25 for n in CONTEXT]])
    blk = CORR_W + H

    uplift, base_mean, b_ics, f_ics = mean_uplift(base, Ftr, Fva, ctx, tgt, ts_tr, ts_va)

    # null: same fit with context rows block-shuffled out of time alignment
    null = np.array([
        mean_uplift(base, Ftr, Fva, ctx, tgt, ts_tr, ts_va,
                    block_perm(len(ts_tr), blk, rng), block_perm(len(ts_va), blk, rng))[0]
        for _ in range(N_SHUFFLE)])
    p = float((null >= uplift).sum() + 1) / (N_SHUFFLE + 1)

    blocks = len(ts_va) // blk
    print(f"\nHORIZON {H}d   val obs {len(ts_va)}   (~{blocks} independent val blocks)")
    print(f"{'pair':<10}{'base IC':>10}{'+ctx IC':>10}{'uplift':>10}")
    print('-' * 40)
    for name, b, f in zip(PAIRS, b_ics, f_ics):
        print(f"{name:<10}{b:>10.4f}{f:>10.4f}{f - b:>+10.4f}")
    print('-' * 40)
    print(f"{'MEAN':<10}{base_mean:>10.4f}{np.mean(f_ics):>10.4f}{uplift:>+10.4f}")
    print(f"shuffle null: mean {null.mean():+.4f}  95th {np.percentile(null, 95):+.4f}  "
          f"-> real uplift p = {p:.3f}")
    return H, uplift, base_mean, p


def main():
    data_dir = resolve_data_dir()
    X = np.load(data_dir / 'X.npy').astype(np.float32)
    dates = pd.to_datetime(pd.read_csv(data_dir / 'dates.csv')['0'].values)
    train_hi = int(np.where(dates <= pd.Timestamp('2021-12-31'))[0].max())
    val_hi   = int(np.where(dates <= pd.Timestamp('2023-12-31'))[0].max())

    trail = trailing_corr(X)
    ls = np.full_like(trail, np.nan)
    ls[CORR_W:] = trail[CORR_W:] - trail[:-CORR_W]

    rng = np.random.default_rng(0)
    print(f"sterilized target | estimation CORR_W={CORR_W} fixed | base IC ~0 | "
          f"{N_SHUFFLE} shuffle nulls")
    results = [run_horizon(X, trail, ls, H, train_hi, val_hi, rng) for H in HORIZONS]

    print("\n" + "=" * 52)
    significant = [r for r in results if r[1] >= PULSE_MIN and r[3] < 0.05]
    best = max(significant, key=lambda r: r[1]) if significant else max(results, key=lambda r: r[1])
    for H, up, base_ic, p in results:
        real = up >= PULSE_MIN and p < 0.05
        tag = "PULSE (survives shuffle)" if real else ("flat" if up < PULSE_MIN
              else "NOISE (uplift in null)")
        print(f"  H={H:>2}d   uplift {up:+.4f}  p {p:.3f}  (base {base_ic:+.4f})   {tag}")
    bH, bup, _, bp = best
    if bup >= PULSE_MIN and bp < 0.05:
        verdict = f"PULSE at H={bH}d, p={bp:.3f} — sweep justified"
    elif bup >= PULSE_MIN:
        verdict = f"uplift at H={bH}d but p={bp:.3f} — noise, NOT justified"
    else:
        verdict = "FLAT everywhere — graph unlikely to help, stop here"
    print(f"\nbest uplift {bup:+.4f} @ H={bH}d  ->  {verdict}")


if __name__ == '__main__':
    main()
