"""
significance_test.py
1. Permutation test (block-aware): shuffle the prediction series against the
   outcome series many times, in contiguous blocks of CORR_W days (plain
   shuffling would destroy the overlap structure and overstate significance).
   The p-value = fraction of shuffles whose |IC| beats the real |IC|.
   p < 0.05 -> the IC is unlikely to be luck. p ~ 0.3+ -> indistinguishable
   from noise.

2. Sign-balance check: share of positive target signs per pair. ~0.5 means
   DirAcc is meaningful; ~0.65 means a sign-imbalance freebie inflates it.
"""
import numpy as np
from pathlib import Path

from src.training.train_graph import CORR_W, HORIZON

NPZ    = Path('checkpoints/attn_test.npz')
BLOCK  = CORR_W + HORIZON # full dependency span: estimation overlap + forecast horizon
N_PERM = 2000
SEED   = 0
PAIRS  = ['EUR-GBP', 'EUR-JPY', 'GBP-JPY']


def ic(pred, true):
    if pred.std() < 1e-12 or true.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, true)[0, 1])


def block_permute(n, block, rng):
    """Permutation of 0..n-1 that shuffles contiguous blocks, preserving
    local (overlap-induced) autocorrelation inside blocks."""
    starts = np.arange(0, n, block)
    rng.shuffle(starts)
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
    return idx


def perm_test(pred, true, rng):
    real = ic(pred, true)
    count = 0
    for _ in range(N_PERM):
        sh = pred[block_permute(len(pred), BLOCK, rng)]
        if abs(ic(sh, true)) >= abs(real):
            count += 1
    return real, (count + 1) / (N_PERM + 1) # add-one: unbiased p


def main():
    d = np.load(NPZ, allow_pickle=True)
    preds, true = d['preds'], d['true']
    rng = np.random.default_rng(SEED)

    print(f"n test days: {len(true)}   (~{len(true)//BLOCK} independent obs)")
    print(f"permutations: {N_PERM}, block = {BLOCK} days\n")

    print("permutation test — graph model")
    pair_ps = []
    for p, name in enumerate(PAIRS):
        real, pv = perm_test(preds[:, p], true[:, p], rng)
        pair_ps.append(pv)
        print(f"  {name}: IC {real:+.4f}   p = {pv:.3f}")
    mean_real = np.mean([ic(preds[:, p], true[:, p]) for p in range(3)])
    print(f"  MEAN IC {mean_real:+.4f}   (per-pair p-values above)")
    print("  read: p < 0.05 = likely real; p > 0.30 = noise\n")

    print("sign-balance check — target")
    for p, name in enumerate(PAIRS):
        pos = float((true[:, p] > 0).mean())
        flag = "  <- imbalanced: DirAcc inflated" if abs(pos - 0.5) > 0.08 else ""
        print(f"  {name}: positive share = {pos:.3f}{flag}")
    print("  read: ~0.50 = DirAcc honest; >0.58 or <0.42 = majority-guess freebie")


if __name__ == '__main__':
    main()
