# FX Correlation-Regime Forecasting

An edge-centric graph transformer with a GRU temporal head, built to test whether
graph structure over a cross-asset market graph predicts shifts in FX pair
correlation beyond what simple linear mechanics already capture. The target is the
forward change in trailing 20-day correlation, `corr[t+H] − corr[t]`, for
EUR-GBP, EUR-JPY, and GBP-JPY.

## Result

At the 5-day horizon, the graph model:
- beats a no-graph plain-GRU control (mean IC 0.288 vs 0.270, per-pair p ≤ 0.002),
- beats a 2-state Gaussian HMM regime baseline (0.288 vs −0.082),
- but only ties — while still trailing — a calibrated linear mean-reversion
  baseline (0.288 vs 0.329; margin −0.046, 95% CI [−0.113, +0.012]).

At the 20-day horizon the graph loses to the linear baseline outright. So across
both horizons it never beats linear mechanics — it ties at best.

Conclusion: the predictable structure in these correlation shifts is largely
linear and mean-reverting; the graph architecture adds measurable value over a
plain GRU but not over a two-feature linear model.

## Why the evaluation is built this way

Early experiments scored deceptively well until a trivial baseline matched the
model — the target was mechanically predictable (mean-reverting), so high IC
reflected mechanics, not skill. The evaluation was redesigned around this: a
calibrated linear baseline defines the mechanical bar, a no-graph ablation
isolates the architecture's contribution, and permutation/bootstrap tests gate
every claim. The goal is to measure skill *beyond* mechanics, and to report a
negative result rather than a flattering one.

## Method

- **Spatial:** a per-day edge transformer over a 25-node graph (3 FX pairs +
  22 macro context nodes — rates, yields, equities, volatility, commodities),
  learning representations on the 6 directed target edges. Backpropagation
  updates edge representations directly rather than deriving edges from node
  embeddings — chosen because the objective is local relational forecasting in
  a noisy domain.
- **Temporal:** a GRU over 20-day windows of edge states predicting the forward
  correlation shift.
- **Evaluation:** time-purged train/val/test splits, train-only standardization,
  a calibrated linear baseline and no-graph ablation, with significance via
  block-permutation tests and bootstrap confidence intervals.

## Repo layout

- `src/models/` — graph transformer, edge GRU
- `src/training/train_graph.py` — end-to-end training
- `src/evaluation/` — evaluate, significance tests, HMM baseline, context-signal gate
- `runs/` — run logs (run005–run007)
- `docs/Engineering_log.md` — dated decision log

## Data

Daily 25-node macro-FX graph, 1999–2026 (not committed — large binaries,
gitignored). Place `X.npy`, `nan_mask.npy`, `dates.csv`, `node_names.csv` in
`data_live_2026/`. Sourced from FRED (set `FRED_API_KEY` in `.env`).

## Running

    pip install -r requirements.txt
    python -m src.training.train_graph                     # trains -> checkpoints/best_model.pt
    $env:FX_HORIZON=5; python -m src.evaluation.evaluate   # scores + significance

## Limitations & future work

- Single seed; seed-robustness not yet run.
- Both neural models dip sharply in 2024Q2 where the linear baseline holds —
  a regime-dependent weakness, not yet diagnosed.
- HMM baseline is a standard 2-state Gaussian; a Student-t or higher-state
  variant is the natural extension, not yet explored.