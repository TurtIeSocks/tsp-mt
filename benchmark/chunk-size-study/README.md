# Chunk-size study

Empirical basis for the `chunk_optimal(n)` default chooser in
`crates/tsp_mt_core/src/io/options.rs`. The **results** here are committed
(unlike the rest of `benchmark/`, which is scratch) so the chooser's tuning is
reproducible and reviewable. The **inputs are not committed** — they are large
(~84 MB) and regenerate deterministically, so they are git-ignored.

## Layout

```
chunk-size-study/
├── inputs/    64 deterministic point files: n{N}_d{D}.txt  (GIT-IGNORED, regenerated)
│              (uniform lat/lng around 37.5,-122.0; seed = N*100 + D)
└── results/   sweep outputs (TSV + Markdown)  (COMMITTED)
```

## Inputs (regenerated, not committed)

Created on-demand by the scripts below and identical across runs (fixed seeds),
so they are git-ignored rather than stored. Running either script recreates any
missing files. Distributions: `d1,d2,d3` for `n <= 25k`; `d1` only for
`n > 25k`. Radius widens with `n` (0.3 / 0.5 / 0.8) to keep density realistic.

## Results

| File prefix | Produced by | Contents |
|---|---|---|
| `chunk-explorer-<ts>-rows.tsv` | `scripts/sweep-chunk-explorer.sh` | raw per-run rows (n, chunk, dist, rep, lk, runtime) |
| `chunk-explorer-<ts>-agg.tsv` | same | median per (n, chunk) + `gap_vs_best_within_n` |
| `chunk-explorer-<ts>.md` | same | human-readable aggregate table |
| `chunk-confirm-<ts>-compare.tsv` | `scripts/confirm-chunk-optimal.sh` | `chunk_optimal(n)`'s pick vs the explorer's best-for-n |
| `chunk-confirm-<ts>.md` | same | human-readable confirmation table |

The committed run is timestamp `20260528T203703Z` (explorer, 424 cells across
38 values of `n` from 1k–250k) and `20260529T023845Z` (confirmation).

## Reproduce

```bash
# Full chunk-size sweep (long; ~3h for the committed run). NO_RUN=1 prints the plan.
scripts/sweep-chunk-explorer.sh

# Fast confirmation that the deployed chunk_optimal(n) lands near the sweep optimum.
scripts/confirm-chunk-optimal.sh [path-to-explorer-agg.tsv]
```

## Headline finding

A sharp regime change near `n ≈ 50k`: below it, solving the whole instance as
one chunk gives the best tour but costs up to ~130s; above it, small chunks
(~2.5k–7k) are Pareto-optimal — both better quality *and* faster than large
chunks. `chunk_optimal(n)` encodes this as a piecewise fit.
