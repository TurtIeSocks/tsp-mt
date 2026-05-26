# Assessment — what to keep, refactor, rewrite, drop

Verdicts based on Phase 1 trace + Phase 2 goals.

## tsp_mt_core (modify in place)

| Module | Verdict | Evidence | Tension with goals | Confidence |
|--------|---------|----------|--------------------|------------|
| `src/algo/runner/common.rs` | Refactor in place | [HOT][BRIDGE] — all `lkh::` imports concentrated here; pure adapter logic the new crate will replace | Direct blocker for "dependency reduction" goal | High |
| `src/algo/runner/mod.rs` | Refactor in place | [HOT][BRIDGE] — `LkhSolver` call sites + return type plumbing | Same | High |
| `src/algo/runner/metric_spike.rs` | Keep | No `lkh::` usage; post-processing only | None | High |
| `src/algo/stitching.rs` | Keep | Independent algorithm | None | High |
| `src/algo/h3_chunking.rs` | Keep | Geo logic, no LKH coupling | None | High |
| `src/geo/*` | Keep | Plane projection + centroid math | None | High |
| `src/io/options.rs:8` | Refactor in place | One-line `use lkh::embedded_lkh;` — drop import; `lkh_exe: PathBuf` field becomes obsolete | Direct blocker | High |
| `src/io/input.rs` | Keep | Input parser, no LKH coupling | None | High |
| `src/lib.rs`, `src/node.rs`, `src/tour.rs`, `src/error.rs`, `src/logging.rs`, `src/file_cleanup.rs` | Keep | Public types, errors, logging — independent of LKH | None | High |

## lkh / lkh_derive crates (untouched per user direction)

| Crate | Verdict | Note |
|-------|---------|------|
| `crates/lkh` | Defer | User: *"leave the lkh and lkh_derive crates for now. I might move them into a separate repo at a later date."* No edits. |
| `crates/lkh_derive` | Defer | Same. |

After refactor: `tsp_mt_core/Cargo.toml` drops the `lkh = { path = "../lkh" }` dep, and the workspace can drop them from `members` once moved out. Until then they remain compiled but unused.

## New crate — `lin_kernighan` (write from scratch)

| Module | Verdict | What it does |
|--------|---------|--------------|
| `src/lib.rs` | Rewrite (new) | Crate root; re-exports public API |
| `src/error.rs` | Rewrite (new) | `Error`, `Result` types; replaces `LkhError`/`LkhResult` |
| `src/coord.rs` | Rewrite (new) | `Point2D { x: f64, y: f64 }`; integer scaling helper |
| `src/distance.rs` | Rewrite (new) | EUC_2D distance with TSPLIB `nint` rounding |
| `src/candidate.rs` | Rewrite (new) | `Candidate { to: NodeId, cost: i64, alpha: i64 }`; NN candidate generation via k-d tree |
| `src/tour.rs` | Rewrite (new) | Doubly-linked list tour (Pred/Suc); Flip, Between operations; tour-as-`Vec<usize>` conversions |
| `src/initial.rs` | Rewrite (new) | Greedy initial tour (per LKH GreedyTour.c); optional input-tour load |
| `src/lk/mod.rs` | Rewrite (new) | Main LK loop: active queue, k-opt search dispatch |
| `src/lk/two_opt.rs` | Rewrite (new) | Best2OptMove + Make2OptMove |
| `src/lk/three_opt.rs` | Rewrite (new) | Best3OptMove + Make3OptMove |
| `src/lk/k_opt.rs` | Rewrite (new) | BestKOptMove + MakeKOptMove (k=4,5) — deferrable |
| `src/lk/gain.rs` | Rewrite (new) | Sequential gain accounting + improvement decision |
| `src/perturb.rs` | Rewrite (new) | KSwapKick perturbation (deferrable; KICK_TYPE=0 default works without it) |
| `src/alpha.rs` | Rewrite (new) — deferred | 1-tree + subgradient ascent + alpha-value candidate sort. Add only if NN quality is insufficient. |
| `src/solver.rs` | Rewrite (new) | `Solver { problem, params, ... }`; main `solve()` entry point; trial loop; time limit |
| `src/params.rs` | Rewrite (new) | `Params { max_candidates, max_trials, time_limit, seed, trace_level, initial_tour }` — narrow vs LKH's 87-field struct |

LOC estimate for new crate: **~3.5k–4.5k Rust LOC** (NN-only candidates, doubly-linked list tour, k≤3 with hooks for k=5). Comparable to the symmetric-TSP-only subset of LKH (~12k C LOC) at typical 3–4× C→Rust compression.

## Dropped from the new codebase

Everything in LKH-3 source NOT in the "Needed" list of trace.md. Notably:
- All non-TSP problem types and their penalty fns
- All non-EUC_2D distance variants
- POPMUSIC, Delaunay, Quadrant candidate sets
- Subproblem partitioning
- Tour pool dedup (Hashing.c)
- Tour merging (Genetic.c, MergeWithTour*.c, gpx.c)
- BIT.c segment tree (defer until profiling shows need)
- Geographic conversion (only EUC_2D supported)
- Status/Statistics reporting (use log crate)

## Net summary

- **Keep**: 9 modules in `tsp_mt_core` (everything not bridging to LKH)
- **Refactor in place**: 3 files (`runner/common.rs`, `runner/mod.rs`, `io/options.rs`)
- **Defer**: 2 crates (`lkh`, `lkh_derive`)
- **Rewrite (new crate)**: ~14 modules in `lin_kernighan`
- **Drop**: ~14k LOC of LKH C source we never need
