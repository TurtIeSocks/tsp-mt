# Goals — LKH → native Rust port

## Driving force

License clarity. LKH-3 README: *"distributed for research use. The author reserves all rights to the code."* tsp-mt is Apache 2.0; vendor crate cannot redistribute LKH binaries cleanly. Removing the dependency makes tsp-mt a single-license Apache 2.0 distribution.

## Goals (ranked)

1. **Dependency reduction** — remove the external LKH binary and its fetch/build/embed dance from `tsp_mt_core`.
2. **Single-license distribution** — every line of TSP-solving code in the published binary should be Apache 2.0.
3. **Maintainability** — Rust port should be readable enough that future fixes don't require LKH C archaeology.
4. **Tour quality** — match LKH within a reasonable margin (target: ≤5% length penalty on benchmark instances ≤10k nodes). Not "beat LKH"; "be good enough for the same use case".
5. **Performance** — same wall-clock order of magnitude as LKH for the working size range (a few × 1k–100k chunks). Going faster is bonus; matching is required.

## Non-goals

- ATSP, SOP, CVRP, MTSP, TSPTW, PDPTW, BWTSP, CTSP, GCTSP, MSCTSP, PCTSP, etc. — only symmetric TSP.
- Non-EUC_2D distance types (no GEO, MAN_2D, ATT, EXPLICIT). EUC_2D + integer rounding (TSPLIB `nint`) only.
- POPMUSIC, Delaunay candidate sets — ALPHA-only is sufficient.
- Subproblem partitioning (tsp_mt_core does its own H3 chunking; LKH's subproblem mode is not used).
- Genetic recombination, GPX2, CLARIST — RUNS is always 1.
- Hashing-based tour pool dedup — RUNS is always 1.
- Three-level segment tree — doubly-linked list with optional two-level tree is enough for tsp_mt_core's chunk sizes.
- Output-file compatibility — solver returns `Vec<usize>` directly, no TSPLIB tour round-trip.

## Constraints

- **Timeline**: open-ended. Single-author project, no merge-freeze.
- **Breaking changes**: allowed within the workspace. `lkh` and `lkh_derive` crates stay published as-is but become unused by `tsp_mt_core` (user plans to move them to a separate repo later).
- **Team**: solo. No coordination cost.
- **Migration**: big-bang. Single PR rewires `tsp_mt_core` from `lkh` to the new crate; `fetch-lkh` feature becomes a no-op or is removed.
- **Backwards-compatibility**: tsp-mt 0.1.x is pre-1.0. `lkh_single` / `lkh_multi_seed` / `lkh_multi_parallel` public fn names can be renamed if desired (suggest `solve_single` / `solve_multi_seed` / `solve_multi_parallel`).

## Open trade-off — already decided

| Question | Decision |
|----------|----------|
| Faithful 1:1 port from C, or clean reimplementation following LK algorithm? | **Clean reimplementation** — port the algorithm, not the C code. Idiomatic Rust, owned state in a `Solver` struct, no globals. |
| ALPHA candidates vs NN candidates? | **NN first (k-d tree), ALPHA later if quality demands it.** ALPHA requires MST + subgradient ascent — adds ~2k LOC. NN gets ~80% of the way for ≤10k nodes per LKH paper. |
| k-opt order? | **Up to k=3 first (2-opt + 3-opt), then k=5 if needed.** LKH default is k=5 but the marginal gain at k=4,5 diminishes; k=3 is well-understood and easier to debug. |
| Tour representation? | **Doubly-linked list (Pred/Suc fields).** Two-level segment tree (`Flip_SL.c`, `Between_SL.c`) is faster for n ≥ ~5000 but ~3× more code. Add if profiling shows bottleneck. |
| Crate name? | **`lin_kernighan`**. Clear, doesn't collide with existing `lkh` crate. |

## Verification plan

1. Cargo workspace builds, all crates compile.
2. Cargo test passes for the new crate (unit tests for distance, candidate set, tour ops, full solve on small instances).
3. `tsp_mt_core` integration tests pass with new solver.
4. Sanity check tour quality on a known benchmark: run on pr2392 or pla7397, compare tour length vs LKH's reported optimum.
5. Wall-clock comparison: run `benchmark/` script with old (LKH) vs new (lin_kernighan) on the same input, document delta.

## Anti-scope (avoid even if tempting)

- Don't rewrite the `lkh`/`lkh_derive` crates. They stay as-is.
- Don't touch `stitching.rs`, `metric_spike.rs`, `h3_chunking.rs` — they don't call LKH directly.
- Don't add new CLI flags. Mirror existing `SolverOptions` exactly.
- Don't change the lat/lng → projected coord pipeline. Keep `PlaneProjection`, keep ×1000 scaling.
- Don't generalize to ATSP "in case someone asks later". Build for the actual use.
