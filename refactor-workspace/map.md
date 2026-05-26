# Refactor Map — LKH → `lin_kernighan`

## New structure

```
crates/
  lin_kernighan/                         (new crate, Apache 2.0)
    Cargo.toml
    src/
      lib.rs                             ← new module; re-exports public API
      error.rs                           ← new; Error/Result types
      coord.rs                           ← new; Point2D, integer scaling
      distance.rs                        ← was LKH Distance.c (EUC_2D branch only) + C.c
                                            Action: port — nint(sqrt(dx²+dy²)) for EUC_2D
      candidate.rs                       ← was LKH CreateNearestNeighborCandidateSet.c +
                                            AddCandidate.c + OrderCandidateSet.c +
                                            TrimCandidateSet.c + SymmetrizeCandidateSet.c
                                            Action: port + simplify
                                            Notes: NN-only first pass (uses kiddo k-d tree;
                                            kiddo already in tsp_mt_core deps so reuse).
                                            ALPHA reserved for alpha.rs if needed later.
      tour.rs                            ← was LKH Flip.c + Between.c (doubly-linked list rep)
                                            + ChooseInitialTour.c WALK case
                                            Action: port + redesign
                                            Notes: own Node struct with Pred/Suc indices
                                            (NodeId = u32); avoid raw pointers entirely.
      initial.rs                         ← was LKH GreedyTour.c
                                            Action: port
                                            Notes: greedy from candidate set; fallback
                                            for nodes with no usable candidate
      lk/
        mod.rs                           ← was LKH LinKernighan.c + Activate.c +
                                            RemoveFirstActive.c
                                            Action: port + redesign
                                            Notes: convert global state (FirstActive,
                                            LastActive linked list) to VecDeque<NodeId>
                                            on the solver struct.
        gain.rs                          ← was LKH Gain23.c + Improvement.c (partial) +
                                            BridgeGain.c
                                            Action: port
                                            Notes: sequential gain criterion check —
                                            integer arithmetic, no float concerns.
        two_opt.rs                       ← was LKH Best2OptMove.c + Make2OptMove.c
                                            Action: port + simplify (drop ATSP branches)
        three_opt.rs                     ← was LKH Best3OptMove.c + Make3OptMove.c
                                            Action: port + simplify
        k_opt.rs                         ← was LKH BestKOptMove.c + MakeKOptMove.c +
                                            Best4OptMove.c + Best5OptMove.c +
                                            Make4OptMove.c + Make5OptMove.c
                                            Action: port — phase 2 (after 2/3-opt land)
                                            Notes: only if quality demands; LKH defaults
                                            to k=5. Hidden behind a feature gate or just
                                            ship k≤3 first.
      perturb.rs                         ← was LKH KSwapKick.c
                                            Action: port (deferrable)
                                            Notes: KICK_TYPE=0 in tsp_mt_core practice;
                                            skip for v1, add for v2 if multi-trial without
                                            kicks converges too fast.
      alpha.rs                           ← was LKH Ascent.c + Minimum1TreeCost.c +
                                            MinimumSpanningTree.c + GenerateCandidates.c
                                            (alpha-value path)
                                            Action: DEFERRED — only port if NN candidates
                                            give insufficient tour quality.
                                            Notes: 1-tree lower bound + subgradient
                                            ascent. ~1.5–2k LOC of Rust if added.
      solver.rs                          ← was LKH FindTour.c + LKHmain.c (main control)
                                            + AllocateStructures.c + FreeStructures.c
                                            Action: port + redesign
                                            Notes: owns all state; trial loop + time
                                            limit check; main `solve()` entry point.
      params.rs                          ← new module, no old equivalent
                                            Action: write fresh
                                            Notes: narrow Params struct mirroring the
                                            7 actual knobs tsp_mt_core uses, not LKH's 87.

crates/
  tsp_mt_core/
    Cargo.toml                           ← Action: drop `lkh = { path = ... }` dep,
                                              add `lin_kernighan = { path = ... }`
    src/algo/runner/
      common.rs                          ← Action: port + redesign
                                              Notes: replace LkhSolver/TsplibProblem/
                                              LkhParameters with lin_kernighan::Solver;
                                              build_problem returns lin_kernighan::Problem;
                                              seeded_params returns lin_kernighan::Params;
                                              maybe_attach_initial_tour_file becomes
                                              params.initial_tour = Some(Vec<usize>)
                                              (in-memory, no .tour file).
      mod.rs                             ← Action: port + redesign
                                              Notes: all `LkhSolver::new(...).run_with_exe(...)`
                                              become `Solver::new(...).solve()` with no
                                              filesystem round-trip. LkhError → crate::Error.
    src/io/options.rs                    ← Action: edit
                                              Notes: remove `use lkh::embedded_lkh;`;
                                              remove `lkh_exe: PathBuf` field; remove
                                              `default_lkh_exe()` and the `fetch-lkh`
                                              feature gating in this file.
```

## Bulk 1:1 ports (LKH .c → Rust mod)

| Old (LKH C) | New (lin_kernighan) | Notes |
|-------------|---------------------|-------|
| `LKH/SRC/Distance.c` (EUC_2D branch) | `src/distance.rs` | Direct port; one fn |
| `LKH/SRC/C.c` | `src/distance.rs` (caching helper if needed) | Probably skip caching at first; n≤100k fits fine |
| `LKH/SRC/GreedyTour.c` | `src/initial.rs::greedy` | Port |
| `LKH/SRC/Flip.c` | `src/tour.rs::flip` | Port to Vec-indexed |
| `LKH/SRC/Between.c` | `src/tour.rs::between` | Port to Vec-indexed |
| `LKH/SRC/ChooseInitialTour.c` (WALK + GREEDY cases only) | `src/initial.rs` | Port subset |
| `LKH/SRC/Activate.c` + `RemoveFirstActive.c` | `src/lk/mod.rs` (VecDeque-based) | Port + redesign |
| `LKH/SRC/Best2OptMove.c` + `Make2OptMove.c` | `src/lk/two_opt.rs` | Port |
| `LKH/SRC/Best3OptMove.c` + `Make3OptMove.c` | `src/lk/three_opt.rs` | Port |
| `LKH/SRC/Improvement.c` (apply-move part) | `src/lk/gain.rs` | Port |
| `LKH/SRC/Gain23.c` | `src/lk/gain.rs` | Port |
| `LKH/SRC/CreateNearestNeighborCandidateSet.c` | `src/candidate.rs::build_nn` | Port; use kiddo k-d tree |
| `LKH/SRC/AddCandidate.c` | inline into `candidate.rs` | Trivial |
| `LKH/SRC/SymmetrizeCandidateSet.c` | `candidate.rs::symmetrize` | Port |
| `LKH/SRC/OrderCandidateSet.c` | `candidate.rs::sort_by_cost` | Port (sort by cost ascending) |
| `LKH/SRC/TrimCandidateSet.c` | `candidate.rs::trim` | Port (truncate to MAX_CANDIDATES per node) |
| `LKH/SRC/FindTour.c` | `src/solver.rs::find_tour` | Port — trial loop |
| `LKH/SRC/LKHmain.c` (main + state init) | `src/solver.rs::Solver::{new, solve}` | Port — orchestration |

## Dropped (not in new codebase)

| File / Concern | Why dropped |
|----------------|-------------|
| All `Penalty_*.c` | Symmetric TSP has no penalty function |
| `Distance_SOP.c`, `Distance_MTSP.c`, `Distance_SPECIAL.c` | EUC_2D only |
| `Genetic.c`, `MergeWithTour*.c`, `gpx.c` | RUNS=1 |
| All `*_InitialTour.c` except `GreedyTour.c` | Only Greedy + Walk needed |
| `Solve*Subproblems.c` | tsp_mt_core does H3 chunking; no LKH subproblem mode |
| `Create_POPMUSIC_CandidateSet.c` | POPMUSIC not used |
| `CreateDelaunayCandidateSet.c`, `Delaunay.c` | Delaunay not used |
| `CreateQuadrantCandidateSet.c` | Quadrant not used |
| `Hashing.c` | Tour pool dedup only matters for RUNS>1 |
| `BIT.c` | Segment-tree optimization; defer until needed |
| `GeoConversion.c` | Only for GEO edge weight; we use EUC_2D |
| `ReadProblem.c`, `ReadParameters.c`, `WriteTour.c` | Solver takes Rust structs, no file I/O |
| `BuildKDTree.c` | Use existing `kiddo` crate already in tsp_mt_core deps |
| Two-level tree (`Flip_SL.c`, `Between_SL.c`, etc.) | Plain doubly-linked list sufficient for chunk sizes; revisit if perf needs it |
| Three-level tree (`Flip_SSL.c`, `Between_SSL.c`) | Same |
| `Statistics.c`, `StatusReport.c`, `PrintParameters.c` | Use `log` crate |
| `CandidateReport.c` | Use `log` crate |
| Specialized output writers (`WriteCandidates.c`, `WritePenalties.c`, etc.) | Not needed |
| `Backbone*`, `IsBackboneCandidate.c` | Backbone heuristic deferred |

## Implementation phases

Phased to allow iteration with feedback rather than one big-bang commit:

**Phase A — scaffold (small)**
- New crate `crates/lin_kernighan` with `Cargo.toml`, `src/lib.rs`, empty modules
- `coord.rs`, `distance.rs`, `error.rs`, `params.rs` — types only
- Unit tests for distance (compare against known TSPLIB pr2392 distances)

**Phase B — minimal solver (medium)**
- `tour.rs` — doubly-linked list ops + tests
- `candidate.rs` — NN candidate set + tests
- `initial.rs` — greedy initial tour
- `lk/two_opt.rs` + `lk/gain.rs` — 2-opt only
- `solver.rs` — single-trial 2-opt convergence
- Sanity test: random 100-point instance converges

**Phase C — full LK (medium)**
- `lk/three_opt.rs` — 3-opt
- `lk/mod.rs` — k-opt dispatch (k=2,3)
- Trial loop + time limit
- Sanity test: pr2392 instance ≤5% over LKH optimum

**Phase D — integration (medium)**
- Rewire `tsp_mt_core/src/algo/runner/common.rs`
- Rewire `tsp_mt_core/src/algo/runner/mod.rs`
- Drop `lkh::embedded_lkh` import in `options.rs`
- Drop `lkh_exe: PathBuf` field + plumbing
- `cargo build --workspace` clean
- `cargo test --workspace` clean

**Phase E — quality push (only if needed)**
- `lk/k_opt.rs` — k=4,5
- `alpha.rs` — ALPHA candidate set
- `perturb.rs` — KSwapKick
- Re-benchmark against LKH; tune until ≤5% margin holds

**Phase F — cleanup (small)**
- Drop `fetch-lkh` feature from workspace + crates
- Remove `lkh` dep from `tsp_mt_core/Cargo.toml`
- Update README to reflect Apache-2.0-only distribution
- (User will move `lkh`/`lkh_derive` crates out of repo at their leisure)
