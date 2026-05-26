# Trace — LKH usage in tsp-mt

Depth: **flow-level** (10k–100k LOC repo + scoped sub-system). Covers every code path that touches LKH.

## Repo facts

- LKH-3 C source: ~27k LOC, 172 files, ~169 extern globals.
- Symmetric Euclidean TSP subset of LKH: ~12k LOC.
- Repo Rust crates: lkh (~3.1k), lkh_derive (~425), tsp_mt_core (~4.1k), tsp_mt_derive (~770), bin (~95).
- LKH-3 README states: *"distributed for research use. The author reserves all rights to the code."* Project is Apache 2.0 — license mismatch is the refactor's motivation.

## LKH-facing surface in tsp_mt_core

Only three files in tsp_mt_core touch the `lkh::` crate:

| File | Lines | What it does |
|------|-------|--------------|
| `crates/tsp_mt_core/src/io/options.rs:8` | 1 | `use lkh::embedded_lkh;` — resolves embedded LKH binary path under `fetch-lkh` feature |
| `crates/tsp_mt_core/src/algo/runner/common.rs:3-9` | 7 | Imports `LkhError`, `LkhResult`, `parameters::{CandidateLimit, LkhParameters}`, `problem::*`, `solver::LkhSolver`, `tour::{TsplibTour, TsplibTourType}` |
| `crates/tsp_mt_core/src/algo/runner/mod.rs:1` | 1 | `use lkh::{LkhError, LkhResult, solver::LkhSolver};` |

LKH binary invocation count: 5 call sites total
- `common.rs:243` — chunk solver
- `mod.rs:43` — lkh_single
- `mod.rs:86-87` — lkh_multi_seed parallel runs
- `mod.rs:174-175` — centroid ordering tour
- `common.rs:240` — LkhSolver::new for chunks

`tsp_mt_core` uses `LkhError`/`LkhResult` as its return surface — these leak into `tsp_mt_core::runner` public fns (lkh_single, lkh_multi_seed, lkh_multi_parallel). That has to be retyped during refactor.

## LKH parameters actually set

Narrow subset of LKH's 87-knob parameter struct:

| Param | Value | Source |
|-------|-------|--------|
| `PROBLEM_FILE` | `<workdir>/<sub>/problem.tsp` | mandatory |
| `MAX_CANDIDATES` | `32 SYMMETRIC` | hardcoded `DEFAULT_MAX_CANDIDATES` |
| `MAX_TRIALS` | `clamp(n * 3, 1_000, 100_000)` | `scaled_max_trials` |
| `RUNS` | `1` | hardcoded `SINGLE_RUNS` |
| `SEED` | per-call u64 | from seed array / DEFAULT_BASE_SEED=12_345 |
| `TIME_LIMIT` | `max(n / 1024, 2)` seconds | `scaled_time_limit_seconds` (halved for `lkh_single`) |
| `TRACE_LEVEL` | 0 or 1 | 1 for chunks/seeds, 0 for centroid order |
| `INITIAL_TOUR_FILE` | optional path | when `options.use_initial_tour=true`; tour = `1..=n` (identity) |

Defaults LKH picks itself (not set by tsp_mt_core, so refactor needs to match LKH defaults):
- `CANDIDATE_SET_TYPE` = `ALPHA` (1-tree + subgradient ascent)
- `INITIAL_TOUR_ALGORITHM` = `WALK` for TSP (random walk) → unless INITIAL_TOUR_FILE set
- `MOVE_TYPE` = 5 (up to 5-opt)
- `KICK_TYPE` = 0 (no kicks)
- `PATCHING_A` = 1, `PATCHING_C` = 1
- `SUBGRADIENT` = YES
- `BACKTRACKING` = NO
- `GAIN23` = YES, `GAIN_CRITERION` = YES

## Input format (tsp_mt_core writes)

```
NAME: PROBLEM
TYPE: TSP
DIMENSION: <n>
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1  <(y*1000).round()>  <(x*1000).round()>
2  ...
EOF
```

- Coords pre-projected to a plane (PlaneProjection), then scaled ×1000 and rounded to integers (LKH expects integer-ish coords for EUC_2D).
- Note the swap: `y` is fed as the X column, `x` as Y. Refactor must preserve identical scaling so distances match.

## Output format (tsp_mt_core parses)

```
NAME: PROBLEM.tour
TYPE: TOUR
DIMENSION: <n>
TOUR_SECTION
<1-based id 1>
<1-based id 2>
...
-1
EOF
```

Parser at `crates/lkh/src/tour.rs:84-96` (`zero_based_tour`) just subtracts 1 from each id. Refactor: solver returns `Vec<usize>` directly, skipping the TSPLIB round-trip.

## Algorithm path (LKH-3 source, symmetric Euclidean TSP only)

Entry: `LKHmain.c:1-130`
```
main()
├─ ReadParameters()                            # parameter file parse
├─ ReadProblem()                               # .tsp file parse
├─ AllocateStructures()                        # Node[Dimension+1], BestTour, etc.
├─ CreateCandidateSet()                        # ALPHA candidates
│   ├─ MinimumSpanningTree()                   # 1-tree
│   ├─ Ascent()                                # subgradient ascent → Pi values
│   ├─ GenerateCandidates()                    # alpha-values from Pi → sort
│   └─ AdjustCandidateSet()
└─ for Run in 1..=Runs:
    └─ FindTour()                              # one run
        ├─ ChooseInitialTour()                 # WALK or GREEDY or load file
        └─ for Trial in 1..=MaxTrials or until time limit:
            ├─ LinKernighan()                  # main LK loop
            │   ├─ Activate(all nodes)
            │   └─ while active node t1:
            │       └─ try BestKOptMove(t1, k=2..MoveType)
            │           ├─ Best2OptMove / Best3OptMove / ... / Best5OptMove
            │           ├─ if Gain > 0: Make<k>OptMove + Flip
            │           └─ else: backtrack
            ├─ if better: RecordBetterTour
            └─ KSwapKick if KICK_TYPE > 0
```

For tsp_mt_core's invariants:
- `MOVE_TYPE` defaults to 5
- `KICK_TYPE` = 0 → no perturbation between trials (each trial restarts from current tour without kicks — see LKH source for what actually happens with KICK_TYPE=0)
- `RUNS=1` → no tour pool, no merging, no genetic

## Files in/out of scope for rewrite

### Needed (~12k LOC of LKH source)

| Category | Files |
|----------|-------|
| Entry + control | LKHmain.c, FindTour.c |
| LK main | LinKernighan.c, Best{2,3,4,5}OptMove.c, BestKOptMove.c, Make{2,3,4,5}OptMove.c, MakeKOptMove.c |
| Tour rep | Flip.c, Flip_SL.c, Between.c, Between_SL.c, RestoreTour.c |
| Candidate set | CreateCandidateSet.c, GenerateCandidates.c, MinimumSpanningTree.c, Minimum1TreeCost.c, Ascent.c, AddCandidate.c, OrderCandidateSet.c, TrimCandidateSet.c, SymmetrizeCandidateSet.c, AdjustCandidateSet.c, AddExtraCandidates.c, AddTourCandidates.c |
| Initial tour | ChooseInitialTour.c, GreedyTour.c |
| Gain | BridgeGain.c, Gain23.c, Improvement.c |
| Distance | Distance.c (EUC_2D only), C.c |
| Active queue | Activate.c, RemoveFirstActive.c |
| Move helpers | Excludable.c, Exclude.c, Forbidden.c (always-NO for TSP), IsCandidate.c, IsPossibleCandidate.c |
| I/O (not needed in Rust port — pass structs in/out) | ReadProblem.c, ReadParameters.c, WriteTour.c |

### NOT needed (~14k LOC excluded)

- All `Penalty_*.c` for non-TSP problems (~35 files)
- `Distance_SOP.c`, `Distance_MTSP.c`, `Distance_SPECIAL.c` — only EUC_2D matters
- Subproblem solvers (`Solve*Subproblems.c`) — RUNS=1 + no SUBPROBLEM_SIZE
- Genetic / recombination (`Genetic.c`, `MergeWithTour*.c`, `gpx.c`) — RUNS=1
- POPMUSIC (`Create_POPMUSIC_CandidateSet.c`)
- Delaunay (`Delaunay.c`, `CreateDelaunayCandidateSet.c`) — using ALPHA default
- Specialized initial tours (CTSP_, CVRP_, GCTSP_, MTSP_, SOP_, TSPDL_, PCTSP_, STTSP2TSP, MTSP2TSP)
- TSP variant reducers (TSPTW_Reduce, VRPB_Reduce, PDPTW_Reduce)
- BIT (segment-tree optimization for very large tours; deferrable)
- Hashing (tour pool dedup; deferrable since RUNS=1)
- GeoConversion (only for `GEO` edge weight type)
- StatusReport, Statistics — log-only

## Heuristic signals on tsp_mt_core

| File | Tags | Note |
|------|------|------|
| `crates/lkh/src/lib.rs` | [HOT][REPLACE-WHOLE] | Whole crate becomes optional once refactor lands |
| `crates/tsp_mt_core/src/algo/runner/common.rs` | [HOT][BRIDGE] | Bridge layer to LKH; primary rewrite target |
| `crates/tsp_mt_core/src/algo/runner/mod.rs` | [HOT][BRIDGE] | Same |
| `crates/tsp_mt_core/src/algo/runner/metric_spike.rs` | [INDEPENDENT] | Post-processing only, untouched by refactor |
| `crates/tsp_mt_core/src/algo/stitching.rs` | [INDEPENDENT] | Independent of LKH |
| `crates/tsp_mt_core/src/io/options.rs:8` | [REMOVE] | `lkh::embedded_lkh` import goes away |

## Global state (concern for Rust port)

LKH C declares ~169 extern globals in LKH.h. Refactor strategy: single `Solver` struct owning all state; pass `&mut Solver` through fn calls. Avoid `static mut`. Per-thread instances for parallel runs (since tsp_mt_core uses rayon for multi-seed/multi-chunk).
