# TSPLIB Problem and Tour Specifications (as modeled by this crate)

This document describes the TSPLIB/LKH problem and tour surfaces currently
modeled by:

- `lkh::problem::TsplibProblem` (`.tsp` / problem files)
- `lkh::tour::TsplibTour` (`.tour` files)

It is intentionally writer-oriented for `TsplibProblem` and parser-oriented for
`TsplibTour`.

## `TsplibProblem` (`.tsp`) Format

### Header Style

`TsplibProblem` writes header lines using:

- `KEY: VALUE`

Always written:

- `NAME: ...`
- `TYPE: ...`

Optionally written when populated:

- `COMMENT: ...` (repeatable)
- `DIMENSION: ...`
- `CAPACITY: ...`
- `DISTANCE: ...`
- `EDGE_WEIGHT_TYPE: ...`
- `EDGE_WEIGHT_FORMAT: ...`
- `EDGE_DATA_FORMAT: ...`
- `NODE_COORD_TYPE: ...`
- `DISPLAY_DATA_TYPE: ...`
- `DEMAND_DIMENSION: ...`
- `GRID_SIZE: ...`
- `GROUPS: ...`
- `GVRP_SETS: ...`
- `RELAXATION_LEVEL: ...`
- `RISK_THRESHOLD: ...`
- `SALESMEN: ...`
- `VEHICLES: ...`
- `SCALE: ...`
- `SERVICE_TIME: ...`

### Supported `TYPE` Values

- `TSP`
- `ATSP`
- `SOP`
- `HCP`
- `HPP`
- `BWTSP`
- `CLUVRP`
- `CCVRP`
- `CVRP`
- `ACVRP`
- `ADCVRP`
- `CVRPTW`
- `KTSP`
- `MLP`
- `MSCTSP`
- `OVRP`
- `PCTSP`
- `PDPTW`
- `PDTSP`
- `PDTSPF`
- `PDTSPL`
- `PTSP`
- `TRP`
- `RCTVRP`
- `RCTVRPTW`
- `SOFTCLUVRP`
- `STTSP`
- `TSPTW`
- `VRPB`
- `VRPBTW`
- `VRPPD`
- `1-PDTSP`
- `M-PDTSP`
- `M1-PDTSP`
- `TSPDL`
- `CTSP`
- `CTSP-D`
- `GCTSP`
- `CCCTSP`
- `CBTSP`
- `CBNTSP`
- `TOUR`

### Supported `EDGE_WEIGHT_TYPE` Values

- `EXPLICIT`
- `EUC_2D`
- `EUC_3D`
- `MAX_2D`
- `MAX_3D`
- `MAN_2D`
- `MAN_3D`
- `CEIL_2D`
- `CEIL_3D`
- `EXACT_2D`
- `EXACT_3D`
- `FLOOR_2D`
- `FLOOR_3D`
- `GEO`
- `GEOM`
- `GEO_MEEUS`
- `GEOM_MEEUS`
- `TOR_2D`
- `TOR_3D`
- `ATT`
- `XRAY1`
- `XRAY2`
- `SPECIAL`

### Supported `EDGE_WEIGHT_FORMAT` Values

- `FUNCTION`
- `FULL_MATRIX`
- `UPPER_ROW`
- `LOWER_ROW`
- `UPPER_DIAG_ROW`
- `LOWER_DIAG_ROW`
- `UPPER_COL`
- `LOWER_COL`
- `UPPER_DIAG_COL`
- `LOWER_DIAG_COL`

### Supported `EDGE_DATA_FORMAT` Values

- `EDGE_LIST`
- `ADJ_LIST`

### Supported `NODE_COORD_TYPE` Values

- `TWOD_COORDS`
- `THREED_COORDS`
- `NO_COORDS`

### Supported `DISPLAY_DATA_TYPE` Values

- `COORD_DISPLAY`
- `TWOD_DISPLAY`
- `NO_DISPLAY`

## Supported Sections

Section emission is conditional: empty vectors are omitted.

- `NODE_COORD_SECTION`
  - rows: `<id> <x> <y> [z]`
- `EDGE_DATA_SECTION`
  - `EDGE_LIST` rows: `<from> <to> [weight]`
  - `ADJ_LIST` rows: `<node> <neighbor1> ... -1`
  - section terminator: `-1`
- `FIXED_EDGES_SECTION`
  - rows: `<from> <to>`
  - section terminator: `-1`
- `DISPLAY_DATA_SECTION`
  - rows: `<id> <x> <y>`
- `EDGE_WEIGHT_SECTION`
  - one matrix row per line, space-separated
- `TOUR_SECTION`
  - one node id per line
  - section terminator: `-1`
- `BACKHAUL_SECTION`
  - one node id per line
  - section terminator: `-1`
- `CTSP_SET_SECTION`
  - rows: `<set_id> <member...> -1`
- `DEMAND_SECTION`
  - rows: `<id> <demand...>`
- `DEPOT_SECTION`
  - one depot id per line
  - section terminator: `-1`
- `DRAFT_LIMIT_SECTION`
  - rows: `<id> <draft_limit>`
- `GCTSP_SECTION`
  - matrix rows of integer flags (space-separated)
- `GCTSP_SET_SECTION`
  - rows: `<set_id> <member...> -1`
- `GROUP_SECTION`
  - rows: `<set_id> <member...> -1`
- `GVRP_SET_SECTION`
  - rows: `<set_id> <member...> -1`
- `PICKUP_AND_DELIVERY_SECTION`
  - rows: `<id> <demand> <earliest> <latest> <service_time> <pickup> <delivery>`
- `REQUIRED_NODES_SECTION`
  - one node id per line
  - section terminator: `-1`
- `SERVICE_TIME_SECTION`
  - rows: `<id> <service_time>`
- `TIME_WINDOW_SECTION`
  - rows: `<id> <earliest> <latest>`

Optional final marker:

- `EOF` (controlled by `emit_eof`, default `true` in `TsplibProblem::new`).

## Convenience Constructor

`TsplibProblem::from_euc2d_points(...)` initializes:

- `TYPE = TSP`
- `EDGE_WEIGHT_TYPE = EUC_2D`
- `NODE_COORD_TYPE = TWOD_COORDS`
- `DIMENSION = points.len()`
- `NODE_COORD_SECTION` with 1-based node ids

## `TsplibTour` (`.tour`) Format

### Parsed Headers

The parser recognizes:

- `NAME: ...`
- `COMMENT: ...` (repeatable)
- `TYPE: TOUR`
- `DIMENSION: ...`
- `OPTIMUM: ...` or `OPTIMUM = ...`
- `TOUR_SECTION`
- `EOF`

Unknown headers are ignored.

Header assignments are accepted with either `:` or `=` delimiters.

### Tour Section Rules

- `TOUR_SECTION` is required.
- Node ids are expected to be TSPLIB 1-based ids.
- `-1` terminates the section.
- `EOF` also terminates parsing.

### Permissive Parser Behavior

The parser is intentionally permissive:

- Non-positive node ids in `TOUR_SECTION` other than the `-1` terminator are skipped.
- If `DIMENSION` is present, it must match the parsed node count.
- If `DIMENSION` is absent, no count check is enforced.

### Conversion to Internal Ordering

`TsplibTour::zero_based_tour()` converts parsed ids from 1-based to 0-based:

- input id `1` -> output `0`
- input id `N` -> output `N-1`

## Practical Interop Notes

- `TsplibProblem` is write-oriented in this crate (no parser implemented here).
- `TsplibTour` is both readable and writable.
- For end-to-end solving, `LkhSolver` writes `.tsp` + `.par`, executes LKH,
  and reads back `.tour`.
