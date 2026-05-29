#!/usr/bin/env bash
# Chunk-size data collection sweep — generates the dataset that the
# eventual adaptive `chunk_optimal(n)` chooser will train on.
#
# This is NOT a regression check vs LKH. Lk_new is recorded raw and
# ranked within each `n` (gap_vs_best_within_n) so chunk-size effects
# can be compared without an external baseline.
#
# Configuration (per item 1, 2, 4 from the discussion):
#
#   * Item 1 — chunk geometry: 5 chunks per `n` along a geometric
#     spacing between 1k and `min(50k, n)`. Denser at the small-chunk
#     side where the quality curve typically lives. Rounded to the
#     nearest 500 and de-duplicated.
#
#   * Item 2 — distribution count regime split:
#       - n ≤ 25k:  3 spatial distributions (d1, d2, d3).
#       - n > 25k:  1 distribution (d1).
#     Smaller `n` are below the time-budget cap and respond to
#     per-distribution variance; larger `n` are dominated by the
#     budget itself, so extra distributions buy little.
#
#   * Item 4 — variance at the scaling tail:
#       - n ≤ 25k:  1 rep per (n, chunk, dist).
#       - n > 25k:  2 reps per (n, chunk, dist), median reported.
#     Wall-clock variance dominates above the budget cap; reps catch
#     scheduler / RNG-walk jitter.
#
# Inputs are generated on-demand into benchmark/chunk-size-study/inputs/
# (uniform random in a radius around 37.5, -122.0; seed = n*100 + dist).
# These fixtures are deterministic and committed to the repo.
#
# Output (committed under benchmark/chunk-size-study/results/):
#   - Raw rows TSV  : chunk-explorer-<ts>-rows.tsv
#   - Aggregated TSV: chunk-explorer-<ts>-agg.tsv
#                     (median per (n, chunk) with gap_vs_best_within_n)
#   - Markdown      : chunk-explorer-<ts>.md
#
# Set NO_RUN=1 to print the planned cells and exit (no binary calls).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEW_BIN="$ROOT_DIR/target/release/tsp-mt"
[[ -f "${NEW_BIN}.exe" ]] && NEW_BIN="${NEW_BIN}.exe"
INPUTS_DIR="$ROOT_DIR/benchmark/chunk-size-study/inputs"
RESULTS_DIR="$ROOT_DIR/benchmark/chunk-size-study/results"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

PYTHON="$(command -v python3 || command -v python || true)"
if [[ -z "$PYTHON" ]]; then
  echo "error: python3/python not found on PATH" >&2
  exit 1
fi

# On MSYS2/Cygwin/Git Bash, native Python and tsp-mt.exe don't understand
# /c/foo paths. cygpath -m emits C:/foo (forward slashes, accepted by both).
# On real Unix cygpath is absent and the helper passes paths through.
to_native_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$1"
  else
    printf '%s' "$1"
  fi
}

ROWS_TSV="$RESULTS_DIR/chunk-explorer-${TIMESTAMP}-rows.tsv"
AGG_TSV="$RESULTS_DIR/chunk-explorer-${TIMESTAMP}-agg.tsv"
MD="$RESULTS_DIR/chunk-explorer-${TIMESTAMP}.md"
PLAN="${TMPDIR:-/tmp}/chunk-explorer-plan-${TIMESTAMP}.tsv"

mkdir -p "$INPUTS_DIR" "$RESULTS_DIR"

INPUTS_DIR_NATIVE="$(to_native_path "$INPUTS_DIR")"
ROWS_TSV_NATIVE="$(to_native_path "$ROWS_TSV")"
AGG_TSV_NATIVE="$(to_native_path "$AGG_TSV")"
MD_NATIVE="$(to_native_path "$MD")"
PLAN_NATIVE="$(to_native_path "$PLAN")"

# ----- generate the cell list + ensure input files exist -----
"$PYTHON" - <<PYEOF
import math, os, random, sys

OUT_DIR = "$INPUTS_DIR_NATIVE"
PLAN_PATH = "$PLAN_NATIVE"

# n grid: every 1k from 1k-10k, every 5k from 10k-50k, every 10k from 50k-250k.
n_values = sorted(set(
    list(range(1000, 10001, 1000)) +
    list(range(10000, 50001, 5000)) +
    list(range(50000, 250001, 10000))
))

def chunks_for(n: int):
    """Geometric spacing of 5 chunks between 1k and min(50k, n), rounded
    to nearest 500 and deduplicated. For n < 1k the sequence collapses
    to a single value."""
    lo = 1000
    hi = min(50000, n)
    if hi <= lo:
        return [lo]
    raw = [int(round(lo * (hi / lo) ** (i / 4))) for i in range(5)]
    raw = [max(500, (v + 250) // 500 * 500) for v in raw]
    seen, out = set(), []
    for v in raw:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def dists_for(n: int):
    return [1, 2, 3] if n <= 25000 else [1]

def reps_for(n: int):
    return 2 if n > 25000 else 1

def radius_for(n: int):
    if n <= 5_000:
        return 0.3
    if n <= 25_000:
        return 0.5
    return 0.8

def ensure_input(n: int, dist: int) -> str:
    # Forward-slash join so the path written to the plan TSV is consistent
    # on both Windows (C:/...) and Unix (/...); both accept forward slashes.
    path = f"{OUT_DIR}/n{n}_d{dist}.txt"
    if os.path.exists(path):
        with open(path) as f:
            if sum(1 for _ in f) == n:
                return path
    radius = radius_for(n)
    random.seed(n * 100 + dist)
    with open(path, "w") as f:
        for _ in range(n):
            lat = 37.5 + random.uniform(-radius, radius)
            lng = -122.0 + random.uniform(-radius, radius)
            f.write(f"{lat:.6f},{lng:.6f}\n")
    return path

# Emit one row per planned run: n, chunk, dist, rep, input_path.
plan_rows = []
for n in n_values:
    chunks = chunks_for(n)
    dists = dists_for(n)
    reps = reps_for(n)
    for dist in dists:
        path = ensure_input(n, dist)
        for chunk in chunks:
            for rep in range(1, reps + 1):
                plan_rows.append((n, chunk, dist, rep, path))

with open(PLAN_PATH, "w") as f:
    f.write("n\tchunk\tdist\trep\tinput_path\n")
    for r in plan_rows:
        f.write("\t".join(str(x) for x in r) + "\n")

print(f"planned {len(plan_rows)} cells across {len(n_values)} n values", file=sys.stderr)
PYEOF

TOTAL=$(awk 'NR>1' "$PLAN" | wc -l | tr -d ' ')
echo "total cells: $TOTAL" >&2

if [[ "${NO_RUN:-0}" == "1" ]]; then
  echo "NO_RUN=1, printing plan and exiting" >&2
  column -t -s $'\t' "$PLAN" | head -40
  echo "..." >&2
  exit 0
fi

# ----- run each cell, append to rows TSV -----
echo -e "n\tchunk\tdist\trep\tlk_new\truntime_s" >"$ROWS_TSV"

i=0
while IFS=$'\t' read -r n chunk dist rep input_path; do
  [[ "$n" == "n" ]] && continue   # header
  i=$((i + 1))
  echo "[$i/$TOTAL] n=$n chunk=$chunk d=$dist rep=$rep" >&2
  stderr=$(mktemp); stdout=$(mktemp)
  start=$("$PYTHON" -c 'import time;print(time.time())')
  "$NEW_BIN" --log-level=info --max-chunk-size="$chunk" --input="$input_path" --output="$stdout" 2>"$stderr"
  end=$("$PYTHON" -c 'import time;print(time.time())')
  runtime=$("$PYTHON" -c "print(f'{$end - $start:.2f}')")
  lk_new=$(grep "INFO metrics:" "$stderr" | tail -n1 | grep -oE 'total_m=[0-9]+' | cut -d= -f2)
  printf '%d\t%d\t%d\t%d\t%s\t%s\n' "$n" "$chunk" "$dist" "$rep" "$lk_new" "$runtime" >>"$ROWS_TSV"
  rm -f "$stderr" "$stdout"
done < "$PLAN"

# ----- aggregate: median lk_new per (n, chunk), gap_vs_best_within_n -----
"$PYTHON" - <<PYEOF
import csv, statistics

rows = list(csv.DictReader(open("$ROWS_TSV_NATIVE"), delimiter="\t"))
for r in rows:
    r["n"] = int(r["n"])
    r["chunk"] = int(r["chunk"])
    r["lk_new"] = int(r["lk_new"])
    r["runtime_s"] = float(r["runtime_s"])

# Median per (n, chunk) over (dist, rep).
grouped = {}
for r in rows:
    grouped.setdefault((r["n"], r["chunk"]), []).append(r)

agg = []
for (n, chunk), bucket in grouped.items():
    lks = sorted(rr["lk_new"] for rr in bucket)
    rts = sorted(rr["runtime_s"] for rr in bucket)
    agg.append({
        "n": n,
        "chunk": chunk,
        "reps_total": len(bucket),
        "lk_median": statistics.median(lks),
        "lk_min": lks[0],
        "lk_max": lks[-1],
        "runtime_median": statistics.median(rts),
    })

# gap_vs_best_within_n = (lk_median - min_chunk_lk_median_for_this_n) / min_chunk_lk_median_for_this_n
best_per_n = {}
for a in agg:
    cur = best_per_n.get(a["n"])
    if cur is None or a["lk_median"] < cur:
        best_per_n[a["n"]] = a["lk_median"]

for a in agg:
    best = best_per_n[a["n"]]
    a["gap_vs_best"] = (a["lk_median"] - best) / best * 100.0

agg.sort(key=lambda a: (a["n"], a["chunk"]))

with open("$AGG_TSV_NATIVE", "w") as f:
    f.write("n\tchunk\treps\tlk_median\tlk_min\tlk_max\tgap_vs_best_pct\truntime_median_s\n")
    for a in agg:
        f.write(
            f'{a["n"]}\t{a["chunk"]}\t{a["reps_total"]}\t'
            f'{a["lk_median"]}\t{a["lk_min"]}\t{a["lk_max"]}\t'
            f'{a["gap_vs_best"]:+.3f}\t{a["runtime_median"]:.2f}\n'
        )

with open("$MD_NATIVE", "w") as f:
    f.write("# Chunk-explorer sweep\n\n")
    f.write("Rows show median across distributions and reps. ")
    f.write("\`gap_vs_best_pct\` is relative to the best chunk seen for each \`n\`.\n\n")
    f.write("| n | chunk | reps | lk median | gap_vs_best | runtime |\n")
    f.write("|---:|---:|---:|---:|---:|---:|\n")
    for a in agg:
        f.write(
            f'| {a["n"]} | {a["chunk"]} | {a["reps_total"]} | '
            f'{a["lk_median"]} | {a["gap_vs_best"]:+.3f}% | '
            f'{a["runtime_median"]:.2f}s |\n'
        )

print(f"aggregated {len(agg)} (n, chunk) cells", flush=True)
PYEOF

echo "" >&2
echo "rows  : $ROWS_TSV" >&2
echo "agg   : $AGG_TSV" >&2
echo "md    : $MD" >&2
