#!/usr/bin/env bash
# Confirmation sweep for the data-driven `chunk_optimal(n)` chooser.
#
# Unlike sweep-chunk-explorer.sh (which sweeps an explicit chunk grid), this
# runs the solver through the REAL production path with NO --max-chunk-size, so
# the deployed `chunk_optimal(n)` chooser fires. It captures the chunk the
# binary actually picked (from the `dynamic_chunk_size` log line) and the
# resulting tour length, then compares against the best chunk seen for that `n`
# in a prior chunk-explorer aggregate.
#
# This specifically confirms the INTERPOLATED picks (e.g. 12000, 6000) that the
# geometric explorer grid never measured directly.
#
# Methodology mirrors the explorer so medians are comparable:
#   * n <= 25k:  3 distributions (d1,d2,d3), 1 rep each, median reported.
#   * n  > 25k:  1 distribution (d1), 2 reps, median reported.
#
# Usage:
#   scripts/confirm-chunk-optimal.sh [path-to-explorer-agg.tsv]
# Defaults to the most recent chunk-explorer-*-agg.tsv in benchmark/results.
#
# Set NO_RUN=1 to print the planned cells (n -> chosen-chunk) and exit.

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

# Resolve the explorer aggregate to compare against.
AGG_BASELINE="${1:-}"
if [[ -z "$AGG_BASELINE" ]]; then
  AGG_BASELINE="$(ls -t "$RESULTS_DIR"/chunk-explorer-*-agg.tsv 2>/dev/null | head -1 || true)"
fi
if [[ -z "$AGG_BASELINE" || ! -f "$AGG_BASELINE" ]]; then
  echo "error: no explorer agg TSV found; pass one as \$1" >&2
  exit 1
fi
echo "baseline: $AGG_BASELINE" >&2

ROWS_TSV="$RESULTS_DIR/chunk-confirm-${TIMESTAMP}-rows.tsv"
CMP_TSV="$RESULTS_DIR/chunk-confirm-${TIMESTAMP}-compare.tsv"
MD="$RESULTS_DIR/chunk-confirm-${TIMESTAMP}.md"

mkdir -p "$INPUTS_DIR" "$RESULTS_DIR"

INPUTS_DIR_NATIVE="$(to_native_path "$INPUTS_DIR")"
ROWS_TSV_NATIVE="$(to_native_path "$ROWS_TSV")"
CMP_TSV_NATIVE="$(to_native_path "$CMP_TSV")"
MD_NATIVE="$(to_native_path "$MD")"
AGG_BASELINE_NATIVE="$(to_native_path "$AGG_BASELINE")"

# Representative n spanning all three chunk_optimal regimes. Mid/large entries
# are chosen so chunk_optimal lands on values the explorer grid did NOT measure
# (10k,20k,30k,40k,50k -> n/3; 90k,120k,150k -> n/25), plus grid-matching
# controls (3k full-instance; 60k,200k,250k).
N_VALUES="${N_VALUES:-3000 10000 20000 30000 40000 50000 60000 90000 120000 150000 200000 250000}"

# ----- ensure input files exist (same generator as the explorer) -----
"$PYTHON" - <<PYEOF
import os, random
OUT_DIR = "$INPUTS_DIR_NATIVE"
n_values = [int(x) for x in "$N_VALUES".split()]

def dists_for(n): return [1, 2, 3] if n <= 25000 else [1]
def radius_for(n):
    if n <= 5_000: return 0.3
    if n <= 25_000: return 0.5
    return 0.8

for n in n_values:
    for dist in dists_for(n):
        path = f"{OUT_DIR}/n{n}_d{dist}.txt"
        if os.path.exists(path):
            with open(path) as f:
                if sum(1 for _ in f) == n:
                    continue
        random.seed(n * 100 + dist)
        r = radius_for(n)
        with open(path, "w") as f:
            for _ in range(n):
                f.write(f"{37.5 + random.uniform(-r, r):.6f},{-122.0 + random.uniform(-r, r):.6f}\n")
print("inputs ready")
PYEOF

if [[ "${NO_RUN:-0}" == "1" ]]; then
  echo "NO_RUN=1: planned n values: $N_VALUES" >&2
  exit 0
fi

# ----- run each cell through the real chooser (no --max-chunk-size) -----
echo -e "n\tdist\trep\tchosen_chunk\tlk\truntime_s" >"$ROWS_TSV"

for n in $N_VALUES; do
  if (( n <= 25000 )); then dists="1 2 3"; reps=1; else dists="1"; reps=2; fi
  for dist in $dists; do
    inp_native="$(to_native_path "$INPUTS_DIR/n${n}_d${dist}.txt")"
    for rep in $(seq 1 "$reps"); do
      echo "[n=$n d=$dist rep=$rep] running (auto chunk)..." >&2
      stderr=$(mktemp); stdout=$(mktemp)
      start=$("$PYTHON" -c 'import time;print(time.time())')
      "$NEW_BIN" --log-level=info --input="$inp_native" --output="$(to_native_path "$stdout")" 2>"$stderr"
      end=$("$PYTHON" -c 'import time;print(time.time())')
      runtime=$("$PYTHON" -c "print(f'{$end - $start:.2f}')")
      chunk=$(grep -oE 'dynamic_chunk_size = [0-9]+' "$stderr" | tail -1 | grep -oE '[0-9]+')
      lk=$(grep "INFO metrics:" "$stderr" | tail -n1 | grep -oE 'total_m=[0-9]+' | cut -d= -f2)
      printf '%d\t%d\t%d\t%s\t%s\t%s\n' "$n" "$dist" "$rep" "${chunk:-0}" "${lk:-0}" "$runtime" >>"$ROWS_TSV"
      rm -f "$stderr" "$stdout"
    done
  done
done

# ----- compare median(confirm) vs explorer best-for-n -----
"$PYTHON" - <<PYEOF
import csv, statistics

rows = list(csv.DictReader(open("$ROWS_TSV_NATIVE"), delimiter="\t"))
for r in rows:
    r["n"] = int(r["n"]); r["chosen_chunk"] = int(r["chosen_chunk"])
    r["lk"] = int(r["lk"]); r["runtime_s"] = float(r["runtime_s"])

# Explorer baseline: best (min) lk_median per n, and which chunk achieved it.
base = {}
for b in csv.DictReader(open("$AGG_BASELINE_NATIVE"), delimiter="\t"):
    n = int(b["n"]); lk = float(b["lk_median"]); ch = int(b["chunk"])
    if n not in base or lk < base[n][0]:
        base[n] = (lk, ch)

# Median confirm lk + runtime per n.
byn = {}
for r in rows:
    byn.setdefault(r["n"], []).append(r)

out = []
for n in sorted(byn):
    bucket = byn[n]
    lk_med = statistics.median(sorted(rr["lk"] for rr in bucket))
    rt_med = statistics.median(sorted(rr["runtime_s"] for rr in bucket))
    chosen = bucket[0]["chosen_chunk"]
    if n in base:
        best_lk, best_ch = base[n]
        gap = (lk_med - best_lk) / best_lk * 100.0
        note = "grid-match" if chosen == best_ch else "interpolated"
    else:
        best_lk, best_ch, gap, note = float("nan"), 0, float("nan"), "no-baseline"
    out.append(dict(n=n, chosen=chosen, lk_med=lk_med, rt=rt_med,
                    best_ch=best_ch, best_lk=best_lk, gap=gap, note=note))

with open("$CMP_TSV_NATIVE", "w") as f:
    f.write("n\tchosen_chunk\tconfirm_lk_median\truntime_s\tsweep_best_chunk\tsweep_best_lk\tgap_vs_best_pct\tnote\n")
    for o in out:
        f.write(f'{o["n"]}\t{o["chosen"]}\t{o["lk_med"]:.0f}\t{o["rt"]:.2f}\t'
                f'{o["best_ch"]}\t{o["best_lk"]:.0f}\t{o["gap"]:+.3f}\t{o["note"]}\n')

with open("$MD_NATIVE", "w") as f:
    f.write("# chunk_optimal confirmation sweep\n\n")
    f.write("Solver run with NO --max-chunk-size, so the deployed ")
    f.write("\`chunk_optimal(n)\` chooser selects the chunk. ")
    f.write("\`gap_vs_best_pct\` compares the confirmed tour to the best chunk ")
    f.write("seen for that \`n\` in the explorer sweep.\n\n")
    f.write("| n | chosen chunk | confirm lk | runtime | sweep best chunk | gap vs best | note |\n")
    f.write("|---:|---:|---:|---:|---:|---:|:--|\n")
    for o in out:
        f.write(f'| {o["n"]} | {o["chosen"]} | {o["lk_med"]:.0f} | {o["rt"]:.2f}s | '
                f'{o["best_ch"]} | {o["gap"]:+.3f}% | {o["note"]} |\n')

worst = max((o for o in out if o["gap"] == o["gap"]), key=lambda o: o["gap"])
print(f"confirmed {len(out)} n values; worst gap_vs_best = {worst['gap']:+.3f}% at n={worst['n']}", flush=True)
PYEOF

echo "" >&2
echo "rows    : $ROWS_TSV" >&2
echo "compare : $CMP_TSV" >&2
echo "md      : $MD" >&2
