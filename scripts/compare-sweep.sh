#!/usr/bin/env bash
# Chunk-size sweep across the full input corpus.
# For each input, walks ascending chunk sizes until one is >= n;
# further sizes would all behave identically (single-chunk path)
# so they're skipped.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LKH_BIN="${LKH_BIN:-/Users/rin/GitHub/tsp-mt/target/release/tsp-mt}"
NEW_BIN="$ROOT_DIR/target/release/tsp-mt"
INPUTS_DIR="$ROOT_DIR/benchmark/inputs"
RESULTS_DIR="$ROOT_DIR/benchmark/results"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TSV="$RESULTS_DIR/sweep-${TIMESTAMP}.tsv"
MD="$RESULTS_DIR/sweep-${TIMESTAMP}.md"
RUNS="${RUNS:-1}"

mkdir -p "$RESULTS_DIR"

CHUNK_SIZES=(500 1000 2500 5000 10000 25000)

INPUTS=()
while IFS= read -r f; do
  INPUTS+=("$f")
done < <(find "$INPUTS_DIR" -maxdepth 1 -type f -name '*.txt' -print | LC_ALL=C sort -V)

if [[ ! -x "$NEW_BIN" ]]; then
  (cd "$ROOT_DIR" && cargo build --release >&2)
fi

echo -e "solver\tinput\tn\tchunk_size\trun\ttotal_m\truntime_s" >"$TSV"

run_once() {
  local solver_label="$1" bin="$2" input="$3" chunk="$4" run="$5"
  local stderr=$(mktemp) stdout=$(mktemp)
  local start end runtime total
  start=$(python3 -c 'import time;print(time.time())')
  set +e
  "$bin" \
    --log-level=info \
    --max-chunk-size="$chunk" \
    --input="$input" \
    --output="$stdout" \
    2>"$stderr"
  set -e
  end=$(python3 -c 'import time;print(time.time())')
  runtime=$(python3 -c "print(f'{$end - $start:.3f}')")
  total=$(grep "INFO metrics:" "$stderr" | tail -n1 | grep -oE 'total_m=[0-9]+' | cut -d= -f2)
  local n
  n=$(awk 'NF { c++ } END { print c + 0 }' "$input")
  printf '%s\t%s\t%d\t%d\t%d\t%s\t%s\n' \
    "$solver_label" "$(basename "$input")" "$n" "$chunk" "$run" "${total:-NA}" "$runtime" \
    >>"$TSV"
  rm -f "$stderr" "$stdout"
}

# Pre-count runs for progress display
total_runs=0
for input in "${INPUTS[@]}"; do
  n=$(awk 'NF { c++ } END { print c + 0 }' "$input")
  prev=-1
  for chunk in "${CHUNK_SIZES[@]}"; do
    if (( prev >= n )); then break; fi
    total_runs=$(( total_runs + 2 * RUNS ))
    prev=$chunk
  done
done

current=0
for input in "${INPUTS[@]}"; do
  n=$(awk 'NF { c++ } END { print c + 0 }' "$input")
  prev=-1
  for chunk in "${CHUNK_SIZES[@]}"; do
    if (( prev >= n )); then break; fi
    for run in $(seq 1 "$RUNS"); do
      for solver in lkh lin_kernighan; do
        current=$((current + 1))
        bin="$NEW_BIN"
        [[ "$solver" == "lkh" ]] && bin="$LKH_BIN"
        echo "[$current/$total_runs] solver=$solver input=$(basename "$input") n=$n chunk=$chunk run=$run" >&2
        run_once "$solver" "$bin" "$input" "$chunk" "$run"
      done
    done
    prev=$chunk
  done
done

# Aggregate report
python3 - <<PYEOF
import csv, statistics, collections
rows = []
with open("$TSV") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append(r)

# (input, n, chunk) -> {solver: [(total, runtime), ...]}
buckets = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    try:
        key = (r["input"], int(r["n"]), int(r["chunk_size"]))
        buckets[key][r["solver"]].append((int(r["total_m"]), float(r["runtime_s"])))
    except ValueError:
        continue

with open("$MD", "w") as out:
    out.write(f"# Chunk-Size Sweep (RUNS={$RUNS})\n\n")
    out.write("Per-input chunk sweep. Skips chunk sizes once the previous chunk was >= n (single-chunk path is equivalent for all larger chunk sizes).\n\n")
    out.write("| input | n | chunk | lkh tour_m | lk tour_m | gap | lkh runtime | lk runtime | ratio |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for (inp, n, chunk), solvers in sorted(buckets.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        lkh = solvers.get("lkh")
        lk = solvers.get("lin_kernighan")
        if not lkh or not lk: continue
        m_lkh = statistics.mean(t for t, _ in lkh)
        m_lk = statistics.mean(t for t, _ in lk)
        rt_lkh = statistics.mean(r for _, r in lkh)
        rt_lk = statistics.mean(r for _, r in lk)
        gap = (m_lk - m_lkh) / m_lkh * 100.0 if m_lkh else 0.0
        ratio = rt_lk / rt_lkh if rt_lkh else 0.0
        out.write(f"| {inp} | {n} | {chunk} | {m_lkh:.0f} | {m_lk:.0f} | {gap:+.2f}% | {rt_lkh:.2f}s | {rt_lk:.2f}s | {ratio:.2f}× |\n")

    out.write("\n## Best chunk per input (by tour length)\n\n")
    out.write("| input | n | lkh best chunk | lkh best tour_m | lk best chunk | lk best tour_m | best-vs-best gap |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|\n")
    by_input = collections.defaultdict(lambda: {"lkh": [], "lin_kernighan": []})
    for (inp, n, chunk), solvers in buckets.items():
        for s, recs in solvers.items():
            m = statistics.mean(t for t, _ in recs)
            by_input[(inp, n)][s].append((chunk, m))
    for (inp, n), d in sorted(by_input.items(), key=lambda kv: kv[0][1]):
        if not d["lkh"] or not d["lin_kernighan"]: continue
        best_lkh = min(d["lkh"], key=lambda x: x[1])
        best_lk = min(d["lin_kernighan"], key=lambda x: x[1])
        gap = (best_lk[1] - best_lkh[1]) / best_lkh[1] * 100.0
        out.write(f"| {inp} | {n} | {best_lkh[0]} | {best_lkh[1]:.0f} | {best_lk[0]} | {best_lk[1]:.0f} | {gap:+.2f}% |\n")
PYEOF

echo "tsv:    $TSV"
echo "report: $MD"
