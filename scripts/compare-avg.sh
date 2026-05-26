#!/usr/bin/env bash
# Averaged comparison: run each (solver, input, chunk) triple N times
# and report mean tour length + mean runtime. Reduces the run-to-run
# noise from multi-seed parallel solvers that show ~0.5–1pp variance
# between identical re-runs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LKH_BIN="${LKH_BIN:-/Users/rin/GitHub/tsp-mt/target/release/tsp-mt}"
NEW_BIN="$ROOT_DIR/target/release/tsp-mt"
INPUTS_DIR="$ROOT_DIR/benchmark/inputs"
RESULTS_DIR="$ROOT_DIR/benchmark/results"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TSV="$RESULTS_DIR/avg-${TIMESTAMP}.tsv"
MD="$RESULTS_DIR/avg-${TIMESTAMP}.md"
RUNS="${RUNS:-3}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"

mkdir -p "$RESULTS_DIR"

INPUTS=()
while IFS= read -r f; do
  INPUTS+=("$f")
done < <(find "$INPUTS_DIR" -maxdepth 1 -type f -name '*.txt' -print | LC_ALL=C sort)

if [[ ! -x "$NEW_BIN" ]]; then
  (cd "$ROOT_DIR" && cargo build --release >&2)
fi

echo -e "solver\tinput\tn\trun\ttotal_m\truntime_s" >"$TSV"

run_once() {
  local solver_label="$1" bin="$2" input="$3" run="$4"
  local stderr=$(mktemp) stdout=$(mktemp)
  local start end runtime total
  start=$(python3 -c 'import time;print(time.time())')
  set +e
  "$bin" \
    --log-level=info \
    --max-chunk-size="$CHUNK_SIZE" \
    --input="$input" \
    --output="$stdout" \
    2>"$stderr"
  set -e
  end=$(python3 -c 'import time;print(time.time())')
  runtime=$(python3 -c "print(f'{$end - $start:.3f}')")
  total=$(grep "INFO metrics:" "$stderr" | tail -n1 | grep -oE 'total_m=[0-9]+' | cut -d= -f2)
  local n
  n=$(awk 'NF { c++ } END { print c + 0 }' "$input")
  printf '%s\t%s\t%d\t%d\t%s\t%s\n' "$solver_label" "$(basename "$input")" "$n" "$run" "${total:-NA}" "$runtime" >>"$TSV"
  rm -f "$stderr" "$stdout"
}

total_runs=$((${#INPUTS[@]} * RUNS * 2))
current=0
for input in "${INPUTS[@]}"; do
  for run in $(seq 1 "$RUNS"); do
    for solver in lkh lin_kernighan; do
      current=$((current + 1))
      bin="$NEW_BIN"
      [[ "$solver" == "lkh" ]] && bin="$LKH_BIN"
      echo "[$current/$total_runs] solver=$solver input=$(basename "$input") run=$run" >&2
      run_once "$solver" "$bin" "$input" "$run"
    done
  done
done

python3 - <<PYEOF
import csv, statistics
rows = []
with open("$TSV") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append(r)

by = {}
for r in rows:
    k = (r["solver"], r["input"], int(r["n"]))
    by.setdefault(k, []).append((int(r["total_m"]), float(r["runtime_s"])))

with open("$MD", "w") as out:
    out.write(f"# Averaged Solver Comparison (RUNS={$RUNS}, CHUNK_SIZE={$CHUNK_SIZE})\n\n")
    out.write("| input | n | lkh mean tour_m | lkh stdev | lk mean tour_m | lk stdev | gap | lkh mean runtime | lk mean runtime | ratio |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    inputs = sorted({k[1] for k in by})
    for input in inputs:
        rows_lkh = by.get(("lkh", input, None))
        # lookup with any n
        n = None
        for k, v in by.items():
            if k[0] == "lkh" and k[1] == input:
                n = k[2]; rows_lkh = v; break
        rows_lk = None
        for k, v in by.items():
            if k[0] == "lin_kernighan" and k[1] == input:
                rows_lk = v; break
        if not rows_lkh or not rows_lk: continue
        lkh_tour = [t for t, _ in rows_lkh]
        lkh_rt = [r for _, r in rows_lkh]
        lk_tour = [t for t, _ in rows_lk]
        lk_rt = [r for _, r in rows_lk]
        m_lkh = statistics.mean(lkh_tour); s_lkh = statistics.pstdev(lkh_tour)
        m_lk = statistics.mean(lk_tour); s_lk = statistics.pstdev(lk_tour)
        rt_lkh = statistics.mean(lkh_rt); rt_lk = statistics.mean(lk_rt)
        gap = (m_lk - m_lkh) / m_lkh * 100.0
        ratio = rt_lk / rt_lkh if rt_lkh else 0.0
        out.write(f"| {input} | {n} | {m_lkh:.0f} | {s_lkh:.0f} | {m_lk:.0f} | {s_lk:.0f} | {gap:+.2f}% | {rt_lkh:.2f}s | {rt_lk:.2f}s | {ratio:.2f}× |\n")
PYEOF

echo "tsv:    $TSV"
echo "report: $MD"
