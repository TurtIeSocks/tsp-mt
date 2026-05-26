#!/usr/bin/env bash
# Compare the legacy LKH-binary solver against the native lin_kernighan
# Rust solver across a fixed grid of chunk sizes and input files.
#
# Inputs are picked up from benchmark/inputs/*.txt. The legacy binary is
# expected at /Users/rin/GitHub/tsp-mt/target/release/tsp-mt (main worktree,
# built with --features fetch-lkh). The new binary is built fresh from the
# current worktree.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LKH_BIN="${LKH_BIN:-/Users/rin/GitHub/tsp-mt/target/release/tsp-mt}"
NEW_BIN="$ROOT_DIR/target/release/tsp-mt"
INPUTS_DIR="$ROOT_DIR/benchmark/inputs"
RESULTS_DIR="$ROOT_DIR/benchmark/results"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY_TSV="$RESULTS_DIR/compare-${TIMESTAMP}.tsv"
REPORT_MD="$RESULTS_DIR/compare-${TIMESTAMP}.md"

mkdir -p "$RESULTS_DIR"

CHUNK_SIZES=(500 1000 2000 5000)
INPUTS=()
while IFS= read -r f; do
  INPUTS+=("$f")
done < <(find "$INPUTS_DIR" -maxdepth 1 -type f -name '*.txt' -print | LC_ALL=C sort)

if [[ "${#INPUTS[@]}" -eq 0 ]]; then
  echo "no inputs in $INPUTS_DIR" >&2
  exit 1
fi

if [[ ! -x "$LKH_BIN" ]]; then
  echo "legacy lkh binary not found at $LKH_BIN" >&2
  exit 1
fi

if [[ ! -x "$NEW_BIN" ]]; then
  echo "building lin_kernighan binary..." >&2
  (cd "$ROOT_DIR" && cargo build --release >&2)
fi

echo -e "solver\tinput\tn\tchunk_size\texit_code\ttotal_m\tlongest_m\tavg_m\tspikes\truntime_s" >"$SUMMARY_TSV"

run_one() {
  local solver_label="$1" bin="$2" input="$3" chunk="$4"
  local n
  n=$(awk 'NF { c++ } END { print c + 0 }' "$input")
  if (( chunk > n + 100 )); then
    return 1
  fi
  local stderr_file
  stderr_file=$(mktemp)
  local stdout_file
  stdout_file=$(mktemp)
  local start end runtime exit_code total longest avg spikes
  start=$(python3 -c 'import time;print(time.time())')
  set +e
  "$bin" \
    --log-level=info \
    --max-chunk-size="$chunk" \
    --input="$input" \
    --output="$stdout_file" \
    2>"$stderr_file"
  exit_code=$?
  set -e
  end=$(python3 -c 'import time;print(time.time())')
  runtime=$(python3 -c "print(f'{$end - $start:.3f}')")
  local metrics_line
  metrics_line=$(grep "INFO metrics:" "$stderr_file" | tail -n1 || true)
  total=$(echo "$metrics_line" | grep -oE 'total_m=[0-9]+' | cut -d= -f2)
  longest=$(echo "$metrics_line" | grep -oE 'longest_m=[0-9]+' | cut -d= -f2)
  avg=$(echo "$metrics_line" | grep -oE 'avg_m=[0-9]+' | cut -d= -f2)
  spikes=$(echo "$metrics_line" | grep -oE 'spikes=[0-9]+' | cut -d= -f2)
  printf '%s\t%s\t%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\n' \
    "$solver_label" "$(basename "$input")" "$n" "$chunk" "$exit_code" \
    "${total:-NA}" "${longest:-NA}" "${avg:-NA}" "${spikes:-NA}" "$runtime" \
    >>"$SUMMARY_TSV"
  rm -f "$stderr_file" "$stdout_file"
  return 0
}

total_runs=$((${#INPUTS[@]} * ${#CHUNK_SIZES[@]} * 2))
current=0
for input in "${INPUTS[@]}"; do
  for chunk in "${CHUNK_SIZES[@]}"; do
    for solver in lkh lin_kernighan; do
      current=$((current + 1))
      if [[ "$solver" == "lkh" ]]; then
        bin="$LKH_BIN"
      else
        bin="$NEW_BIN"
      fi
      echo "[run ${current}/${total_runs}] solver=${solver} input=$(basename "$input") chunk=${chunk}" >&2
      run_one "$solver" "$bin" "$input" "$chunk" || true
    done
  done
done

# Markdown report — pair lkh vs lin_kernighan side by side
python3 - <<PYEOF
import csv, collections, sys
rows = []
with open("$SUMMARY_TSV") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append(r)

paired = collections.defaultdict(dict)
for r in rows:
    key = (r["input"], r["chunk_size"])
    paired[key][r["solver"]] = r

with open("$REPORT_MD", "w") as out:
    out.write("# Solver Comparison: LKH vs lin_kernighan\n\n")
    out.write("All runs default to ``--solver-mode=multi-parallel`` (the project's main path); chunk_size varies per row.\n\n")
    out.write("Tour length (\`total_m\`) measured in meters via the great-circle metric in tsp_mt_core. ``Delta_pct`` = ``(lin_kernighan - lkh) / lkh * 100`` for tour length, negative means lin_kernighan tour is shorter.\n\n")
    out.write("| input | n | chunk_size | lkh total_m | lk total_m | total_m delta% | lkh runtime_s | lk runtime_s | runtime delta% | lkh spikes | lk spikes |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for (inp, chunk), pair in sorted(paired.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
        lkh = pair.get("lkh"); lk = pair.get("lin_kernighan")
        if not lkh or not lk: continue
        try:
            tot_lkh = int(lkh["total_m"]); tot_lk = int(lk["total_m"])
            rt_lkh = float(lkh["runtime_s"]); rt_lk = float(lk["runtime_s"])
            tot_delta = (tot_lk - tot_lkh) / tot_lkh * 100.0 if tot_lkh else 0.0
            rt_delta = (rt_lk - rt_lkh) / rt_lkh * 100.0 if rt_lkh else 0.0
        except ValueError:
            continue
        out.write(f"| {inp} | {lkh['n']} | {chunk} | {tot_lkh} | {tot_lk} | {tot_delta:+.2f} | {rt_lkh:.2f} | {rt_lk:.2f} | {rt_delta:+.2f} | {lkh['spikes']} | {lk['spikes']} |\n")
    out.write("\n## Per-input aggregates\n\n")
    out.write("| input | n | best lkh total_m | best lk total_m | lk vs lkh best | avg lkh runtime | avg lk runtime | runtime ratio |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    by_input = collections.defaultdict(lambda: {"lkh": [], "lin_kernighan": []})
    for r in rows:
        try:
            by_input[r["input"]][r["solver"]].append((int(r["total_m"]), float(r["runtime_s"]), int(r["n"])))
        except ValueError:
            continue
    for inp in sorted(by_input):
        d = by_input[inp]
        if not d["lkh"] or not d["lin_kernighan"]: continue
        n = d["lkh"][0][2]
        best_lkh = min(t for t, _, _ in d["lkh"])
        best_lk = min(t for t, _, _ in d["lin_kernighan"])
        avg_rt_lkh = sum(rt for _, rt, _ in d["lkh"]) / len(d["lkh"])
        avg_rt_lk = sum(rt for _, rt, _ in d["lin_kernighan"]) / len(d["lin_kernighan"])
        delta = (best_lk - best_lkh) / best_lkh * 100.0 if best_lkh else 0.0
        ratio = avg_rt_lk / avg_rt_lkh if avg_rt_lkh else 0.0
        out.write(f"| {inp} | {n} | {best_lkh} | {best_lk} | {delta:+.2f}% | {avg_rt_lkh:.2f} | {avg_rt_lk:.2f} | {ratio:.2f}× |\n")
PYEOF

echo "summary: $SUMMARY_TSV"
echo "report:  $REPORT_MD"
