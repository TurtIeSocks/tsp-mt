#!/usr/bin/env bash
# Resume the chunk-size sweep skipping chunk=50000 (LKH single-chunk
# path on n≥50k takes >20 minutes per run). Appends to the most-recent
# sweep TSV; regenerates the markdown report at the end.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LKH_BIN="${LKH_BIN:-/Users/rin/GitHub/tsp-mt/target/release/tsp-mt}"
NEW_BIN="$ROOT_DIR/target/release/tsp-mt"
INPUTS_DIR="$ROOT_DIR/benchmark/inputs"
RESULTS_DIR="$ROOT_DIR/benchmark/results"
TSV="$(ls -t "$RESULTS_DIR"/sweep-*.tsv | head -1)"
MD="${TSV%.tsv}.md"

CHUNK_SIZES=(500 1000 2500 5000 10000 25000)  # 50000 dropped

INPUTS=()
while IFS= read -r f; do
  INPUTS+=("$f")
done < <(find "$INPUTS_DIR" -maxdepth 1 -type f -name '*.txt' -print | LC_ALL=C sort -V)

already_done() {
  local solver="$1" input_base="$2" chunk="$3"
  awk -F'\t' -v s="$solver" -v i="$input_base" -v c="$chunk" \
    'NR>1 && $1==s && $2==i && $4==c {found=1} END {exit !found}' "$TSV"
}

run_once() {
  local solver_label="$1" bin="$2" input="$3" chunk="$4"
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
    "$solver_label" "$(basename "$input")" "$n" "$chunk" 1 "${total:-NA}" "$runtime" \
    >>"$TSV"
  rm -f "$stderr" "$stdout"
}

# Plan missing combos
declare -a TODO=()
for input in "${INPUTS[@]}"; do
  base=$(basename "$input")
  n=$(awk 'NF { c++ } END { print c + 0 }' "$input")
  prev=-1
  for chunk in "${CHUNK_SIZES[@]}"; do
    if (( prev >= n )); then break; fi
    for solver in lkh lin_kernighan; do
      if ! already_done "$solver" "$base" "$chunk"; then
        TODO+=("$solver|$input|$chunk")
      fi
    done
    prev=$chunk
  done
done

total=${#TODO[@]}
echo "$total runs to do" >&2

current=0
for entry in "${TODO[@]}"; do
  current=$((current + 1))
  IFS='|' read -r solver input chunk <<< "$entry"
  bin="$NEW_BIN"
  [[ "$solver" == "lkh" ]] && bin="$LKH_BIN"
  echo "[$current/$total] solver=$solver input=$(basename "$input") chunk=$chunk" >&2
  run_once "$solver" "$bin" "$input" "$chunk"
done

# Regenerate markdown
python3 - <<PYEOF
import csv, statistics, collections
rows = []
with open("$TSV") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append(r)

buckets = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    try:
        key = (r["input"], int(r["n"]), int(r["chunk_size"]))
        buckets[key][r["solver"]].append((int(r["total_m"]), float(r["runtime_s"])))
    except ValueError:
        continue

with open("$MD", "w") as out:
    out.write(f"# Chunk-Size Sweep (chunk=50000 skipped)\n\n")
    out.write("| input | n | chunk | lkh tour_m | lk tour_m | gap | lkh rt | lk rt | ratio |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for (inp, n, chunk), solvers in sorted(buckets.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        lkh = solvers.get("lkh"); lk = solvers.get("lin_kernighan")
        if not lkh or not lk: continue
        m_lkh = statistics.mean(t for t, _ in lkh)
        m_lk = statistics.mean(t for t, _ in lk)
        rt_lkh = statistics.mean(r for _, r in lkh)
        rt_lk = statistics.mean(r for _, r in lk)
        gap = (m_lk - m_lkh) / m_lkh * 100.0 if m_lkh else 0.0
        ratio = rt_lk / rt_lkh if rt_lkh else 0.0
        out.write(f"| {inp} | {n} | {chunk} | {m_lkh:.0f} | {m_lk:.0f} | {gap:+.2f}% | {rt_lkh:.2f}s | {rt_lk:.2f}s | {ratio:.2f}× |\n")

    out.write("\n## Best chunk per input (lowest tour)\n\n")
    out.write("| input | n | lkh best chunk | lkh best tour | lk best chunk | lk best tour | gap |\n")
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
