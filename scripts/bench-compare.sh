#!/usr/bin/env bash
# Compare ripvec performance across device backends.
#
# Runs the search pipeline against the Gutenberg corpus with profiling
# and captures timing, throughput, and peak RSS for each configuration.
#
# Prerequisites:
#   ./scripts/fetch-gutenberg.sh   # download corpus first
#   cargo build --release           # CPU+Accelerate
#   cargo build --release --features metal  # if testing Metal
#
# Usage: ./scripts/bench-compare.sh [corpus_dir] [query]

set -euo pipefail

CORPUS="${1:-tests/corpus/gutenberg}"
QUERY="${2:-connection pool database}"
BINARY="./target/release/ripvec"
TOP_K=5
RUNS=3  # number of runs per config (take best)

if [[ ! -d "$CORPUS" ]]; then
    echo "Corpus not found at $CORPUS"
    echo "Run: ./scripts/fetch-gutenberg.sh"
    exit 1
fi

if [[ ! -x "$BINARY" ]]; then
    echo "Binary not found. Run: cargo build --release"
    exit 1
fi

file_count=$(find "$CORPUS" -name '*.txt' | wc -l | tr -d ' ')
corpus_size=$(du -sh "$CORPUS" | cut -f1)
echo "=== ripvec benchmark ==="
echo "Corpus: $CORPUS ($file_count files, $corpus_size)"
echo "Query:  \"$QUERY\""
echo "Runs:   $RUNS per config (best-of)"
echo ""

# Run a single benchmark, capture profile output and peak RSS
# Args: $1=label $2..=extra args to ripvec
run_bench() {
    local label="$1"
    shift
    local best_time=999999
    local best_rate=0
    local best_output=""

    for i in $(seq 1 $RUNS); do
        # Use /usr/bin/time for peak RSS on macOS
        local tmpout
        tmpout=$(mktemp)
        local tmptime
        tmptime=$(mktemp)

        /usr/bin/time -l "$BINARY" "$QUERY" "$CORPUS" \
            --profile -n "$TOP_K" --format plain \
            "$@" \
            >"$tmpout" 2>"$tmptime" || true

        # Extract metrics from profile output
        local total_line
        total_line=$(rg 'total:' "$tmpout" 2>/dev/null || echo "")
        local embed_line
        embed_line=$(rg 'embed:.*done' "$tmpout" 2>/dev/null || echo "")
        local rss_kb
        rss_kb=$(rg 'maximum resident' "$tmptime" 2>/dev/null | rg -o '[0-9]+' | head -1 || echo "0")

        if [[ -n "$total_line" ]]; then
            local time_s
            time_s=$(echo "$total_line" | rg -o '[0-9]+\.[0-9]+s' | head -1 | tr -d 's')
            local rate
            rate=$(echo "$embed_line" | rg -o '[0-9]+\.[0-9]+/s\)' | tail -1 | tr -d '/s)' || echo "0")
            local rss_mb=$(( rss_kb / 1024 / 1024 ))

            # Check if this is the best run
            if (( $(echo "$time_s < $best_time" | bc -l) )); then
                best_time="$time_s"
                best_rate="$rate"
                best_output="$tmpout"
            fi
        fi

        rm -f "$tmptime"
        [[ "$tmpout" != "$best_output" ]] && rm -f "$tmpout"
    done

    if [[ -n "$best_output" && -f "$best_output" ]]; then
        local embed_done
        embed_done=$(rg 'embed:.*done' "$best_output" 2>/dev/null || echo "N/A")
        local chunk_line
        chunk_line=$(rg 'chunk:' "$best_output" 2>/dev/null || echo "")

        printf "%-20s  %8ss  %8s/s  %s\n" "$label" "$best_time" "$best_rate" "$embed_done"
        rm -f "$best_output"
    else
        printf "%-20s  %8s  %8s  %s\n" "$label" "FAILED" "-" "-"
    fi
}

printf "%-20s  %9s  %9s  %s\n" "Config" "Time" "Rate" "Details"
printf "%-20s  %9s  %9s  %s\n" "------" "----" "----" "-------"

# CPU (always available)
run_bench "cpu" --device cpu

# Metal (if built with feature)
if "$BINARY" --help 2>&1 | rg -q 'metal'; then
    run_bench "metal" --device metal
fi

echo ""
echo "Done."
