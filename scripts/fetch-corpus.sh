#!/usr/bin/env bash
# Download code repositories for benchmarking ripvec.
#
# Shallow-clones popular open source projects into tests/corpus/.
# These provide a realistic mix of languages and file sizes.
#
# Usage: ./scripts/fetch-corpus.sh [--small]
#   --small: download only 3 repos (~15MB)
#   default: download 8 repos (~80MB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORPUS_DIR="$SCRIPT_DIR/../tests/corpus/code"

# Format: "org/repo"
REPOS_FULL=(
    "BurntSushi/ripgrep"        # Rust, ~15MB, 100+ files
    "pallets/flask"             # Python, ~5MB
    "expressjs/express"         # JS, ~3MB
    "golang/go"                 # Go, ~300MB (but depth=1)
    "spring-projects/spring-boot"  # Java, ~50MB
    "redis/redis"               # C, ~15MB
    "facebook/react"            # JS/TS, ~30MB
    "tokio-rs/tokio"            # Rust, ~20MB
    "torvalds/linux"		# Linux,  ~89468 source files, 2.0G on disk
)

REPOS_SMALL=(
    "BurntSushi/ripgrep"        # Rust
    "pallets/flask"             # Python
    "expressjs/express"         # JS
)

if [[ "${1:-}" == "--small" ]]; then
    REPOS=("${REPOS_SMALL[@]}")
    echo "Downloading small corpus (${#REPOS_SMALL[@]} repos)..."
else
    REPOS=("${REPOS_FULL[@]}")
    echo "Downloading full corpus (${#REPOS_FULL[@]} repos)..."
fi

mkdir -p "$CORPUS_DIR"

for repo in "${REPOS[@]}"; do
    name="${repo##*/}"
    target="$CORPUS_DIR/$name"
    if [[ -d "$target" ]]; then
        echo "  skip $name (exists)"
        continue
    fi
    echo "  clone $repo..."
    git clone --depth 1 --quiet "https://github.com/$repo.git" "$target" 2>/dev/null || {
        echo "  FAILED: $repo"
        continue
    }
    # Remove .git to save space
    rm -rf "$target/.git"
done

# Summary
total_files=$(find "$CORPUS_DIR" -type f \( -name '*.rs' -o -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.go' -o -name '*.java' -o -name '*.c' -o -name '*.h' -o -name '*.cpp' \) | wc -l | tr -d ' ')
total_size=$(du -sh "$CORPUS_DIR" 2>/dev/null | cut -f1)

echo ""
echo "Corpus: $CORPUS_DIR"
echo "  Total: ~$total_files source files, $total_size on disk"

# Ensure .gitignore excludes corpus
GITIGNORE="$SCRIPT_DIR/../.gitignore"
if ! rg -q 'tests/corpus' "$GITIGNORE" 2>/dev/null; then
    echo "tests/corpus/" >> "$GITIGNORE"
    echo "  Added tests/corpus/ to .gitignore"
fi
