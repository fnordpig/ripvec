#!/usr/bin/env bash
# Download a corpus from Project Gutenberg for benchmarking ripvec.
#
# Downloads ~50 books as plain text into tests/corpus/gutenberg/.
# The texts are split into individual .txt files per book (~20-50MB total).
# This gives a realistic large-text corpus with varied document sizes.
#
# Usage: ./scripts/fetch-gutenberg.sh [--small]
#   --small: download only 10 books (~4MB) for quick tests

set -euo pipefail

CORPUS_DIR="tests/corpus/gutenberg"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CORPUS_PATH="$PROJECT_DIR/$CORPUS_DIR"

# Book IDs from Project Gutenberg — mix of sizes and styles
# Fiction, non-fiction, poetry, plays, technical writing
BOOKS_FULL=(
    1342    # Pride and Prejudice
    11      # Alice's Adventures in Wonderland
    1661    # Sherlock Holmes
    84      # Frankenstein
    98      # A Tale of Two Cities
    2701    # Moby Dick
    1952    # The Yellow Wallpaper
    174     # The Picture of Dorian Gray
    345     # Dracula
    1232    # The Prince (Machiavelli)
    76      # Adventures of Huckleberry Finn
    2542    # The Iliad (Pope translation)
    100     # Complete Works of Shakespeare
    16328   # Beowulf
    514     # Little Women
    1260    # Jane Eyre
    768     # Wuthering Heights
    43      # The Strange Case of Dr. Jekyll and Mr. Hyde
    1080    # A Modest Proposal
    5200    # Metamorphosis (Kafka)
    4300    # Ulysses (Joyce)
    1184    # The Count of Monte Cristo
    3207    # Leviathan (Hobbes)
    8800    # The Divine Comedy
    2600    # War and Peace
    35      # The Time Machine
    36      # The War of the Worlds
    55      # The Wonderful Wizard of Oz
    120     # Treasure Island
    2500    # Siddhartha
    1400    # Great Expectations
    6130    # The Iliad (Butler)
    2591    # Grimm's Fairy Tales
    1399    # Anna Karenina
    244     # A Study in Scarlet
    2814    # Dubliners
    25344   # The Scarlet Letter
    158     # Emma
    161     # Sense and Sensibility
    30254   # The Romance of Lust
    45     # Anne of Green Gables
    74      # The Adventures of Tom Sawyer
    219     # Heart of Darkness
    408     # The Souls of Black Folk
    996     # Don Quixote
    1497    # Republic (Plato)
    46       # A Christmas Carol
    1727    # The Odyssey
    3600    # Thus Spake Zarathustra
    28054   # The Brothers Karamazov
)

BOOKS_SMALL=(
    1342    # Pride and Prejudice
    11      # Alice's Adventures in Wonderland
    84      # Frankenstein
    1952    # The Yellow Wallpaper
    43      # Jekyll and Hyde
    5200    # Metamorphosis
    35      # The Time Machine
    46      # A Christmas Carol
    219     # Heart of Darkness
    1080    # A Modest Proposal
)

if [[ "${1:-}" == "--small" ]]; then
    BOOKS=("${BOOKS_SMALL[@]}")
    echo "Downloading small corpus (10 books)..."
else
    BOOKS=("${BOOKS_FULL[@]}")
    echo "Downloading full corpus (${#BOOKS_FULL[@]} books)..."
fi

mkdir -p "$CORPUS_PATH"

downloaded=0
skipped=0
failed=0

for id in "${BOOKS[@]}"; do
    outfile="$CORPUS_PATH/pg${id}.txt"
    if [[ -f "$outfile" ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    url="https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
    if curl -fsSL --retry 2 --max-time 30 -o "$outfile" "$url" 2>/dev/null; then
        downloaded=$((downloaded + 1))
    else
        # Try alternate URL format
        url="https://www.gutenberg.org/files/${id}/${id}-0.txt"
        if curl -fsSL --retry 2 --max-time 30 -o "$outfile" "$url" 2>/dev/null; then
            downloaded=$((downloaded + 1))
        else
            failed=$((failed + 1))
            rm -f "$outfile"
        fi
    fi
done

# Summary
total_files=$(find "$CORPUS_PATH" -name '*.txt' | wc -l | tr -d ' ')
total_size=$(du -sh "$CORPUS_PATH" 2>/dev/null | cut -f1)

echo ""
echo "Corpus: $CORPUS_PATH"
echo "  Downloaded: $downloaded new, $skipped cached, $failed failed"
echo "  Total: $total_files files, $total_size"

# Ensure .gitignore excludes corpus
GITIGNORE="$PROJECT_DIR/.gitignore"
if ! rg -q 'tests/corpus' "$GITIGNORE" 2>/dev/null; then
    echo "tests/corpus/" >> "$GITIGNORE"
    echo "  Added tests/corpus/ to .gitignore"
fi
