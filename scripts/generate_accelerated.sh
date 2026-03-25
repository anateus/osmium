#!/usr/bin/env bash
set -euo pipefail

CLIPS_DIR="$(cd "$(dirname "$0")/../samples/clips" && pwd)"
ACCEL_DIR="$CLIPS_DIR/accelerated"

SPEEDS=(2 2.3 2.6 2.8 3 3.2 3.8)
MODES=(uniform no-mimi neural)

for speed in "${SPEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
        outdir="$ACCEL_DIR/${speed}x/$mode"
        mkdir -p "$outdir"

        for wav in "$CLIPS_DIR"/*.wav; do
            base="$(basename "${wav%.wav}")"
            outfile="$outdir/${base}.mp3"

            if [[ -f "$outfile" ]]; then
                echo "SKIP $outfile (exists)"
                continue
            fi

            flags="-s $speed -o $outfile"
            case "$mode" in
                uniform) flags="$flags --uniform" ;;
                no-mimi) ;; # default mel-based mode
                neural)  flags="$flags --mimi" ;;
            esac

            echo ">> ${speed}x $mode: $(basename "$wav")"
            uv run osmium "$wav" $flags
        done
    done
done

echo "Done. Output in $ACCEL_DIR"
