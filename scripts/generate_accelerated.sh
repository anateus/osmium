#!/usr/bin/env bash
set -euo pipefail

CLIPS_DIR="$(cd "$(dirname "$0")/../samples/clips" && pwd)"
ACCEL_DIR="$CLIPS_DIR/accelerated"

SPEEDS=(2 2.3 2.6 2.8 3 3.2 3.8)
MODES=(uniform no-mimi neural gate-denoise deep-denoise)

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
                uniform)       flags="$flags --uniform" ;;
                no-mimi)       ;;
                neural)        flags="$flags --mimi" ;;
                gate-denoise)  flags="$flags --denoise gate" ;;
                deep-denoise)  flags="$flags --denoise deep" ;;
            esac

            echo ">> ${speed}x $mode: $(basename "$wav")"
            if ! uv run osmium "$wav" $flags; then
                echo "SKIP $mode (dependency not installed or error)"
                rm -f "$outfile"
                break
            fi
        done
    done
done

echo "Done. Output in $ACCEL_DIR"
