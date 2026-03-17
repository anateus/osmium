#!/usr/bin/env python3
"""ABX perceptual testing for osmium output quality.

Plays sample A and B (in random order), then plays X (which is one of them).
Asks listener to identify whether X is A or B. Tracks accuracy across trials.

Usage:
    python scripts/abx_test.py /tmp/osmium_phoneme_nohpss/clip_01_opening_3x.wav /tmp/osmium_comparison/v5c/clip_01_opening_neural.wav
    python scripts/abx_test.py --dir /tmp/osmium_phoneme_nohpss --pattern "*_3x.wav" --dir-b /tmp/osmium_comparison/v5c --pattern-b "*_neural.wav"
    python scripts/abx_test.py --generate-pairs 3.0 --phoneme-vs-mimi
"""
import argparse
import glob
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def play_audio(path: str, label: str = ""):
    if label:
        print(f"  Playing {label}...", flush=True)
    else:
        print(f"  Playing...", flush=True)
    subprocess.run(
        ["ffplay", "-v", "quiet", "-nodisp", "-autoexit", str(path)],
        check=True,
    )
    time.sleep(0.3)


def run_trial(path_a: str, path_b: str, trial_num: int, total: int) -> bool | None:
    print(f"\n{'='*50}")
    print(f"Trial {trial_num}/{total}")
    print(f"  A: {Path(path_a).name}")
    print(f"  B: {Path(path_b).name}")
    print(f"{'='*50}")

    x_is_a = random.random() < 0.5
    x_path = path_a if x_is_a else path_b

    order = random.random() < 0.5
    if order:
        first, first_label = path_a, "A"
        second, second_label = path_b, "B"
    else:
        first, first_label = path_b, "B"
        second, second_label = path_a, "A"

    play_audio(first, first_label)
    play_audio(second, second_label)

    print(f"\n  Now playing X...")
    play_audio(x_path, "X")

    while True:
        resp = input("\n  Is X the same as A or B? [a/b/r/q] (r=replay, q=quit): ").strip().lower()
        if resp == "q":
            return None
        if resp == "r":
            print()
            play_audio(first, first_label)
            play_audio(second, second_label)
            print()
            play_audio(x_path, "X")
            continue
        if resp in ("a", "b"):
            correct = (resp == "a") == x_is_a
            answer = "A" if x_is_a else "B"
            if correct:
                print(f"  Correct! X was {answer}.")
            else:
                print(f"  Wrong — X was {answer}.")
            return correct

    return None


def find_matching_pairs(dir_a, pattern_a, dir_b, pattern_b):
    files_a = sorted(glob.glob(os.path.join(dir_a, pattern_a)))
    files_b = sorted(glob.glob(os.path.join(dir_b, pattern_b)))

    pairs = []
    for fa in files_a:
        name_a = Path(fa).stem
        for fb in files_b:
            name_b = Path(fb).stem
            clip_a = name_a.split("_")[0:3]
            clip_b = name_b.split("_")[0:3]
            if clip_a == clip_b:
                pairs.append((fa, fb))
                break

    return pairs


def generate_pairs(speed: float, mode: str):
    clips = sorted(glob.glob("samples/clips/clip_*.wav"))
    if not clips:
        print("No clips found in samples/clips/")
        sys.exit(1)

    pairs = []
    out_dir = Path("/tmp/osmium_abx")
    out_dir.mkdir(exist_ok=True)

    for clip in clips:
        name = Path(clip).stem
        if mode == "phoneme-vs-mimi":
            path_a = out_dir / f"{name}_{speed}x_mimi.wav"
            path_b = out_dir / f"{name}_{speed}x_phoneme.wav"
            if not path_a.exists():
                print(f"Generating {path_a.name}...")
                subprocess.run([
                    sys.executable, "-m", "osmium", clip,
                    "-s", str(speed), "-o", str(path_a),
                ], check=True, capture_output=True)
            if not path_b.exists():
                print(f"Generating {path_b.name}...")
                subprocess.run([
                    sys.executable, "-m", "osmium", clip,
                    "-s", str(speed), "--phoneme", "-o", str(path_b),
                ], check=True, capture_output=True)
            pairs.append((str(path_a), str(path_b)))
        elif mode == "neural-vs-uniform":
            path_a = out_dir / f"{name}_{speed}x_neural.wav"
            path_b = out_dir / f"{name}_{speed}x_uniform.wav"
            if not path_a.exists():
                print(f"Generating {path_a.name}...")
                subprocess.run([
                    sys.executable, "-m", "osmium", clip,
                    "-s", str(speed), "-o", str(path_a),
                ], check=True, capture_output=True)
            if not path_b.exists():
                print(f"Generating {path_b.name}...")
                subprocess.run([
                    sys.executable, "-m", "osmium", clip,
                    "-s", str(speed), "--no-model", "-o", str(path_b),
                ], check=True, capture_output=True)
            pairs.append((str(path_a), str(path_b)))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="ABX perceptual testing for osmium")
    parser.add_argument("file_a", nargs="?", help="First audio file")
    parser.add_argument("file_b", nargs="?", help="Second audio file")
    parser.add_argument("--dir", help="Directory for A files")
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern for A files")
    parser.add_argument("--dir-b", help="Directory for B files")
    parser.add_argument("--pattern-b", default="*.wav", help="Glob pattern for B files")
    parser.add_argument("--generate-pairs", type=float, metavar="SPEED", help="Generate test pairs at this speed")
    parser.add_argument("--phoneme-vs-mimi", action="store_true")
    parser.add_argument("--neural-vs-uniform", action="store_true")
    parser.add_argument("--trials", type=int, default=0, help="Number of trials (0=all pairs)")

    args = parser.parse_args()

    if args.generate_pairs:
        mode = "phoneme-vs-mimi" if args.phoneme_vs_mimi else "neural-vs-uniform"
        pairs = generate_pairs(args.generate_pairs, mode)
    elif args.file_a and args.file_b:
        pairs = [(args.file_a, args.file_b)]
    elif args.dir and args.dir_b:
        pairs = find_matching_pairs(args.dir, args.pattern, args.dir_b, args.pattern_b)
    else:
        parser.error("Provide two files, --dir/--dir-b, or --generate-pairs")

    if not pairs:
        print("No pairs found!")
        sys.exit(1)

    if args.trials > 0:
        random.shuffle(pairs)
        pairs = pairs[:args.trials]

    print(f"\nABX Test — {len(pairs)} trial(s)")
    print("Listen to A and B, then identify X.\n")

    correct = 0
    total = 0

    for i, (fa, fb) in enumerate(pairs, 1):
        result = run_trial(fa, fb, i, len(pairs))
        if result is None:
            break
        total += 1
        if result:
            correct += 1

    print(f"\n{'='*50}")
    print(f"Results: {correct}/{total} correct ({100*correct/total:.0f}%)" if total > 0 else "No trials completed")
    if total > 0:
        chance = 50
        print(f"Chance level: {chance}%")
        if correct / total > 0.7:
            print("You can reliably distinguish these — the difference is perceptible.")
        elif correct / total > 0.55:
            print("Marginal — you can sometimes tell, but it's subtle.")
        else:
            print("At chance — these sound equivalent to you.")


if __name__ == "__main__":
    main()
