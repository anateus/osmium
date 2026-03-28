#!/usr/bin/env python
import argparse
import subprocess
import tempfile
from pathlib import Path


def run_osmium(input_file, speed, output_file, extra_args=None):
    cmd = ["uv", "run", "osmium", input_file, "-s", str(speed), "-o", output_file]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def run_wer(ref_file, test_file, whisper_model="small"):
    cmd = [
        "uv", "run", "python", "scripts/eval_wer.py",
        "--ref", ref_file, "--test", test_file,
        "--whisper-model", whisper_model,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in result.stdout.strip().split("\n"):
        if "WER" in line and "CER" in line:
            parts = line.split()
            wer = float(parts[parts.index("WER") + 1].rstrip("%,"))
            cer = float(parts[parts.index("CER") + 1].rstrip("%,"))
            return wer, cer
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Compare phoneme-aware configurations")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--speeds", default="2.0,3.0,4.0,5.0", help="Comma-separated speeds")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--output-dir", default=None, help="Dir for temp outputs")
    args = parser.parse_args()

    speeds = [float(s) for s in args.speeds.split(",")]
    configs = [
        ("mel-only", ["--no-phoneme"]),
        ("phoneme-class", []),
        ("phoneme-align", ["--phoneme-align"]),
    ]

    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(args.output_dir) if args.output_dir else Path(tmpdir)
        outdir.mkdir(parents=True, exist_ok=True)

        for speed in speeds:
            results[speed] = {}
            for name, extra in configs:
                outfile = str(outdir / f"{name}_{speed}x.wav")
                try:
                    run_osmium(args.input, speed, outfile, extra)
                    wer, cer = run_wer(args.input, outfile, args.whisper_model)
                    results[speed][name] = {"wer": wer, "cer": cer}
                except Exception as e:
                    results[speed][name] = {"wer": None, "cer": None, "error": str(e)}

    print("\n## Phoneme-Aware Rate Scheduling Comparison\n")
    print(f"Input: `{args.input}`\n")
    print("| Speed | Config | WER (%) | CER (%) |")
    print("|---|---|---|---|")
    for speed in speeds:
        for name in ["mel-only", "phoneme-class", "phoneme-align"]:
            r = results[speed].get(name, {})
            wer = f"{r['wer']:.1f}" if r.get("wer") is not None else "N/A"
            cer = f"{r['cer']:.1f}" if r.get("cer") is not None else "N/A"
            err = f" ({r['error'][:40]})" if r.get("error") else ""
            print(f"| {speed}x | {name} | {wer}{err} | {cer} |")


if __name__ == "__main__":
    main()
