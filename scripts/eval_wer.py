#!/usr/bin/env python
"""Whisper WER evaluation harness for osmium output quality.

Usage:
    # Compare default vs uniform at 3x:
    uv run scripts/eval_wer.py samples/clips/clip_01_opening.wav -s 3.0

    # Test specific parameter sweep:
    uv run scripts/eval_wer.py samples/clips/*.wav -s 3.0 --sweep smoothing 0.0 0.3 0.5 0.7 1.0 1.5

    # Compare importance resolutions:
    uv run scripts/eval_wer.py samples/clips/*.wav -s 3.0 --sweep resolution 5 10 20 40 80

    # Compare importance weights:
    uv run scripts/eval_wer.py samples/clips/*.wav -s 3.0 --sweep hf_boost 1.0 1.5 2.0 3.0

    # Just eval an existing file against a reference:
    uv run scripts/eval_wer.py --ref original.wav --test accelerated.wav
"""
import argparse
import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    name: str
    wer: float
    cer: float
    output_duration: float
    process_time: float
    file: str = ""


@dataclass
class EvalReport:
    reference_text: str
    clip: str
    speed: float
    results: list[EvalResult] = field(default_factory=list)

    def print(self):
        print(f"\n{'='*70}")
        print(f"  {self.clip} @ {self.speed}x")
        print(f"{'='*70}")
        print(f"  {'Config':<30s} {'WER':>6s} {'CER':>6s} {'Time':>6s}")
        print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6}")
        best_wer = min(r.wer for r in self.results)
        for r in self.results:
            marker = " *" if r.wer == best_wer and len(self.results) > 1 else ""
            print(f"  {r.name:<30s} {r.wer:>5.1%} {r.cer:>5.1%} {r.process_time:>5.1f}s{marker}")


_whisper_model = None


def _get_whisper(model_size="small"):
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def transcribe(path: str, model_size="small") -> str:
    model = _get_whisper(model_size)
    result = model.transcribe(path, language="en")
    return result["text"]


def compute_wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    import jiwer
    return jiwer.wer(reference, hypothesis), jiwer.cer(reference, hypothesis)


def eval_file(ref_path: str, test_path: str, model_size="small") -> tuple[float, float]:
    ref_text = transcribe(ref_path, model_size)
    test_text = transcribe(test_path, model_size)
    return compute_wer_cer(ref_text, test_text)


def eval_osmium(
    input_path: str,
    speed: float,
    configs: dict[str, dict] | None = None,
    model_size: str = "small",
) -> EvalReport:
    from osmium.tsm.vocos_mlx import extract_mel, vocos_mlx_variable_rate, vocos_mlx_stretch, _load_model
    from osmium.analyzer.mel_importance import compute_mel_importance
    from osmium.analyzer.importance import resample_importance
    from osmium.tsm.rate_schedule import importance_to_rate_schedule

    _load_model()

    samples, sr = sf.read(input_path, dtype="float32")
    clip_name = Path(input_path).stem
    ref_text = transcribe(input_path, model_size)
    report = EvalReport(reference_text=ref_text, clip=clip_name, speed=speed)

    if configs is None:
        configs = {
            "default": {},
            "uniform": {"uniform": True},
        }

    for name, cfg in configs.items():
        t0 = time.time()

        if cfg.get("uniform"):
            out = vocos_mlx_stretch(samples, speed, sr, cfg.get("smoothing", 0.7))
        else:
            mel = extract_mel(samples, sr)
            duration = len(samples) / sr
            imp = compute_mel_importance(
                mel, duration,
                weight_flux=cfg.get("flux_w", 0.65),
                weight_energy=cfg.get("energy_w", 0.35),
                hf_boost=cfg.get("hf_boost", 2.5),
            )
            imp = resample_importance(imp, cfg.get("resolution", 0.02))
            rate_curve, rate_times = importance_to_rate_schedule(
                imp.scores, imp.times, target_speed=speed,
                gamma=cfg.get("rate_gamma", 1.5),
            )
            out = vocos_mlx_variable_rate(
                samples, rate_curve, rate_times, sr,
                smoothing_sigma=cfg.get("smoothing", 0.7),
            )

        elapsed = time.time() - t0
        tmp = f"/tmp/osmium_eval_{clip_name}_{name}.wav"
        sf.write(tmp, out, sr)

        test_text = transcribe(tmp, model_size)
        wer, cer = compute_wer_cer(ref_text, test_text)

        report.results.append(EvalResult(
            name=name, wer=wer, cer=cer,
            output_duration=len(out) / sr,
            process_time=elapsed, file=tmp,
        ))

    return report


def build_sweep_configs(param: str, values: list[str]) -> dict[str, dict]:
    configs = {}
    for v in values:
        vf = float(v)
        label = f"{param}={v}"
        if param == "smoothing":
            configs[label] = {"smoothing": vf}
        elif param == "resolution":
            configs[label] = {"resolution": vf / 1000.0}
        elif param == "hf_boost":
            configs[label] = {"hf_boost": vf}
        elif param == "flux_w":
            configs[label] = {"flux_w": vf, "energy_w": 1.0 - vf}
        elif param == "rate_gamma":
            configs[label] = {"rate_gamma": vf}
        elif param == "speed":
            configs[label] = {"_speed_override": vf}
        else:
            print(f"Unknown sweep param: {param}")
            sys.exit(1)
    return configs


def main():
    parser = argparse.ArgumentParser(description="Osmium WER evaluation harness")
    parser.add_argument("inputs", nargs="*", help="Input audio files")
    parser.add_argument("-s", "--speed", type=float, default=3.0)
    parser.add_argument("--sweep", nargs="+", metavar=("PARAM", "VALUES"), help="Sweep a parameter: --sweep smoothing 0.0 0.5 1.0")
    parser.add_argument("--ref", help="Reference file for direct comparison")
    parser.add_argument("--test", help="Test file for direct comparison")
    parser.add_argument("--whisper-model", default="small", help="Whisper model size")
    args = parser.parse_args()

    if args.ref and args.test:
        wer, cer = eval_file(args.ref, args.test, args.whisper_model)
        print(f"WER: {wer:.1%}  CER: {cer:.1%}")
        return

    if not args.inputs:
        parser.error("Provide input files or use --ref/--test")

    configs = None
    if args.sweep:
        param = args.sweep[0]
        values = args.sweep[1:]
        configs = build_sweep_configs(param, values)

    for input_path in args.inputs:
        report = eval_osmium(input_path, args.speed, configs, args.whisper_model)
        report.print()


if __name__ == "__main__":
    main()
