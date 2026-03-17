import json
import os
import sys
import time
import click
import numpy as np
from pathlib import Path

from osmium.io.decode import decode, decode_streaming
from osmium.io.encode import encode, encode_pcm_stdout
from osmium.tsm.stream import process_streaming


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-s", "--speed", required=True, type=float, help="Target speed factor (e.g., 2.0, 3.5)")
@click.option("-o", "--output", "output_file", type=click.Path(), help="Output file path")
@click.option("--stream", is_flag=True, help="Stream raw f32le PCM to stdout")
@click.option("--engine", default="phase_vocoder", type=click.Choice(["phase_vocoder", "psola", "hybrid", "crepe_hybrid", "crepe_pv"]), help="TSM engine")
@click.option("--resolution", default="20ms", help="Importance map resolution (e.g., 10ms, 20ms, 80ms)")
@click.option("--window", "window_size", default=2048, type=int, help="STFT window size")
@click.option("--no-model", is_flag=True, help="Skip neural analysis, use uniform-rate TSM")
@click.option("--hpss", is_flag=True, help="Enable HPSS (experimental, phase_vocoder engine only)")
@click.option("--phoneme", is_flag=True, help="Enable phoneme alignment for importance (requires torch)")
@click.option("--parallel", "parallel_chunks", type=float, default=0, help="Chunk duration in seconds for parallel processing (0=auto)")
@click.option("--workers", type=int, default=0, help="Number of parallel workers (0=auto)")
@click.option("--analyze-only", is_flag=True, help="Output importance map as JSON")
def main(input_file, speed, output_file, stream, engine, resolution, window_size, no_model, hpss, phoneme, parallel_chunks, workers, analyze_only):
    """Osmium — high-quality speech acceleration."""
    if not stream and not output_file and not analyze_only:
        raise click.UsageError("Either --output, --stream, or --analyze-only is required")

    if speed <= 0:
        raise click.UsageError("Speed must be positive")

    resolution_s = _parse_resolution(resolution)

    click.echo(f"osmium: {Path(input_file).name} @ {speed}x (engine={engine})", err=True)

    if stream:
        _stream_mode(input_file, speed, window_size)
    else:
        _batch_mode(input_file, speed, engine, window_size, output_file, no_model, hpss, phoneme, resolution_s, analyze_only, parallel_chunks, workers)


def _batch_mode(input_file, speed, engine, window_size, output_file, no_model, hpss, phoneme, resolution_s, analyze_only, parallel_chunks, workers):
    t0 = time.time()
    click.echo("  decoding...", err=True)
    audio = decode(input_file)
    decode_time = time.time() - t0
    duration = len(audio.samples) / audio.sample_rate
    click.echo(f"  decoded {duration:.1f}s ({len(audio.samples)} samples) in {decode_time:.1f}s", err=True)

    use_parallel = parallel_chunks > 0 or duration > 600
    chunk_dur = parallel_chunks if parallel_chunks > 0 else 3600.0
    max_workers = workers if workers > 0 else min(os.cpu_count() or 1, max(1, int(duration / chunk_dur)))

    rate_curve = None
    rate_times = None

    if no_model:
        t1 = time.time()
    else:
        try:
            from osmium.analyzer.mimi import encode as mimi_encode
            from osmium.analyzer.importance import compute_importance, resample_importance
        except ImportError:
            raise click.ClickException(
                "Neural analysis requires: uv pip install -e '.[neural]'\n"
                "Or use --no-model for uniform-rate mode."
            )

        t1 = time.time()
        try:
            from osmium.analyzer.mimi_mlx import encode_mlx
            click.echo("  analyzing (mimi via mlx)...", err=True)
            codes = encode_mlx(audio.samples, audio.sample_rate)
        except (ImportError, Exception):
            click.echo("  analyzing (mimi via rustymimi)...", err=True)
            codes = mimi_encode(audio.samples, audio.sample_rate)
        analyze_time = time.time() - t1
        click.echo(f"  analyzed in {analyze_time:.1f}s ({duration/analyze_time:.1f}x realtime)", err=True)

        if phoneme:
            click.echo("  analyzing (phoneme alignment)...", err=True)
            t_ph = time.time()

        imp = compute_importance(codes, audio.samples, audio.sample_rate, use_phoneme_alignment=phoneme)

        if phoneme:
            click.echo(f"  phoneme analysis in {time.time() - t_ph:.1f}s", err=True)
        imp = resample_importance(imp, resolution_s)

        click.echo(f"  importance: mean={imp.scores.mean():.2f}, low(<0.2)={100*(imp.scores < 0.2).mean():.0f}%, high(>0.5)={100*(imp.scores > 0.5).mean():.0f}%", err=True)

        if analyze_only:
            result = {
                "duration": duration,
                "frame_rate": imp.frame_rate,
                "resolution_s": resolution_s,
                "n_frames": len(imp.scores),
                "scores": imp.scores.tolist(),
                "times": imp.times.tolist(),
            }
            out = output_file or "-"
            if out == "-":
                json.dump(result, sys.stdout, indent=2)
            else:
                with open(out, "w") as f:
                    json.dump(result, f, indent=2)
                click.echo(f"  wrote importance map to {out}", err=True)
            return

        from osmium.tsm.rate_schedule import importance_to_rate_schedule
        rate_curve, rate_times = importance_to_rate_schedule(
            imp.scores, imp.times, target_speed=speed,
        )
        click.echo(f"  rate range: {rate_curve.min():.1f}x – {rate_curve.max():.1f}x", err=True)

    if use_parallel and max_workers > 1:
        click.echo(f"  parallel {engine} @ {speed}x ({max_workers} workers, {chunk_dur:.0f}s chunks)...", err=True)
        from osmium.parallel import process_parallel

        def on_progress(done, total):
            click.echo(f"    chunk {done}/{total} complete", err=True)

        output_samples = process_parallel(
            audio.samples, speed=speed, engine=engine,
            window_size=window_size, sample_rate=audio.sample_rate,
            chunk_duration=chunk_dur, max_workers=max_workers,
            rate_curve=rate_curve, rate_times=rate_times,
            on_progress=on_progress,
        )
    else:
        output_samples = _stretch_single(
            audio.samples, speed, engine, window_size, audio.sample_rate,
            hpss, rate_curve, rate_times,
        )

    input_rms = np.sqrt(np.mean(audio.samples ** 2))
    output_samples = _soft_clip_and_normalize(output_samples, input_rms)

    stretch_time = time.time() - t1
    out_duration = len(output_samples) / audio.sample_rate
    click.echo(f"  output: {out_duration:.1f}s in {stretch_time:.1f}s", err=True)

    if output_file:
        t2 = time.time()
        click.echo(f"  encoding → {output_file}...", err=True)
        encode(output_samples, audio.sample_rate, output_file)
        encode_time = time.time() - t2
        click.echo(f"  done in {encode_time:.1f}s (total: {time.time() - t0:.1f}s)", err=True)


def _stretch_single(samples, speed, engine, window_size, sample_rate, hpss, rate_curve, rate_times):
    if engine == "psola":
        from osmium.tsm.psola_engine import psola_stretch, variable_rate_psola
        if rate_curve is not None:
            return variable_rate_psola(samples, rate_curve, rate_times, sample_rate)
        return psola_stretch(samples, speed, sample_rate)

    if engine == "crepe_hybrid":
        from osmium.tsm.crepe_hybrid import crepe_hybrid_stretch, crepe_hybrid_variable_rate
        if rate_curve is not None:
            return crepe_hybrid_variable_rate(samples, rate_curve, rate_times, window_size, sample_rate)
        return crepe_hybrid_stretch(samples, speed, window_size, sample_rate)

    if engine == "crepe_pv":
        from osmium.tsm.crepe_enhanced_pv import crepe_enhanced_stretch, crepe_enhanced_variable_rate
        if rate_curve is not None:
            return crepe_enhanced_variable_rate(samples, rate_curve, rate_times, window_size, sample_rate)
        return crepe_enhanced_stretch(samples, speed, window_size, sample_rate)

    if engine == "hybrid":
        from osmium.tsm.voiced_split import hybrid_voiced_stretch, hybrid_voiced_variable_rate
        if rate_curve is not None:
            return hybrid_voiced_variable_rate(samples, rate_curve, rate_times, window_size, sample_rate)
        return hybrid_voiced_stretch(samples, speed, window_size, sample_rate)

    if hpss:
        from osmium.tsm.hybrid import hybrid_stretch, hybrid_variable_rate_stretch
        if rate_curve is not None:
            return hybrid_variable_rate_stretch(samples, rate_curve, rate_times, window_size, sample_rate)
        return hybrid_stretch(samples, speed, window_size, sample_rate)

    from osmium.tsm.phase_vocoder import phase_vocoder_stretch, variable_rate_phase_vocoder
    if rate_curve is not None:
        return variable_rate_phase_vocoder(samples, rate_curve, rate_times, window_size, sample_rate)
    return phase_vocoder_stretch(samples, speed, window_size, sample_rate)


def _stream_mode(input_file, speed, window_size):
    click.echo("  streaming mode (f32le mono 24kHz)...", err=True)
    chunks = decode_streaming(input_file, chunk_seconds=5.0)
    for output_chunk in process_streaming(chunks, speed=speed, window_size=window_size):
        sys.stdout.buffer.write(encode_pcm_stdout(output_chunk))
    click.echo("  stream complete", err=True)


def _soft_clip_and_normalize(samples: np.ndarray, target_rms: float) -> np.ndarray:
    peak = np.max(np.abs(samples))
    if peak > 0.8:
        threshold = 0.8
        above = np.abs(samples) > threshold
        sign = np.sign(samples)
        excess = np.abs(samples) - threshold
        samples = np.where(
            above,
            sign * (threshold + np.tanh(excess / (peak - threshold + 1e-10)) * (1.0 - threshold)),
            samples,
        )

    output_rms = np.sqrt(np.mean(samples ** 2))
    if output_rms > 0 and target_rms > 0:
        samples = samples * (target_rms / output_rms)

    peak = np.max(np.abs(samples))
    if peak > 0.99:
        samples = samples * (0.99 / peak)

    return samples.astype(np.float32)


def _parse_resolution(res: str) -> float:
    res = res.strip().lower()
    if res.endswith("ms"):
        return float(res[:-2]) / 1000.0
    elif res.endswith("s"):
        return float(res[:-1])
    else:
        return float(res) / 1000.0
