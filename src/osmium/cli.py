import json
import sys
import time
import click
import numpy as np
from pathlib import Path

from osmium.io.decode import decode, decode_streaming, probe_duration
from osmium.io.encode import encode, encode_pcm_stdout


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-s", "--speed", required=True, type=float, help="Target speed factor (e.g., 2.0, 3.5)")
@click.option("-o", "--output", "output_file", type=click.Path(), help="Output file path")
@click.option("--stream", is_flag=True, help="Stream raw f32le PCM to stdout")
@click.option("--resolution", default="20ms", help="Importance map resolution (e.g., 10ms, 20ms, 80ms)")
@click.option("--uniform", is_flag=True, help="Skip importance analysis, use uniform-rate")
@click.option("--mimi", is_flag=True, help="Use Mimi neural codec for importance (slower, slightly better)")
@click.option("--smoothing", default=0.7, type=float, help="Mel temporal smoothing sigma (0=off)")
@click.option("--chunks", "chunk_duration", type=float, default=0, help="Chunk duration in seconds for long files (0=auto for >10min)")
@click.option("--analyze-only", is_flag=True, help="Output importance map as JSON")
def main(input_file, speed, output_file, stream, resolution, uniform, mimi, smoothing, chunk_duration, analyze_only):
    """Osmium — high-quality speech acceleration."""
    if not stream and not output_file and not analyze_only:
        raise click.UsageError("Either --output, --stream, or --analyze-only is required")

    if speed <= 0:
        raise click.UsageError("Speed must be positive")

    if stream:
        _stream_mode(input_file, speed)
    else:
        _batch_mode(input_file, speed, output_file, uniform, mimi, resolution, smoothing, analyze_only, chunk_duration)


def _batch_mode(input_file, speed, output_file, uniform, use_mimi, resolution, smoothing, analyze_only, chunk_duration):
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

    console = Console(stderr=True)
    filename = Path(input_file).name
    resolution_s = _parse_resolution(resolution)

    console.print(f"[bold]osmium[/bold] {filename} @ [cyan]{speed}x[/cyan]")

    t0 = time.time()

    probe_dur = probe_duration(input_file)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:

        decode_task = progress.add_task("Decoding", total=probe_dur, status="")

        def on_decode_progress(decoded_s):
            progress.update(decode_task, completed=decoded_s, status=f"{decoded_s:.0f}s")

        if probe_dur and probe_dur > 30:
            audio = decode(input_file, progress_callback=on_decode_progress)
        else:
            audio = decode(input_file)

        duration = len(audio.samples) / audio.sample_rate
        progress.update(decode_task, completed=duration, total=duration, status=f"{duration:.0f}s")
        progress.remove_task(decode_task)

        rate_curve = None
        rate_times = None

        if not uniform:
            from osmium.analyzer.importance import resample_importance
            from osmium.tsm.rate_schedule import importance_to_rate_schedule

            if use_mimi:
                imp = _analyze_mimi(audio, progress, console)
            else:
                imp = _analyze_mel(audio, progress)

            imp = resample_importance(imp, resolution_s)

            if analyze_only:
                _write_analysis(imp, resolution_s, duration, output_file, console)
                return

            rate_curve, rate_times = importance_to_rate_schedule(
                imp.scores, imp.times, target_speed=speed,
            )

        use_chunked = chunk_duration > 0 or duration > 600
        chunk_dur = chunk_duration if chunk_duration > 0 else 300.0

        if use_chunked:
            n_chunks = max(1, int(np.ceil(duration / chunk_dur)))
            stretch_task = progress.add_task("Stretching", total=n_chunks, status=f"0/{n_chunks} chunks")
            from osmium.parallel import process_chunked

            def on_chunk_progress(done, total):
                progress.update(stretch_task, completed=done, status=f"{done}/{total} chunks")

            output_samples = process_chunked(
                audio.samples, speed=speed, sample_rate=audio.sample_rate,
                chunk_duration=chunk_dur,
                rate_curve=rate_curve, rate_times=rate_times,
                smoothing=smoothing, on_progress=on_chunk_progress,
            )
        else:
            stretch_task = progress.add_task("Stretching", total=None, status="vocos")

            try:
                from osmium.tsm.vocos_mlx import vocos_mlx_stretch, vocos_mlx_variable_rate
                progress.update(stretch_task, status="vocos (mlx)")
                if rate_curve is not None:
                    output_samples = vocos_mlx_variable_rate(
                        audio.samples, rate_curve, rate_times,
                        sample_rate=audio.sample_rate, smoothing_sigma=smoothing,
                    )
                else:
                    output_samples = vocos_mlx_stretch(
                        audio.samples, speed,
                        sample_rate=audio.sample_rate, smoothing_sigma=smoothing,
                    )
            except (ImportError, Exception):
                from osmium.tsm.vocos_engine import vocos_stretch, vocos_variable_rate
                progress.update(stretch_task, status="vocos (cpu)")
                if rate_curve is not None:
                    output_samples = vocos_variable_rate(
                        audio.samples, rate_curve, rate_times,
                        sample_rate=audio.sample_rate, smoothing_sigma=smoothing,
                    )
                else:
                    output_samples = vocos_stretch(
                        audio.samples, speed,
                        sample_rate=audio.sample_rate, smoothing_sigma=smoothing,
                    )

        progress.remove_task(stretch_task)

        input_rms = np.sqrt(np.mean(audio.samples ** 2))
        output_samples = _soft_clip_and_normalize(output_samples, input_rms)

        if output_file:
            encode_task = progress.add_task("Encoding", total=None, status=Path(output_file).suffix)
            encode(output_samples, audio.sample_rate, output_file)
            progress.remove_task(encode_task)

    out_duration = len(output_samples) / audio.sample_rate
    total_time = time.time() - t0

    console.print(
        f"  [green]done[/green] {duration:.0f}s → {out_duration:.1f}s "
        f"in {total_time:.1f}s ({duration/total_time:.0f}x realtime)"
    )
    if output_file:
        console.print(f"  [dim]→ {output_file}[/dim]")


def _analyze_mel(audio, progress):
    from osmium.tsm.vocos_mlx import extract_mel
    from osmium.analyzer.mel_importance import compute_mel_importance

    task = progress.add_task("Analyzing", total=None, status="mel importance")
    duration = len(audio.samples) / audio.sample_rate
    mel = extract_mel(audio.samples, audio.sample_rate)
    imp = compute_mel_importance(mel, duration)
    progress.remove_task(task)
    return imp


def _analyze_mimi(audio, progress, console):
    from osmium.analyzer.importance import compute_importance

    task = progress.add_task("Analyzing", total=None, status="loading mimi...")

    try:
        from osmium.analyzer.mimi_mlx import encode_mlx
        progress.update(task, status="mimi (mlx)")
        codes = encode_mlx(audio.samples, audio.sample_rate)
    except (ImportError, Exception):
        try:
            from osmium.analyzer.mimi import encode as mimi_encode
            progress.update(task, status="mimi (cpu)")
            codes = mimi_encode(audio.samples, audio.sample_rate)
        except ImportError:
            console.print("[red]Mimi requires:[/red] uv pip install -e '.[neural]'")
            raise SystemExit(1)

    imp = compute_importance(codes, audio.samples, audio.sample_rate)
    progress.remove_task(task)
    return imp


def _write_analysis(imp, resolution_s, duration, output_file, console):
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
        console.print(f"  wrote importance map to {out}")


def _stream_mode(input_file, speed):
    from osmium.tsm.stream import process_streaming
    click.echo("  streaming mode (f32le mono 24kHz)...", err=True)
    chunks = decode_streaming(input_file, chunk_seconds=5.0)
    for output_chunk in process_streaming(chunks, speed=speed, window_size=2048):
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
