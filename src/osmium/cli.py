import sys
import time
import click
import numpy as np
from pathlib import Path

from osmium.io.decode import decode
from osmium.io.encode import encode, encode_pcm_stdout
from osmium.tsm.phase_vocoder import phase_vocoder_stretch, variable_rate_phase_vocoder
from osmium.tsm.rate_schedule import uniform_rate_schedule


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-s", "--speed", required=True, type=float, help="Target speed factor (e.g., 2.0, 3.5)")
@click.option("-o", "--output", "output_file", type=click.Path(), help="Output file path")
@click.option("--stream", is_flag=True, help="Stream raw f32le PCM to stdout")
@click.option("--resolution", default="20ms", help="Importance map resolution (e.g., 10ms, 20ms, 80ms)")
@click.option("--window", "window_size", default=2048, type=int, help="STFT window size")
@click.option("--device", default="auto", type=click.Choice(["auto", "mlx", "cuda", "cpu"]), help="Compute device")
@click.option("--no-model", is_flag=True, help="Skip neural analysis, use uniform-rate TSM")
@click.option("--analyze-only", is_flag=True, help="Output importance map as JSON")
def main(input_file, speed, output_file, stream, resolution, window_size, device, no_model, analyze_only):
    """Osmium — high-quality speech acceleration."""
    if not stream and not output_file:
        raise click.UsageError("Either --output or --stream is required")

    if speed <= 0:
        raise click.UsageError("Speed must be positive")

    res_ms = _parse_resolution(resolution)

    click.echo(f"osmium: {Path(input_file).name} @ {speed}x", err=True)

    t0 = time.time()
    click.echo("  decoding...", err=True)
    audio = decode(input_file)
    decode_time = time.time() - t0
    duration = len(audio.samples) / audio.sample_rate
    click.echo(f"  decoded {duration:.1f}s ({len(audio.samples)} samples) in {decode_time:.1f}s", err=True)

    t1 = time.time()
    click.echo(f"  stretching @ {speed}x (window={window_size})...", err=True)
    output_samples = phase_vocoder_stretch(
        audio.samples,
        speed=speed,
        window_size=window_size,
        sample_rate=audio.sample_rate,
    )
    stretch_time = time.time() - t1
    out_duration = len(output_samples) / audio.sample_rate
    click.echo(f"  output: {out_duration:.1f}s in {stretch_time:.1f}s", err=True)

    if stream:
        sys.stdout.buffer.write(encode_pcm_stdout(output_samples))
    else:
        t2 = time.time()
        click.echo(f"  encoding → {output_file}...", err=True)
        encode(output_samples, audio.sample_rate, output_file)
        encode_time = time.time() - t2
        click.echo(f"  done in {encode_time:.1f}s (total: {time.time() - t0:.1f}s)", err=True)


def _parse_resolution(res: str) -> float:
    res = res.strip().lower()
    if res.endswith("ms"):
        return float(res[:-2]) / 1000.0
    elif res.endswith("s"):
        return float(res[:-1])
    else:
        return float(res) / 1000.0
