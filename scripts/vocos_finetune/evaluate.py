import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from scripts.vocos_finetune.click_detector import clicks_per_second


def load_finetuned_model(checkpoint_path: Path, model_type: str = "finetune"):
    if model_type == "phase_reg":
        from scripts.vocos_finetune.train import create_phase_reg_model
        model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=1)
    else:
        from scripts.vocos_finetune.train import create_model
        model = create_model(pretrain_mel_steps=0, initial_learning_rate=1e-4, max_steps=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model


def generate_samples(
    checkpoint_path: Path,
    val_filelist: Path,
    output_dir: Path,
    n_utterances: int = 5,
    speeds: list[float] = [2.0, 3.0, 4.0],
    sample_rate: int = 24000,
    model_type: str = "finetune",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(val_filelist) as f:
        utterances = [line.strip() for line in f if line.strip()][:n_utterances]

    from vocos.pretrained import Vocos
    baseline = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    baseline.eval()

    finetuned = load_finetuned_model(checkpoint_path, model_type=model_type)

    readme_lines = ["# A/B Comparison Samples\n"]

    for utt_path in utterances:
        data, sr = sf.read(utt_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        audio = torch.from_numpy(data)
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)

        max_samples = 5 * sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        audio_batch = audio.unsqueeze(0)
        utt_name = Path(utt_path).stem

        for speed in speeds:
            with torch.no_grad():
                features = baseline.feature_extractor(audio_batch)
                T = features.shape[2]
                target_T = max(1, int(T / speed))
                resampled = torch.nn.functional.interpolate(
                    features, size=target_T, mode="linear", align_corners=True,
                )
                baseline_out = baseline.decode(resampled)
                baseline_np = baseline_out.squeeze().numpy()

                features_ft = finetuned.feature_extractor(audio_batch)
                resampled_ft = torch.nn.functional.interpolate(
                    features_ft, size=target_T, mode="linear", align_corners=True,
                )
                x_ft = finetuned.backbone(resampled_ft)
                finetuned_out = finetuned.head(x_ft)
                finetuned_np = finetuned_out.squeeze().numpy()

            baseline_path = output_dir / f"{utt_name}_{speed}x_baseline.wav"
            finetuned_path = output_dir / f"{utt_name}_{speed}x_finetuned.wav"
            sf.write(str(baseline_path), baseline_np, sample_rate)
            sf.write(str(finetuned_path), finetuned_np, sample_rate)

            baseline_clicks = clicks_per_second(baseline_np, sample_rate)
            finetuned_clicks = clicks_per_second(finetuned_np, sample_rate)

            readme_lines.append(f"## {utt_name} @ {speed}x")
            readme_lines.append(f"- Baseline clicks/s: {baseline_clicks:.1f}")
            readme_lines.append(f"- Fine-tuned clicks/s: {finetuned_clicks:.1f}")
            readme_lines.append("")

    readme_path = output_dir / "README.txt"
    readme_path.write_text("\n".join(readme_lines))
    print(f"Samples saved to {output_dir}")
    print(f"Summary: {readme_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-filelist", type=Path, default=Path("training/data/filelists/val.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-utterances", type=int, default=5)
    parser.add_argument("--model-type", choices=["finetune", "phase_reg"], default="finetune")
    args = parser.parse_args()

    generate_samples(
        checkpoint_path=args.checkpoint,
        val_filelist=args.val_filelist,
        output_dir=args.output_dir,
        n_utterances=args.n_utterances,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
