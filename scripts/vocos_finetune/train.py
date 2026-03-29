import argparse
import os
import random
from pathlib import Path

import pytorch_lightning as pl
import torch
import transformers

from vocos.experiment import VocosExp
from vocos.feature_extractors import MelSpectrogramFeatures
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone
from vocos.pretrained import Vocos

from scripts.vocos_finetune.augment import random_resample_roundtrip


def compute_aug_ratio(global_step: int) -> float:
    if global_step <= 1000:
        return 0.15
    elif global_step <= 3000:
        return 0.15 + 0.10 * (global_step - 1000) / 2000
    else:
        return 0.25


class VocosFineTuneExp(VocosExp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_ratio = 0.3

    def _maybe_augment(self, features: torch.Tensor) -> torch.Tensor:
        if self.training and random.random() < self.aug_ratio:
            return random_resample_roundtrip(features, min_rate=1.5, max_rate=5.0, presmooth_sigma=2.0)
        return features

    def _forward_with_augment(self, audio_input, **kwargs):
        T = audio_input.shape[-1]
        features = self.feature_extractor(audio_input, **kwargs)
        features = self._maybe_augment(features)
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)[..., :T]
        return audio_output

    def training_step(self, batch, batch_idx, optimizer_idx, **kwargs):
        audio_input = batch

        if optimizer_idx == 0 and self.train_discriminator:
            with torch.no_grad():
                audio_hat = self._forward_with_augment(audio_input, **kwargs)
            self._cached_audio_hat = audio_hat

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(
                y=audio_input, y_hat=audio_hat, **kwargs,
            )
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(
                y=audio_input, y_hat=audio_hat, **kwargs,
            )
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd

            self.log("discriminator/total", loss, prog_bar=True)
            self.log("discriminator/multi_period_loss", loss_mp)
            self.log("discriminator/multi_res_loss", loss_mrd)
            return loss

        if optimizer_idx == 1:
            seed = batch_idx + self.global_step
            random.seed(seed)
            torch.manual_seed(seed)

            audio_hat = self._forward_with_augment(audio_input, **kwargs)

            if self.train_discriminator:
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
                loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
                loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

                self.log("generator/multi_period_loss", loss_gen_mp)
                self.log("generator/multi_res_loss", loss_gen_mrd)
                self.log("generator/feature_matching_mp", loss_fm_mp)
                self.log("generator/feature_matching_mrd", loss_fm_mrd)
            else:
                loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0

            mel_loss = self.melspec_loss(audio_hat, audio_input)
            loss = (
                loss_gen_mp
                + self.hparams.mrd_loss_coeff * loss_gen_mrd
                + loss_fm_mp
                + self.hparams.mrd_loss_coeff * loss_fm_mrd
                + self.mel_loss_coeff * mel_loss
            )

            self.log("generator/total_loss", loss, prog_bar=True)
            self.log("mel_loss_coeff", self.mel_loss_coeff)
            self.log("generator/mel_loss", mel_loss)

            if self.global_step % 1000 == 0 and self.global_rank == 0:
                try:
                    self.logger.experiment.add_audio(
                        "train/audio_in", audio_input[0].data.cpu(), self.global_step, self.hparams.sample_rate
                    )
                    self.logger.experiment.add_audio(
                        "train/audio_pred", audio_hat[0].data.cpu(), self.global_step, self.hparams.sample_rate
                    )
                except Exception:
                    pass

            return loss

    def on_train_batch_start(self, *args):
        super().on_train_batch_start(*args)
        self.aug_ratio = compute_aug_ratio(self.global_step)

    def validation_step(self, batch, batch_idx, **kwargs):
        audio_input = batch
        from scripts.vocos_finetune.augment import resample_roundtrip
        from scripts.vocos_finetune.click_detector import clicks_per_second
        import numpy as np

        T = audio_input.shape[-1]
        features_normal = self.feature_extractor(audio_input)
        x_normal = self.backbone(features_normal)
        audio_hat_normal = self.head(x_normal)[..., :T]
        mel_loss_normal = self.melspec_loss(audio_hat_normal, audio_input)

        result = {
            "val_loss": mel_loss_normal,
            "mel_loss_normal": mel_loss_normal,
            "audio_input": audio_input[0],
            "audio_pred_normal": audio_hat_normal[0],
        }

        mel_losses_aug = []
        for rate in [2.0, 3.0, 4.0]:
            features_aug = resample_roundtrip(features_normal.clone(), rate=rate, presmooth_sigma=2.0)
            x_aug = self.backbone(features_aug)
            audio_hat_aug = self.head(x_aug)[..., :T]
            mel_loss_aug = self.melspec_loss(audio_hat_aug, audio_input)
            mel_losses_aug.append(mel_loss_aug)

            audio_np = audio_hat_aug[0].detach().cpu().numpy()
            click_rate = clicks_per_second(audio_np, sample_rate=self.hparams.sample_rate)

            rate_key = f"{rate}x".replace(".", "_")
            result[f"mel_loss_{rate_key}"] = mel_loss_aug
            result[f"click_rate_{rate_key}"] = torch.tensor(click_rate)

        avg_aug_loss = torch.stack(mel_losses_aug).mean()
        composite = 0.5 * mel_loss_normal + 0.5 * avg_aug_loss
        result["mel_loss_augmented"] = avg_aug_loss
        result["val_loss"] = composite

        self.log("val/mel_loss_normal", mel_loss_normal, prog_bar=True)
        self.log("val/mel_loss_augmented", avg_aug_loss, prog_bar=True)
        self.log("val/composite_loss", composite, prog_bar=True)

        return result

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0 and outputs:
            try:
                first = outputs[0]
                self.logger.experiment.add_audio(
                    "val/audio_in", first["audio_input"].data.cpu().numpy(),
                    self.global_step, self.hparams.sample_rate,
                )
                self.logger.experiment.add_audio(
                    "val/audio_normal", first["audio_pred_normal"].data.cpu().numpy(),
                    self.global_step, self.hparams.sample_rate,
                )
            except Exception:
                pass

        avg_normal = torch.stack([x["mel_loss_normal"] for x in outputs]).mean()
        avg_aug = torch.stack([x["mel_loss_augmented"] for x in outputs]).mean()
        avg_composite = torch.stack([x["val_loss"] for x in outputs]).mean()

        self.log("val_loss", avg_composite, sync_dist=True)
        self.log("val/mel_loss_normal", avg_normal, sync_dist=True)
        self.log("val/mel_loss_augmented", avg_aug, sync_dist=True)

        for rate in [2.0, 3.0, 4.0]:
            rate_key = f"{rate}x".replace(".", "_")
            mel_key = f"mel_loss_{rate_key}"
            click_key = f"click_rate_{rate_key}"
            if mel_key in outputs[0]:
                avg_mel = torch.stack([x[mel_key] for x in outputs]).mean()
                avg_click = torch.stack([x[click_key] for x in outputs]).mean()
                self.log(f"val/mel_loss_{rate_key}", avg_mel, sync_dist=True)
                self.log(f"val/click_rate_{rate_key}", avg_click, sync_dist=True)


class QualityGateCallback(pl.Callback):
    def __init__(self):
        self._baseline_mel_loss = None

    def on_validation_end(self, trainer, pl_module):
        mel_loss = trainer.callback_metrics.get("val/mel_loss_normal")
        if mel_loss is None:
            return
        val = mel_loss.item()
        if self._baseline_mel_loss is None:
            self._baseline_mel_loss = val
        elif val > self._baseline_mel_loss * 1.1:
            print(f"WARNING: mel loss {val:.4f} regressed beyond 110% of baseline {self._baseline_mel_loss:.4f}")


class EvalSampleCallback(pl.Callback):
    def __init__(self, val_filelist, output_base, every_n_steps=2000):
        self.val_filelist = val_filelist
        self.output_base = Path(output_base)
        self.every_n_steps = every_n_steps
        self._last_step = -1

    def on_validation_end(self, trainer, pl_module):
        step = pl_module.global_step
        if step == self._last_step or step % self.every_n_steps != 0 or step == 0:
            return
        self._last_step = step

        output_dir = self.output_base / f"step_{step}"
        if output_dir.exists():
            return

        print(f"\n==> Generating A/B samples at step {step}")

        ckpt_path = self.output_base / f"_tmp_eval_step_{step}.ckpt"
        trainer.save_checkpoint(str(ckpt_path))

        try:
            from scripts.vocos_finetune.evaluate import generate_samples
            generate_samples(
                checkpoint_path=ckpt_path,
                val_filelist=self.val_filelist,
                output_dir=output_dir,
                n_utterances=5,
            )
        except Exception as e:
            print(f"Warning: eval sample generation failed at step {step}: {e}")
        finally:
            ckpt_path.unlink(missing_ok=True)


def create_model(
    pretrain_mel_steps: int = 0,
    initial_learning_rate: float = 2e-5,
    max_steps: int = 20000,
) -> VocosFineTuneExp:
    feature_extractor = MelSpectrogramFeatures(
        sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100
    )
    backbone = VocosBackbone(
        input_channels=100, dim=512, intermediate_dim=1536, num_layers=8
    )
    head = ISTFTHead(dim=512, n_fft=1024, hop_length=256)

    pretrained = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    backbone.load_state_dict(pretrained.backbone.state_dict())
    head.load_state_dict(pretrained.head.state_dict())

    model = VocosFineTuneExp(
        feature_extractor=feature_extractor,
        backbone=backbone,
        head=head,
        sample_rate=24000,
        initial_learning_rate=initial_learning_rate,
        pretrain_mel_steps=pretrain_mel_steps,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-filelist", required=True)
    parser.add_argument("--val-filelist", required=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints/vocos_finetune")
    parser.add_argument("--log-dir", default="logs/vocos_finetune")
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrain-mel-steps", type=int, default=999999)
    args = parser.parse_args()

    from scripts.vocos_finetune.dataset import AudioDataset
    from torch.utils.data import DataLoader

    model = create_model(
        pretrain_mel_steps=args.pretrain_mel_steps,
        initial_learning_rate=args.lr,
        max_steps=args.max_steps,
    )

    train_dataset = AudioDataset(
        filelist_path=args.train_filelist, num_samples=24000, sample_rate=24000, train=True
    )
    val_dataset = AudioDataset(
        filelist_path=args.val_filelist, num_samples=24000, sample_rate=24000, train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor="val/composite_loss",
        save_top_k=3,
        every_n_train_steps=2000,
        save_last=True,
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=args.log_dir, name="vocos_finetune")

    resume_path = None
    if args.resume:
        last_ckpt = os.path.join(args.checkpoint_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            resume_path = last_ckpt

    eval_callback = EvalSampleCallback(
        val_filelist=args.val_filelist,
        output_base=Path("training/eval_samples"),
    )

    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        max_steps=args.max_steps,
        val_check_interval=1000,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, QualityGateCallback(), eval_callback],
        logger=logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
