import argparse
import os
import random

import pytorch_lightning as pl
import torch
import transformers

from vocos.experiment import VocosExp
from vocos.feature_extractors import MelSpectrogramFeatures
from vocos.heads import ISTFTHead
from vocos.helpers import plot_spectrogram_to_numpy
from vocos.models import VocosBackbone
from vocos.modules import safe_log
from vocos.pretrained import Vocos

from scripts.vocos_finetune.augment import random_resample_roundtrip


def compute_aug_ratio(global_step: int) -> float:
    if global_step <= 2000:
        return 0.3
    elif global_step <= 4000:
        return 0.3 + 0.2 * (global_step - 2000) / 2000
    else:
        return 0.5


class VocosFineTuneExp(VocosExp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_ratio = 0.3

    def _maybe_augment(self, features: torch.Tensor) -> torch.Tensor:
        if self.training and random.random() < self.aug_ratio:
            return random_resample_roundtrip(features, min_rate=1.5, max_rate=5.0, presmooth_sigma=2.0)
        return features

    def _forward_with_augment(self, audio_input, **kwargs):
        features = self.feature_extractor(audio_input, **kwargs)
        features = self._maybe_augment(features)
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)
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
                self.logger.experiment.add_audio(
                    "train/audio_in", audio_input[0].data.cpu(), self.global_step, self.hparams.sample_rate
                )
                self.logger.experiment.add_audio(
                    "train/audio_pred", audio_hat[0].data.cpu(), self.global_step, self.hparams.sample_rate
                )
                with torch.no_grad():
                    mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
                    mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
                self.logger.experiment.add_image(
                    "train/mel_target",
                    plot_spectrogram_to_numpy(mel.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    "train/mel_pred",
                    plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )

            return loss

    def on_train_batch_start(self, *args):
        super().on_train_batch_start(*args)
        self.aug_ratio = compute_aug_ratio(self.global_step)


class QualityGateCallback(pl.Callback):
    def __init__(self):
        self._baseline_mel_loss = None

    def on_validation_end(self, trainer, pl_module):
        mel_loss = trainer.callback_metrics.get("val/mel_loss")
        if mel_loss is None:
            return
        val = mel_loss.item()
        if self._baseline_mel_loss is None:
            self._baseline_mel_loss = val
        elif val > self._baseline_mel_loss * 1.1:
            print(f"WARNING: mel loss {val:.4f} regressed beyond 110% of baseline {self._baseline_mel_loss:.4f}")


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
    args = parser.parse_args()

    from scripts.vocos_finetune.dataset import AudioDataset
    from torch.utils.data import DataLoader

    model = create_model(
        pretrain_mel_steps=0,
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
        monitor="val_loss",
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

    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        max_steps=args.max_steps,
        val_check_interval=2000,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, QualityGateCallback()],
        logger=logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
