import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision
import yaml
from omegaconf import DictConfig, OmegaConf

from src.dataset.datamodule import IMG_MEAN, IMG_STD, get_input_size_wo_pad, show_stft
from src.dataset.dataset import WaveformDataset
from src.postprocess.visualize import plot_rec


class LitModel(pl.LightningModule):
    def __init__(
        self, conf: DictConfig, dataset_len: int = 72899, logger_name="tensorboard"
    ) -> None:
        super().__init__()
        self.save_hyperparameters()  # type: ignore
        # self.hparams = conf  # type: ignore[misc]
        self.conf = conf
        self.dataset_len = dataset_len
        self.logger_name = logger_name

        print("\t >>do segmentation")
        self.num_inchannels = len(self.conf.stft_targets) * 3
        self.classes_num = self.num_inchannels

        smp_params = OmegaConf.to_container(conf.model.smp_params)
        smp_params["classes"] = self.classes_num

        smp_arch = smp_params.pop("arch_name")
        if smp_arch == "unet":
            smp_func = smp.Unet
        elif smp_arch == "unetpp":
            smp_func = smp.UnetPlusPlus
        elif smp_arch == "manet":
            smp_func = smp.MAnet
        elif smp_arch == "deeplabv3":
            smp_func = smp.DeepLabV3
        elif smp_arch == "deeplabv3p":
            smp_func = smp.DeepLabV3Plus

        self.model = smp_func(**smp_params)
        # self.model = nn.Sequential(smp_func(**smp_params),)

        if self.num_inchannels != 3:
            patch_first_conv(self.model, in_channels=self.num_inchannels)

        if self.conf.model.channels_last:
            # Need to be done once, after model initialization (or load)
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.conf.model.loss == "mse":
            self.criterion = torch.nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError

        self.loss_ch = (
            len(self.conf.stft_targets) * 1
            if self.conf.train_energy_only
            else len(self.conf.stft_targets) * 3
        )

        if self.conf.model.metrics == "mse":
            self.metrics = pl.metrics.MeanSquaredError()
        else:
            raise NotImplementedError

        if self.conf.model.last_act == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif self.conf.model.last_act == "tanh":
            self.activation = torch.nn.Tanh()
        elif self.conf.model.last_act == "identity":
            self.activation = torch.nn.Identity()
        else:
            raise NotImplementedError

        self.val_sync_dist = self.conf.trainer.gpus > 1
        self.is_debug = self.conf.is_debug

        self.h_, self.w_ = get_input_size_wo_pad(
            n_fft=self.conf.stft_params.n_fft, input_width=self.conf.input_width
        )

    def on_fit_start(self):
        self._set_image_normalization()

    def on_test_start(self):
        self._set_image_normalization()

    def forward(self, x):
        x = self.model(x)
        return x

    def _remove_pad(
        self, inputs: torch.Tensor, pred: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        return (
            inputs[:, :, : self.h_, : self.w_],
            pred[:, :, : self.h_, : self.w_],
            targets[:, :, : self.h_, : self.w_],
        )

    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        if self.conf.model.channels_last:
            # Need to be done for every input
            inputs = inputs.to(memory_format=torch.channels_last)

        targets = batch["target_image"]
        outputs = self.model(inputs)
        pred = self.activation(outputs)
        # if self.conf.model.last_act == "tanh":
        #     pred = pred * 2.0
        inputs, pred, targets = self._remove_pad(
            inputs=inputs, pred=pred, targets=targets
        )
        if self.conf.gt_as_mask:
            loss = self.criterion(
                pred, targets - (inputs * self._img_std + self._img_mean)
            )[:, : self.loss_ch].mean()
        else:
            loss = self.criterion(pred, targets - inputs)[:, : self.loss_ch].mean()

        if self.logger_name == "tensorboard":
            self.log("train_loss", loss)
        elif self.logger_name == "neptune":
            self.logger.experiment["loss/train"].log(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        if self.conf.model.channels_last:
            # Need to be done for every input
            inputs = inputs.to(memory_format=torch.channels_last)

        targets = batch["target_image"]
        outputs = self.model(inputs)
        pred = self.activation(outputs)
        # if self.conf.model.last_act == "tanh":
        #     pred = pred * 2.0

        inputs, pred, targets = self._remove_pad(
            inputs=inputs, pred=pred, targets=targets
        )
        if self.conf.gt_as_mask:
            loss = self.criterion(
                pred, targets - (inputs * self._img_std + self._img_mean)
            )[:, : self.loss_ch].mean()
            pred = pred + (inputs * self._img_std + self._img_mean)
        else:
            loss = self.criterion(pred, targets - inputs)[:, : self.loss_ch].mean()
            pred = pred + inputs

        # only for checkpoint call back
        self.log("val_loss", loss)
        sequence_results = self.convert_img_pred_to_sequence(pred=pred, batch=batch)
        abs_error = sequence_results.pop("abs_error")
        metrics = np.mean(abs_error)

        if batch_idx in [0, 2]:
            pred = torch.clamp(pred, 0.0, 1.0)
            epoch = (
                self.trainer.global_step * self.conf.batch_size
            ) // self.dataset_len

            num_ = 3
            ba_ind = 0
            imgs = {
                "inputs": inputs[:num_].cpu().numpy().transpose((0, 2, 3, 1)),
                "pred": pred[:num_].detach().cpu().numpy().transpose((0, 2, 3, 1)),
                "targets": targets[:num_]
                .detach()
                .cpu()
                .numpy()
                .transpose((0, 2, 3, 1)),
            }
            # === PLOT ===
            nrows = 3
            ncols = 3
            ch_ = 0
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(12, 6), sharey=True, sharex=True,
            )
            fig.suptitle(
                "_".join(
                    [
                        str(batch["phone"][ba_ind]),
                        str(batch["millisSinceGpsEpoch"][ba_ind].cpu().numpy()),
                        str(batch["phone_time"][ba_ind].cpu().numpy()),
                        "epoch",
                        str(epoch),
                    ]
                )
            )

            D_mats = {}
            for j, (key, img) in enumerate(imgs.items()):
                gt_as_mask = (key in ["pred", "targets"]) and self.conf.gt_as_mask
                abs_, cos_, sin_ = WaveformDataset.handle_stft_normalize(
                    img=img.copy(),
                    cnum=len(self.conf.stft_targets),
                    is_encode=False,
                    img_std=self._img_std.cpu().numpy().transpose((0, 2, 3, 1)),
                    img_mean=self._img_mean.cpu().numpy().transpose((0, 2, 3, 1)),
                    gt_as_mask=gt_as_mask,
                )
                show_stft(
                    conf=self.conf,
                    D_abs=abs_[ba_ind][..., ch_],
                    D_cos=cos_[ba_ind][..., ch_],
                    D_sin=sin_[ba_ind][..., ch_],
                    ax=axes,
                    stft_ind=j,
                    stft_name=key,
                )
                D_mats[key] = {
                    "D_abs": abs_[ba_ind][..., ch_],
                    "D_theta": np.arctan2(
                        sin_[ba_ind][..., ch_], cos_[ba_ind][..., ch_]
                    ),
                }

            if self.logger_name == "tensorboard":
                self.logger.experiment.add_figure(
                    "prediction_fig", fig, global_step=self.trainer.global_step,
                )
            elif self.logger_name == "neptune":
                self.logger.experiment[f"val/pred_{batch_idx}_{ba_ind}"].log(fig)
            plt.close()

            x_gts = batch[self.conf.stft_targets[0].replace("_diff", "_gt_diff")]
            x = batch[self.conf.stft_targets[0]]
            plot_rec(
                x=x[ba_ind].cpu().numpy(),
                x_gt=x_gts[ba_ind].cpu().numpy(),
                D_abs=D_mats["pred"]["D_abs"],
                D_theta=D_mats["pred"]["D_theta"],
                D_abs_gt=D_mats["targets"]["D_abs"],
                D_theta_gt=D_mats["targets"]["D_theta"],
                length=x_gts[ba_ind].shape[0],
                is_db=self.conf.stft_params.is_db,
                hop_length=self.conf.stft_params.hop_length,
                win_length=self.conf.stft_params.win_length,
                logger=self.logger,
                logger_name=self.logger_name,
                global_step=self.trainer.global_step,
                log_name=f"val/pred_{batch_idx}_{ba_ind}_line",
                target_name=self.conf.stft_targets[0],
            )

        return {"loss": loss, "metrics": metrics, "sequence_results": sequence_results}

    def validation_epoch_end(self, validation_step_outputs):
        keys = list(validation_step_outputs[0].keys())
        met_dict = {key: [] for key in keys}
        for pred in validation_step_outputs:
            for key in keys:
                met_dict[key].append(pred[key])
        sequence_results = {key: [] for key in met_dict["sequence_results"][0].keys()}

        for key in keys:
            if key == "sequence_results":
                for seq_res in met_dict[key]:
                    for seq_key, values in seq_res.items():
                        if not isinstance(values, np.ndarray):
                            values = np.array(values)
                        sequence_results[seq_key].append(values)
            elif isinstance(met_dict[key][0], torch.Tensor):
                met_dict[key] = torch.mean(torch.stack(met_dict[key])).cpu().numpy()

            else:
                met_dict[key] = np.mean(np.stack(met_dict[key]))

        for seq_key, values in sequence_results.items():
            sequence_results[seq_key] = np.concatenate(values)

        pred_df = self.generate_pred_df(
            sequence_results=sequence_results, is_test=False, agg_mode="mean"
        )
        pred_df.to_csv(f"./val_{self.trainer.global_step}.csv", index=False)
        phone_mae = []
        # the fisrt step, there is no velocity info
        pred_df.dropna(axis=0, inplace=True)
        for phone, df_ in pred_df.groupby("phone"):
            gt_targets = [
                target.replace("_diff", "_gt_diff") for target in self.conf.stft_targets
            ]
            phone_mae.append(
                np.mean(
                    np.abs(df_[self.conf.stft_targets].values - df_[gt_targets].values)
                )
            )

        self.log("phone_mae", np.mean(phone_mae).tolist())
        if self.logger_name == "tensorboard":
            self.log("epoch_loss/val", met_dict["loss"].tolist())
            self.log("epoch_metrics/val", met_dict["metrics"].tolist())
        elif self.logger_name == "neptune":
            self.logger.experiment["metrics/phone_mae"].log(np.mean(phone_mae).tolist())
            self.logger.experiment["loss/val"].log(met_dict["loss"].tolist())
            self.logger.experiment["metrics/val"].log(met_dict["metrics"].tolist())

    def convert_img_pred_to_sequence(
        self, pred: torch.Tensor, batch: Dict[str, torch.Tensor], is_test: bool = False
    ) -> Dict[str, np.ndarray]:
        pred = torch.clamp(pred, 0.0, 1.0)
        D_abs_b, D_cos_b, D_sin_b = WaveformDataset.handle_stft_normalize(
            img=pred.clone(),
            cnum=len(self.conf.stft_targets),
            is_encode=False,
            img_std=self._img_std,
            img_mean=self._img_mean,
            gt_as_mask=True,
            image_fomat="torch",
        )
        D_theta_b = torch.atan2(D_sin_b, D_cos_b)

        D_abs_b = D_abs_b.cpu().numpy().transpose((0, 2, 3, 1))
        D_theta_b = D_theta_b.cpu().numpy().transpose((0, 2, 3, 1))

        stft_preds = {}
        for i, key in enumerate(self.conf.stft_targets):
            stft_preds[key] = {
                "D_abs_b": D_abs_b[..., i],
                "D_theta_b": D_theta_b[..., i],
            }

        abs_errors = []
        sequence_results = {}
        for stft_target, D_mats in stft_preds.items():
            # batch
            x_rec_b = []
            for D_abs, D_theta in zip(D_mats["D_abs_b"], D_mats["D_theta_b"]):
                if self.conf.stft_params.is_db:
                    D_abs = librosa.db_to_amplitude(D_abs, ref=1.0)
                x_rec = librosa.istft(
                    np.exp(1j * D_theta) * D_abs,
                    hop_length=self.conf.stft_params.hop_length,
                    win_length=self.conf.stft_params.win_length,
                    length=self.conf.input_width,
                )
                x_rec_b.append(x_rec)

            x_rec_b = np.stack(x_rec_b)

            sequence_results[stft_target] = x_rec_b
            if not is_test:
                stft_target_gt = stft_target.replace("_diff", "_gt_diff")
                sequence_results[stft_target_gt] = batch[stft_target_gt].cpu().numpy()
                error = np.abs(x_rec_b - sequence_results[stft_target_gt])
                abs_errors.append(error)
                sequence_results["abs_error"] = np.stack(abs_errors)

        sequence_results.update(
            {
                "phone": batch["phone"],
                "phone_time": batch["phone_time"].cpu().numpy(),
                "millisSinceGpsEpoch": batch["millisSinceGpsEpoch"].cpu().numpy(),
            }
        )
        return sequence_results

    def generate_pred_df(
        self, sequence_results: dict, is_test: bool = False, agg_mode: str = "mean"
    ) -> pd.DataFrame:
        phone_results = {
            str(phone): {} for phone in np.unique(sequence_results["phone"])
        }
        df = []

        stft_targets = OmegaConf.to_container(self.conf.stft_targets)

        if not is_test:
            gt_targets = [
                target.replace("_diff", "_gt_diff") for target in stft_targets
            ]
            stft_targets = stft_targets + gt_targets
            samplling_delta = self.conf.val_sampling_delta
        else:
            samplling_delta = self.conf.test_sampling_delta

        for phone in np.unique(sequence_results["phone"]):
            phone_mask = sequence_results["phone"] == phone
            phone_time = sequence_results["phone_time"][phone_mask]
            phone_results[phone].update({"phone_time": phone_time})
            millisSinceGpsEpoch = sequence_results["millisSinceGpsEpoch"][phone_mask]

            millisSinceGpsEpoch = np.arange(
                millisSinceGpsEpoch.min() - 1000,
                millisSinceGpsEpoch.max() + self.conf.input_width * 1000,
                1000,
                dtype=np.int64,
            )
            phone_results[phone].update({"millisSinceGpsEpoch": millisSinceGpsEpoch})

            for stft_target in stft_targets:
                num_preds = np.sum(phone_mask)
                assert (
                    num_preds
                    <= (phone_time.max() + self.conf.input_width) // samplling_delta
                )
                pred_sequence = np.zeros(
                    (num_preds, phone_time.max() + self.conf.input_width,),
                    dtype=np.float64,
                )
                pred_mask = np.zeros(
                    (num_preds, phone_time.max() + self.conf.input_width,),
                    dtype=np.bool,
                )

                for i, (ph_t, seq) in enumerate(
                    zip(phone_time, sequence_results[stft_target][phone_mask])
                ):
                    pred_sequence[i, ph_t : ph_t + self.conf.input_width] = seq
                    pred_mask[i, ph_t : ph_t + self.conf.input_width] = True

                if agg_mode == "mean":
                    pred_sequence = np.sum(pred_sequence, axis=0) / np.sum(
                        pred_mask, axis=0
                    )
                else:
                    raise NotImplementedError

                phone_results[phone].update({stft_target: pred_sequence})

            for_df = {
                "phone": np.repeat(phone, pred_sequence.shape[0]),
                "millisSinceGpsEpoch": millisSinceGpsEpoch,
            }

            for_df.update(
                {
                    stft_target: phone_results[phone][stft_target]
                    for stft_target in stft_targets
                }
            )

            df.append(pd.DataFrame(for_df))
            # df.append(np.stack(for_df).transpose(1, 0))
        pred_df = pd.concat(df, axis=0)
        dtypes = {target: np.float64 for target in stft_targets}
        pred_df = pred_df.astype(dtypes)
        return pred_df

    def flip_tta(
        self, model, inputs: torch.Tensor, pred: torch.Tensor,
    ):
        transforms = [
            torchvision.transforms.functional.hflip,
        ]
        inverts = [
            torchvision.transforms.functional.hflip,
        ]
        for trans_, invert_ in zip(transforms, inverts):
            outputs = self.model(trans_(inputs))
            pred_aug = self.activation(outputs)
            pred += invert_(pred_aug)
        pred *= 1.0 / (len(transforms) + 1)
        return pred

    def test_step(self, batch, batch_idx):
        inputs = batch["image"]
        if self.conf.model.channels_last:
            # Need to be done for every input
            inputs = inputs.to(memory_format=torch.channels_last)

        outputs = self.model(inputs)
        pred = self.activation(outputs)
        if self.conf.use_flip_tta:
            pred = self.flip_tta(model=self.model, inputs=inputs, pred=pred)

        inputs, pred, _ = self._remove_pad(inputs=inputs, pred=pred, targets=pred)
        # if self.conf.model.last_act == "tanh":
        #     pred = pred * 2.0

        if self.conf.gt_as_mask:
            pred = pred + (inputs * self._img_std + self._img_mean)
        else:
            pred = pred + inputs

        sequence_results = self.convert_img_pred_to_sequence(
            pred=pred, batch=batch, is_test=True
        )

        return {"sequence_results": sequence_results}

    def test_epoch_end(self, test_step_outputs):
        keys = list(test_step_outputs[0].keys())
        met_dict = {key: [] for key in keys}
        for pred in test_step_outputs:
            for key in keys:
                met_dict[key].append(pred[key])
        sequence_results = {key: [] for key in met_dict["sequence_results"][0].keys()}

        for key in keys:
            if key == "sequence_results":
                for seq_res in met_dict[key]:
                    for seq_key, values in seq_res.items():
                        if not isinstance(values, np.ndarray):
                            values = np.array(values)
                        sequence_results[seq_key].append(values)
            elif isinstance(met_dict[key][0], torch.Tensor):
                met_dict[key] = torch.mean(torch.stack(met_dict[key])).cpu().numpy()

            else:
                met_dict[key] = np.mean(np.stack(met_dict[key]))

        for seq_key, values in sequence_results.items():
            sequence_results[seq_key] = np.concatenate(values)

        pred_df = self.generate_pred_df(
            sequence_results=sequence_results, is_test=True, agg_mode="mean"
        )
        fname = f"pred_test_flip_{self.conf.use_flip_tta}_d{self.conf.test_sampling_delta}.csv"
        save_path = os.path.join(os.path.dirname(self.conf.ckpt_path), fname)
        print("test prediction csv", save_path)
        pred_df.to_csv(save_path, index=False)

    def _set_image_normalization(self) -> None:
        img_mean = IMG_MEAN[: self.num_inchannels]  # type: ignore[union-attr]
        img_std = IMG_STD[: self.num_inchannels]  # type: ignore[union-attr]
        self._img_std = torch.tensor(
            np.array(img_std, dtype=np.float32)[None, :, None, None], device=self.device
        )
        self._img_mean = torch.tensor(
            np.array(img_mean, dtype=np.float32)[None, :, None, None],
            device=self.device,
        )

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_idx,
        closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if not self.conf.find_lr:
            if self.trainer.global_step < self.warmup_steps:
                lr_scale = min(
                    1.0, float(self.trainer.global_step + 1) / self.warmup_steps
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.conf.lr
            else:
                pct = (self.trainer.global_step - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps
                )
                pct = min(1.0, pct)
                for pg in optimizer.param_groups:
                    pg["lr"] = self._annealing_cos(pct, start=self.conf.lr, end=0.0)

        if self.logger_name == "neptune":
            self.logger.experiment["train/lr"].log(optimizer.param_groups[0]["lr"])
        optimizer.step(closure=closure)
        optimizer.zero_grad()

    def _annealing_cos(self, pct: float, start: float = 0.1, end: float = 0.0) -> float:
        """
        https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
        Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.
        """
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def configure_optimizers(self):
        self.total_steps = (
            self.dataset_len // self.conf.batch_size
        ) * self.conf.trainer.max_epochs
        self.warmup_steps = int(self.total_steps * self.conf.warmup_ratio)

        if self.conf.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.conf.lr, momentum=0.9, weight_decay=4e-5,
            )
        elif self.conf.optim_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.conf.lr)
        else:
            raise NotImplementedError
        # steps_per_epoch = self.hparams.dataset_len // self.hparams.batch_size
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.hparams.lr,
        #     max_epochs=self.hparams.max_epochs,
        #     steps_per_epoch=steps_per_epoch,
        # )
        # return [optimizer], [scheduler]
        return optimizer


def patch_first_conv(model, in_channels: int = 4) -> None:
    """
    from segmentation_models_pytorch/encoders/_utils.py
    Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    # reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    elif in_channels == 4:
        weight = torch.nn.Parameter(torch.cat([weight, weight[:, -1:, :, :]], dim=1))
    elif in_channels % 3 == 0:
        weight = torch.nn.Parameter(torch.cat([weight] * (in_channels // 3), dim=1))

    module.weight = weight
