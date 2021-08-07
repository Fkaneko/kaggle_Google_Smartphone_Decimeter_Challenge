import glob
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as albu
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.dataset.dataset import WaveformDataset
from src.dataset.utils import calc_triangle_center, get_groundtruth
from src.postprocess.postporcess import apply_gauss_smoothing, apply_kf_smoothing
from src.postprocess.visualize import add_distance_diff

IMG_MEAN = (0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.485, 0.456, 0.406)


class GsdcDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        conf: DictConfig,
        val_fold: int = 0,
        batch_size: int = 64,
        num_workers: int = 16,
        aug_mode: int = 0,
        is_debug: bool = False,
    ) -> None:
        super().__init__()
        self.conf = conf
        self.batch_size = batch_size
        self.aug_mode = aug_mode
        self.num_workers = num_workers
        self.is_debug = is_debug
        self.val_fold = val_fold
        self.input_width = conf["input_width"]
        self.num_inchannels = len(conf["stft_targets"]) * 3

        self.img_mean = np.array(IMG_MEAN[: self.num_inchannels])
        self.img_std = np.array(IMG_STD[: self.num_inchannels])

    def prepare_data(self):
        # check
        assert Path(get_original_cwd(), self.conf["data_dir"]).is_dir()

    def _onehot_to_set(self, onehot: np.ndarray):
        return set(np.where(onehot == 1)[0].astype(str).tolist())

    def _use_cached_kalman(self, df: pd.DataFrame, is_test=False) -> pd.DataFrame:
        print("apply kalman filttering")
        processed_kf_path = (
            "../input/kf_test.csv" if is_test else "../input/kf_train.csv"
        )
        processed_kf_path = Path(get_original_cwd(), processed_kf_path)
        try:
            df = pd.read_csv(processed_kf_path)
        except Exception:
            df = apply_kf_smoothing(df=df)  # nan each phone first or last row
            df.to_csv(processed_kf_path, index=False)
        return df

    def setup(self, stage: Optional[str] = None):
        # Assign Train/val split(s) for use in Dataloaders

        conf = self.conf
        if stage == "fit" or stage is None:

            # read data
            data_dir = Path(get_original_cwd(), self.conf["data_dir"])
            self.train_df = pd.read_csv(data_dir / "baseline_locations_train.csv")
            df_path = pd.read_csv(
                Path(get_original_cwd(), "./src/meta_data/path_meta_info.csv")
            )

            # merge graoundtruth
            self.train_df = self.train_df.merge(
                get_groundtruth(data_dir),
                on=["collectionName", "phoneName", "millisSinceGpsEpoch"],
            )

            if self.conf.apply_kalman_filtering:
                self.train_df = self._use_cached_kalman(df=self.train_df, is_test=False)

            # there is nan at being and end...
            if self.conf.stft_targets[0].find("center") > -1:
                self.train_df = calc_triangle_center(
                    df=self.train_df,
                    targets=["latDeg", "lngDeg", "latDeg_gt", "lngDeg_gt"],
                )
            else:
                self.train_df = add_distance_diff(df=self.train_df, is_test=False)

            # train/val split

            df_path = make_split(df=df_path, n_splits=3)
            self.train_df = merge_split_info(data_df=self.train_df, split_df=df_path)
            self.train_df = choose_paths(df=self.train_df, target=self.conf.target_path)

            train_df = self.train_df.loc[self.train_df["fold"] != self.val_fold, :]

            val_df = self.train_df.loc[self.train_df["fold"] == self.val_fold, :]
            if self.conf.data_aug_with_kf:
                train_phone = train_df.phone.unique()
                if self.conf.apply_kalman_filtering:
                    orig_df = pd.read_csv(data_dir / "baseline_locations_train.csv")
                    orig_df = orig_df.merge(
                        get_groundtruth(data_dir),
                        on=["collectionName", "phoneName", "millisSinceGpsEpoch"],
                    )
                else:
                    orig_df = self._use_cached_kalman(df=train_df, is_test=False)
                orig_df = orig_df.loc[orig_df.phone.isin(train_phone)]
                if self.conf.stft_targets[0].find("center") > -1:
                    orig_df = calc_triangle_center(
                        df=orig_df,
                        targets=["latDeg", "lngDeg", "latDeg_gt", "lngDeg_gt"],
                    )
                else:
                    orig_df = add_distance_diff(df=orig_df, is_test=False)
                split_info_df = train_df.loc[
                    :, ["phone", "millisSinceGpsEpoch", "location", "fold", "length"]
                ]
                orig_df = pd.merge(
                    left=orig_df,
                    right=split_info_df,
                    on=["phone", "millisSinceGpsEpoch"],
                )
                orig_df["phone"] = orig_df["phone"] + "_kf_aug"

                train_df = pd.concat([train_df, orig_df], axis=0).reset_index(drop=True)

            if self.conf.data_aug_with_gaussian:
                train_phone = train_df.phone.unique()
                orig_df = pd.read_csv(data_dir / "baseline_locations_train.csv")
                orig_df = orig_df.merge(
                    get_groundtruth(data_dir),
                    on=["collectionName", "phoneName", "millisSinceGpsEpoch"],
                )
                orig_df = orig_df.loc[orig_df.phone.isin(train_phone)]
                orig_df = apply_gauss_smoothing(
                    df=orig_df, params={"sz_1": 0.85, "sz_2": 5.65, "sz_crit": 1.5}
                )
                if self.conf.stft_targets[0].find("center") > -1:
                    orig_df = calc_triangle_center(
                        df=orig_df,
                        targets=["latDeg", "lngDeg", "latDeg_gt", "lngDeg_gt"],
                    )
                else:
                    orig_df = add_distance_diff(df=orig_df, is_test=False)
                split_info_df = train_df.loc[
                    :, ["phone", "millisSinceGpsEpoch", "location", "fold", "length"]
                ]
                orig_df = pd.merge(
                    left=orig_df,
                    right=split_info_df,
                    on=["phone", "millisSinceGpsEpoch"],
                )
                orig_df["phone"] = orig_df["phone"] + "_gauss"
                train_df = pd.concat([train_df, orig_df], axis=0).reset_index(drop=True)

            train_df, train_list = make_sampling_list(
                df=train_df,
                input_width=conf["input_width"],
                sampling_delta=conf["train_sampling_delta"],
                stft_targets=conf["stft_targets"],
                is_test=False,
                remove_starts=True,
                remove_ends=False
                if self.conf.stft_targets[0].find("prev") > -1
                else True,
            )
            train_sequences = get_phone_sequences(
                df=train_df, targets=conf["stft_targets"], is_test=False
            )

            val_df, val_list = make_sampling_list(
                df=val_df,
                input_width=conf["input_width"],
                sampling_delta=conf["val_sampling_delta"],
                stft_targets=conf["stft_targets"],
                is_test=False,
                remove_starts=True,
                remove_ends=False
                if self.conf.stft_targets[0].find("prev") > -1
                else True,
            )
            val_df.to_csv("./val.csv")
            val_sequences = get_phone_sequences(
                df=val_df, targets=conf["stft_targets"], is_test=False
            )

            self.train_dataset = WaveformDataset(
                sampling_list=train_list,
                phone_sequences=train_sequences,
                stft_targets=conf["stft_targets"],
                stft_params=conf["stft_params"],
                input_width=conf["input_width"],
                image_transforms=self.train_transform(),
                is_test=False,
                gt_as_mask=self.conf.gt_as_mask,
                rand_freq=self.conf.rand_freq,
                rand_ratio=self.conf.rand_ratio,
                sigma=self.conf.sigma,
            )

            self.val_dataset = WaveformDataset(
                sampling_list=val_list,
                phone_sequences=val_sequences,
                stft_targets=conf["stft_targets"],
                stft_params=conf["stft_params"],
                input_width=conf["input_width"],
                image_transforms=self.val_transform(),
                is_test=False,
                gt_as_mask=self.conf.gt_as_mask,
            )
            self.plot_dataset(self.train_dataset)
            self.train_df = train_df
            self.val_df = val_df

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            # read data
            data_dir = Path(get_original_cwd(), self.conf["data_dir"])

            if self.conf.test_with_val:
                self.train_df = pd.read_csv(data_dir / "baseline_locations_train.csv")
                df_path = pd.read_csv(
                    Path(get_original_cwd(), "../input/path_meta_info.csv")
                )
                if self.conf.apply_kalman_filtering:
                    self.train_df = self._use_cached_kalman(
                        df=self.train_df, is_test=False
                    )
                # train/val split
                df_path = make_split(df=df_path, n_splits=3)
                self.train_df = merge_split_info(
                    data_df=self.train_df, split_df=df_path
                )
                self.test_df = self.train_df.loc[
                    self.train_df["fold"] == self.val_fold, :
                ]

            else:
                self.test_df = pd.read_csv(data_dir / "baseline_locations_test.csv")
                if self.conf.apply_kalman_filtering:
                    self.test_df = self._use_cached_kalman(
                        df=self.test_df, is_test=True
                    )

            # there is nan at being and end...
            if self.conf.stft_targets[0].find("center") > -1:
                self.test_df = calc_triangle_center(
                    df=self.test_df, targets=["latDeg", "lngDeg"],
                )
            else:
                self.test_df = add_distance_diff(df=self.test_df, is_test=True)

            if self.conf.tta_with_kf:
                test_phone = self.test_df.phone.unique()
                if self.conf.apply_kalman_filtering:
                    orig_df = pd.read_csv(data_dir / "baseline_locations_test.csv")
                    orig_df = orig_df.merge(
                        get_groundtruth(data_dir),
                        on=["collectionName", "phoneName", "millisSinceGpsEpoch"],
                    )
                else:
                    orig_df = self._use_cached_kalman(df=self.test_df, is_test=True)
                orig_df = orig_df.loc[orig_df.phone.isin(test_phone)]
                if self.conf.stft_targets[0].find("center") > -1:
                    orig_df = calc_triangle_center(
                        df=orig_df,
                        targets=["latDeg", "lngDeg", "latDeg_gt", "lngDeg_gt"],
                    )
                else:
                    orig_df = add_distance_diff(df=orig_df, is_test=True)
                split_info_df = self.test_df.loc[
                    :, ["phone", "millisSinceGpsEpoch", "location", "fold", "length"]
                ]
                orig_df = pd.merge(
                    left=orig_df,
                    right=split_info_df,
                    on=["phone", "millisSinceGpsEpoch"],
                )
                orig_df["phone"] = orig_df["phone"] + "_kf_aug"

                self.test_df = pd.concat([self.test_df, orig_df], axis=0).reset_index(
                    drop=True
                )

            self.test_df, test_list = make_sampling_list(
                df=self.test_df,
                input_width=conf["input_width"],
                sampling_delta=conf["test_sampling_delta"],
                stft_targets=conf["stft_targets"],
                is_test=True,
                remove_starts=True,
                remove_ends=False
                if self.conf.stft_targets[0].find("prev") > -1
                else True,
            )
            self.test_df.to_csv("./test_input.csv", index=False)
            test_sequences = get_phone_sequences(
                df=self.test_df, targets=conf["stft_targets"], is_test=True
            )
            self.test_dataset = WaveformDataset(
                sampling_list=test_list,
                phone_sequences=test_sequences,
                stft_targets=conf["stft_targets"],
                stft_params=conf["stft_params"],
                input_width=conf["input_width"],
                image_transforms=self.test_transform(),
                is_test=True,
                gt_as_mask=self.conf.gt_as_mask,
            )
            self.plot_dataset(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_transform(self):
        return self.get_transforms(mode=self.aug_mode)

    def val_transform(self):
        return self.get_transforms(mode=0)

    def test_transform(self):
        return self.get_transforms(mode=0)

    def get_transforms(self, mode: int = 0) -> albu.Compose:

        self.input_size = WaveformDataset.calc_stft_resize(
            input_width=self.conf.input_width, n_fft=self.conf.stft_params.n_fft
        )

        def pad_image(
            image: np.ndarray,
            input_size: List[int],
            constant_values: float = 255.0,
            **kwargs,
        ):
            pad_size = (input_size[0] - image.shape[0], input_size[1] - image.shape[1])
            if np.any(np.array(pad_size) > 0):
                image = np.pad(
                    image, [[0, pad_size[0]], [0, pad_size[1]], [0, 0]], mode="reflect",
                )
                # image[:, :, orig_width:] = constant_values
            return image

        add_pad_img = partial(
            pad_image, input_size=self.input_size, constant_values=255.0
        )
        add_pad_mask = partial(
            pad_image, input_size=self.input_size, constant_values=1.0
        )

        if mode == 0:
            transforms = [
                albu.Lambda(image=add_pad_img, mask=add_pad_mask, name="padding"),
                albu.Normalize(mean=self.img_mean, std=self.img_std),
            ]
        elif mode == 1:
            transforms = [
                albu.HorizontalFlip(p=0.5),
                albu.Lambda(image=add_pad_img, mask=add_pad_mask, name="padding"),
                albu.Normalize(mean=self.img_mean, std=self.img_std),
            ]
        else:
            raise NotImplementedError
        if self.conf.gt_as_mask:
            additional_targets = {"target_image": "mask"}
        else:
            additional_targets = {"target_image": "image"}

        composed = albu.Compose(transforms, additional_targets=additional_targets)
        return composed

    def plot_dataset(
        self, dataset, plot_num: int = 3, df: Optional[pd.DataFrame] = None,
    ) -> None:

        inds = np.random.choice(len(dataset), plot_num)
        h_, w_ = get_input_size_wo_pad(
            n_fft=self.conf.stft_params.n_fft, input_width=self.conf.input_width
        )
        for i in inds:
            plt.figure(figsize=(16, 8))
            data = dataset[i]
            im = data["image"].numpy().transpose(1, 2, 0)
            im = im[:h_, :w_]
            # === PLOT ===
            nrows = 3
            ncols = 3
            fig, ax = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(12, 6), sharey=True, sharex=True,
            )

            fig.suptitle(
                "_".join(
                    [
                        data["phone"],
                        str(data["millisSinceGpsEpoch"]),
                        str(data["phone_time"]),
                    ]
                )
            )
            cnum = len(self.conf["stft_targets"])
            D_abs, D_cos, D_sin = WaveformDataset.handle_stft_normalize(
                img=im,
                cnum=cnum,
                is_encode=False,
                is_db=self.conf["stft_params"]["is_db"],
                img_mean=self.img_mean,
                img_std=self.img_std,
            )

            for stft_ind, stft_name in enumerate(self.conf["stft_targets"]):
                show_stft(
                    conf=self.conf,
                    D_abs=D_abs[..., stft_ind],
                    D_cos=D_cos[..., stft_ind],
                    D_sin=D_sin[..., stft_ind],
                    ax=ax,
                    stft_ind=stft_ind,
                    stft_name=stft_name,
                )

            if data["target_image"].shape[0] != 0:

                im = data["target_image"].numpy().transpose(1, 2, 0)
                im = im[:h_, :w_]
                # === PLOT ===
                nrows = 3
                ncols = 3
                fig, ax = plt.subplots(
                    nrows=nrows, ncols=ncols, figsize=(12, 6), sharey=True, sharex=True,
                )

                fig.suptitle(
                    "_".join(
                        [
                            data["phone"],
                            str(data["millisSinceGpsEpoch"]),
                            str(data["phone_time"]),
                        ]
                    )
                )
                cnum = len(self.conf["stft_targets"])
                D_abs, D_cos, D_sin = WaveformDataset.handle_stft_normalize(
                    img=im,
                    cnum=cnum,
                    is_encode=False,
                    is_db=self.conf["stft_params"]["is_db"],
                    img_mean=self.img_mean,
                    img_std=self.img_std,
                    gt_as_mask=self.conf.gt_as_mask,
                )

                for stft_ind, stft_name in enumerate(self.conf["stft_targets"]):
                    show_stft(
                        conf=self.conf,
                        D_abs=D_abs[..., stft_ind],
                        D_cos=D_cos[..., stft_ind],
                        D_sin=D_sin[..., stft_ind],
                        ax=ax,
                        stft_ind=stft_ind,
                        stft_name=stft_name.replace("_diff", "_gt_diff"),
                    )


def get_input_size_wo_pad(n_fft: int = 256, input_width: int = 128) -> Tuple[int, int]:
    input_height = n_fft // 2 + 1
    input_width = input_width + 1
    return input_height, input_width


def show_stft(
    conf: DictConfig,
    D_abs: np.ndarray,
    D_cos: np.ndarray,
    D_sin: np.ndarray,
    ax: plt.axes,
    stft_ind: int,
    stft_name: str = None,
) -> None:
    for nrow, mat in enumerate([D_abs, D_cos, D_sin]):
        img = librosa.display.specshow(
            mat,
            sr=1,
            hop_length=conf["stft_params"]["hop_length"],
            x_axis="time",
            y_axis="hz",
            cmap="cool",
            ax=ax[nrow][stft_ind],
        )
        plt.colorbar(img, ax=ax[nrow][stft_ind])
    ax[0][stft_ind].set_title(stft_name)


def choose_paths(df: pd.DataFrame, target: str = "short") -> pd.DataFrame:
    if target is not None:
        return df.loc[df["length"].apply(lambda x: x.split("-")[0]) == target, :]
    else:
        return df


def make_split(df: pd.DataFrame, n_splits: int = 3) -> pd.DataFrame:
    df["fold"] = -1
    df["groups"] = df["location"].apply(lambda x: x.split("-")[0])
    df["groups"] = df["groups"] + "_" + df["length"]
    # gkf = GroupKFold(n_splits=n_splits)
    gkf = StratifiedKFold(n_splits=n_splits)

    for i, (train_idx, valid_idx) in enumerate(gkf.split(df, df["groups"])):
        df.loc[valid_idx, "fold"] = i

    return df


def merge_split_info(data_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    split_col = ["collectionName", "location", "length", "fold"]
    df = pd.merge(data_df, split_df.loc[:, split_col], on="collectionName")
    return df


def interpolate_vel(
    velocity: np.ndarray,
    base_time: np.ndarray,
    ref_time: np.ndarray,
    drop_first_vel: bool = True,
) -> np.ndarray:

    if velocity.ndim == 1:
        raise NotImplementedError

    if ref_time.max() > base_time.max():
        assert ref_time.max() - base_time.max() <= 1000
        base_time = np.pad(
            base_time, [0, 1], mode="constant", constant_values=base_time.max() + 1000
        )
        velocity = np.pad(velocity, [[0, 1], [0, 0]], mode="edge")

    if drop_first_vel:
        assert np.all(velocity[0] == np.nan) or np.all(velocity[0] == 0.0)
        velocity = velocity[
            1:,
        ]
    # (sequence, feats)
    rel_posi = np.cumsum(velocity, axis=0)
    rel_posi = np.pad(rel_posi, [[1, 0], [0, 0]], mode="constant", constant_values=0.0)
    rel_posi_ref = scipy.interpolate.interp1d(base_time, rel_posi, axis=0)(ref_time)
    vel_ref = np.diff(rel_posi_ref, axis=0)
    if drop_first_vel:
        vel_ref = np.pad(
            vel_ref, [[1, 0], [0, 0]], mode="constant", constant_values=np.nan
        )
    return vel_ref


def make_sampling_list(
    df: pd.DataFrame,
    input_width: int = 256,
    sampling_delta: int = 1,
    remove_starts: bool = True,
    remove_ends: bool = False,
    stft_targets: List[str] = ["latDeg_diff_prev", "lngDeg_diff_prev"],
    is_test: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sampling_list = []
    initial_time_offset = 0
    dfs = []
    if isinstance(stft_targets, ListConfig):
        stft_targets = OmegaConf.to_container(stft_targets)
    if not is_test:
        gt_targets = [target.replace("_diff", "_gt_diff") for target in stft_targets]
        stft_targets = stft_targets + gt_targets

    if remove_starts:
        # for remove initial
        initial_time_offset += 1

    for phone, df_ in df.groupby("phone"):
        # length includes the min value, so we need "+1"
        second_length = (
            np.ceil(
                (df_["millisSinceGpsEpoch"].max() - df_["millisSinceGpsEpoch"].min())
                / 1000,
            ).astype(np.int64)
            + 1
        )

        inter_gps_epochs = (
            np.arange(0, second_length, dtype=np.int64,) * 1000
            + df_["millisSinceGpsEpoch"].min()
        )
        assert inter_gps_epochs[-1] // 1000 >= df_["millisSinceGpsEpoch"].max() // 1000
        inter_targets = interpolate_vel(
            velocity=df_.loc[:, stft_targets].fillna(0.0).values,
            base_time=df_.loc[:, "millisSinceGpsEpoch"].values,
            ref_time=inter_gps_epochs,
            drop_first_vel=True,
        )
        for_df = {
            "phone": np.repeat(phone, inter_gps_epochs.shape[0]),
            "millisSinceGpsEpoch": inter_gps_epochs,
        }

        for_df.update({key: inter_targets[:, i] for i, key in enumerate(stft_targets)})
        inter_df = pd.DataFrame(for_df)

        end_point = (
            second_length - input_width * 2
            if remove_ends
            else second_length - input_width
        )
        samplings = np.linspace(
            initial_time_offset,
            end_point,
            np.ceil((end_point - initial_time_offset) / sampling_delta + 1).astype(
                np.int64
            ),
            dtype=np.int64,
            endpoint=True,
        )
        inter_df["phone_time"] = inter_df.reset_index().index.values

        assert inter_df.iloc[samplings[-1] :].shape[0] == input_width
        sampling_list.append(
            inter_df.loc[:, ["phone", "millisSinceGpsEpoch", "phone_time"]].iloc[
                samplings
            ]
        )
        if inter_gps_epochs[-1] > df_["millisSinceGpsEpoch"].max():
            pass
        else:
            if np.any(np.diff(df_["millisSinceGpsEpoch"]) != 1000):
                if np.all(np.diff(df_["millisSinceGpsEpoch"]) % 1000 == 0):
                    assert np.all(
                        inter_df.loc[:, stft_targets].values[-1]
                        == df_.loc[:, stft_targets].values[-1]
                    )

        dfs.append(inter_df)
    sampling_list = pd.concat(sampling_list)
    df = pd.concat(dfs)
    sampling_list.reset_index(inplace=True, drop=True)
    return df, sampling_list


def get_phone_sequences(
    df: pd.DataFrame,
    targets: List[str] = [
        "latDeg_diff_center",
        "latDeg_diff_right",
        "latDeg_diff_left",
    ],
    is_test: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    phone_sequences = {}
    if not is_test:
        gt_targets = [target.replace("_diff", "_gt_diff") for target in targets]
        targets = targets + gt_targets
    for phone, df_ in df.groupby("phone"):
        values = {target: df_[target].values for target in targets}
        phone_sequences[phone] = values
    return phone_sequences


def check_img_files(ext_dir: Path, file_ext: str = "png") -> List[str]:
    common_id = []
    for color in ["red", "green", "blue", "yellow"]:
        ext_files = glob.glob(str(ext_dir / f"*{color}.{file_ext}"))
        ext_file_ids = [
            Path(file).name.replace(f"_{color}.{file_ext}", "") for file in ext_files
        ]
        print(f"{color} image files:", len(ext_file_ids))
        if color == "red":
            common_id = ext_file_ids
        else:
            common_id = list(set(common_id) & set(ext_file_ids))
    print("common file id num", len(set(common_id)))
    return common_id
