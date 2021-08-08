import copy
import os
from functools import partial
from pathlib import Path
from typing import List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.neighbors import KDTree

from src.dataset.datamodule import GsdcDatamodule, interpolate_vel
from src.dataset.utils import get_groundtruth
from src.modeling.pl_model import LitModel
from src.postprocess.metric import print_metric
from src.postprocess.postporcess import (apply_kf_smoothing, filter_outlier,
                                         mean_with_other_phones)
from src.postprocess.visualize import add_distance_diff
from src.utils.util import set_random_seed

pd.set_option("display.max_rows", 100)
SEED = 42


def check_test_df(path_a, path_b):
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    df_a = df_a.rename(columns={"latDeg": "latDeg_gt", "lngDeg": "lngDeg_gt"})
    df = pd.merge(df_a, df_b, on=["phone", "millisSinceGpsEpoch"])
    met_df = print_metric(df=df)
    return met_df


def load_dataset(is_test: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(
        get_original_cwd(), "../input/google-smartphone-decimeter-challenge"
    )
    fname = "test" if is_test else "train"
    df = pd.read_csv(data_dir / f"baseline_locations_{fname}.csv")

    if not is_test:
        # merge graoundtruth
        df = df.merge(
            get_groundtruth(data_dir),
            on=["collectionName", "phoneName", "millisSinceGpsEpoch"],
        )

    # area_df from
    # https://www.kaggle.com/columbia2131/area-knn-prediction-train-hand-label
    area_df = pd.read_csv(
        Path(get_original_cwd()) / f"./src/meta_data/{fname}_area.csv"
    )

    df = apply_kf_smoothing(df=df)
    df = add_distance_diff(df=df, is_test=is_test)

    if is_test:
        area_df = area_df.rename(columns={"area_pred": "area_target"})

    df = pd.merge(df, area_df[["collectionName", "area_target"]], on=["collectionName"])
    return df, area_df


def interpolate_vel_df(
    vel_pred_df: pd.DataFrame,
    posi_pred_df: pd.DataFrame,
    abs_targets: List[str] = ["latDeg", "lngDeg"],
    pred_targets: List[str] = ["latDeg_diff_prev", "lngDeg_diff_prev"],
):
    T_ref = posi_pred_df.loc[:, "millisSinceGpsEpoch"].values
    rel_positions = vel_pred_df.loc[:, pred_targets].fillna(0.0).values
    T_rel = vel_pred_df.loc[:, "millisSinceGpsEpoch"].values

    # padding and interpolate, exclude the fisrt nan
    delta_xy_hat = interpolate_vel(
        velocity=rel_positions, base_time=T_rel, ref_time=T_ref, drop_first_vel=True
    )
    return T_ref, delta_xy_hat


def plot_velocity(
    vel_pred_df: pd.DataFrame,
    posi_pred_df: pd.DataFrame,
    phone: str,
    pred_targets: List[str] = ["latDeg_diff_prev", "lngDeg_diff_prev"],
    is_test: bool = False,
) -> None:
    posi_pred_df = posi_pred_df[posi_pred_df["phone"] == phone].fillna(0.0)
    posi_pred_df = add_distance_diff(df=posi_pred_df, is_test=is_test)

    vel_pred_df = vel_pred_df[vel_pred_df["phone"] == phone].fillna(0.0)

    T_ref, delta_xy_hat = interpolate_vel_df(
        vel_pred_df=vel_pred_df, posi_pred_df=posi_pred_df, pred_targets=pred_targets,
    )
    for_df = {
        "phone": np.repeat(phone, delta_xy_hat.shape[0]),
        "millisSinceGpsEpoch": T_ref,
    }
    for_df.update(
        {stft_target: delta_xy_hat[:, i] for i, stft_target in enumerate(pred_targets)}
    )
    vel_pred_df = pd.DataFrame(for_df).fillna(0.0)
    fig, axes = plt.subplots(figsize=(15, 8), nrows=3, sharex=True)

    axes[0].plot(
        posi_pred_df["millisSinceGpsEpoch"],
        posi_pred_df["latDeg_diff_prev"],
        # "--",
        label="baseline",
    )
    axes[0].plot(
        vel_pred_df["millisSinceGpsEpoch"],
        vel_pred_df["latDeg_diff_prev"],
        label="pred",
    )

    axes[1].plot(
        posi_pred_df["millisSinceGpsEpoch"],
        posi_pred_df["lngDeg_diff_prev"],
        label="baseline",
    )
    axes[1].plot(
        vel_pred_df["millisSinceGpsEpoch"],
        vel_pred_df["lngDeg_diff_prev"],
        label="pred",
    )
    if not is_test:
        axes[0].plot(
            posi_pred_df["millisSinceGpsEpoch"],
            posi_pred_df["latDeg_gt_diff_prev"],
            "--",
            label="gt",
        )
        axes[1].plot(
            posi_pred_df["millisSinceGpsEpoch"],
            posi_pred_df["lngDeg_gt_diff_prev"],
            "--",
            label="gt",
        )

        axes[2].plot(
            posi_pred_df["millisSinceGpsEpoch"],
            np.abs(
                posi_pred_df["latDeg_diff_prev"].values
                - posi_pred_df["latDeg_gt_diff_prev"].values
            ),
            "--",
            label="baseline",
        )
        axes[2].plot(
            posi_pred_df["millisSinceGpsEpoch"],
            np.abs(
                vel_pred_df["latDeg_diff_prev"].values
                - posi_pred_df["latDeg_gt_diff_prev"].values
            ),
            label="pred",
        )
    yscale = "linear"

    axes[0].set_yscale(yscale)
    axes[0].set_ylabel("deg velocity")

    axes[1].set_yscale(yscale)
    axes[1].set_ylabel("deg velocity")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1.0e-8, 1.0e-4)

    axes[1].set_xlabel("time step")
    axes[0].grid()
    axes[1].grid()
    axes[2].grid()

    title = phone
    fig.suptitle(title)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    save_path = os.path.join(os.getcwd(), f"{phone}_vel.png")
    print(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def rel_pred(
    phone: str,
    vel_pred_df: pd.DataFrame,
    posi_pred_df: pd.DataFrame,
    stft_targets: List[str],
    is_test: bool = False,
) -> pd.DataFrame:
    posi_pred_df = posi_pred_df[posi_pred_df["phone"] == phone].fillna(0.0)
    posi_pred_df = add_distance_diff(df=posi_pred_df, is_test=is_test)
    vel_pred_df = vel_pred_df[vel_pred_df["phone"] == phone].fillna(0.0)

    targets = ["latDeg", "lngDeg"]

    T_ref, delta_xy_hat = interpolate_vel_df(
        vel_pred_df=vel_pred_df, posi_pred_df=posi_pred_df, pred_targets=stft_targets
    )
    delta_xy_hat[:1] = 0.0
    pred_posi = np.cumsum(delta_xy_hat, axis=0)
    rel_posi_mean = pred_posi.mean(axis=0)

    abs_posi_mean = posi_pred_df.loc[:, targets].values.mean(axis=0)

    pred_posi = pred_posi - rel_posi_mean + abs_posi_mean
    print(pred_posi.mean(axis=0) - abs_posi_mean)
    posi_pred_df.loc[:, targets] = pred_posi

    return posi_pred_df


def calc_avg_vel(df: pd.DataFrame, is_database: bool = False, add_future: bool = False):
    if is_database:
        global_targets = ["latDeg_gt", "lngDeg_gt"]
    else:
        global_targets = ["latDeg", "lngDeg"]

    local_targets = [key + "_diff_prev" for key in global_targets]
    window_sizes = [5, 15, 45]
    new_targets = []
    dfs = []
    for phone, df_ in df.groupby("phone"):
        for window_size in window_sizes:
            tri_center = (
                df_[local_targets]
                .rolling(window=window_size, min_periods=1, center=False)
                .mean()
            ).fillna(0.0)
            for target in local_targets:
                new_targets.append(target + "_" + str(window_size))
                df_[new_targets[-1]] = tri_center[target]
                if add_future:
                    next_target = new_targets[-1].replace("prev", "next")
                    df_[next_target] = (
                        df_[new_targets[-1]].shift(-window_size).fillna(0.0)
                    )
                    new_targets.append(next_target)
        dfs.append(df_)

    new_targets = sorted(list(set(new_targets)))
    df = pd.concat(dfs, axis=0)
    local_targets.extend(new_targets)
    local_targets = local_targets + ["latDeg_diff_next", "lngDeg_diff_next"]
    df.reset_index(drop=False, inplace=True)
    return df, global_targets, local_targets


def knn_search(
    phone: str,
    vel_pred_df: pd.DataFrame,
    posi_pred_df: pd.DataFrame,
    data_df: pd.DataFrame,
    global_targets_gt=List[str],
    local_targets_gt=List[str],
    is_test: bool = False,
):

    # choose phone data
    posi_pred_df = posi_pred_df[posi_pred_df["phone"] == phone]
    posi_pred_df = add_distance_diff(df=posi_pred_df, is_test=is_test)
    posi_pred_df = posi_pred_df.fillna(0.0)
    vel_pred_df = vel_pred_df[vel_pred_df["phone"] == phone].fillna(0.0)
    area = posi_pred_df.area_target.to_numpy()[0]
    if posi_pred_df.area_target.to_numpy()[0] != 2:
        return posi_pred_df

    un_used = posi_pred_df[posi_pred_df["phone"] == phone].collectionName.unique()[0]
    data_df = data_df.loc[data_df.collectionName != un_used]
    leaf_size = 40
    k = 10 if area == 2 else 5
    k_local = 3
    local_leaf_size = 10
    global_tree = KDTree(data_df[global_targets_gt], leaf_size=leaf_size)

    posi_pred_df, global_targets, local_targets = calc_avg_vel(df=posi_pred_df)
    dists, inds = global_tree.query(posi_pred_df[global_targets], k=k)
    # print("mean, std", np.mean(dists), np.std(dists))
    for i, (dist, ind) in enumerate(zip(dists, inds)):
        if area == 2:
            pass
        elif area == 1:
            deg_vel = vel_pred_df.iloc[i][["latDeg_diff_prev", "lngDeg_diff_prev"]]
            if np.any(deg_vel > 5.0e-6):
                continue
        else:
            continue
        local_data_df = data_df.iloc[ind]
        query_state = posi_pred_df.iloc[i]
        local_tree = KDTree(local_data_df[local_targets_gt], leaf_size=local_leaf_size)
        local_dist, local_ind = local_tree.query(
            query_state[local_targets].to_numpy().reshape(1, -1), k=k_local
        )

        posi_pred_df.loc[
            posi_pred_df.millisSinceGpsEpoch == query_state.millisSinceGpsEpoch,
            global_targets,
        ] = (
            local_data_df.iloc[local_ind[0]][global_targets_gt].to_numpy().mean(axis=0)
        )
    return posi_pred_df


def mask_with_velocity(
    phone: str,
    vel_pred_df: pd.DataFrame,
    posi_pred_df: pd.DataFrame,
    stft_targets: List[str],
    is_test: bool = False,
) -> pd.DataFrame:
    targets = ["latDeg", "lngDeg"]

    posi_pred_df = posi_pred_df[posi_pred_df["phone"] == phone].fillna(0.0)
    posi_pred_df = add_distance_diff(df=posi_pred_df, is_test=is_test)
    vel_pred_df = vel_pred_df[vel_pred_df["phone"] == phone].fillna(0.0)

    deg_preds = posi_pred_df.loc[:, targets].values

    T_ref, delta_xy_hat = interpolate_vel_df(
        vel_pred_df=vel_pred_df, posi_pred_df=posi_pred_df, pred_targets=stft_targets
    )

    kf_vel = posi_pred_df.loc[:, stft_targets].values

    delta_xy_hat[:1] = 0.0
    kf_vel[:1] = 0.0

    low_pred_vel = np.all(np.abs(delta_xy_hat) < 5e-7, axis=1)

    prev_part = 0

    low_vel_area = list(np.where(np.diff(T_ref[low_pred_vel], axis=0) > 5000)[0])
    low_vel_area.append(low_pred_vel.sum())
    local_means = []
    for part_ind in low_vel_area:
        part_ind = part_ind + 1
        slice_target = np.where(low_pred_vel)[0][prev_part:part_ind]
        local_degs = np.take(deg_preds, slice_target, axis=0)
        local_span = np.take(T_ref, slice_target, axis=0)
        if local_span.max() - local_span.min() > 5000:
            local_mean = np.median(local_degs, axis=0, keepdims=True)
            local_means.append(np.repeat(local_mean, local_degs.shape[0], axis=0))
        else:
            local_means.append(local_degs)

        prev_part = part_ind

    deg_preds[low_pred_vel] = np.concatenate(local_means, axis=0)

    kf_vel = np.diff(deg_preds, axis=0)
    kf_vel = np.pad(kf_vel, [[1, 0], [0, 0]], mode="constant", constant_values=0.0)
    disagreement_mask = np.all(np.abs(delta_xy_hat - kf_vel) > 0.09e-4, axis=1)
    disagreement_mask[:-1] += disagreement_mask[1:]

    disagreement_mask[0] = False
    disagreement_mask[-1] = False

    deg_preds_filtered = copy.deepcopy(deg_preds[~disagreement_mask])
    T_ref_filtered = copy.deepcopy(T_ref[~disagreement_mask])
    deg_preds = scipy.interpolate.interp1d(T_ref_filtered, deg_preds_filtered, axis=0)(
        T_ref
    )

    posi_pred_df.loc[:, targets] = deg_preds
    return posi_pred_df


@hydra.main(config_path="./src/config", config_name="test_config")
def main(conf: DictConfig) -> None:
    set_random_seed(SEED)

    is_test = not conf.test_with_val
    print("start conv position prediction")
    for model_path in conf.test_weights.model_paths:
        model_path = Path(
            get_original_cwd(), conf.test_weights.weights_dir, model_path[1]
        )
        conf_path = model_path / conf.test_weights.conf_name
        model_conf = OmegaConf.load(conf_path)
        ckpt_path = list(model_path.glob(conf.test_weights.ckpt_regex))

        assert len(ckpt_path) == 1
        model_conf.ckpt_path = str(ckpt_path[0])
        print("\t\t ==== TEST MODE ====")
        print("load from: ", model_conf.ckpt_path)

        # add missing keys
        model_conf.tta_with_kf = conf.tta_with_kf
        model_conf.use_flip_tta = conf.use_flip_tta
        model_conf.test_with_val = conf.test_with_val
        model_conf.test_sampling_delta = conf.test_sampling_delta
        datamodule = GsdcDatamodule(
            conf=model_conf,
            val_fold=model_conf.val_fold,
            batch_size=model_conf.batch_size,
            aug_mode=model_conf.aug_mode,
            num_workers=model_conf.num_workers,
            is_debug=model_conf.is_debug,
        )
        datamodule.prepare_data()
        datamodule.setup(stage="test")
        model = LitModel.load_from_checkpoint(
            model_conf.ckpt_path, conf=model_conf, dataset_len=-1
        )
        pl.trainer.seed_everything(seed=SEED)
        trainer = pl.Trainer(gpus=1)
        trainer.test(model, datamodule=datamodule)
        torch.cuda.empty_cache()

    csv_name = f"pred_test_flip_{conf.use_flip_tta}_d{conf.test_sampling_delta}.csv"
    vel_pred_paths = [
        [
            model_path[0],
            Path(get_original_cwd(), conf.test_weights.weights_dir, model_path[1])
            / csv_name,
        ]
        for model_path in conf.test_weights.model_paths
    ]
    vel_preds = []

    for path in vel_pred_paths:
        df = pd.read_csv(path[1])
        vel_preds.append(path[0] * df.loc[:, conf.stft_targets].values)

    vel_pred_df = pd.read_csv(vel_pred_paths[0][1])
    vel_pred_df.loc[:, conf.stft_targets] = np.sum(np.stack(vel_preds), axis=0)

    print("loading baseline positions")
    posi_pred_df, area_df = load_dataset(is_test=is_test)

    phone_list = vel_pred_df["phone"].unique()
    # baseline
    df = posi_pred_df.loc[posi_pred_df.phone.isin(phone_list)]
    if not is_test:
        print("baseline")
        met_df = print_metric(df=df)
        print(met_df)

    print("Ensemble, conv pred & ligtgbm")
    gbm_df = pd.read_csv(Path(get_original_cwd(), conf.gbm_pred_path))
    conv_df = pd.read_csv(Path(get_original_cwd(), conf.conv_pred_path))

    targets = ["latDeg", "lngDeg"]
    for phone in phone_list:
        df_ = gbm_df.loc[gbm_df.phone == phone]
        cname = phone.split("_")[0]
        area = area_df.loc[area_df.collectionName == cname]["area_target"].to_numpy()[0]
        if len(df_) > 0:
            if area == 0:
                posi_pred_df.loc[posi_pred_df.phone == phone, targets] = (
                    df_.loc[:, targets].to_numpy() * 0.6
                    + conv_df.loc[conv_df.phone == phone, targets].to_numpy() * 0.4
                )
            else:
                posi_pred_df.loc[posi_pred_df.phone == phone, targets] = (
                    df_.loc[:, targets].to_numpy() * 0.3
                    + conv_df.loc[conv_df.phone == phone, targets].to_numpy() * 0.7
                )
        else:
            if area == 0:
                posi_pred_df.loc[posi_pred_df.phone == phone, targets] = (
                    posi_pred_df.loc[posi_pred_df.phone == phone, targets].to_numpy()
                    * 0.6
                    + conv_df.loc[conv_df.phone == phone, targets].to_numpy() * 0.4
                )
            elif area == 1:
                posi_pred_df.loc[posi_pred_df.phone == phone, targets] = (
                    posi_pred_df.loc[posi_pred_df.phone == phone, targets].to_numpy()
                    * 0.2
                    + conv_df.loc[conv_df.phone == phone, targets].to_numpy() * 0.8
                )
            elif area == 2:
                posi_pred_df.loc[posi_pred_df.phone == phone, targets] = (
                    posi_pred_df.loc[posi_pred_df.phone == phone, targets].to_numpy()
                    * 0.10
                    + conv_df.loc[conv_df.phone == phone, targets].to_numpy() * 0.9
                )

    if not is_test:
        met_df = print_metric(df=posi_pred_df)
        print(met_df)

    print("conv speed & disagreement mask")
    dfs = []
    for phone in phone_list:
        dfs.append(
            mask_with_velocity(
                phone=phone,
                vel_pred_df=vel_pred_df,
                posi_pred_df=posi_pred_df,
                stft_targets=conf.stft_targets,
                is_test=is_test,
            )
        )
    df = pd.concat(dfs, axis=0)
    if not is_test:
        met_df = print_metric(df=df)
        print(met_df)

    print("outlier")
    df = filter_outlier(df=df, one_direction=True)
    if not is_test:
        print_metric(df=df)

    print("mean pred")
    df = mean_with_other_phones(df=df)
    if not is_test:
        met_df = print_metric(df=df)
        print(met_df)

    print("knn at downtown")
    dfs = []
    data_df, _ = load_dataset(is_test=False)
    data_df = data_df.fillna(0.0)

    data_df, global_targets_gt, local_targets_gt = calc_avg_vel(
        df=data_df, is_database=True
    )
    for phone in phone_list:
        dfs.append(
            knn_search(
                phone=phone,
                vel_pred_df=vel_pred_df,
                posi_pred_df=df,
                data_df=data_df,
                global_targets_gt=global_targets_gt,
                local_targets_gt=local_targets_gt,
                is_test=is_test,
            )
        )
    df = pd.concat(dfs, axis=0)
    if not is_test:
        met_df = print_metric(df=df)
        print(met_df)

    print("kalmann filtering ")
    df_down = df.loc[df.area_target == 2]
    df_down = apply_kf_smoothing(df=df_down)
    df = df.loc[df.area_target != 2]
    df = pd.concat([df, df_down], axis=0)
    df = df.sort_values(["phone", "millisSinceGpsEpoch"]).reset_index(drop=True)
    if not is_test:
        met_df = print_metric(df=df)
        print(met_df)

    if is_test:
        targets = ["phone", "millisSinceGpsEpoch", "latDeg", "lngDeg"]
        df = df.loc[:, targets]
        sample_sub = os.path.join(
            get_original_cwd(),
            "../input/google-smartphone-decimeter-challenge/sample_submission.csv",
        )
        sample_sub = pd.read_csv(sample_sub)
        orig_len = (len(sample_sub), len(df))
        sample_sub = sample_sub.loc[:, targets[:2]]
        df = pd.merge(left=sample_sub, right=df, on=targets[:2])
        after_len = (len(sample_sub), len(df))
        print(orig_len, after_len)
        save_path = Path.cwd() / "./submission.csv"
        df.loc[:, targets].to_csv(save_path, index=False)
        print(f"Save submission file on {str(save_path)}")


if __name__ == "__main__":
    main()
